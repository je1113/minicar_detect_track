#!/usr/bin/env python3
"""
자동차 위치로 Nav2 goal 을 보내 standoff 거리까지 접근하고 취소 요청을 처리하는 노드.

토픽 이름, 메시지 타입, 상태 문자열은 mission_manager / localizer 와의 팀 인터페이스다.
  target_topic  geometry_msgs/PointStamped (frame=map) : 자동차의 지도 좌표
  cancel_topic  std_msgs/Empty                         : 이동 취소 요청
  status_topic  std_msgs/String (transient local)      : IDLE/MOVING/ARRIVED/FAILED/CANCELED

goal 은 자동차 위치 자체로 보내고, Nav2 feedback 의 distance_remaining(경로상 남은 길이)이
standoff_distance 아래로 내려가면 goal 을 취소하고 ARRIVED 로 알린다.
  - goal 을 '로봇->자동차 직선 위 standoff 앞' 에 두면 사이에 벽이 있을 때 goal 이 벽 앞에 찍힌다.
    자동차 위치로 보내면 Nav2 가 벽을 돌아가는 경로를 만들고, 경로 길이로 도착을 판정한다.
  - 이동 중에도 자동차가 goal_move_threshold 이상 움직이면 goal 을 다시 보낸다 (goal_period 간격 이상).
  - 도착 뒤 자동차가 거의 그대로면 도착 상태를 유지하고, 많이 움직이면 다시 출발한다.
로봇 위치는 TF(map -> base_link) 로 구한다.

Nav2 가 실패하면(서버 없음/거절/abort) fallback_waypoints 를 거쳐 자동차 standoff 앞까지
cmd_vel 로 직접 이동하고(모서리를 원호로 보간한 경로를 pure pursuit 로 추종),
그 결과를 최종 상태로 쓴다.
"""
from dataclasses import dataclass, fields
from enum import Enum
from functools import partial
import math
from typing import NamedTuple, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener

NODE_NAME = 'approach'
SUBSCRIPTION_QUEUE_DEPTH = 10
FALLBACK_PERIOD_SEC = 0.1

# 늦게 구독한 노드(mission_manager)도 마지막 상태를 받을 수 있도록 transient local 을 쓴다.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class ApproachStatus(str, Enum):
    """status 토픽으로 발행하는 상태. 값은 팀 인터페이스 문자열이다."""

    IDLE = 'IDLE'
    MOVING = 'MOVING'
    ARRIVED = 'ARRIVED'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'


NAV_RESULT_TO_STATUS = {
    GoalStatus.STATUS_SUCCEEDED: ApproachStatus.ARRIVED,
    GoalStatus.STATUS_CANCELED: ApproachStatus.CANCELED,
}


# ---------------------------------------------------------------- pure geometry
class Point2D(NamedTuple):
    """map 프레임의 2D 좌표 [m]."""

    x: float
    y: float


class Pose2D(NamedTuple):
    """map 프레임의 2D 자세 (위치 [m], yaw [rad])."""

    x: float
    y: float
    yaw: float

    @property
    def position(self) -> Point2D:
        """방향(yaw)을 제외한 위치."""
        return Point2D(self.x, self.y)


class QuaternionXYZW(NamedTuple):
    """ROS 메시지와 같은 (x, y, z, w) 순서의 quaternion."""

    x: float
    y: float
    z: float
    w: float


def yaw_to_quaternion(yaw: float) -> QuaternionXYZW:
    """Z 축 회전 yaw [rad] 를 quaternion 으로 변환한다."""
    return QuaternionXYZW(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(q: QuaternionXYZW) -> float:
    """Quaternion 의 z 축 회전 성분(yaw) [rad] 을 구한다."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

def normalize_angle(angle: float) -> float:
    """각도를 [-pi, pi) 로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_goal_pose(robot: Point2D, car: Point2D) -> Pose2D:
    """
    자동차 위치 자체를 goal 로 하고, 로봇에서 자동차를 바라보는 방향을 yaw 로 한다.

    Nav2 는 경로상 standoff 거리가 남으면 취소되므로 실제로 자동차까지 가지는 않는다.
    """
    return Pose2D(car.x, car.y, math.atan2(car.y - robot.y, car.x - robot.x))


# --------------------------------------------------------------- cmd_vel route
class Velocity(NamedTuple):
    """Cmd_vel 로 보낼 직진 [m/s], 회전 [rad/s] 속도."""

    linear: float
    angular: float


STOP = Velocity(0.0, 0.0)
# 제자리 회전 속도 = ROTATE_GAIN * 방향 오차 [1/s]
ROTATE_GAIN = 1.5


@dataclass(frozen=True)
class RouteConfig:
    """RouteFollower 제어 값."""

    linear_speed: float = 0.15
    angular_speed: float = 0.6
    min_linear_speed: float = 0.03
    min_angular_speed: float = 0.15
    # 가감속 한계. 매 주기 속도 변화를 이 값 * 주기 이하로 보간한다 [m/s^2], [rad/s^2]
    linear_accel: float = 0.3
    angular_accel: float = 1.5
    # pure pursuit 전방 주시 거리 [m]
    lookahead: float = 0.25
    # 경로 끝에서 이 거리 안에 들어오면 도착한 것으로 본다 [m]
    position_tolerance: float = 0.03
    # 경로 끝을 진행 방향으로 지나쳤을 때 이 옆 방향 오차 안이면 도착으로 본다 [m]
    overshoot_tolerance: float = 0.1
    # 제자리 회전을 끝내는 방향 오차 [rad]
    yaw_tolerance: float = 0.05
    # 출발 전 제자리 회전에서 이 오차 안이면 주행을 시작한다 [rad]
    align_tolerance: float = 0.15
    # 주행 중 전방 주시점과의 방향 오차가 이보다 커지면 감속 후 제자리 회전 [rad]
    heading_tolerance: float = 0.8


class RoutePhase(str, Enum):
    """RouteFollower 진행 단계."""

    ROTATE = 'ROTATE'
    TRACK = 'TRACK'
    FINAL_ROTATE = 'FINAL_ROTATE'
    DONE = 'DONE'


def distance_to_segment(p: Point2D, a: Point2D, b: Point2D) -> float:
    """점 p 와 선분 ab 사이의 최단 거리 [m]."""
    abx, aby = b.x - a.x, b.y - a.y
    length_sq = abx * abx + aby * aby
    t = 0.0 if length_sq == 0.0 else ((p.x - a.x) * abx + (p.y - a.y) * aby) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))


def route_start_index(robot: Point2D, route: list[Point2D]) -> int:
    """
    route(시작점 포함) 에서 로봇이 다음으로 향할 점의 인덱스를 구한다.

    로봇에서 가장 가까운 구간의 끝점을 고르므로, Nav2 가 일부 이동한 뒤에도
    지나온 점으로 되돌아가거나 경로를 건너뛰지 않는다. 거리가 같으면 뒤 구간을 고른다.
    """
    return min(range(1, len(route)),
               key=lambda i: (distance_to_segment(robot, route[i - 1], route[i]), -i))


def _densify(points: list[Point2D], step: float) -> list[Point2D]:
    """연속한 점 사이를 step [m] 이하 간격으로 선형 보간한다."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.hypot(b.x - a.x, b.y - a.y) / step))
        out.extend(Point2D(a.x + (b.x - a.x) * k / n, a.y + (b.y - a.y) * k / n)
                   for k in range(1, n))
        out.append(b)
    return out


def smooth_path(points: list[Point2D], corner_radius: float,
                step: float = 0.02) -> list[Point2D]:
    """
    꺾은선 경로의 꼭짓점을 반지름 corner_radius 원호로 둥글게 하고 step 간격으로 보간한다.

    원호의 접선 길이는 양쪽 구간 길이의 절반을 넘지 않도록 줄인다.
    corner_radius 가 0 이면 꺾은선을 그대로 보간한다.
    """
    pts = [points[0]]
    for p in points[1:]:
        if math.hypot(p.x - pts[-1].x, p.y - pts[-1].y) > 1e-3:
            pts.append(p)

    corners = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        len_in = math.hypot(b.x - a.x, b.y - a.y)
        len_out = math.hypot(c.x - b.x, c.y - b.y)
        heading_in = math.atan2(b.y - a.y, b.x - a.x)
        turn = normalize_angle(math.atan2(c.y - b.y, c.x - b.x) - heading_in)
        if corner_radius <= 0.0 or abs(turn) < 1e-3 or abs(turn) > math.pi - 1e-3:
            corners.append(b)
            continue
        half_tan = math.tan(abs(turn) / 2.0)
        tangent = min(corner_radius * half_tan, 0.5 * len_in, 0.5 * len_out)
        radius = tangent / half_tan
        ux, uy = (b.x - a.x) / len_in, (b.y - a.y) / len_in
        entry = Point2D(b.x - ux * tangent, b.y - uy * tangent)
        side = math.copysign(1.0, turn)
        center = Point2D(entry.x - uy * radius * side, entry.y + ux * radius * side)
        start = math.atan2(entry.y - center.y, entry.x - center.x)
        n = max(2, math.ceil(radius * abs(turn) / step))
        corners.extend(
            Point2D(center.x + radius * math.cos(start + turn * k / n),
                    center.y + radius * math.sin(start + turn * k / n))
            for k in range(n + 1))
    corners.append(pts[-1])
    return _densify(corners, step)


def trim_path_end(path: list[Point2D], length: float) -> list[Point2D]:
    """
    경로 끝에서 length [m] 만큼 잘라낸다. 경로가 length 보다 짧으면 시작점만 남긴다.

    fallback 이 자동차 standoff 앞에서 멈추도록 자동차 위치까지의 경로를 자르는 데 쓴다.
    """
    keep = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:])) - length
    out = [path[0]]
    for a, b in zip(path, path[1:]):
        seg = math.hypot(b.x - a.x, b.y - a.y)
        if seg >= keep:
            if keep > 0.0:
                t = keep / seg
                out.append(Point2D(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
            return out
        keep -= seg
        out.append(b)
    return out


def _approach(current: float, target: float, max_delta: float) -> float:
    """값을 target 쪽으로 최대 max_delta 만큼 옮긴다."""
    return current + max(-max_delta, min(max_delta, target - current))


class RouteFollower:
    """
    보간한 경로를 pure pursuit 로 따라가고 마지막에 goal yaw 로 맞추는 제어기.

    출발 방향이 크게 틀어져 있으면 먼저 제자리 회전하고, 이후에는 곡선으로 주행한다.
    속도는 가감속 한계와 남은 거리에 따른 감속으로 매 주기 보간한다.
    ROS 와 독립적이며, period 마다 step(현재 map 자세) 로 속도를 얻는다.
    """

    def __init__(self, path: list[Point2D], final_yaw: float, config: RouteConfig,
                 period: float) -> None:
        """smooth_path 로 만든 path 를 따라가고 마지막에 final_yaw 로 맞춘다."""
        self._path = path
        self._arc = [0.0]
        for a, b in zip(path, path[1:]):
            self._arc.append(self._arc[-1] + math.hypot(b.x - a.x, b.y - a.y))
        self._final_yaw = final_yaw
        self._cfg = config
        self._period = period
        self._last = STOP
        self.progress = 0
        self.phase = RoutePhase.ROTATE

    @property
    def done(self) -> bool:
        """Goal 위치와 yaw 까지 모두 맞췄는지."""
        return self.phase is RoutePhase.DONE

    @property
    def remaining(self) -> float:
        """현재 진행 위치부터 경로 끝까지 남은 길이 [m]."""
        return self._arc[-1] - self._arc[self.progress]

    def step(self, pose: Pose2D) -> Velocity:
        """현재 자세에서 보낼 속도를 계산하고 단계를 진행한다."""
        if self.phase is RoutePhase.DONE:
            return STOP
        if self.phase is RoutePhase.FINAL_ROTATE:
            return self._rotate(self._final_yaw - pose.yaw, self._cfg.yaw_tolerance,
                                RoutePhase.DONE)

        self._update_progress(pose)
        if self._arrived(pose):
            # 남은 직진 속도는 _rotate 에서 가속도 한계로 줄인다.
            self.phase = RoutePhase.FINAL_ROTATE
            return self._rotate(self._final_yaw - pose.yaw, self._cfg.yaw_tolerance,
                                RoutePhase.DONE)

        target = self._lookahead_point()
        dx, dy = target.x - pose.x, target.y - pose.y
        alpha = normalize_angle(math.atan2(dy, dx) - pose.yaw)
        if self.phase is RoutePhase.ROTATE:
            return self._rotate(alpha, self._cfg.align_tolerance, RoutePhase.TRACK)
        if abs(alpha) > self._cfg.heading_tolerance:
            self.phase = RoutePhase.ROTATE
            return self._rotate(alpha, self._cfg.align_tolerance, RoutePhase.TRACK)

        # pure pursuit: 전방 주시점을 지나는 원호의 곡률
        curvature = 2.0 * math.sin(alpha) / max(math.hypot(dx, dy), 1e-3)
        speed = self._cfg.linear_speed
        if abs(curvature) > 1e-6:
            speed = min(speed, self._cfg.angular_speed / abs(curvature))
        # 도착 판정 거리에서 속도가 min_linear_speed 까지 떨어지도록,
        # 가속 한계의 절반으로 여유 있게 감속한다.
        stop_distance = max(0.0, self.remaining - self._cfg.position_tolerance)
        speed = min(speed, math.sqrt(self._cfg.linear_accel * stop_distance))
        speed = max(speed, self._cfg.min_linear_speed)
        linear = _approach(self._last.linear, speed, self._cfg.linear_accel * self._period)
        angular = _approach(self._last.angular, curvature * linear,
                            self._cfg.angular_accel * self._period)
        return self._output(Velocity(linear, angular))

    def _arrived(self, pose: Pose2D) -> bool:
        end = self._path[-1]
        dx, dy = end.x - pose.x, end.y - pose.y
        if math.hypot(dx, dy) < self._cfg.position_tolerance:
            return True
        if len(self._path) < 2 or self.remaining > self._cfg.lookahead:
            return False
        # AMCL 보정 등으로 끝점을 살짝 지나쳤으면 되돌아가지 않는다.
        prev = self._path[-2]
        length = math.hypot(end.x - prev.x, end.y - prev.y)
        ux, uy = (end.x - prev.x) / length, (end.y - prev.y) / length
        along = dx * ux + dy * uy
        lateral = abs(-dx * uy + dy * ux)
        return along <= 0.0 and lateral < self._cfg.overshoot_tolerance

    def _update_progress(self, pose: Pose2D) -> None:
        # 경로 앞쪽만 찾아 진행 위치가 뒤로 가지 않게 한다.
        window = range(self.progress, len(self._path))
        self.progress = min(
            window, key=lambda i: math.hypot(self._path[i].x - pose.x, self._path[i].y - pose.y))

    def _lookahead_point(self) -> Point2D:
        goal_arc = self._arc[self.progress] + self._cfg.lookahead
        for i in range(self.progress, len(self._path)):
            if self._arc[i] >= goal_arc:
                return self._path[i]
        return self._path[-1]

    def _rotate(self, yaw_error: float, tolerance: float, next_phase: RoutePhase) -> Velocity:
        yaw_error = normalize_angle(yaw_error)
        if abs(yaw_error) < tolerance and abs(self._last.linear) < 1e-3:
            self.phase = next_phase
            return self._output(Velocity(0.0, self._last.angular)
                                if next_phase is RoutePhase.TRACK else STOP)
        # 오차가 줄수록 감속해서 목표 각도에 부드럽게 멈춘다.
        speed = max(self._cfg.min_angular_speed,
                    min(self._cfg.angular_speed, ROTATE_GAIN * abs(yaw_error)))
        dv = self._cfg.linear_accel * self._period
        dw = self._cfg.angular_accel * self._period
        return self._output(Velocity(
            _approach(self._last.linear, 0.0, dv),
            _approach(self._last.angular, math.copysign(speed, yaw_error), dw)))

    def _output(self, velocity: Velocity) -> Velocity:
        self._last = velocity
        return velocity


# ------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ApproachParams:
    """
    approach 노드 파라미터와 기본값.

    기본값은 namespace 없는 중립 값이고, 로봇별 값은 config/params.yaml 에서 준다.
    target/cancel/status 는 팀 인터페이스라 namespace 에 따라 바뀌지 않도록 절대 이름이다.
    """

    nav_action: str = 'navigate_to_pose'
    target_topic: str = '/approach/target_point'
    cancel_topic: str = '/approach/cancel'
    status_topic: str = '/approach/status'
    map_frame: str = 'map'
    # 로봇 위치는 TF map_frame -> base_frame 으로 구한다
    base_frame: str = 'base_link'

    # 경로상 자동차까지 이 거리 [m] 가 남으면 goal 을 취소하고 ARRIVED
    standoff_distance: float = 1.0
    # goal 재전송 최소 간격 [s]
    goal_period: float = 0.5
    # 자동차가 진행 중인 goal(또는 도착 위치)에서 이만큼 [m] 이상 움직여야 goal 을 다시 보낸다
    goal_move_threshold: float = 0.2

    # Nav2 실패 시 cmd_vel 로 직접 이동 (fallback)
    fallback_enabled: bool = True
    cmd_vel_topic: str = 'cmd_vel'
    # 경로 시작점과 goal 전에 거칠 경로점 [x0, y0, x1, y1, ...] (map) [m]
    fallback_waypoints: tuple = (0.0, 0.0, 2.0, 0.0, 1.79, 1.77)
    # 경로점 모서리를 둥글게 보간하는 원호 반지름 [m]. 0 이면 꺾은선 그대로
    fallback_corner_radius: float = 0.3
    fallback_lookahead: float = 0.25
    fallback_linear_speed: float = 0.15
    fallback_angular_speed: float = 0.6
    fallback_linear_accel: float = 0.3
    fallback_angular_accel: float = 1.5
    fallback_position_tolerance: float = 0.03
    fallback_yaw_tolerance: float = 0.05
    fallback_heading_tolerance: float = 0.8
    # TF 로 로봇 위치를 이 시간 동안 못 구하면 정지 후 FAILED [s]
    fallback_pose_timeout: float = 1.0
    # fallback 전체 제한 시간 [s]
    fallback_timeout: float = 90.0

    @property
    def waypoints(self) -> list[Point2D]:
        """Fallback_waypoints 를 (x, y) 점 목록으로 바꾼다."""
        flat = list(self.fallback_waypoints)
        if len(flat) % 2 or not flat:
            raise ValueError('fallback_waypoints must be non-empty [x0, y0, x1, y1, ...]')
        return [Point2D(float(x), float(y)) for x, y in zip(flat[::2], flat[1::2])]

    @property
    def route_config(self) -> RouteConfig:
        """Fallback 파라미터로 RouteConfig 를 만든다."""
        return RouteConfig(
            linear_speed=self.fallback_linear_speed,
            angular_speed=self.fallback_angular_speed,
            linear_accel=self.fallback_linear_accel,
            angular_accel=self.fallback_angular_accel,
            lookahead=self.fallback_lookahead,
            position_tolerance=self.fallback_position_tolerance,
            yaw_tolerance=self.fallback_yaw_tolerance,
            heading_tolerance=self.fallback_heading_tolerance,
        )

    @classmethod
    def declare_and_load(cls, node: Node) -> 'ApproachParams':
        """모든 필드를 ROS 파라미터로 선언하고 현재 값을 읽는다."""
        defaults = cls()
        values = {}
        for field in fields(cls):
            node.declare_parameter(field.name, getattr(defaults, field.name))
            values[field.name] = node.get_parameter(field.name).value
        return cls(**values)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# ------------------------------------------------------------------------- node
@dataclass
class _GoalRequest:
    """
    전송한 goal 한 건의 상태.

    콜백은 자신의 _GoalRequest 가 현재 활성 goal 인지 identity 로 비교해서,
    교체된 이전 goal 의 응답/feedback/결과를 무시한다.
    """

    car: Point2D
    goal: Pose2D
    handle: Optional[ClientGoalHandle] = None
    cancel_requested: bool = False


class ApproachNode(Node):
    """자동차 좌표를 받아 standoff 앞까지 Nav2 로 이동하고, 상태를 status 토픽으로 알린다."""

    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        self._params = ApproachParams.declare_and_load(self)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._active_goal: Optional[_GoalRequest] = None
        self._last_send = None
        # 도착 판정 당시 자동차 위치. 자동차가 여기서 많이 움직이면 다시 출발한다.
        self._arrived_car: Optional[Point2D] = None
        self._status = ApproachStatus.IDLE

        self._waypoints = self._params.waypoints
        self._route: Optional[RouteFollower] = None
        self._route_started = None
        self._last_pose_time = None

        self._nav = ActionClient(self, NavigateToPose, self._params.nav_action)
        self.create_subscription(
            PointStamped, self._params.target_topic, self._on_target,
            SUBSCRIPTION_QUEUE_DEPTH)
        self.create_subscription(
            Empty, self._params.cancel_topic, self._on_cancel, SUBSCRIPTION_QUEUE_DEPTH)
        self._status_pub = self.create_publisher(
            String, self._params.status_topic, LATCHED_QOS)
        self._cmd_vel_pub = self.create_publisher(
            TwistStamped, self._params.cmd_vel_topic, SUBSCRIPTION_QUEUE_DEPTH)
        self.create_timer(FALLBACK_PERIOD_SEC, self._on_route_timer)

        self._set_status(ApproachStatus.IDLE)
        self.get_logger().info(
            f'approach ready: action={self._params.nav_action}, '
            f'target={self._params.target_topic}, cancel={self._params.cancel_topic}, '
            f'standoff={self._params.standoff_distance}m, '
            f'fallback={self._params.fallback_enabled} '
            f'waypoints={[(p.x, p.y) for p in self._waypoints]}')

    def shutdown(self) -> None:
        """종료 전에 진행 중인 goal 이 있으면 취소 요청을 보내고, fallback 이동은 멈춘다."""
        if self._active_goal is not None and self._active_goal.handle is not None:
            self._active_goal.handle.cancel_goal_async()
        if self._route is not None:
            self._route = None
            self._publish_velocity(STOP)

    # ------------------------------------------------------------ callbacks
    def _on_target(self, msg: PointStamped) -> None:
        if not self._validate_target(msg):
            return
        if self._route is not None:
            return  # fallback 주행 중에는 목표를 바꾸지 않는다
        car = Point2D(msg.point.x, msg.point.y)

        # 이미 도착했고 자동차가 거의 그대로면 도착 상태 유지 (goal 재전송/취소 반복 방지)
        if (self._arrived_car is not None
                and _distance(car, self._arrived_car) < self._params.goal_move_threshold):
            return

        robot = self._robot_pose()
        if robot is None:
            self.get_logger().warning(
                f'TF {self._params.map_frame}->{self._params.base_frame} 없음, target ignored',
                throttle_duration_sec=3.0)
            return

        # 이미 standoff 안이면 goal 을 보내지 않고 도착으로 본다
        straight = _distance(car, robot.position)
        if straight < self._params.standoff_distance:
            self._arrive(car, f'직선거리 {straight:.2f}m')
            return

        active = self._active_goal
        if active is not None and _distance(car, active.car) <= self._params.goal_move_threshold:
            return  # 진행 중인 goal 과 거의 같은 위치 -> 재전송하면 매번 경로 재계획
        if (self._last_send is not None
                and (self.get_clock().now() - self._last_send).nanoseconds / 1e9
                < self._params.goal_period):
            return
        self._send_goal(car, robot)

    def _on_goal_response(self, future: Future, request: _GoalRequest) -> None:
        handle: ClientGoalHandle = future.result()
        if request is not self._active_goal:
            self._discard_stale_goal(handle)
            return
        if not handle.accepted:
            self.get_logger().error('goal rejected by Nav2')
            self._active_goal = None
            self._fail_or_fallback(request)
            return
        self._track_accepted_goal(request, handle)

    def _on_feedback(self, msg, request: _GoalRequest) -> None:
        if request is not self._active_goal:
            return
        remaining = msg.feedback.distance_remaining
        # 경로 계획 전 feedback 은 0 이 올 수 있어 제외한다
        if 0.01 < remaining < self._params.standoff_distance:
            self._arrive(request.car, f'경로상 남은 거리 {remaining:.2f}m')

    def _on_result(self, future: Future, request: _GoalRequest) -> None:
        if request is not self._active_goal:
            return
        self._active_goal = None
        self._report_nav_result(future.result().status, request)

    def _on_cancel(self, _msg: Empty) -> None:
        self._arrived_car = None
        if self._route is not None:
            self.get_logger().info('canceling fallback route')
            self._finish_route(ApproachStatus.CANCELED)
            return
        if self._status is not ApproachStatus.MOVING or self._active_goal is None:
            self.get_logger().info('cancel requested but nothing is moving')
            return
        self._active_goal.cancel_requested = True
        self._cancel_goal(self._active_goal)

    # ------------------------------------------------------- target -> goal
    def _validate_target(self, msg: PointStamped) -> bool:
        if msg.header.frame_id != self._params.map_frame:
            self.get_logger().warning(
                f"target frame is '{msg.header.frame_id}', "
                f"expected '{self._params.map_frame}'. ignored"
            )
            return False

        return True

    def _robot_pose(self) -> Optional[Pose2D]:
        """TF map_frame -> base_frame 로 구한 로봇 자세. 없으면 None."""
        try:
            t = self._tf_buffer.lookup_transform(
                self._params.map_frame, self._params.base_frame, Time())
        except TransformException:
            return None
        q = t.transform.rotation
        return Pose2D(t.transform.translation.x, t.transform.translation.y,
                      quaternion_to_yaw(QuaternionXYZW(q.x, q.y, q.z, q.w)))

    def _send_goal(self, car: Point2D, robot: Pose2D) -> None:
        goal_pose = compute_goal_pose(robot.position, car)
        request = _GoalRequest(car=car, goal=goal_pose)
        self._arrived_car = None
        self._last_send = self.get_clock().now()

        if not self._nav.server_is_ready():
            self.get_logger().error('Nav2 action server not available')
            self._fail_or_fallback(request)
            return

        goal_msg = NavigateToPose.Goal(pose=self._to_pose_stamped(goal_pose))

        # 새 goal 을 보내면 Nav2 가 이전 goal 을 선점(preempt)하므로 이전 goal 을 따로 취소하지 않는다.
        self._active_goal = request

        self.get_logger().info(
            f'car=({car.x:.3f}, {car.y:.3f}) goal sent '
            f'(stop when {self._params.standoff_distance:.1f}m left on path)')

        self._set_status(ApproachStatus.MOVING)

        future = self._nav.send_goal_async(
            goal_msg, feedback_callback=partial(self._on_feedback, request=request))
        future.add_done_callback(partial(self._on_goal_response, request=request))

    def _arrive(self, car: Point2D, why: str) -> None:
        """진행 중인 goal 을 취소하고 ARRIVED 를 알린다."""
        request = self._active_goal
        self._active_goal = None  # 취소 결과(CANCELED)는 _on_result 에서 무시된다
        if request is not None:
            request.cancel_requested = True
            if request.handle is not None:
                request.handle.cancel_goal_async()
        self._arrived_car = car
        if self._status is not ApproachStatus.ARRIVED:
            self.get_logger().info(f'arrived ({why}): car=({car.x:.2f}, {car.y:.2f})')
            self._set_status(ApproachStatus.ARRIVED)

    def _to_pose_stamped(self, pose: Pose2D) -> PoseStamped:
        q = yaw_to_quaternion(pose.yaw)
        msg = PoseStamped()
        msg.header.frame_id = self._params.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.orientation.x = q.x
        msg.pose.orientation.y = q.y
        msg.pose.orientation.z = q.z
        msg.pose.orientation.w = q.w
        return msg

    # ---------------------------------------------------- goal lifecycle
    def _discard_stale_goal(self, handle: ClientGoalHandle) -> None:
        # 응답 전에 새 goal 이 나갔거나 도착 처리됐다면, 뒤늦게 수락된 이 goal 이 로봇을 움직이지 않게 취소한다.
        if handle.accepted:
            handle.cancel_goal_async()

    def _track_accepted_goal(self, request: _GoalRequest, handle: ClientGoalHandle) -> None:
        request.handle = handle
        if request.cancel_requested:
            self._cancel_goal(request)
        handle.get_result_async().add_done_callback(
            partial(self._on_result, request=request))

    def _cancel_goal(self, request: _GoalRequest) -> None:
        if request.handle is None:
            # 수락 응답이 오면 _track_accepted_goal 이 cancel_requested 를 보고 취소한다.
            self.get_logger().info('cancel pending (goal not accepted yet)')
            return
        self.get_logger().info('canceling goal')
        # 최종 CANCELED 상태는 Nav2 결과(_on_result)로 발행한다.
        request.handle.cancel_goal_async()

    def _report_nav_result(self, nav_status: int, request: _GoalRequest) -> None:
        status = NAV_RESULT_TO_STATUS.get(nav_status)
        if status is None:
            self.get_logger().error(f'navigation failed (status code {nav_status})')
            if request.cancel_requested:
                self._set_status(ApproachStatus.FAILED)
            else:
                self._fail_or_fallback(request)
            return
        if status is ApproachStatus.ARRIVED:
            self._arrived_car = request.car
        self._set_status(status)

    # --------------------------------------------------- cmd_vel fallback
    def _fail_or_fallback(self, request: _GoalRequest) -> None:
        if not self._params.fallback_enabled:
            self._set_status(ApproachStatus.FAILED)
            return
        robot = self._robot_pose()
        if robot is None:
            self.get_logger().error('fallback unavailable: no TF robot pose')
            self._set_status(ApproachStatus.FAILED)
            return

        car = request.car
        points = self._waypoints + [car]
        start = route_start_index(robot.position, points)
        path = smooth_path([robot.position] + points[start:], self._params.fallback_corner_radius)
        # 자동차 standoff 앞에서 멈추고 마지막에 자동차를 바라본다
        path = trim_path_end(path, self._params.standoff_distance)
        if len(path) < 2:
            self._arrive(car, 'fallback 경로가 standoff 보다 짧음')
            return
        end = path[-1]
        final_yaw = math.atan2(car.y - end.y, car.x - end.x)
        self._route = RouteFollower(
            path, final_yaw, self._params.route_config, FALLBACK_PERIOD_SEC)
        self._route_started = self.get_clock().now()
        self._last_pose_time = self._route_started
        self.get_logger().warning(
            f'Nav2 failed -> cmd_vel fallback from ({robot.x:.2f}, {robot.y:.2f}) via '
            f'{[(round(p.x, 2), round(p.y, 2)) for p in points[start:]]} '
            f'until {self._params.standoff_distance:.1f}m before car')
        # mission_manager 가 취소를 보낼 수 있도록 이동 중 상태를 유지한다.
        self._set_status(ApproachStatus.MOVING)

    def _on_route_timer(self) -> None:
        if self._route is None:
            return
        now = self.get_clock().now()
        if (now - self._route_started).nanoseconds / 1e9 > self._params.fallback_timeout:
            self.get_logger().error('fallback route timed out')
            self._finish_route(ApproachStatus.FAILED)
            return

        pose = self._robot_pose()
        if pose is None:
            if (now - self._last_pose_time).nanoseconds / 1e9 > self._params.fallback_pose_timeout:
                self.get_logger().error('fallback stopped: TF robot pose is stale')
                self._finish_route(ApproachStatus.FAILED)
            return
        self._last_pose_time = now

        phase = self._route.phase
        velocity = self._route.step(pose)
        if self._route.phase is not phase:
            self.get_logger().info(
                f'fallback {self._route.phase.value} remaining={self._route.remaining:.2f}m '
                f'pose=({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})')
        if self._route.done:
            self._finish_route(ApproachStatus.ARRIVED)
            return
        self._publish_velocity(velocity)

    def _finish_route(self, status: ApproachStatus) -> None:
        self._route = None
        self._publish_velocity(STOP)
        self._set_status(status)

    def _publish_velocity(self, velocity: Velocity) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = velocity.linear
        msg.twist.angular.z = velocity.angular
        self._cmd_vel_pub.publish(msg)

    def _set_status(self, status: ApproachStatus) -> None:
        self._status = status
        self._status_pub.publish(String(data=status.value))
        self.get_logger().info(f'status: {status.value}')


def main(args: Optional[list[str]] = None) -> None:
    """Approach 노드를 실행하고, 종료 시 진행 중인 goal 을 취소한다."""
    rclpy.init(args=args)
    node = ApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
