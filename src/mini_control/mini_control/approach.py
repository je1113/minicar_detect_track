#!/usr/bin/env python3
"""
자동차 근처로 Nav2 goal 을 보내 이동하고 취소 요청을 처리하는 노드.

토픽 이름, 메시지 타입, 상태 문자열은 mission_manager / localizer 와의 팀 인터페이스다.
  target_topic  geometry_msgs/PointStamped (frame=map) : 자동차의 지도 좌표
  cancel_topic  std_msgs/Empty                         : 이동 취소 요청
  status_topic  std_msgs/String (transient local)      : IDLE/MOVING/ARRIVED/FAILED/CANCELED

목표 좌표를 계산했다고 도달이 보장되지는 않으므로, 최종 상태는 Nav2 결과로 정한다.
Nav2 가 실패하면(서버 없음/거절/abort) fallback_waypoints 를 거쳐 goal 까지
cmd_vel 로 직접 이동하고(제자리 회전 → 직진 반복), 그 결과를 최종 상태로 쓴다.
"""
from dataclasses import dataclass, fields
from enum import Enum
from functools import partial
import math
from typing import NamedTuple, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (
    PointStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped)
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.task import Future
from std_msgs.msg import Empty, String

NODE_NAME = 'approach'
SUBSCRIPTION_QUEUE_DEPTH = 10
FALLBACK_PERIOD_SEC = 0.1

# 늦게 구독한 노드(mission_manager)도 마지막 상태를 받고, AMCL 이 latched 로 발행한
# 마지막 pose 도 받을 수 있도록 transient local 을 쓴다.
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


def relative_pose(base: Pose2D, pose: Pose2D) -> Pose2D:
    """기준 자세 base 의 좌표계에서 본 pose 의 상대 자세를 구한다."""
    dx, dy = pose.x - base.x, pose.y - base.y
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    return Pose2D(c * dx + s * dy, -s * dx + c * dy, normalize_angle(pose.yaw - base.yaw))


def compose_pose(base: Pose2D, delta: Pose2D) -> Pose2D:
    """기준 자세 base 에서 그 좌표계 기준 delta 만큼 움직인 자세를 구한다(relative_pose 의 역)."""
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    return Pose2D(base.x + c * delta.x - s * delta.y,
                  base.y + s * delta.x + c * delta.y,
                  normalize_angle(base.yaw + delta.yaw))


def compute_view_pose(
    car: Point2D,
    offset_x: float,
    offset_y: float
) -> Pose2D:
    """
    자동차 위치를 기준으로 OAK-D에서 잘 보이는 위치를 계산한다.

    goal 위치:
        goal_x = car_x + offset_x
        goal_y = car_y + offset_y

    yaw:
        goal 위치에서 자동차를 바라보도록 계산
    """

    goal_x = car.x + offset_x
    goal_y = car.y + offset_y

    yaw = math.atan2(
        car.y - goal_y,
        car.x - goal_x
    )

    return Pose2D(
        goal_x,
        goal_y,
        yaw
    )


# --------------------------------------------------------------- cmd_vel route
class Velocity(NamedTuple):
    """Cmd_vel 로 보낼 직진 [m/s], 회전 [rad/s] 속도."""

    linear: float
    angular: float


STOP = Velocity(0.0, 0.0)


@dataclass(frozen=True)
class RouteConfig:
    """RouteFollower 제어 값."""

    linear_speed: float = 0.15
    angular_speed: float = 0.6
    min_angular_speed: float = 0.15
    # 이 거리 안에 들어오면 경로점에 도착한 것으로 본다 [m]
    position_tolerance: float = 0.08
    # 제자리 회전을 끝내는 방향 오차 [rad]
    yaw_tolerance: float = 0.05
    # 직진 중 방향 오차가 이보다 커지면 멈추고 다시 제자리 회전 [rad]
    heading_tolerance: float = 0.4


class RoutePhase(str, Enum):
    """RouteFollower 진행 단계."""

    ROTATE = 'ROTATE'
    DRIVE = 'DRIVE'
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


class RouteFollower:
    """
    경로점을 차례로 '제자리 회전 → 직진' 하고 마지막에 goal yaw 로 맞추는 제어기.

    ROS 와 독립적이며, 매 주기 step(현재 map 자세) 로 속도를 얻는다.
    """

    def __init__(self, points: list[Point2D], final_yaw: float, config: RouteConfig,
                 start_index: int = 0) -> None:
        """points[start_index] 부터 차례로 이동하고 마지막에 final_yaw 로 맞춘다."""
        self._points = points
        self._final_yaw = final_yaw
        self._cfg = config
        self.index = start_index
        self.phase = RoutePhase.ROTATE

    @property
    def done(self) -> bool:
        """Goal 위치와 yaw 까지 모두 맞췄는지."""
        return self.phase is RoutePhase.DONE

    def step(self, pose: Pose2D) -> Velocity:
        """현재 자세에서 보낼 속도를 계산하고 단계를 진행한다."""
        if self.phase is RoutePhase.DONE:
            return STOP
        if self.phase is RoutePhase.FINAL_ROTATE:
            return self._rotate_to(self._final_yaw - pose.yaw, RoutePhase.DONE)

        target = self._points[self.index]
        dx, dy = target.x - pose.x, target.y - pose.y
        if math.hypot(dx, dy) < self._cfg.position_tolerance:
            self.index += 1
            self.phase = (RoutePhase.FINAL_ROTATE if self.index >= len(self._points)
                          else RoutePhase.ROTATE)
            return STOP

        heading_error = normalize_angle(math.atan2(dy, dx) - pose.yaw)
        if self.phase is RoutePhase.ROTATE:
            return self._rotate_to(heading_error, RoutePhase.DRIVE)

        if abs(heading_error) > self._cfg.heading_tolerance:
            self.phase = RoutePhase.ROTATE
            return STOP
        angular = max(-self._cfg.angular_speed,
                      min(self._cfg.angular_speed, 2.0 * heading_error))
        return Velocity(self._cfg.linear_speed, angular)

    def _rotate_to(self, yaw_error: float, next_phase: RoutePhase) -> Velocity:
        yaw_error = normalize_angle(yaw_error)
        if abs(yaw_error) < self._cfg.yaw_tolerance:
            self.phase = next_phase
            return STOP
        speed = max(self._cfg.min_angular_speed,
                    min(self._cfg.angular_speed, 1.5 * abs(yaw_error)))
        return Velocity(0.0, math.copysign(speed, yaw_error))


# ------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ApproachParams:
    """
    approach 노드 파라미터와 기본값.

    기본값은 namespace 없는 중립 값이고, 로봇별 값은 config/params.yaml 에서 준다.
    target/cancel/status 는 팀 인터페이스라 namespace 에 따라 바뀌지 않도록 절대 이름이다.
    """

    nav_action: str = 'navigate_to_pose'
    # TF 대신 amcl_pose 를 쓰면 namespace 별 TF 리매핑 없이 로봇 위치를 얻을 수 있다.
    pose_topic: str = 'amcl_pose'
    target_topic: str = '/approach/target_point'
    cancel_topic: str = '/approach/cancel'
    status_topic: str = '/approach/status'
    map_frame: str = 'map'

    # 자동차 기준, OAK-D가 잘 보이는 AMR 위치
    view_offset_x: float = 0.6755
    view_offset_y: float = 0.8150

    # Nav2 실패 시 cmd_vel 로 직접 이동 (fallback)
    fallback_enabled: bool = True
    cmd_vel_topic: str = 'cmd_vel'
    # amcl_pose 갱신 사이의 이동을 odom 으로 보간한다.
    odom_topic: str = 'odom'
    # 경로 시작점과 goal 전에 거칠 경로점 [x0, y0, x1, y1, ...] (map) [m]
    fallback_waypoints: tuple = (0.0, 0.0, 2.0, 0.0, 1.79, 1.77)
    fallback_linear_speed: float = 0.15
    fallback_angular_speed: float = 0.6
    fallback_position_tolerance: float = 0.08
    fallback_yaw_tolerance: float = 0.05
    fallback_heading_tolerance: float = 0.4
    # odom 이 이 시간 동안 안 들어오면 정지 후 FAILED [s]
    fallback_odom_timeout: float = 1.0
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


# ------------------------------------------------------------------------- node
@dataclass
class _GoalRequest:
    """
    전송한 goal 한 건의 상태.

    콜백은 자신의 _GoalRequest 가 현재 활성 goal 인지 identity 로 비교해서,
    교체된 이전 goal 의 응답/결과를 무시한다.
    """

    car: Point2D
    goal: Pose2D
    handle: Optional[ClientGoalHandle] = None
    cancel_requested: bool = False


class ApproachNode(Node):
    """자동차 좌표를 받아 그 앞까지 Nav2 로 이동하고, 상태를 status 토픽으로 알린다."""

    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        self._params = ApproachParams.declare_and_load(self)

        # map 자세 = 마지막 amcl_pose + (그 이후 odom 이동량)
        self._amcl_pose: Optional[Pose2D] = None
        self._odom_at_amcl: Optional[Pose2D] = None
        self._odom_pose: Optional[Pose2D] = None
        self._odom_stamp = None
        self._active_goal: Optional[_GoalRequest] = None
        self._status = ApproachStatus.IDLE

        self._waypoints = self._params.waypoints
        self._route: Optional[RouteFollower] = None
        self._route_started = None

        self._nav = ActionClient(self, NavigateToPose, self._params.nav_action)
        self.create_subscription(
            PoseWithCovarianceStamped, self._params.pose_topic, self._on_pose, LATCHED_QOS)
        self.create_subscription(
            Odometry, self._params.odom_topic, self._on_odom, qos_profile_sensor_data)
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
    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        o = pose.orientation
        self._amcl_pose = Pose2D(
            pose.position.x, pose.position.y,
            quaternion_to_yaw(QuaternionXYZW(o.x, o.y, o.z, o.w)))
        self._odom_at_amcl = self._odom_pose

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        o = pose.orientation
        self._odom_pose = Pose2D(
            pose.position.x, pose.position.y,
            quaternion_to_yaw(QuaternionXYZW(o.x, o.y, o.z, o.w)))
        self._odom_stamp = self.get_clock().now()
        if self._odom_at_amcl is None:
            self._odom_at_amcl = self._odom_pose

    def _on_target(self, msg: PointStamped) -> None:
        if not self._validate_target(msg):
            return
        if self._status is ApproachStatus.MOVING:
            self.get_logger().info('already moving, target ignored')
            return
        self._send_goal(Point2D(msg.point.x, msg.point.y))

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

    def _on_result(self, future: Future, request: _GoalRequest) -> None:
        if request is not self._active_goal:
            return
        self._active_goal = None
        self._report_nav_result(future.result().status, request)

    def _on_cancel(self, _msg: Empty) -> None:
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

    def _send_goal(self, car: Point2D) -> None:
        goal_pose = compute_view_pose(car, self._params.view_offset_x, self._params.view_offset_y)
        request = _GoalRequest(car=car, goal=goal_pose)

        if not self._nav.server_is_ready():
            self.get_logger().error('Nav2 action server not available')
            self._fail_or_fallback(request)
            return

        goal_msg = NavigateToPose.Goal(pose=self._to_pose_stamped(goal_pose))

        # 새 goal 을 보내면 Nav2 가 이전 goal 을 선점(preempt)하므로 이전 goal 을 따로 취소하지 않는다.
        self._active_goal = request

        self.get_logger().info(
            f'car=({car.x:.3f}, {car.y:.3f}) '
            f'-> goal=({goal_pose.x:.3f}, {goal_pose.y:.3f}) '
            f'yaw={goal_pose.yaw:.3f}'
        )

        self._set_status(ApproachStatus.MOVING)

        future = self._nav.send_goal_async(goal_msg)
        future.add_done_callback(partial(self._on_goal_response, request=request))

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
        # 응답 전에 새 goal 이 나갔다면, 뒤늦게 수락된 이 goal 이 로봇을 움직이지 않게 취소한다.
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
        self._set_status(status)

    # --------------------------------------------------- cmd_vel fallback
    def _fail_or_fallback(self, request: _GoalRequest) -> None:
        if not self._params.fallback_enabled:
            self._set_status(ApproachStatus.FAILED)
            return
        robot = self._current_pose()
        if robot is None:
            self.get_logger().error('fallback unavailable: no amcl_pose yet')
            self._set_status(ApproachStatus.FAILED)
            return

        goal = request.goal
        points = self._waypoints + [goal.position]
        start = route_start_index(robot.position, points)
        self._route = RouteFollower(points, goal.yaw, self._params.route_config, start)
        self._route_started = self.get_clock().now()
        self.get_logger().warning(
            f'Nav2 failed -> cmd_vel fallback from ({robot.x:.2f}, {robot.y:.2f}) via '
            f'{[(round(p.x, 2), round(p.y, 2)) for p in points[start:]]} '
            f'yaw={goal.yaw:.2f}')
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
        if (self._odom_stamp is None or (now - self._odom_stamp).nanoseconds / 1e9
                > self._params.fallback_odom_timeout):
            self.get_logger().error('fallback stopped: odom is stale')
            self._finish_route(ApproachStatus.FAILED)
            return

        pose = self._current_pose()
        phase, index = self._route.phase, self._route.index
        velocity = self._route.step(pose)
        if (self._route.phase, self._route.index) != (phase, index):
            self.get_logger().info(
                f'fallback {self._route.phase.value} point={self._route.index} '
                f'pose=({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})')
        if self._route.done:
            self._finish_route(ApproachStatus.ARRIVED)
            return
        self._publish_velocity(velocity)

    def _finish_route(self, status: ApproachStatus) -> None:
        self._route = None
        self._publish_velocity(STOP)
        self._set_status(status)

    def _current_pose(self) -> Optional[Pose2D]:
        if self._amcl_pose is None:
            return None
        if self._odom_pose is None or self._odom_at_amcl is None:
            return self._amcl_pose
        return compose_pose(self._amcl_pose, relative_pose(self._odom_at_amcl, self._odom_pose))

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
