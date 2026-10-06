#!/usr/bin/env python3
"""
자동차 근처로 Nav2 goal 을 보내 이동하고 취소 요청을 처리하는 노드.

토픽 이름, 메시지 타입, 상태 문자열은 mission_manager / localizer 와의 팀 인터페이스다.
  target_topic  geometry_msgs/PointStamped (frame=map) : 자동차의 지도 좌표
  cancel_topic  std_msgs/Empty                         : 이동 취소 요청
  status_topic  std_msgs/String (transient local)      : IDLE/MOVING/ARRIVED/FAILED/CANCELED

목표 좌표를 계산했다고 도달이 보장되지는 않으므로, 최종 상태는 Nav2 결과로만 정한다.
"""
from dataclasses import dataclass, fields
from enum import Enum
from functools import partial
import math
from typing import NamedTuple, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from std_msgs.msg import Empty, String

NODE_NAME = 'approach'
SUBSCRIPTION_QUEUE_DEPTH = 10

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
    handle: Optional[ClientGoalHandle] = None
    cancel_requested: bool = False


class ApproachNode(Node):
    """자동차 좌표를 받아 그 앞까지 Nav2 로 이동하고, 상태를 status 토픽으로 알린다."""

    def __init__(self) -> None:
        super().__init__(NODE_NAME)
        self._params = ApproachParams.declare_and_load(self)

        self._robot_pose: Optional[Pose2D] = None
        self._active_goal: Optional[_GoalRequest] = None
        self._status = ApproachStatus.IDLE

        self._nav = ActionClient(self, NavigateToPose, self._params.nav_action)
        self.create_subscription(
            PoseWithCovarianceStamped, self._params.pose_topic, self._on_pose, LATCHED_QOS)
        self.create_subscription(
            PointStamped, self._params.target_topic, self._on_target,
            SUBSCRIPTION_QUEUE_DEPTH)
        self.create_subscription(
            Empty, self._params.cancel_topic, self._on_cancel, SUBSCRIPTION_QUEUE_DEPTH)
        self._status_pub = self.create_publisher(
            String, self._params.status_topic, LATCHED_QOS)

        self._set_status(ApproachStatus.IDLE)
        self.get_logger().info(
            f'approach ready: action={self._params.nav_action}, '
            f'target={self._params.target_topic}, cancel={self._params.cancel_topic}')

    def shutdown(self) -> None:
        """종료 전에 진행 중인 goal 이 있으면 취소 요청을 보낸다."""
        if self._active_goal is not None and self._active_goal.handle is not None:
            self._active_goal.handle.cancel_goal_async()

    # ------------------------------------------------------------ callbacks
    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        o = pose.orientation
        self._robot_pose = Pose2D(
            pose.position.x, pose.position.y,
            quaternion_to_yaw(QuaternionXYZW(o.x, o.y, o.z, o.w)))

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
            self._set_status(ApproachStatus.FAILED)
            return
        self._track_accepted_goal(request, handle)

    def _on_result(self, future: Future, request: _GoalRequest) -> None:
        if request is not self._active_goal:
            return
        self._active_goal = None
        self._report_nav_result(future.result().status)

    def _on_cancel(self, _msg: Empty) -> None:
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
        if not self._nav.server_is_ready():
            self.get_logger().error('Nav2 action server not available')
            self._set_status(ApproachStatus.FAILED)
            return

        goal_pose = compute_view_pose(car, self._params.view_offset_x, self._params.view_offset_y)
        goal_msg = NavigateToPose.Goal(pose=self._to_pose_stamped(goal_pose))

        # 새 goal 을 보내면 Nav2 가 이전 goal 을 선점(preempt)하므로 이전 goal 을 따로 취소하지 않는다.
        request = _GoalRequest(car=car)
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

    def _report_nav_result(self, nav_status: int) -> None:
        status = NAV_RESULT_TO_STATUS.get(nav_status)
        if status is None:
            self.get_logger().error(f'navigation failed (status code {nav_status})')
            status = ApproachStatus.FAILED
        self._set_status(status)

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
