"""
웹캠 발견 → Nav2 접근 → AMR 카메라 인계 → 자동차 추종을 관리한다.

상태:
SEARCHING
APPROACHING
HANDOVER
FOLLOWING
LOST
DOCKING  (배터리 부족: 도크 앞까지 Nav2 이동 후 Dock 액션)
DOCKED

approach 연결:
  웹캠 좌표를 approach 목표 토픽으로 넘기고,
  approach 상태(ARRIVED/FAILED/CANCELED)로 다음 상태를 정한다.

배터리:
  battery_low_threshold 미만이 되면 어떤 상태든 미션을 멈추고 DOCKING으로 간다.
"""

import math

from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import Dock
from irobot_create_msgs.msg import DockStatus
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import BatteryState

from std_msgs.msg import Empty, String
from vision_msgs.msg import Detection2DArray

from .approach import ApproachStatus
from .follow import compute_velocity, stop_velocity

# approach가 이동을 끝냈음을 뜻하는 상태
APPROACH_FINISHED = (
    ApproachStatus.ARRIVED,
    ApproachStatus.FAILED,
    ApproachStatus.CANCELED,
)


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        # =====================================================
        # 1. ROS2 파라미터 선언
        # =====================================================

        self.declare_parameter(
            'target_position_topic',
            '/target/map_position'
        )

        self.declare_parameter(
            'amr_detection_topic',
            '/amr/detections'
        )

        self.declare_parameter(
            'approach_target_topic',
            '/approach/target_point'
        )

        self.declare_parameter(
            'approach_cancel_topic',
            '/approach/cancel'
        )

        self.declare_parameter(
            'approach_status_topic',
            '/approach/status'
        )

        self.declare_parameter(
            'cmd_vel_topic',
            '/robot2/cmd_vel'
        )

        self.declare_parameter(
            'map_frame',
            'map'
        )

        self.declare_parameter(
            'image_width',
            640
        )

        self.declare_parameter(
            'target_distance',
            0.8
        )

        self.declare_parameter(
            'min_distance',
            0.5
        )

        self.declare_parameter(
            'linear_gain',
            0.5
        )

        self.declare_parameter(
            'angular_gain',
            1.0
        )

        self.declare_parameter(
            'max_linear_speed',
            0.31
        )

        self.declare_parameter(
            'max_angular_speed',
            1.0
        )

        # 자동차 중심이 화면 가운데 이 비율 안에 있으면 회전하지 않는다
        # (1/3 → 화면 가운데 1/3 구간)
        self.declare_parameter(
            'center_deadband_ratio',
            1.0 / 3.0
        )

        self.declare_parameter(
            'detection_timeout',
            0.5
        )

        # 자동차가 이 시간 동안 한 번도 감지되지 않아야 LOST로 간다 [s]
        self.declare_parameter(
            'lost_timeout',
            2.0
        )

        # LOST 상태에서 자동차를 다시 찾을 때 제자리 회전 속도 [rad/s]
        # 마지막으로 본 쪽으로 돈다. 본 적이 없으면 부호대로 돈다.
        # 양수: 반시계(좌회전), 음수: 시계(우회전)
        self.declare_parameter(
            'lost_angular_speed',
            0.03
        )

        # approach가 도착(yaw 정렬까지 완료)한 뒤 정지한 채
        # 자동차를 감지하며 기다리는 시간 [s]
        self.declare_parameter(
            'arrival_hold_time',
            2.0
        )

        # -----------------------------------------------------
        # 배터리 / 도킹
        # -----------------------------------------------------
        self.declare_parameter(
            'battery_topic',
            '/robot2/battery_state'
        )

        # 배터리 잔량(0~1)이 이 값 미만이면 도킹하러 간다
        self.declare_parameter(
            'battery_low_threshold',
            0.2
        )

        self.declare_parameter(
            'dock_status_topic',
            '/robot2/dock_status'
        )

        self.declare_parameter(
            'nav_action',
            '/robot2/navigate_to_pose'
        )

        self.declare_parameter(
            'dock_action',
            '/robot2/dock'
        )

        # 도크 앞 대기 자세 (map) [m, m, rad]
        # 이 자세에서 Dock 액션이 도크를 볼 수 있어야 한다 (도크 정면 약 0.5~1m)
        self.declare_parameter(
            'dock_staging_x',
            0.0
        )

        self.declare_parameter(
            'dock_staging_y',
            0.0
        )

        self.declare_parameter(
            'dock_staging_yaw',
            0.0
        )

        # Dock 액션 실패 시 다시 시도하는 횟수
        self.declare_parameter(
            'dock_max_retries',
            2
        )

        # =====================================================
        # 2. 파라미터 읽기
        # =====================================================

        self.target_position_topic = (
            self.get_parameter(
                'target_position_topic'
            ).value
        )

        self.amr_detection_topic = (
            self.get_parameter(
                'amr_detection_topic'
            ).value
        )

        self.approach_target_topic = (
            self.get_parameter(
                'approach_target_topic'
            ).value
        )

        self.approach_cancel_topic = (
            self.get_parameter(
                'approach_cancel_topic'
            ).value
        )

        self.approach_status_topic = (
            self.get_parameter(
                'approach_status_topic'
            ).value
        )

        self.cmd_vel_topic = (
            self.get_parameter(
                'cmd_vel_topic'
            ).value
        )

        self.map_frame = (
            self.get_parameter(
                'map_frame'
            ).value
        )

        self.image_width = int(
            self.get_parameter(
                'image_width'
            ).value
        )

        self.target_distance = float(
            self.get_parameter(
                'target_distance'
            ).value
        )

        self.min_distance = float(
            self.get_parameter(
                'min_distance'
            ).value
        )

        self.linear_gain = float(
            self.get_parameter(
                'linear_gain'
            ).value
        )

        self.angular_gain = float(
            self.get_parameter(
                'angular_gain'
            ).value
        )

        self.max_linear_speed = float(
            self.get_parameter(
                'max_linear_speed'
            ).value
        )

        self.max_angular_speed = float(
            self.get_parameter(
                'max_angular_speed'
            ).value
        )

        self.center_deadband_ratio = float(
            self.get_parameter(
                'center_deadband_ratio'
            ).value
        )

        self.detection_timeout = float(
            self.get_parameter(
                'detection_timeout'
            ).value
        )

        self.lost_timeout = float(
            self.get_parameter(
                'lost_timeout'
            ).value
        )

        self.lost_angular_speed = float(
            self.get_parameter(
                'lost_angular_speed'
            ).value
        )

        self.arrival_hold_time = float(
            self.get_parameter(
                'arrival_hold_time'
            ).value
        )

        self.battery_topic = (
            self.get_parameter(
                'battery_topic'
            ).value
        )

        self.battery_low_threshold = float(
            self.get_parameter(
                'battery_low_threshold'
            ).value
        )

        self.dock_status_topic = (
            self.get_parameter(
                'dock_status_topic'
            ).value
        )

        self.nav_action = (
            self.get_parameter(
                'nav_action'
            ).value
        )

        self.dock_action = (
            self.get_parameter(
                'dock_action'
            ).value
        )

        self.dock_staging_x = float(
            self.get_parameter(
                'dock_staging_x'
            ).value
        )

        self.dock_staging_y = float(
            self.get_parameter(
                'dock_staging_y'
            ).value
        )

        self.dock_staging_yaw = float(
            self.get_parameter(
                'dock_staging_yaw'
            ).value
        )

        self.dock_max_retries = int(
            self.get_parameter(
                'dock_max_retries'
            ).value
        )

        # =====================================================
        # 3. 상태값
        # =====================================================
        self.state = 'SEARCHING'

        self.target_detected = False
        self.target_center_x = None

        # amr_detector가 /amr/detections에 넣어 주는 자동차까지 거리 [m]
        # depth 측정이 한 프레임 실패해도 직전 유효 거리를 잠시 쓴다.
        self.distance = None
        self.distance_time = None

        self.last_detection_time = None

        self.handover_start_time = None

        # HANDOVER 진입 후 추종을 시작하지 않고 정지해 있을 시간 [s]
        self.handover_hold_time = 0.0

        # =====================================================
        # 4. approach 상태
        # =====================================================
        self.approach_status = None

        # AMR 카메라가 먼저 자동차를 찾아 Nav2 취소를 요청한 상태
        self.handover_cancel_requested = False

        # =====================================================
        # 4-1. 배터리 / 도킹 상태
        # =====================================================
        self.battery_percentage = None
        self.is_docked = False

        # 도킹 단계: WAIT_APPROACH → NAVIGATING → DOCKING_ACTION
        self.dock_phase = None
        self.dock_start_time = None
        self.dock_retries = 0

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.nav_action
        )

        self.dock_client = ActionClient(
            self,
            Dock,
            self.dock_action
        )

        # =====================================================
        # 5. approach Publisher
        # =====================================================
        self.approach_target_pub = self.create_publisher(
            PointStamped,
            self.approach_target_topic,
            10
        )

        self.approach_cancel_pub = self.create_publisher(
            Empty,
            self.approach_cancel_topic,
            10
        )

        # =====================================================
        # 6. TurtleBot4 속도 Publisher
        # =====================================================
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10
        )

        # =====================================================
        # 7. Subscriber
        # =====================================================

        # 웹캠으로 계산된 자동차 map 위치
        self.create_subscription(
            PointStamped,
            self.target_position_topic,
            self.target_position_callback,
            10
        )

        # approach는 상태를 transient local로 발행하지만,
        # 이전 실행의 오래된 상태를 받지 않도록 volatile로 구독한다.
        self.create_subscription(
            String,
            self.approach_status_topic,
            self.approach_status_callback,
            10
        )

        # AMR 카메라 자동차 감지
        self.create_subscription(
            Detection2DArray,
            self.amr_detection_topic,
            self.amr_detection_callback,
            10
        )

        # TurtleBot4 배터리 상태
        self.create_subscription(
            BatteryState,
            self.battery_topic,
            self.battery_callback,
            10
        )

        # TurtleBot4 도킹 여부
        self.create_subscription(
            DockStatus,
            self.dock_status_topic,
            self.dock_status_callback,
            10
        )

        # =====================================================
        # 8. 제어 Timer
        # =====================================================
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Mission Manager 시작'
        )

    # =========================================================
    # 웹캠 자동차 map 위치 수신
    # =========================================================
    def target_position_callback(self, msg):

        if msg.header.frame_id != self.map_frame:
            self.get_logger().warning(
                f'잘못된 frame: {msg.header.frame_id}'
            )
            return

        # 첫 자동차 위치만 approach에 전달
        if self.state != 'SEARCHING':
            return

        self.get_logger().info(
            f'car map position=({msg.point.x:.3f}, '
            f'{msg.point.y:.3f})'
        )

        self.approach_target_pub.publish(msg)

        self.state = 'APPROACHING'

        self.get_logger().info(
            'SEARCHING -> APPROACHING'
        )

    # =========================================================
    # approach 상태 수신
    # =========================================================
    def approach_status_callback(self, msg):

        try:
            status = ApproachStatus(msg.data)
        except ValueError:
            self.get_logger().warning(
                f'알 수 없는 approach 상태: {msg.data}'
            )
            return

        self.approach_status = status

        if self.state != 'APPROACHING':
            return

        if status not in APPROACH_FINISHED:
            return

        # AMR 카메라가 이미 자동차를 보고 있으므로
        # Nav2가 어떻게 끝났든 추종 인계로 넘어간다.
        if self.handover_cancel_requested:
            self.handover_cancel_requested = False
            self.notify_approach_complete()
            return

        # 도착 후 자세가 잡힌 상태에서 잠시 멈춰 감지한다
        if status == ApproachStatus.ARRIVED:
            self.notify_approach_complete(
                hold_time=self.arrival_hold_time
            )
            return

        self.state = 'SEARCHING'

        self.get_logger().warning(
            f'APPROACHING -> SEARCHING (approach {status.value})'
        )

    # =========================================================
    # approach 이동 취소 요청
    # =========================================================
    def cancel_approach(self):

        self.handover_cancel_requested = True

        self.approach_cancel_pub.publish(Empty())

        self.get_logger().info(
            'AMR 카메라 감지, approach 취소 요청'
        )

    # =========================================================
    # 접근 완료 → AMR 카메라 인계
    # =========================================================
    def notify_approach_complete(self, hold_time=0.0):

        if self.state != 'APPROACHING':
            return

        self.start_handover('APPROACHING', hold_time)

    # =========================================================
    # HANDOVER 진입: 정지 후 hold_time이 지나고 자동차가 보이면 추종한다
    # =========================================================
    def start_handover(self, from_state, hold_time=0.0):

        self.state = 'HANDOVER'
        self.handover_start_time = self.get_clock().now()
        self.handover_hold_time = hold_time

        self.stop_robot()

        self.get_logger().info(
            f'{from_state} -> HANDOVER (hold {hold_time:.1f}s)'
        )

    # =========================================================
    # AMR 카메라 감지 결과
    # =========================================================
    def amr_detection_callback(self, msg):
        # car 클래스만 필터링
        car_detections = []

        for detection in msg.detections:
            if len(detection.results) == 0:
                continue

            class_id = detection.results[0].hypothesis.class_id

            if class_id == 'car':
                car_detections.append(detection)

        # car가 없는 프레임은 감지 실패로만 표시한다.
        # 한두 프레임 놓친 것으로 상태를 바꾸지 않도록
        # 마지막 위치/거리는 지우지 않고, LOST 판정은 lost_timeout에 맡긴다.
        if len(car_detections) == 0:
            self.target_detected = False
            return

        # car가 여러 개라면 confidence 가장 높은 car 사용
        detection = max(
            car_detections,
            key=lambda d: float(
                d.results[0].hypothesis.score
            )
        )

        self.target_detected = True

        self.target_center_x = float(
            detection.bbox.center.position.x
        )

        distance = float(
            detection.results[0].pose.pose.position.z
        )

        self.last_detection_time = (
            self.get_clock().now()
        )

        # 0.0은 depth 측정 실패이므로 직전 유효 거리를 유지한다.
        if distance > 0.0:
            self.distance = distance
            self.distance_time = self.last_detection_time

        score = float(
            detection.results[0].hypothesis.score
        )

        self.get_logger().info(
            f'AMR car detected: '
            f'x={self.target_center_x:.1f}, '
            f'score={score:.2f}'
        )

    # =========================================================
    # 추종 시작
    # =========================================================
    def start_following(self, from_state):

        self.state = 'FOLLOWING'

        self.get_logger().info(
            f'{from_state} -> FOLLOWING'
        )

    # =========================================================
    # 감지 결과가 아직 유효한지 확인
    # =========================================================
    def seconds_since(self, stamp):

        if stamp is None:
            return float('inf')

        return (
            self.get_clock().now() - stamp
        ).nanoseconds / 1e9

    def detection_is_fresh(self):

        return (
            self.seconds_since(self.last_detection_time)
            <= self.detection_timeout
        )

    # =========================================================
    # lost_timeout 동안 자동차를 한 번도 못 봤는지 확인
    # =========================================================
    def target_is_lost(self, since=None):

        elapsed = self.seconds_since(self.last_detection_time)

        # HANDOVER는 진입 시각부터 기다린다
        if since is not None:
            elapsed = min(elapsed, self.seconds_since(since))

        return elapsed > self.lost_timeout

    # =========================================================
    # 배터리 상태 수신
    # =========================================================
    def battery_callback(self, msg):

        self.battery_percentage = float(msg.percentage)

        if self.state in ('DOCKING', 'DOCKED'):
            return

        if self.battery_percentage >= self.battery_low_threshold:
            return

        self.get_logger().warning(
            f'배터리 {self.battery_percentage * 100:.0f}% '
            f'(< {self.battery_low_threshold * 100:.0f}%), 도킹 시작'
        )

        self.start_docking()

    def dock_status_callback(self, msg):

        self.is_docked = bool(msg.is_docked)

    # =========================================================
    # DOCKING 진입: 미션을 멈추고 approach 이동을 취소한다
    # =========================================================
    def start_docking(self):

        from_state = self.state

        # 이미 도크 위면 Nav2로 끌어내지 않는다
        if self.is_docked:
            self.state = 'DOCKED'
            self.stop_robot()

            self.get_logger().info(
                f'{from_state} -> DOCKED (이미 도킹됨)'
            )
            return

        self.state = 'DOCKING'
        self.dock_phase = 'WAIT_APPROACH'
        self.dock_start_time = self.get_clock().now()
        self.dock_retries = 0

        self.stop_robot()

        # approach의 Nav2 goal / fallback 주행이 남아 있으면 취소한다
        self.approach_cancel_pub.publish(Empty())

        self.get_logger().info(
            f'{from_state} -> DOCKING'
        )

    # =========================================================
    # 도크 앞 대기 자세로 Nav2 이동
    # =========================================================
    def send_dock_staging_goal(self):

        self.dock_phase = 'NAVIGATING'

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f'{self.nav_action} 서버 없음, 바로 Dock 시도'
            )
            self.send_dock_goal()
            return

        goal = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.dock_staging_x
        pose.pose.position.y = self.dock_staging_y
        pose.pose.orientation.z = math.sin(self.dock_staging_yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.dock_staging_yaw / 2.0)

        goal.pose = pose

        self.get_logger().info(
            f'도크 앞으로 이동: ({self.dock_staging_x:.2f}, '
            f'{self.dock_staging_y:.2f}, {self.dock_staging_yaw:.2f})'
        )

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.dock_staging_goal_response)

    def dock_staging_goal_response(self, future):

        handle = future.result()

        if not handle.accepted:
            self.get_logger().error('도크 앞 이동 goal 거절, 바로 Dock 시도')
            self.send_dock_goal()
            return

        handle.get_result_async().add_done_callback(
            self.dock_staging_result
        )

    def dock_staging_result(self, future):

        status = future.result().status

        # Nav2가 실패해도 도크 근처일 수 있으므로 Dock은 시도한다
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warning(
                f'도크 앞 이동 실패 (status {status}), Dock 시도'
            )

        self.send_dock_goal()

    # =========================================================
    # Create3 Dock 액션
    # =========================================================
    def send_dock_goal(self):

        self.dock_phase = 'DOCKING_ACTION'

        if not self.dock_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f'{self.dock_action} 서버 없음, 도킹 실패'
            )
            self.dock_phase = 'FAILED'
            return

        self.get_logger().info('Dock 액션 호출')

        future = self.dock_client.send_goal_async(Dock.Goal())
        future.add_done_callback(self.dock_goal_response)

    def dock_goal_response(self, future):

        handle = future.result()

        if not handle.accepted:
            self.get_logger().error('Dock goal 거절')
            self.dock_failed()
            return

        handle.get_result_async().add_done_callback(
            self.dock_result
        )

    def dock_result(self, future):

        result = future.result()

        if (
            result.status == GoalStatus.STATUS_SUCCEEDED
            and result.result.is_docked
        ):
            self.state = 'DOCKED'
            self.dock_phase = None

            self.get_logger().info('DOCKING -> DOCKED')
            return

        self.get_logger().error(
            f'Dock 실패 (status {result.status})'
        )

        self.dock_failed()

    def dock_failed(self):

        if self.dock_retries < self.dock_max_retries:
            self.dock_retries += 1

            self.get_logger().warning(
                f'도크 앞으로 다시 이동 후 재시도 '
                f'({self.dock_retries}/{self.dock_max_retries})'
            )

            self.send_dock_staging_goal()
            return

        self.dock_phase = 'FAILED'

        self.get_logger().error('도킹 실패, 정지한 채 대기')

    # =========================================================
    # 상태 제어
    # =========================================================
    def control_loop(self):

        # -----------------------------------------------------
        # DOCKING / DOCKED
        #
        # Nav2와 Dock 액션이 제어한다. cmd_vel을 보내지 않는다.
        # -----------------------------------------------------
        if self.state == 'DOCKING':

            # approach 취소가 끝날 때까지(최대 3초) 기다렸다가 Nav2 goal을 보낸다.
            # 바로 보내면 approach goal을 선점해 approach가 fallback으로 갈 수 있다.
            if self.dock_phase == 'WAIT_APPROACH':
                if (
                    self.approach_status != ApproachStatus.MOVING
                    or self.seconds_since(self.dock_start_time) > 3.0
                ):
                    self.send_dock_staging_goal()

            return

        if self.state == 'DOCKED':
            return

        # -----------------------------------------------------
        # SEARCHING
        # -----------------------------------------------------
        if self.state == 'SEARCHING':
            return

        # -----------------------------------------------------
        # APPROACHING
        #
        # 이 상태에서는 Nav2 / approach.py가 제어한다.
        # cmd_vel을 여기서 보내지 않는다.
        # -----------------------------------------------------
        if self.state == 'APPROACHING':

            # approach가 MOVING을 알리기 전에 취소하면 무시되므로 기다린다.
            if (
                not self.handover_cancel_requested
                and self.approach_status == ApproachStatus.MOVING
                and self.target_detected
                and self.detection_is_fresh()
            ):
                self.cancel_approach()

            return

        # -----------------------------------------------------
        # HANDOVER
        # -----------------------------------------------------
        if self.state == 'HANDOVER':

            # hold_time 동안은 감지만 하고 정지해 있는다
            holding = (
                self.seconds_since(self.handover_start_time)
                < self.handover_hold_time
            )

            # hold가 끝난 뒤 자동차가 보이면 바로 추종
            if (
                not holding
                and self.target_detected
                and self.detection_is_fresh()
            ):
                self.approach_cancel_pub.publish(
                    Empty()
                )

                self.start_following('HANDOVER')

                return

            self.stop_robot()

            if holding:
                return

            # 멈춘 뒤 lost_timeout 동안 자동차가 안 보이면 다시 회전하며 찾는다
            if self.target_is_lost(since=self.handover_start_time):
                self.state = 'LOST'

                self.get_logger().warning(
                    'HANDOVER -> LOST'
                )

            return

        # -----------------------------------------------------
        # FOLLOWING
        # -----------------------------------------------------
        if self.state == 'FOLLOWING':

            # lost_timeout 동안 한 번도 못 봤을 때만 LOST
            if self.target_is_lost():
                self.state = 'LOST'

                self.stop_robot()

                self.get_logger().warning(
                    'FOLLOWING -> LOST'
                )

                return

            # 잠깐 놓쳤거나 거리가 오래되면 멈춰서 다시 보일 때까지 기다린다
            if (
                not self.detection_is_fresh()
                or self.seconds_since(self.distance_time)
                > self.detection_timeout
            ):

                self.stop_robot()

                return

            # 직전 감지(detection_timeout 이내)로 계속 추종한다
            linear_x, angular_z = compute_velocity(
                target_detected=True,
                target_center_x=self.target_center_x,
                image_width=self.image_width,
                distance=self.distance,
                target_distance=self.target_distance,
                min_distance=self.min_distance,
                linear_gain=self.linear_gain,
                angular_gain=self.angular_gain,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed,
                center_deadband=self.center_deadband_ratio,
            )

            self.publish_velocity(
                linear_x,
                angular_z
            )

            return

        # -----------------------------------------------------
        # LOST
        # -----------------------------------------------------
        if self.state == 'LOST':

            # 자동차가 한 번이라도 보이면 바로 추종
            if (
                self.target_detected
                and self.detection_is_fresh()
            ):
                self.start_following('LOST')

                return

            # 찾을 때까지 마지막으로 본 쪽으로 제자리 회전
            self.publish_velocity(
                0.0,
                self.lost_search_angular_speed()
            )

            return

    # =========================================================
    # LOST 회전 방향: 화면 왼쪽에서 사라졌으면 좌회전, 오른쪽이면 우회전
    # =========================================================
    def lost_search_angular_speed(self):

        if self.target_center_x is None:
            return self.lost_angular_speed

        speed = abs(self.lost_angular_speed)

        if self.target_center_x < self.image_width / 2.0:
            return speed

        return -speed

    # =========================================================
    # 속도 명령 발행
    # =========================================================
    def publish_velocity(
        self,
        linear_x,
        angular_z
    ):

        msg = TwistStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(angular_z)

        self.cmd_vel_pub.publish(msg)

    # =========================================================
    # TurtleBot4 정지
    # =========================================================
    def stop_robot(self):

        linear_x, angular_z = stop_velocity()

        self.publish_velocity(
            linear_x,
            angular_z
        )


def main(args=None):

    rclpy.init(args=args)

    node = MissionManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
