"""
웹캠 발견 → Nav2 접근 → AMR 카메라로 Nav2 추종 → 놓치면 360도 탐색을 관리한다.

상태:
SEARCHING    웹캠 좌표를 기다린다
APPROACHING  웹캠 좌표(자동차 위치)를 approach 로 넘겨 Nav2 로 접근한다.
             웹캠 좌표 근처에 처음 도착하기 전에는 AMR 카메라에 보여도 추종으로 넘어가지 않는다.
FOLLOWING    AMR 카메라 bbox 방향 + depth 거리로 자동차 위치를 카메라 frame 에 만들고
             TF 로 map 좌표로 바꿔 approach 로 넘긴다 → 추종 중에도 Nav2 가 장애물을 피한다.
             approach 가 ARRIVED(경로상 standoff 이내)면 제자리 회전으로 자동차를 화면 가운데에 둔다.
LOST         lost_timeout 동안 못 보면 approach 를 취소하고 마지막으로 본 쪽으로 제자리 360도 회전.
             보이면 FOLLOWING, 한 바퀴 돌아도 없으면 웹캠 좌표로 다시 APPROACHING (없으면 SEARCHING).
DOCKING      배터리 부족: 도크 앞까지 Nav2 이동 후 Dock 액션
DOCKED

approach 연결:
  자동차 map 좌표를 approach 목표 토픽으로 계속 넘긴다. goal 재전송 간격/이동 임계값과
  standoff 도착 판정은 approach 가 한다. 상태는 /approach/status 로 받는다.

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
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from sensor_msgs.msg import BatteryState

from std_msgs.msg import Empty, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from .approach import ApproachStatus


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
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

        # 탐색 회전량을 잴 때 쓰는 frame (map 은 AMCL 보정으로 yaw 가 튈 수 있다)
        self.declare_parameter(
            'odom_frame',
            'odom'
        )

        self.declare_parameter(
            'base_frame',
            'base_link'
        )

        # 감지 결과 header 에 frame_id 가 없을 때 쓰는 카메라 frame
        self.declare_parameter(
            'camera_frame',
            'oakd_rgb_camera_optical_frame'
        )

        self.declare_parameter(
            'image_width',
            704
        )

        # OAK-D RGB 수평 화각 [deg]. bbox 가로 위치 → 방향각 변환에 사용
        self.declare_parameter(
            'camera_hfov_deg',
            69.0
        )

        # 도착(standoff 이내) 후 제자리 회전: 화면 중심 오차 → 회전 속도 gain
        self.declare_parameter(
            'angular_gain',
            1.2
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

        # 추종 중 자동차를 이 시간 동안 한 번도 못 보면 LOST [s]
        # (그 전까지는 진행 중인 Nav2 goal 을 그대로 둔다)
        self.declare_parameter(
            'lost_timeout',
            3.0
        )

        # LOST 360도 탐색 회전 속도 [rad/s]. 마지막으로 본 쪽으로 돈다.
        # 빠르면 영상이 흔들려 YOLO 가 놓친다
        self.declare_parameter(
            'lost_angular_speed',
            0.03
        )

        # 회전량을 못 재는 경우(TF 없음 등) 대비 탐색 최대 시간 [s]
        self.declare_parameter(
            'lost_search_timeout',
            25.0
        )

        # 웹캠 좌표가 이 시간 [s] 보다 오래되면 접근에 쓰지 않는다
        self.declare_parameter(
            'webcam_target_timeout',
            4.0
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
        p = self.get_parameter

        self.target_position_topic = p('target_position_topic').value
        self.amr_detection_topic = p('amr_detection_topic').value
        self.approach_target_topic = p('approach_target_topic').value
        self.approach_cancel_topic = p('approach_cancel_topic').value
        self.approach_status_topic = p('approach_status_topic').value
        self.cmd_vel_topic = p('cmd_vel_topic').value

        self.map_frame = p('map_frame').value
        self.odom_frame = p('odom_frame').value
        self.base_frame = p('base_frame').value
        self.camera_frame = p('camera_frame').value

        self.image_width = int(p('image_width').value)
        self.camera_hfov = math.radians(float(p('camera_hfov_deg').value))
        self.angular_gain = float(p('angular_gain').value)
        self.max_angular_speed = float(p('max_angular_speed').value)
        self.center_deadband_ratio = float(p('center_deadband_ratio').value)

        self.detection_timeout = float(p('detection_timeout').value)
        self.lost_timeout = float(p('lost_timeout').value)
        self.lost_angular_speed = float(p('lost_angular_speed').value)
        self.lost_search_timeout = float(p('lost_search_timeout').value)
        self.webcam_target_timeout = float(p('webcam_target_timeout').value)

        self.battery_topic = p('battery_topic').value
        self.battery_low_threshold = float(p('battery_low_threshold').value)
        self.dock_status_topic = p('dock_status_topic').value
        self.nav_action = p('nav_action').value
        self.dock_action = p('dock_action').value
        self.dock_staging_x = float(p('dock_staging_x').value)
        self.dock_staging_y = float(p('dock_staging_y').value)
        self.dock_staging_yaw = float(p('dock_staging_yaw').value)
        self.dock_max_retries = int(p('dock_max_retries').value)

        # =====================================================
        # 3. 상태값
        # =====================================================
        self.state = 'SEARCHING'

        # 웹캠 자동차 map 좌표 (수신 시각 기준으로 최신성 판단)
        self.webcam_target = None
        self.webcam_target_time = None

        # 웹캠 좌표 근처에 한 번이라도 도착했는지.
        # 그 전에는 AMR 카메라에 보여도 추종으로 넘어가지 않는다.
        self.arrived_once = False

        self.target_detected = False
        self.target_center_x = None
        self.detection_frame = None

        # amr_detector가 /amr/detections에 넣어 주는 자동차까지 거리 [m]
        # depth 측정이 한 프레임 실패해도 직전 유효 거리를 잠시 쓴다.
        self.distance = None
        self.distance_time = None

        self.last_detection_time = None

        # 제자리 회전 중 (cmd_vel 직접 발행 중)
        self.rotating = False

        # LOST 탐색 상태: {'dir', 'turned', 'prev_yaw', 'start'}
        self.search = None

        # =====================================================
        # 4. approach 상태
        # =====================================================
        self.approach_status = None

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
        # 4-2. TF (카메라 → map 변환, 탐색 회전량)
        # =====================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        self.webcam_target = msg
        self.webcam_target_time = self.get_clock().now()

        if self.state == 'SEARCHING':
            self.start_approach('SEARCHING')
            return

        # 접근 중에는 최신 좌표를 계속 넘긴다
        # (자동차가 움직이면 approach가 goal을 다시 보낸다)
        if self.state == 'APPROACHING':
            self.approach_target_pub.publish(msg)

    def webcam_target_is_fresh(self):

        return (
            self.webcam_target is not None
            and self.seconds_since(self.webcam_target_time)
            <= self.webcam_target_timeout
        )

    # =========================================================
    # 웹캠 좌표로 접근 시작
    # =========================================================
    def start_approach(self, from_state):

        self.stop_rotation()

        self.state = 'APPROACHING'

        self.approach_target_pub.publish(self.webcam_target)

        self.get_logger().info(
            f'{from_state} -> APPROACHING '
            f'(car map=({self.webcam_target.point.x:.3f}, '
            f'{self.webcam_target.point.y:.3f}))'
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

        # FAILED / CANCELED 는 다음 좌표가 오면 approach가 goal을 다시 보낸다
        if (
            self.state == 'APPROACHING'
            and status == ApproachStatus.ARRIVED
            and not self.arrived_once
        ):
            self.arrived_once = True

            self.get_logger().info(
                '웹캠 좌표 도착: AMR 카메라에 자동차가 보이면 추종 시작'
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

        self.detection_frame = msg.header.frame_id or self.camera_frame

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
            f'dist={self.distance}, '
            f'score={score:.2f}',
            throttle_duration_sec=1.0
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

    # 지금 자동차가 보이고 거리도 유효한지
    def car_is_visible(self):

        return (
            self.target_detected
            and self.detection_is_fresh()
            and self.seconds_since(self.distance_time)
            <= self.detection_timeout
        )

    # 화면 가로 위치 오차: -1(왼쪽) ~ 1(오른쪽)
    def horizontal_error(self):

        center = self.image_width / 2.0

        return (self.target_center_x - center) / center

    # =========================================================
    # AMR 감지 → 자동차 map 좌표 (TF 변환)
    #
    # bbox 가로 위치를 화각으로 방향각으로 바꾸고, depth(광축 방향 거리)로
    # 카메라 frame 의 점을 만든 뒤 TF 로 map 으로 변환한다.
    # 카메라와 로봇의 시계가 다를 수 있어 최신 TF 를 쓴다.
    # =========================================================
    def car_map_position(self):

        lateral = (
            self.distance
            * self.horizontal_error()
            * math.tan(self.camera_hfov / 2.0)
        )

        point = PointStamped()
        point.header.frame_id = self.detection_frame

        if self.detection_frame.endswith('optical_frame'):
            # optical: x 오른쪽, y 아래, z 앞
            point.point.x = lateral
            point.point.z = self.distance
        else:
            # 일반 ROS frame: x 앞, y 왼쪽
            point.point.x = self.distance
            point.point.y = -lateral

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.detection_frame,
                Time()
            )
        except TransformException as e:
            self.get_logger().warning(
                f'TF {self.map_frame}->{self.detection_frame} 없음: {e}',
                throttle_duration_sec=3.0
            )
            return None

        car = do_transform_point(point, transform)
        car.header.frame_id = self.map_frame
        car.header.stamp = self.get_clock().now().to_msg()

        return car

    def odom_yaw(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                Time()
            )
        except TransformException:
            return None

        return _yaw_from_quaternion(transform.transform.rotation)

    # =========================================================
    # 추종 시작 / LOST 탐색 시작
    # =========================================================
    def start_following(self, from_state):

        self.state = 'FOLLOWING'

        self.get_logger().info(
            f'{from_state} -> FOLLOWING (Nav2로 추종)'
        )

    def start_search(self):

        # 진행 중인 Nav2 goal 취소 (제자리 회전과 충돌 방지)
        self.approach_cancel_pub.publish(Empty())

        # 마지막에 오른쪽에서 봤으면 시계 방향
        direction = 1.0
        if (
            self.target_center_x is not None
            and self.horizontal_error() > 0.0
        ):
            direction = -1.0

        self.search = {
            'dir': direction,
            'turned': 0.0,
            'prev_yaw': self.odom_yaw(),
            'start': self.get_clock().now(),
        }

        self.state = 'LOST'

        self.get_logger().warning(
            f'FOLLOWING -> LOST ({self.lost_timeout:.0f}초간 못 봄, '
            f'{"시계" if direction < 0 else "반시계"} 방향 360도 탐색)'
        )

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
        # Nav2 / approach.py가 제어한다. cmd_vel을 여기서 보내지 않는다.
        # 웹캠 좌표에 한 번 도착한 뒤 AMR 카메라에 보이면 추종한다.
        # -----------------------------------------------------
        if self.state == 'APPROACHING':

            if self.arrived_once and self.car_is_visible():
                self.start_following('APPROACHING')

            return

        # -----------------------------------------------------
        # FOLLOWING
        # -----------------------------------------------------
        if self.state == 'FOLLOWING':

            if self.seconds_since(self.last_detection_time) > self.lost_timeout:
                self.stop_rotation()
                self.start_search()
                return

            # 잠깐 놓친 경우(lost_timeout 이내)는 진행 중인 Nav2 goal을 그대로 둔다
            if not self.car_is_visible():
                self.stop_rotation()
                return

            car = self.car_map_position()

            if car is not None:
                self.approach_target_pub.publish(car)

            # standoff 이내: 이동 없이 제자리 회전으로 화면 가운데 유지
            # Nav2가 움직이는 중에는 cmd_vel을 보내지 않는다
            if self.approach_status == ApproachStatus.ARRIVED:
                self.rotate_to_center()
            else:
                self.stop_rotation()

            return

        # -----------------------------------------------------
        # LOST
        # -----------------------------------------------------
        if self.state == 'LOST':
            self.search_step()
            return

    # =========================================================
    # 도착 후 제자리 회전: 자동차를 화면 가운데에 둔다
    # =========================================================
    def rotate_to_center(self):

        error = self.horizontal_error()

        angular_z = 0.0

        if abs(error) > self.center_deadband_ratio:
            angular_z = _clamp(
                -self.angular_gain * error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )

        self.publish_velocity(0.0, angular_z)

        self.rotating = angular_z != 0.0

    def stop_rotation(self):

        if self.rotating:
            self.stop_robot()
            self.rotating = False

    # =========================================================
    # LOST: 제자리 360도 회전 탐색
    # =========================================================
    def search_step(self):

        if self.car_is_visible():
            self.stop_rotation()
            self.search = None
            self.get_logger().info('탐색 회전 중 자동차 발견')
            self.start_following('LOST')
            return

        st = self.search

        yaw = self.odom_yaw()

        if yaw is not None and st['prev_yaw'] is not None:
            d = math.atan2(
                math.sin(yaw - st['prev_yaw']),
                math.cos(yaw - st['prev_yaw'])
            )
            st['turned'] += abs(d)

        if yaw is not None:
            st['prev_yaw'] = yaw

        if (
            st['turned'] >= 2.0 * math.pi
            or self.seconds_since(st['start']) > self.lost_search_timeout
        ):
            self.stop_rotation()
            self.search = None

            self.get_logger().warning(
                f'{math.degrees(st["turned"]):.0f}도 회전했지만 자동차 못 찾음'
            )

            if self.webcam_target_is_fresh():
                self.start_approach('LOST')
            else:
                self.state = 'SEARCHING'
                self.get_logger().warning(
                    'LOST -> SEARCHING (웹캠 좌표 대기)'
                )

            return

        self.publish_velocity(
            0.0,
            st['dir'] * abs(self.lost_angular_speed)
        )

        self.rotating = True

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

        self.publish_velocity(0.0, 0.0)


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
