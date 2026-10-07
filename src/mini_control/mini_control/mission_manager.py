"""
웹캠 발견 → Nav2 접근 → AMR 카메라 인계 → 자동차 추종을 관리한다.

상태:
SEARCHING
APPROACHING
HANDOVER
FOLLOWING
LOST

approach 연결:
  웹캠 좌표를 approach 목표 토픽으로 넘기고,
  approach 상태(ARRIVED/FAILED/CANCELED)로 다음 상태를 정한다.

추종 방식(follow_mode 파라미터):
  cmd_vel (기본)  AMR 카메라 감지로 속도를 계산해 cmd_vel 로 직접 보낸다.
  nav             AMR 카메라 감지를 map 좌표로 바꿔 approach 목표 토픽으로 보내고
                  Nav2 가 따라가게 한다. 이때 추종 중에는 cmd_vel 을 보내지 않고,
                  추종을 멈출 때는 approach 취소 토픽을 쓴다.
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import (
    PointStamped,
    PoseWithCovarianceStamped,
    TwistStamped,
)
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Empty, String
from vision_msgs.msg import Detection2DArray

from .approach import (
    ApproachStatus,
    LATCHED_QOS,
    quaternion_to_yaw,
    QuaternionXYZW,
)
from .follow import (
    compute_velocity,
    detection_to_map_point,
    should_publish_follow_target,
    stop_velocity,
)

FOLLOW_MODE_CMD_VEL = 'cmd_vel'
FOLLOW_MODE_NAV = 'nav'
FOLLOW_MODES = (FOLLOW_MODE_CMD_VEL, FOLLOW_MODE_NAV)

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

        # 추종 방식: 'cmd_vel'(속도 직접 제어) 또는 'nav'(Nav2 goal)
        self.declare_parameter(
            'follow_mode',
            FOLLOW_MODE_CMD_VEL
        )

        # nav 모드에서 AMR 감지를 map 좌표로 바꿀 때 쓴다.
        self.declare_parameter(
            'camera_info_topic',
            '/robot2/oakd/rgb/camera_info'
        )

        self.declare_parameter(
            'robot_pose_topic',
            '/robot2/amcl_pose'
        )

        # 시간이 원인인지 확인하려고 추가했다: nav 모드에서 추종 목표를 보낼 때마다
        # 시각 정보를 CSV 로 저장하는 폴더, 빈 문자열이면 저장 안 함
        self.declare_parameter(
            'time_log_dir',
            '~/minicar_time_logs'
        )

        # nav 모드에서 자동차 좌표를 approach 로 보내는 최소 간격 [s]
        self.declare_parameter(
            'follow_publish_period',
            0.5
        )

        # 마지막으로 보낸 좌표에서 이만큼 움직여야 다시 보낸다 [m]
        self.declare_parameter(
            'follow_min_move',
            0.2
        )

        # 움직이지 않아도 이 시간마다 다시 보낸다 [s]
        self.declare_parameter(
            'follow_refresh_period',
            3.0
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

        self.follow_mode = str(
            self.get_parameter(
                'follow_mode'
            ).value
        )

        # 오타가 나면 조용히 다른 방식으로 움직이지 않도록 시작할 때 막는다.
        if self.follow_mode not in FOLLOW_MODES:
            raise ValueError(
                f'follow_mode 는 {FOLLOW_MODES} 중 하나여야 합니다: '
                f'{self.follow_mode!r}'
            )

        self.camera_info_topic = (
            self.get_parameter(
                'camera_info_topic'
            ).value
        )

        self.robot_pose_topic = (
            self.get_parameter(
                'robot_pose_topic'
            ).value
        )

        self.time_log_dir = str(
            self.get_parameter(
                'time_log_dir'
            ).value
        )

        self.follow_publish_period = float(
            self.get_parameter(
                'follow_publish_period'
            ).value
        )

        self.follow_min_move = float(
            self.get_parameter(
                'follow_min_move'
            ).value
        )

        self.follow_refresh_period = float(
            self.get_parameter(
                'follow_refresh_period'
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

        # nav 모드: AMR 감지를 map 좌표로 바꾸는 데 필요한 값
        self.camera_fx = None
        self.camera_cx = None

        # amcl_pose 로 받은 로봇 map 위치 (x, y, yaw)
        self.robot_pose = None

        # 시간이 원인인지 확인하려고 추가했다: 각 입력의 header 시각
        self.last_detection_stamp = None
        self.robot_pose_stamp = None

        self.time_csv = None
        self.time_csv_writer = None

        if self.follow_mode == FOLLOW_MODE_NAV:
            self.open_time_csv()

        # nav 모드: 마지막으로 approach 에 보낸 자동차 좌표 (x, y)와 시각
        self.last_follow_target = None
        self.last_follow_time = None

        # =====================================================
        # 4. approach 상태
        # =====================================================
        self.approach_status = None

        # AMR 카메라가 먼저 자동차를 찾아 Nav2 취소를 요청한 상태
        self.handover_cancel_requested = False

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

        # nav 모드에서만 필요하다. cmd_vel 모드는 기존과 구독이 같다.
        if self.follow_mode == FOLLOW_MODE_NAV:

            # 카메라 내부 파라미터(fx, cx)
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data
            )

            # approach 와 같이 amcl_pose 로 로봇의 map 위치를 얻는다.
            self.create_subscription(
                PoseWithCovarianceStamped,
                self.robot_pose_topic,
                self.robot_pose_callback,
                LATCHED_QOS
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

        # nav 모드에서는 approach 가 이미 끝났으므로 0 속도를 보내지 않는다.
        if self.follow_mode == FOLLOW_MODE_CMD_VEL:
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

        self.last_detection_stamp = msg.header.stamp

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

        if self.follow_mode == FOLLOW_MODE_NAV:

            # 추종 첫 좌표는 바로 보내도록 이전 기록을 지운다.
            self.last_follow_target = None
            self.last_follow_time = None

            # LOST 에서 제자리 회전하던 마지막 속도 명령이 남지 않게 한 번 멈춘다.
            if from_state == 'LOST':
                self.stop_robot()

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
    # 상태 제어
    # =========================================================
    def control_loop(self):

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

            if self.follow_mode == FOLLOW_MODE_CMD_VEL:
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

                if self.follow_mode == FOLLOW_MODE_NAV:
                    # Nav2 가 마지막 목표로 계속 가지 않게 한 번만 취소한다.
                    self.cancel_follow_goal()
                else:
                    self.stop_robot()

                self.get_logger().warning(
                    'FOLLOWING -> LOST'
                )

                return

            # nav 모드: 속도를 직접 만들지 않고 Nav2 목표를 보낸다.
            if self.follow_mode == FOLLOW_MODE_NAV:

                # 감지나 거리가 오래됐으면 새 좌표를 만들지 않고
                # 마지막 목표를 그대로 둔다.
                if (
                    self.detection_is_fresh()
                    and self.seconds_since(self.distance_time)
                    <= self.detection_timeout
                ):
                    self.follow_with_nav()

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
    # nav 모드: 카메라 내부 파라미터 / 로봇 위치 수신
    # =========================================================
    def camera_info_callback(self, msg):

        fx = float(msg.k[0])
        cx = float(msg.k[2])

        if fx <= 0.0:
            self.get_logger().warning(
                f'camera_info 의 fx 가 올바르지 않습니다: {fx}',
                throttle_duration_sec=5.0
            )
            return

        first = self.camera_fx is None

        self.camera_fx = fx
        self.camera_cx = cx

        if first:
            self.get_logger().info(
                f'camera_info 수신: fx={fx:.1f}, cx={cx:.1f}, '
                f'{msg.width}x{msg.height}'
            )

    def robot_pose_callback(self, msg):

        self.robot_pose_stamp = msg.header.stamp

        pose = msg.pose.pose
        q = pose.orientation

        self.robot_pose = (
            pose.position.x,
            pose.position.y,
            quaternion_to_yaw(
                QuaternionXYZW(q.x, q.y, q.z, q.w)
            ),
        )

    # =========================================================
    # nav 모드: 감지된 자동차를 Nav2 목표로 보냄
    # =========================================================
    def follow_with_nav(self):

        # 거리를 못 쟀으면 새 좌표를 만들 수 없다. 마지막 목표는 그대로 둔다.
        if self.distance is None:
            return

        if self.camera_fx is None or self.robot_pose is None:
            self.get_logger().warning(
                'camera_info / amcl_pose 수신 대기 중',
                throttle_duration_sec=2.0
            )
            return

        robot_x, robot_y, robot_yaw = self.robot_pose

        car = detection_to_map_point(
            self.target_center_x,
            self.distance,
            self.camera_fx,
            self.camera_cx,
            robot_x,
            robot_y,
            robot_yaw,
        )

        now = self.get_clock().now()

        elapsed = None
        if self.last_follow_time is not None:
            elapsed = (
                now - self.last_follow_time
            ).nanoseconds / 1e9

        if not should_publish_follow_target(
            self.last_follow_target,
            car,
            elapsed,
            self.follow_publish_period,
            self.follow_min_move,
            self.follow_refresh_period,
        ):
            return

        msg = PointStamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.map_frame

        msg.point.x = float(car[0])
        msg.point.y = float(car[1])
        msg.point.z = 0.0

        # goal 교체 여부와 stand-off 위치는 approach 가 정한다.
        self.approach_target_pub.publish(msg)

        self.last_follow_target = car
        self.last_follow_time = now

        # 시간이 원인인지 확인하려고 추가했다.
        # 감지 지연: 영상 촬영 후 지금까지 경과 시간
        # amcl_pose 나이: 이 좌표 계산에 쓴 로봇 위치가 몇 초 전 것인지
        detection_age = self.stamp_age(self.last_detection_stamp)
        pose_age = self.stamp_age(self.robot_pose_stamp)

        self.get_logger().info(
            f'follow car map position=({car[0]:.3f}, {car[1]:.3f}) '
            f'(distance={self.distance:.2f} m, '
            f'감지 지연={self.format_age(detection_age)}, '
            f'amcl_pose 나이={self.format_age(pose_age)})'
        )

        self.write_follow_row(
            detection_age,
            pose_age,
            self.distance,
            car,
            self.robot_pose,
        )

    # =========================================================
    # header 시각이 지금 기준 몇 초 전인지 (시각을 안 채웠으면 None)
    # =========================================================
    def stamp_age(self, stamp):

        if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
            return None

        return (
            self.get_clock().now() - Time.from_msg(stamp)
        ).nanoseconds / 1e9

    def format_age(self, age):

        return 'N/A' if age is None else f'{age:.2f}s'

    # =========================================================
    # 추종 목표를 보낼 때마다 시각 정보를 CSV 로 저장한다
    #
    # 한 줄 = approach 로 자동차 좌표를 한 번 보낸 시점. 컬럼:
    #   wall_time       PC 시각 (epoch 초)
    #   detection_age   감지 영상 촬영 후 경과 시간 [s]
    #   pose_age        좌표 계산에 쓴 amcl_pose 의 나이 [s]
    #   distance        AMR 카메라 거리 [m]
    #   car_x, car_y    계산한 자동차 map 좌표
    #   robot_x, robot_y, robot_yaw   계산에 쓴 로봇 위치
    # =========================================================
    TIME_CSV_COLUMNS = (
        'wall_time', 'detection_age', 'pose_age', 'distance',
        'car_x', 'car_y', 'robot_x', 'robot_y', 'robot_yaw',
    )

    def open_time_csv(self):

        directory = self.time_log_dir.strip()

        if not directory:
            return

        try:
            directory = os.path.expanduser(directory)
            os.makedirs(directory, exist_ok=True)

            path = os.path.join(
                directory,
                time.strftime('mission_manager_%Y%m%d_%H%M%S.csv')
            )

            self.time_csv = open(
                path, 'w', newline='', encoding='utf-8'
            )
            self.time_csv_writer = csv.writer(self.time_csv)
            self.time_csv_writer.writerow(self.TIME_CSV_COLUMNS)
            self.time_csv.flush()

            self.get_logger().info(f'시간 로그 저장: {path}')

        except OSError as error:
            self.time_csv = None
            self.time_csv_writer = None

            self.get_logger().error(
                f'시간 로그 파일을 열 수 없습니다: {error}'
            )

    def write_follow_row(
        self,
        detection_age,
        pose_age,
        distance,
        car,
        robot_pose
    ):

        if self.time_csv is None:
            return

        def number(value, digits):
            return '' if value is None else f'{value:.{digits}f}'

        try:
            self.time_csv_writer.writerow([
                f'{time.time():.3f}',
                number(detection_age, 3),
                number(pose_age, 3),
                number(distance, 3),
                number(car[0], 3),
                number(car[1], 3),
                number(robot_pose[0], 3),
                number(robot_pose[1], 3),
                number(robot_pose[2], 4),
            ])
            self.time_csv.flush()

        except OSError as error:
            self.time_csv = None
            self.time_csv_writer = None

            self.get_logger().error(
                f'시간 로그 저장 중 오류, 저장을 멈춥니다: {error}'
            )

    def destroy_node(self):

        if self.time_csv is not None:
            self.time_csv.close()
            self.time_csv = None

        return super().destroy_node()

    # =========================================================
    # nav 모드: 추종 goal 취소
    # =========================================================
    def cancel_follow_goal(self):

        self.last_follow_target = None
        self.last_follow_time = None

        self.approach_cancel_pub.publish(Empty())

        self.get_logger().info(
            '추종 중단, approach 취소 요청'
        )

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

        # nav 모드에서는 Nav2 가 마지막 목표로 계속 가지 않게 취소한다.
        if node.follow_mode == FOLLOW_MODE_NAV:
            node.approach_cancel_pub.publish(Empty())

        node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
