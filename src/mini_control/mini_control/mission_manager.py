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
                  Nav2 가 따라가게 한다. 이때 cmd_vel 은 보내지 않고,
                  추종을 멈출 때는 approach 취소 토픽을 쓴다.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

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

        self.declare_parameter(
            'handover_detection_count',
            3
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

        self.handover_detection_count = int(
            self.get_parameter(
                'handover_detection_count'
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
        self.distance = None

        self.last_detection_time = None

        self.handover_count = 0

        # nav 모드: AMR 감지를 map 좌표로 바꾸는 데 필요한 값
        self.camera_fx = None
        self.camera_cx = None

        # amcl_pose 로 받은 로봇 map 위치 (x, y, yaw)
        self.robot_pose = None

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

        # SEARCHING 상태일 때 자동차 위치를 찾으면 접근 시작
        if self.state == 'SEARCHING':

            self.approach_target_pub.publish(msg)

            self.state = 'APPROACHING'

            self.get_logger().info(
                'SEARCHING -> APPROACHING'
            )

            return

        # 취소를 요청한 뒤 새 목표를 보내면 Nav2가 다시 움직이므로 보내지 않는다.
        if (
            self.state == 'APPROACHING'
            and not self.handover_cancel_requested
        ):
            # 계속 들어오는 좌표의 goal 교체 여부는 approach가 판단한다.
            self.approach_target_pub.publish(msg)

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

        if status == ApproachStatus.ARRIVED:
            self.notify_approach_complete()
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
    def notify_approach_complete(self):

        if self.state != 'APPROACHING':
            return

        self.state = 'HANDOVER'
        self.handover_count = 0

        # nav 모드에서는 approach 가 이미 끝났으므로 0 속도를 보내지 않는다.
        if self.follow_mode == FOLLOW_MODE_CMD_VEL:
            self.stop_robot()

        self.get_logger().info(
            'APPROACHING -> HANDOVER'
        )

    # =========================================================
    # AMR 카메라 감지 결과
    # =========================================================
    def amr_detection_callback(self, msg):

        # 자동차 감지 실패
        if len(msg.detections) == 0:

            self.target_detected = False
            self.handover_count = 0
            self.distance = None

            return

        detection = msg.detections[0]

        self.target_detected = True

        self.target_center_x = float(
            detection.bbox.center.position.x
        )

        # amr_detector가 넣어 준 거리 [m], 0 이하는 측정 실패
        distance = 0.0
        if detection.results:
            distance = float(
                detection.results[0].pose.pose.position.z
            )
        self.distance = distance if distance > 0.0 else None

        self.last_detection_time = (
            self.get_clock().now()
        )

        # HANDOVER 상태에서 연속 감지 확인
        if self.state == 'HANDOVER':

            self.handover_count += 1

            if (
                self.handover_count
                >= self.handover_detection_count
            ):
                # Nav2 접근 제어 취소 요청
                self.approach_cancel_pub.publish(
                    Empty()
                )

                if self.follow_mode == FOLLOW_MODE_CMD_VEL:
                    self.stop_robot()

                # 추종 첫 좌표는 바로 보내도록 이전 기록을 지운다.
                self.last_follow_target = None
                self.last_follow_time = None

                self.state = 'FOLLOWING'

                self.get_logger().info(
                    'HANDOVER -> FOLLOWING'
                )

    # =========================================================
    # 감지 결과가 아직 유효한지 확인
    # =========================================================
    def detection_is_fresh(self):

        if self.last_detection_time is None:
            return False

        elapsed = (
            self.get_clock().now()
            - self.last_detection_time
        ).nanoseconds / 1e9

        return elapsed <= self.detection_timeout

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

            if self.follow_mode == FOLLOW_MODE_CMD_VEL:
                self.stop_robot()

            return

        # -----------------------------------------------------
        # FOLLOWING
        # -----------------------------------------------------
        if self.state == 'FOLLOWING':

            # 자동차 감지 손실
            if (
                not self.target_detected
                or not self.detection_is_fresh()
            ):
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

                self.follow_with_nav()

                return

            # 거리 측정에 실패했으면 정지
            if self.distance is None:

                self.stop_robot()

                return

            linear_x, angular_z = compute_velocity(
                target_detected=self.target_detected,
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

            # nav 모드는 FOLLOWING -> LOST 로 넘어갈 때 이미 취소했다.
            if self.follow_mode == FOLLOW_MODE_CMD_VEL:
                self.stop_robot()

            # 다시 AMR 카메라에서 자동차를 찾음
            if (
                self.target_detected
                and self.detection_is_fresh()
            ):
                self.state = 'FOLLOWING'

                self.get_logger().info(
                    'LOST -> FOLLOWING'
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

        self.get_logger().info(
            f'follow car map position=({car[0]:.3f}, {car[1]:.3f}) '
            f'(distance={self.distance:.2f} m)'
        )

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
