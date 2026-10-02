"""
역할:
웹캠 발견 → AMR 접근 → AMR 카메라 인계 → 추종 순서를 관리한다.

현재는 webcam_localizer.py, amr_detector.py, approach.py의
실제 구현이 완료되기 전이므로 ROS2 연결 틀을 구성한다.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, TwistStamped
from vision_msgs.msg import Detection2DArray

from .follow import compute_velocity, stop_velocity


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        # =====================================================
        # 1. ROS2 파라미터
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
            'cmd_vel_topic',
            '/robot2/cmd_vel'
        )

        self.declare_parameter(
            'map_frame',
            'map'
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
            'image_width',
            640
        )

        self.declare_parameter(
            'detection_timeout',
            0.5
        )

        # =====================================================
        # 2. 파라미터 값 읽기
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

        self.image_width = int(
            self.get_parameter(
                'image_width'
            ).value
        )

        self.detection_timeout = float(
            self.get_parameter(
                'detection_timeout'
            ).value
        )

        # =====================================================
        # 3. 상태
        # =====================================================
        self.state = 'SEARCHING'

        # =====================================================
        # 4. 웹캠이 계산한 자동차 지도 위치
        # =====================================================
        self.target_map_position = None

        # =====================================================
        # 5. AMR 카메라 추종 데이터
        # =====================================================
        self.target_detected = False
        self.target_center_x = None

        # 거리값은 amr_detector 실제 구현 후 연결
        self.distance = None

        self.last_detection_time = None

        # =====================================================
        # 6. Subscriber
        # =====================================================
        self.target_position_sub = self.create_subscription(
            PointStamped,
            self.target_position_topic,
            self.target_position_callback,
            10
        )

        self.amr_detection_sub = self.create_subscription(
            Detection2DArray,
            self.amr_detection_topic,
            self.amr_detection_callback,
            10
        )

        # =====================================================
        # 7. TurtleBot4 cmd_vel Publisher
        # =====================================================
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10
        )

        # =====================================================
        # 8. 제어 Loop
        # =====================================================
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Mission Manager 시작'
        )

    # =========================================================
    # 웹캠 지도 위치 수신
    # =========================================================
    def target_position_callback(self, msg):

        self.target_map_position = msg

        if self.state == 'SEARCHING':
            self.state = 'APPROACHING'

            self.get_logger().info(
                'SEARCHING -> APPROACHING'
            )

            # TODO:
            # approach.py 구현 완료 후
            # 여기서 접근 목표 전송

    # =========================================================
    # AMR 카메라 감지 결과 수신
    # =========================================================
    def amr_detection_callback(self, msg):

        # 감지 결과 없음
        if len(msg.detections) == 0:
            self.target_detected = False
            return

        detection = msg.detections[0]

        self.target_detected = True

        self.target_center_x = (
            detection.bbox.center.position.x
        )

        self.last_detection_time = (
            self.get_clock().now()
        )

        # TODO:
        # amr_detector.py에서 실제 거리값이 제공되면
        # self.distance에 연결한다.

    # =========================================================
    # 접근 완료 알림용 함수
    # =========================================================
    def notify_approach_complete(self):

        if self.state != 'APPROACHING':
            return

        self.state = 'HANDOVER'

        self.stop_robot()

        self.get_logger().info(
            'APPROACHING -> HANDOVER'
        )

    # =========================================================
    # 인계 완료 알림용 함수
    # =========================================================
    def notify_handover_complete(self):

        if self.state != 'HANDOVER':
            return

        self.state = 'FOLLOWING'

        self.get_logger().info(
            'HANDOVER -> FOLLOWING'
        )

    # =========================================================
    # 감지 데이터 시간 확인
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
    # 전체 상태 제어
    # =========================================================
    def control_loop(self):

        # -----------------------------------------------------
        # SEARCHING
        # -----------------------------------------------------
        if self.state == 'SEARCHING':
            return

        # -----------------------------------------------------
        # APPROACHING
        # Nav2가 제어
        # -----------------------------------------------------
        if self.state == 'APPROACHING':
            return

        # -----------------------------------------------------
        # HANDOVER
        # Nav2 제어 해제 후 추종으로 전환 준비
        # -----------------------------------------------------
        if self.state == 'HANDOVER':
            self.stop_robot()
            return

        # -----------------------------------------------------
        # FOLLOWING
        # -----------------------------------------------------
        if self.state == 'FOLLOWING':

            # 자동차를 놓쳤거나 오래된 감지 데이터
            if (
                not self.target_detected
                or not self.detection_is_fresh()
            ):
                self.state = 'LOST'
                self.stop_robot()

                self.get_logger().warning(
                    'FOLLOWING -> LOST'
                )
                return

            # 아직 거리 정보가 없으면 이동하지 않음
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

            self.stop_robot()

            # 다시 자동차를 찾으면 추종 재개
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
    # 속도 발행
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
    # 정지
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
