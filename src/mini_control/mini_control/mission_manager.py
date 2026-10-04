"""
웹캠 발견 → Nav2 접근 → AMR 카메라 인계 → 자동차 추종을 관리한다.

상태:
SEARCHING
APPROACHING
HANDOVER
FOLLOWING
LOST
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Empty, String
from vision_msgs.msg import Detection2DArray

from .follow import compute_velocity, stop_velocity


LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
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

        # =====================================================
        # 3. 상태값
        # =====================================================
        self.state = 'SEARCHING'

        self.target_detected = False
        self.target_center_x = None

        # amr_detector의 실제 거리 출력이 추가되면 연결한다.
        self.distance = None

        self.last_detection_time = None

        self.handover_count = 0

        # =====================================================
        # 4. Approach Publisher
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
        # 5. TurtleBot4 속도 Publisher
        # =====================================================
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            10
        )

        # =====================================================
        # 6. Subscriber
        # =====================================================

        # 웹캠으로 계산된 자동차 map 위치
        self.create_subscription(
            PointStamped,
            self.target_position_topic,
            self.target_position_callback,
            10
        )

        # approach.py 상태
        self.create_subscription(
            String,
            self.approach_status_topic,
            self.approach_status_callback,
            LATCHED_QOS
        )

        # AMR 카메라 자동차 감지
        self.create_subscription(
            Detection2DArray,
            self.amr_detection_topic,
            self.amr_detection_callback,
            10
        )

        # =====================================================
        # 7. 제어 Timer
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

    # =========================================================
    # approach.py 상태 수신
    # =========================================================
    def approach_status_callback(self, msg):

        status = msg.data

        # Nav2가 이동 중
        if status == 'MOVING':
            self.state = 'APPROACHING'
            return

        # 자동차 근처 도착
        if (
            status == 'ARRIVED'
            and self.state == 'APPROACHING'
        ):
            self.state = 'HANDOVER'
            self.handover_count = 0

            self.get_logger().info(
                'APPROACHING -> HANDOVER'
            )
            return

        # 이동 실패
        if status == 'FAILED':

            self.stop_robot()

            self.state = 'SEARCHING'

            self.get_logger().warning(
                '접근 실패 -> SEARCHING'
            )
            return

        # 접근 제어 취소 완료
        if (
            status == 'CANCELED'
            and self.state == 'HANDOVER'
        ):
            return

    # =========================================================
    # AMR 카메라 감지 결과
    # =========================================================
    def amr_detection_callback(self, msg):

        # 자동차 감지 실패
        if len(msg.detections) == 0:

            self.target_detected = False
            self.handover_count = 0

            return

        detection = msg.detections[0]

        self.target_detected = True

        self.target_center_x = float(
            detection.bbox.center.position.x
        )

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

                self.stop_robot()

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
            return

        # -----------------------------------------------------
        # HANDOVER
        # -----------------------------------------------------
        if self.state == 'HANDOVER':

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

                self.stop_robot()

                self.get_logger().warning(
                    'FOLLOWING -> LOST'
                )

                return

            # 아직 실제 거리값이 연결되지 않았으면 정지
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
