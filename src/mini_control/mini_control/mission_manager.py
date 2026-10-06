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
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, TwistStamped

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
            0.3
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

        # amr_detector가 /amr/detections에 넣어 주는 자동차까지 거리 [m]
        self.distance = None

        self.last_detection_time = None

        self.handover_count = 0

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

        self.stop_robot()

        self.get_logger().info(
            'APPROACHING -> HANDOVER'
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

        # car가 하나도 없으면 detection 실패로 처리
        if len(car_detections) == 0:
            self.target_detected = False
            self.handover_count = 0
            self.distance = None
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

        distance = 0.0

        if detection.results:
            distance = float(
                detection.results[0].pose.pose.position.z
            )

        self.distance = (
            distance if distance > 0.0 else None
        )

        
        self.last_detection_time = (
            self.get_clock().now()
        )

        score = float(
            detection.results[0].hypothesis.score
        )

        self.get_logger().info(
            f'AMR car detected: '
            f'x={self.target_center_x:.1f}, '
            f'score={score:.2f}'
        )

        # HANDOVER 상태에서 연속 car 감지 확인
        if self.state == 'HANDOVER':

            self.handover_count += 1

            if (
                self.handover_count
                >= self.handover_detection_count
            ):
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
