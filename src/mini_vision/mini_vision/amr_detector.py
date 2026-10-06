"""
TurtleBot4 AMR 카메라 토픽 → YOLO 감지 + depth 거리 → /amr/detections.

[거리 추가를 위해 수정했다]
OAK-D depth 영상(/robot2/oakd/stereo/image_raw)을 함께 구독해서
박스 중심 주변의 depth 값으로 자동차까지 거리를 구한다.
depth 영상은 RGB preview에 정렬되어 있고 해상도도 같다고 본다.
그래서 박스 중심 픽셀 (u, v)를 depth 영상에 그대로 쓴다.

거리는 표준 메시지의 빈 칸에 넣는다. 새 토픽은 만들지 않는다.
  /amr/detections (vision_msgs/Detection2DArray)
    detections[0].bbox.center.position.x/y   박스 중심 픽셀 (기존)
    detections[0].results[0].pose.pose.position.z
                                             자동차까지 거리 [m] (추가)
                                             측정 실패 시 0.0
"""

import os

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.common import detect, run_node


class AmrDetector(Node):
    def __init__(self):
        super().__init__('amr_detector')

        # mini_vision 패키지의 설치 경로를 가져온다.
        package_share = get_package_share_directory('mini_vision')

        # 기본 YOLO 모델 경로:
        # install/mini_vision/share/mini_vision/models/amr_best.pt
        default_model_path = os.path.join(
            package_share,
            'models',
            'amr_best.pt'
        )

        # 카메라 토픽/모델/검출 관련 기본 파라미터
        defaults = {
            'camera_topic': '/robot2/oakd/rgb/preview/image_raw',
            'model_path': default_model_path,
            'detection_topic': '/amr/detections',
            'target_class': 'car',
            'confidence': 0.8,
            'device': 'cpu',
            'show_window': True,
            # 거리 추가를 위해 수정했다: depth 관련 파라미터
            'depth_topic': '/robot2/oakd/stereo/image_raw',
            'depth_patch_size': 7,      # 박스 중심 주변 N×N 픽셀
            'depth_min_valid': 5,       # 유효 depth 픽셀이 이보다 적으면 실패
            'depth_max_dt': 0.1,        # RGB와 depth 촬영 시각 허용 차이 (초)
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.p = {
            name: self.get_parameter(name).value
            for name in defaults
        }

        # 파라미터 유효성 확인
        if not 0 < self.p['confidence'] <= 1:
            raise ValueError('0 < confidence <= 1 이어야 합니다.')

        # 모델 파일 존재 여부 확인
        if not os.path.isfile(self.p['model_path']):
            raise FileNotFoundError(
                f"YOLO 모델 파일을 찾을 수 없습니다: "
                f"{self.p['model_path']}"
            )

        self.get_logger().info(
            f"YOLO model: {self.p['model_path']}"
        )

        # 학습한 YOLO 모델 로드
        self.model = YOLO(self.p['model_path'])

        self.get_logger().info(
            f"YOLO classes: {self.model.names}"
        )

        # target_class 확인
        if self.p['target_class'] not in self.model.names.values():
            raise ValueError(
                f"모델에 클래스가 없습니다: "
                f"{self.p['target_class']}, "
                f"available={self.model.names}"
            )

        # ROS publisher
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Detection2DArray,
            self.p['detection_topic'],
            10
        )

        # AMR 카메라 영상 구독
        # sensor_data QoS(best effort, depth 5)는 카메라 드라이버가
        # reliable / best effort 어느 쪽으로 발행해도 받을 수 있다.
        self.create_subscription(
            Image,
            self.p['camera_topic'],
            self.image_callback,
            qos_profile_sensor_data
        )

        # 거리 추가를 위해 수정했다: depth 영상 구독
        # 최신 depth 영상(단위 m)과 촬영 시각만 저장해 두고
        # image_callback 에서 박스 중심 거리를 읽는다.
        self.depth_m = None
        self.depth_stamp = None

        self.create_subscription(
            Image,
            self.p['depth_topic'],
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"camera topic: {self.p['camera_topic']}, "
            f"depth topic: {self.p['depth_topic']}"
        )

    def depth_callback(self, image):
        # 거리 추가를 위해 수정했다.
        # depth_checker.py 와 같은 방식으로 passthrough 변환한다.
        try:
            depth = self.bridge.imgmsg_to_cv2(
                image,
                desired_encoding='passthrough'
            )
        except Exception as error:
            self.get_logger().error(
                f'depth 변환 실패: {error}',
                throttle_duration_sec=3.0
            )
            return

        # 16UC1 은 mm, 32FC1 은 m 단위다. 둘 다 m 로 맞춘다.
        if depth.dtype == np.uint16:
            self.depth_m = depth.astype(np.float32) / 1000.0
        else:
            self.depth_m = depth.astype(np.float32)

        self.depth_stamp = Time.from_msg(image.header.stamp)

    def measure_distance(self, u, v, frame_shape, image_stamp):
        """
        거리 추가를 위해 수정했다.

        박스 중심 (u, v) 주변 N×N depth 값 중 0/NaN 을 뺀 중앙값 [m].
        측정할 수 없으면 0.0 을 반환한다.
        """
        if self.depth_m is None:
            self.get_logger().warning(
                'depth 영상을 아직 받지 못했습니다.',
                throttle_duration_sec=3.0
            )
            return 0.0

        # RGB와 depth 가 정렬·같은 해상도라는 전제를 확인한다.
        if self.depth_m.shape[:2] != frame_shape[:2]:
            self.get_logger().warning(
                f'RGB {frame_shape[:2]} 와 depth {self.depth_m.shape[:2]} '
                f'해상도가 다릅니다. 거리를 측정하지 않습니다.',
                throttle_duration_sec=3.0
            )
            return 0.0

        # 너무 오래된 depth 로 재지 않도록 촬영 시각 차이를 확인한다.
        dt = abs(
            (Time.from_msg(image_stamp) - self.depth_stamp).nanoseconds
        ) / 1e9
        if dt > self.p['depth_max_dt']:
            self.get_logger().warning(
                f'RGB와 depth 시각 차이 {dt:.3f}s > '
                f"{self.p['depth_max_dt']}s",
                throttle_duration_sec=3.0
            )
            return 0.0

        height, width = self.depth_m.shape[:2]
        half = max(int(self.p['depth_patch_size']) // 2, 0)

        u = min(max(int(round(u)), 0), width - 1)
        v = min(max(int(round(v)), 0), height - 1)

        patch = self.depth_m[
            max(v - half, 0):v + half + 1,
            max(u - half, 0):u + half + 1
        ]

        # depth 0 은 측정 실패(구멍)이므로 뺀다.
        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if valid.size < self.p['depth_min_valid']:
            return 0.0

        return float(np.median(valid))

    def image_callback(self, image):
        # 감지 결과는 카메라 영상의 촬영 시각과 frame_id를 그대로 쓴다.
        header = image.header

        try:
            # ROS 영상 메시지 → OpenCV BGR 이미지
            frame = self.bridge.imgmsg_to_cv2(
                image,
                desired_encoding='bgr8'
            )

            # 학습한 amr_best.pt로 객체 탐지
            message, result = detect(
                self.model,
                frame,
                header,
                self.p['target_class'],
                self.p['confidence'],
                self.p['device']
            )

            # 거리 추가를 위해 수정했다:
            # 박스 중심 거리를 results[0].pose.pose.position.z 에 넣는다.
            distance = 0.0
            if message.detections:
                detection = message.detections[0]
                distance = self.measure_distance(
                    detection.bbox.center.position.x,
                    detection.bbox.center.position.y,
                    frame.shape,
                    header.stamp
                )
                detection.results[0].pose.pose.position.z = distance

            # 감지하지 못하면 detect()가 빈 Detection2DArray를 반환한다.
            self.publisher.publish(message)

            # 디버깅 창
            if self.p['show_window']:
                plot = result.plot()

                # 거리 추가를 위해 수정했다: 화면에 거리 표시
                if message.detections:
                    text = (
                        f'dist {distance:.2f} m'
                        if distance > 0.0 else 'dist N/A'
                    )
                    cv2.putText(
                        plot, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )

                cv2.imshow(
                    'AMR YOLO',
                    plot
                )
                cv2.waitKey(1)

        except Exception as error:
            # 변환이나 탐지 중 문제가 발생해도 빈 결과 publish
            empty = Detection2DArray()
            empty.header = header

            self.publisher.publish(empty)

            self.get_logger().error(
                str(error),
                throttle_duration_sec=3.0
            )

    def destroy_node(self):
        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(AmrDetector, args)
