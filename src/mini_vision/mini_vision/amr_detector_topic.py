"""
TurtleBot4 AMR 카메라 토픽 → car/dummy 감지 + car depth 거리 → /amr/detections + /amr/image_annotated.

[car / dummy 를 함께 내보내도록 수정했다]
car 최대 1개 + dummy 최대 1개를 /amr/detections 에 담는다 (amr_detector.py 와 같은 방식).
class_id 로 구분하며, car 가 안 보이고 dummy 만 보이면 detections[0] 은 dummy 다.

[거리 추가를 위해 수정했다]
amr_detector.py 와 같은 방식으로 depth 영상을 구독해서 car 박스 중심 거리 [m]를
그 detection 의 results[0].pose.pose.position.z 에 넣는다.
(측정 실패 시 0.0, dummy 는 거리를 재지 않고 항상 0.0)
/amr/image_annotated 에는 car 박스 아래에 거리를 글자로 표시한다.
RGB/depth 는 해상도를 맞춰 둔 압축 토픽(둘 다 704x704)을 쓴다.
  RGB   /robot2/oakd/rgb/image_raw/compressed          (JPEG, bgr8)
  depth /robot2/oakd/stereo/image_raw/compressedDepth  (PNG, 16UC1 mm)
mission_manager 수정 방법은 amr_detector.py 맨 위 설명을 참고한다.
"""

import os

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.amr_detector import draw_distances
from mini_vision.common import detect, run_node
from mini_vision.webcam_detector_topic import draw_detections

# compressedDepth 메시지는 PNG 앞에 12바이트 헤더(압축 방식, 깊이 변환값)가 붙는다.
COMPRESSED_DEPTH_HEADER_SIZE = 12


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
            'camera_topic': '/robot2/oakd/rgb/image_raw/compressed',
            'model_path': default_model_path,
            'detection_topic': '/amr/detections',
            'target_classes': ['car', 'dummy'],
            'confidence': 0.8,
            'device': 'cpu',
            'show_window': True,
            'annotated_topic': '/amr/image_annotated',
            # 거리 추가를 위해 수정했다: depth 관련 파라미터
            'depth_topic': '/robot2/oakd/stereo/image_raw/compressedDepth',
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

        # target_classes 확인
        if not self.p['target_classes']:
            raise ValueError('target_classes에 클래스 이름을 지정하세요.')

        for name in self.p['target_classes']:
            if name not in self.model.names.values():
                raise ValueError(
                    f"모델에 클래스가 없습니다: "
                    f"{name}, "
                    f"available={self.model.names}"
                )

        # ROS publisher
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Detection2DArray,
            self.p['detection_topic'],
            10
        )

        # 감지 결과(publish한 car / dummy)를 그린 영상
        self.annotated_publisher = self.create_publisher(
            Image,
            self.p['annotated_topic'],
            1
        )

        # AMR 카메라 영상 구독
        # sensor_data QoS(best effort, depth 5)는 카메라 드라이버가
        # reliable / best effort 어느 쪽으로 발행해도 받을 수 있다.
        self.create_subscription(
            CompressedImage,
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
            CompressedImage,
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
        # compressedDepth 는 cv_bridge 로 풀 수 없다.
        # 12바이트 헤더를 건너뛰고 PNG 를 직접 디코딩한다.
        if not image.format.startswith('16UC1'):
            self.get_logger().error(
                f'지원하지 않는 depth 형식입니다: {image.format}',
                throttle_duration_sec=3.0
            )
            return

        depth = cv2.imdecode(
            np.frombuffer(image.data, np.uint8)[
                COMPRESSED_DEPTH_HEADER_SIZE:
            ],
            cv2.IMREAD_UNCHANGED
        )

        if depth is None or depth.dtype != np.uint16:
            self.get_logger().error(
                'depth PNG 디코딩 실패',
                throttle_duration_sec=3.0
            )
            return

        # 16UC1 은 mm 단위다. m 로 맞춘다.
        self.depth_m = depth.astype(np.float32) / 1000.0

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
        frame = None

        try:
            # 압축(JPEG) 영상 메시지 → OpenCV BGR 이미지
            frame = self.bridge.compressed_imgmsg_to_cv2(
                image,
                desired_encoding='bgr8'
            )

            # 학습한 amr_best.pt로 객체 탐지
            message, result = detect(
                self.model,
                frame,
                header,
                self.p['target_classes'],
                self.p['confidence'],
                self.p['device']
            )

            # 거리 추가를 위해 수정했다:
            # car 박스 중심 거리를 results[0].pose.pose.position.z 에 넣는다.
            # dummy 는 거리가 필요 없어서 재지 않는다 (z 는 0.0 그대로).
            distances = []
            for detection in message.detections:
                if detection.results[0].hypothesis.class_id != 'car':
                    distances.append(None)
                    continue

                distance = self.measure_distance(
                    detection.bbox.center.position.x,
                    detection.bbox.center.position.y,
                    frame.shape,
                    header.stamp
                )
                detection.results[0].pose.pose.position.z = distance
                distances.append(distance)

            # 감지하지 못하면 detect()가 빈 Detection2DArray를 반환한다.
            self.publisher.publish(message)

            # publish한 감지 결과만 그린 영상
            annotated = draw_detections(frame, message)

            # 거리 추가를 위해 수정했다: car 박스 아래에 거리 표시
            draw_distances(annotated, message, distances)
            self.publish_annotated(annotated, header)

            # 디버깅 창
            if self.p['show_window']:
                cv2.imshow(
                    'AMR YOLO',
                    annotated
                )
                cv2.waitKey(1)

        except Exception as error:
            # 변환이나 탐지 중 문제가 발생해도 빈 결과 publish
            empty = Detection2DArray()
            empty.header = header

            self.publisher.publish(empty)

            # 영상 변환에 성공했다면 박스 없는 원본을 publish
            if frame is not None:
                self.publish_annotated(frame, header)

            self.get_logger().error(
                str(error),
                throttle_duration_sec=3.0
            )

    def publish_annotated(self, image, header):
        message = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )
        message.header = header

        self.annotated_publisher.publish(message)

    def destroy_node(self):
        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(AmrDetector, args)
