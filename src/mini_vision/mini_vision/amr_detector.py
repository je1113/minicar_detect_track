"""
TurtleBot4 AMR 카메라 토픽 → YOLO 감지 + depth 거리 → /amr/detections.

[거리 추가를 위해 수정했다]
해상도를 맞춰 둔 압축 토픽 두 개를 구독한다 (둘 다 704x704).
  RGB   /robot2/oakd/rgb/image_raw/compressed          (JPEG, bgr8)
  depth /robot2/oakd/stereo/image_raw/compressedDepth  (PNG, 16UC1 mm)
박스 중심 주변의 depth 값으로 자동차까지 거리를 구한다.
depth 영상은 RGB에 정렬되어 있고 해상도도 같다.
그래서 박스 중심 픽셀 (u, v)를 depth 영상에 그대로 쓴다.

[RGB 와 depth 시각이 어긋나 거리가 N/A 로 깜빡이던 문제를 고치기 위해 수정했다]
depth 는 mini_vision.depth_distance 의 별도 노드가 따로 받아 두고,
RGB 촬영 시각에 가장 가까운 depth 로 거리를 잰다 (자세한 내용은 그 파일 참고).
RGB 구독 큐 깊이는 1 로 줄여서 YOLO 가 느려도 가장 최근 영상만 처리한다.
거리를 못 재도 감지 결과는 그대로 나간다.

[dummy 를 화면에 그리도록 수정했다]
target_classes (기본 ['car', 'dummy']) 를 모두 감지해서 YOLO 화면에 박스를 그린다.
  - amr_detector        'AMR YOLO' 창
  - amr_detector_topic  'AMR YOLO' 창과 /amr/image_annotated (car 초록, dummy 주황)
토픽(/amr/detections)에는 publish_classes (기본 ['car']) 만 내보낸다.
dummy 는 중심 좌표도 거리도 필요 없어서 토픽에 싣지 않는다.

거리는 표준 메시지의 빈 칸에 넣는다. 새 토픽은 만들지 않는다.
  /amr/detections (vision_msgs/Detection2DArray)
    detections[0].bbox.center.position.x/y   car 박스 중심 픽셀 (기존)
    detections[0].results[0].pose.pose.position.z
                                             car 까지 거리 [m] (추가)
                                             측정 실패 시 0.0
  publish_classes 에 'dummy' 를 넣으면 dummy 도 토픽에 실리지만 거리는 0.0 이다.
  그때는 dummy 만 보이면 detections[0] 이 dummy 가 되므로 받는 쪽이 class_id 로
  car 를 골라야 한다.
"""

import os

import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.common import detect, run_node
from mini_vision.depth_distance import DEPTH_DEFAULTS, DepthDistance


def draw_distances(image, message, distances):
    """
    거리 추가를 위해 수정했다.

    박스 아래에 그 박스까지의 거리를 글자로 쓴다. 측정 실패는 N/A.
    distances 는 message.detections 와 같은 순서이고,
    거리를 재지 않은 박스(dummy)는 None 이라서 건너뛴다.
    """
    for detection, distance in zip(message.detections, distances):
        if distance is None:
            continue

        x1 = int(
            detection.bbox.center.position.x - detection.bbox.size_x / 2
        )
        y2 = int(
            detection.bbox.center.position.y + detection.bbox.size_y / 2
        )

        text = f'{distance:.2f} m' if distance > 0.0 else 'N/A'

        # 박스 아래에 쓰고, 화면 아래로 넘어가면 박스 안쪽에 쓴다.
        ty = y2 + 22 if y2 + 22 < image.shape[0] else y2 - 8

        cv2.putText(
            image, text, (max(x1, 0), ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )


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
            'target_classes': ['car', 'dummy'],   # 감지해서 화면에 그릴 클래스
            'publish_classes': ['car'],           # 토픽에 실을 클래스
            'confidence': 0.8,
            'device': 'cpu',
            'show_window': True,
            # RGB 구독 큐 깊이. 1 이면 YOLO 가 밀려도 가장 최근 영상만 처리한다.
            'image_queue_depth': 1,
            # 거리 추가를 위해 수정했다: depth 관련 파라미터 (depth_distance.py)
            **DEPTH_DEFAULTS,
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

        for name in self.p['publish_classes']:
            if name not in self.p['target_classes']:
                raise ValueError(
                    f"publish_classes 의 {name} 이(가) "
                    f"target_classes 에 없습니다."
                )

        # ROS publisher
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Detection2DArray,
            self.p['detection_topic'],
            10
        )

        # AMR 카메라 영상 구독
        # sensor_data 와 같은 best effort QoS 는 카메라 드라이버가
        # reliable / best effort 어느 쪽으로 발행해도 받을 수 있다.
        # 큐 깊이는 1 로 줄여서 YOLO 가 느려도 오래된 영상부터 처리하지 않는다.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(int(self.p['image_queue_depth']), 1),
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            CompressedImage,
            self.p['camera_topic'],
            self.image_callback,
            image_qos
        )

        # 거리 추가를 위해 수정했다: depth 영상은 별도 노드가 따로 받는다.
        # image_callback 에서는 RGB 촬영 시각에 가장 가까운 depth 로 거리를 잰다.
        self.depth_distance = DepthDistance(self.p, self.get_logger())

        self.get_logger().info(
            f"camera topic: {self.p['camera_topic']}, "
            f"depth topic: {self.p['depth_topic']}"
        )

    def image_callback(self, image):
        # 감지 결과는 카메라 영상의 촬영 시각과 frame_id를 그대로 쓴다.
        header = image.header

        try:
            # 압축(JPEG) 영상 메시지 → OpenCV BGR 이미지
            frame = self.bridge.compressed_imgmsg_to_cv2(
                image,
                desired_encoding='bgr8'
            )

            # 학습한 amr_best.pt로 객체 탐지 (target_classes: car, dummy)
            detected, result = detect(
                self.model,
                frame,
                header,
                self.p['target_classes'],
                self.p['confidence'],
                self.p['device']
            )

            # 토픽에는 publish_classes(기본 car)만 내보낸다.
            # dummy 는 화면에만 그리고 중심 좌표·거리는 내보내지 않는다.
            message = Detection2DArray()
            message.header = detected.header
            message.detections = [
                detection for detection in detected.detections
                if detection.results[0].hypothesis.class_id
                in self.p['publish_classes']
            ]

            # 거리 추가를 위해 수정했다:
            # car 박스 중심 거리를 results[0].pose.pose.position.z 에 넣는다.
            # dummy 는 거리가 필요 없어서 재지 않는다 (z 는 0.0 그대로).
            distances = []
            for detection in message.detections:
                if detection.results[0].hypothesis.class_id != 'car':
                    distances.append(None)
                    continue

                distance = self.depth_distance.measure(
                    detection.bbox.center.position.x,
                    detection.bbox.center.position.y,
                    frame.shape,
                    header.stamp
                )
                detection.results[0].pose.pose.position.z = distance
                distances.append(distance)

            # 감지하지 못하면 detect()가 빈 Detection2DArray를 반환한다.
            self.publisher.publish(message)

            # 디버깅 창
            if self.p['show_window']:
                plot = result.plot()

                # 거리 추가를 위해 수정했다: car 박스 아래에 거리 표시
                draw_distances(plot, message, distances)

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
        self.depth_distance.close()

        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(AmrDetector, args)
