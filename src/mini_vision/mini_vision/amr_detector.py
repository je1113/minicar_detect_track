"""
TurtleBot4 AMR 카메라 토픽 → YOLO 감지 + depth 거리 → /amr/detections.

[거리 추가를 위해 수정했다]
해상도를 맞춰 둔 압축 토픽 두 개를 구독한다 (둘 다 704x704).
  RGB   /robot2/oakd/rgb/image_raw/compressed          (JPEG, bgr8)
  depth /robot2/oakd/stereo/image_raw/compressedDepth  (PNG, 16UC1 mm)
박스 중심 주변의 depth 값으로 자동차까지 거리를 구한다.
depth 영상은 RGB에 정렬되어 있고 해상도도 같다.
그래서 박스 중심 픽셀 (u, v)를 depth 영상에 그대로 쓴다.

거리는 표준 메시지의 빈 칸에 넣는다. 새 토픽은 만들지 않는다.
  /amr/detections (vision_msgs/Detection2DArray)
    detections[0].bbox.center.position.x/y   박스 중심 픽셀 (기존)
    detections[0].results[0].pose.pose.position.z
                                             자동차까지 거리 [m] (추가)
                                             측정 실패 시 0.0

----------------------------------------------------------------------
mission_manager 수정 방법 (mini_control/mission_manager.py)
----------------------------------------------------------------------
지금 mission_manager는 self.distance 가 항상 None 이라서
FOLLOWING 상태에서 정지 명령만 보낸다. 아래 두 곳만 고치면 된다.
구독 토픽(/amr/detections)과 launch 는 바꿀 필요가 없다.

1) amr_detection_callback() 의 "자동차 감지 실패" 분기에 추가

       if len(msg.detections) == 0:
           self.target_detected = False
           self.handover_count = 0
           self.distance = None          # 추가

2) amr_detection_callback() 에서 target_center_x 를 저장한 바로 아래에 추가

       self.target_center_x = float(
           detection.bbox.center.position.x
       )

       # 추가: amr_detector 가 넣어 준 거리 [m], 0 이하는 측정 실패
       distance = 0.0
       if detection.results:
           distance = float(
               detection.results[0].pose.pose.position.z
           )
       self.distance = distance if distance > 0.0 else None

그 외에는 지금 코드가 그대로 동작한다.
  - 거리가 None 이면 control_loop 의 FOLLOWING 에서 정지한다 (기존 코드).
  - 거리와 박스가 같은 메시지로 오므로 detection_timeout 검사가
    거리에도 그대로 적용된다.
  - 확인할 것: mission_manager 의 image_width 파라미터(기본 640)가
    AMR 영상의 실제 가로 폭(704)과 같아야 회전 계산이 맞다.
"""

import os

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.common import detect, run_node

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
            'target_class': 'car',
            'confidence': 0.8,
            'device': 'cpu',
            'show_window': True,
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
