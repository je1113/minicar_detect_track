"""고정 USB 웹캠 → YOLO 감지 → /webcam/detections."""

import os

import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.common import detect, run_node


class WebcamDetector(Node):
    def __init__(self):
        super().__init__('webcam_detector')

        # mini_vision 패키지의 설치 경로를 가져온다.
        package_share = get_package_share_directory('mini_vision')

        # 기본 YOLO 모델 경로:
        # install/mini_vision/share/mini_vision/models/webcam_best.pt
        default_model_path = os.path.join(
            package_share,
            'models',
            'webcam_best.pt'
        )

        # 웹캠/모델/검출 관련 기본 파라미터
        defaults = {
            'camera_index': 4,
            'model_path': default_model_path,
            'target_class': 'car',
            'confidence': 0.8,          # 기존 0.5 -> 0.8 추천
            'device': 'cpu',
            'image_width': 640,
            'image_height': 480,
            'rate_hz': 10.0,
            'show_window': True,
            'frame_id': 'webcam_optical_frame',
            'max_detections': 1,        # 자동차는 1개만 사용
            'min_box_area': 1500,       # 너무 작은 박스는 무시
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.p = {
            name: self.get_parameter(name).value
            for name in defaults
        }

        # 파라미터 유효성 확인
        if self.p['rate_hz'] <= 0:
            raise ValueError('rate_hz > 0 이어야 합니다.')

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

        # USB 웹캠 열기
        self.camera = cv2.VideoCapture(
            self.p['camera_index'],
            cv2.CAP_V4L2
        )

        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(
                '웹캠을 열 수 없습니다. camera_index를 확인하세요.'
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.p['image_width']
        )
        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.p['image_height']
        )
        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        # ROS publisher
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Detection2DArray,
            '/webcam/detections',
            10
        )

        self.image_publisher = self.create_publisher(
            Image,
            '/webcam/image_raw',
            1
        )

        # 주기적으로 영상 처리
        self.create_timer(
            1.0 / self.p['rate_hz'],
            self.tick
        )

    def tick(self):
        # 웹캠 프레임 읽기
        ok, frame = self.camera.read()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.p['frame_id']

        message = Detection2DArray()
        message.header = header

        # 프레임 읽기 실패
        if not ok:
            self.publisher.publish(message)

            self.get_logger().warning(
                '웹캠 영상 읽기 실패',
                throttle_duration_sec=3.0
            )
            return

        # 실제 해상도 확인
        if frame.shape[:2] != (
            self.p['image_height'],
            self.p['image_width']
        ):
            self.publisher.publish(message)

            self.get_logger().error(
                '실제 해상도와 설정이 다릅니다.',
                throttle_duration_sec=3.0
            )
            return

        # 원본 영상 ROS topic으로 publish
        image = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )
        image.header = header

        self.image_publisher.publish(image)

        try:
            # 학습한 webcam_best.pt로 객체 탐지
            message, result = detect(
                self.model,
                frame,
                header,
                self.p['target_class'],
                self.p['confidence'],
                self.p['device']
            )

            self.publisher.publish(message)

            # 디버깅 창
            if self.p['show_window']:
                cv2.imshow(
                    'Webcam YOLO',
                    result.plot()
                )
                cv2.waitKey(1)

        except Exception as error:
            # 탐지 중 문제가 발생해도 빈 결과 publish
            empty = Detection2DArray()
            empty.header = header

            self.publisher.publish(empty)

            self.get_logger().error(
                str(error),
                throttle_duration_sec=3.0
            )

    def destroy_node(self):
        self.camera.release()

        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(WebcamDetector, args)