"""고정 USB 웹캠 → car/dummy 감지 → /webcam/detections."""

import os

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from mini_vision.common import detect, run_node


class WebcamDetector(Node):
    def __init__(self):
        super().__init__('webcam_detector')

        # 설치된 mini_vision 패키지 안의 학습 모델을 사용한다.
        package_share = get_package_share_directory('mini_vision')
        default_model_path = os.path.join(
            package_share, 'models', 'webcam_best.pt'
        )

        # YAML 없이도 아래 기본값으로 실행할 수 있다.
        defaults = {
            'camera_index': 4,
            'model_path': default_model_path,
            'target_classes': ['car', 'dummy'],
            'confidence': 0.8,  # 사용자가 설정한 값. 실제 탐지 결과에 맞춰 조정한다.
            'device': 'cuda:0',
            'image_width': 640,
            'image_height': 480,
            'rate_hz': 5.0,
            'show_window': True,
            'frame_id': 'webcam_optical_frame',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {
            name: self.get_parameter(name).value for name in defaults
        }

        if self.p['rate_hz'] <= 0:
            raise ValueError('rate_hz > 0 이어야 합니다.')
        if not 0 < self.p['confidence'] <= 1:
            raise ValueError('0 < confidence <= 1 이어야 합니다.')
        if not self.p['target_classes']:
            raise ValueError('target_classes에 클래스 이름을 지정하세요.')
        if not os.path.isfile(self.p['model_path']):
            raise FileNotFoundError(
                f"YOLO 모델 파일을 찾을 수 없습니다: {self.p['model_path']}"
            )

        self.get_logger().info(f"YOLO model: {self.p['model_path']}")
        self.model = YOLO(self.p['model_path'])
        self.get_logger().info(f'YOLO classes: {self.model.names}')

        # 요청한 두 클래스가 실제 학습 모델에 있는지 확인한다.
        for name in self.p['target_classes']:
            if name not in self.model.names.values():
                raise ValueError(
                    f'모델에 클래스가 없습니다: {name}, '
                    f'available={self.model.names}'
                )

        # USB 웹캠을 열고 요청 해상도를 설정한다.
        self.camera = cv2.VideoCapture(self.p['camera_index'], cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError('웹캠을 열 수 없습니다. camera_index를 확인하세요.')
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.p['image_width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.p['image_height'])
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Detection2DArray, '/webcam/detections', 10
        )
        self.image_publisher = self.create_publisher(
            Image, '/webcam/image_raw', 1
        )
        self.create_timer(1.0 / self.p['rate_hz'], self.tick)

    def tick(self):
        # 프레임을 읽은 직후의 시간을 이미지와 감지 메시지에 동일하게 사용한다.
        ok, frame = self.camera.read()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.p['frame_id']

        empty = Detection2DArray()
        empty.header = header
        if not ok:
            self.publisher.publish(empty)
            self.get_logger().warning(
                '웹캠 영상 읽기 실패', throttle_duration_sec=3.0
            )
            return

        # 보정에 사용하는 해상도와 달라지는 것을 막는다.
        if frame.shape[:2] != (
            self.p['image_height'], self.p['image_width']
        ):
            self.publisher.publish(empty)
            self.get_logger().error(
                '실제 해상도와 설정이 다릅니다.', throttle_duration_sec=3.0
            )
            return

        image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header = header
        self.image_publisher.publish(image)

        try:
            # common.detect가 car와 dummy를 클래스별 최대 1개씩 선택한다.
            message, result = detect(
                self.model,
                frame,
                header,
                self.p['target_classes'],
                self.p['confidence'],
                self.p['device'],
            )
            self.publisher.publish(message)

            # 이 창에는 클래스별 선택 전의 YOLO 결과 전체가 표시된다.
            # ROS 메시지는 common.detect에서 선택한 결과만 담는다.
            if self.p['show_window']:
                cv2.imshow('Webcam YOLO', result.plot())
                cv2.waitKey(1)
        except Exception as error:
            self.publisher.publish(empty)
            self.get_logger().error(str(error), throttle_duration_sec=3.0)

    def destroy_node(self):
        self.camera.release()
        if self.p['show_window']:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    run_node(WebcamDetector, args)


if __name__ == '__main__':
    main()
