"""고정 USB 웹캠 → car/dummy 감지 → /webcam/detections + /webcam/image_annotated."""

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

BOX_COLOR = (0, 200, 0)  # BGR, CLASS_COLORS에 없는 클래스의 기본 색
CLASS_COLORS = {
    'car': (0, 200, 0),       # 초록
    'dummy': (0, 140, 255),   # 주황
}


def draw_detections(frame, message):
    """Detection2DArray에 담긴 박스만 클래스별 색으로 원본 복사본에 그린다."""

    image = frame.copy()

    for detection in message.detections:
        cx = detection.bbox.center.position.x
        cy = detection.bbox.center.position.y
        w = detection.bbox.size_x
        h = detection.bbox.size_y

        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        hypothesis = detection.results[0].hypothesis
        text = f'{hypothesis.class_id} {hypothesis.score:.2f}'
        color = CLASS_COLORS.get(hypothesis.class_id, BOX_COLOR)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (int(cx), int(cy)), 4, color, -1)

        # 박스 위에 글자 배경을 깔고 텍스트 표시 (화면 위로 넘어가면 박스 안쪽)
        (tw, th), base = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        ty = y1 - 6 if y1 - th - base - 6 >= 0 else y1 + th + 6
        cv2.rectangle(
            image,
            (x1, ty - th - base),
            (x1 + tw + 4, ty + base),
            color,
            -1
        )
        cv2.putText(
            image, text, (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    return image


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
            'target_classes': ['car', 'dummy'],
            'confidence': 0.8,          # 기존 0.5 -> 0.8 추천
            'device': 'cpu',
            'image_width': 640,
            'image_height': 480,
            'rate_hz': 10.0,
            'show_window': True,
            'frame_id': 'webcam_optical_frame',
            'annotated_topic': '/webcam/image_annotated',
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

        # 감지 결과(publish한 car 1개)를 그린 영상
        self.annotated_publisher = self.create_publisher(
            Image,
            self.p['annotated_topic'],
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
                self.p['target_classes'],
                self.p['confidence'],
                self.p['device']
            )

            self.publisher.publish(message)

            # publish한 감지 결과만 그린 영상
            annotated = draw_detections(frame, message)
            self.publish_annotated(annotated, header)

            # 디버깅 창
            if self.p['show_window']:
                cv2.imshow(
                    'Webcam YOLO',
                    annotated
                )
                cv2.waitKey(1)

        except Exception as error:
            # 탐지 중 문제가 발생해도 빈 결과 publish
            empty = Detection2DArray()
            empty.header = header

            self.publisher.publish(empty)

            # 영상이 끊기지 않도록 박스 없는 원본을 publish
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
        self.camera.release()

        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(WebcamDetector, args)