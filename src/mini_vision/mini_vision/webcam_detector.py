# 역할:
# 스탠드에 설치한 웹캠 영상을 읽고 YOLO로 자동차를 감지한다.

# 작성할 내용:
# 1. ROS 노드를 생성한다.
# 2. 웹캠 번호, 모델 경로, 감지 임계값을 파라미터로 받는다.
# 3. 웹캠을 열고 YOLO 모델을 불러온다.
# 4. 주기적으로 영상을 읽어 자동차를 감지한다.
# 5. 자동차 바운딩 박스와 중심 픽셀 좌표를 구한다.
# 6. 감지 결과를 /webcam/detections 토픽으로 발행한다.
# 7. 감지하지 못한 경우에도 빈 감지 결과를 발행한다.
# 8. 종료할 때 웹캠과 ROS 노드를 정리한다.

# 출력 좌표:
# 영상의 픽셀 좌표이며, 아직 지도 좌표가 아니다.

# main():
# ROS 초기화 → 노드 생성 → 실행 → 종료 처리

"""고정 USB 웹캠 → YOLO 감지 → /webcam/detections."""

import cv2
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
        defaults = {
            'camera_index': 0, 'model_path': 'yolov8n.pt',
            'target_class': 'car', 'confidence': 0.5, 'device': 'cpu',
            'image_width': 640, 'image_height': 480, 'rate_hz': 10.0,
            'show_window': True, 'frame_id': 'webcam_optical_frame',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if self.p['rate_hz'] <= 0 or not 0 < self.p['confidence'] <= 1:
            raise ValueError('rate_hz > 0, 0 < confidence <= 1 이어야 합니다.')

        self.model = YOLO(self.p['model_path'])
        if self.p['target_class'] not in self.model.names.values():
            raise ValueError(f"모델에 클래스가 없습니다: {self.p['target_class']}")
        self.camera = cv2.VideoCapture(self.p['camera_index'], cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError('웹캠을 열 수 없습니다. camera_index를 확인하세요.')
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.p['image_width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.p['image_height'])
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Detection2DArray, '/webcam/detections', 10)
        self.image_publisher = self.create_publisher(Image, '/webcam/image_raw', 1)
        self.create_timer(1.0 / self.p['rate_hz'], self.tick)

    def tick(self):
        ok, frame = self.camera.read()
        header = Header()
        # USB read 시각을 사용한다. 하드웨어 노출 시각은 아니다.
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.p['frame_id']
        message = Detection2DArray()
        message.header = header
        if not ok:
            self.publisher.publish(message)
            self.get_logger().warning('웹캠 영상 읽기 실패', throttle_duration_sec=3.0)
            return
        # 보정과 다른 해상도를 조용히 사용하지 않는다.
        if frame.shape[:2] != (self.p['image_height'], self.p['image_width']):
            self.publisher.publish(message)
            self.get_logger().error('실제 해상도와 설정이 다릅니다.', throttle_duration_sec=3.0)
            return
        image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header = header
        self.image_publisher.publish(image)
        try:
            message, result = detect(
                self.model, frame, header, self.p['target_class'],
                self.p['confidence'], self.p['device'])
            self.publisher.publish(message)
            if self.p['show_window']:
                cv2.imshow('Webcam YOLO', result.plot())
                cv2.waitKey(1)
        except Exception as error:
            empty = Detection2DArray()
            empty.header = header
            self.publisher.publish(empty)
            self.get_logger().error(str(error), throttle_duration_sec=3.0)

    def destroy_node(self):
        self.camera.release()
        if self.p['show_window']:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    run_node(WebcamDetector, args)
