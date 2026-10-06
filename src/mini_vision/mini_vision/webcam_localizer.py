# 역할:
# 웹캠에서 감지한 자동차 위치를 AMR이 사용하는 지도 좌표로 변환한다.

# 작성할 내용:
# 1. ROS 노드를 생성한다.
# 2. camera_mapping.yaml의 보정 파라미터를 읽는다.
# 3. /webcam/detections 토픽을 구독한다.
# 4. 감지 결과에서 자동차의 대표 픽셀 위치를 선택한다.
# 5. 미리 구한 호모그래피로 픽셀 위치를 지도 좌표로 변환한다.
# 6. 보정 영역 안의 유효한 위치인지 확인한다.
# 7. /target/map_position으로 위치를 발행한다.
#    메시지는 geometry_msgs/PointStamped를 사용하는 설계로 한다.
#    frame_id는 map으로 설정하고 감지 시각을 유지한다.
# 8. 자동차가 없으면 새로운 위치를 발행하지 않는다.

# 주의:
# 보정 기준점의 실제 좌표가 AMR의 map 좌표와 연결되어야 한다.
# 이전 위치의 유효 시간은 mission_manager에서 확인한다.
# 자동차 높이에 따른 위치 오차는 실제 측정으로 확인한다.

# main():
# ROS 초기화 → 노드 생성 → 실행 → 종료 처리


"""바닥의 네 대응점으로 웹캠 픽셀을 map 좌표(미터)로 변환한다."""

import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from mini_vision.common import run_node


class WebcamLocalizer(Node):
    def __init__(self):
        super().__init__('webcam_localizer')

        self.get_logger().info('1. node created')

        defaults = {
            'calibrated': False, 'map_frame': 'map',
            'pixel_points': [0.0] * 8, 'map_points': [0.0] * 8,
            'camera_matrix': [0.0] * 9, 'distortion': [0.0] * 5,
            'undistort': False, 'max_age': 0.5,
            'calibration_width': 640, 'calibration_height': 480,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.publisher = self.create_publisher(PointStamped, '/target/map_position', 10)
        self.get_logger().info('2. publisher created')
        
        self.homography = None
        self.image_size_ok = False
        self.create_subscription(Image, '/webcam/image_raw', self.on_image, 1)
        if self.p['calibrated']:
            self.get_logger().info('3. starting calibration')
            self.setup_calibration()
            self.get_logger().info('4. calibration complete')
        else:
            self.get_logger().warning('보정 전: 지도 위치를 발행하지 않습니다.')
        self.create_subscription(Detection2DArray, '/webcam/detections', self.on_detection, 10)
        self.get_logger().info('5. localizer ready')

    def on_image(self, message):
        self.image_size_ok = (message.width, message.height) == (
            self.p['calibration_width'], self.p['calibration_height'])

    def undistort(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not self.p['undistort']:
            return points
        matrix = np.asarray(self.p['camera_matrix']).reshape(3, 3)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError('유효한 내부 카메라 행렬이 필요합니다.')
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2), matrix,
            np.asarray(self.p['distortion']), P=matrix).reshape(-1, 2)

    def setup_calibration(self):
        # 점 순서는 경계를 따라 시계 또는 반시계 방향이어야 한다.
        source = self.undistort(self.p['pixel_points']).astype(np.float32)
        destination = np.asarray(self.p['map_points'], dtype=np.float32).reshape(-1, 2)
        for points in (source, destination):
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError('유효한 대응점 4개가 필요합니다.')
            if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1e-6:
                raise ValueError('기준점은 넓이가 있는 볼록 사각형이어야 합니다.')
        self.homography = cv2.getPerspectiveTransform(source, destination)
        if not np.isfinite(self.homography).all() or np.linalg.matrix_rank(self.homography) < 3:
            raise ValueError('호모그래피 계산 실패')
        self.region = source

    def on_detection(self, message):
        if self.homography is None:
            return

        if not self.image_size_ok:
            self.get_logger().warning(
                'image size mismatch or image not received yet',
                throttle_duration_sec=3.0
            )
            return
        # car 클래스만 선택
        car_detections = []

        for detection in message.detections:
            if len(detection.results) == 0:
                continue

            class_id = detection.results[0].hypothesis.class_id

            if class_id == 'car':
                car_detections.append(detection)

        # car가 없으면 아무것도 발행하지 않음
        if len(car_detections) == 0:
            return

        # car가 여러 개면 confidence 가장 높은 것 사용
        detection = max(
            car_detections,
            key=lambda d: float(d.results[0].hypothesis.score)
        )

        score = float(
            detection.results[0].hypothesis.score
        )

        age = (
            self.get_clock().now().nanoseconds / 1e9
            - message.header.stamp.sec
            - message.header.stamp.nanosec / 1e9
        )

        if not 0 <= age <= self.p['max_age']:
            return

        center = detection.bbox.center.position

        self.get_logger().info(
            f'car pixel=({center.x:.1f}, {center.y:.1f}), '
            f'score={score:.2f}',
            throttle_duration_sec=5.0
        )

        point = self.undistort(
            [[center.x, center.y]]
        )[0]

        if not np.isfinite(point).all():
            return

        if cv2.pointPolygonTest(
            self.region,
            tuple(map(float, point)),
            False
        ) < 0:
            self.get_logger().warning(
                'car outside calibration region'
            )
            return

        transformed = self.homography @ np.array(
            [*point, 1.0]
        )

        if abs(transformed[2]) < 1e-9:
            return

        x, y = transformed[:2] / transformed[2]

        if not np.isfinite([x, y]).all():
            return

        output = PointStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.p['map_frame']
        output.point.x = float(x)
        output.point.y = float(y)
        output.point.z = 0.0

        self.publisher.publish(output)

        self.get_logger().info(
            f'car map=({x:.3f}, {y:.3f})',
            throttle_duration_sec=5.0

        )


def main(args=None):
    run_node(WebcamLocalizer, args)
