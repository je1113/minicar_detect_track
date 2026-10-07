"""웹캠 픽셀을 지도 좌표로 변환한다. 보정 사각형 밖도 계산한다.

바닥의 네 대응점으로 호모그래피를 구한다.
마커 영역 밖은 같은 평면으로 가정하여 외삽한다.
기준점은 AMR 지도 좌표와 대응해야 하며, 카메라가 이동하면 재보정한다.
자동차 높이에 따른 오차와 영역 밖 정확도는 실제 측정으로 확인한다.
"""

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
            'calibrated': False,
            'map_frame': 'map',
            'pixel_points': [0.0] * 8,
            'map_points': [0.0] * 8,
            'camera_matrix': [0.0] * 9,
            'distortion': [0.0] * 5,
            'undistort': False,
            'max_age': 0.5,
            'calibration_width': 640,
            'calibration_height': 480,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        self.publisher = self.create_publisher(
            PointStamped, '/target/map_position', 10
        )
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
        self.create_subscription(
            Detection2DArray, '/webcam/detections', self.on_detection, 10
        )
        self.get_logger().info('5. localizer ready (보정 영역 밖 좌표 계산 허용)')

    def on_image(self, message):
        # 현재 영상과 보정에 사용한 영상의 해상도가 같아야 한다.
        self.image_size_ok = (message.width, message.height) == (
            self.p['calibration_width'], self.p['calibration_height']
        )

    def undistort(self, points):
        # 기준점과 자동차 위치에 동일한 왜곡 보정을 적용한다.
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not self.p['undistort']:
            return points
        matrix = np.asarray(self.p['camera_matrix']).reshape(3, 3)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError('유효한 내부 카메라 행렬이 필요합니다.')
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2), matrix,
            np.asarray(self.p['distortion']), P=matrix
        ).reshape(-1, 2)

    def setup_calibration(self):
        # 두 배열의 점은 같은 지점끼리 대응하며 경계 순서로 나열한다.
        source = self.undistort(self.p['pixel_points']).astype(np.float32)
        destination = np.asarray(
            self.p['map_points'], dtype=np.float32
        ).reshape(-1, 2)
        for points in (source, destination):
            if points.shape != (4, 2) or not np.isfinite(points).all():
                raise ValueError('유효한 대응점 4개가 필요합니다.')
            if (not cv2.isContourConvex(points)
                    or abs(cv2.contourArea(points)) < 1e-6):
                raise ValueError('기준점은 넓이가 있는 볼록 사각형이어야 합니다.')
        self.homography = cv2.getPerspectiveTransform(source, destination)
        if (not np.isfinite(self.homography).all()
                or np.linalg.matrix_rank(self.homography) < 3):
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

        # dummy를 제외하고 car 중 신뢰도가 가장 높은 감지를 사용한다.
        car_detections = []
        for detection in message.detections:
            if not detection.results:
                continue
            if detection.results[0].hypothesis.class_id == 'car':
                car_detections.append(detection)
        if not car_detections:
            return
        detection = max(
            car_detections,
            key=lambda d: float(d.results[0].hypothesis.score)
        )
        score = float(detection.results[0].hypothesis.score)

        # 오래된 감지와 미래 시각의 감지는 사용하지 않는다.
        age = (
            self.get_clock().now().nanoseconds / 1e9
            - message.header.stamp.sec
            - message.header.stamp.nanosec / 1e9
        )
        if not 0 <= age <= self.p['max_age']:
            return

        center = detection.bbox.center.position
        self.get_logger().info(
            f'car pixel=({center.x:.1f}, {center.y:.1f}), score={score:.2f}',
            throttle_duration_sec=5.0
        )
        point = self.undistort([[center.x, center.y]])[0]
        if not np.isfinite(point).all():
            return

        # 변경: 보정 사각형 밖이어도 return하지 않고 좌표 계산을 계속한다.
        # 사각형은 발행 제한이 아닌 외삽 여부를 알려주는 기준으로 사용한다.
        if cv2.pointPolygonTest(
            self.region, tuple(map(float, point)), False
        ) < 0:
            self.get_logger().info(
                'car outside calibration region: 외삽으로 지도 좌표 계산 중',
                throttle_duration_sec=5.0
            )

        transformed = self.homography @ np.array([*point, 1.0])
        # 분모가 거의 0이거나 계산값이 유효하지 않으면 발행하지 않는다.
        if not np.isfinite(transformed).all():
            return
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
            f'car map=({x:.3f}, {y:.3f})', throttle_duration_sec=5.0
        )


def main(args=None):
    run_node(WebcamLocalizer, args)


if __name__ == '__main__':
    main()
