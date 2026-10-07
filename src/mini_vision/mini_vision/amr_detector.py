"""
TurtleBot4 AMR 카메라 토픽 → YOLO 감지 + depth 거리 → /amr/detections.

[거리 추가를 위해 수정했다]
해상도를 맞춰 둔 압축 토픽 두 개를 구독한다 (둘 다 704x704).
  RGB   /robot2/oakd/rgb/image_raw/compressed          (JPEG, bgr8)
  depth /robot2/oakd/stereo/image_raw/compressedDepth  (PNG, 16UC1 mm)
박스 중심 주변의 depth 값으로 자동차까지 거리를 구한다.
depth 영상은 RGB에 정렬되어 있고 해상도도 같다.
그래서 박스 중심 픽셀 (u, v)를 depth 영상에 그대로 쓴다.

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

import csv
import os
import time

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
            'confidence': 0.85,
            'device': 'cuda:0',
            'show_window': True,
            # 거리 추가를 위해 수정했다: depth 관련 파라미터
            'depth_topic': '/robot2/oakd/stereo/image_raw/compressedDepth',
            'depth_patch_size': 7,      # 박스 중심 주변 N×N 픽셀
            'depth_min_valid': 5,       # 유효 depth 픽셀이 이보다 적으면 실패
            'depth_max_dt': 0.1,        # RGB와 depth 촬영 시각 허용 차이 (초)
            # 시간이 원인인지 확인하려고 추가했다: 요약 로그 주기 (초), 0 이면 끈다
            'time_log_period': 2.0,
            # 프레임마다 시각 정보를 CSV 로 저장하는 폴더, 빈 문자열이면 저장 안 함
            'time_log_dir': '~/minicar_time_logs',
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

        # 시간이 원인인지 확인하려고 추가했다.
        self.reset_time_stats()
        self.open_time_csv()

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

        if self.p['time_log_period'] > 0:
            self.create_timer(
                float(self.p['time_log_period']),
                self.log_time_stats
            )

    # =========================================================
    # 시간이 원인인지 확인하려고 추가했다: 시간 진단 통계
    #
    # 주기마다 아래를 한 줄로 요약한다.
    #   - RGB / depth 입력 속도
    #   - 촬영 → 수신 지연 (PC 시각 - header.stamp)
    #   - RGB 와 직전 depth 의 촬영 시각 차이
    #   - 거리 측정 실패 횟수와 이유
    # =========================================================
    def reset_time_stats(self):
        self.stat_start = time.monotonic()
        self.stat_rgb = 0
        self.stat_depth = 0
        self.stat_latency = []   # 촬영 → 수신 지연 [s]
        self.stat_process = []   # image_callback 처리 시간 [ms]
        self.stat_dt = []        # RGB 시각 - 직전 depth 시각 [s]
        self.stat_dist_calls = 0
        self.stat_dist_fail = 0
        self.stat_dt_over = 0    # depth_max_dt 초과로 실패
        self.stat_no_depth = 0   # depth 를 아직 못 받아 실패

    def record_frame_times(self, header, row):
        self.stat_rgb += 1

        stamp = Time.from_msg(header.stamp)

        # 시각을 채우지 않은 메시지는 건너뛴다.
        if stamp.nanoseconds <= 0:
            return

        latency = (self.get_clock().now() - stamp).nanoseconds / 1e9

        self.stat_latency.append(latency)

        row['rgb_stamp'] = stamp.nanoseconds / 1e9
        row['latency'] = latency

        if self.depth_stamp is not None:
            dt = (stamp - self.depth_stamp).nanoseconds / 1e9

            self.stat_dt.append(dt)

            row['depth_stamp'] = self.depth_stamp.nanoseconds / 1e9
            row['rgb_minus_depth'] = dt

    # =========================================================
    # 프레임마다 시각 정보를 CSV 로 저장한다
    #
    # 한 줄 = RGB 한 프레임. 컬럼:
    #   wall_time        PC 시각 (epoch 초)
    #   rgb_stamp        RGB 촬영 시각 (header.stamp)
    #   depth_stamp      그 시점에 비교한 직전 depth 의 촬영 시각
    #   rgb_minus_depth  RGB - depth 촬영 시각 차 [s]
    #   latency          wall_time - rgb_stamp, 촬영 → 수신 지연 [s]
    #   process_ms       image_callback 처리 시간 [ms]
    #   cars             토픽에 실은 car 개수
    #   distance         첫 car 의 측정 거리 [m] (실패하면 0.0)
    #   fail_reason      거리 실패 이유 (no_depth / size_mismatch / dt_over / hole)
    # =========================================================
    TIME_CSV_COLUMNS = (
        'wall_time', 'rgb_stamp', 'depth_stamp', 'rgb_minus_depth',
        'latency', 'process_ms', 'cars', 'distance', 'fail_reason',
    )

    def open_time_csv(self):
        self.time_csv = None
        self.time_csv_writer = None

        directory = str(self.p['time_log_dir']).strip()

        if not directory:
            return

        try:
            directory = os.path.expanduser(directory)
            os.makedirs(directory, exist_ok=True)

            path = os.path.join(
                directory,
                time.strftime('amr_detector_%Y%m%d_%H%M%S.csv')
            )

            self.time_csv = open(
                path, 'w', newline='', encoding='utf-8'
            )
            self.time_csv_writer = csv.writer(self.time_csv)
            self.time_csv_writer.writerow(self.TIME_CSV_COLUMNS)
            self.time_csv.flush()

            self.get_logger().info(f'시간 로그 저장: {path}')

        except OSError as error:
            self.time_csv = None
            self.time_csv_writer = None

            self.get_logger().error(
                f'시간 로그 파일을 열 수 없습니다: {error}'
            )

    def write_time_row(self, row):
        if self.time_csv is None:
            return

        def number(key, digits):
            value = row.get(key)

            return '' if value is None else f'{value:.{digits}f}'

        try:
            self.time_csv_writer.writerow([
                f'{time.time():.3f}',
                number('rgb_stamp', 6),
                number('depth_stamp', 6),
                number('rgb_minus_depth', 6),
                number('latency', 6),
                number('process_ms', 2),
                row.get('cars', 0),
                number('distance', 3),
                row.get('fail_reason', ''),
            ])
            self.time_csv.flush()

        except OSError as error:
            self.time_csv = None
            self.time_csv_writer = None

            self.get_logger().error(
                f'시간 로그 저장 중 오류, 저장을 멈춥니다: {error}'
            )

    def log_time_stats(self):
        elapsed = max(time.monotonic() - self.stat_start, 1e-6)

        rgb_hz = self.stat_rgb / elapsed
        depth_hz = self.stat_depth / elapsed

        parts = [
            f'RGB {rgb_hz:.1f}Hz',
            f'depth {depth_hz:.1f}Hz',
        ]

        latency = self.stat_latency

        if latency:
            parts.append(
                f'촬영→수신 지연 평균 {sum(latency) / len(latency):.3f}s '
                f'(최소 {min(latency):.3f}, 최대 {max(latency):.3f})'
            )

        if self.stat_process:
            process = self.stat_process

            parts.append(
                f'처리 평균 {sum(process) / len(process):.0f}ms '
                f'(최대 {max(process):.0f})'
            )

        dt = self.stat_dt

        if dt:
            parts.append(
                f'RGB-depth 시각차 평균 {sum(dt) / len(dt):+.3f}s '
                f'(절대 최대 {max(abs(x) for x in dt):.3f}, '
                f"허용 {self.p['depth_max_dt']}s)"
            )

        if self.stat_dist_calls:
            other = (
                self.stat_dist_fail
                - self.stat_dt_over
                - self.stat_no_depth
            )

            parts.append(
                f'거리 실패 {self.stat_dist_fail}/{self.stat_dist_calls} '
                f'(시각차 초과 {self.stat_dt_over}, '
                f'depth 없음 {self.stat_no_depth}, 기타 {other})'
            )

        self.get_logger().info('[시간] ' + ' | '.join(parts))

        # 원인 판단에 도움이 되는 경고
        if latency and min(latency) < 0.0:
            self.get_logger().warning(
                '[시간] 촬영 시각이 PC 시각보다 미래입니다 '
                f'(최소 지연 {min(latency):.3f}s). '
                '로봇과 PC 시계가 어긋났을 수 있습니다.'
            )

        if self.stat_dt_over > 0 and rgb_hz > 0.0:
            self.get_logger().warning(
                f'[시간] RGB-depth 시각차가 허용치를 넘어 거리 실패 '
                f'{self.stat_dt_over}회. 프레임 간격은 '
                f'{1.0 / rgb_hz:.3f}s 입니다. 최신 depth 한 장만 '
                f'비교하는 방식의 한계일 수 있습니다.'
            )

        self.reset_time_stats()

    def depth_callback(self, image):
        # 시간이 원인인지 확인하려고 추가했다.
        self.stat_depth += 1

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
        self.distance_fail_reason = ''

        if self.depth_m is None:
            self.stat_no_depth += 1
            self.distance_fail_reason = 'no_depth'

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
            self.distance_fail_reason = 'size_mismatch'
            return 0.0

        # 너무 오래된 depth 로 재지 않도록 촬영 시각 차이를 확인한다.
        dt = abs(
            (Time.from_msg(image_stamp) - self.depth_stamp).nanoseconds
        ) / 1e9
        if dt > self.p['depth_max_dt']:
            self.stat_dt_over += 1
            self.distance_fail_reason = 'dt_over'

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
            self.distance_fail_reason = 'hole'
            return 0.0

        return float(np.median(valid))

    def image_callback(self, image):
        # 감지 결과는 카메라 영상의 촬영 시각과 frame_id를 그대로 쓴다.
        header = image.header

        # 시간이 원인인지 확인하려고 추가했다.
        callback_start = time.perf_counter()
        row = {}
        self.record_frame_times(header, row)

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

                distance = self.measure_distance(
                    detection.bbox.center.position.x,
                    detection.bbox.center.position.y,
                    frame.shape,
                    header.stamp
                )
                detection.results[0].pose.pose.position.z = distance
                distances.append(distance)

                # 시간이 원인인지 확인하려고 추가했다.
                self.stat_dist_calls += 1

                if distance <= 0.0:
                    self.stat_dist_fail += 1

                if 'distance' not in row:
                    row['distance'] = distance
                    row['fail_reason'] = (
                        self.distance_fail_reason if distance <= 0.0 else ''
                    )

            # 감지하지 못하면 detect()가 빈 Detection2DArray를 반환한다.
            self.publisher.publish(message)

            row['cars'] = len(message.detections)

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

        finally:
            # 시간이 원인인지 확인하려고 추가했다.
            process_ms = (time.perf_counter() - callback_start) * 1000.0

            self.stat_process.append(process_ms)

            row['process_ms'] = process_ms
            self.write_time_row(row)

    def destroy_node(self):
        if self.time_csv is not None:
            self.time_csv.close()
            self.time_csv = None

        if self.p['show_window']:
            cv2.destroyAllWindows()

        return super().destroy_node()


def main(args=None):
    run_node(AmrDetector, args)
