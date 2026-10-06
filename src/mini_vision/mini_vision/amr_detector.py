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
예전에는 depth 를 최신 1장만 저장하고 지금 처리하는 RGB 와 시각을 비교했다.
depth 가 RGB 보다 늦게 도착하면 둘의 시각이 0.2~1.1 초 벌어졌고,
시각 차이가 0.1 초를 넘으면 거리를 포기했다.
이 파일의 DepthBuffer / DepthDistance 가 다음과 같이 처리한다.
  1. depth 는 별도 노드/스레드로 받는다. YOLO 가 돌아도 멈추지 않는다.
  2. 최근 depth 를 시간 기준(기본 2 초)으로 모아 두고,
     RGB 촬영 시각에 가장 가까운 depth 를 골라 쓴다.
  3. 맞는 depth 가 아직 도착하지 않았으면 포기하지 않고 그 프레임의
     박스 위치를 보류해 두었다가, depth 가 도착하면 그때 잰다.
     (같은 시각의 박스와 depth 를 쓰므로 위치가 어긋나지 않는다.)
  4. depth 구멍에 대비해 패치를 한 번 더 키워서 다시 재 본다.
  5. 측정에 실패해도 마지막 유효 거리를 잠깐(기본 1 초) 유지하고
     최근 값들의 중앙값을 내보낸다. 유지 시간은 값이 나온 시점부터 센다.
RGB 구독 큐 깊이는 1 로 줄여서 YOLO 가 느려도 가장 최근 영상만 처리한다.
depth 가 늦게 오면 거리는 그만큼 이전 시각의 값이다 (예: 0.5 초 늦으면 0.5 초 전).
거리를 못 재도 감지 결과는 그대로 나간다. depth 가 끊겨도 car 를 놓치지 않는다.

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

from collections import deque
import os
import statistics
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray

from ament_index_python.packages import get_package_share_directory

from mini_vision.common import detect, run_node


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


# compressedDepth 메시지는 PNG 앞에 12바이트 헤더(압축 방식, 깊이 변환값)가 붙는다.
COMPRESSED_DEPTH_HEADER_SIZE = 12

# 두 감지 노드가 같은 기본값을 쓰도록 한곳에 둔다.
DEPTH_DEFAULTS = {
    'depth_topic': '/robot2/oakd/stereo/image_raw/compressedDepth',
    'depth_patch_size': 7,        # 박스 중심 주변 N×N 픽셀
    'depth_retry_patch_size': 15,  # 실패하면 이 크기로 한 번 더 (0 이면 안 함)
    'depth_min_valid': 5,         # 유효 depth 픽셀이 이보다 적으면 실패
    'depth_min_range': 0.2,       # 이보다 가까운 값은 무효 [m]
    'depth_max_range': 10.0,      # 이보다 먼 값은 무효 [m]
    'depth_max_dt': 0.1,          # RGB와 가장 가까운 depth 시각 허용 차이 (초)
    'depth_buffer_sec': 2.0,      # depth 를 모아 두는 시간, 늦은 depth 를 기다리는 한도 (초)
    'depth_hold_sec': 1.0,        # 측정 실패 시 마지막 유효 거리를 유지할 시간 (초)
    'depth_filter_count': 3,      # 중앙값을 낼 최근 유효 거리 개수 (1 이면 안 함)
}


def stamp_to_ns(stamp):
    """builtin_interfaces/Time 을 정수 나노초로 바꾼다."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def patch_median(depth_mm, u, v, patch_size, min_valid, min_range, max_range):
    """
    박스 중심 (u, v) 주변 patch_size×patch_size 유효 depth 의 중앙값 [m].

    depth_mm 은 16UC1(mm) 배열이다. 0(구멍)과 범위 밖 값은 뺀다.
    유효 픽셀이 min_valid 보다 적으면 (None, 유효 개수)를 반환한다.
    """
    height, width = depth_mm.shape[:2]
    half = max(int(patch_size) // 2, 0)

    u = min(max(int(round(u)), 0), width - 1)
    v = min(max(int(round(v)), 0), height - 1)

    patch = depth_mm[
        max(v - half, 0):v + half + 1,
        max(u - half, 0):u + half + 1
    ].astype(np.float32) / 1000.0

    valid = patch[
        np.isfinite(patch) & (patch >= min_range) & (patch <= max_range)
    ]

    if valid.size < min_valid:
        return None, int(valid.size)

    return float(np.median(valid)), int(valid.size)


class DistanceFilter:
    """최근 유효 거리를 잠깐 유지하고 중앙값으로 흔들림을 줄인다."""

    def __init__(self, hold_sec, count):
        self.hold_ns = int(float(hold_sec) * 1e9)
        self.recent = deque(maxlen=max(int(count), 1))

    def add(self, stamp_ns, distance, resolved_ns=None):
        """
        촬영 시각 stamp_ns 의 측정값을 넣는다. 실패면 distance 는 None.

        resolved_ns 는 그 값이 나온(depth 가 도착해 잰) 시각이다.
        유지 시간은 이 시각부터 센다. depth 가 늦게 와도 값이 나오자마자
        만료되지 않게 하려는 것이다. 주지 않으면 stamp_ns 와 같다.
        """
        if resolved_ns is None:
            resolved_ns = stamp_ns

        # 시각이 거꾸로 가면(시계 재설정 등) 예전 값은 버린다.
        if self.recent and resolved_ns < self.recent[-1][0]:
            self.recent.clear()

        if distance is not None:
            self.recent.append((resolved_ns, distance))

    def value(self, stamp_ns):
        """
        stamp_ns 시점에 내보낼 거리. hold_sec 안의 유효 값이 없으면 None.

        유지 시간 안의 최근 값들의 중앙값이다.
        """
        while self.recent and stamp_ns - self.recent[0][0] > self.hold_ns:
            self.recent.popleft()

        if not self.recent:
            return None

        return float(statistics.median(d for _, d in self.recent))

    def update(self, stamp_ns, distance):
        """새 측정값을 넣고 내보낼 거리를 바로 돌려준다."""
        self.add(stamp_ns, distance)

        return self.value(stamp_ns)


class DepthBuffer(Node):
    """
    depth 영상을 따로 받아 시간순으로 모아 두는 노드.

    감지 노드의 YOLO 와 다른 스레드에서 spin 하므로 YOLO 가 오래 걸려도
    depth 를 제때 받는다. 감지 노드의 cv2.imshow 는 원래 스레드에 그대로 둔다.
    """

    def __init__(self, topic, keep_sec):
        super().__init__('amr_depth_buffer')

        self.keep_ns = int(float(keep_sec) * 1e9)
        self.lock = threading.Lock()
        self.frames = deque()   # (촬영 시각 ns, 16UC1 depth 배열)

        self.create_subscription(
            CompressedImage,
            topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        # Node 에 executor 속성이 이미 있어서 다른 이름을 쓴다.
        self.spin_executor = SingleThreadedExecutor()
        self.spin_executor.add_node(self)
        self.stop_event = threading.Event()
        self.spin_thread = threading.Thread(
            target=self.spin_loop,
            daemon=True
        )
        self.spin_thread.start()

    def spin_loop(self):
        while not self.stop_event.is_set() and rclpy.ok():
            try:
                self.spin_executor.spin_once(timeout_sec=0.1)
            except Exception as error:
                # 종료 중 context 가 닫히면 조용히 끝낸다.
                if self.stop_event.is_set() or not rclpy.ok():
                    break

                # 콜백 오류로 depth 수신이 영영 멈추지 않게 계속 돈다.
                self.get_logger().error(
                    f'depth 수신 중 오류: {error}',
                    throttle_duration_sec=3.0
                )

    def depth_callback(self, image):
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

        stamp_ns = stamp_to_ns(image.header.stamp)

        with self.lock:
            # 시각이 거꾸로 가면 예전 영상은 버린다.
            if self.frames and stamp_ns < self.frames[-1][0]:
                self.frames.clear()

            self.frames.append((stamp_ns, depth))

            while stamp_ns - self.frames[0][0] > self.keep_ns:
                self.frames.popleft()

    def nearest(self, stamp_ns):
        """
        촬영 시각이 stamp_ns 에 가장 가까운 depth 를 찾는다.

        (depth 배열, 시각 차이 초, 모아 둔 장수, 가장 최근 depth 시각 ns)를
        반환한다. 아직 받은 depth 가 없으면 (None, None, 0, None).
        """
        with self.lock:
            if not self.frames:
                return None, None, 0, None

            count = len(self.frames)
            newest_ns = self.frames[-1][0]
            frame_ns, depth = min(
                self.frames, key=lambda frame: abs(frame[0] - stamp_ns)
            )

        return depth, abs(frame_ns - stamp_ns) / 1e9, count, newest_ns

    def close(self):
        self.stop_event.set()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        self.spin_executor.shutdown()
        self.destroy_node()


class DepthDistance:
    """RGB 박스 중심 (u, v) 의 거리를 depth 에서 구한다."""

    def __init__(self, params, logger, buffer=None):
        self.p = params
        self.logger = logger
        self.buffer = buffer or DepthBuffer(
            params['depth_topic'],
            params['depth_buffer_sec']
        )
        self.filter = DistanceFilter(
            params['depth_hold_sec'],
            params['depth_filter_count']
        )
        # 맞는 depth 가 아직 도착하지 않은 요청: (촬영 시각 ns, u, v, 영상 shape)
        self.pending = deque()

    def measure(self, u, v, frame_shape, image_stamp):
        """
        박스 중심 (u, v) 주변 depth 의 중앙값 [m].

        RGB 촬영 시각에 가장 가까운 depth 를 쓴다. depth 가 늦게 오면
        도착한 뒤에 재고, 그동안은 마지막 유효 거리를 유지한다.
        측정할 수 없으면 0.0 을 반환한다.
        """
        stamp_ns = stamp_to_ns(image_stamp)

        self.pending.append((stamp_ns, u, v, frame_shape))
        self.resolve_pending(stamp_ns)

        held = self.filter.value(stamp_ns)

        return 0.0 if held is None else held

    def resolve_pending(self, now_ns):
        """도착한 depth 부터 보류해 둔 요청을 시간순으로 거리로 바꾼다."""
        wait_ns = int(float(self.p['depth_buffer_sec']) * 1e9)

        while self.pending:
            stamp_ns, u, v, frame_shape = self.pending[0]
            depth, dt, count, newest_ns = self.buffer.nearest(stamp_ns)

            if depth is not None and dt <= self.p['depth_max_dt']:
                distance = self.measure_depth(depth, u, v, frame_shape)

            elif (
                now_ns - stamp_ns <= wait_ns
                and (depth is None or newest_ns < stamp_ns)
            ):
                # depth 가 아직 이 RGB 의 촬영 시각까지 오지 않았다.
                # 포기하지 않고 두었다가 도착하면 그때 잰다.
                self.logger.info(
                    f'depth 가 RGB 보다 늦게 도착하는 중입니다 '
                    f'(대기 {len(self.pending)}건).',
                    throttle_duration_sec=5.0
                )
                break

            else:
                self.log_too_far(depth, dt, count)
                distance = None

            self.pending.popleft()
            self.filter.add(stamp_ns, distance, now_ns)

    def log_too_far(self, depth, dt, count):
        if depth is None:
            self.logger.warning(
                'depth 영상을 받지 못했습니다.',
                throttle_duration_sec=3.0
            )
            return

        self.logger.warning(
            f'RGB에 가장 가까운 depth 와 시각 차이 {dt:.3f}s > '
            f"{self.p['depth_max_dt']}s (모아 둔 depth {count}장)",
            throttle_duration_sec=3.0
        )

    def measure_depth(self, depth, u, v, frame_shape):
        """고른 depth 한 장에서 박스 중심 거리를 잰다. 실패하면 None."""
        # RGB와 depth 가 정렬·같은 해상도라는 전제를 확인한다.
        if depth.shape[:2] != frame_shape[:2]:
            self.logger.warning(
                f'RGB {frame_shape[:2]} 와 depth {depth.shape[:2]} '
                f'해상도가 다릅니다. 거리를 측정하지 않습니다.',
                throttle_duration_sec=3.0
            )
            return None

        sizes = [self.p['depth_patch_size']]
        retry = int(self.p['depth_retry_patch_size'])
        if retry > sizes[0]:
            sizes.append(retry)

        counts = []
        for size in sizes:
            distance, valid = patch_median(
                depth, u, v, size,
                self.p['depth_min_valid'],
                self.p['depth_min_range'],
                self.p['depth_max_range']
            )
            if distance is not None:
                return distance
            counts.append(f'{size}x{size} 유효 {valid}개')

        self.logger.warning(
            f'depth 유효 픽셀 부족: 박스 중심 ({u:.0f}, {v:.0f}) '
            f"{', '.join(counts)} < {self.p['depth_min_valid']}개",
            throttle_duration_sec=3.0
        )
        return None

    def close(self):
        self.buffer.close()


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
            # 거리 추가를 위해 수정했다: depth 관련 파라미터 (위 DEPTH_DEFAULTS)
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
