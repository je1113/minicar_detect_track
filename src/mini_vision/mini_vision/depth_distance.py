"""
AMR 카메라 depth 로 박스 중심 거리를 구하는 공용 코드.

amr_detector 와 amr_detector_topic 이 함께 쓴다.

[RGB 와 depth 시각이 어긋나 N/A 가 나던 문제를 고치기 위해 만들었다]
예전에는 depth 를 최신 1장만 저장하고, 지금 처리하는 RGB 와 시각을 비교했다.
depth 가 RGB 보다 늦게 도착하면 둘의 시각이 0.2~1.1 초 벌어졌고,
시각 차이가 0.1 초를 넘으면 거리를 포기했다.

  1. depth 는 별도 노드/스레드로 받는다. YOLO 가 돌아도 멈추지 않는다.
  2. 최근 depth 를 시간 기준(기본 2 초)으로 모아 두고,
     RGB 촬영 시각에 가장 가까운 depth 를 골라 쓴다.
  3. 맞는 depth 가 아직 도착하지 않았으면 포기하지 않고 그 프레임의
     박스 위치를 보류해 두었다가, depth 가 도착하면 그때 잰다.
     (같은 시각의 박스와 depth 를 쓰므로 위치가 어긋나지 않는다.)
  4. depth 구멍에 대비해 패치를 한 번 더 키워서 다시 재 본다.
  5. 측정에 실패해도 마지막 유효 거리를 잠깐(기본 1 초) 유지하고
     최근 값들의 중앙값을 내보낸다. 유지 시간은 값이 나온 시점부터 센다.

depth 가 늦게 오면 거리는 그만큼 이전 시각의 값이다 (예: 0.5 초 늦으면 0.5 초 전).
거리를 못 재도 감지 결과는 그대로 나간다. depth 가 끊겨도 car 를 놓치지 않는다.
측정할 수 없으면 0.0 을 반환한다 (받는 쪽은 0 이하를 측정 실패로 본다).
"""

from collections import deque
import statistics
import threading

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

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
