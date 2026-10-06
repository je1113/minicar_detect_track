"""depth_distance 테스트: 패치 중앙값, 유지·중앙값 필터, 늦게 오는 depth 처리."""

from mini_vision.depth_distance import (
    DEPTH_DEFAULTS,
    DepthDistance,
    DistanceFilter,
    patch_median,
    stamp_to_ns,
)
import numpy as np

NS = 1_000_000_000
RANGE = {'min_range': 0.2, 'max_range': 10.0}
SHAPE = (64, 64, 3)


class Stamp:
    def __init__(self, seconds):
        self.sec = int(seconds)
        self.nanosec = int(round((seconds - int(seconds)) * NS))


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)

    def warning(self, message, **kwargs):
        self.messages.append(message)


class FakeBuffer:
    """DepthBuffer 대신 쓰는 가짜. 시각과 depth 를 직접 넣는다."""

    def __init__(self):
        self.frames = []

    def add(self, seconds, value_mm=1500):
        self.frames.append((int(seconds * NS), depth_image(value_mm)))

    def nearest(self, stamp_ns):
        if not self.frames:
            return None, None, 0, None

        newest_ns = max(frame[0] for frame in self.frames)
        frame_ns, depth = min(
            self.frames, key=lambda frame: abs(frame[0] - stamp_ns)
        )

        return depth, abs(frame_ns - stamp_ns) / 1e9, len(self.frames), newest_ns

    def close(self):
        pass


def depth_image(value_mm=1500, size=64):
    return np.full((size, size), value_mm, dtype=np.uint16)


def make_measurer(buffer):
    return DepthDistance(dict(DEPTH_DEFAULTS), FakeLogger(), buffer)


def test_stamp_to_ns():
    assert stamp_to_ns(Stamp(3.25)) == 3_250_000_000


def test_patch_median_uniform():
    distance, valid = patch_median(
        depth_image(1500), 32, 32, 7, 5, **RANGE
    )
    assert distance == 1.5
    assert valid == 49


def test_patch_median_ignores_holes_and_outliers():
    depth = depth_image(1500)
    depth[30:35, 30:35] = 0          # 구멍
    depth[32, 32] = 9000             # 튀는 값
    distance, _ = patch_median(depth, 32, 32, 7, 5, **RANGE)
    assert distance == 1.5


def test_patch_median_too_few_valid_pixels():
    depth = np.zeros((64, 64), dtype=np.uint16)
    depth[32, 30:34] = 1500          # 유효 4개 < min_valid 5
    distance, valid = patch_median(depth, 32, 32, 7, 5, **RANGE)
    assert distance is None
    assert valid == 4


def test_patch_median_range_gate():
    near = patch_median(depth_image(100), 32, 32, 7, 5, **RANGE)
    far = patch_median(depth_image(12000), 32, 32, 7, 5, **RANGE)
    assert near[0] is None
    assert far[0] is None


def test_patch_median_larger_patch_recovers():
    depth = np.zeros((64, 64), dtype=np.uint16)
    depth[26:39, 26:39] = 2000       # 가운데 7x7 은 비어 있다
    depth[29:36, 29:36] = 0
    assert patch_median(depth, 32, 32, 7, 5, **RANGE)[0] is None
    assert patch_median(depth, 32, 32, 15, 5, **RANGE)[0] == 2.0


def test_patch_median_clamps_to_image_border():
    distance, _ = patch_median(depth_image(1500), -5, 999, 7, 5, **RANGE)
    assert distance == 1.5


def test_filter_median_of_recent_values():
    distance_filter = DistanceFilter(hold_sec=0.5, count=3)
    assert distance_filter.update(0 * NS, 1.0) == 1.0
    assert distance_filter.update(1 * NS // 10, 1.1) == 1.05
    # 튀는 값 5.0 하나는 중앙값에서 걸러진다.
    assert distance_filter.update(2 * NS // 10, 5.0) == 1.1


def test_filter_holds_last_value_then_gives_up():
    distance_filter = DistanceFilter(hold_sec=0.5, count=3)
    distance_filter.update(0, 2.0)

    assert distance_filter.update(3 * NS // 10, None) == 2.0
    assert distance_filter.update(5 * NS // 10, None) == 2.0
    assert distance_filter.update(6 * NS // 10, None) is None


def test_filter_without_any_value():
    assert DistanceFilter(0.5, 3).update(0, None) is None


def test_filter_count_one_passes_value_through():
    distance_filter = DistanceFilter(hold_sec=0.5, count=1)
    distance_filter.update(0, 1.0)
    assert distance_filter.update(1 * NS // 10, 3.0) == 3.0


def test_filter_holds_from_when_value_was_resolved():
    distance_filter = DistanceFilter(hold_sec=1.0, count=3)
    # 촬영은 0초, depth 가 늦게 와서 1.2초에 잰 값
    distance_filter.add(0, 2.0, resolved_ns=12 * NS // 10)

    assert distance_filter.value(12 * NS // 10) == 2.0
    assert distance_filter.value(20 * NS // 10) == 2.0
    assert distance_filter.value(23 * NS // 10) is None


def test_filter_clears_when_time_goes_backwards():
    distance_filter = DistanceFilter(hold_sec=0.5, count=3)
    distance_filter.update(100 * NS, 1.0)
    assert distance_filter.update(1 * NS, None) is None


def test_measure_with_matching_depth():
    buffer = FakeBuffer()
    buffer.add(1.0)
    measurer = make_measurer(buffer)

    assert measurer.measure(32, 32, SHAPE, Stamp(1.02)) == 1.5
    assert not measurer.pending


def test_measure_waits_for_late_depth_then_resolves():
    buffer = FakeBuffer()
    buffer.add(0.5)                  # RGB(1.0) 보다 오래된 depth 만 왔다
    measurer = make_measurer(buffer)

    # 맞는 depth 가 아직 없다: 포기하지 않고 보류한다.
    assert measurer.measure(32, 32, SHAPE, Stamp(1.0)) == 0.0
    assert len(measurer.pending) == 1

    # 늦게 도착한 depth(1.0)로 보류했던 요청을 푼다.
    buffer.add(1.0, 2000)
    assert measurer.measure(32, 32, SHAPE, Stamp(1.25)) == 2.0
    assert len(measurer.pending) == 1    # 1.25 의 depth 는 아직 안 왔다


def test_measure_gives_up_when_depth_never_arrives():
    buffer = FakeBuffer()
    buffer.add(0.0)
    measurer = make_measurer(buffer)

    assert measurer.measure(32, 32, SHAPE, Stamp(1.0)) == 0.0
    assert len(measurer.pending) == 1

    # 기다리는 한도(2초)를 넘기면 그 요청은 실패로 버린다.
    assert measurer.measure(32, 32, SHAPE, Stamp(4.0)) == 0.0
    assert len(measurer.pending) == 1    # 4.0 요청만 남는다


def test_measure_gap_between_depth_frames_fails_immediately():
    buffer = FakeBuffer()
    buffer.add(0.0)
    buffer.add(1.0)                  # 0.5 부근에는 depth 가 없다
    measurer = make_measurer(buffer)

    assert measurer.measure(32, 32, SHAPE, Stamp(0.5)) == 0.0
    assert not measurer.pending          # 기다려도 오지 않으니 바로 실패


def test_measure_holds_last_value_after_failure():
    buffer = FakeBuffer()
    buffer.add(1.0)
    buffer.add(2.0)
    measurer = make_measurer(buffer)

    assert measurer.measure(32, 32, SHAPE, Stamp(1.0)) == 1.5
    # 1.5 부근은 depth 공백 → 실패, 그래도 1초 안이라 마지막 값을 유지한다.
    assert measurer.measure(32, 32, SHAPE, Stamp(1.5)) == 1.5
    # 마지막 유효 값(1.0)에서 1초가 넘으면 N/A.
    assert measurer.measure(32, 32, SHAPE, Stamp(2.6)) == 0.0


def test_measure_resolution_mismatch_is_not_measured():
    buffer = FakeBuffer()
    buffer.add(1.0)
    measurer = make_measurer(buffer)

    assert measurer.measure(32, 32, (128, 128, 3), Stamp(1.0)) == 0.0
