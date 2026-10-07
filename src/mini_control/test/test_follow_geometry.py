"""follow.py 의 map 좌표 변환 / goal 전송 판단 순수 함수 단위 테스트."""
import math

from mini_control.follow import (
    detection_to_map_point,
    should_publish_follow_target,
)
import pytest

# amr_detector 영상이 704x704 라는 가정의 예시 값
FX = 500.0
CX = 352.0


def to_map(center_x, distance, robot_x=0.0, robot_y=0.0, robot_yaw=0.0):
    return detection_to_map_point(
        center_x, distance, FX, CX, robot_x, robot_y, robot_yaw)


def test_car_at_image_center_is_straight_ahead():
    x, y = to_map(CX, 1.0, robot_x=1.0, robot_y=2.0, robot_yaw=math.pi / 2)

    # 북쪽(+y)을 보고 있으면 정면 1 m 는 y 로 1 m 위
    assert (x, y) == pytest.approx((1.0, 3.0))


@pytest.mark.parametrize('yaw, expected', [
    (0.0, (2.0, 0.0)),
    (math.pi / 2, (0.0, 2.0)),
    (math.pi, (-2.0, 0.0)),
    (-math.pi / 2, (0.0, -2.0)),
])
def test_straight_ahead_follows_robot_heading(yaw, expected):
    assert to_map(CX, 2.0, robot_yaw=yaw) == pytest.approx(expected)


def test_car_on_right_of_image_is_to_the_right_of_robot():
    # 로봇이 +x(동쪽)를 볼 때 화면 오른쪽 = 로봇 오른쪽 = -y
    x, y = to_map(CX + FX, 1.0, robot_yaw=0.0)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(-1.0)


def test_car_on_left_of_image_is_to_the_left_of_robot():
    x, y = to_map(CX - FX, 1.0, robot_yaw=0.0)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(1.0)


def test_right_of_robot_facing_north_is_east():
    # 로봇이 +y(북쪽)를 볼 때 오른쪽은 +x(동쪽)
    x, y = to_map(CX + FX, 1.0, robot_yaw=math.pi / 2)

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(1.0)


def test_lateral_offset_scales_with_distance():
    near = to_map(CX + 100.0, 1.0)
    far = to_map(CX + 100.0, 3.0)

    # 같은 픽셀이라도 멀수록 옆으로 더 벌어진다 (X = (u - cx) * Z / fx)
    assert near[1] == pytest.approx(-0.2)
    assert far[1] == pytest.approx(-0.6)


def test_distance_from_robot_matches_pinhole_geometry():
    robot_x, robot_y = 3.0, -1.0
    x, y = to_map(CX + 200.0, 2.0, robot_x, robot_y, robot_yaw=0.7)

    lateral = 200.0 * 2.0 / FX
    assert math.hypot(x - robot_x, y - robot_y) == pytest.approx(
        math.hypot(2.0, lateral))


@pytest.mark.parametrize('fx', [0.0, -1.0])
def test_non_positive_fx_is_rejected(fx):
    with pytest.raises(ValueError):
        detection_to_map_point(CX, 1.0, fx, CX, 0.0, 0.0, 0.0)


PERIOD = 0.5
MIN_MOVE = 0.2
REFRESH = 3.0


def should_publish(last, new, elapsed):
    return should_publish_follow_target(
        last, new, elapsed, PERIOD, MIN_MOVE, REFRESH)


def test_first_target_is_always_published():
    assert should_publish(None, (1.0, 1.0), None) is True


def test_nothing_is_published_within_min_period():
    # 많이 움직여도 min_period 안에서는 goal 을 바꾸지 않는다
    assert should_publish((0.0, 0.0), (5.0, 5.0), 0.3) is False


@pytest.mark.parametrize('new, expected', [
    ((0.1, 0.0), False),
    ((0.2, 0.0), True),    # 경계값: 정확히 min_move 면 보낸다
    ((0.0, 1.0), True),
])
def test_publish_only_when_car_moved_enough(new, expected):
    assert should_publish((0.0, 0.0), new, 1.0) is expected


def test_stationary_car_is_refreshed_after_refresh_period():
    assert should_publish((0.0, 0.0), (0.0, 0.0), 2.9) is False
    assert should_publish((0.0, 0.0), (0.0, 0.0), 3.0) is True
