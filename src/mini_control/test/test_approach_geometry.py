"""approach.py 의 ROS 비의존 순수 함수 단위 테스트."""
import math

from mini_control.approach import (
    ApproachStatus,
    compute_view_pose,
    Point2D,
    quaternion_to_yaw,
    yaw_to_quaternion,
)
import pytest

VIEW_OFFSET_X = 0.6755
VIEW_OFFSET_Y = 0.8150


def test_goal_is_car_position_plus_offset():
    car = Point2D(-0.0355, 1.025)

    goal = compute_view_pose(car, VIEW_OFFSET_X, VIEW_OFFSET_Y)

    assert goal.x == pytest.approx(car.x + VIEW_OFFSET_X)
    assert goal.y == pytest.approx(car.y + VIEW_OFFSET_Y)


def test_goal_does_not_depend_on_car_heading_or_distance():
    near = compute_view_pose(Point2D(0.0, 0.0), VIEW_OFFSET_X, VIEW_OFFSET_Y)
    far = compute_view_pose(Point2D(10.0, -5.0), VIEW_OFFSET_X, VIEW_OFFSET_Y)

    assert far.x - near.x == pytest.approx(10.0)
    assert far.y - near.y == pytest.approx(-5.0)
    assert far.yaw == pytest.approx(near.yaw)


@pytest.mark.parametrize('offset_x, offset_y, expected_yaw', [
    (-1.0, -1.0, math.pi / 4),           # goal 이 3사분면 쪽 → 1사분면 방향을 봄
    (1.0, -1.0, 3 * math.pi / 4),
    (1.0, 1.0, -3 * math.pi / 4),
    (-1.0, 1.0, -math.pi / 4),
    (-1.0, 0.0, 0.0),                    # +x 축
    (0.0, -1.0, math.pi / 2),            # +y 축
])
def test_goal_yaw_faces_car(offset_x, offset_y, expected_yaw):
    car = Point2D(2.0, 3.0)

    goal = compute_view_pose(car, offset_x, offset_y)

    assert goal.yaw == pytest.approx(expected_yaw)
    heading_to_car = math.atan2(car.y - goal.y, car.x - goal.x)
    assert goal.yaw == pytest.approx(heading_to_car)


@pytest.mark.parametrize('yaw', [
    0.0, math.pi / 6, math.pi / 2, 2.5, -0.7, -math.pi / 2, -3.0,
])
def test_yaw_quaternion_roundtrip(yaw):
    assert quaternion_to_yaw(yaw_to_quaternion(yaw)) == pytest.approx(yaw)


def test_yaw_quaternion_is_unit_length():
    q = yaw_to_quaternion(1.234)

    assert math.hypot(q.x, q.y, q.z, q.w) == pytest.approx(1.0)


def test_status_values_match_team_interface_strings():
    assert [s.value for s in ApproachStatus] == [
        'IDLE', 'MOVING', 'ARRIVED', 'FAILED', 'CANCELED']
