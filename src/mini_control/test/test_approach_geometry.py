"""approach.py 의 ROS 비의존 순수 함수 단위 테스트."""
import math

from mini_control.approach import (
    ApproachStatus,
    compute_goal_pose,
    Point2D,
    quaternion_to_yaw,
    trim_path_end,
    yaw_to_quaternion,
)
import pytest


def test_goal_is_car_position():
    car = Point2D(-0.0355, 1.025)

    goal = compute_goal_pose(Point2D(2.0, 2.0), car)

    assert (goal.x, goal.y) == pytest.approx((car.x, car.y))


@pytest.mark.parametrize('robot, expected_yaw', [
    (Point2D(1.0, 3.0), 0.0),                 # 로봇이 -x 쪽 → +x 방향을 봄
    (Point2D(2.0, 2.0), math.pi / 2),
    (Point2D(3.0, 4.0), -3 * math.pi / 4),
])
def test_goal_yaw_faces_car_from_robot(robot, expected_yaw):
    goal = compute_goal_pose(robot, Point2D(2.0, 3.0))

    assert goal.yaw == pytest.approx(expected_yaw)


def path_length(path):
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:]))


def test_trim_path_end_removes_standoff_length():
    path = [Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(2.0, 2.0)]

    trimmed = trim_path_end(path, 1.0)

    assert path_length(trimmed) == pytest.approx(3.0)
    assert trimmed[-1] == pytest.approx(Point2D(2.0, 1.0))


def test_trim_path_end_cuts_inside_earlier_segment():
    path = [Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(2.0, 0.5)]

    trimmed = trim_path_end(path, 1.0)

    assert trimmed[-1] == pytest.approx(Point2D(1.5, 0.0))


def test_trim_path_end_shorter_than_standoff_keeps_start_only():
    path = [Point2D(0.0, 0.0), Point2D(0.5, 0.0)]

    assert trim_path_end(path, 1.0) == [Point2D(0.0, 0.0)]


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
