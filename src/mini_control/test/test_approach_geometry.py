"""approach.py 의 ROS 비의존 순수 함수 단위 테스트."""
import math

from mini_control.approach import (
    ApproachStatus,
    compute_approach_pose,
    Point2D,
    quaternion_to_yaw,
    should_retarget,
    yaw_to_quaternion,
)
import pytest

APPROACH_DISTANCE = 0.5


def test_far_car_goal_is_exactly_approach_distance_in_front():
    robot = Point2D(0.0, 0.0)
    car = Point2D(3.0, 4.0)

    goal = compute_approach_pose(robot, car, APPROACH_DISTANCE)

    assert math.hypot(car.x - goal.x, car.y - goal.y) == pytest.approx(APPROACH_DISTANCE)
    assert goal.x == pytest.approx(3.0 - 0.5 * 3.0 / 5.0)
    assert goal.y == pytest.approx(4.0 - 0.5 * 4.0 / 5.0)


def test_far_car_goal_lies_between_robot_and_car():
    robot = Point2D(1.0, 1.0)
    car = Point2D(-2.0, 5.0)

    goal = compute_approach_pose(robot, car, APPROACH_DISTANCE)

    robot_to_goal = math.hypot(goal.x - robot.x, goal.y - robot.y)
    robot_to_car = math.hypot(car.x - robot.x, car.y - robot.y)
    assert robot_to_goal == pytest.approx(robot_to_car - APPROACH_DISTANCE)


@pytest.mark.parametrize('car', [
    Point2D(1.3, 2.0),              # 0.3 m
    Point2D(1.0 + APPROACH_DISTANCE, 2.0),  # 경계값: 정확히 approach_distance
])
def test_close_car_keeps_robot_position_and_only_turns(car):
    robot = Point2D(1.0, 2.0)

    goal = compute_approach_pose(robot, car, APPROACH_DISTANCE)

    assert (goal.x, goal.y) == (robot.x, robot.y)
    assert goal.yaw == pytest.approx(math.atan2(car.y - robot.y, car.x - robot.x))


def test_car_at_robot_position_does_not_divide_by_zero():
    robot = Point2D(1.0, 1.0)

    goal = compute_approach_pose(robot, robot, APPROACH_DISTANCE)

    assert (goal.x, goal.y) == (robot.x, robot.y)


@pytest.mark.parametrize('car, expected_yaw', [
    (Point2D(2.0, 2.0), math.pi / 4),           # 1사분면
    (Point2D(-2.0, 2.0), 3 * math.pi / 4),      # 2사분면
    (Point2D(-2.0, -2.0), -3 * math.pi / 4),    # 3사분면
    (Point2D(2.0, -2.0), -math.pi / 4),         # 4사분면
    (Point2D(3.0, 0.0), 0.0),                   # +x 축
    (Point2D(0.0, 3.0), math.pi / 2),           # +y 축
])
def test_goal_yaw_faces_car_in_each_quadrant(car, expected_yaw):
    robot = Point2D(0.0, 0.0)

    goal = compute_approach_pose(robot, car, APPROACH_DISTANCE)

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


@pytest.mark.parametrize('new_car, expected', [
    (Point2D(0.1, 0.0), False),
    (Point2D(0.3, 0.0), True),   # 경계값: 정확히 threshold 면 교체
    (Point2D(0.0, 1.0), True),
])
def test_should_retarget_only_when_car_moved_enough(new_car, expected):
    assert should_retarget(Point2D(0.0, 0.0), new_car, threshold=0.3) is expected


def test_status_values_match_team_interface_strings():
    assert [s.value for s in ApproachStatus] == [
        'IDLE', 'MOVING', 'ARRIVED', 'FAILED', 'CANCELED']
