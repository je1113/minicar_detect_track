"""approach.py 의 cmd_vel fallback 경로 제어 단위 테스트."""
import math

from mini_control.approach import (
    compose_pose,
    route_start_index,
    normalize_angle,
    Point2D,
    Pose2D,
    relative_pose,
    RouteConfig,
    RouteFollower,
    RoutePhase,
)
import pytest

DT = 0.1
WAYPOINTS = [Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(1.79, 1.77)]
GOAL = Pose2D(0.64, 1.84, -2.27)


def simulate(follower, pose, max_steps=5000):
    """속도 명령을 그대로 적분하는 이상적인 diff-drive 로 이동시킨다."""
    turns = []
    for _ in range(max_steps):
        v = follower.step(pose)
        if follower.done:
            return pose, turns
        if v.angular and not v.linear:
            turns.append(math.copysign(1, v.angular))
        yaw = pose.yaw + v.angular * DT
        pose = Pose2D(pose.x + v.linear * math.cos(yaw) * DT,
                      pose.y + v.linear * math.sin(yaw) * DT,
                      normalize_angle(yaw))
    raise AssertionError('route did not finish')


@pytest.mark.parametrize('angle, expected', [
    (0.0, 0.0), (math.pi / 2, math.pi / 2), (3 * math.pi / 2, -math.pi / 2),
    (-3 * math.pi / 2, math.pi / 2), (2 * math.pi, 0.0),
])
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


def test_compose_is_inverse_of_relative():
    base = Pose2D(1.0, -2.0, 0.7)
    pose = Pose2D(-0.5, 3.0, -2.5)

    restored = compose_pose(base, relative_pose(base, pose))

    assert restored == pytest.approx(pose)


def test_odom_delta_is_applied_in_map_frame():
    # map 에서 +y 를 보는 로봇이 odom 상 전진 1m 하면 map 에서는 +y 로 1m
    amcl = Pose2D(1.0, 1.0, math.pi / 2)
    odom_at_amcl = Pose2D(5.0, 5.0, 0.0)
    odom_now = Pose2D(6.0, 5.0, 0.0)

    pose = compose_pose(amcl, relative_pose(odom_at_amcl, odom_now))

    assert pose == pytest.approx(Pose2D(1.0, 2.0, math.pi / 2))


@pytest.mark.parametrize('robot, expected', [
    (Point2D(0.0, 0.0), 1),    # 출발점: goal 이 P1 보다 가깝지만 P1 로 간다
    (Point2D(1.0, 0.1), 1),    # 첫 직진 도중
    (Point2D(1.95, 1.0), 2),   # 두 번째 직진 도중
    (Point2D(1.0, 1.85), 3),   # 마지막 직진 도중
])
def test_route_start_follows_nearest_segment(robot, expected):
    assert route_start_index(robot, WAYPOINTS + [GOAL.position]) == expected


def test_route_reaches_goal_with_two_left_turns():
    cfg = RouteConfig()
    route = WAYPOINTS + [GOAL.position]
    follower = RouteFollower(route, GOAL.yaw, cfg, route_start_index(Point2D(0.0, 0.0), route))

    final, turns = simulate(follower, Pose2D(0.0, 0.0, 0.0))

    assert math.hypot(final.x - GOAL.x, final.y - GOAL.y) < cfg.position_tolerance
    assert abs(normalize_angle(final.yaw - GOAL.yaw)) < cfg.yaw_tolerance
    # 좌회전(+) 만 하고 우회전하지 않는다
    assert turns and all(t > 0 for t in turns)


def test_route_rotates_in_place_before_driving():
    follower = RouteFollower([Point2D(0.0, 1.0)], 0.0, RouteConfig())

    v = follower.step(Pose2D(0.0, 0.0, 0.0))

    assert v.linear == 0.0
    assert v.angular > 0.0
    assert follower.phase is RoutePhase.ROTATE


def test_route_stops_when_done():
    follower = RouteFollower([Point2D(0.0, 0.0)], 0.0, RouteConfig())
    pose = Pose2D(0.0, 0.0, 0.0)

    follower.step(pose)  # 경로점 도착 → FINAL_ROTATE
    follower.step(pose)  # yaw 일치 → DONE

    assert follower.done
    assert follower.step(pose) == (0.0, 0.0)
