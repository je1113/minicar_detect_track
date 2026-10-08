"""approach.py 의 cmd_vel fallback 경로 보간/추종 단위 테스트."""
import math

from mini_control.approach import (
    distance_to_segment,
    normalize_angle,
    Point2D,
    Pose2D,
    route_start_index,
    RouteConfig,
    RouteFollower,
    RoutePhase,
    smooth_path,
)
import pytest

DT = 0.1
WAYPOINTS = [Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(1.79, 1.77)]
GOAL = Pose2D(0.64, 1.84, -2.27)
ROUTE = WAYPOINTS + [GOAL.position]


def simulate(follower, pose, max_steps=5000):
    """속도 명령을 그대로 적분하는 이상적인 diff-drive 로 이동시킨다."""
    commands = []
    poses = [pose]
    for _ in range(max_steps):
        v = follower.step(pose)
        commands.append(v)
        if follower.done:
            return pose, commands, poses
        yaw = pose.yaw + v.angular * DT
        pose = Pose2D(pose.x + v.linear * math.cos(yaw) * DT,
                      pose.y + v.linear * math.sin(yaw) * DT,
                      normalize_angle(yaw))
        poses.append(pose)
    raise AssertionError('route did not finish')


def follow_route(start_pose, cfg=RouteConfig(), corner_radius=0.3):
    start = route_start_index(start_pose.position, ROUTE)
    path = smooth_path([start_pose.position] + ROUTE[start:], corner_radius)
    return simulate(RouteFollower(path, GOAL.yaw, cfg, DT), start_pose)


def distance_to_route(p):
    return min(distance_to_segment(p, a, b) for a, b in zip(ROUTE, ROUTE[1:]))


@pytest.mark.parametrize('angle, expected', [
    (0.0, 0.0), (math.pi / 2, math.pi / 2), (3 * math.pi / 2, -math.pi / 2),
    (-3 * math.pi / 2, math.pi / 2), (2 * math.pi, 0.0),
])
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize('robot, expected', [
    (Point2D(0.0, 0.0), 1),    # 출발점: goal 이 P1 보다 가깝지만 P1 로 간다
    (Point2D(1.0, 0.1), 1),    # 첫 직진 도중
    (Point2D(1.95, 1.0), 2),   # 두 번째 직진 도중
    (Point2D(1.0, 1.85), 3),   # 마지막 직진 도중
])
def test_route_start_follows_nearest_segment(robot, expected):
    assert route_start_index(robot, ROUTE) == expected


def test_smooth_path_is_dense_and_keeps_endpoints():
    path = smooth_path(ROUTE, 0.3, step=0.02)

    assert path[0] == ROUTE[0]
    assert path[-1] == ROUTE[-1]
    gaps = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:])]
    assert max(gaps) <= 0.02 + 1e-9


def test_smooth_path_rounds_corner_within_radius():
    # 90도 코너: 원호는 꼭짓점에서 r(√2 - 1) 만큼 안쪽을 지난다
    r = 0.3
    path = smooth_path([Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(2.0, 2.0)], r)

    corner = Point2D(2.0, 0.0)
    closest = min(math.hypot(p.x - corner.x, p.y - corner.y) for p in path)
    assert closest == pytest.approx(r * (math.sqrt(2) - 1), abs=0.01)
    assert Point2D(2.0, 0.0) not in path


def test_smooth_path_without_radius_keeps_corners():
    path = smooth_path(ROUTE, 0.0)

    for corner in ROUTE:
        assert corner in path


def test_route_reaches_goal_turning_left_only():
    cfg = RouteConfig()

    final, commands, poses = follow_route(Pose2D(0.0, 0.0, 0.0), cfg)

    assert math.hypot(final.x - GOAL.x, final.y - GOAL.y) < cfg.position_tolerance
    assert abs(normalize_angle(final.yaw - GOAL.yaw)) < cfg.yaw_tolerance
    assert all(c.angular > -0.05 for c in commands)
    # 모서리를 둥글게 돌아도 원래 경로에서 크게 벗어나지 않는다
    assert max(distance_to_route(p.position) for p in poses) < 0.2


def test_route_drives_through_corners_without_stopping():
    _, commands, _ = follow_route(Pose2D(0.0, 0.0, 0.0))

    driving = [c for c in commands if c.linear > 0.0]
    first = commands.index(driving[0])
    last = commands.index(driving[-1])
    # 주행을 시작한 뒤 goal 까지 멈추지 않는다 (모서리에서 제자리 회전 없음)
    assert all(c.linear > 0.0 for c in commands[first:last + 1])


def test_velocity_changes_within_accel_limits():
    cfg = RouteConfig()

    _, commands, _ = follow_route(Pose2D(0.0, 0.0, 0.0), cfg)

    for a, b in zip(commands, commands[1:]):
        assert abs(b.linear - a.linear) <= cfg.linear_accel * DT + 1e-9
    max_dw = max(abs(b.angular - a.angular) for a, b in zip(commands, commands[1:]))
    # 제자리 회전 종료 시 min_angular_speed 에서 0 으로 멈추는 것만 허용
    assert max_dw <= max(cfg.angular_accel * DT, cfg.min_angular_speed) + 1e-9


def test_route_rotates_in_place_when_facing_away():
    path = smooth_path([Point2D(0.0, 0.0), Point2D(0.0, 1.0)], 0.3)
    follower = RouteFollower(path, 0.0, RouteConfig(), DT)

    v = follower.step(Pose2D(0.0, 0.0, 0.0))

    assert v.linear == 0.0
    assert v.angular > 0.0
    assert follower.phase is RoutePhase.ROTATE


def test_route_resumes_mid_way():
    final, _, _ = follow_route(Pose2D(1.95, 1.0, math.pi / 2))

    assert math.hypot(final.x - GOAL.x, final.y - GOAL.y) < RouteConfig().position_tolerance


def test_route_accepts_small_overshoot_without_turning_back():
    path = smooth_path([Point2D(0.0, 0.0), Point2D(1.0, 0.0)], 0.3)
    follower = RouteFollower(path, 0.0, RouteConfig(), DT)
    follower.phase = RoutePhase.TRACK

    follower.step(Pose2D(1.05, 0.04, 0.5))  # 끝점을 5cm 지나침

    assert follower.phase is RoutePhase.FINAL_ROTATE


def test_route_stops_when_done():
    follower = RouteFollower([Point2D(0.0, 0.0)], 0.0, RouteConfig(), DT)
    pose = Pose2D(0.0, 0.0, 0.0)

    follower.step(pose)  # goal 도착 → FINAL_ROTATE
    follower.step(pose)  # yaw 일치 → DONE

    assert follower.done
    assert follower.step(pose) == (0.0, 0.0)
