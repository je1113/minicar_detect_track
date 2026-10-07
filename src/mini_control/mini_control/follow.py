"""
역할:
자동차의 상대 방향과 거리로 추종 속도를 계산한다.

이 파일은 별도 실행 ROS2 노드로 만들지 않는다.
계산 결과를 mission_manager에 반환한다.
"""

import math

from geometry_msgs.msg import TwistStamped


def _clamp(value, minimum, maximum):
    """
    값을 최소값과 최대값 사이로 제한한다.
    """
    return max(minimum, min(value, maximum))


def stop_velocity():
    """
    정지 속도를 반환한다.

    반환값:
        linear_x = 0.0
        angular_z = 0.0
    """
    return 0.0, 0.0


def compute_velocity(
    target_detected,
    target_center_x,
    image_width,
    distance,
    target_distance,
    min_distance,
    linear_gain,
    angular_gain,
    max_linear_speed,
    max_angular_speed,
):
    """
    자동차의 상대 방향과 거리로
    전진 속도와 회전 속도를 계산한다.

    반환값:
        (linear_x, angular_z)
    """

    # ========================================================
    # 1. 자동차 감지 여부 확인
    # ========================================================
    if not target_detected:
        return stop_velocity()

    # ========================================================
    # 2. 영상 정보 확인
    # ========================================================
    if target_center_x is None:
        return stop_velocity()

    if image_width is None or image_width <= 0:
        return stop_velocity()

    # ========================================================
    # 3. 거리 정보 확인
    # ========================================================
    if distance is None or distance <= 0:
        return stop_velocity()

    # ========================================================
    # 4. 자동차 중심과 영상 중심의 차이 계산
    # ========================================================
    image_center_x = image_width / 2.0

    horizontal_error = (
        target_center_x - image_center_x
    ) / image_center_x

    # ========================================================
    # 5. 회전 속도 계산
    #
    # 자동차가 화면 오른쪽:
    # horizontal_error > 0
    #
    # 자동차가 화면 왼쪽:
    # horizontal_error < 0
    # ========================================================
    angular_z = -angular_gain * horizontal_error

    angular_z = _clamp(
        angular_z,
        -max_angular_speed,
        max_angular_speed,
    )

    # ========================================================
    # 6. 거리 오차로 전진 속도 계산
    # ========================================================
    distance_error = distance - target_distance

    linear_x = linear_gain * distance_error

    # ========================================================
    # 7. 너무 가까우면 전진 정지
    # ========================================================
    if distance <= min_distance:
        linear_x = 0.0

    # 후진하지 않음
    if linear_x < 0.0:
        linear_x = 0.0

    # ========================================================
    # 8. 최대 전진 속도 제한
    # ========================================================
    linear_x = _clamp(
        linear_x,
        0.0,
        max_linear_speed,
    )

    # ========================================================
    # 9. mission_manager에 계산 결과 반환
    # ========================================================
    return linear_x, angular_z


def make_cmd_vel_message(
    linear_x,
    angular_z,
    stamp=None,
):
    """
    계산된 속도를 TurtleBot4의 ROS2 cmd_vel 메시지 형식으로 만든다.

    실제 /robot2/cmd_vel 타입:
        geometry_msgs/msg/TwistStamped

    이 함수는 publish하지 않는다.
    생성된 메시지를 mission_manager에 반환한다.
    """

    msg = TwistStamped()

    if stamp is not None:
        msg.header.stamp = stamp

    msg.twist.linear.x = float(linear_x)
    msg.twist.linear.y = 0.0
    msg.twist.linear.z = 0.0

    msg.twist.angular.x = 0.0
    msg.twist.angular.y = 0.0
    msg.twist.angular.z = float(angular_z)

    return msg


def detection_to_map_point(
    center_x,
    distance,
    fx,
    cx,
    robot_x,
    robot_y,
    robot_yaw,
):
    """
    AMR 카메라 감지(박스 중심 픽셀 x, 거리)를 map 좌표로 바꾼다.

    3_3_d_depth_to_nav_goal_ts 와 같은 핀홀 역투영을 쓴다.
        X = (u - cx) * Z / fx    카메라 오른쪽이 +
        Z = distance             카메라 앞쪽이 +
    세로(Y)는 지도에 쓰지 않는다.

    카메라 좌표를 로봇 기준(앞쪽 f = Z, 왼쪽 l = -X)으로 보고,
    amcl_pose 의 위치와 방향으로 map 에 옮긴다.
        map_x = robot_x + f * cos(yaw) - l * sin(yaw)
        map_y = robot_y + f * sin(yaw) + l * cos(yaw)

    카메라가 로봇 중심에서 떨어진 오프셋은 무시한다.

    반환값:
        (map_x, map_y) [m]
    """
    if fx <= 0.0:
        raise ValueError(f'fx 는 0보다 커야 합니다: {fx}')

    lateral_x = (center_x - cx) * distance / fx

    forward = distance
    left = -lateral_x

    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)

    return (
        robot_x + forward * cos_yaw - left * sin_yaw,
        robot_y + forward * sin_yaw + left * cos_yaw,
    )


def should_publish_follow_target(
    last_point,
    new_point,
    elapsed,
    min_period,
    min_move,
    refresh_period,
):
    """
    FOLLOWING 중 새 자동차 좌표를 approach 로 보낼지 정한다.

    approach 는 좌표를 받을 때마다 Nav2 goal 을 보낼 수 있다.
    goal 이 자주 바뀌면 로봇이 끊기므로 보내는 쪽에서도 횟수를 줄인다.
        - 처음이면 보낸다.
        - 마지막으로 보낸 뒤 min_period 가 지나지 않았으면 보내지 않는다.
        - 자동차가 min_move 이상 움직였거나 refresh_period 가 지났으면 보낸다.
          (움직이지 않아도 가끔 다시 보내 goal 이 실패했을 때 재시도한다.)

    last_point, new_point 는 (x, y), elapsed 는 마지막으로 보낸 뒤 경과 시간 [s].
    """
    if last_point is None:
        return True

    if elapsed < min_period:
        return False

    moved = math.hypot(
        new_point[0] - last_point[0],
        new_point[1] - last_point[1],
    )

    return moved >= min_move or elapsed >= refresh_period
