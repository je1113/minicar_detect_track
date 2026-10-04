"""
역할:
자동차의 상대 방향과 거리로 추종 속도를 계산한다.

이 파일은 별도 실행 ROS2 노드로 만들지 않는다.
계산 결과를 mission_manager에 반환한다.
"""

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