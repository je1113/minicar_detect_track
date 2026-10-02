"""자동차 상대 위치와 거리를 이용한 추종 속도 계산."""


def stop_velocity():
    """정지 속도를 반환한다."""
    return 0.0, 0.0


def _clamp(value, minimum, maximum):
    """값을 minimum ~ maximum 범위로 제한한다."""
    return max(minimum, min(value, maximum))


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
    자동차의 상대 위치와 거리로 전진/회전 속도를 계산한다.

    반환값:
        (linear_x, angular_z)
    """

    # 자동차를 감지하지 못했으면 정지
    if not target_detected:
        return stop_velocity()

    # 영상 정보가 유효하지 않으면 정지
    if target_center_x is None:
        return stop_velocity()

    if image_width is None or image_width <= 0:
        return stop_velocity()

    # 거리 정보가 유효하지 않으면 정지
    if distance is None or distance <= 0:
        return stop_velocity()

    # ---------------------------------------------------------
    # 1. 영상 중심과 자동차 중심의 차이 계산
    # ---------------------------------------------------------
    image_center_x = image_width / 2.0

    horizontal_error = (
        target_center_x - image_center_x
    ) / image_center_x

    # ---------------------------------------------------------
    # 2. 회전 속도 계산
    #
    # 영상 오른쪽: horizontal_error > 0
    # 영상 왼쪽:   horizontal_error < 0
    #
    # ROS angular.z 기준에 맞춰 부호를 반대로 적용한다.
    # ---------------------------------------------------------
    angular_z = -angular_gain * horizontal_error

    angular_z = _clamp(
        angular_z,
        -max_angular_speed,
        max_angular_speed,
    )

    # ---------------------------------------------------------
    # 3. 거리 차이 계산
    # ---------------------------------------------------------
    distance_error = distance - target_distance

    linear_x = linear_gain * distance_error

    # ---------------------------------------------------------
    # 4. 너무 가까우면 전진 정지
    # ---------------------------------------------------------
    if distance <= min_distance:
        linear_x = 0.0

    # 뒤로 가지 않도록 제한
    if linear_x < 0.0:
        linear_x = 0.0

    # ---------------------------------------------------------
    # 5. 최대 전진 속도 제한
    # ---------------------------------------------------------
    linear_x = _clamp(
        linear_x,
        0.0,
        max_linear_speed,
    )

    return linear_x, angular_z
