미니카를 감지하고 추적하는 프로젝트

> PR 테스트용 변경입니다.

## 구성

| 패키지 | 노드 | 역할 |
|---|---|---|
| mini_vision | `webcam_detector` | 고정 웹캠 YOLO 감지 → `/webcam/detections` |
| mini_vision | `webcam_localizer` | 웹캠 픽셀 → map 좌표 → `/target/map_position` |
| mini_vision | `amr_detector` | TurtleBot4 OAK-D YOLO 감지 + depth 거리 → `/amr/detections` |
| mini_control | `approach` | 목표 좌표로 Nav2 이동 → `/approach/status` |
| mini_control | `mission_manager` | 상태 전환 및 자동차 추종 → `/robot2/cmd_vel` |

상태 흐름: `SEARCHING → APPROACHING → HANDOVER → FOLLOWING ⇄ LOST`

- 웹캠이 자동차를 찾으면 approach로 접근한다 (APPROACHING).
- 도착하거나, 이동 중 AMR 카메라가 자동차를 보면 Nav2를 취소하고 HANDOVER로 넘어간다.
- AMR 카메라에서 `handover_detection_count`번 연속 감지되면 FOLLOWING으로 추종을 시작한다.
- 추종 거리는 `amr_detector`가 `/amr/detections`의 `results[0].pose.pose.position.z`에 넣어 준 depth 거리를 쓴다.

## 통합 테스트

기준: `~/minicar_ws`, ROS 2 Jazzy, 로봇 namespace `/robot2`

### 1. 빌드

```bash
cd ~/minicar_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select mini_vision mini_control
source install/setup.bash
```

새 터미널을 열 때마다 `source ~/minicar_ws/install/setup.bash`를 먼저 실행한다.

### 2. TurtleBot4 위치 추정 + Nav2 (이미 실행 중이면 생략)

`system.launch.py`는 Nav2를 실행하지 않는다.
approach가 `/robot2/navigate_to_pose`, `/robot2/amcl_pose`를 사용하므로 먼저 띄운다.

```bash
# 터미널 2: 위치 추정
ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot2 map:=$HOME/minicar_ws/arena_map.yaml

# 터미널 3: Nav2
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot2

# 터미널 4: RViz에서 "2D Pose Estimate"로 초기 위치 지정
ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/robot2
```

필요한 토픽 확인:

```bash
ros2 topic hz /robot2/oakd/rgb/image_raw/compressed
ros2 topic hz /robot2/oakd/stereo/image_raw/compressedDepth
ros2 topic echo --once /robot2/amcl_pose
```

### 3. 전체 시스템 실행

```bash
ros2 launch mini_control system.launch.py
```

인자 변경 예:

```bash
ros2 launch mini_control system.launch.py camera_index:=0 target_distance:=0.8 max_linear_speed:=0.2
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `camera_index` | `2` | 고정 웹캠 번호 (`ls /dev/video*`로 확인) |
| `amr_camera_topic` | `/robot2/oakd/rgb/image_raw/compressed` | AMR 카메라 토픽 (depth는 `stereo/image_raw/compressedDepth`, 둘 다 704x704) |
| `cmd_vel_topic` | `/robot2/cmd_vel` | 추종 속도 명령 토픽 |
| `target_distance` | `0.8` | 추종 시 유지할 거리 [m] |
| `min_distance` | `0.5` | 이 거리 이하면 전진 정지 (회전은 유지) [m] |
| `max_linear_speed` | `0.31` | 최대 전진 속도 [m/s] |
| `max_angular_speed` | `1.0` | 최대 회전 속도 [rad/s] |
| `detection_timeout` | `0.5` | 감지 결과 유효 시간 [s] |
| `handover_detection_count` | `3` | 추종 전환에 필요한 연속 감지 횟수 |

- `amr_detector`는 감지 창을 띄우므로(`show_window: True`) 디스플레이가 있는 환경에서 실행한다.
- `mission_manager`의 `image_width`(기본 640)가 AMR 영상의 가로 폭(704)과 같아야 회전 계산이 맞다.

### 4. 동작 확인

상태 전환은 launch 터미널 로그에 출력된다 (`SEARCHING -> APPROACHING` 등).

```bash
ros2 topic echo /target/map_position   # 웹캠이 계산한 자동차 위치
ros2 topic echo /approach/status       # MOVING / ARRIVED / FAILED / CANCELED
ros2 topic echo /amr/detections        # AMR 감지 결과 + 거리(results[0].pose.pose.position.z)
ros2 topic echo /robot2/cmd_vel        # 추종 속도 명령
```

### 단계별 테스트: 웹캠 없이 접근만 확인

목표 좌표를 직접 보낸다 (맵 안의 빈 곳으로 지정).

```bash
ros2 topic pub --once /target/map_position geometry_msgs/msg/PointStamped \
  "{header: {frame_id: map}, point: {x: 1.0, y: 0.5, z: 0.0}}"
```

이동 중 AMR 카메라에 자동차가 보이면 approach가 취소되고 HANDOVER로 넘어가야 정상이다.

### 정지

launch 터미널에서 `Ctrl+C`. `mission_manager`가 종료 시 정지 명령을 보낸다.
