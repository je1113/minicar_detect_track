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

상태 흐름: `SEARCHING → APPROACHING → FOLLOWING ⇄ LOST` (LOST에서 못 찾으면 `APPROACHING`/`SEARCHING`)

- 웹캠이 자동차를 찾으면 자동차 map 좌표를 approach로 넘긴다 (APPROACHING). 자동차가 움직이면 좌표를 계속 넘긴다.
- approach는 **자동차 위치 자체를 Nav2 goal**로 보내고, feedback의 경로상 남은 거리가 `standoff_distance`(1m) 아래가 되면 goal을 취소하고 `ARRIVED`를 알린다. 로봇 위치는 TF(`map → base_link`)로 구한다.
- 웹캠 좌표에 한 번 도착한 뒤 AMR 카메라에 자동차가 보이면 FOLLOWING으로 간다 (도착 전에는 보여도 접근을 계속한다).
- FOLLOWING: bbox 가로 위치(화각 69°) + depth 거리로 카메라 frame 점을 만들고, TF로 map 좌표로 바꿔 approach로 넘긴다 → 추종 중에도 Nav2가 장애물을 피한다. standoff 안이면 제자리 회전으로 자동차를 화면 가운데에 둔다.
- `lost_timeout`(3초) 동안 못 보면 approach를 취소하고 마지막으로 본 쪽으로 360도 회전한다 (LOST). 보이면 FOLLOWING, 한 바퀴 돌아도 없으면 웹캠 좌표로 다시 접근한다.
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

### 2. TurtleBot4 위치 추정 + Nav2 + RViz

`system.launch.py`가 위치 추정(`localization.launch.py`), Nav2(`nav2.launch.py`), RViz(`view_navigation.launch.py`)를 같이 실행한다.
맵은 `mini_control/maps/arena_map.yaml`, Nav2 파라미터는 `mini_control/config/nav2.yaml`(TurtleBot4 기본값에서 global costmap `inflation_radius`만 0.05로 변경)을 쓴다.

launch 후 RViz에서 "2D Pose Estimate"로 초기 위치를 지정해야 `map → base_link` TF가 나오고 approach가 동작한다.

비전/제어 노드만 다시 띄울 때 초기 위치를 매번 다시 잡지 않으려면, 위치 추정·Nav2·RViz를 따로 띄워 두고 system.launch에서는 끈다.

```bash
# 각각 별도 터미널에서: 위치 추정 / Nav2 / RViz
ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot2 map:=$HOME/minicar_ws/src/mini_control/maps/arena_map.yaml
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot2 params_file:=$HOME/minicar_ws/src/mini_control/config/nav2.yaml
ros2 launch turtlebot4_viz view_navigation.launch.py namespace:=/robot2

# 그다음 터미널: 나머지 노드
ros2 launch mini_control system.launch.py use_localization:=false use_nav2:=false use_rviz:=false
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
ros2 launch mini_control system.launch.py camera_index:=0 standoff_distance:=1.0
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `namespace` | `/robot2` | 위치 추정 / Nav2 / RViz에 쓸 로봇 namespace |
| `use_localization` | `true` | 위치 추정(AMCL + map_server) 실행 여부 |
| `use_nav2` | `true` | Nav2 실행 여부 |
| `use_rviz` | `true` | RViz 실행 여부 |
| `map` | `mini_control/maps/arena_map.yaml` | 위치 추정에 쓸 맵 |
| `nav2_params_file` | `mini_control/config/nav2.yaml` | Nav2 파라미터 파일 |
| `camera_index` | `2` | 고정 웹캠 번호 (`ls /dev/video*`로 확인) |
| `amr_camera_topic` | `/robot2/oakd/rgb/image_raw/compressed` | AMR 카메라 토픽 (depth는 `stereo/image_raw/compressedDepth`, 둘 다 704x704) |
| `cmd_vel_topic` | `/robot2/cmd_vel` | 추종 속도 명령 토픽 |
| `standoff_distance` | `1.0` | 경로상 자동차까지 이 거리가 남으면 Nav2 goal 취소 [m] |
| `angular_gain` | `1.2` | 도착 후 제자리 회전 gain |
| `max_angular_speed` | `1.0` | 최대 회전 속도 [rad/s] |
| `detection_timeout` | `0.5` | 감지 결과 유효 시간 [s] |

- `amr_detector`는 감지 창을 띄우므로(`show_window: True`) 디스플레이가 있는 환경에서 실행한다.
- `mission_manager`의 `image_width`(기본 704)가 AMR 영상의 가로 폭과 같아야 방향각 계산이 맞다.
- approach / mission_manager는 TF를 쓰므로 launch에서 `/tf`, `/tf_static`을 `<namespace>/tf`로 리매핑한다.

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

경로상 1m 남으면 approach가 goal을 취소하고 `ARRIVED`를 알려야 정상이다 (`ros2 topic echo /approach/status`).
그 뒤 AMR 카메라에 자동차가 보이면 `APPROACHING -> FOLLOWING`으로 넘어간다.

### 정지

launch 터미널에서 `Ctrl+C`. `mission_manager`가 종료 시 정지 명령을 보낸다.
