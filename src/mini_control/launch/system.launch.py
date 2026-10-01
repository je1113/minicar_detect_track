# 역할:
# 프로젝트 노드와 설정을 한 번에 실행한다.

# 작성할 내용:
# 1. mini_vision과 mini_control의 설치 경로를 찾는다.
# 2. 모델 경로, 웹캠 번호, AMR 카메라 토픽 등을 인자로 받는다.
# 3. webcam_detector 노드를 실행한다.
# 4. webcam_localizer 노드에 보정 설정을 전달한다.
# 5. amr_detector 노드에 AMR 카메라 토픽을 전달한다.
# 6. mission_manager 노드에 제어 설정을 전달한다.
# 7. 실제 로봇의 namespace와 토픽에 맞춰 연결한다.

# Nav2:
# 로봇 bringup에서 이미 실행 중이면 중복 실행하지 않는다.
# 이 launch에서 실행할지는 팀의 실행 방식에 맞춰 정한다.

# generate_launch_description():
# 실행할 노드와 인자를 LaunchDescription으로 반환한다.