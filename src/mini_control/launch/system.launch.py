"""
프로젝트 전체 ROS2 노드를 한 번에 실행한다.

실행:
- mini_vision / webcam_detector
- mini_vision / webcam_localizer
- mini_vision / amr_detector
- mini_control / approach
- mini_control / mission_manager

위치 추정(localization) / Nav2 / RViz는 여기서 실행하지 않는다.
먼저 따로 실행해 두고, RViz의 "2D Pose Estimate"로 초기 위치를 지정해야
amcl_pose가 나오고 approach가 goal을 보낼 수 있다.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # =========================================================
    # 1. 패키지 설치 경로
    # =========================================================
    mini_vision_share = get_package_share_directory(
        'mini_vision'
    )

    mini_control_share = get_package_share_directory(
        'mini_control'
    )

    # =========================================================
    # 2. 설정 / 모델 경로
    # =========================================================
    camera_mapping_file = os.path.join(
        mini_vision_share,
        'config',
        'camera_mapping.yaml'
    )

    control_params_file = os.path.join(
        mini_control_share,
        'config',
        'params.yaml'
    )

    webcam_model_default = os.path.join(
        mini_vision_share,
        'models',
        'webcam_best.pt'
    )

    amr_model_default = os.path.join(
        mini_vision_share,
        'models',
        'amr_best.pt'
    )

    # =========================================================
    # 3. Launch 인자
    # =========================================================
    webcam_model_arg = DeclareLaunchArgument(
        'webcam_model_path',
        default_value=webcam_model_default
    )

    amr_model_arg = DeclareLaunchArgument(
        'amr_model_path',
        default_value=amr_model_default
    )

    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='4'
    )

    amr_camera_topic_arg = DeclareLaunchArgument(
        'amr_camera_topic',
        default_value='/robot2/oakd/rgb/image_raw/compressed'
    )

    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/robot2/cmd_vel'
    )

    target_distance_arg = DeclareLaunchArgument(
        'target_distance',
        default_value='0.8'
    )

    min_distance_arg = DeclareLaunchArgument(
        'min_distance',
        default_value='0.5'
    )

    linear_gain_arg = DeclareLaunchArgument(
        'linear_gain',
        default_value='0.5'
    )

    angular_gain_arg = DeclareLaunchArgument(
        'angular_gain',
        default_value='1.0'
    )

    max_linear_speed_arg = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.31'
    )

    max_angular_speed_arg = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='1.0'
    )

    detection_timeout_arg = DeclareLaunchArgument(
        'detection_timeout',
        default_value='0.5'
    )

    # =========================================================
    # 4. 추종 방식
    # =========================================================
    # cmd_vel: 속도 직접 제어 (기존), nav: Nav2 goal 로 추종
    follow_mode_arg = DeclareLaunchArgument(
        'follow_mode',
        default_value='cmd_vel'
    )

    # =========================================================
    # 5. 고정 웹캠 YOLO 감지
    # =========================================================
    webcam_detector_node = Node(
        package='mini_vision',
        executable='webcam_detector',
        name='webcam_detector',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration(
                    'webcam_model_path'
                ),
                'camera_index': ParameterValue(
                    LaunchConfiguration('camera_index'),
                    value_type=int
                ),
            }
        ]
    )

    # =========================================================
    # 6. 웹캠 픽셀 → map 위치 변환
    # =========================================================
    webcam_localizer_node = Node(
        package='mini_vision',
        executable='webcam_localizer',
        name='webcam_localizer',
        output='screen',
        parameters=[
            camera_mapping_file
        ]
    )

    # =========================================================
    # 7. TurtleBot4 AMR 카메라 YOLO 감지
    # =========================================================
    amr_detector_node = Node(
        package='mini_vision',
        executable='amr_detector',
        name='amr_detector',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration(
                    'amr_model_path'
                ),

                'camera_topic': LaunchConfiguration(
                    'amr_camera_topic'
                ),

                'depth_topic': '/robot2/oakd/stereo/image_raw/compressedDepth',
            }
        ]
    )

    # =========================================================
    # 8. Nav2 접근 노드
    #
    # params.yaml의 /robot2 설정을 approach에 전달한다.
    # =========================================================
    approach_node = Node(
        package='mini_control',
        executable='approach',
        name='approach',
        output='screen',
        parameters=[
            control_params_file
        ]
    )

    # =========================================================
    # 9. 추종 / 상태 전환 통합 노드
    # =========================================================
    mission_manager_node = Node(
        package='mini_control',
        executable='mission_manager',
        name='mission_manager',
        output='screen',
        parameters=[
            control_params_file,
            {
                'cmd_vel_topic': LaunchConfiguration(
                    'cmd_vel_topic'
                ),

                'target_distance': ParameterValue(
                    LaunchConfiguration('target_distance'),
                    value_type=float
                ),

                'min_distance': ParameterValue(
                    LaunchConfiguration('min_distance'),
                    value_type=float
                ),

                'linear_gain': ParameterValue(
                    LaunchConfiguration('linear_gain'),
                    value_type=float
                ),

                'angular_gain': ParameterValue(
                    LaunchConfiguration('angular_gain'),
                    value_type=float
                ),

                'max_linear_speed': ParameterValue(
                    LaunchConfiguration('max_linear_speed'),
                    value_type=float
                ),

                'max_angular_speed': ParameterValue(
                    LaunchConfiguration('max_angular_speed'),
                    value_type=float
                ),

                'detection_timeout': ParameterValue(
                    LaunchConfiguration('detection_timeout'),
                    value_type=float
                ),

                'follow_mode': LaunchConfiguration(
                    'follow_mode'
                ),
            }
        ]
    )

    # =========================================================
    # 10. 전체 노드 반환
    # =========================================================
    return LaunchDescription([
        webcam_model_arg,
        amr_model_arg,
        camera_index_arg,
        amr_camera_topic_arg,
        cmd_vel_topic_arg,

        target_distance_arg,
        min_distance_arg,
        linear_gain_arg,
        angular_gain_arg,
        max_linear_speed_arg,
        max_angular_speed_arg,
        detection_timeout_arg,
        follow_mode_arg,

        webcam_detector_node,
        webcam_localizer_node,
        amr_detector_node,
        approach_node,
        mission_manager_node,
    ])
