from setuptools import find_packages, setup

package_name = 'turtlebot4_beep'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hv-02',
    maintainer_email='jje320594@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'beep_node = turtlebot4_beep.beep_node:main',
            '2_0_a_image_publisher = turtlebot4_beep.2_0_a_image_publisher:main',
            '2_0_b_image_subscriber = turtlebot4_beep.2_0_b_image_subscriber:main',
            '2_0_c_data_publisher = turtlebot4_beep.2_0_c_data_publisher:main',
            '2_0_d_data_subscriber = turtlebot4_beep.2_0_d_data_subscriber:main',
            '2_1_d_capture_image = turtlebot4_beep.2_1_d_capture_image:main',
            'depth_checker = turtlebot4_beep.depth_checker:main',
            '2_1_e_capture_comp_image = turtlebot4_beep.2_1_e_capture_comp_image:main',
            '3_1_a_nav_to_pose = turtlebot4_beep.3_1_a_nav_to_pose:main',
            '3_1_b_nav_through_poses = turtlebot4_beep.3_1_b_nav_through_poses:main',
            '3_1_c_follow_waypoints = turtlebot4_beep.3_1_c_follow_waypoints:main',
            '3_1_d_create_path = turtlebot4_beep.3_1_d_create_path:main',
            '3_1_e_mail_delivery = turtlebot4_beep.3_1_e_mail_delivery:main',
            '3_1_f_patrol_loop = turtlebot4_beep.3_1_f_patrol_loop:main',
        ],
    },
)
