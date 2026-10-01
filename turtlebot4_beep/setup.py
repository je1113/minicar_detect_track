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
            '2_0_d_data_subscriber = turtlebot4_beep.2_0_d_data_subscriber:main'

        ],
    },
)
