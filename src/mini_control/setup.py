from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'mini_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'maps'),
            glob('maps/*'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    extras_require={
        'test': [
            'pytest',
        ],
    },
    maintainer='maymayko',
    maintainer_email='maymayko9559@gmail.com',
    description='AMR approach and car following control',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_manager = mini_control.mission_manager:main',
            'approach = mini_control.approach:main',
        ],
    },
)
