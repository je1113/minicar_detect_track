from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'mini_vision'

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
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'models'),
            glob('models/*.pt'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maymayko',
    maintainer_email='maymayko9559@gmail.com',
    description='Webcam and AMR camera car detection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'webcam_detector = mini_vision.webcam_detector:main',
            'webcam_localizer = mini_vision.webcam_localizer:main',
            'amr_detector = mini_vision.amr_detector:main',
            'aruco_calibrator = mini_vision.aruco_calibrator:main',
        ],
    },
)