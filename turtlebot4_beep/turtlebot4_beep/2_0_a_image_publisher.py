import rclpy
from rclpy.node import Node
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')

        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        self.bridge = CvBridge()

        image_path = Path.home() / 'minicar_ws' / 'test.jpg'

        self.image = cv2.imread(str(image_path))

        if self.image is None:
            self.get_logger().error('이미지를 불러오지 못했습니다.')
            return

        # 10 FPS
        self.timer = self.create_timer(
            0.1,
            self.publish_image
        )

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(
            self.image,
            encoding='bgr8'
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ImagePublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
