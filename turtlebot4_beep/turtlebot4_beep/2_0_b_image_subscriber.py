import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        self.get_logger().info(
            'Image subscriber started!'
        )

    def image_callback(self, msg):

        # ROS Image → OpenCV image
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        self.get_logger().info(
            f'image received: {frame.shape}'
        )

        # 화면에 표시
        cv2.imshow('ROS Image', frame)
        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
