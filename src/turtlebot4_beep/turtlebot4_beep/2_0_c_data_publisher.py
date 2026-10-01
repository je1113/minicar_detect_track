import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class TextPublisher(Node):

    def __init__(self):
        super().__init__('text_publisher')

        self.publisher = self.create_publisher(
            String,
            '/text_topic',
            10
        )

        self.timer = self.create_timer(
            1.0,
            self.publish_text
        )

        self.count = 0

    def publish_text(self):

        msg = String()

        msg.data = f'Hello ROS 2! count={self.count}'

        self.publisher.publish(msg)

        self.get_logger().info(
            f'Published: {msg.data}'
        )

        self.count += 1


def main(args=None):

    rclpy.init(args=args)

    node = TextPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
