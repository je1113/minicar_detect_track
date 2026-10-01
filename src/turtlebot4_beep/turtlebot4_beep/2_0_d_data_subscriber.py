import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class TextSubscriber(Node):

    def __init__(self):
        super().__init__('text_subscriber')

        self.subscription = self.create_subscription(
            String,
            '/text_topic',
            self.text_callback,
            10
        )

        self.get_logger().info('Text subscriber started!')

    def text_callback(self, msg):
        self.get_logger().info(
            f'Received: {msg.data}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = TextSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
