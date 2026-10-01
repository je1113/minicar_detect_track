"""TurtleBot4 비프음 노드 (cmd_audio 발행)

  python3 audio_beep.py --ros-args -p robot_ns:=robot2
  python3 audio_beep.py --ros-args -p robot_ns:=robot2 -p repeat:=3
"""
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from irobot_create_msgs.msg import AudioNote, AudioNoteVector


class AudioBeep(Node):
    def __init__(self):
        super().__init__('audio_beep')
        self.declare_parameter('robot_ns', 'robot2')
        self.declare_parameter('repeat', 1)        # 몇 번 재생할지
        self.declare_parameter('period', 1.0)      # 재생 간격(초)

        ns = self.get_parameter('robot_ns').value.strip('/')
        self.repeat = self.get_parameter('repeat').value
        period = self.get_parameter('period').value

        topic = f'/{ns}/cmd_audio' if ns else '/cmd_audio'
        self.pub = self.create_publisher(AudioNoteVector, topic, 10)
        self.count = 0
        self.done = False
        self.timer = self.create_timer(period, self.publish_audio)
        self.get_logger().info(f'발행 토픽: {topic}')

    def build_msg(self):
        msg = AudioNoteVector()
        msg.header.frame_id = ''
        msg.append = False
        for freq in [880, 440, 880, 440]:
            note = AudioNote()
            note.frequency = freq
            note.max_runtime = Duration(sec=0, nanosec=300_000_000)
            msg.notes.append(note)
        return msg

    def publish_audio(self):
        # 로봇이 구독하기 전에 보내면 메시지가 유실되므로 연결될 때까지 대기
        if self.pub.get_subscription_count() == 0:
            self.get_logger().info('로봇 연결 대기 중...')
            return

        self.pub.publish(self.build_msg())
        self.count += 1
        self.get_logger().info(f'비프음 발행 ({self.count}/{self.repeat})')

        if self.count >= self.repeat:
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = AudioBeep()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()