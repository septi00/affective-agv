import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from affective_msgs.msg import ProxemicCircleArray


class CircleLogger(Node):
    """Append every published proxemic circle to a CSV for offline analysis.

    One row per circle per message. Columns are analysis-ready so study runs
    and thesis figures can be plotted directly from the file. For a full,
    replayable capture of every topic, `ros2 bag record` is the better tool
    and needs no node; this logger exists specifically to produce tidy CSV.
    """

    def __init__(self):
        super().__init__('circle_logger')

        self.declare_parameter('topic', '/proxemics/circles')
        self.declare_parameter(
            'output_dir', os.path.expanduser('~/proxemics_logs'))
        self.declare_parameter('flush_every', 20)

        topic = self.get_parameter('topic').value
        out_dir = self.get_parameter('output_dir').value
        self.flush_every = int(self.get_parameter('flush_every').value)

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(out_dir, f'circles_{stamp}.csv')

        self.file = open(self.path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'stamp_sec', 'stamp_nanosec', 'frame_id',
            'person_id', 'x', 'y', 'radius', 'peak_cost'])
        self._rows_since_flush = 0

        self.sub = self.create_subscription(
            ProxemicCircleArray, topic, self.on_circles, 10)

        self.get_logger().info(
            f'logging circles from {topic} -> {self.path}')

    def on_circles(self, msg):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        frame = msg.header.frame_id
        for c in msg.circles:
            self.writer.writerow([
                sec, nsec, frame,
                c.person_id, c.x, c.y, c.radius, c.peak_cost])
            self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every:
            self.file.flush()
            self._rows_since_flush = 0

    def destroy_node(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CircleLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
