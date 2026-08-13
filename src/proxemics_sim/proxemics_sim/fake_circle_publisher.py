import math

import rclpy
from rclpy.node import Node

from affective_msgs.msg import ProxemicCircle, ProxemicCircleArray


class FakeCirclePublisher(Node):
    """Stand-in for perception during home dev.

    Publishes a ProxemicCircleArray of synthetic people so the nav side
    (topic echo, and later the costmap layer) has something to consume
    without a camera attached. Person 0 is parked ahead; person 1 slowly
    circles, so the stream is visibly live.
    """

    def __init__(self):
        super().__init__('fake_circle_publisher')
        self.pub = self.create_publisher(
            ProxemicCircleArray, '/proxemics/circles', 10)
        self.t = 0.0
        self.dt = 0.2  # 5 Hz
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            'publishing synthetic circles on /proxemics/circles')

    def tick(self):
        self.t += self.dt

        msg = ProxemicCircleArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Person 0: stationary, 2 m straight ahead
        p0 = ProxemicCircle()
        p0.person_id = 0
        p0.x = 2.0
        p0.y = 0.0
        p0.radius = 0.8
        p0.peak_cost = 0.9

        # Person 1: slowly circling, proves the stream is live
        p1 = ProxemicCircle()
        p1.person_id = 1
        p1.x = 3.0 + math.cos(self.t * 0.5)
        p1.y = math.sin(self.t * 0.5)
        p1.radius = 1.2
        p1.peak_cost = 0.6

        msg.circles = [p0, p1]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeCirclePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
