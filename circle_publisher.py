"""
circle_publisher.py
Repo-root ROS1 bridge from perception to navigation. Sits next to run.py, the
one file allowed to touch I/O and ROS -- video/common/ stays ROS-free.

Takes the VideoResult that vp.process() already returns each frame and
publishes its circles as affective_msgs/ProxemicCircleArray on
/proxemics/circles, the topic the costmap layer subscribes to. Reads ONLY
VideoResult.circles -- affect never crosses. Radius carries the affect signal;
peak_cost is the fixed global weight (confidence/arousal-driven peak_cost is
stated future work).
"""
import rospy

from std_msgs.msg import Header
from affective_msgs.msg import ProxemicCircle, ProxemicCircleArray

PEAK_COST = 0.85           # fixed global weight; tune to taste
BASE_FRAME = "base_footprint"   # matches myAGV odometry child_frame_id


class CirclePublisher:
    def __init__(self, topic="/proxemics/circles", frame=BASE_FRAME,
                 peak_cost=PEAK_COST, queue_size=10):
        self.frame = frame
        self.peak_cost = float(peak_cost)
        self.pub = rospy.Publisher(
            topic, ProxemicCircleArray, queue_size=queue_size)

    def publish(self, video_result, stamp=None):
        """video_result: a VideoResult from vp.process(). Only .circles read."""
        msg = ProxemicCircleArray()
        msg.header = Header()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.frame

        for c in video_result.circles:
            circle = ProxemicCircle()
            circle.person_id = int(c.track_id)
            circle.x = float(c.cx)
            circle.y = float(c.cy)
            circle.radius = float(c.radius)
            circle.peak_cost = self.peak_cost
            msg.circles.append(circle)

        self.pub.publish(msg)
        return msg
