#!/usr/bin/env python
"""
circle_relay_receiver.py   (Pi4 / Noetic side -- ROS1)

Listens on the UDP socket for circle datagrams from the CM5, unpacks them
(see WIRE_FORMAT.md), and republishes as affective_msgs/ProxemicCircleArray
on /proxemics/circles -- the exact topic the proxemic costmap layer already
subscribes to. The layer doesn't change; circles just appear as if published
locally.

This is the ONLY place the two ROS worlds meet: raw numbers come off the wire,
a ROS1 message goes out. No ros1_bridge, no version-gap machinery.
"""
import socket
import struct

import rospy
from std_msgs.msg import Header
from affective_msgs.msg import ProxemicCircle, ProxemicCircleArray

PORT = 9944
BASE_FRAME = "base_footprint"
HEADER = struct.Struct("<I")
CIRCLE = struct.Struct("<iffff")


def main():
    rospy.init_node("circle_relay_receiver")
    pub = rospy.Publisher("/proxemics/circles", ProxemicCircleArray, queue_size=10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.5)   # so we can check for shutdown between packets
    rospy.loginfo("circle relay receiver listening on UDP %d -> /proxemics/circles", PORT)

    while not rospy.is_shutdown():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            continue

        if len(data) < HEADER.size:
            continue
        (n,) = HEADER.unpack_from(data, 0)

        msg = ProxemicCircleArray()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = BASE_FRAME

        off = HEADER.size
        ok = True
        for _ in range(n):
            if off + CIRCLE.size > len(data):
                ok = False
                break
            pid, x, y, radius, peak = CIRCLE.unpack_from(data, off)
            off += CIRCLE.size
            c = ProxemicCircle()
            c.person_id = int(pid)
            c.x = float(x)
            c.y = float(y)
            c.radius = float(radius)
            c.peak_cost = float(peak)
            msg.circles.append(c)

        if ok:
            pub.publish(msg)

    sock.close()


if __name__ == "__main__":
    main()
