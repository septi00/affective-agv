"""
circle_relay_sender.py   (CM5 / Jazzy side -- NO ROS)

Replaces the ROS publisher. Takes the VideoResult your vp.process() already
returns and sends its circles as a small UDP datagram to the Pi4 over the
dedicated Ethernet link. See WIRE_FORMAT.md.

Perception stays exactly as-is; run.py calls relay.send(result) once per frame
instead of publishing a ROS message. No ROS on this side at all -- so it runs
fine on Jazzy (or anywhere), sidestepping the ROS1<->ROS2 gap entirely.
"""
import socket
import struct

# --- SET THESE to the Pi4's Ethernet IP and the port the receiver listens on
PI4_IP = "192.168.50.2"
PORT = 9944


class CircleRelaySender:
    def __init__(self, pi4_ip=PI4_IP, port=PORT):
        self.addr = (pi4_ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, video_result):
        """video_result: a VideoResult from vp.process(). Only .circles read."""
        circles = video_result.circles
        buf = struct.pack("<I", len(circles))
        for c in circles:
            # ProxemicsCircle: track_id, cx, cy, radius (base frame). peak_cost
            # is the fixed global constant (radius carries the affect signal).
            buf += struct.pack("<iffff",
                               int(c.track_id),
                               float(c.cx),
                               float(c.cy),
                               float(c.radius),
                               0.85)          # peak_cost constant
        try:
            self.sock.sendto(buf, self.addr)
        except OSError:
            # cable unplugged / receiver down: drop this frame, keep perceiving
            pass

    def close(self):
        self.sock.close()
