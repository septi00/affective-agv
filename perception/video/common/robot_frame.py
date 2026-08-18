"""
robot_frame.py
Static camera -> robot-base transform for person positions.

Perception emits person anchors (torso_centroid) in the CAMERA optical frame:
    +x right, +y down, +z forward      (OpenCV / depthai convention)

Navigation wants them in the robot BASE frame (ROS REP-103):
    +x forward, +y left, +z up         origin on the floor under the base

Both are metres. This module is the ONE place that knows how the OAK-D is
bolted to the robot, so every mounting number lives here and nowhere else --
the same geometry the DOA bearing-to-person matching will reuse.

Nothing here is measured yet: the constants below are stubs. Measure them on
the actual robot with a tape and a level, drop the numbers in, and the seam to
the nav layer is live with no other code change.
"""
import numpy as np

# --- MOUNTING CONSTANTS: measure on the robot, then replace -----------------
# OAK-D optical centre position in the base frame, metres.
CAM_X = 0.0    # forward of base origin   (TODO measure)
CAM_Y = 0.0    # left of base origin      (TODO measure; ~0 if centred)
CAM_Z = 0.0    # height above the floor    (TODO measure)

# Camera pitch: downward tilt of the optical axis, radians.
# 0.0 = optical axis horizontal; positive = tilted DOWN toward the floor.
CAM_PITCH = 0.0    # (TODO measure)
# ---------------------------------------------------------------------------


# Optical axes (x right, y down, z fwd) -> base axes (x fwd, y left, z up),
# rotation only. base = _OPTICAL_TO_BASE @ cam.
_OPTICAL_TO_BASE = np.array([
    [0.0,  0.0, 1.0],    # base_x (fwd)  =  cam_z
    [-1.0, 0.0, 0.0],    # base_y (left) = -cam_x
    [0.0, -1.0, 0.0],    # base_z (up)   = -cam_y
])


def _pitch_about_base_y(theta):
    """Rotation about the base Y (left) axis, to undo a downward camera tilt.
    With theta = CAM_PITCH (camera tilted down), this lifts points back to
    level. Verify the sign once against a known test point when you measure."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c],
    ])


# Precomputed at import from the constants above (they're static per robot).
_R = _pitch_about_base_y(CAM_PITCH) @ _OPTICAL_TO_BASE
_T = np.array([CAM_X, CAM_Y, CAM_Z])


def camera_to_base(xyz_cam):
    """Person anchor: camera frame -> robot base frame, metres.

    xyz_cam: np.array([x, y, z]) from torso_centroid, or None.
    Returns np.array([x_fwd, y_left, z_up]), or None if given None.

    The proxemics circle lives on the base-frame ground plane: take
    (x_fwd, y_left) and ignore z_up -- a person's height doesn't change the
    footprint the robot must keep clear.
    """
    if xyz_cam is None:
        return None
    return _R @ np.asarray(xyz_cam, dtype=float) + _T


def ground_xy(xyz_cam):
    """Convenience: straight to the base-frame ground-plane (x_fwd, y_left)
    the planner needs. Returns np.array([x_fwd, y_left]) or None."""
    base = camera_to_base(xyz_cam)
    return None if base is None else base[:2]
