"""
pose_features.py
Openness + activity from YOLOv8-pose 3D joints (COCO-17).

Feed it the per-frame joint dict you already build in pose_live.py:
    pts[i] = (px, py, conf, x3d, y3d, z_m)

Openness  = mean 3D perpendicular distance of wrists from the torso midline
            (nose -> shoulder midpoint), normalised by shoulder width.
            Keypoints without valid stereo depth are reprojected at the median
            body depth, so depth holes don't kill the reading but the full
            Z component is used when data is available.
Activity  = mean 3D joint speed over a short window, normalised by body scale.
"""
import numpy as np
from collections import deque

# COCO-17 indices
NOSE = 0
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_AN, R_AN = 15, 16
L_EL, R_EL = 7, 8
EXTREMITIES = [L_WR, R_WR]
# upper-body joints whose 3D convex-hull VOLUME = how much space the body occupies
VOLUME_JOINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]
JITTER_FLOOR = 0.05   # per-joint normalised movement below this = keypoint noise, ignored


def _xyz(pts, i):
    """Return np.array([x3d, y3d, z_m]) for joint i, or None if no valid 3D."""
    if i in pts and pts[i][3] is not None:
        return np.array(pts[i][3:6], dtype=float)
    return None


def _est_xyz(pts, i, fallback_z, intrinsics):
    """Best-effort 3D position: use stored 3D if valid, else reproject pixel
    position at fallback_z using camera intrinsics."""
    if i not in pts:
        return None
    p = pts[i]
    if p[3] is not None:
        return np.array(p[3:6], dtype=float)
    if fallback_z is not None and intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        x3d = (p[0] - cx) * fallback_z / fx
        y3d = (p[1] - cy) * fallback_z / fy
        return np.array([x3d, y3d, fallback_z])
    return None


def body_scale(pts):
    """Shoulder width in metres (fallback: shoulder-hip length). None if unknown."""
    ls, rs = _xyz(pts, L_SH), _xyz(pts, R_SH)
    if ls is not None and rs is not None:
        w = np.linalg.norm(ls - rs)
        if w > 0.05:
            return w
    lh, rh = _xyz(pts, L_HIP), _xyz(pts, R_HIP)
    if ls is not None and lh is not None:
        return max(np.linalg.norm(ls - lh), 0.05)
    return None


def body_center(pts):
    """Shoulder midpoint in 3D."""
    ls, rs = _xyz(pts, L_SH), _xyz(pts, R_SH)
    if ls is not None and rs is not None:
        return (ls + rs) / 2.0
    return None


def openness(pts, intrinsics=None):
    """Openness = mean 3D distance of BOTH wrists from the body centre (shoulder
    midpoint), in shoulder-width units. Full 3D -> reaching to the SIDE, UP, or
    toward the CAMERA all count. Only hands near the body read closed.

    Robust to depth holes: if a wrist/shoulder has no valid stereo depth, its 3D
    position is reprojected at the median body depth, so openness ALWAYS computes
    when the joints are detected (it never freezes on missing wrist depth).
    """
    zs = [p[5] for p in pts.values() if p[5] is not None]
    fallback_z = float(np.median(zs)) if zs else None

    ls = _est_xyz(pts, L_SH, fallback_z, intrinsics)
    rs = _est_xyz(pts, R_SH, fallback_z, intrinsics)
    if ls is None or rs is None:
        return None
    sh_mid   = (ls + rs) / 2.0
    sh_width = np.linalg.norm(ls - rs)
    if sh_width < 0.02:
        return None

    reaches = []
    for i in (L_WR, R_WR):
        w = _est_xyz(pts, i, fallback_z, intrinsics)      # falls back to body depth if no stereo
        if w is not None:
            reaches.append(np.linalg.norm(w - sh_mid))
    if len(reaches) < 2:                                   # need both wrists detected
        return None

    return float(np.mean(reaches) / sh_width)

class PoseFeatures:
    """Tracks openness and activity over a sliding window."""
    def __init__(self, window=15, intrinsics=None, open_window=7):
        self.prev = None
        self.speeds = deque(maxlen=window)
        self.open_hist = deque(maxlen=open_window)   # smooth noisy openness
        self.intrinsics = intrinsics

    def update(self, pts):
        zs = [p[5] for p in pts.values() if p[5] is not None]
        fallback_z = float(np.median(zs)) if zs else None

        cur = {i: p for i in (L_WR, R_WR, L_SH, R_SH)
               if (p := _est_xyz(pts, i, fallback_z, self.intrinsics)) is not None}

        sh_scale = (np.linalg.norm(cur[L_SH] - cur[R_SH])
                    if L_SH in cur and R_SH in cur else None)

        # torso (shoulder-centre) motion this frame -- subtract it so walking/
        # translating the whole body doesn't count as arm activity. Only motion
        # of the wrists RELATIVE to the torso is "activity".
        torso_now  = ((cur[L_SH] + cur[R_SH]) / 2.0) if (L_SH in cur and R_SH in cur) else None
        torso_prev = ((self.prev[L_SH] + self.prev[R_SH]) / 2.0) \
                     if (self.prev is not None and L_SH in self.prev and R_SH in self.prev) else None

        if self.prev is not None and sh_scale and sh_scale > 0.02 and torso_now is not None and torso_prev is not None:
            torso_disp = torso_now - torso_prev
            wrists = [i for i in (L_WR, R_WR) if i in cur and i in self.prev]
            if wrists:
                # each wrist's movement MINUS the torso's movement = motion relative to body
                per_joint = [np.linalg.norm((cur[i] - self.prev[i]) - torso_disp) / sh_scale
                             for i in wrists]
                per_joint = [m if m > JITTER_FLOOR else 0.0 for m in per_joint]
                self.speeds.append(float(np.mean(per_joint)))
        self.prev = cur

        raw_op = openness(pts, self.intrinsics)
        if raw_op is not None:
            self.open_hist.append(raw_op)
        op = float(np.median(self.open_hist)) if self.open_hist else None  # median = robust to depth spikes

        mean  = float(np.mean(self.speeds)) if self.speeds else None   # sustained energy
        burst = float(np.max(self.speeds))  if self.speeds else None   # sharpest recent movement
        return op, mean, burst