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
EXTREMITIES = [L_WR, R_WR]


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
    """Mean 3D perpendicular distance of wrists from the torso midline,
    normalised by shoulder width. Hands by sides/pockets -> low; arms spread
    or extended toward camera -> high.

    Midline = nose to shoulder midpoint. Missing depth is filled by reprojecting
    at the median Z of all detected keypoints, so the reading is continuous."""
    zs = [p[5] for p in pts.values() if p[5] is not None]
    fallback_z = float(np.median(zs)) if zs else None

    nose = _est_xyz(pts, NOSE,  fallback_z, intrinsics)
    ls   = _est_xyz(pts, L_SH,  fallback_z, intrinsics)
    rs   = _est_xyz(pts, R_SH,  fallback_z, intrinsics)
    if nose is None or ls is None or rs is None:
        return None

    sh_mid   = (ls + rs) / 2.0
    axis     = sh_mid - nose
    axis_len = np.linalg.norm(axis)
    sh_width = np.linalg.norm(ls - rs)
    if axis_len < 0.01 or sh_width < 0.02:
        return None
    axis_hat = axis / axis_len

    ds = []
    for i in EXTREMITIES:
        w = _est_xyz(pts, i, fallback_z, intrinsics)
        if w is None:
            continue
        v = w - nose
        perp = v - np.dot(v, axis_hat) * axis_hat
        ds.append(np.linalg.norm(perp))

    if not ds:
        return None
    return float(np.mean(ds) / sh_width)


class PoseFeatures:
    """Tracks openness and activity over a sliding window."""
    def __init__(self, window=30, intrinsics=None):
        self.prev = None
        self.speeds = deque(maxlen=window)
        self.intrinsics = intrinsics

    def update(self, pts):
        zs = [p[5] for p in pts.values() if p[5] is not None]
        fallback_z = float(np.median(zs)) if zs else None

        cur = {i: p for i in (L_WR, R_WR, L_SH, R_SH)
               if (p := _est_xyz(pts, i, fallback_z, self.intrinsics)) is not None}

        sh_scale = (np.linalg.norm(cur[L_SH] - cur[R_SH])
                    if L_SH in cur and R_SH in cur else None)

        if self.prev is not None and sh_scale and sh_scale > 0.02:
            shared = [i for i in cur if i in self.prev]
            if shared:
                moved = np.mean([np.linalg.norm(cur[i] - self.prev[i]) for i in shared])
                self.speeds.append(float(moved / sh_scale))
        self.prev = cur

        op       = openness(pts, self.intrinsics)
        activity = float(np.mean(self.speeds)) if self.speeds else None
        return op, activity