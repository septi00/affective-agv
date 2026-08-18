"""
face_detector.py
Haar-cascade face detector for the FER channel.

One job: full BGR frame -> face boxes (x1,y1,x2,y2), largest first.

detectMultiScale3 always returned EVERY face; the previous version discarded
all but the biggest inside this file. detect_faces_scored now exposes the full
list, and the two largest-face functions are thin wrappers over it.

WHY THE WRAPPERS STAY: face_clip.sum_face_logits calls detect_largest_face on
CREMA-D clips, and score_crema_d.py / fuse_crema_d.py depend on it. Those
harnesses produced the 0.522 / 0.738 / 0.742 numbers in the results chapter.
The wrapper picks max area with ties broken by detection order -- identical to
the old inline max() -- so the offline path is bit-for-bit unchanged and those
numbers do not have to be re-run.

This is the OFFLINE detector (decoded video frames). On the live OAK-D you will
likely run the camera's own on-device face detector instead -- in that case feed
its boxes straight into FacePerception.process(frame, box=...) and this file is
bypassed. It lives apart precisely so that swap touches one file only.
"""
from __future__ import annotations

import cv2

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def _box_area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def detect_faces_scored(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """frame_bgr: (H,W,3) uint8 BGR.

    Returns [((x1,y1,x2,y2), level_weight), ...] for EVERY face found, sorted
    largest-area first. Empty list if none.

    level_weight is Haar's own cascade-stage score -- its certainty that the box
    is a face at all, NOT a calibrated probability and NOT an emotion
    confidence. face_quality.face_quality() squashes it into [0,1].
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces, _reject_levels, level_weights = _CASCADE.detectMultiScale3(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=min_size, outputRejectLevels=True)
    if len(faces) == 0:
        return []

    out = [((int(x), int(y), int(x + w), int(y + h)), float(level_weights[k]))
           for k, (x, y, w, h) in enumerate(faces)]
    # stable sort: equal-area boxes keep detection order, matching the old
    # max()-picks-the-first-maximum behaviour exactly
    out.sort(key=lambda fb: -_box_area(fb[0]))
    return out


def detect_largest_face_scored(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """Largest face plus Haar's certainty for it.
    Returns ((x1,y1,x2,y2), weight), or (None, None) if nothing was found."""
    faces = detect_faces_scored(frame_bgr, scale_factor, min_neighbors, min_size)
    if not faces:
        return None, None
    return faces[0]


def detect_largest_face(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """frame_bgr: (H,W,3) uint8 BGR. Returns (x1,y1,x2,y2) of the largest face, or None."""
    box, _ = detect_largest_face_scored(frame_bgr, scale_factor, min_neighbors, min_size)
    return box
