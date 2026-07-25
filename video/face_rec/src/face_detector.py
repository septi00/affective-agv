"""
face_detector.py
Haar-cascade largest-face detector for the FER channel.

One job: full BGR frame -> the biggest face box (x1,y1,x2,y2), or None.

This is the OFFLINE detector (decoded video frames). On the live OAK-D you will
likely run the camera's own on-device face detector instead -- in that case feed
its box straight into FacePerception.process(frame, box=...) and this file is
bypassed. It lives apart precisely so that swap touches one file only.
"""
from __future__ import annotations

import cv2

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_largest_face(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """frame_bgr: (H,W,3) uint8 BGR. Returns (x1,y1,x2,y2) of the largest face, or None."""
    box, _ = detect_largest_face_scored(frame_bgr, scale_factor, min_neighbors, min_size)
    return box


def detect_largest_face_scored(frame_bgr, scale_factor=1.1, min_neighbors=5, min_size=(48, 48)):
    """Same as detect_largest_face, but also returns Haar's own certainty that the
    box is a face at all (its cascade-stage level-weight, not a calibrated probability).
    Returns ((x1,y1,x2,y2), weight), or (None, None) if nothing was found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces, _reject_levels, level_weights = _CASCADE.detectMultiScale3(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=min_size, outputRejectLevels=True)
    if len(faces) == 0:
        return None, None
    i = max(range(len(faces)), key=lambda k: faces[k][2] * faces[k][3])
    x, y, w, h = faces[i]
    return (x, y, x + w, y + h), float(level_weights[i])
