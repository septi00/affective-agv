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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=scale_factor,
                                      minNeighbors=min_neighbors, minSize=min_size)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, x + w, y + h)
