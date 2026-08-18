"""
face_quality.py
Single job: turn a detected face box + frame (+ optional detector confidence)
into one scalar quality score in [0,1] -- how much to trust this reading.

This is deliberately independent of FaceResult.conf, which is just the emotion
model's own max softmax probability. A confidently-wrong read on a blurry,
tiny, or profile face still scores high on conf -- quality is the orthogonal
signal for "was this a good look at a face at all."

Three signals, averaged (det_weight is skipped, not defaulted, when unknown --
e.g. a box handed in from an external detector like the OAK-D's on-device one):
  - box area relative to the frame: tiny/far faces carry less signal
  - detector confidence: Haar's own certainty this box is a face at all
  - frontality: HSEmotion was trained on frontal faces, so a profile view is
    out-of-distribution input independent of how sure either detector is
"""
from __future__ import annotations

import cv2
import numpy as np

# Reference area fraction at which the area term saturates to 1.0. A face
# filling ~1/6 of the frame (typical webcam headshot distance) counts as
# "big enough" -- quality shouldn't keep climbing for closer-still faces.
_AREA_SATURATE = 1.0 / 6.0

# Haar's detectMultiScale3 level-weight is an uncalibrated cascade-stage
# score, not a probability -- strong frontal-face hits typically land
# around 3-6 with this module's default scale_factor/min_neighbors. Squash
# it into [0,1] against that empirical range.
_DET_CONF_SATURATE = 5.0

_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def _area_score(box, frame_shape) -> float:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    frac = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)
    return min(frac / _AREA_SATURATE, 1.0)


def _det_conf_score(det_weight: float) -> float:
    return min(max(det_weight, 0.0) / _DET_CONF_SATURATE, 1.0)


def _frontality_score(frame_bgr, box) -> float:
    """Proxy for pose, not identity: two eyes, level and symmetric about the
    face's midline, reads as frontal. One eye (profile) or none (steep angle /
    occlusion) reads as not-frontal."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    min_eye = max(1, (x2 - x1) // 8)
    eyes = _EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(min_eye, min_eye))
    if len(eyes) < 2:
        return 0.2 if len(eyes) == 1 else 0.0

    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(
        eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    cx1, cy1 = ex1 + ew1 / 2.0, ey1 + eh1 / 2.0
    cx2, cy2 = ex2 + ew2 / 2.0, ey2 + eh2 / 2.0
    fw, fh = x2 - x1, y2 - y1
    face_cx = fw / 2.0

    vertical_misalign = abs(cy1 - cy2) / fh
    # eyes equidistant from the face midline (in opposite directions) reads
    # as symmetric regardless of which eye detectMultiScale returned first
    horizontal_asym = abs((cx1 - face_cx) + (cx2 - face_cx)) / fw
    penalty = min(2.0 * (vertical_misalign + horizontal_asym), 1.0)
    return 1.0 - penalty


def face_quality(frame_bgr, box, det_weight: float | None = None) -> float:
    """box: (x1,y1,x2,y2) as used for this FaceResult. det_weight: Haar's raw
    level-weight for this detection (from face_detector.detect_largest_face_scored),
    or None if box came from an external detector -- the det-confidence term
    is then left out of the average rather than assumed."""
    scores = [_area_score(box, frame_bgr.shape), _frontality_score(frame_bgr, box)]
    if det_weight is not None:
        scores.append(_det_conf_score(det_weight))
    return float(np.mean(scores))