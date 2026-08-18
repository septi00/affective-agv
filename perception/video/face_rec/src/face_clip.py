"""
face_clip.py
Offline helper: turn a video clip into summed HSEmotion 8-class logits.

Shared by score_crema_d.py and fuse_crema_d.py so they sample frames and run
the face model IDENTICALLY. If this loop drifted between the two harnesses,
the FACE-only numbers in each would stop being comparable -- the same
train/inference-parity trap as the audio raw-mic/ch0 mismatch.

This is the OFFLINE counterpart to the live per-frame path in
face_perception.py. It lives apart because it does file I/O
(cv2.VideoCapture), which the live perception core must not.

Returns the SUM of 8-logits over the frames where a face was found. argmax is
scale-invariant (so scoring is unaffected by sum-vs-mean) and the fusion
harness softmaxes the sum exactly as it did before -- existing results are
unchanged.
"""
from __future__ import annotations

import numpy as np
import cv2

from face_features import crop_face_rgb
from face_detector import detect_largest_face


def sum_face_logits(path, clf, frames_per_clip=3, lo=0.35, hi=0.65):
    """path: video file. clf: FaceClassifier. Samples the expressive middle of
    the clip, detects the largest face, crops, runs HSEmotion, sums the logits.

    Returns (summed_8logits, n_frames_used), or (None, 0) if the clip won't open
    or no face is found in any sampled frame."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 0

    summed = np.zeros(8)
    n = 0
    for pos in np.linspace(lo, hi, frames_per_clip) * total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        box = detect_largest_face(frame)
        if box is None:
            continue
        summed += clf.logits(crop_face_rgb(frame, box))
        n += 1
    cap.release()
    if n == 0:
        return None, 0
    return summed, n
