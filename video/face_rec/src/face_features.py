"""
face_features.py
Shared face-crop preprocessing for the FER channel (HSEmotion backend).

One job: turn a detector's face box + a frame into a clean RGB crop.
HSEmotion does its own resize and normalization internally, so this module
does NOT resize or normalize -- it only crops squarely with margin and returns
RGB uint8. Keeping this as the single crop path means the CREMA-D scoring
harness and the live OAK loop feed HSEmotion identically (train/inference
parity, the lesson from the audio raw-mic/ch0 mismatch).
"""

from __future__ import annotations
import cv2
import numpy as np

BOX_MARGIN = 0.30   # expand the squared box by 30% per side (keeps forehead/chin context)


def _square_with_margin(box):
    """(x1,y1,x2,y2) -> square box, same centre, expanded by BOX_MARGIN. Not clamped
    here; padding in crop_face_rgb keeps an edge face centred instead of squashed."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * BOX_MARGIN)
    half = side / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def crop_face_rgb(frame_bgr: np.ndarray, box) -> np.ndarray:
    """frame_bgr: full frame (H,W,3) uint8 BGR.  box: (x1,y1,x2,y2) in pixels.
    Returns a square RGB uint8 crop, ready to hand to HSEmotion.predict_emotions()."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _square_with_margin(box)

    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    crop = frame_bgr[y1c:y2c, x1c:x2c]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)