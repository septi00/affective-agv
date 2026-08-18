"""
emotion_smoother.py
Exponential moving average over the emotion probability vector, so the label
on the live overlay doesn't flip every inference tick just because two close
classes trade the argmax from one noisy read to the next.

The update weight is scaled by the frame's face_quality (small/profile/blurry
detections nudge the average only slightly; confident frontal ones pull it
toward themselves quickly) -- reuses the same trust signal already computed
per frame rather than treating every reading as equally reliable.
"""
from __future__ import annotations

import numpy as np


class EmaSmoother:
    """alpha: max per-update weight, applied at quality == 1.0.
    quality_floor: minimum weight a reading gets even at quality == 0, so a
    long run of poor detections still drifts (rather than freezing) the average."""

    def __init__(self, alpha: float = 0.3, quality_floor: float = 0.15):
        self.alpha = alpha
        self.quality_floor = quality_floor
        self._probs: np.ndarray | None = None

    def update(self, probs, quality: float | None) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self._probs is None:
            self._probs = probs
            return self._probs

        q = self.quality_floor if quality is None else max(quality, self.quality_floor)
        weight = self.alpha * q
        self._probs = weight * probs + (1 - weight) * self._probs
        return self._probs

    def reset(self):
        self._probs = None
