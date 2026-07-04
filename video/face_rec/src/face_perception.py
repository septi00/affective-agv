"""
face_perception.py
Face-emotion perception core: one frame in, one FaceResult out.

I/O-free and device-agnostic -- no camera, no drawing, no printing -- mirroring
perception.py on the audio side. The live OAK-D runner will go through here.

process(frame, box=None):
  box=None  -> detect the largest face internally (Haar fallback)
  box given -> use it as-is (e.g. the OAK-D's on-device face detector)

FaceResult carries the full 6-class probability vector, not just the winning
label, so late fusion with the audio channel stays a plain weighted average of
two 6-vectors in the SAME order [anger, disgust, fear, happy, neutral, sad].

NOTE ON PARITY WITH THE HARNESSES: this core reduces per frame (softmax to a
6-vector). The offline harnesses instead SUM 8-logits across sampled frames and
reduce once, via face_clip.sum_face_logits -- a different, deliberate order of
operations for clip-level scoring. Both feed HSEmotion through the same
FaceClassifier + detector + crop, so the model input is identical; only the
temporal aggregation differs by regime (live stream vs whole clip).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_features import crop_face_rgb
from face_detector import detect_largest_face
from face_classifier import FaceClassifier, SIX, probs6


@dataclass
class FaceResult:
    box: tuple | None            # (x1,y1,x2,y2) used for this result, or None
    probs: np.ndarray | None     # 6-vector in SIX order, or None if no face
    label: str | None = None     # convenience: argmax label
    conf: float | None = None    # convenience: max prob

    @property
    def found(self) -> bool:
        return self.probs is not None


class FacePerception:
    """Owns the face model; turns a frame into a FaceResult. No I/O."""

    def __init__(self, model_name="enet_b0_8_best_vgaf"):
        self.clf = FaceClassifier(model_name=model_name)

    def process(self, frame_bgr, box=None) -> FaceResult:
        if box is None:
            box = detect_largest_face(frame_bgr)
        if box is None:
            return FaceResult(box=None, probs=None)

        crop = crop_face_rgb(frame_bgr, box)
        p = probs6(self.clf.logits(crop))
        i = int(p.argmax())
        return FaceResult(box=tuple(box), probs=p, label=SIX[i], conf=float(p[i]))
