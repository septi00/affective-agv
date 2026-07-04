"""
face_classifier.py
HSEmotion (ONNX) wrapper for the face-emotion channel.

One job: run HSEmotion on an already-cropped RGB face and return the raw
8-class logits, plus the constants that map those 8 classes onto your 6.
Detection and cropping live elsewhere (face_detector.py / face_features.py) --
this file owns only the model. Video analog of sound_classifier.py.

Deliberately returns the honest 8-class logits, NOT a reduced 6-vector: the
consumer decides how to reduce (softmax over the kept 6 for fusion, or an
argmax-over-8 decision rule for scoring). Same idea as the audio side -- one
model output, several consumers.
"""
from __future__ import annotations

import urllib.request   # keep: fixes hsemotion-onnx 0.3.1 weight-download bug (import BEFORE the lib)

import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# HSEmotion's 8 classes, in the model's own output order.
HSE = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
HSE_TO_6 = {'Anger': 'anger', 'Disgust': 'disgust', 'Fear': 'fear',
            'Happiness': 'happy', 'Neutral': 'neutral', 'Sadness': 'sad'}
SIX = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
# Indices into the 8-logit vector that line up with SIX, in SIX order.
# (Contempt=1 and Surprise=7 have no CREMA-D equivalent and are dropped.)
HSE_KEEP = [0, 2, 3, 4, 5, 6]


def probs6(logits8):
    """8-class logits -> 6-class probability vector in SIX order (softmax over kept 6)."""
    v = np.asarray(logits8, dtype=np.float64)[HSE_KEEP]
    e = np.exp(v - v.max())
    return e / e.sum()


class FaceClassifier:
    """Thin HSEmotion wrapper. Feed it a cropped RGB face -> raw 8-class logits."""

    def __init__(self, model_name="enet_b0_8_best_vgaf"):
        self.fer = HSEmotionRecognizer(model_name=model_name)

    def logits(self, face_rgb) -> np.ndarray:
        """face_rgb: square RGB uint8 crop from face_features.crop_face_rgb.
        Returns the raw 8-class logit vector (float64)."""
        _, logits = self.fer.predict_emotions(face_rgb, logits=True)
        return np.asarray(logits, dtype=np.float64)
