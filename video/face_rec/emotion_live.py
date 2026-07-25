"""
emotion_live.py
Live webcam emotion recognition using HSEmotion (enet_b0_8_best_vgaf).
Uses the same crop path as the CREMA-D scoring harness for parity.

Run from: video/face_rec/
    python emotion_live.py
Press q to quit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import urllib.request  # fixes hsemotion-onnx 0.3.1 download bug; keep this import
import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from face_features import crop_face_rgb
from face_detector import detect_largest_face_scored
from face_quality import face_quality
from emotion_smoother import EmaSmoother

MODEL    = "enet_b0_8_best_vgaf"
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
INFER_EVERY = 4  # run inference every N frames; raise to 6-8 if too slow

# BGR colours per emotion
COLORS = {
    'Happiness': (0,   220,   0),
    'Neutral':   (180, 180, 180),
    'Sadness':   (255, 100,   0),
    'Anger':     (0,     0, 230),
    'Fear':      (0,   140, 255),
    'Disgust':   (0,   180, 100),
    'Surprise':  (0,   200, 220),
    'Contempt':  (120,  80, 200),
}

# box border colour by quality tier, independent of the emotion colour used for the label
QUALITY_COLORS = [(0, 0, 230), (0, 200, 230), (0, 200, 0)]  # red / yellow / green
QUALITY_THRESHOLDS = (0.35, 0.6)


def quality_color(quality):
    if quality is None:
        return (150, 150, 150)
    lo, hi = QUALITY_THRESHOLDS
    if quality < lo:
        return QUALITY_COLORS[0]
    if quality < hi:
        return QUALITY_COLORS[1]
    return QUALITY_COLORS[2]


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def draw_overlay(frame, box, scores, dominant, quality):
    h, w = frame.shape[:2]
    color = COLORS.get(dominant, (0, 255, 0))

    # face box (border colour = quality tier) + emotion/quality label
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), quality_color(quality), 2)
        q_text = f"  Q:{quality:.2f}" if quality is not None else ""
        cv2.putText(frame, dominant.upper() + q_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # sidebar bar chart
    panel_x = w - 210
    cv2.rectangle(frame, (panel_x - 8, 8), (w - 4, len(EMOTIONS) * 30 + 16),
                  (30, 30, 30), -1)
    y = 32
    for name in EMOTIONS:
        score = scores.get(name, 0.0)
        bar_len = int(score * 185)
        c = COLORS.get(name, (150, 150, 150))
        cv2.rectangle(frame, (panel_x, y - 14), (panel_x + bar_len, y + 4), c, -1)
        weight = 2 if name == dominant else 1
        cv2.putText(frame, f"{name[:3]}  {score:.2f}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), weight)
        y += 30


def main():
    print(f"Loading {MODEL} …")
    fer = HSEmotionRecognizer(model_name=MODEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0)")

    box      = None
    scores   = {e: 0.0 for e in EMOTIONS}
    dominant = "Neutral"
    quality  = None
    frame_n  = 0
    smoother = EmaSmoother()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        if frame_n % INFER_EVERY == 0:
            b, det_weight = detect_largest_face_scored(frame)
            if b is not None:
                box     = b
                quality = face_quality(frame, box, det_weight)
                crop = crop_face_rgb(frame, box)
                _, logits = fer.predict_emotions(crop, logits=True)
                probs     = softmax(np.array(logits, dtype=float))
                smoothed  = smoother.update(probs, quality)
                scores    = {name: float(smoothed[i]) for i, name in enumerate(EMOTIONS)}
                dominant  = max(scores, key=scores.__getitem__)

        draw_overlay(frame, box, scores, dominant, quality)
        cv2.imshow("Emotion", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()