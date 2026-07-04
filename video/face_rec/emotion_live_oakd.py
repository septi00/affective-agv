"""
emotion_live_oakd.py
Live OAK-D emotion recognition using HSEmotion (enet_b0_8_best_vgaf).
Copy of emotion_live.py with cv2.VideoCapture replaced by the OAK-D RGB stream.

Run from: video/face_rec/
    python emotion_live_oakd.py
Press q to quit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import urllib.request  # fixes hsemotion-onnx 0.3.1 download bug; keep this import
import cv2
import numpy as np
import depthai as dai
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from face_features import crop_face_rgb

MODEL    = "enet_b0_8_best_vgaf"
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
INFER_EVERY = 4
CAM_W, CAM_H = 640, 360

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

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame_bgr):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, x + w, y + h)


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def draw_overlay(frame, box, scores, dominant):
    h, w = frame.shape[:2]
    color = COLORS.get(dominant, (0, 255, 0))

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, dominant.upper(), (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

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

    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam     = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                preview = cam.requestOutput((CAM_W, CAM_H), type=dai.ImgFrame.Type.BGR888p)
                q_frame = preview.createOutputQueue(maxSize=4, blocking=False)

                pipeline.start()
                print("Running — press q to quit")

                box      = None
                scores   = {e: 0.0 for e in EMOTIONS}
                dominant = "Neutral"
                frame_n  = 0

                while True:
                    frame_data = q_frame.tryGet()
                    if frame_data is None:
                        continue

                    frame = frame_data.getCvFrame()
                    frame_n += 1

                    if frame_n % INFER_EVERY == 0:
                        b = detect_face(frame)
                        if b is not None:
                            box = b
                            crop = crop_face_rgb(frame, box)
                            _, logits = fer.predict_emotions(crop, logits=True)
                            probs     = softmax(np.array(logits, dtype=float))
                            scores    = {name: float(probs[i]) for i, name in enumerate(EMOTIONS)}
                            dominant  = max(scores, key=scores.__getitem__)

                    draw_overlay(frame, box, scores, dominant)
                    cv2.imshow("Emotion (OAK-D)", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()