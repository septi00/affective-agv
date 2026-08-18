"""
emotion_live_oakd.py
Live OAK-D face emotion (HSEmotion enet_b0_8_best_vgaf), one independent
reading per face.

WHAT CHANGED AND WHY: the previous version called detect_largest_face_scored
and ran a single EmaSmoother. Two problems, both silent.

  1. Only the biggest face was ever classified. Everyone else was invisible.
  2. EmaSmoother holds ONE probability vector. Had a second face been fed to
     it, it would have exponentially averaged two people's emotional states
     together and reported the blend as a single confident answer.

Fix mirrors pose_live.py: PersonTracker gives each face a stable ID across
frames, PerTrackState gives each ID its own EmaSmoother, TargetSelector picks
the attended face. Smoothing is temporal, temporal needs identity.

Also fixed: `box` used to keep its previous value when detection failed, so the
overlay drew a stale box with stale scores at the old location. Tracks that
stop being detected now expire instead.

TRACKER UNITS ARE INFERENCE TICKS, NOT FRAMES: detection runs every
INFER_EVERY frames, and the tracker is updated only on those, so MAX_AGE=8
means 8 ticks -- about 32 frames, roughly a second at 30fps.

Run from: video/face_rec/
    python emotion_live_oakd.py
Press q to quit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import urllib.request  # fixes hsemotion-onnx 0.3.1 download bug; keep this import
import cv2
import numpy as np
import depthai as dai
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from face_features import crop_face_rgb
from face_detector import detect_faces_scored
from face_classifier import probs6, SIX
from face_quality import face_quality
from emotion_smoother import EmaSmoother
from person_tracker import PersonTracker, iou
from per_track_state import PerTrackState
from target_selector import TargetSelector

MODEL    = "enet_b0_8_best_vgaf"
# Same 6 classes as the rest of the system (probs6 drops Surprise/Contempt).
EMOTIONS = SIX
INFER_EVERY = 4
CAM_W, CAM_H = 640, 360

# True  = classify every tracked face (nicer demo, cost scales with headcount)
# False = classify only the attended face (fixed cost, what the CM5 wants once
#         SRP-PHAT, SER, pose and fusion are all running concurrently)
CLASSIFY_ALL = True

# Tracker units are INFERENCE TICKS (see module docstring). Haar drops out
# often on slight head turns, so MIN_HITS is low -- waiting 3 ticks to show a
# face is 12 frames of nothing.
IOU_THRESH = 0.3
MAX_AGE    = 8
MIN_HITS   = 2

SEL_MARGIN   = 1.25
SEL_PATIENCE = 3

# box border colour by quality tier, independent of the emotion colour used for the label
QUALITY_COLORS = [(0, 0, 230), (0, 200, 230), (0, 200, 0)]  # red / yellow / green
QUALITY_THRESHOLDS = (0.35, 0.6)

# BGR colours per emotion (keys = SIX)
COLORS = {
    'happy':   (0,   220,   0),
    'neutral': (180, 180, 180),
    'sad':     (255, 100,   0),
    'anger':   (0,     0, 230),
    'fear':    (0,   140, 255),
    'disgust': (0,   180, 100),
}


def quality_color(quality):
    if quality is None:
        return (150, 150, 150)
    lo, hi = QUALITY_THRESHOLDS
    if quality < lo:
        return QUALITY_COLORS[0]
    if quality < hi:
        return QUALITY_COLORS[1]
    return QUALITY_COLORS[2]


def detection_index(tbox, boxes, used):
    """Which detection a track matched to this tick. See pose_live.py -- same
    helper, same reasoning: exact box scores IoU 1.0, `used` keeps it one-to-one."""
    best_i, best_v = None, 0.0
    for i, b in enumerate(boxes):
        if i in used:
            continue
        v = iou(tbox, b)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def draw_faces(frame, results, target):
    """results: {track_id: (box, scores, dominant, quality)}"""
    for tid, (box, scores, dominant, quality) in results.items():
        is_target = (tid == target)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      quality_color(quality), 3 if is_target else 1)
        mark = "> " if is_target else ""
        label = f"{mark}T{tid} {dominant.upper()}  Q:{quality:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS.get(dominant, (0, 255, 0)), 2)


def draw_bars(frame, scores, dominant, tid):
    """Sidebar for the ATTENDED face only. Six bars per person would not fit,
    and fusion consumes one vector anyway -- so the panel shows the vector that
    would actually be fused."""
    h, w = frame.shape[:2]
    panel_x = w - 210
    cv2.rectangle(frame, (panel_x - 8, 8), (w - 4, len(EMOTIONS) * 30 + 40),
                  (30, 30, 30), -1)
    cv2.putText(frame, f"target T{tid}", (panel_x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y = 58
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

    tracker   = PersonTracker(iou_thresh=IOU_THRESH, max_age=MAX_AGE, min_hits=MIN_HITS)
    selector  = TargetSelector(margin=SEL_MARGIN, patience=SEL_PATIENCE)
    # one EmaSmoother per tracked face -- the whole point: an EMA is only
    # meaningful over one person's own history
    ema_state = PerTrackState(EmaSmoother)

    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                cam     = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                preview = cam.requestOutput((CAM_W, CAM_H), type=dai.ImgFrame.Type.BGR888p)
                q_frame = preview.createOutputQueue(maxSize=4, blocking=False)

                pipeline.start()
                print("Running — press q to quit")

                results = {}      # track_id -> (box, scores, dominant, quality)
                target  = None
                frame_n = 0

                while True:
                    frame_data = q_frame.tryGet()
                    if frame_data is None:
                        continue

                    frame = frame_data.getCvFrame()
                    frame_n += 1

                    if frame_n % INFER_EVERY == 0:
                        faces   = detect_faces_scored(frame)     # ALL faces now
                        boxes   = [b for b, _w in faces]
                        weights = [w for _b, w in faces]

                        tracks = tracker.update(boxes)
                        target = selector.update(tracks)

                        results = {}
                        used = set()
                        for tid, tbox in tracks:
                            di = detection_index(tbox, boxes, used)
                            if di is None:
                                continue
                            used.add(di)
                            box, det_weight = boxes[di], weights[di]

                            if not (CLASSIFY_ALL or tid == target):
                                continue

                            quality  = face_quality(frame, box, det_weight)
                            crop     = crop_face_rgb(frame, box)
                            _, logits = fer.predict_emotions(crop, logits=True)
                            probs    = probs6(np.array(logits, dtype=float))  # 6-vector, SIX order
                            smoothed = ema_state.get(tid).update(probs, quality)
                            scores   = {name: float(smoothed[i])
                                        for i, name in enumerate(EMOTIONS)}
                            dominant = max(scores, key=scores.__getitem__)
                            results[tid] = (box, scores, dominant, quality)

                        ema_state.evict(tracker.dead_ids)

                    draw_faces(frame, results, target)
                    if target in results:
                        _box, scores, dominant, _q = results[target]
                        draw_bars(frame, scores, dominant, target)

                    cv2.imshow("Emotion (OAK-D)", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
