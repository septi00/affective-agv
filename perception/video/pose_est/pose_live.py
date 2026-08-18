"""
pose_live.py
Live OAK-D pose + body-affect features, one independent reading per person.

WHAT CHANGED AND WHY: the previous version created a single PoseFeatures before
the pipeline and called pf.update(pts) for every detection in the frame. That
object is stateful (self.prev wrists, self.speeds window, self.open_hist), so
with two people in frame it subtracted person A's wrist position from person B's
and called the difference a velocity. Nobody had to move for that number to be
large -- it was the distance between two bodies. `act` and `burst` pinned high,
hot hit 2, and pose_emotion applied BOOST_HIGH on every frame regardless of
posture. The single on-screen readout hid it, because every detection
overwrote last_txt and only the final person's numbers were ever displayed.

The fix is identity: PersonTracker assigns a stable ID per person, and
PerTrackState hands each ID its own PoseFeatures. Temporal features are only
meaningful within one person's history, so state is now keyed by who it
belongs to.

TargetSelector picks which person the robot attends to. Every track is still
computed -- only the choice of which one drives behaviour and fusion is
narrowed. Because state lives per track, attention can move back and forth with
no reset and no cold start.

TEST FOR THE FIX: two people, both standing still. `act` and `burst` should sit
near zero for both. Before the fix they pin high. Worth capturing on video --
that before/after is a figure.
"""
import sys, os

import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))

from pose_features import PoseFeatures
from pose_emotion import OPEN_THRESH
from person_tracker import PersonTracker, iou
from per_track_state import PerTrackState
from target_selector import TargetSelector

ARCHIVE_PATH = "models/yolov8n-pose.rvc2.tar.xz"

KEYPOINT_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
    'l_knee', 'r_knee', 'l_ankle', 'r_ankle'
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CONF_THRESH = 0.3
NN_W, NN_H  = 512, 288

# Tracker units are NN RESULTS, not display frames: update() is called once per
# q_nn message, so max_age=15 means 15 pose inferences without a match.
IOU_THRESH = 0.3
MAX_AGE    = 15
MIN_HITS   = 3

# Attention: a challenger needs 25% more box area, held for 5 NN results.
SEL_MARGIN   = 1.25
SEL_PATIENCE = 5

TARGET_COLOR = (0, 255, 255)     # attended person: yellow, thick
OTHER_COLOR  = (200, 200, 0)     # everyone else: dim teal, thin

# BGR colors: left side = greens/cyans, right side = reds/oranges, face = yellows
KEYPOINT_COLORS = [
    (0,   255, 255),  # 0  nose        yellow
    (255, 200,   0),  # 1  l_eye       cyan-green
    (0,   200, 255),  # 2  r_eye       orange
    (255, 255,   0),  # 3  l_ear       cyan
    (0,   165, 255),  # 4  r_ear       deep orange
    (0,   255,   0),  # 5  l_shoulder  green
    (0,     0, 255),  # 6  r_shoulder  red
    (100, 255,   0),  # 7  l_elbow     lime-green
    (0,    50, 255),  # 8  r_elbow     red-orange
    (200, 255,   0),  # 9  l_wrist     yellow-green
    (0,     0, 200),  # 10 r_wrist     dark red
    (255, 100,   0),  # 11 l_hip       teal
    (0,   100, 255),  # 12 r_hip       amber
    (255,   0, 100),  # 13 l_knee      blue-violet
    (0,   150, 200),  # 14 r_knee      dark orange
    (255,   0,   0),  # 15 l_ankle     blue
    (0,     0, 180),  # 16 r_ankle     dark red
]


# ------------------------------------------------------------------ extraction

def build_pts(det, depth_frame, intr):
    """One detection -> {kp_index: (px, py, conf, x3d, y3d, z_m)}.

    Pure extraction: no feature update, no drawing, no state. Splitting this out
    is what lets keypoints be built for EVERY person first, and features updated
    per track afterwards -- the two-pass structure the fix depends on."""
    fx, fy, cx, cy = intr
    pts = {}
    for i, kp in enumerate(det.getKeypoints()):
        px = int(kp.imageCoordinates.x * NN_W)
        py = int(kp.imageCoordinates.y * NN_H)
        conf = kp.confidence

        if conf <= CONF_THRESH:
            continue

        if depth_frame is not None:
            px_c = int(np.clip(px, 0, NN_W - 1))
            py_c = int(np.clip(py, 0, NN_H - 1))
            z_mm = depth_frame[py_c, px_c]
            if z_mm > 0:
                z_m = z_mm / 1000.0
                pts[i] = (px, py, conf,
                          (px - cx) * z_m / fx,
                          (py - cy) * z_m / fy,
                          z_m)
                continue
        pts[i] = (px, py, conf, None, None, None)
    return pts


def detection_index(tbox, boxes, used):
    """Which detection a track matched to this frame.

    The tracker returns the exact box it was handed, so the correct detection
    scores IoU 1.0. Matching by best IoU rather than tuple equality keeps this
    correct if boxes are ever rounded or scaled, and `used` enforces one
    detection per track so identical boxes can't both claim the same person."""
    best_i, best_v = None, 0.0
    for i, b in enumerate(boxes):
        if i in used:
            continue
        v = iou(tbox, b)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


# -------------------------------------------------------------------- drawing

def draw_skeleton(frame, pts, show_depth=False):
    for a, b in SKELETON:
        if a not in pts or b not in pts:
            continue
        xa, ya = pts[a][:2]
        xb, yb = pts[b][:2]
        ca, cb = KEYPOINT_COLORS[a], KEYPOINT_COLORS[b]
        line_color = tuple(int((ca[j] + cb[j]) / 2) for j in range(3))
        cv2.line(frame, (xa, ya), (xb, yb), line_color, 2)

    for i, pt in pts.items():
        px, py = pt[:2]
        cv2.circle(frame, (px, py), 4, KEYPOINT_COLORS[i], -1)
        # depth labels only on the attended person -- with several people the
        # per-joint text covers the whole frame and nothing is readable
        if show_depth and pt[5] is not None:
            cv2.putText(frame, f"{pt[5]:.2f}m", (px + 5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def draw_person_box(frame, box, tid, is_target):
    x1, y1, x2, y2 = (int(v) for v in box)
    col = TARGET_COLOR if is_target else OTHER_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if is_target else 1)
    tag = f"T{tid}" + ("  <target>" if is_target else "")
    cv2.putText(frame, tag, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def readout_line(tid, op, mean, burst, is_target):
    """One text line per tracked person. Showing EVERY track, not just one, is
    deliberate -- it is what makes the per-person independence visible, and what
    would have exposed the shared-state bug immediately."""
    mark = ">" if is_target else " "
    if op is None:
        return f"{mark}T{tid}  open --", (150, 150, 150)
    state = "OPEN" if op > OPEN_THRESH else "CLOSED"
    col = (0, 255, 0) if op > OPEN_THRESH else (0, 160, 255)
    txt = f"{mark}T{tid}  open {op:.2f} [{state}]"
    if mean is not None:
        txt += f"   act {mean:.2f}  burst {burst:.2f}"
    return txt, col


def draw_readouts(frame, readouts):
    y = 26
    for tid in sorted(readouts):
        txt, col = readouts[tid]
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        y += 24


# ----------------------------------------------------------------------- main

def main():
    nn_archive = dai.NNArchive(ARCHIVE_PATH)

    try:
        with dai.Device() as device:
            calib = device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_W, NN_H)
            fx, fy = intrinsics[0][0], intrinsics[1][1]
            cx, cy = intrinsics[0][2], intrinsics[1][2]
            cam_intrinsics = (fx, fy, cx, cy)

            tracker    = PersonTracker(iou_thresh=IOU_THRESH, max_age=MAX_AGE,
                                       min_hits=MIN_HITS)
            selector   = TargetSelector(margin=SEL_MARGIN, patience=SEL_PATIENCE)
            # one PoseFeatures per tracked person, built on first sight
            pose_state = PerTrackState(lambda: PoseFeatures(intrinsics=cam_intrinsics))

            with dai.Pipeline(device) as pipeline:
                cam_rgb   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                cam_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
                cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

                stereo = pipeline.create(dai.node.StereoDepth)
                cam_left.requestFullResolutionOutput().link(stereo.left)
                cam_right.requestFullResolutionOutput().link(stereo.right)
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                stereo.setOutputSize(NN_W, NN_H)

                nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
                    cam_rgb, nn_archive
                )
                nn.input.setBlocking(False)
                nn.input.setMaxSize(1)

                q_frame = nn.passthrough.createOutputQueue(maxSize=2, blocking=False)
                q_nn    = nn.out.createOutputQueue(maxSize=2, blocking=False)
                q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

                pipeline.start()
                print("Running — press q to quit")

                readouts = {}

                while True:
                    frame_data = q_frame.tryGet()
                    if frame_data is None:
                        continue

                    frame      = frame_data.getCvFrame()
                    nn_data    = q_nn.tryGet()
                    depth_data = q_depth.tryGet()

                    depth_frame = depth_data.getFrame() if depth_data is not None else None

                    if nn_data is not None:
                        dets = list(nn_data.detections)

                        # --- pass 1: geometry for EVERYONE, no state touched ---
                        boxes = [(int(d.xmin * NN_W), int(d.ymin * NN_H),
                                  int(d.xmax * NN_W), int(d.ymax * NN_H)) for d in dets]
                        pts_by_det = [build_pts(d, depth_frame, cam_intrinsics) for d in dets]

                        # --- identity, then attention ---
                        tracks = tracker.update(boxes)
                        target = selector.update(tracks)

                        # --- pass 2: features per person, into that person's own state ---
                        readouts = {}
                        used = set()
                        for tid, tbox in tracks:
                            di = detection_index(tbox, boxes, used)
                            if di is None:
                                continue
                            used.add(di)
                            pts = pts_by_det[di]

                            op, mean, burst = pose_state.get(tid).update(pts)

                            is_target = (tid == target)
                            draw_skeleton(frame, pts, show_depth=is_target)
                            draw_person_box(frame, tbox, tid, is_target)
                            readouts[tid] = readout_line(tid, op, mean, burst, is_target)

                        # not optional: without this, every person who ever
                        # walked past keeps a PoseFeatures alive for the whole run
                        pose_state.evict(tracker.dead_ids)

                    draw_readouts(frame, readouts)

                    cv2.imshow("Pose", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
