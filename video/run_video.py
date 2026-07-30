"""
run_video.py
Live OAK-D runner for the video half of the system: ONE camera feeding BOTH
the body channel and the face channel, fused per tracked person.

This is the file that makes cross-channel association possible. pose_live.py
and emotion_live_oakd.py each open the OAK-D themselves, so they cannot run at
the same time, and even if they could they would be reading different frames --
a face box from moment A and a person box from moment B share no coordinate
space. Here one frame goes to both detectors, so a face box and a person box
are directly comparable and "whose face is this" has an answer.

The only file here that touches hardware, mirroring run_live.py on the audio
side. All perception is in video_perception.py, which is pure compute.

WHAT IS NOT WIRED YET: audio. VideoPerception.process() already accepts an
audio channel as (probs6, available, quality) and fuses it as a second peer
alongside face; there is simply no mic in this process yet. When run_live.py
and this merge, that argument is where the SER output goes and nothing else
changes.

Run from: video/
    python run_video.py
Press q to quit.
"""
import sys, os

import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork


_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("common", "pose_est", os.path.join("face_rec", "src")):
    sys.path.insert(0, os.path.join(_HERE, _p))

import urllib.request  # fixes hsemotion-onnx 0.3.1 download bug; keep BEFORE the lib
from video_perception import VideoPerception
from late_fusion import SIX
from open_oakd import open_device

ARCHIVE_PATH = os.path.join(_HERE, "pose_est", "models", "yolov8n-pose.rvc2.tar.xz")

CONF_THRESH = 0.3
NN_W, NN_H  = 512, 288

# Haar's default min_size (48,48) is tuned for full-resolution frames. The pose
# passthrough is 512x288, where a face at ~2m is around 40px -- with the default
# the face channel would silently never fire. See VideoPerception docstring.
FACE_MIN_SIZE = (32, 32)

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

TARGET_COLOR = (0, 255, 255)
OTHER_COLOR  = (200, 200, 0)
FACE_COLOR   = (255, 160, 0)

COLORS = {
    'happy':   (0,   220,   0),
    'neutral': (180, 180, 180),
    'sad':     (255, 100,   0),
    'anger':   (0,     0, 230),
    'fear':    (0,   140, 255),
    'disgust': (0,   180, 100),
}


# ------------------------------------------------------------------ extraction

def build_pts(det, depth_frame, intr):
    """One pose detection -> {kp_index: (px, py, conf, x3d, y3d, z_m)}.

    Device-specific (reads depthai keypoint objects), so it lives in the runner
    rather than the perception core. Pure extraction: no state, no features."""
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


# -------------------------------------------------------------------- drawing

def draw_person(frame, r, is_target):
    x1, y1, x2, y2 = (int(v) for v in r.box)
    col = TARGET_COLOR if is_target else OTHER_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if is_target else 1)

    # the face box drawn INSIDE the person box is the association, visible.
    # If a face ever appears attached to the wrong body, it shows up here.
    if r.face_box is not None:
        fx1, fy1, fx2, fy2 = (int(v) for v in r.face_box)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), FACE_COLOR, 2)

    tag = f"T{r.track_id} {r.label.upper()}"
    if is_target:
        tag += "  <target>"
    cv2.putText(frame, tag, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(r.label, col), 2)


def draw_skeleton(frame, pts):
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(frame, pts[a][:2], pts[b][:2], (90, 90, 90), 1)


def draw_channels(frame, result):
    """Per-person channel availability. This readout IS the modality-dropout
    story: watch F flip to '--' when someone turns away and the fused label
    keep working on body alone."""
    y = 22
    for r in result.people:
        mark = ">" if r.track_id == result.target_id else " "
        f = f"F {r.face_quality:.2f}" if r.face_available else "F  --"
        b = f"B {r.openness:.2f}" if r.body_available else "B  --"
        txt = f"{mark}T{r.track_id}  {f}  {b}  -> {r.label}"
        col = TARGET_COLOR if r.track_id == result.target_id else (170, 170, 170)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y += 20


def draw_bars(frame, r):
    """Fused 6-vector for the attended person -- the vector the robot acts on."""
    h, w = frame.shape[:2]
    px = w - 200
    cv2.rectangle(frame, (px - 8, 8), (w - 4, len(SIX) * 26 + 40), (30, 30, 30), -1)
    cv2.putText(frame, f"fused  T{r.track_id}", (px, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y = 56
    for i, name in enumerate(SIX):
        v = float(r.fused[i])
        c = COLORS.get(name, (150, 150, 150))
        cv2.rectangle(frame, (px, y - 12), (px + int(v * 175), y + 3), c, -1)
        cv2.putText(frame, f"{name[:3]} {v:.2f}", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    2 if name == r.label else 1)
        y += 26


# ----------------------------------------------------------------------- main

def main():
    nn_archive = dai.NNArchive(ARCHIVE_PATH)

    try:
        with open_device() as device:
            calib = device.readCalibration()
            intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_W, NN_H)
            cam_intrinsics = (intr[0][0], intr[1][1], intr[0][2], intr[1][2])

            print("Loading face model (HSEmotion) …")
            vp = VideoPerception(intrinsics=cam_intrinsics, face_min_size=FACE_MIN_SIZE)

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
                    cam_rgb, nn_archive)
                nn.input.setBlocking(False)
                nn.input.setMaxSize(1)

                q_frame = nn.passthrough.createOutputQueue(maxSize=2, blocking=False)
                q_nn    = nn.out.createOutputQueue(maxSize=2, blocking=False)
                q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

                pipeline.start()
                print("Running — press q to quit")

                result = None

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
                        people = []
                        for d in dets:
                            box = (int(d.xmin * NN_W), int(d.ymin * NN_H),
                                   int(d.xmax * NN_W), int(d.ymax * NN_H))
                            people.append((box, build_pts(d, depth_frame, cam_intrinsics)))

                        # ONE frame, both channels, same coordinate space.
                        # audio=None until run_live.py merges in here.
                        result = vp.process(frame, people, audio=None)

                        for (_box, pts) in people:
                            draw_skeleton(frame, pts)

                    if result is not None:
                        for r in result.people:
                            draw_person(frame, r, r.track_id == result.target_id)
                        draw_channels(frame, result)
                        if result.target is not None:
                            draw_bars(frame, result.target)

                    cv2.imshow("run_video", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
