"""
run.py
Repo-root entry point. Live OAK-D + ReSpeaker runner for the video half of the system: ONE camera feeding BOTH
the body channel and the face channel, fused per tracked person.

This is the file that makes cross-channel association possible. pose_live.py
and emotion_live_oakd.py each open the OAK-D themselves, so they cannot run at
the same time, and even if they could they would be reading different frames --
a face box from moment A and a person box from moment B share no coordinate
space. Here one frame goes to both detectors, so a face box and a person box
are directly comparable and "whose face is this" has an answer.

The only file here that touches hardware, mirroring run_live.py on the audio
side. All perception is in video_perception.py, which is pure compute.

ALL THREE CHANNELS. Face and body come from the OAK-D in this loop. Audio comes
from the ReSpeaker on a worker thread via AudioBridge, because the channels run
on different clocks -- video turns at ~30fps, the SER needs ~1s of sound before
it can say anything. Calling perception.feed() inline would either stall the
display or starve the audio, so the bridge publishes its newest result into a
slot this loop reads without waiting.

The audio path reuses noise/src/perception.py unchanged -- the same core
run_live.py drives. Two front-ends over one perception core, which is the whole
reason that file is I/O-free.

AUDIO IS ROBOT-ONLY. The ReSpeaker is on the CM5. Set USE_AUDIO=False (or just
run without the mic attached) to fall back to face+body on the MacBook -- the
fusion call is identical, audio simply reports unavailable.

Run from the repo root:
    python run.py
Press q to quit.

Paths below are relative to the ROOT, not to video/. If this file is ever moved
back inside video/, the sys.path block and the two model paths are the only
things that change.
"""
import sys, os, time

import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

_HERE = os.path.dirname(os.path.abspath(__file__))          # repo root
for _p in (os.path.join("video", "common"),
           os.path.join("video", "pose_est"),
           os.path.join("video", "face_rec", "src"),
           os.path.join("noise", "src")):                    # audio core, imported not copied
    sys.path.insert(0, os.path.join(_HERE, _p))

import urllib.request  # fixes hsemotion-onnx 0.3.1 download bug; keep BEFORE the lib
from video_perception import VideoPerception
from late_fusion import SIX
from open_oakd import open_device
from audio_bridge import AudioBridge

ARCHIVE_PATH = os.path.join(_HERE, "video", "pose_est", "models",
                            "yolov8n-pose.rvc2.tar.xz")

CONF_THRESH = 0.3
NN_W, NN_H  = 512, 288

# --- audio channel -------------------------------------------------------
# False = face + body only. Use on the MacBook, or to isolate a video problem
# without the mic in the picture.
USE_AUDIO   = True
AUDIO_MODELS = os.path.join(_HERE, "noise", "models")
YAMNET_PATH  = os.path.join(AUDIO_MODELS, "yamnet.tflite")
YAMNET_MAP   = os.path.join(AUDIO_MODELS, "yamnet_class_map.csv")
SER_PATH     = os.path.join(AUDIO_MODELS, "emotion_fusion_resnet18_merged.pt")

# stdout cadence. A status line every frame would be 30/s of unreadable spam,
# so print on a fixed interval -- plus immediately whenever the attended
# person's label changes, because a change is the event worth seeing and
# waiting up to a second for it makes the system feel unresponsive.
REPORT_EVERY_S = 1.0

# An X_LINK_ERROR tears down the pipeline and rebuilds it instead of ending the
# run. Audio keeps running across a reconnect -- it is a separate device on a
# separate thread and has no reason to stop.
MAX_RECONNECTS   = 5
RECONNECT_DELAY_S = 3.0

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


class Reporter:
    """The only place a PerceptionResult becomes stdout, mirroring _report() in
    run_live.py. Keeps printing out of the perception and drawing code."""

    def __init__(self, every_s=REPORT_EVERY_S):
        self.every_s = every_s
        self._last_t = 0.0
        self._last_label = None

    def maybe(self, result, audio_now, doa):
        target = result.target if result is not None else None
        label = target.label if target is not None else None
        now = time.monotonic()
        if label == self._last_label and now - self._last_t < self.every_s:
            return
        self._last_t = now
        self._last_label = label
        print(self._line(target, audio_now, doa), flush=True)

    @staticmethod
    def _line(target, audio_now, doa):
        _probs, a_avail, a_quality = audio_now
        parts = [time.strftime("%H:%M:%S")]

        if target is None:
            parts.append("no person")
        else:
            conf = float(target.fused[int(np.argmax(target.fused))])
            parts.append(f"T{target.track_id} {target.label:<7} {conf:.2f}")
            # per-channel availability: which modalities actually contributed
            f = f"F {target.face_quality:.2f}" if target.face_available else "F  -- "
            b = f"B {target.openness:.2f}" if target.body_available else "B  -- "
            a = f"A {a_quality:.2f}" if a_avail else "A  -- "
            parts.append(f"{f}  {b}  {a}")

        if doa:
            parts.append("DOA " + "  ".join(
                f"S{t.id} {t.angle_deg:6.1f}deg +/-{t.std_deg:.1f}" for t in doa))
        else:
            parts.append("DOA --")

        return "  |  ".join(parts)


def draw_audio(frame, audio, age, doa=()):
    """Audio is ONE reading for the scene, not per person: the mic array cannot
    yet say which tracked person spoke, so it is applied to the attended person
    only. Closing that gap means matching a DOA bearing against the target's
    pixel-x through the camera FOV -- the remaining piece of the association
    story, and the reason this line is drawn separately from the per-person
    ones rather than alongside them."""
    _probs, avail, quality = audio
    txt = f" A {quality:.2f}  ({age:.1f}s)" if avail else " A  --"
    if doa:
        txt += "   DOA " + " ".join(f"S{t.id} {t.angle_deg:.0f}deg" for t in doa)
    col = (0, 220, 220) if avail else (120, 120, 120)
    h = frame.shape[0]
    cv2.putText(frame, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


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

def start_audio():
    """Returns a started AudioBridge, or None if audio is off or unavailable.

    Failure is non-fatal on purpose: a missing ReSpeaker should degrade the
    system to face+body, not stop it. That is the same modality-dropout
    principle applied at startup instead of per frame."""
    if not USE_AUDIO:
        return None
    try:
        from perception import Perception
        from config import BLOCK, DEVICE, FS, MIC_CHS, PROC_CH, TOTAL_CH

        print("Loading audio models (YAMNet + SER) …")
        perception = Perception(yamnet_path=YAMNET_PATH, yamnet_labels=YAMNET_MAP,
                                ser_path=SER_PATH, block=BLOCK)
        bridge = AudioBridge(perception, device=DEVICE, samplerate=FS,
                             channels=TOTAL_CH, blocksize=BLOCK,
                             proc_ch=PROC_CH, mic_chs=MIC_CHS)
        bridge.start()
        print("Audio channel running")
        return bridge
    except Exception as e:
        print(f"[audio disabled] {e}")
        return None


def run_session(nn_archive, audio, reporter):
    """One device session. Returns True if the user quit, False if the link
    dropped and a reconnect is worth trying.

    Split out from main() so an X_LINK_ERROR can tear the pipeline down and
    rebuild it cleanly rather than killing the process. The Myriad X stays
    booted after an unclean exit, so leaving the device open on the way out is
    what makes the NEXT run start in a stale state -- the "sometimes it works"
    behaviour. Exiting through the context managers on every path, including
    the error path, is what prevents that."""
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
                        # 1 ms yield, NOT a bare continue. A bare continue spins
                        # this loop as fast as the CPU allows -- millions of
                        # tryGet() calls per second, one core pegged, while
                        # HSEmotion, Haar, YAMNet and the SER compete for the
                        # same machine. depthai runs its XLink communication on
                        # its own host threads; starve those and the streams
                        # time out, which surfaces as X_LINK_ERROR on all four
                        # queues at once. Frames arrive every ~33 ms, so 1 ms
                        # costs nothing.
                        time.sleep(0.001)
                        # keep the window responsive even when frames stall,
                        # otherwise q only works while data is flowing
                        if cv2.waitKey(1) == ord("q"):
                            return True
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

                        # ONE frame, both video channels, same coordinate
                        # space. Audio joins as a third channel from its own
                        # thread -- latest() expires stale readings, so a
                        # stopped audio worker drops out instead of being
                        # replayed as current.
                        audio_now = audio.latest() if audio else (None, False, 0.0)
                        result = vp.process(frame, people, audio=audio_now)
                        reporter.maybe(result, audio_now,
                                       audio.tracks() if audio else [])

                        for (_box, pts) in people:
                            draw_skeleton(frame, pts)

                    if result is not None:
                        for r in result.people:
                            draw_person(frame, r, r.track_id == result.target_id)
                        draw_channels(frame, result)
                        if result.target is not None:
                            draw_bars(frame, result.target)
                    draw_audio(frame,
                               audio.latest() if audio else (None, False, 0.0),
                               audio.age if audio else float("inf"),
                               audio.tracks() if audio else [])

                    cv2.imshow("run", frame)
                    if cv2.waitKey(1) == ord("q"):
                        return True
    except KeyboardInterrupt:
        return True
    except RuntimeError as e:
        # depthai raises RuntimeError for XLink communication failures. The
        # `with` blocks above still unwind, so the device is released properly
        # before we retry -- that release is the point.
        print(f"\n[device error] {e}")
        return False
    return True


def main():
    nn_archive = dai.NNArchive(ARCHIVE_PATH)
    audio = start_audio()
    reporter = Reporter()

    try:
        attempt = 0
        while True:
            if run_session(nn_archive, audio, reporter):
                break
            attempt += 1
            if attempt >= MAX_RECONNECTS:
                print(f"giving up after {attempt} reconnects")
                break
            print(f"reconnecting ({attempt}/{MAX_RECONNECTS}) …")
            cv2.destroyAllWindows()
            time.sleep(RECONNECT_DELAY_S)
    finally:
        if audio is not None:
            audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()