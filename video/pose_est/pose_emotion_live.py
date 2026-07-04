"""
pose_emotion_live.py
Watch the body channel work end to end: pose -> openness + activity (mean/burst)
-> grid cell -> soft emotion prior, drawn on screen so you can see it react.

Same OAK pipeline as pose_live.py, plus the emotion mapping overlaid. Move
around and watch: open up -> "open", gesture hard -> "active", and the boosted
emotions light up.

Needs pose_features.py and pose_emotion.py in the same folder.
"""
import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from pose_features import PoseFeatures
from pose_emotion import (body_prior, SIX, OPEN_THRESH, BURST_THRESH, MEAN_THRESH,
                          GRID)

ARCHIVE_PATH = "/Users/septi/dis_ws/video/pose_est/models/yolov8n-pose.rvc2.tar.xz"
CONF_THRESH = 0.3
NN_W, NN_H = 512, 288

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def main():
    nn_archive = dai.NNArchive(ARCHIVE_PATH)
    pf = None
    try:
        with dai.Device() as device:
            calib = device.readCalibration()
            intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_W, NN_H)
            fx, fy, cx, cy = intr[0][0], intr[1][1], intr[0][2], intr[1][2]
            pf = PoseFeatures(intrinsics=(fx, fy, cx, cy))

            with dai.Pipeline(device) as pipeline:
                cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                cam_l = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
                cam_r = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
                stereo = pipeline.create(dai.node.StereoDepth)
                cam_l.requestFullResolutionOutput().link(stereo.left)
                cam_r.requestFullResolutionOutput().link(stereo.right)
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                stereo.setOutputSize(NN_W, NN_H)
                nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(cam_rgb, nn_archive)
                nn.input.setBlocking(False); nn.input.setMaxSize(1)
                q_frame = nn.passthrough.createOutputQueue(maxSize=2, blocking=False)
                q_nn = nn.out.createOutputQueue(maxSize=2, blocking=False)
                q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
                pipeline.start()
                print("Running — press q to quit")

                info = {}
                while True:
                    fd = q_frame.tryGet()
                    if fd is None:
                        continue
                    frame = fd.getCvFrame()
                    nn_data = q_nn.tryGet()
                    dd = q_depth.tryGet()
                    depth = dd.getFrame() if dd is not None else None

                    if nn_data is not None:
                        for det in nn_data.detections:
                            pts = {}
                            for i, kp in enumerate(det.getKeypoints()):
                                px = int(kp.imageCoordinates.x * NN_W)
                                py = int(kp.imageCoordinates.y * NN_H)
                                if kp.confidence <= CONF_THRESH:
                                    continue
                                x3d = y3d = z_m = None
                                if depth is not None:
                                    zc = depth[int(np.clip(py,0,NN_H-1)), int(np.clip(px,0,NN_W-1))]
                                    if zc > 0:
                                        z_m = zc/1000.0
                                        x3d = (px-cx)*z_m/fx; y3d = (py-cy)*z_m/fy
                                pts[i] = (px, py, kp.confidence, x3d, y3d, z_m)

                            for a, b in SKELETON:
                                if a in pts and b in pts:
                                    cv2.line(frame, pts[a][:2], pts[b][:2], (0,255,0), 2)
                            for i, p in pts.items():
                                cv2.circle(frame, p[:2], 3, (0,200,255), -1)

                            op, mean, burst = pf.update(pts)
                            if op is not None and mean is not None and burst is not None:
                                o = 'open' if op > OPEN_THRESH else 'closed'
                                hot = (burst > BURST_THRESH) + (mean > MEAN_THRESH)
                                a = 'active' if hot >= 1 else 'calm'
                                boosted = GRID.get((o, a), []) if a == 'active' or True else []
                                boosted = GRID.get((o, a), [])
                                prior = body_prior(op, mean, burst)
                                info = {'op':op,'mean':mean,'burst':burst,'o':o,'a':a,
                                        'hot':hot,'boosted':boosted,'prior':prior}

                    # ---- overlay panel ----
                    if info:
                        y = 22
                        cv2.putText(frame, f"open {info['op']:.2f} [{info['o']}]  mean {info['mean']:.2f}  burst {info['burst']:.2f}",
                                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2); y += 24
                        cv2.putText(frame, f"cell: {info['o'].upper()} + {info['a'].upper()}  (energy {info['hot']}/2)",
                                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2); y += 22
                        cv2.putText(frame, f"-> boosting: {', '.join(info['boosted']) or 'nothing'}",
                                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 2); y += 26
                        # per-class prior bars
                        for c, v in zip(SIX, info['prior']):
                            col = (0,255,0) if v > 1.0 else (150,150,150)
                            cv2.putText(frame, f"{c:8} {v:.2f}", (8, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                            cv2.rectangle(frame, (150, y-10), (150+int((v-1)*180), y-2), col, -1)
                            y += 18

                    cv2.imshow("Body emotion", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()