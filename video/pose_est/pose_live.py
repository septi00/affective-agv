import cv2
import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from pose_features import PoseFeatures

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


def main():
    nn_archive = dai.NNArchive(ARCHIVE_PATH)
    pf = None  # initialised after camera intrinsics are read

    try:
        with dai.Device() as device:
            calib = device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_W, NN_H)
            fx, fy = intrinsics[0][0], intrinsics[1][1]
            cx, cy = intrinsics[0][2], intrinsics[1][2]
            cam_intrinsics = (fx, fy, cx, cy)
            pf = PoseFeatures(intrinsics=cam_intrinsics)

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

                last_txt = None

                while True:
                    frame_data = q_frame.tryGet()
                    if frame_data is None:
                        continue

                    frame      = frame_data.getCvFrame()
                    nn_data    = q_nn.tryGet()
                    depth_data = q_depth.tryGet()

                    depth_frame = depth_data.getFrame() if depth_data is not None else None

                    if nn_data is not None:
                        for det in nn_data.detections:
                            x1 = int(det.xmin * NN_W)
                            y1 = int(det.ymin * NN_H)
                            x2 = int(det.xmax * NN_W)
                            y2 = int(det.ymax * NN_H)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)

                            pts = {}
                            for i, kp in enumerate(det.getKeypoints()):
                                px = int(kp.imageCoordinates.x * NN_W)
                                py = int(kp.imageCoordinates.y * NN_H)
                                conf = kp.confidence

                                if conf <= CONF_THRESH:
                                    continue

                                z_m = None
                                if depth_frame is not None:
                                    px_c = np.clip(px, 0, NN_W - 1)
                                    py_c = np.clip(py, 0, NN_H - 1)
                                    z_mm = depth_frame[py_c, px_c]
                                    if z_mm > 0:
                                        z_m = z_mm / 1000.0
                                        x3d = (px - cx) * z_m / fx
                                        y3d = (py - cy) * z_m / fy
                                        pts[i] = (px, py, conf, x3d, y3d, z_m)
                                        continue
                                pts[i] = (px, py, conf, None, None, None)

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
                                z_m = pt[5]
                                cv2.circle(frame, (px, py), 4, KEYPOINT_COLORS[i], -1)
                                if z_m is not None:
                                    cv2.putText(frame, f"{z_m:.2f}m", (px + 5, py),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                            op, act = pf.update(pts)
                            if op is not None:
                                last_txt = f"open {op:.2f}" + (f"  act {act:.2f}" if act is not None else "")

                    if last_txt is not None:
                        cv2.putText(frame, last_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("Pose", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
