import cv2
import depthai as dai
import sys


def main():
    print("Starting example based on luxonis/oak-examples V3 patterns...")

    # 1. Establish the Device context first (V3 standard)
    with dai.Device() as device:

        # 2. Open the Pipeline using the device hook context
        with dai.Pipeline(device) as pipeline:

            # 3. Create the unified V3 Camera node
            cam = pipeline.create(dai.node.Camera).build()

            # Request output with dimensions optimized for the Zoo architecture (512x288)
            preview = cam.requestOutput((512, 288), type=dai.ImgFrame.Type.RGB888p)

            # 4. Initialize NeuralNetwork using the exact string descriptor pattern
            model_path = "luxonis/yolov8-nano-pose-estimation:coco-512x288"
            pose_nn = pipeline.create(dai.node.NeuralNetwork).build(cam, model_path)

            # 5. Connect streams via output queue endpoints
            q_rgb = preview.createOutputQueue(maxSize=2, blocking=False)
            q_nn = pose_nn.out.createOutputQueue(maxSize=2, blocking=False)

            # 6. Start pipeline processing on the device
            pipeline.start()
            print("Pipeline successfully deployed to OAK-D Lite. Streaming...")

            while True:
                # Retrieve packets from the queues
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()

                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

                    if in_nn is not None:
                        # FIXED: In V3, extract the raw multi-dimensional matrix as a NumPy array
                        tensor_data = in_nn.getFirstTensor()

                        # Visual notification that the model layers are actively streaming
                        cv2.putText(
                            frame,
                            f"NN Shape: {tensor_data.shape}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )

                    # Display frame output
                    cv2.imshow("OAK-D Lite - luxonis/oak-examples Pattern", frame)

                # Exit gracefully on 'q'
                if cv2.waitKey(1) == ord('q'):
                    break


if __name__ == "__main__":
    main()