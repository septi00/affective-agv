#!/usr/bin/env python3
"""
Live SRP-PHAT DOA estimation from the ReSpeaker Mic Array v3.0.
Reads channels 1..4 (raw mics) from the 6-channel firmware,
runs SRP-PHAT every block, prints azimuth in degrees.
On speech, also runs the ResNet18 SER and prints the emotion.
"""
import os
from collections import deque

import numpy as np
import pyroomacoustics as pra
import sounddevice as sd

from config import (
    AZ_GRID, BLOCK, CLASSIFY_EVERY, DEVICE, FS,
    MIC_CHS, MIC_POS, NFFT, RMS_GATE, SER_FRAME,
    SPEECH_THRESH, TOTAL_CH,
)
from multi_tracker import MultiTracker
from peak_picking import pick_peaks
from sound_classifier import EmotionClassifier, SoundClassifier, YAMNET_FRAME


def list_devices():
    print(sd.query_devices())


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))

    doa = pra.doa.srp.SRP(
        L=MIC_POS,
        fs=FS,
        nfft=NFFT,
        azimuth=AZ_GRID,
        num_src=1,
        mode="far",
    )

    clf = SoundClassifier(
        os.path.join(HERE, "../models/yamnet.tflite"),
        os.path.join(HERE, "../models/yamnet_class_map.csv"),
        top_k=3,
    )
    ser = EmotionClassifier(os.path.join(HERE, "../models/emotion_fusion_resnet18_merged.pt"))
    tracker = MultiTracker(dt=BLOCK / FS)

    buf = deque(maxlen=SER_FRAME)  # rolling mono audio, auto-drops past 3 s
    classify_count = 0

    def callback(indata, frames, t, status):
        nonlocal classify_count

        if status:
            print(f"[stream] {status}")

        sig = indata[:, MIC_CHS].T.astype(np.float32)

        # --- Sound / emotion path -------------------------------------------
        mono = sig.mean(axis=0)
        buf.extend(mono)
        classify_count += 1
        if classify_count >= CLASSIFY_EVERY and len(buf) >= YAMNET_FRAME:
            classify_count = 0
            window = np.array(buf, dtype=np.float32)
            if clf.speech_score(window) >= SPEECH_THRESH:
                emotion, conf = ser.classify(window)
                print(f"[emotion] {emotion} ({conf:.2f})")
            else:
                label, score = clf.classify(window)[0]
                print(f"[sound] {label} ({score:.2f})")

        # --- DOA path -------------------------------------------------------
        rms = float(np.sqrt(np.mean(sig**2)))
        if rms < RMS_GATE:
            tracker.step([])
            return

        X = pra.transform.stft.analysis(sig.T, NFFT, NFFT // 2)
        X = X.transpose(2, 1, 0)
        doa.locate_sources(X)

        detections = pick_peaks(doa.grid.azimuth, doa.grid.values)
        confirmed = tracker.step(detections)

        if not confirmed:
            return

        parts = [
            f"T{t.id}: {np.rad2deg(t.angle):6.1f}° ±{np.rad2deg(t.angle_std):.1f}° "
            f"(age={t.age}, miss={t.misses})"
            for t in confirmed
        ]
        print("  ".join(parts))

    with sd.InputStream(
        device=DEVICE,
        channels=TOTAL_CH,
        samplerate=FS,
        blocksize=BLOCK,
        dtype="float32",
        callback=callback,
    ):
        print("Listening — Ctrl+C to stop.")
        print(f"Frame size: {BLOCK} samples = {1000*BLOCK/FS:.0f} ms\n")
        try:
            sd.sleep(10**9)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
