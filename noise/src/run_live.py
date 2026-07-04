#!/usr/bin/env python3
"""
Live runner for the perception pipeline (replaces the old srp_phat.py).

Everything hardware- and I/O-specific lives HERE and only here:
  * open the ReSpeaker audio stream
  * split each 6-ch block into raw mics (ch1..ch4) and processed ch0
  * hand blocks to a worker thread so the real-time audio callback never
    blocks on ML inference (STFT + SRP + YAMNet + ResNet18)
  * print results

To move to ROS2 later, swap this file for a node that publishes
perception.feed(...) results to topics. perception.py does not change.
"""
import os
import queue
import threading

import numpy as np
import sounddevice as sd

from config import BLOCK, DEVICE, FS, MIC_CHS, PROC_CH, QUEUE_MAXSIZE, TOTAL_CH
from perception import Perception


def _report(result):
    """The only place that turns a PerceptionResult into stdout."""
    if result.sound is not None:
        s = result.sound
        if s.is_speech:
            print(f"[emotion] {s.emotion} ({s.emotion_conf:.2f})")
        else:
            print(f"[sound] {s.sound_label} ({s.sound_conf:.2f})")

    if result.tracks:
        parts = [
            f"T{t.id}: {t.angle_deg:6.1f}° ±{t.std_deg:.1f}° "
            f"(age={t.age}, miss={t.misses})"
            for t in result.tracks
        ]
        print("  ".join(parts))


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    perception = Perception(
        yamnet_path=os.path.join(HERE, "../models/yamnet.tflite"),
        yamnet_labels=os.path.join(HERE, "../models/yamnet_class_map.csv"),
        ser_path=os.path.join(HERE, "../models/emotion_fusion_resnet18_merged.pt"),
        block=BLOCK,
    )

    q: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop = threading.Event()
    stats = {"dropped": 0}

    # --- audio thread: stays real-time safe, only copies + enqueues ---------
    def audio_cb(indata, frames, t, status):
        if status:
            print(f"[stream] {status}")
        try:
            q.put_nowait(indata.copy())   # copy: sounddevice reuses its buffer
        except queue.Full:
            stats["dropped"] += 1         # worker behind -> drop, stay real-time

    # --- worker thread: does all the heavy lifting off the audio thread -----
    def worker():
        while not stop.is_set():
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue
            mics = block[:, MIC_CHS].T.astype(np.float32)   # (4, N) raw mics
            proc = block[:, PROC_CH].astype(np.float32)     # (N,)  processed ch0
            _report(perception.feed(mics, proc))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    with sd.InputStream(
        device=DEVICE, channels=TOTAL_CH, samplerate=FS,
        blocksize=BLOCK, dtype="float32", callback=audio_cb,
    ):
        print("Listening — Ctrl+C to stop.")
        print(f"Block: {BLOCK} samples = {1000 * BLOCK / FS:.0f} ms\n")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass

    stop.set()
    th.join(timeout=2.0)
    if stats["dropped"]:
        print(f"\n[warn] dropped {stats['dropped']} block(s) — worker fell behind")


if __name__ == "__main__":
    main()
