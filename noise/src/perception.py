"""
Perception pipeline: DOA tracking + speech-gated emotion/sound classification.

This module is deliberately I/O-free and device-agnostic. It does NOT touch
sounddevice, does NOT print, and does NOT know channel numbers. You feed it
per-block audio already split into:

    mics : (4, N) float32  -- the four RAW mics (ch1..ch4), for SRP-PHAT
    proc : (N,)  float32   -- the single PROCESSED channel (ch0), for the
                              speech / emotion classifiers

and it returns a PerceptionResult. Whoever calls feed() decides what to do with
that result: print it now, publish it to a ROS2 topic later, etc. That is the
whole point of keeping this file clean -- the live runner and a future ROS2
node are interchangeable front-ends over the same perception core.

Why the two inputs differ (defense-relevant):
  * SRP-PHAT needs inter-mic TIME DELAYS, so it must see the raw 4-mic array.
    A single channel carries no delay information.
  * The SER model was trained on clean single-channel speech, so it must see
    the board's beamformed/denoised channel (ch0) to match its training
    distribution. Feeding it a raw 4-mic downmix is a train/inference mismatch
    -- same audio stream, two different consumers, two different slices.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyroomacoustics as pra

from config import (
    AZ_GRID, CLASSIFY_EVERY, FS, MIC_POS, NFFT, RMS_GATE,
    SER_FRAME, SPEECH_HOLD_WINDOWS, SPEECH_THRESH, TRACK_STD_MAX_DEG,
)
from multi_tracker import MultiTracker
from peak_picking import pick_peaks
from sound_classifier import (
    CLASS_NAMES, EmotionClassifier, SoundClassifier, YAMNET_FRAME,
)


@dataclass
class TrackView:
    """A confirmed DOA track, ready to report (angles already in degrees)."""
    id: int
    angle_deg: float
    std_deg: float
    age: int
    misses: int


@dataclass
class SoundResult:
    """Outcome of one classification. Either an emotion (speech) or a sound."""
    is_speech: bool
    emotion: str | None = None
    emotion_conf: float | None = None
    emotion_probs: np.ndarray | None = None   # 6-vector, CLASS_NAMES order
    sound_label: str | None = None
    sound_conf: float | None = None


@dataclass
class PerceptionResult:
    """What feed() returns for one audio block."""
    tracks: list = field(default_factory=list)   # list[TrackView]
    sound: SoundResult | None = None             # None unless a classify fired


class Perception:
    """
    Owns the DOA stack (SRP + tracker) and the classifiers (YAMNet + SER),
    plus the rolling audio buffer and classify cadence.

    Single entry point: feed(mics, proc) per block -> PerceptionResult.
    """

    def __init__(self, yamnet_path, yamnet_labels, ser_path, block, device="cpu"):
        self.doa = pra.doa.srp.SRP(
            L=MIC_POS, fs=FS, nfft=NFFT, azimuth=AZ_GRID,
            num_src=1, mode="far",
        )
        self.tracker = MultiTracker(dt=block / FS)
        self.clf = SoundClassifier(yamnet_path, yamnet_labels, top_k=3)
        self.ser = EmotionClassifier(ser_path, device=device)

        self._buf = deque(maxlen=SER_FRAME)   # rolling ch0 mono, ~3 s
        self._since_classify = 0
        self._speech_hold = 0   # classify windows remaining before DOA stops tracking

    def feed(self, mics, proc) -> PerceptionResult:
        """Process one block. mics:(4,N) raw mics, proc:(N,) processed ch0."""
        result = PerceptionResult()

        # --- speech / emotion path (ch0) ------------------------------------
        self._buf.extend(proc)
        self._since_classify += 1
        if self._since_classify >= CLASSIFY_EVERY and len(self._buf) >= YAMNET_FRAME:
            self._since_classify = 0
            window = np.array(self._buf, dtype=np.float32)
            result.sound = self._classify(window)
            if result.sound.is_speech:
                self._speech_hold = SPEECH_HOLD_WINDOWS
            elif self._speech_hold > 0:
                self._speech_hold -= 1

        # --- DOA path (raw mics) --------------------------------------------
        result.tracks = self._update_doa(mics)
        return result

    # -- internals -----------------------------------------------------------

    def _classify(self, window) -> SoundResult:
        if self.clf.speech_score(window) >= SPEECH_THRESH:
            # classify_probs, not classify: late fusion needs the whole
            # distribution. An argmax label cannot be weighted against the face
            # channel or multiplied by the body prior -- reducing here would
            # throw away the only thing fusion can actually use, and the vector
            # is computed either way. emotion/emotion_conf are still filled so
            # run_live.py and any existing consumer are unaffected.
            probs = self.ser.classify_probs(window)
            i = int(probs.argmax())
            return SoundResult(is_speech=True,
                               emotion=CLASS_NAMES[i],
                               emotion_conf=float(probs[i]),
                               emotion_probs=probs)
        label, score = self.clf.classify(window)[0]
        return SoundResult(is_speech=False, sound_label=label, sound_conf=score)

    def _update_doa(self, mics) -> list:
        rms = float(np.sqrt(np.mean(mics ** 2)))
        if rms < RMS_GATE or self._speech_hold == 0:
            self.tracker.step([])
            return []

        X = pra.transform.stft.analysis(mics.T, NFFT, NFFT // 2)
        X = X.transpose(2, 1, 0)
        self.doa.locate_sources(X)

        detections = pick_peaks(self.doa.grid.azimuth, self.doa.grid.values)
        confirmed = self.tracker.step(detections)

        views = []
        for t in confirmed:
            std_deg = float(np.rad2deg(t.angle_std))
            if std_deg > TRACK_STD_MAX_DEG:   # hide low-confidence tracks
                continue
            views.append(TrackView(
                id=t.id,
                angle_deg=float(np.rad2deg(t.angle)),
                std_deg=std_deg,
                age=t.age,
                misses=t.misses,
            ))
        return views