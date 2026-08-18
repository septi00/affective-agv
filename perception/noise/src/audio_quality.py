"""
audio_quality.py
Single job: turn an audio window into one scalar quality score in [0,1] --
how much to trust this SER reading.

Audio counterpart of face_quality.py, and deliberately the same shape: a few
cheap extrinsic signals, saturating maps, averaged. Extrinsic means it measures
the INPUT, not the model's opinion of it.

WHY NOT JUST USE emotion_conf: because a confidently wrong prediction scores
high on confidence. That is not hypothetical here -- when the SER was fed the
raw 4-mic downmix instead of ch0, it returned "disgust" on everything with high
confidence, because the input was out of distribution and the softmax had no
way to know. Posterior confidence cannot detect bad input; SNR can. Both signals
exist in late_fusion for exactly that reason, and this file supplies the
extrinsic half.

Two signals, MULTIPLIED not averaged:
  - SNR against a rolling noise floor: quiet speech far from the mic carries
    less signal than close speech, independent of what was said
  - clipping: a saturated window is distorted regardless of how loud it is,
    and distortion is out-of-distribution for a model trained on clean clips

Multiplied because clipping is a gate on validity, not a peer measurement.
Averaging them gave room tone a score of 0.5 -- silence is clean, so the clip
term voted 1.0 and dragged a zero-SNR window up to the middle. A distorted
window is untrustworthy however loud it is, and a silent one is untrustworthy
however clean, so both terms must be able to veto.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# SNR in dB at which quality saturates to 1.0. Conversational speech at ~1m on
# the ReSpeaker sits around 20-25 dB above room tone; beyond that, more SNR does
# not make the reading more trustworthy.
_SNR_SATURATE_DB = 20.0

# Below this, treat the window as noise rather than speech.
_SNR_FLOOR_DB = 3.0

# Fraction of samples at full scale before a window counts as clipped.
_CLIP_FRAC_BAD = 0.01


def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _snr_score(window_rms: float, noise_floor: float) -> float:
    if noise_floor <= 1e-9 or window_rms <= 1e-9:
        return 0.0
    snr_db = 20.0 * np.log10(window_rms / noise_floor)
    if snr_db <= _SNR_FLOOR_DB:
        return 0.0
    return float(np.clip((snr_db - _SNR_FLOOR_DB) /
                         (_SNR_SATURATE_DB - _SNR_FLOOR_DB), 0.0, 1.0))


def _clip_score(window) -> float:
    """1.0 = clean, 0.0 = heavily clipped."""
    x = np.abs(np.asarray(window, dtype=float))
    if x.size == 0:
        return 0.0
    peak = x.max()
    if peak <= 0:
        return 0.0
    # samples within 1% of this window's peak AND near full scale
    frac = float(np.mean(x >= 0.99)) if peak >= 0.99 else 0.0
    return float(np.clip(1.0 - frac / _CLIP_FRAC_BAD, 0.0, 1.0))


class AudioQuality:
    """Stateful because the noise floor has to be learned from the room.

    Keeps a rolling history of window RMS and takes a low percentile as the
    floor. A percentile rather than a minimum: the minimum latches onto a single
    anomalously quiet block and never recovers, which would inflate SNR for the
    rest of the run.

    history: number of past windows the floor is estimated over. At the ~1s SER
        cadence, 30 is about half a minute of room tone -- long enough to be
        stable, short enough to follow the robot moving between rooms.
    """

    def __init__(self, history: int = 30, floor_percentile: float = 20.0):
        self._rms_hist = deque(maxlen=history)
        self.floor_percentile = floor_percentile

    def update(self, window) -> float:
        """window: 1-D float samples for this SER window (ch0, the beamformed
        channel -- the same signal the classifier sees, so quality describes the
        actual classifier input rather than a different microphone).

        Returns quality in [0,1]."""
        w = np.asarray(window, dtype=float)
        r = rms(w)
        self._rms_hist.append(r)

        # need some history before an SNR claim means anything; until then be
        # neutral rather than confidently wrong in either direction
        if len(self._rms_hist) < 5:
            return 0.5

        floor = float(np.percentile(self._rms_hist, self.floor_percentile))
        return float(_snr_score(r, floor) * _clip_score(w))

    def reset(self):
        self._rms_hist.clear()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q = AudioQuality()

    print("room tone only:")
    for _ in range(8):
        print(f"  {q.update(rng.normal(0, 0.01, 16000)):.2f}")

    print("speech-level window:")
    print(f"  {q.update(rng.normal(0, 0.15, 16000)):.2f}")

    print("clipped window:")
    clipped = np.clip(rng.normal(0, 2.0, 16000), -1.0, 1.0)
    print(f"  {q.update(clipped):.2f}")
