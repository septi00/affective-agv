"""
audio_bridge.py
Runs the audio channel on its own thread and publishes the latest SER result
for the video loop to consume.

WHY A BRIDGE AND NOT A CALL: the channels run on different clocks. The video
loop turns at ~30fps; the SER needs roughly a second of audio before it can say
anything, and Perception only classifies every CLASSIFY_EVERY blocks. Calling
feed() from the video loop would either block the display or starve the audio.
So audio runs continuously on a worker and drops its most recent answer into a
slot; the video loop reads that slot without waiting.

STALENESS IS THE POINT. One SER result covers roughly thirty video frames, so a
result that stops updating would otherwise be presented as current indefinitely
-- the audio equivalent of the stale-face-box bug. latest() therefore expires:
past max_age_s it reports available=False, the channel drops out of the weighted
sum, and fusion carries on with face and body. That is modality dropout working,
not a failure path.

WHAT THIS FILE DOES NOT DO: it does not classify. Perception owns the buffer,
the YAMNet gate, the SER, and the DOA path. This is transport and timing only,
which keeps run_live.py and run_video.py using literally the same perception
core rather than two drifting copies.

DEVICE NOTE: the ReSpeaker lives on the robot. This can only run on the CM5, not
the MacBook it cannot be tested offline.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from audio_quality import AudioQuality


def _default_to_probs6(sound):
    """PerceptionResult.sound -> 6-vector, or None.

    Non-speech blocks (YAMNet gated the window out as a door slam, music, room
    noise) legitimately carry no emotion. Returning None makes the channel
    unavailable for that window rather than inventing a distribution -- an
    absent channel is honest, a guessed one poisons the fusion."""
    probs = getattr(sound, "emotion_probs", None)
    if probs is None or not getattr(sound, "is_speech", False):
        return None
    return np.asarray(probs, dtype=float)


@dataclass
class AudioReading:
    probs: np.ndarray | None    # 6-vector in SIX order, or None
    quality: float
    stamp: float                # time.monotonic() when produced
    label: str | None = None    # for the overlay only; fusion uses probs


class AudioBridge:
    """Owns the mic stream and the SER worker. Read it with latest().

    perception: your Perception instance from noise/src/perception.py.
        Constructed by the caller so this file does not need to know its
        configuration.
    to_probs6: callable turning a PerceptionResult.sound into a 6-vector.
        Defaults to reading .emotion_probs, which perception.py fills from
        EmotionClassifier.classify_probs(). Overridable so this file does not
        have to change if the audio result grows a different shape.
    proc_ch / mic_chs: channel routing, from noise/src/config.py. ch0 is the
        ReSpeaker's beamformed output and is what the classifier must see; the
        raw mics feed SRP-PHAT. Mixing these up was the original disgust bug.
    max_age_s: how long an SER reading stays valid. Should exceed the SER
        cadence (~1s) with margin, but stay short enough that a stopped audio
        thread is noticed within a second or so.
    doa_max_age_s: same idea for DOA tracks, but much shorter -- the DOA path
        runs on EVERY block (~256 ms), not on the classify cadence, so a track
        that has not refreshed in a second is genuinely gone rather than merely
        between windows.
    """

    def __init__(self, perception, to_probs6=None, device="ReSpeaker",
                 samplerate=16000, channels=6, blocksize=4096,
                 proc_ch=0, mic_chs=(1, 2, 3, 4), max_age_s=2.5,
                 doa_max_age_s=1.0):
        self.perception = perception
        self.to_probs6 = to_probs6 or _default_to_probs6
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.proc_ch = proc_ch
        self.mic_chs = list(mic_chs)
        self.max_age_s = max_age_s
        self.doa_max_age_s = doa_max_age_s

        self._quality = AudioQuality()
        self._lock = threading.Lock()
        self._latest = AudioReading(None, 0.0, 0.0)
        self._tracks: list = []
        self._tracks_stamp = 0.0
        self._stream = None
        self._thread = None
        self._stop = threading.Event()
        self._queue: list = []
        self._queue_lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle

    def start(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            # Real-time callback: copy and leave. Any work done here blocks the
            # audio driver and shows up as input overflow and DOA stutter --
            # the lesson that moved inference onto a worker thread originally.
            if status:
                pass
            with self._queue_lock:
                if len(self._queue) < 32:          # bounded: drop rather than lag
                    self._queue.append(indata.copy())

        self._stream = sd.InputStream(
            device=self.device, channels=self.channels,
            samplerate=self.samplerate, blocksize=self.blocksize,
            dtype='float32', callback=callback)
        self._stream.start()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- worker

    def _worker(self):
        while not self._stop.is_set():
            block = None
            with self._queue_lock:
                if self._queue:
                    block = self._queue.pop(0)
            if block is None:
                time.sleep(0.005)
                continue

            mics = block[:, self.mic_chs].T.astype(np.float32)   # (4, N) for SRP
            proc = block[:, self.proc_ch].astype(np.float32)     # ch0, beamformed

            # Quality is updated on EVERY block, not only when a classification
            # fires. The noise floor is estimated from history, so sampling it
            # at the block rate (~256 ms) rather than the classify rate (~1 s)
            # gives four times the evidence and tracks the room faster.
            quality = self._quality.update(proc)

            try:
                result = self.perception.feed(mics, proc)
            except Exception as e:
                print("audio perception error:", e)
                continue

            # sound is None on non-classify blocks -- that is the cadence, not
            # a dropout, so the previous reading stays valid until it expires
            # DOA refreshes on EVERY block; the SER only every CLASSIFY_EVERY.
            # Store tracks BEFORE the sound check or bearings would only ever
            # update on classify blocks -- a quarter of the real DOA rate.
            with self._lock:
                self._tracks = list(getattr(result, "tracks", None) or [])
                self._tracks_stamp = time.monotonic()

            sound = getattr(result, "sound", None)
            if sound is None:
                continue

            probs = self.to_probs6(sound)

            with self._lock:
                self._latest = AudioReading(
                    probs=probs,
                    quality=quality,
                    stamp=time.monotonic(),
                    label=getattr(sound, "emotion", None))

    # ----------------------------------------------------------------- read

    def latest(self):
        """Returns (probs6, available, quality) -- the tuple
        VideoPerception.process(audio=...) expects.

        available is False when nothing has been classified yet, when the last
        reading has expired, or when the SER could not produce a vector."""
        with self._lock:
            r = self._latest

        if r.probs is None:
            return None, False, 0.0
        if time.monotonic() - r.stamp > self.max_age_s:
            return None, False, 0.0
        return r.probs, True, r.quality

    def tracks(self) -> list:
        """Confirmed DOA tracks, or [] if stale. Each has .id, .angle_deg,
        .std_deg, .age, .misses (TrackView from perception.py).

        BEARINGS ARE IN THE MIC ARRAY'S FRAME: 0 degrees is the ReSpeaker's
        silkscreen mark, not the camera's optical axis. Comparing these against
        a person's position on screen needs the mounting offset between the
        two -- see fusion/bearing.py, which does exactly that comparison, and
        video_perception.py's `doa_tracks` parameter, which is what actually
        attributes audio to a specific tracked person instead of to the scene."""
        with self._lock:
            if not self._tracks:
                return []
            if time.monotonic() - self._tracks_stamp > self.doa_max_age_s:
                return []
            return list(self._tracks)

    @property
    def age(self) -> float:
        """Seconds since the last classification. For the overlay -- watching
        this climb past max_age_s as the room goes quiet is the clearest way to
        see the audio channel drop out."""
        with self._lock:
            stamp = self._latest.stamp
        return float('inf') if stamp == 0.0 else time.monotonic() - stamp