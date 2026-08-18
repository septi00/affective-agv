import numpy as np

# --- Array geometry -------------------------------------------------------
# Mics at corners of a square inscribed in a ~60mm circle on the board.
# Frame: +X toward DOA 0° silkscreen mark, +Y toward DOA 90°, +Z out of board.
# Columns are mic1, mic2, mic3, mic4 — matching firmware channel order.

r = 0.032                       # mic distance from board center, meters
d = r * np.cos(np.pi / 4)       # = r * sqrt(2)/2  ≈ 0.02121

MIC_POS = np.array([
    [ +d, -d, -d, +d ],   # x:  mic1 NE, mic2 NW, mic3 SW, mic4 SE
    [ +d, +d, -d, -d ],   # y
    [  0,  0,  0,  0 ],   # z (planar)
])

# --- Audio capture --------------------------------------------------------
FS         = 16000                              # sample rate (must match firmware)
BLOCK      = 4096                              # 256 ms of audio per DOA estimate
NFFT       = 512                               # ~32 ms STFT window
DEVICE     = "ReSpeaker"                       # substring match for sounddevice
TOTAL_CH   = 6                                 # 6-ch firmware

# Channel routing — the ONE place channel numbers live.
#   ch0  = board-processed (beamformed + denoised) -> emotion/sound classifiers
#   ch1..ch4 = raw mics                            -> SRP-PHAT DOA
#   ch5  = playback reference (unused here)
PROC_CH    = 0                                 # processed channel for the SER
MIC_CHS    = [1, 2, 3, 4]                      # raw mics for DOA

AZ_GRID    = np.deg2rad(np.arange(0, 360, 2))  # 2° resolution
RMS_GATE   = 0.001                             # skip silent blocks (float32, -1..1)

# Real-time safety: audio callback drops blocks into a bounded queue that a
# worker thread drains. 8 blocks ≈ 2 s of slack to absorb classify spikes.
QUEUE_MAXSIZE = 8

# --- DOA tracking ---------------------------------------------------------
# Deployment: don't report tracks the filter isn't confident about. A track
# whose angle std exceeds this is still tracked internally but hidden from
# output (it's usually a half-formed or dying track).
TRACK_STD_MAX_DEG = 30.0

# --- Sound classification -------------------------------------------------
# Buffer fills every block; classify every CLASSIFY_EVERY blocks.
# 4 blocks × 256 ms ≈ 1.02 s, which matches YAMNet's 0.975 s window.
CLASSIFY_EVERY = 4
SER_FRAME      = FS * 3    # 3 s buffer for SER (matches training)
SPEECH_THRESH  = 0.30      # tune on your own recordings
# DOA keeps tracking for this many classify windows after the last speech hit.
# 3 windows × ~1 s/window = ~3 s of tolerance for natural pauses.
SPEECH_HOLD_WINDOWS = 3
