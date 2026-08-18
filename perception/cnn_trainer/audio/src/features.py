# features.py — turns a .wav file into a log-mel spectrogram for the CNN.

import numpy as np
import librosa

SR = 16000                          # match the robot's mic
DURATION = 3.0                      # seconds per clip; fits CREMA-D / C-BESD avg length
N_SAMPLES = int(SR * DURATION)      # 48000 samples per clip

N_MELS = 64                         # frequency rows (standard for speech)
N_FFT = 1024                        # FFT window size
HOP = 512                           # window step (50% overlap)
TIME_STEPS = 1 + N_SAMPLES // HOP   # time columns -> spectrogram is (64, 94)


def wav_to_logmel(path):
    """Load a .wav file and return its log-mel spectrogram, shape (64, 94)."""

    # load audio, force 16 kHz mono; we ignore the returned sample rate
    y, _ = librosa.load(path, sr=SR, mono=True)

    # force exactly 48000 samples — CNN needs a fixed input size
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))   # too short: pad with silence
    else:
        y = y[:N_SAMPLES]                        # too long: cut off the end

    # compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)

    logmel = librosa.power_to_db(mel, ref=np.max) # energy to db
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6) # normalization

    # make sure time axis is exactly TIME_STEPS columns
    if logmel.shape[1] < TIME_STEPS:
        logmel = np.pad(logmel, ((0, 0), (0, TIME_STEPS - logmel.shape[1])))
    else:
        logmel = logmel[:, :TIME_STEPS]

    return logmel.astype(np.float32)[..., np.newaxis]

