import numpy as np
import librosa
import torch

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)

N_MELS = 64
N_FFT = 1024
HOP = 512
TIME_STEPS = 94


def array_to_fusion_tensor(y):
    """Mono 16 kHz float array -> 3-channel fusion tensor (3, 64, 94).

    Single source of truth for feature extraction: training (via
    wav_to_fusion_tensor) and live inference both call this, so they
    can never drift apart.
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    # CHANNEL 1: log-mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    # CHANNEL 2: pitch contour
    pitches, magnitudes = librosa.piptrack(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    pitch_contour = np.array([pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])])
    pitch_contour = (pitch_contour - pitch_contour.mean()) / (pitch_contour.std() + 1e-6)
    pitch_matrix = np.tile(pitch_contour, (N_MELS, 1))

    # CHANNEL 3: zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-6)
    zcr_matrix = np.tile(zcr, (N_MELS, 1))

    def match_time_steps(m):
        if m.shape[1] < TIME_STEPS:
            return np.pad(m, ((0, 0), (0, TIME_STEPS - m.shape[1])))
        return m[:, :TIME_STEPS]

    fusion_stack = np.stack([
        match_time_steps(logmel),
        match_time_steps(pitch_matrix),
        match_time_steps(zcr_matrix),
    ], axis=0)

    return torch.tensor(fusion_stack, dtype=torch.float32)


def wav_to_fusion_tensor(path):
    """Load a .wav and return the fusion tensor. Used by the trainer."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return array_to_fusion_tensor(y)