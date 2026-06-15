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


def wav_to_fusion_tensor(path):
    """Load a .wav file and return a 3-channel feature fusion tensor, shape (3, 64, 94)."""

    # 1. Load raw audio
    y, _ = librosa.load(path, sr=SR, mono=True)

    # Force a uniform 3-second audio length
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

        # CHANNEL 1: Compute Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    # CHANNEL 2: Extract Pitch Contour
    pitches, magnitudes = librosa.piptrack(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    pitch_contour = np.array([pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])])
    pitch_contour = (pitch_contour - pitch_contour.mean()) / (pitch_contour.std() + 1e-6)
    pitch_matrix = np.tile(pitch_contour, (N_MELS, 1))

    # CHANNEL 3: Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP)
    zcr_contour = zcr[0]
    zcr_contour = (zcr_contour - zcr_contour.mean()) / (zcr_contour.std() + 1e-6)
    zcr_matrix = np.tile(zcr_contour, (N_MELS, 1))

    # Strict spatial dimension alignment function
    def match_time_steps(matrix):
        if matrix.shape[1] < TIME_STEPS:
            return np.pad(matrix, ((0, 0), (0, TIME_STEPS - matrix.shape[1])))
        else:
            return matrix[:, :TIME_STEPS]

    logmel = match_time_steps(logmel)
    pitch_matrix = match_time_steps(pitch_matrix)
    zcr_matrix = match_time_steps(zcr_matrix)

    # Stack channels-first for PyTorch Conv2D -> Final Shape: (3, 64, 94)
    fusion_stack = np.stack([logmel, pitch_matrix, zcr_matrix], axis=0)

    return torch.tensor(fusion_stack, dtype=torch.float32)