"""
fuse_crema_d_confidence.py
Confidence-weighted late fusion on CREMA-D: face (HSEmotion) + audio (your SER).

Same test spine as fuse_crema_d.py -- scores FACE-only, AUDIO-only, and FUSED on
the SAME clips against the SAME ground truth. The ONLY change from the plain
average (method 1) is the fusion step:

    method 1:  fused = a * face + (1 - a) * audio          (fixed a, swept on test)
    method 4:  fused = (w_face * face + w_audio * audio) / (w_face + w_audio)
               w_face  = r_face  * available_face  * quality_face
               w_audio = r_audio * available_audio * quality_audio

WHY THIS IS DEFENSIBLE, IN ONE BREATH:
  - Still a LINEAR pool (a mixture) -> neither channel can veto the other, which
    is what you want when face (~0.55) is your weak channel.
  - r_* is a fixed RELIABILITY PRIOR taken from validated unimodal accuracy
    (audio > face). It is set a priori and is NOT searched to maximise the fused
    score -- so, unlike the alpha sweep, there is no free parameter fit on the
    test set. (Ideally r_* comes from a split disjoint from this test set; here
    it is anchored to the unimodal CREMA-D validation numbers -- note as a
    limitation.)
  - available in {0,1} gives modality dropout for free: camera dies -> face
    weight 0 -> the formula collapses to audio, no special-casing.
  - quality in [0,1] is per-clip certainty (1 - normalised entropy): a flat,
    unsure distribution down-weights itself; a peaked one earns its weight.
    For face, certainty is scaled by frame coverage (fraction of sampled frames
    that yielded a detectable face) -- the honest "how much visual evidence did
    we actually get" signal. Audio has no offline coverage analog (SNR would go
    here in the live version).

No alpha to sweep -> nothing tuned on test -> the single fused number is honest.

Run from a folder where features_fusion.py and face_features.py are importable.
Set CONFIG paths first.
    pip install hsemotion-onnx opencv-python torch torchvision librosa numpy scikit-learn matplotlib
"""
import urllib.request                       # hsemotion-onnx 0.3.1 download fix
import os, csv, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "video", "face_rec", "src"))
sys.path.insert(0, os.path.join(_BASE, "cnn_trainer", "audio", "src"))
from face_features import crop_face_rgb
from features_fusion import wav_to_fusion_tensor

# ---------------------------------------------------------------- CONFIG
CREMA       = os.path.join(_BASE, "cnn_trainer", "datasets", "crema-d")
VIDEO_DIR   = os.path.join(CREMA, "VideoFlash")
AUDIO_DIR   = os.path.join(CREMA, "AudioWAV")
SUMMARY_CSV = os.path.join(CREMA, "processedResults", "summaryTable.csv")
SER_WEIGHTS = os.path.join(_BASE, "cnn_trainer", "audio", "models", "emotion_fusion_resnet18_merged.pt")
FER_MODEL   = "enet_b0_8_best_vgaf"
FRAMES_PER_CLIP = 3
GT_MODE     = "multimodal"      # "multimodal" | "intended" | "facevote" | "voicevote"

# Reliability prior r_*: fixed a priori from validated unimodal CREMA-D accuracy.
# NOT tuned to maximise the fused score. Audio earns more weight because it is
# the stronger validated channel. Change these ONLY to match your reported
# unimodal numbers -- never to chase a better fused result.
RELIABILITY = {"face": 0.548, "audio": 0.738}

LIMIT       = None              # set 50 for a smoke run
OUT_DIR     = os.path.join(_BASE, "video", "face_rec", "results")
# -----------------------------------------------------------------------

SIX = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
VOTE = {'A':'anger','D':'disgust','F':'fear','H':'happy','N':'neutral','S':'sad'}
FILENAME = {'ANG':'anger','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
GT_COL = {"multimodal":"MultiModalVote", "facevote":"FaceVote", "voicevote":"VoiceVote"}

# HSEmotion 8-class -> the 6 indices that map to your order [anger,disgust,fear,happy,neutral,sad]
HSE_KEEP = [0, 2, 3, 4, 5, 6]


class EmotionFusionCNN(nn.Module):
    """Your SER architecture (the head your .pt loads into)."""
    def __init__(self, n=6):
        super().__init__()
        self.resnet = resnet18(weights=None)
        d = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(d, n))

    def forward(self, x):
        return self.resnet(x)


def softmax(v):
    e = np.exp(np.asarray(v, dtype=np.float64) - np.max(v))
    return e / e.sum()


def certainty(p):
    """Per-clip quality in [0,1]: 1 - normalised Shannon entropy.
    1.0 = fully confident (one class), 0.0 = uniform (no information)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    h = -(p * np.log(p)).sum()
    return float(1.0 - h / np.log(len(p)))


def get_gt(row, name):
    if GT_MODE == "intended":
        return FILENAME.get(name.split('_')[2])
    raw = row[GT_COL[GT_MODE]]
    if ':' in raw:                      # ambiguous human vote -> skip
        return None
    return VOTE.get(raw)


def face_probs(flv, fer, cascade):
    """Returns (probs6, quality) in SIX order, or (None, 0.0) if no face.
    quality = frame_coverage * certainty(probs)."""
    cap = cv2.VideoCapture(flv)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None, 0.0
    summed, n = np.zeros(8), 0
    for pos in np.linspace(0.35, 0.65, FRAMES_PER_CLIP) * total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        _, logits = fer.predict_emotions(crop_face_rgb(frame, (x, y, x + w, y + h)), logits=True)
        summed += np.array(logits); n += 1
    cap.release()
    if n == 0:
        return None, 0.0
    probs = softmax(summed[HSE_KEEP])          # 6-vector, order = SIX
    coverage = n / FRAMES_PER_CLIP             # how much visual evidence we got
    return probs, coverage * certainty(probs)


def audio_probs(wav, model):
    """Returns (probs6, quality) in SIX order. Audio is always available offline."""
    t = wav_to_fusion_tensor(wav)
    t = torch.as_tensor(np.asarray(t), dtype=torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)                    # (1,3,64,94)
    with torch.no_grad():
        logits = model(t).cpu().numpy()[0]
    probs = softmax(logits)                    # 6-vector, order = SIX
    return probs, certainty(probs)


def confidence_fuse(FP, FQ, FA, AP, AQ, AA):
    """Confidence-weighted linear pool over all clips (vectorised).
    FP/AP: (N,6) prob matrices.  FQ/AQ: (N,) qualities.  FA/AA: (N,) availability 0/1.
    Falls back to reliability-only weights for any clip where quality zeroes out."""
    WF = RELIABILITY["face"]  * FA * FQ        # (N,)
    WA = RELIABILITY["audio"] * AA * AQ
    denom = WF + WA
    # guard: both channels contributed zero weight (e.g. both fully uniform)
    dead = denom <= 1e-12
    if dead.any():
        WF = np.where(dead, RELIABILITY["face"]  * FA, WF)
        WA = np.where(dead, RELIABILITY["audio"] * AA, WA)
        denom = WF + WA
    fused = (WF[:, None] * FP + WA[:, None] * AP) / denom[:, None]
    return fused


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=SIX, average='macro', zero_division=0)
    print(f"  {name:24} acc {acc:.3f} | macro-F1 {mf1:.3f}")
    return acc, mf1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (VIDEO_DIR, AUDIO_DIR, SUMMARY_CSV, SER_WEIGHTS):
        if not os.path.exists(p):
            sys.exit(f"Missing path: {p}")

    print("Loading face model (HSEmotion) and audio model (your SER)...")
    fer = HSEmotionRecognizer(model_name=FER_MODEL)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ser = EmotionFusionCNN()
    ser.load_state_dict(torch.load(SER_WEIGHTS, map_location='cpu'))
    ser.eval()

    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    if LIMIT:
        rows = rows[:LIMIT]

    y_true = []
    FP, FQ, FA = [], [], []        # face: probs, quality, availability
    AP, AQ, AA = [], [], []        # audio: probs, quality, availability
    skipped = noface = missing = 0
    for i, row in enumerate(rows, 1):
        name = row['FileName']
        gt = get_gt(row, name)
        if gt is None:
            skipped += 1; continue
        flv = os.path.join(VIDEO_DIR, name + '.flv')
        wav = os.path.join(AUDIO_DIR, name + '.wav')
        if not (os.path.isfile(flv) and os.path.isfile(wav)):
            missing += 1; continue
        fp, fq = face_probs(flv, fer, cascade)
        if fp is None:
            noface += 1; continue          # keep parity with method-1 harness
        ap, aq = audio_probs(wav, ser)
        y_true.append(gt)
        FP.append(fp); FQ.append(fq); FA.append(1.0)
        AP.append(ap); AQ.append(aq); AA.append(1.0)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} | scored {len(y_true)}")

    FP, AP = np.array(FP), np.array(AP)
    FQ, AQ = np.array(FQ), np.array(AQ)
    FA, AA = np.array(FA), np.array(AA)

    # class-order invariant: both channels must be 6-wide in the SAME SIX order
    assert FP.shape[1] == len(SIX) and AP.shape[1] == len(SIX), "probs must be 6-class in SIX order"

    print("\n" + "=" * 64)
    print(f"Ground truth: {GT_MODE} | scored {len(y_true)} clips "
          f"(skipped {skipped}, no-face {noface}, missing {missing})")
    print(f"Reliability prior (fixed a priori): {RELIABILITY}")
    print(f"Mean quality -> face {FQ.mean():.3f} | audio {AQ.mean():.3f}")
    print("=" * 64)

    face_pred  = [SIX[i] for i in FP.argmax(1)]
    audio_pred = [SIX[i] for i in AP.argmax(1)]
    evaluate("FACE only", y_true, face_pred)
    evaluate("AUDIO only", y_true, audio_pred)

    # reference: plain unweighted mixture (method 1 at a=0.5) -- shows if
    # confidence-weighting actually helped over a dumb average
    plain = 0.5 * FP + 0.5 * AP
    evaluate("FUSED plain (a=0.5)", y_true, [SIX[i] for i in plain.argmax(1)])

    # primary: confidence-weighted linear pool (method 4)
    fused = confidence_fuse(FP, FQ, FA, AP, AQ, AA)
    fpred = [SIX[i] for i in fused.argmax(1)]
    acc, mf1 = evaluate("FUSED conf-weighted", y_true, fpred)

    print("\n--- confidence-weighted fused, per-class ---")
    print(classification_report(y_true, fpred, labels=SIX, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, fpred, labels=SIX)
    np.savetxt(os.path.join(OUT_DIR, "fused_conf_confusion.csv"), cm, fmt='%d',
               delimiter=',', header=','.join(SIX), comments='')
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Purples')
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SIX, rotation=45, ha='right'); ax.set_yticklabels(SIX)
        ax.set_xlabel('Predicted'); ax.set_ylabel(f'{GT_MODE} (true)')
        ax.set_title(f'Confidence-weighted fused  (acc {acc:.2f})')
        for r in range(6):
            for c in range(6):
                ax.text(c, r, cm[r, c], ha='center', va='center',
                        color='white' if cm[r, c] > cm.max() / 2 else 'black')
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fused_conf_confusion.png"), dpi=150)
        print("Saved fused_conf_confusion.png to", OUT_DIR)
    except Exception as e:
        print("(plot skipped:", e, ")")


if __name__ == "__main__":
    main()
