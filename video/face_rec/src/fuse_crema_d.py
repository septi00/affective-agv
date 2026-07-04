"""
fuse_crema_d.py
Late-fusion test on CREMA-D: face (HSEmotion) + audio (your SER).

Scores FACE-only, AUDIO-only, and FUSED on the SAME clips against the SAME
ground truth -- the actual test of whether fusion beats either modality alone.
This is the spine of the results chapter.

Both channels emit a 6-class probability vector in the SAME order
[anger, disgust, fear, happy, neutral, sad], so fusion is a weighted average of
the two vectors (alpha = face weight). We sweep a few alphas.

FACE path now goes through the shared core (face_clip.sum_face_logits), the
SAME code score_crema_d.py uses, so the two harnesses' face numbers are
comparable by construction. AUDIO path is unchanged:
  - EmotionFusionCNN: resnet18 + Sequential(Dropout, Linear) head (loads your .pt)
  - features_fusion.wav_to_fusion_tensor: the SAME feature code as training

Run from a folder where features_fusion.py and the face_*.py modules are
importable. Set CONFIG paths first.
    pip install hsemotion-onnx opencv-python torch torchvision librosa numpy scikit-learn matplotlib
"""
import os, csv, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "video", "face_rec", "src"))
sys.path.insert(0, os.path.join(_BASE, "cnn_trainer", "audio", "src"))
from features_fusion import wav_to_fusion_tensor
from face_classifier import FaceClassifier, HSE_KEEP
from face_clip import sum_face_logits

# ---------------------------------------------------------------- CONFIG
CREMA       = os.path.join(_BASE, "cnn_trainer", "datasets", "crema-d")
VIDEO_DIR   = os.path.join(CREMA, "VideoFlash")
AUDIO_DIR   = os.path.join(CREMA, "AudioWAV")
SUMMARY_CSV = os.path.join(CREMA, "processedResults", "summaryTable.csv")
SER_WEIGHTS = os.path.join(_BASE, "cnn_trainer", "audio", "models", "emotion_fusion_resnet18_merged.pt")
FER_MODEL   = "enet_b0_8_best_vgaf"
FRAMES_PER_CLIP = 3
GT_MODE     = "multimodal"      # "multimodal" | "intended" | "facevote" | "voicevote"
ALPHAS      = [0.3, 0.5, 0.7]   # face weight in the fused average
LIMIT       = None              # set 50 for a smoke run
OUT_DIR     = os.path.join(_BASE, "video", "face_rec", "results")
# -----------------------------------------------------------------------

SIX = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
VOTE = {'A':'anger','D':'disgust','F':'fear','H':'happy','N':'neutral','S':'sad'}
FILENAME = {'ANG':'anger','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
GT_COL = {"multimodal":"MultiModalVote", "facevote":"FaceVote", "voicevote":"VoiceVote"}


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


def get_gt(row, name):
    if GT_MODE == "intended":
        return FILENAME.get(name.split('_')[2])
    raw = row[GT_COL[GT_MODE]]
    if ':' in raw:                      # ambiguous human vote -> skip
        return None
    return VOTE.get(raw)


def face_probs(flv, clf):
    """Shared face core -> 6-vector in SIX order, or None if no face."""
    summed, n = sum_face_logits(flv, clf, FRAMES_PER_CLIP)
    if n == 0:
        return None
    return softmax(summed[HSE_KEEP])          # 6-vector, order = SIX


def audio_probs(wav, model):
    t = wav_to_fusion_tensor(wav)
    t = torch.as_tensor(np.asarray(t), dtype=torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(0)                    # (1,3,64,94)
    with torch.no_grad():
        logits = model(t).cpu().numpy()[0]
    return softmax(logits)                     # 6-vector, order = SIX


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=SIX, average='macro', zero_division=0)
    print(f"  {name:18} acc {acc:.3f} | macro-F1 {mf1:.3f}")
    return acc, mf1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (VIDEO_DIR, AUDIO_DIR, SUMMARY_CSV, SER_WEIGHTS):
        if not os.path.exists(p):
            sys.exit(f"Missing path: {p}")

    print("Loading face model (HSEmotion) and audio model (your SER)...")
    clf = FaceClassifier(model_name=FER_MODEL)
    ser = EmotionFusionCNN()
    ser.load_state_dict(torch.load(SER_WEIGHTS, map_location='cpu'))
    ser.eval()

    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    if LIMIT:
        rows = rows[:LIMIT]

    y_true, FP, AP = [], [], []
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
        fp = face_probs(flv, clf)
        if fp is None:
            noface += 1; continue
        ap = audio_probs(wav, ser)
        y_true.append(gt); FP.append(fp); AP.append(ap)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} | scored {len(y_true)}")

    FP, AP = np.array(FP), np.array(AP)
    print("\n" + "=" * 60)
    print(f"Ground truth: {GT_MODE} | scored {len(y_true)} clips "
          f"(skipped {skipped}, no-face {noface}, missing {missing})")
    print("=" * 60)

    face_pred  = [SIX[i] for i in FP.argmax(1)]
    audio_pred = [SIX[i] for i in AP.argmax(1)]
    evaluate("FACE only", y_true, face_pred)
    evaluate("AUDIO only", y_true, audio_pred)

    best = None
    for a in ALPHAS:
        fused = a * FP + (1 - a) * AP
        fpred = [SIX[i] for i in fused.argmax(1)]
        acc, mf1 = evaluate(f"FUSED a={a:.1f}", y_true, fpred)
        if best is None or mf1 > best[2]:
            best = (a, acc, mf1, fpred)

    a, acc, mf1, fpred = best
    print("\n--- best fused (alpha={:.1f}) per-class ---".format(a))
    print(classification_report(y_true, fpred, labels=SIX, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, fpred, labels=SIX)
    np.savetxt(os.path.join(OUT_DIR, "fused_confusion.csv"), cm, fmt='%d',
               delimiter=',', header=','.join(SIX), comments='')
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Purples')
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SIX, rotation=45, ha='right'); ax.set_yticklabels(SIX)
        ax.set_xlabel('Predicted'); ax.set_ylabel(f'{GT_MODE} (true)')
        ax.set_title(f'Fused face+audio  alpha={a:.1f}  (acc {acc:.2f})')
        for r in range(6):
            for c in range(6):
                ax.text(c, r, cm[r, c], ha='center', va='center',
                        color='white' if cm[r, c] > cm.max() / 2 else 'black')
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fused_confusion.png"), dpi=150)
        print("Saved fused_confusion.png to", OUT_DIR)
    except Exception as e:
        print("(plot skipped:", e, ")")


if __name__ == "__main__":
    main()
