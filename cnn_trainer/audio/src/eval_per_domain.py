import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from features_fusion import wav_to_fusion_tensor   # same extractor as training

# --- point these at your two dataset roots --------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_DATASETS  = os.path.join(_HERE, "..", "..", "datasets")
BESD_DIR   = os.path.join(_DATASETS, "besd")
CREMAD_DIR = os.path.join(_DATASETS, "crema-d", "AudioWAV")
WEIGHTS    = os.path.join(_HERE, "..", "models", "emotion_fusion_resnet18_merged.pt")
# --------------------------------------------------------------------------

EMOTIONS    = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}
CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
CREMAD_CODE = {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
               "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- parsers: must reproduce the SAME speaker_id strings training used -----
def parse_besd(path):
    name = os.path.splitext(os.path.basename(path))[0].strip()
    low = name.lower()
    emotion = next((e for e in EMOTIONS if e in low), None)
    if emotion is None or "." not in name:
        return None
    serial, rest = name.split(".", 1)
    if len(rest) < 2:
        return None
    gender = rest[1].upper()
    m = re.search(r"\d+", rest[2:])
    age = m.group() if m else "0"
    return emotion, f"child_{gender}_{age}_{serial}", "child"


def parse_cremad(path):
    # CREMA-D: ActorID_Sentence_EMO_Intensity.wav  e.g. 1001_IEO_ANG_HI.wav
    parts = os.path.splitext(os.path.basename(path))[0].strip().split("_")
    if len(parts) < 3:
        return None
    emotion = CREMAD_CODE.get(parts[2].upper())
    if emotion is None:
        return None
    return emotion, f"adult_actor_{parts[0]}", "adult"


def load_all():
    X, y, speakers, domains = [], [], [], []
    for root, parser in [(BESD_DIR, parse_besd), (CREMAD_DIR, parse_cremad)]:
        files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        for path in tqdm(files, desc=f"Loading {os.path.basename(root)}"):
            parsed = parser(path)
            if parsed is None:
                continue
            emotion, spk, dom = parsed
            X.append(wav_to_fusion_tensor(path))
            y.append(EMOTIONS[emotion])
            speakers.append(spk)
            domains.append(dom)
    return X, np.array(y), speakers, np.array(domains)


def test_mask(speakers):
    # MUST mirror split_train_test() in the trainer: same seed, same logic
    unique = sorted(set(speakers))
    rng = np.random.default_rng(seed=42)
    n_test = max(1, round(0.15 * len(unique)))
    test_speakers = set(rng.choice(unique, size=n_test, replace=False))
    return np.array([s in test_speakers for s in speakers])


class EmotionFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        n = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(n, len(CLASS_NAMES)))

    def forward(self, x):
        return self.resnet(x)


def report(tag, y_true, y_pred):
    n = len(y_true)
    if n == 0:
        print(f"\n[{tag}] no test clips")
        return
    acc = (y_pred == y_true).mean() * 100
    print(f"\n[{tag}] test clips = {n} | accuracy = {acc:.2f}%")
    for c, name in enumerate(CLASS_NAMES):
        sel = y_true == c
        tot = int(sel.sum())
        rec = 100 * (y_pred[sel] == c).mean() if tot else 0.0
        print(f"    {name:8s} n={tot:4d}  recall={rec:5.1f}%")


def main():
    X, y, speakers, domains = load_all()
    mask = test_mask(speakers)

    model = EmotionFusionCNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    idx = np.where(mask)[0]
    preds = np.empty(len(idx), dtype=int)
    with torch.no_grad():
        for j, i in enumerate(tqdm(idx, desc="Evaluating")):
            preds[j] = int(model(X[i].unsqueeze(0).to(device)).argmax(1).item())

    y_test, dom_test = y[idx], domains[idx]
    report("all",   y_test, preds)
    report("child", y_test[dom_test == "child"], preds[dom_test == "child"])
    report("adult", y_test[dom_test == "adult"], preds[dom_test == "adult"])


if __name__ == "__main__":
    main()