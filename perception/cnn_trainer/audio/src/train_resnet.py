"""
train_resnet.py
ResNet18 backbone, pretrained on ImageNet, fine-tuned on the same
log-mel(+fusion) tensors as train_plain.py / train_fusion.py. Set DATASET
and FUSION below to pick children (BESD) only vs merged child+adult data,
and log-mel-only vs the 3-channel log-mel+pitch+ZCR fusion input.

Replaces train_fusion_resnet.py, train_resnet_plain.py,
train_fusion_resnet_combined.py, train_resnet_plain_combined.py, and the
chart-instrumented train_fusion_resnet_combined_chart.py -- same math, one
file, one place to configure. Per-(DATASET, FUSION) weights filenames are
kept identical to those old scripts' outputs so nothing downstream (eval.py,
the runtime perception pipeline) needs to change.

RESNET18 EXPECTS 3 CHANNELS (conv1 is pretrained on RGB). FUSION=True feeds
it the fusion tensor unmodified. FUSION=False swaps conv1 for a genuine
1-channel convolution, initialized by averaging the pretrained 3-channel
weights across the input-channel dimension (the standard grayscale-
adaptation trick) -- keeps most of the ImageNet pretraining signal while
being honestly single-channel, rather than replicating the one channel 3x
(which would keep conv1 untouched but make the "extra" 2 channels carry no
new information).

SAVE_HISTORY=True also writes ../results/<tag>_history.csv (epoch,
train/val loss/acc) for plot_training_curves.py -- this is what
train_fusion_resnet_combined_chart.py used to do as a separate script.

Evaluation (confusion matrix, per-domain breakdown) lives in eval.py, not
here -- this script only trains and saves weights.
"""
import os
import re
import csv
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

from features_fusion import wav_to_fusion_tensor

# --- SET THESE:
# DATASET: "children" = BESD only, "combined" = BESD + CREMA-D merged
# FUSION:  True = log-mel+pitch+zcr (3ch), False = log-mel only (1ch, conv1 surgery)
DATASET = "children"
FUSION = True

# The deployed combined+fusion checkpoint was originally trained for 60
# epochs; every other (DATASET, FUSION) combo used 40. Bump this if you're
# replicating that exact run.
EPOCHS = 40
SAVE_HISTORY = False   # True: also write ../results/<tag>_history.csv per epoch (for plot_training_curves.py)

_HERE = os.path.dirname(os.path.abspath(__file__))
CHILD_DATA_DIR = os.path.join(_HERE, "..", "..", "datasets", "besd")
ADULT_DATA_DIR = os.path.join(_HERE, "..", "..", "datasets", "crema-d", "AudioWAV")
_RESULTS_DIR = os.path.join(_HERE, "..", "results")
_MODELS_DIR = os.path.join(_HERE, "..", "models")

DATASET_CFG = {
    "children": dict(class_weights=None, desc="children (BESD) only"),
    "combined": dict(class_weights=[1.0, 1.5, 1.5, 1.0, 1.0, 1.0],
                     desc="merged child (BESD) + adult (CREMA-D) data"),
}

# (DATASET, FUSION) -> weights filename, matching the old per-combo scripts' outputs
WEIGHTS_FILE = {
    ("children", True):  "emotion_fusion_resnet18.pt",
    ("children", False): "emotion_resnet_plain.pt",
    ("combined", True):  "emotion_fusion_resnet18_merged.pt",
    ("combined", False): "emotion_resnet_plain_merged.pt",
}

EMOTIONS = {
    "angry":   0,
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "sad":     5,
}

CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware execution engine active on device: {device}")


class SpeechEmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_child_filename(path):
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

    speaker_id = f"child_{gender}_{age}_{serial}"
    return emotion, speaker_id


def parse_cremad_filename(path):
    name = os.path.splitext(os.path.basename(path))[0].strip()
    parts = name.split("_")

    if len(parts) != 4:
        return None

    actor_id = parts[0]
    raw_emotion = parts[2]

    crema_map = {
        "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5,
    }
    if raw_emotion not in crema_map:
        return None

    emotion_id = crema_map[raw_emotion]
    speaker_id = f"adult_actor_{actor_id}"
    return emotion_id, speaker_id


def load_dataset():
    X, y, speakers = [], [], []

    child_files = glob.glob(os.path.join(CHILD_DATA_DIR, "**", "*.wav"), recursive=True)
    if not child_files:
        raise SystemExit(f"No .wav files found under '{CHILD_DATA_DIR}'. Check the path.")
    print(f"Found {len(child_files)} child audio files.")
    for path in tqdm(child_files, desc="Extracting tensors (child)"):
        parsed = parse_child_filename(path)
        if parsed is None:
            continue
        emotion, speaker = parsed
        tensor = wav_to_fusion_tensor(path)
        X.append(tensor if FUSION else tensor[0:1])
        y.append(EMOTIONS[emotion])
        speakers.append(speaker)

    if DATASET == "combined":
        if os.path.exists(ADULT_DATA_DIR):
            adult_files = glob.glob(os.path.join(ADULT_DATA_DIR, "**", "*.wav"), recursive=True)
            print(f"\nFound {len(adult_files)} adult audio files.")
            for path in tqdm(adult_files, desc="Extracting tensors (adult)"):
                parsed = parse_cremad_filename(path)
                if parsed is None:
                    continue
                emotion_id, speaker = parsed
                tensor = wav_to_fusion_tensor(path)
                X.append(tensor if FUSION else tensor[0:1])
                y.append(emotion_id)
                speakers.append(speaker)
        else:
            print(f"\n[Warning] Adult directory not found at '{ADULT_DATA_DIR}'. Skipping adult data.")

    return X, np.array(y), speakers


def split_train_test(X, y, speakers):
    unique_speakers = sorted(set(speakers))
    rng = np.random.default_rng(seed=42)
    n_test = max(1, round(0.15 * len(unique_speakers)))
    test_speakers = set(rng.choice(unique_speakers, size=n_test, replace=False))

    train_X, train_y, test_X, test_y = [], [], [], []
    for idx, spk in enumerate(speakers):
        if spk in test_speakers:
            test_X.append(X[idx])
            test_y.append(y[idx])
        else:
            train_X.append(X[idx])
            train_y.append(y[idx])

    print(f"\nSpeakers: {len(unique_speakers)} total, {n_test} held out for testing")
    print(f"Train clips = {len(train_y)}, Test clips = {len(test_y)}")
    return train_X, train_y, test_X, test_y


class EmotionResnetCNN(nn.Module):
    """ResNet18 classifier. Reads the module-level FUSION flag at
    construction time: True keeps the pretrained 3-channel conv1 as-is
    (fed the full fusion tensor); False swaps conv1 for a genuine
    1-channel convolution -- see module docstring for why and how it's
    initialized."""

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        if not FUSION:
            old_conv1 = self.resnet.conv1
            new_conv1 = nn.Conv2d(1, old_conv1.out_channels,
                                  kernel_size=old_conv1.kernel_size,
                                  stride=old_conv1.stride,
                                  padding=old_conv1.padding,
                                  bias=(old_conv1.bias is not None))
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
            self.resnet.conv1 = new_conv1

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),  # prevent memorization of small datasets
            nn.Linear(num_ftrs, len(CLASS_NAMES))
        )

    def forward(self, x):
        return self.resnet(x)


def main():
    cfg = DATASET_CFG[DATASET]
    weights_file = WEIGHTS_FILE[(DATASET, FUSION)]
    print(f"=== ResNet18, {'with' if FUSION else 'no'} feature fusion, {cfg['desc']} ===\n")

    X, y, speakers = load_dataset()
    X_train, y_train, X_test, y_test = split_train_test(X, y, speakers)

    train_loader = DataLoader(SpeechEmotionDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SpeechEmotionDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = EmotionResnetCNN().to(device)
    if cfg["class_weights"] is not None:
        weights = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = []
    print(f"\nStarting {EPOCHS}-epoch training loop with ResNet18...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_running_loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss = val_running_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if SAVE_HISTORY:
            history.append(dict(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                                 val_loss=val_loss, val_acc=val_acc))
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            tag = os.path.splitext(weights_file)[0]
            with open(os.path.join(_RESULTS_DIR, f"{tag}_history.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                writer.writeheader()
                writer.writerows(history)

        scheduler.step(val_acc)

    os.makedirs(_MODELS_DIR, exist_ok=True)
    weights_path = os.path.join(_MODELS_DIR, weights_file)
    torch.save(model.state_dict(), weights_path)
    print(f"\nWrote finalized ResNet18 weights to {weights_path}")


if __name__ == "__main__":
    main()
