import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

# Import the fusion extractor we built
from features_fusion import wav_to_fusion_tensor

# --- PATH CONFIGURATION ---
CHILD_DATA_DIR = "/Users/septi/dis_ws/cnn_trainer/besd"
ADULT_DATA_DIR = "/Users/septi/dis_ws/cnn_trainer/crema-d-mirror/AudioWAV"  # <-- Update to your CREMA-D folder path

EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
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


# 1. PARSER FOR CHILD DATASET (BESD)
def parse_child_filename(path):
    name = os.path.splitext(os.path.basename(path))[0].strip()
    low = name.lower()
    emotion = next((e for e in EMOTIONS if e in low), None)

    if emotion is None or "." not in name:
        return None

    serial, rest = name.split(".", 1)
    if len(rest) < 2:
        return None

    language = rest[0].upper()
    gender = rest[1].upper()

    m = re.search(r"\d+", rest[2:])
    age = m.group() if m else "0"

    speaker_id = f"child_{gender}_{age}_{serial}"
    return emotion, speaker_id


# 2. PARSER FOR ADULT DATASET (CREMA-D: e.g., 1001_IEO_ANG_HI.wav)
def parse_cremad_filename(path):
    name = os.path.splitext(os.path.basename(path))[0].strip()
    parts = name.split("_")

    if len(parts) != 4:
        return None

    actor_id = parts[0]  # e.g., "1001"
    raw_emotion = parts[2]  # e.g., "ANG"

    # Map CREMA-D's three-letter uppercase codes directly to your integer tracks
    crema_map = {
        "ANG": 0,  # angry
        "DIS": 1,  # disgust
        "FEA": 2,  # fear
        "HAP": 3,  # happy
        "NEU": 4,  # neutral
        "SAD": 5,  # sad
    }

    if raw_emotion not in crema_map:
        return None

    emotion_id = crema_map[raw_emotion]
    speaker_id = f"adult_actor_{actor_id}"  # Keeps adult IDs totally unique from child IDs

    return emotion_id, speaker_id


def load_dataset():
    X, y, speakers = [], [], []

    # --- STEP A: LOAD CHILD VOICES (BESD) ---
    child_files = glob.glob(os.path.join(CHILD_DATA_DIR, "**", "*.wav"), recursive=True)
    print(f"Found {len(child_files)} child audio files.")
    for path in tqdm(child_files, desc="Extracting Child Tensors"):
        parsed = parse_child_filename(path)
        if parsed is None:
            continue
        emotion, speaker = parsed
        X.append(wav_to_fusion_tensor(path))
        y.append(EMOTIONS[emotion])
        speakers.append(speaker)

    # --- STEP B: LOAD ADULT VOICES (CREMA-D) ---
    if os.path.exists(ADULT_DATA_DIR):
        adult_files = glob.glob(os.path.join(ADULT_DATA_DIR, "**", "*.wav"), recursive=True)
        print(f"\nFound {len(adult_files)} adult audio files.")
        for path in tqdm(adult_files, desc="Extracting Adult Tensors"):
            parsed = parse_cremad_filename(path)
            if parsed is None:
                continue
            emotion_id, speaker = parsed
            X.append(wav_to_fusion_tensor(path))
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

    print(f"\nTotal Speakers (Child + Adult): {len(unique_speakers)}")
    print(f"Held out {n_test} speakers entirely for testing.")
    print(f"Train samples = {len(train_y)} | Test samples = {len(test_y)}")
    return train_X, train_y, test_X, test_y


class EmotionFusionCNN(nn.Module):
    def __init__(self):
        super(EmotionFusionCNN, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, len(CLASS_NAMES))
        )

    def forward(self, x):
        return self.resnet(x)


def main():
    X, y, speakers = load_dataset()
    X_train, y_train, X_test, y_test = split_train_test(X, y, speakers)

    train_loader = DataLoader(SpeechEmotionDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SpeechEmotionDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = EmotionFusionCNN().to(device)
    weights = torch.tensor([1.0, 1.5, 1.5, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print("\nStarting execution loop with ResNet18 on Merged Dataset...")
    for epoch in range(40):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1:02d}/40 | Loss: {running_loss / len(train_loader):.4f} | Merged Test Acc: {test_acc:.2f}%")

        scheduler.step(test_acc)

    torch.save(model.state_dict(), "emotion_fusion_resnet18_merged.pt")
    print("\nWrote finalized configuration out to emotion_fusion_resnet18_merged.pt")


if __name__ == "__main__":
    main()