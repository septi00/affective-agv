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

DATA_DIR = "/Users/septi/dis_ws/cnn_trainer/besd"

EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}

CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

# Configure Apple Silicon GPU Hardware targeting
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


def parse_filename(path):
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

    speaker_id = f"{gender}_{age}_{serial}"
    return language, emotion, speaker_id


def load_dataset():
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    if not files:
        raise SystemExit(f"No .wav files found under '{DATA_DIR}'. Check the path.")

    X, y, speakers = [], [], []
    for path in tqdm(files, desc="Extracting 3-Channel Fusion Tensors"):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        language, emotion, speaker = parsed
        X.append(wav_to_fusion_tensor(path))
        y.append(EMOTIONS[emotion])
        speakers.append(speaker)
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


# --- THE NEW RESNET18 ARCHITECTURE ---
class EmotionFusionCNN(nn.Module):
    def __init__(self):
        super(EmotionFusionCNN, self).__init__()

        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the final 1000-class ImageNet layer with our 6-class emotion layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),  # Prevent memorization of small datasets
            nn.Linear(num_ftrs, len(CLASS_NAMES))
        )

    def forward(self, x):
        # Pass the 3-channel audio straight into ResNet
        return self.resnet(x)


def main():
    X, y, speakers = load_dataset()
    X_train, y_train, X_test, y_test = split_train_test(X, y, speakers)

    train_loader = DataLoader(SpeechEmotionDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SpeechEmotionDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = EmotionFusionCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer with L2 Regularization (weight_decay) to combat overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Scheduler to drop learning rate when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print("\nStarting execution training loop with ResNet18...")
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

        # Evaluation step
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
            f"Epoch {epoch + 1:02d}/40 | Loss: {running_loss / len(train_loader):.4f} | Speaker-Ind. Test Acc: {test_acc:.2f}%")

        # Tell the scheduler how the model performed this epoch
        scheduler.step(test_acc)

    torch.save(model.state_dict(), "emotion_fusion_resnet18.pt")
    print("\nWrote finalized ResNet18 weights configuration out to emotion_fusion_resnet18.pt")


if __name__ == "__main__":
    main()