import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from ai_edge_litert.interpreter import Interpreter
from features_fusion import array_to_fusion_tensor

# YAMNet's fixed input: 0.975 s @ 16 kHz mono, float32 in [-1, 1].
YAMNET_FRAME = 15600

CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


class SoundClassifier:
    """
    Thin wrapper around the YAMNet tflite model.

    feed it a mono 16 kHz buffer -> get back the top sound classes.
    """

    def __init__(self, model_path, labels_path, top_k=3):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.labels = self._load_labels(labels_path)
        self.top_k = top_k

    @staticmethod
    def _load_labels(path):
        # yamnet_class_map.csv format: index,mid,display_name
        names = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)                      # skip header row
            for row in reader:
                names.append(row[2])          # display_name column
        return names

    def _scores(self, mono):
        x = np.asarray(mono, dtype=np.float32)
        if len(x) < YAMNET_FRAME:
            x = np.pad(x, (0, YAMNET_FRAME - len(x)))
        elif len(x) > YAMNET_FRAME:
            x = x[-YAMNET_FRAME:]
        self.interpreter.set_tensor(self.inp["index"], x.reshape(self.inp["shape"]))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out["index"])[0]   # shape (521,)

    def classify(self, mono):
        """mono: 1-D float32 16 kHz. Returns top_k (label, score), best first."""
        scores = self._scores(mono)
        top = np.argsort(scores)[::-1][:self.top_k]
        return [(self.labels[i], float(scores[i])) for i in top]

    def speech_score(self, mono, speech_ids=(0, 1, 2, 3, 4)):
        """
        Total probability over the speech family (Speech, Child speech,
        Conversation, Narration, Babbling). Gate on this instead of waiting
        for one label to win top_k -- child speech rarely wins outright.
        VERIFY the indices against your yamnet_class_map.csv.
        """
        return float(self._scores(mono)[list(speech_ids)].sum())


class _EmotionFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)          # no ImageNet download on robot
        n = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(n, len(CLASS_NAMES)))

    def forward(self, x):
        return self.resnet(x)


class EmotionClassifier:
    """
    feed it a mono 16 kHz buffer (ideally ~3 s of speech) -> (emotion, score).
    Uses array_to_fusion_tensor from features_fusion.py, so the live features
    are identical to training.
    """

    def __init__(self, weights_path, device="cpu"):
        self.device = torch.device(device)
        m = _EmotionFusionCNN()
        m.load_state_dict(torch.load(weights_path, map_location=self.device))
        m.eval().to(self.device)
        self.model = m

    @torch.no_grad()
    def classify(self, mono):
        x = array_to_fusion_tensor(mono).unsqueeze(0).to(self.device)   # (1, 3, 64, 94)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        i = int(probs.argmax())
        return CLASS_NAMES[i], float(probs[i])


if __name__ == "__main__":
    clf = SoundClassifier("yamnet.tflite", "yamnet_class_map.csv")
    ser = EmotionClassifier("../models/emotion_fusion_resnet18_merged.pt")

    noise = (np.random.randn(48000) * 0.05).astype(np.float32)
    print("speech score:", clf.speech_score(noise))
    print("emotion:", ser.classify(noise))