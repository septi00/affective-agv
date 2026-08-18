"""
Train a small CNN to classify children's speech emotion from log-mel
spectrograms, using the C-BESD (Children Bilingual Emotion Speech) dataset.

Filename convention (C-BESD):  1.TF_12_happy_1.wav
  1     serial number of the child
  T     language  (T = Telugu, E = English)
  F     gender    (M = male, F = female)
  12    age
  happy emotion   (angry, happy, sad, disgust, neutral, fear)
  1     utterance number

Run:  python train_emotion_children.py
Produces: emotion_cnn_child.tflite  +  emotion_labels_child.txt
"""
import os
import re
import glob
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from features import wav_to_logmel, N_MELS, TIME_STEPS

# Point this at the folder kagglehub printed for the C-BESD download.
DATA_DIR = "/Users/septi/dis_ws/cnn_trainer/besd"

# Language filter:  None = use both, "E" = English only, "T" = Telugu only.
LANGUAGE = None

# Emotion word in the filename -> class index.
# Indices match the CREMA-D adult model, so the two stay label-compatible.
EMOTIONS = {
    "angry":   0,   # anger
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "sad":     5,
}
CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def parse_filename(path):
    """
    Return (language, emotion, speaker_id) or None if it doesn't parse.
    Real C-BESD names look like:  9.EF_8 Angry_4.wav
      9     serial    E language   F gender   8 age   Angry emotion   4 utt
    Note: a SPACE separates age and emotion, and emotion case varies.
    """
    name = os.path.splitext(os.path.basename(path))[0].strip()   # "9.EF_8 Angry_4"

    # Emotion: find any known emotion word anywhere, case-insensitive.
    low = name.lower()
    emotion = next((e for e in EMOTIONS if e in low), None)
    if emotion is None or "." not in name:
        return None

    serial, rest = name.split(".", 1)        # "9", "EF_8 Angry_4"
    seg = rest.split("_")                     # ["EF", "8 Angry", "4"]
    if len(seg) < 2 or len(seg[0]) < 2:
        return None
    language = seg[0][0].upper()              # E / T
    gender   = seg[0][1].upper()              # M / F
    m = re.search(r"\d+", seg[1])             # leading digits of "8 Angry"
    age = m.group() if m else "0"

    # Speaker = the child. Excludes language (same kid speaks both).
    speaker_id = f"{gender}_{age}_{serial}"
    return language, emotion, speaker_id


def load_dataset(data_dir):
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise SystemExit(f"No .wav files found under '{data_dir}'. Check the path.")

    records, skipped = [], 0
    lang_count = {}
    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            skipped += 1; continue
        language, emotion, speaker = parsed
        if emotion not in EMOTIONS:
            skipped += 1; continue
        if LANGUAGE and language != LANGUAGE:
            continue
        lang_count[language] = lang_count.get(language, 0) + 1
        records.append((f, emotion, speaker))

    if not records:
        raise SystemExit("Parsed 0 usable files — filename format may differ.")

    # Hold out ~15% of the CHILDREN (not clips) for an honest test score.
    speakers = sorted({r[2] for r in records})
    n_test = max(1, round(0.15 * len(speakers)))
    test_speakers = set(speakers[-n_test:])

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for f, emotion, speaker in tqdm(records, desc="extracting features"):
        feat = wav_to_logmel(f)
        if speaker in test_speakers:
            X_te.append(feat); y_te.append(EMOTIONS[emotion])
        else:
            X_tr.append(feat); y_tr.append(EMOTIONS[emotion])

    print(f"files found = {len(files)}   used = {len(records)}   skipped = {skipped}")
    print(f"language counts = {lang_count}")
    print(f"children: {len(speakers)} total, {len(test_speakers)} held out for test")
    return (np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te))


def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((N_MELS, TIME_STEPS, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    X_tr, y_tr, X_te, y_te = load_dataset(DATA_DIR)
    print(f"train clips = {len(y_tr)}   test clips = {len(y_te)}   "
          f"feature shape = {X_tr.shape[1:]}")

    classes, counts = np.unique(y_tr, return_counts=True)
    cw = {int(c): len(y_tr) / (len(classes) * n) for c, n in zip(classes, counts)}

    model = build_model()
    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=40, batch_size=32, class_weight=cw, verbose=2)

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nSpeaker-independent test accuracy: {acc:.3f}")

    pred = model.predict(X_te, verbose=0).argmax(1)
    cm = tf.math.confusion_matrix(y_te, pred).numpy()
    print("confusion matrix (rows = true, cols = predicted):")
    print("          " + " ".join(f"{n[:4]:>5}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        print(f"{name:>9} " + " ".join(f"{v:>5}" for v in row))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("emotion_cnn_child.tflite", "wb") as fp:
        fp.write(tflite_model)
    with open("../models/emotion_labels_child.txt", "w") as fp:
        fp.write("\n".join(CLASS_NAMES))
    print("\nWrote emotion_cnn_child.tflite and emotion_labels_child.txt")


if __name__ == "__main__":
    main()
