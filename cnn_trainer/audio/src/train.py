import os
import re
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from features import wav_to_logmel, N_MELS, TIME_STEPS

DATA_DIR = "/Users/septi/dis_ws/cnn_trainer/besd"

EMOTIONS = {
    "angry":   0,
    "disgust": 1,
    "fear":    2,
    "happy":   3,
    "neutral": 4,
    "sad":     5,
}

CLASS_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

def parse_filename(path):
    name = os.path.splitext(os.path.basename(path))[0].strip()

    low = name.lower()
    emotion = next((e for e in EMOTIONS if e in low), None)

    if emotion is None or "." not in name:
        return None

    serial, rest = name.split(".", 1)

    # langgender is always the first 2 chars after the dot, e.g. "TM" or "EF"
    if len(rest) < 2:
        return None
    language = rest[0].upper()
    gender = rest[1].upper()

    # age is the first run of digits in the rest of the filename
    m = re.search(r"\d+", rest[2:])
    age = m.group() if m else "0"

    speaker_id = f"{gender}_{age}_{serial}"
    return language, emotion, speaker_id


def load_dataset():

    files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

    if not files:
        raise SystemExit(f"No .wav files found under '{DATA_DIR}'. Check the path.")

    X, y, speakers = [], [], []

    for path in tqdm(files, desc="extracting features"):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        language, emotion, speaker = parsed
        X.append(wav_to_logmel(path))
        y.append(EMOTIONS[emotion])
        speakers.append(speaker)

    return np.array(X), np.array(y), speakers


def split_train_test(X, y, speakers):
    unique_speakers = sorted(set(speakers))

    rng = np.random.default_rng(seed=42)
    n_test = max(1, round(0.15 * len(unique_speakers)))
    test_speakers = set(rng.choice(unique_speakers, size=n_test, replace=False))

    test_idx = [i for i, s in enumerate(speakers) if s in test_speakers]
    train_idx = [i for i, s in enumerate(speakers) if s not in test_speakers]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"speakers: {len(unique_speakers)} total, {n_test} held out for testing")
    print(f"train clips = {len(y_train)}, test clips = {len(y_test)}")
    return X_train, y_train, X_test, y_test


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((N_MELS, TIME_STEPS, 1)),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    X, y, speakers = load_dataset()
    X_train, y_train, X_test, y_test = split_train_test(X, y, speakers)
    model = build_model()

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=40,
              batch_size=32,
              verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nSpeaker-independent test accuracy: {acc:.3f}")

    pred = model.predict(X_test, verbose=0).argmax(1)
    cm = tf.math.confusion_matrix(y_test, pred, num_classes=len(CLASS_NAMES)).numpy()
    print("confusion matrix (rows = true, cols = predicted):")
    print("        " + " ".join(f"{n[:4]:>5}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        print(f"{name:>9} " + " ".join(f"{v:>5}" for v in row))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("emotion_cnn.tflite", "wb") as fp:
        fp.write(tflite_model)
    with open("emotion_labels.txt", "w") as fp:
        fp.write("\n".join(CLASS_NAMES))
    print("\nWrote emotion_cnn.tflite and emotion_labels.txt")

if __name__ == "__main__":
    main()
