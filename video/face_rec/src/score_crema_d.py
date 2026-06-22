"""
score_crema_d.py
Score HSEmotion on CREMA-D faces against the human FaceVote labels.

Produces your "why this model, measured on my own data" number: overall
accuracy, macro-F1, per-class F1 (watch fear/disgust/sad/anger), and a
confusion matrix.

Ground truth = FaceVote (the face-only human majority rating from
processedResults/summaryTable.csv), NOT the intended emotion in the filename,
because the face channel predicts perceived facial emotion. The filename
("intended") label is also reported as a cross-check.

DECISION RULE -- two modes (set DECISION below):
  "drop_then_pick": restrict the 8 logits to your 6 classes, then argmax.
       (Silently converts Surprise/Contempt predictions into one of your 6,
        which can distort minority classes like anger.)
  "pick_then_map": argmax over all 8 first (the model's honest answer); if it
       lands on Surprise/Contempt, count it WRONG rather than redistributing.
       This is the "8-class fix" -- it stops putting words in the model's mouth.

Run "pick_then_map" and "drop_then_pick" and compare anger recall: if anger
jumps under pick_then_map, the weakness was the decision rule, not the model.

Run:
    pip install hsemotion-onnx opencv-python numpy scikit-learn matplotlib
    python score_crema_d.py
Set CREMA_DIR and SUMMARY_CSV below to your local paths first.
"""

import urllib.request          # <-- fixes hsemotion-onnx 0.3.1 download bug; keep this import
import os, csv, sys
import numpy as np
import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ---------------------------------------------------------------- CONFIG
CREMA_DIR   = os.path.expanduser("/cnn_trainer/datasets/crema-d/VideoFlash")              # folder of .flv clips
SUMMARY_CSV = os.path.expanduser("/cnn_trainer/datasets/crema-d/processedResults/summaryTable.csv")
MODEL_NAME  = "enet_b0_8_best_vgaf"   # b0 beat b2 on your data; keep b0
FRAMES_PER_CLIP = 3                   # frames sampled from the expressive middle of each clip
SKIP_TIES   = True                    # skip ambiguous FaceVotes like "A:F" (cleaner number)
LIMIT       = None                    # set e.g. 50 for a quick smoke run; None = full dataset
DECISION    = "pick_then_map"         # "pick_then_map" (8-class fix) or "drop_then_pick" (old)
OUT_DIR     = os.path.expanduser("/video/face_rec/results")
# -----------------------------------------------------------------------

# HSEmotion 8 classes -> your 6 (Contempt and Surprise have no CREMA-D equivalent)
HSE      = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
HSE_TO_6 = {'Anger':'anger','Disgust':'disgust','Fear':'fear','Happiness':'happy','Neutral':'neutral','Sadness':'sad'}
KEEP_IDX = [i for i, n in enumerate(HSE) if n in HSE_TO_6]   # [0,2,3,4,5,6]
KEEP_CLS = [HSE_TO_6[HSE[i]] for i in KEEP_IDX]
SIX      = ['anger','disgust','fear','happy','neutral','sad']
OUT_OF_SET = "__out__"               # marker: model picked Surprise/Contempt (a real miss)

FACEVOTE = {'A':'anger','D':'disgust','F':'fear','H':'happy','N':'neutral','S':'sad'}
FILENAME = {'ANG':'anger','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}

fer = None
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def decide(summed_logits):
    """Turn averaged 8-logits into a predicted label, per DECISION mode."""
    if DECISION == "drop_then_pick":
        sub = summed_logits[KEEP_IDX]
        return KEEP_CLS[int(np.argmax(sub))]
    elif DECISION == "pick_then_map":
        top = int(np.argmax(summed_logits))          # honest top-1 over all 8
        name = HSE[top]
        return HSE_TO_6.get(name, OUT_OF_SET)        # Surprise/Contempt -> OUT_OF_SET (counted wrong)
    else:
        raise ValueError(f"unknown DECISION: {DECISION}")


def detect_largest_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, x + w, y + h)


def predict_clip(path):
    """Decode, sample middle frames, detect+crop+HSEmotion, average logits, decide.
    Returns predicted label (one of SIX or OUT_OF_SET) or None if no face found."""
    from face_features import crop_face_rgb
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    positions = np.linspace(0.35, 0.65, FRAMES_PER_CLIP) * total
    summed, n = np.zeros(8), 0
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        box = detect_largest_face(frame)
        if box is None:
            continue
        crop = crop_face_rgb(frame, box)
        _, logits = fer.predict_emotions(crop, logits=True)
        summed += np.array(logits); n += 1
    cap.release()
    if n == 0:
        return None
    return decide(summed)


def main():
    global fer
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(CREMA_DIR) or not os.path.isfile(SUMMARY_CSV):
        sys.exit(f"Set CREMA_DIR / SUMMARY_CSV. Got:\n  {CREMA_DIR}\n  {SUMMARY_CSV}")

    print(f"Loading {MODEL_NAME} | decision rule: {DECISION}")
    fer = HSEmotionRecognizer(model_name=MODEL_NAME)

    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    if LIMIT:
        rows = rows[:LIMIT]

    y_true, y_pred, y_intended = [], [], []
    skipped_tie = skipped_noface = missing = out_of_set = 0

    for i, row in enumerate(rows, 1):
        name, facevote = row['FileName'], row['FaceVote']
        if SKIP_TIES and ':' in facevote:
            skipped_tie += 1; continue
        true_cls = FACEVOTE.get(facevote.split(':')[0])
        if true_cls is None:
            continue
        path = os.path.join(CREMA_DIR, name + '.flv')
        if not os.path.isfile(path):
            missing += 1; continue
        pred = predict_clip(path)
        if pred is None:
            skipped_noface += 1; continue
        if pred == OUT_OF_SET:
            out_of_set += 1                     # model said Surprise/Contempt: real miss, keep it
        y_true.append(true_cls); y_pred.append(pred)
        y_intended.append(FILENAME.get(name.split('_')[2]))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} processed | scored={len(y_true)}")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 60)
    print(f"Decision rule : {DECISION}")
    print(f"Scored clips  : {len(y_true)}")
    print(f"Skipped ties  : {skipped_tie} | no face: {skipped_noface} | missing files: {missing}")
    if DECISION == "pick_then_map":
        print(f"Predicted out-of-set (Surprise/Contempt, counted wrong): {out_of_set}")
    print("=" * 60)

    # labels for metrics: include OUT marker only if it occurred, so it shows in the matrix
    labels = SIX + ([OUT_OF_SET] if out_of_set else [])
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=SIX, average='macro', zero_division=0)
    print(f"\nAccuracy vs FaceVote : {acc:.3f}")
    print(f"Macro-F1 vs FaceVote : {mf1:.3f}  (over your 6 classes)\n")
    report = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
    print(report)

    valid = [(t, p) for t, p in zip(y_intended, y_pred) if t is not None]
    if valid:
        ia = accuracy_score([t for t, _ in valid], [p for _, p in valid])
        print(f"(cross-check) Accuracy vs intended/filename label: {ia:.3f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(os.path.join(OUT_DIR, f"confusion_{DECISION}.csv"), cm, fmt='%d', delimiter=',',
               header=','.join(labels), comments='')
    with open(os.path.join(OUT_DIR, f"report_{DECISION}.txt"), 'w') as f:
        f.write(f"{DECISION}\nAccuracy {acc:.3f}  Macro-F1 {mf1:.3f}\n\n{report}\n")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('FaceVote (true)')
        ax.set_title(f'HSEmotion / CREMA-D / {DECISION}  (acc {acc:.2f})')
        for r in range(len(labels)):
            for c in range(len(labels)):
                ax.text(c, r, cm[r, c], ha='center', va='center',
                        color='white' if cm[r, c] > cm.max() / 2 else 'black')
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"confusion_{DECISION}.png"), dpi=150)
        print(f"Saved confusion_{DECISION}.png + report to {OUT_DIR}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()