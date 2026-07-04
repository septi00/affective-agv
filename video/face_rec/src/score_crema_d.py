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

Detection + crop + HSEmotion now come from the shared core (face_detector,
face_features, face_classifier) via face_clip.sum_face_logits, so this harness
and fuse_crema_d.py feed the model identically. The DECISION rule stays here --
it's a consumer choice, not part of the core.

Run:
    pip install hsemotion-onnx opencv-python numpy scikit-learn matplotlib
    python score_crema_d.py
Set CREMA_DIR and SUMMARY_CSV below to your local paths first.
"""
import os, csv, sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from face_classifier import FaceClassifier, HSE, HSE_TO_6, HSE_KEEP, SIX
from face_clip import sum_face_logits

# ---------------------------------------------------------------- CONFIG
CREMA_DIR   = "/Users/septi/dis_ws/cnn_trainer/datasets/crema-d/VideoFlash"
SUMMARY_CSV = "/Users/septi/dis_ws/cnn_trainer/datasets/crema-d/processedResults/summaryTable.csv"
MODEL_NAME  = "enet_b0_8_best_vgaf"   # b0 beat b2 on your data; keep b0
FRAMES_PER_CLIP = 3                   # frames sampled from the expressive middle of each clip
SKIP_TIES   = True                    # skip ambiguous FaceVotes like "A:F" (cleaner number)
LIMIT       = None                    # set e.g. 50 for a quick smoke run; None = full dataset
DECISION    = "pick_then_map"         # "pick_then_map" (8-class fix) or "drop_then_pick" (old)
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
# -----------------------------------------------------------------------

OUT_OF_SET = "__out__"               # marker: model picked Surprise/Contempt (a real miss)

FACEVOTE = {'A':'anger','D':'disgust','F':'fear','H':'happy','N':'neutral','S':'sad'}
FILENAME = {'ANG':'anger','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}


def decide(summed_logits):
    """Turn averaged 8-logits into a predicted label, per DECISION mode."""
    if DECISION == "drop_then_pick":
        sub = summed_logits[HSE_KEEP]
        return SIX[int(np.argmax(sub))]
    elif DECISION == "pick_then_map":
        top = int(np.argmax(summed_logits))          # honest top-1 over all 8
        return HSE_TO_6.get(HSE[top], OUT_OF_SET)    # Surprise/Contempt -> OUT_OF_SET (counted wrong)
    else:
        raise ValueError(f"unknown DECISION: {DECISION}")


def predict_clip(path, clf):
    """Sample frames, run the shared face core, average logits, decide.
    Returns predicted label (one of SIX or OUT_OF_SET) or None if no face found."""
    summed, n = sum_face_logits(path, clf, FRAMES_PER_CLIP)
    if n == 0:
        return None
    return decide(summed)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(CREMA_DIR) or not os.path.isfile(SUMMARY_CSV):
        sys.exit(f"Set CREMA_DIR / SUMMARY_CSV. Got:\n  {CREMA_DIR}\n  {SUMMARY_CSV}")

    print(f"Loading {MODEL_NAME} | decision rule: {DECISION}")
    clf = FaceClassifier(model_name=MODEL_NAME)

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
        pred = predict_clip(path, clf)
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
