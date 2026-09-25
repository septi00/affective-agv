"""
eval.py
Single entry point for testing any trained emotion-CNN checkpoint in this
folder on its own held-out speakers, without retraining. Replaces the old
eval_plain_cnn.py / eval_fusion_cnn.py / eval_resnet_children.py /
eval_combined.py scripts -- same behaviour, one file, one place to configure.

Set EXPERIMENT below to one of the keys in EXPERIMENTS and run:
    python eval.py

Each experiment reuses its training module's own load_dataset() /
split_train_test() / model class rather than reimplementing feature
extraction here -- if a split ever drifts from what a given training run
actually used, that module drifts together with its own eval, since both
come from the same source. Training modules are now config-driven
(train_plain.py / train_fusion.py / train_resnet.py, each with a DATASET
toggle and train_resnet.py also a FUSION toggle) rather than one file per
combination, so "config" below sets those module-level variables before
load_dataset()/the model class are used.

"domain_breakdown" experiments (the merged-dataset ablations) additionally
report child/adult accuracy. That needs each test speaker's domain, which
split_train_test() doesn't return, so those use a local test_mask() that
MUST mirror split_train_test()'s speaker split (same seed=42 logic) in every
train_*.py in this folder.

Saves the confusion matrix as PNG + CSV under ../results/, tagged by
experiment key so re-running a different EXPERIMENT doesn't clobber the
previous one.
"""
import importlib
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# --- SET THIS: key into EXPERIMENTS below, picks which checkpoint gets evaluated
EXPERIMENT = "stage3_resnet_fusion_combined"

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_HERE, "..", "results")

EXPERIMENTS = {
    "plain_children": dict(
        module="train_plain", config=dict(DATASET="children"),
        model_cls="EmotionPlainCNN", weights="emotion_plain_cnn.pt",
        label="Plain CNN, no feature fusion, children (BESD) only",
        domain_breakdown=False,
    ),
    "fusion_children": dict(
        module="train_fusion", config=dict(DATASET="children"),
        model_cls="EmotionFusionCNN", weights="emotion_fusion_cnn.pt",
        label="Plain CNN + feature fusion, children (BESD) only",
        domain_breakdown=False,
    ),
    "resnet_fusion_children": dict(
        module="train_resnet", config=dict(DATASET="children", FUSION=True),
        model_cls="EmotionResnetCNN", weights="emotion_fusion_resnet18.pt",
        label="ResNet18 + feature fusion, children (BESD) only",
        domain_breakdown=False,
    ),
    "stage1_plain_combined": dict(
        module="train_plain", config=dict(DATASET="combined"),
        model_cls="EmotionPlainCNN", weights="emotion_plain_cnn_merged.pt",
        label="Stage 1: plain CNN, no feature fusion, merged data",
        domain_breakdown=True,
    ),
    "stage2_fusion_combined": dict(
        module="train_fusion", config=dict(DATASET="combined"),
        model_cls="EmotionFusionCNN", weights="emotion_fusion_cnn_merged.pt",
        label="Stage 2: plain CNN + feature fusion, merged data",
        domain_breakdown=True,
    ),
    "stage3_resnet_fusion_combined": dict(
        module="train_resnet", config=dict(DATASET="combined", FUSION=True),
        model_cls="EmotionResnetCNN", weights="emotion_fusion_resnet18_merged.pt",
        label="Stage 3: ResNet18 + feature fusion, merged data",
        domain_breakdown=True,
    ),
    "stage4_resnet_plain_combined": dict(
        module="train_resnet", config=dict(DATASET="combined", FUSION=False),
        model_cls="EmotionResnetCNN", weights="emotion_resnet_plain_merged.pt",
        label="Side comparison: ResNet18, no feature fusion, merged data",
        domain_breakdown=True,
    ),
}


def domain_of(speaker_id):
    return "child" if speaker_id.startswith("child_") else "adult"


def test_mask(speakers):
    # MUST mirror split_train_test() in every train_*.py: same seed, same logic.
    unique = sorted(set(speakers))
    rng = np.random.default_rng(seed=42)
    n_test = max(1, round(0.15 * len(unique)))
    test_speakers = set(rng.choice(unique, size=n_test, replace=False))
    return np.array([s in test_speakers for s in speakers])


def predict(model, items, device):
    preds = np.empty(len(items), dtype=int)
    with torch.no_grad():
        for j, x in enumerate(items):
            t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
            preds[j] = int(model(t.unsqueeze(0).to(device)).argmax(1).item())
    return preds


def report(tag, y_true, y_pred, class_names):
    n = len(y_true)
    if n == 0:
        print(f"\n[{tag}] no test clips")
        return
    acc = (y_pred == y_true).mean() * 100
    print(f"\n[{tag}] test clips = {n} | accuracy = {acc:.2f}%")
    for c, name in enumerate(class_names):
        sel = y_true == c
        tot = int(sel.sum())
        rec = 100 * (y_pred[sel] == c).mean() if tot else 0.0
        print(f"    {name:8s} n={tot:4d}  recall={rec:5.1f}%")


def save_confusion_matrix(cm, class_names, label, acc, tag):
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(_RESULTS_DIR, f"{tag}_confusion.csv")
    np.savetxt(csv_path, cm, fmt='%d', delimiter=',',
              header=','.join(class_names), comments='')
    print(f"\nSaved {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"(confusion matrix image skipped: {e})")
        return

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Purples')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f"{label}\nacc {acc:.1f}%")
    for r in range(n):
        for c in range(n):
            ax.text(c, r, cm[r, c], ha='center', va='center',
                    color='white' if cm[r, c] > cm.max() / 2 else 'black')
    fig.colorbar(im); fig.tight_layout()

    png_path = os.path.join(_RESULTS_DIR, f"{tag}_confusion.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    print(f"=== {cfg['label']} ===\n")

    mod = importlib.import_module(cfg["module"])
    for key, value in cfg["config"].items():
        setattr(mod, key, value)

    weights_path = os.path.join(_HERE, "..", "models", cfg["weights"])
    if not os.path.exists(weights_path):
        raise SystemExit(f"No weights at {weights_path} -- run {cfg['module']}.py first, "
                          f"or edit EXPERIMENTS[{EXPERIMENT}]['weights'] to point elsewhere.")

    X, y, speakers = mod.load_dataset()
    class_names = mod.CLASS_NAMES

    model_cls = getattr(mod, cfg["model_cls"])
    model = model_cls().to(mod.device)
    model.load_state_dict(torch.load(weights_path, map_location=mod.device))
    model.eval()

    if cfg["domain_breakdown"]:
        idx = np.where(test_mask(speakers))[0]
        domains = np.array([domain_of(s) for s in speakers])[idx]
        y_test = y[idx]
        preds = predict(model, [X[i] for i in idx], mod.device)

        report("all", y_test, preds, class_names)
        report("child", y_test[domains == "child"], preds[domains == "child"], class_names)
        report("adult", y_test[domains == "adult"], preds[domains == "adult"], class_names)
    else:
        _, _, X_test, y_test = mod.split_train_test(X, y, speakers)
        y_test = np.array(y_test)
        preds = predict(model, X_test, mod.device)

        report("all", y_test, preds, class_names)

    acc = (preds == y_test).mean() * 100

    cm = confusion_matrix(y_test, preds, labels=range(len(class_names)))
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print("        " + " ".join(f"{n[:4]:>5}" for n in class_names))
    for name, row in zip(class_names, cm):
        print(f"{name:>9} " + " ".join(f"{v:>5}" for v in row))

    save_confusion_matrix(cm, class_names, cfg["label"], acc, EXPERIMENT)


if __name__ == "__main__":
    main()
