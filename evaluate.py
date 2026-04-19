"""
evaluate.py — Comprehensive model evaluation with confusion matrix,
per-class metrics, and misclassification analysis.

Usage:
    python evaluate.py --checkpoint models/checkpoints/best_model.pt \
                       --data_dir data/raw
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from torch.utils.data import DataLoader

from models.model import build_model, NUM_CLASSES
from utils.dataset import LightingDataset, CLASSES


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return (np.array(all_probs),
            np.array(all_preds),
            np.array(all_labels))


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix — Lighting Condition Classifier", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_history(history_path, save_path):
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="train", color="#1f77b4")
    ax1.plot(epochs, h["val_loss"],   label="val",   color="#ff7f0e")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax1.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="phase 2")

    ax2.plot(epochs, h["train_acc"], label="train", color="#1f77b4")
    ax2.plot(epochs, h["val_acc"],   label="val",   color="#ff7f0e")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="phase 2")

    plt.suptitle("Training History — Lighting Condition Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training history saved: {save_path}")


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=False, device=device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"[Eval] Loaded: {args.checkpoint}")

    # ── Data (test split) ─────────────────────────────────────────────────
    test_ds = LightingDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    print(f"[Eval] Test samples: {len(test_ds)}")

    # ── Collect predictions ───────────────────────────────────────────────
    probs, preds, labels = collect_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc = (preds == labels).mean()
    cm  = confusion_matrix(labels, preds)

    print(f"\n  Overall accuracy: {acc*100:.2f}%\n")
    print(classification_report(labels, preds, target_names=CLASSES))

    try:
        from sklearn.preprocessing import label_binarize
        labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
        auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
        print(f"  Macro ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, out / "confusion_matrix.png")

    history_path = Path(args.checkpoint).parent / "history.json"
    if history_path.exists():
        plot_training_history(history_path, out / "training_history.png")

    # ── Misclassification analysis ────────────────────────────────────────
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) > 0:
        print(f"\n  Misclassified: {len(wrong_idx)}/{len(labels)}")
        conf_wrong = probs[wrong_idx].max(axis=1)
        print(f"  Avg confidence on wrong predictions: {conf_wrong.mean()*100:.1f}%")
        print("  Most confused pairs:")
        pair_counts = {}
        for i in wrong_idx:
            pair = (CLASSES[labels[i]], CLASSES[preds[i]])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {pair[0]} → {pair[1]}: {count} times")

    print(f"\n[Eval] Outputs saved to: {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir",   default="data/raw")
    parser.add_argument("--output_dir", default="results")
    evaluate(parser.parse_args())
