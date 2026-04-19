"""
train.py — Two-phase training loop for the LightingClassifier.

Phase 1 (warm-up):   Train only the classifier head. Backbone frozen.
                     High LR, fast convergence, avoids destroying pretrained features.

Phase 2 (fine-tune): Unfreeze top backbone blocks. Low LR with differential
                     learning rates (backbone LR << head LR).

Usage:
    python train.py --data_dir data/raw --epochs_warmup 10 --epochs_finetune 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models.model import build_model, NUM_CLASSES
from utils.dataset import get_dataloaders, CLASSES


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.split(":")[0],
                            enabled=(device != "cpu")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def per_class_accuracy(preds, labels):
    preds  = torch.tensor(preds)
    labels = torch.tensor(labels)
    results = {}
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        if mask.sum() == 0:
            results[cls] = None
        else:
            results[cls] = (preds[mask] == labels[mask]).float().mean().item()
    return results


# ── Main training function ────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Train] Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[Train] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[Train] Building model...")
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True, device=device)

    # Class-weighted loss to handle imbalanced datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Warm-up: train head only
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f" PHASE 1 — Head warm-up ({args.epochs_warmup} epochs)")
    print(f"{'='*55}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_warmup, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_warmup, eta_min=1e-5)

    for epoch in range(1, args.epochs_warmup + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, device, scaler)
        va_loss, va_acc, va_preds, va_labels = eval_epoch(model, val_loader,
                                                           criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{args.epochs_warmup}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}  "
              f"({elapsed:.1f}s)")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune: unfreeze top backbone blocks
    # ─────────────────────────────────────────────────────────────────────
    if args.epochs_finetune > 0:
        print(f"\n{'='*55}")
        print(f" PHASE 2 — Fine-tuning ({args.epochs_finetune} epochs)")
        print(f"{'='*55}")

        model.unfreeze_backbone(layers_from_end=args.unfreeze_layers)

        # Differential learning rates: backbone gets 10x lower LR than head
        backbone_params = list(model.features.parameters())
        head_params     = list(model.classifier.parameters())

        optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr_finetune / 10},
            {"params": head_params,     "lr": args.lr_finetune},
        ], weight_decay=1e-4)

        warmup_sched  = LinearLR(optimizer, start_factor=0.1, total_iters=3)
        cosine_sched  = CosineAnnealingLR(optimizer,
                                          T_max=args.epochs_finetune - 3,
                                          eta_min=1e-6)
        scheduler     = SequentialLR(optimizer,
                                     schedulers=[warmup_sched, cosine_sched],
                                     milestones=[3])

        for epoch in range(1, args.epochs_finetune + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                          optimizer, device, scaler)
            va_loss, va_acc, va_preds, va_labels = eval_epoch(model, val_loader,
                                                               criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:02d}/{args.epochs_finetune}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                  f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}  "
                  f"({elapsed:.1f}s)")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # ─────────────────────────────────────────────────────────────────────
    # Final test evaluation
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(" Final evaluation on test set")
    print(f"{'='*55}")

    model.load_state_dict(torch.load(output_dir / "best_model.pt",
                                     map_location=device))
    te_loss, te_acc, te_preds, te_labels = eval_epoch(model, test_loader,
                                                       criterion, device)
    print(f"\n  Test loss: {te_loss:.4f}")
    print(f"  Test acc:  {te_acc:.4f}")

    per_cls = per_class_accuracy(te_preds, te_labels)
    print("\n  Per-class accuracy:")
    for cls, acc in per_cls.items():
        bar = "█" * int((acc or 0) * 20)
        print(f"    {cls:<12s}  {bar:<20s}  {acc:.3f}" if acc else
              f"    {cls:<12s}  (no samples)")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved to {output_dir}/history.json")
    print(f"[Train] Best model saved to {output_dir}/best_model.pt")
    print(f"[Train] Best val acc: {best_val_acc:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lighting Condition Classifier")
    parser.add_argument("--data_dir",         default="data/raw")
    parser.add_argument("--output_dir",       default="models/checkpoints")
    parser.add_argument("--img_size",         type=int,   default=224)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--epochs_warmup",    type=int,   default=10)
    parser.add_argument("--epochs_finetune",  type=int,   default=20)
    parser.add_argument("--lr_warmup",        type=float, default=1e-3)
    parser.add_argument("--lr_finetune",      type=float, default=3e-4)
    parser.add_argument("--unfreeze_layers",  type=int,   default=3,
                        help="Number of top backbone blocks to unfreeze in phase 2")
    args = parser.parse_args()
    train(args)
