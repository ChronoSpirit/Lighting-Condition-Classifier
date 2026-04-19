"""
dataset.py — Dataset loading, augmentation, and splitting for the
Lighting Condition Classifier.

Classes:
    harsh     — hard shadows, direct sunlight, strong artificial light
    soft      — overcast / diffused light, even illumination
    backlit   — light source behind subject, silhouette / halo effect
    low_light — dim / night conditions, high noise
    mixed     — multiple conflicting light sources
"""

import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

CLASSES = ["harsh", "soft", "backlit", "low_light", "mixed"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224):
    """
    Aggressive augmentations for training. We deliberately include
    lighting-related augmentations (brightness, contrast, gamma) to
    teach the model to focus on structural cues rather than absolute
    intensity values.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class LightingDataset(Dataset):
    """
    Expects the following directory layout:

        data/raw/
            harsh/
                img001.jpg
                img002.jpg
                ...
            soft/
            backlit/
            low_light/
            mixed/

    Each subfolder name must match one of the CLASSES strings above.
    """

    def __init__(self, root_dir: str, split: str = "train",
                 img_size: int = 224, val_frac: float = 0.15,
                 test_frac: float = 0.10, seed: int = 42):
        """
        Args:
            root_dir:  Path to data/raw/ folder.
            split:     One of 'train', 'val', 'test'.
            img_size:  Square resize target.
            val_frac:  Fraction of data held out for validation.
            test_frac: Fraction of data held out for test.
            seed:      Random seed for reproducible splits.
        """
        self.root = Path(root_dir)
        self.split = split
        self.transform = (get_train_transforms(img_size)
                          if split == "train"
                          else get_val_transforms(img_size))

        self.samples = []  # list of (path, label_idx)
        rng = np.random.default_rng(seed)

        for cls in CLASSES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                print(f"[WARNING] Class folder not found: {cls_dir}")
                continue

            paths = sorted([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ])
            paths = np.array(paths)
            indices = rng.permutation(len(paths))
            n = len(paths)
            n_test = max(1, int(n * test_frac))
            n_val  = max(1, int(n * val_frac))

            test_idx  = indices[:n_test]
            val_idx   = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]

            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
            chosen = paths[split_map[split]]
            label  = CLASS_TO_IDX[cls]
            self.samples.extend([(str(p), label) for p in chosen])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Convenience loaders ───────────────────────────────────────────────────────

def get_dataloaders(data_dir: str, img_size: int = 224,
                    batch_size: int = 32, num_workers: int = 4):
    """Returns train, val, test DataLoaders."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = LightingDataset(data_dir, split=split, img_size=img_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  {split:5s}: {len(ds):4d} samples")
    return loaders["train"], loaders["val"], loaders["test"]
