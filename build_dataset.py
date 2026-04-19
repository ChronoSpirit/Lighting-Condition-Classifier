"""
build_dataset.py — Download multiple Kaggle datasets and auto-organize
images into the 5 lighting condition classes needed for train.py.

Datasets used:
  1. pratik2901/multiclass-weather-dataset
       cloudy  → soft
       shine   → harsh
       sunrise → backlit
       rainy   → soft (secondary)

  2. zara2099/low-light-image-enhancement-dataset
       low_light → low_light

  3. nikhil7280/weather-type-classification
       Cloudy  → soft
       Sunny   → harsh
       Rainy   → soft (secondary)
       Foggy   → mixed

SETUP (one-time):
  pip install kaggle
  - Go to kaggle.com → Your profile → Settings → API → "Create New Token"
  - This downloads kaggle.json. Place it at:
      Windows:  C:/Users/<you>/.kaggle/kaggle.json
      Mac/Linux: ~/.kaggle/kaggle.json
  - Then run:  python build_dataset.py

OUTPUT:
  data/raw/
    harsh/       (shine + sunny images)
    soft/        (cloudy + overcast images)
    backlit/     (sunrise / contre-jour images)
    low_light/   (low-light enhancement dataset)
    mixed/       (foggy + rainy images — mixed/uneven light)
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR    = Path("data/raw")
STAGE_DIR  = Path("data/_staging")
MAX_PER_CLASS = 300   # cap per class to keep things balanced

# Map: (staging_subfolder_glob, target_class)
# Each tuple tells the organizer where to look and where to put it.
# Map subfolder NAMES (not paths) to lighting classes.
# The script will search the entire staging tree for folders with these names.
FOLDER_CLASS_MAP = {
    "shine":    "harsh",
    "Shine":    "harsh",
    "sunny":    "harsh",
    "Sunny":    "harsh",
    "cloudy":   "soft",
    "Cloudy":   "soft",
    "overcast": "soft",
    "sunrise":  "backlit",
    "Sunrise":  "backlit",
    "rainy":    "mixed",
    "Rainy":    "mixed",
    "Rain":     "mixed",
    "rain":     "mixed",
    "foggy":    "mixed",
    "Foggy":    "mixed",
    "lowlight": "low_light",
}

DATASETS = [
    "pratik2901/multiclass-weather-dataset",
    "nikhil7280/weather-type-classification",
    "zara2099/low-light-image-enhancement-dataset",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_kaggle():
    try:
        import kaggle  # noqa
    except ImportError:
        print("ERROR: kaggle package not found.")
        print("  Run:  pip install kaggle")
        raise SystemExit(1)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: kaggle.json not found.")
        print(f"  Expected at: {kaggle_json}")
        raise SystemExit(1)
    os.chmod(kaggle_json, 0o600)
    print("[Setup] Kaggle credentials found.")


def download_datasets():
    import kaggle
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        name = dataset.split("/")[1]
        dest = STAGE_DIR / name
        if dest.exists() and any(dest.iterdir()):
            print(f"[Download] Already exists, skipping: {name}")
            continue
        print(f"[Download] {dataset} ...")
        dest.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
        print(f"           Done → {dest}")


def find_images(folder: Path):
    """Recursively find all image files under a folder."""
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in VALID_EXTS and p.is_file()
    ]


def organize():
    """
    Walk the entire staging tree, match any folder whose name is in
    FOLDER_CLASS_MAP, and copy its images into data/raw/<class>/.
    This is robust to whatever subfolder structure Kaggle unpacks.
    """
    for cls in ["harsh", "soft", "backlit", "low_light", "mixed"]:
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: 0 for cls in ["harsh", "soft", "backlit", "low_light", "mixed"]}

    # Walk every directory under staging
    matched_dirs = []
    for dirpath, dirnames, _ in os.walk(STAGE_DIR):
        for dirname in dirnames:
            if dirname in FOLDER_CLASS_MAP:
                matched_dirs.append((Path(dirpath) / dirname,
                                     FOLDER_CLASS_MAP[dirname]))

    if not matched_dirs:
        print("[Organize] WARNING: No matching folders found in staging area.")
        print(f"           Staging contents:")
        for p in STAGE_DIR.rglob("*"):
            if p.is_dir():
                print(f"             {p.relative_to(STAGE_DIR)}")
        return counts

    for src_dir, target_cls in matched_dirs:
        images = find_images(src_dir)
        random.shuffle(images)
        print(f"[Organize] {src_dir.relative_to(STAGE_DIR)} → {target_cls}  ({len(images)} images)")

        for img_path in images:
            if counts[target_cls] >= MAX_PER_CLASS:
                break
            dest_name = f"{src_dir.name}_{img_path.name}"
            dest_path = OUT_DIR / target_cls / dest_name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                counts[target_cls] += 1

    return counts


def patch_low_light():
    """
    The low-light dataset may have subfolders like 'low/' and 'high/'.
    We only want the 'low' images. This finds them automatically.
    """
    staging = STAGE_DIR / "low-light-image-enhancement-dataset"
    if not staging.exists():
        return

    lowlight_staging = STAGE_DIR / "lowlight"
    lowlight_staging.mkdir(exist_ok=True)

    # Look for any subfolder named 'low', 'dark', 'input', etc.
    candidates = ["low", "dark", "input", "Low", "Dark", "Input", "lowlight"]
    for sub in candidates:
        src = staging / sub
        if src.exists():
            print(f"[Patch]  Found low-light images in: {src}")
            for img in find_images(src):
                shutil.copy2(img, lowlight_staging / img.name)
            return

    # Fallback: copy everything from the dataset root
    print("[Patch]  No 'low' subfolder found — copying all images from dataset root.")
    for img in find_images(staging):
        shutil.copy2(img, lowlight_staging / img.name)


def print_summary(counts):
    print("\n" + "=" * 45)
    print("  Dataset build complete")
    print("=" * 45)
    total = 0
    for cls, n in counts.items():
        bar = "█" * (n // 10)
        print(f"  {cls:<12s}  {bar:<30s}  {n}")
        total += n
    print(f"\n  Total images: {total}")
    print(f"  Location:     {OUT_DIR.resolve()}")

    low = [cls for cls, n in counts.items() if n < 50]
    if low:
        print(f"\n  WARNING: Low image count for: {', '.join(low)}")
        print("  Consider getting a Kaggle API key and adding unsplash images,")
        print("  or manually adding images for those classes.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("\n=== Lighting Condition Dataset Builder ===\n")
    check_kaggle()
    download_datasets()
    patch_low_light()
    counts = organize()
    print_summary(counts)