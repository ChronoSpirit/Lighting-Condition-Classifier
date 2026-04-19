# Lighting Condition Classifier

A deep learning pipeline that classifies images into five photometric lighting
categories in real time, using a fine-tuned EfficientNet-B0 CNN backbone and
OpenCV-based feature extraction.

## Classes

| Class | Description |
|---|---|
| `harsh` | Direct sunlight, hard shadows, high contrast |
| `soft` | Overcast / diffused light, even illumination |
| `backlit` | Light source behind subject, silhouette / halo |
| `low_light` | Dim / night conditions, high sensor noise |
| `mixed` | Multiple conflicting light sources |

## Architecture

```
Input Image (224×224 RGB)
        │
        ▼
EfficientNet-B0 Backbone (ImageNet pretrained)
  Conv stem + 7× MBConv blocks
        │
        ▼
Global Average Pooling  →  (1280,)
        │
        ▼
Dropout(0.3) → FC(1280→512) → BatchNorm → ReLU → Dropout(0.2)
        │
        ▼
FC(512 → 5)  →  Softmax
        │
        ▼
Lighting Class + Confidence
```

Trained in two phases:
1. **Warm-up (10 epochs):** Head only, backbone frozen. LR = 1e-3.
2. **Fine-tune (20 epochs):** Top 3 backbone blocks unfrozen. LR = 3e-4
   (backbone gets 10× lower LR). Cosine annealing with linear warm-up.

OpenCV features (luminance stats, gradient energy, shadow/highlight ratio,
backlight score) are extracted alongside CNN predictions and displayed in the
real-time HUD.

## Project Structure

```
lighting_classifier/
├── train.py              # Two-phase training loop
├── inference.py          # Real-time webcam / video inference with HUD
├── evaluate.py           # Confusion matrix, per-class metrics, plots
├── scrape_dataset.py     # Automatic dataset builder (Unsplash)
├── requirements.txt
├── data/
│   └── raw/
│       ├── harsh/
│       ├── soft/
│       ├── backlit/
│       ├── low_light/
│       └── mixed/
├── models/
│   ├── model.py          # LightingClassifier definition
│   └── checkpoints/      # Saved during training
├── utils/
│   ├── dataset.py        # Dataset class + augmentations
│   └── features.py       # OpenCV photometric feature extractors
└── results/              # Confusion matrix + training plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Build the dataset

Option A — Kaggle dataset from pratik2901/multiclass-weather-dataset, nikhil7280/weather-type-classification, zara2099/low-light-image-enhancement-dataset.

.kaggle folder in C: with a kaggle.json folder with the api key for users who need assistance using this option.

Option B — Bring your own images. Place them in the folder structure above.
Each subfolder name must match one of the five class names exactly.

## Step 2 — Train

```bash
python train.py \
    --data_dir data/raw \
    --epochs_warmup 10 \
    --epochs_finetune 20 \
    --batch_size 32
```

The best checkpoint is saved to `models/checkpoints/best_model.pt`.

## Step 3 — Evaluate

```bash
python evaluate.py \
    --checkpoint models/checkpoints/best_model.pt \
    --data_dir data/raw
```

Outputs:
- `results/confusion_matrix.png`
- `results/training_history.png`
- Per-class accuracy + macro ROC-AUC in console

## Step 4 — Real-time inference

```bash
# Webcam
python inference.py --checkpoint models/checkpoints/best_model.pt

# Video file
python inference.py --checkpoint models/checkpoints/best_model.pt \
                    --source video.mp4

# Single image
python inference.py --checkpoint models/checkpoints/best_model.pt \
                    --source photo.jpg --image
```

## OpenCV Feature Descriptors

| Feature | What it measures | Diagnostic value |
|---|---|---|
| `lum_mean` | Mean L* luminance | Low → low_light |
| `lum_std` | Luminance spread | High → harsh contrast |
| `grad_energy` | Sobel edge energy | High → hard shadow edges |
| `shadow_frac` | % pixels below threshold | High → backlit / low_light |
| `highlight_frac` | % pixels above threshold | High → harsh overexposure |
| `backlight_score` | Border/centre lum ratio | >1 → backlit |
| `rb_ratio` | Red/Blue channel ratio | Warm vs cool light temperature |

## Expected Performance

With ~200 images per class and 30 total training epochs on a GPU:

| Split | Accuracy |
|---|---|
| Train | ~92% |
| Val | ~85% |
| Test | ~83% |

`soft` and `mixed` are the most commonly confused classes (both involve
multi-directional, moderate illumination).
