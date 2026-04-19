"""
inference.py — Real-time lighting condition classification on webcam or video.

Runs the trained CNN on every frame and overlays:
  - Predicted class + confidence bar
  - Top-3 class probabilities
  - Key OpenCV-extracted photometric features
  - Frame rate counter

Usage:
    # Webcam
    python inference.py --checkpoint models/checkpoints/best_model.pt

    # Video file
    python inference.py --checkpoint models/checkpoints/best_model.pt \
                        --source path/to/video.mp4

    # Single image
    python inference.py --checkpoint models/checkpoints/best_model.pt \
                        --source path/to/image.jpg --image
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.model import build_model, NUM_CLASSES
from utils.dataset import CLASSES
from utils.features import (luminance_stats, shadow_highlight_ratio,
                             gradient_energy, backlight_score)


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "harsh":     (0,   100, 255),   # orange-red
    "soft":      (80,  200, 80),    # green
    "backlit":   (200, 80,  200),   # purple
    "low_light": (60,  60,  200),   # deep blue
    "mixed":     (200, 180, 0),     # amber
}

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Overlay drawing helpers ───────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=10):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_prob_bar(img, x, y, w, label, prob, color, bar_h=16):
    filled = int(w * prob)
    cv2.rectangle(img, (x, y), (x + w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + filled, y + bar_h), color, -1)
    text = f"{label:<10s} {prob*100:5.1f}%"
    cv2.putText(img, text, (x + w + 8, y + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame, probs, pred_idx, features, fps):
    h, w = frame.shape[:2]
    pred_class = CLASSES[pred_idx]
    pred_color = CLASS_COLORS[pred_class]

    # ── Main prediction label ──────────────────────────────────────────────
    panel_w, panel_h = 310, 195
    draw_rounded_rect(frame, 10, 10, 10 + panel_w, 10 + panel_h,
                      (20, 20, 20), alpha=0.65)

    cv2.putText(frame, pred_class.upper().replace("_", " "), (22, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, pred_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"conf: {probs[pred_idx]*100:.1f}%", (22, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Probability bars (top 3) ───────────────────────────────────────────
    sorted_idx = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(sorted_idx):
        draw_prob_bar(frame, 22, 90 + i * 28, 100,
                      CLASSES[idx], probs[idx],
                      CLASS_COLORS[CLASSES[idx]])

    # ── Feature readout ────────────────────────────────────────────────────
    feat_x, feat_y = 22, 173
    feat_text = (f"lum:{features['lum_mean']:.0f}  "
                 f"grad:{features['grad_mean']:.1f}  "
                 f"shad:{features['shadow_frac']:.2f}  "
                 f"back:{features['backlight_score']:.2f}")
    cv2.putText(frame, feat_text, (feat_x, feat_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)

    # ── FPS ────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"{fps:.1f} fps", (w - 85, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    # ── Colored border to signal class ────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), pred_color, 3)

    return frame


# ── Inference engine ──────────────────────────────────────────────────────────

class LightingInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = build_model(num_classes=NUM_CLASSES,
                                 freeze_backbone=False, device=device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Inference] Loaded checkpoint: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, bgr_frame: np.ndarray):
        """
        Run CNN on a single BGR frame.
        Returns (predicted_class_idx, probs_array, feature_dict).
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = INFERENCE_TRANSFORM(rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())

        # OpenCV feature extraction (runs on CPU, negligible cost)
        small = cv2.resize(bgr_frame, (224, 224))
        feats = {}
        feats.update(luminance_stats(small))
        feats.update(shadow_highlight_ratio(small))
        feats.update(gradient_energy(small))
        feats.update(backlight_score(small))

        return pred, probs, feats


# ── Main loops ────────────────────────────────────────────────────────────────

def run_video(engine: LightingInferenceEngine, source=0, output_path=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    prev_t = time.time()
    print("[Inference] Press 'q' to quit, 's' to save screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_idx, probs, feats = engine.predict(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        vis = draw_hud(frame, probs, pred_idx, feats, fps)

        if writer:
            writer.write(vis)

        cv2.imshow("Lighting Condition Classifier", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, vis)
            print(f"  Saved {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_image(engine: LightingInferenceEngine, image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)

    pred_idx, probs, feats = engine.predict(frame)
    vis = draw_hud(frame, probs, pred_idx, feats, fps=0)

    print(f"\nPrediction: {CLASSES[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")
    print("All probabilities:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12s}: {probs[i]*100:5.1f}%")

    out_path = Path(image_path).stem + "_result.jpg"
    cv2.imwrite(out_path, vis)
    print(f"\nResult saved to: {out_path}")
    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", default=0,
                        help="0 for webcam, or path to video/image")
    parser.add_argument("--image", action="store_true",
                        help="Treat --source as a single image file")
    parser.add_argument("--output", default=None,
                        help="Optional path to save annotated video output")
    args = parser.parse_args()

    engine = LightingInferenceEngine(args.checkpoint)

    if args.image:
        run_image(engine, args.source)
    else:
        source = args.source if args.source != "0" else 0
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        run_video(engine, source=source, output_path=args.output)
