"""
features.py — OpenCV-based handcrafted feature extraction for lighting analysis.

These features can be used standalone (classical ML baseline) or concatenated
with CNN embeddings for a hybrid model. They directly measure the photometric
properties that define each lighting class.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


# ── Core photometric descriptors ──────────────────────────────────────────────

def _skewness(arr: np.ndarray) -> float:
    mu = arr.mean()
    sigma = arr.std() + 1e-6
    return float(((arr - mu) ** 3).mean() / sigma ** 3)


def luminance_stats(bgr: np.ndarray) -> Dict[str, float]:
    """
    Global and local brightness statistics via LAB color space.
    L channel isolates luminance from chroma — robust to color casts.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    return {
        "lum_mean":   float(L.mean()),
        "lum_std":    float(L.std()),
        "lum_min":    float(L.min()),
        "lum_max":    float(L.max()),
        "lum_range":  float(L.max() - L.min()),
        "lum_skew":   float(_skewness(L)),
    }


def shadow_highlight_ratio(bgr: np.ndarray,
                           shadow_thresh: int = 50,
                           highlight_thresh: int = 200) -> Dict[str, float]:
    """
    Fraction of pixels in shadow (<shadow_thresh) vs highlight (>highlight_thresh)
    regions. Harsh light → high highlight fraction + high contrast.
    Backlit → high shadow fraction in foreground.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    total = gray.size
    shadow_frac    = float((gray < shadow_thresh).sum() / total)
    highlight_frac = float((gray > highlight_thresh).sum() / total)
    midtone_frac   = 1.0 - shadow_frac - highlight_frac
    return {
        "shadow_frac":    shadow_frac,
        "highlight_frac": highlight_frac,
        "midtone_frac":   midtone_frac,
        "sh_ratio":       float(shadow_frac / (highlight_frac + 1e-6)),
    }


def gradient_energy(bgr: np.ndarray) -> Dict[str, float]:
    """
    Sobel edge energy — proxy for shadow sharpness.
    Harsh / directional light produces crisp, high-energy shadow edges.
    Soft / diffused light produces low gradient energy.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return {
        "grad_mean":   float(mag.mean()),
        "grad_std":    float(mag.std()),
        "grad_max":    float(mag.max()),
        "grad_energy": float((mag**2).mean()),
    }


def histogram_features(bgr: np.ndarray, bins: int = 32) -> Dict[str, float]:
    """
    Normalized histogram entropy and peak statistics per channel.
    Low-light images cluster in low bins; harsh light has bimodal distribution.
    """
    feats: Dict[str, float] = {}
    channel_names = ["B", "G", "R"]
    for i, name in enumerate(channel_names):
        hist = cv2.calcHist([bgr], [i], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
        feats[f"hist_entropy_{name}"] = entropy
        feats[f"hist_peak_{name}"]    = float(hist.argmax()) / bins
        feats[f"hist_spread_{name}"]  = float(hist.std())
    return feats


def backlight_score(bgr: np.ndarray, border_frac: float = 0.15) -> Dict[str, float]:
    """
    Compare mean luminance of border region vs centre.
    Backlit scenes have a bright periphery / dark centre.
    Score > 1.0  → border brighter than centre (backlit indicator).
    Score < 1.0  → centre brighter (front-lit or harsh).
    """
    h, w = bgr.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    border_lum = gray[border_mask].mean()
    center_lum = gray[~border_mask].mean()
    return {
        "backlight_score": float(border_lum / (center_lum + 1e-6)),
        "border_lum":      float(border_lum),
        "center_lum":      float(center_lum),
    }


def color_temperature_proxy(bgr: np.ndarray) -> Dict[str, float]:
    """
    R/B channel ratio as a rough color temperature proxy.
    Warm (tungsten / golden hour) → high R/B.
    Cool (overcast / fluorescent) → low R/B.
    """
    b = bgr[:, :, 0].astype(np.float32).mean()
    g = bgr[:, :, 1].astype(np.float32).mean()
    r = bgr[:, :, 2].astype(np.float32).mean()
    return {
        "rb_ratio": float(r / (b + 1e-6)),
        "rg_ratio": float(r / (g + 1e-6)),
        "gb_ratio": float(g / (b + 1e-6)),
    }


# ── Composite extractor ───────────────────────────────────────────────────────

def extract_all_features(image_path: str) -> np.ndarray:
    """
    Load an image and return a 1-D feature vector of all descriptors above.
    Used for the classical ML baseline (SVM / Random Forest).

    Returns:
        np.ndarray of shape (N_FEATURES,) — float32
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    bgr = cv2.resize(bgr, (224, 224))

    feats: Dict[str, float] = {}
    feats.update(luminance_stats(bgr))
    feats.update(shadow_highlight_ratio(bgr))
    feats.update(gradient_energy(bgr))
    feats.update(histogram_features(bgr))
    feats.update(backlight_score(bgr))
    feats.update(color_temperature_proxy(bgr))

    return np.array(list(feats.values()), dtype=np.float32)


def extract_from_frame(bgr: np.ndarray) -> np.ndarray:
    """Same as extract_all_features but accepts a pre-loaded BGR frame."""
    bgr = cv2.resize(bgr, (224, 224))
    feats: Dict[str, float] = {}
    feats.update(luminance_stats(bgr))
    feats.update(shadow_highlight_ratio(bgr))
    feats.update(gradient_energy(bgr))
    feats.update(histogram_features(bgr))
    feats.update(backlight_score(bgr))
    feats.update(color_temperature_proxy(bgr))
    return np.array(list(feats.values()), dtype=np.float32)


FEATURE_DIM = len(extract_from_frame(np.zeros((224, 224, 3), dtype=np.uint8)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def visualize_features(image_path: str) -> None:
    """
    Print a human-readable feature report for a single image.
    Useful for debugging and understanding model decisions.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    bgr = cv2.resize(bgr, (224, 224))

    groups = [
        ("Luminance",          luminance_stats(bgr)),
        ("Shadow/Highlight",   shadow_highlight_ratio(bgr)),
        ("Gradient Energy",    gradient_energy(bgr)),
        ("Backlight Score",    backlight_score(bgr)),
        ("Color Temperature",  color_temperature_proxy(bgr)),
    ]

    print(f"\n{'='*50}")
    print(f"Feature report: {image_path}")
    print(f"{'='*50}")
    for group_name, feat_dict in groups:
        print(f"\n  [{group_name}]")
        for k, v in feat_dict.items():
            print(f"    {k:<25s} {v:>8.4f}")
    print(f"\n  Total feature dim: {FEATURE_DIM}")