import os

# Let PyTorch fall back to CPU for any op not yet implemented on Apple's GPU (MPS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_visualization(image_path, anomaly_map, score, threshold, out_path,
                       label=None, image_size=None) -> None:
    """Save a 3-panel figure: input | heatmap | overlay."""
    img = Image.open(image_path).convert("RGB")
    h, w = anomaly_map.shape
    img = img.resize((w, h))
    vmax = max(float(anomaly_map.max()), float(threshold) * 1.5, 1e-6)

    verdict = "ANOMALY" if score >= threshold else "OK"
    truth = "" if label is None else f" | truth: {'defect' if label else 'good'}"
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[1].imshow(anomaly_map, cmap="jet", vmin=0, vmax=vmax)
    axes[1].set_title("anomaly map")
    axes[2].imshow(img)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{Path(image_path).name}  score={score:.3f} (thr {threshold:.3f}) -> {verdict}{truth}",
                 fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
