"""Run the detector on new image(s).

    python scripts/predict.py path/to/image.png
    python scripts/predict.py path/to/folder/
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import torch
from PIL import Image

from anomaly import Config, PatchCoreDetector
from anomaly.data import build_transform, list_images
from anomaly.utils import get_device, save_visualization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder of images")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    det = PatchCoreDetector.load(cfg.model_path, device=get_device(cfg.device))
    threshold = args.threshold if args.threshold is not None else det.threshold
    if threshold is None:
        raise SystemExit("No threshold stored in the model. Pass --threshold or run evaluate.py --save-threshold.")

    inp = Path(args.input)
    paths = list_images(inp) if inp.is_dir() else [inp]
    tf = build_transform(det.image_size)
    out_dir = Path(cfg.output_dir) / "predictions"

    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        score, amap = det.predict(x)
        score = float(score[0])
        verdict = "ANOMALY" if score >= threshold else "ok"
        print(f"{p}: score={score:.3f} (threshold {threshold:.3f}) -> {verdict}")
        save_visualization(p, amap[0].numpy(), score, threshold, out_dir / f"{p.stem}.png")
    print(f"heatmaps saved to {out_dir}/")


if __name__ == "__main__":
    main()
