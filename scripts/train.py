"""Build the memory bank from good images and calibrate a decision threshold.

    python scripts/train.py --config config.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np

from anomaly import Config, PatchCoreDetector
from anomaly.data import get_train_paths, make_loader
from anomaly.utils import get_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"device: {device}")

    paths = get_train_paths(cfg.data_root)
    rng = np.random.RandomState(cfg.seed)
    order = rng.permutation(len(paths))
    n_val = int(round(len(paths) * cfg.val_fraction)) if len(paths) >= 10 else 0
    val_paths = [paths[i] for i in order[:n_val]]
    fit_paths = [paths[i] for i in order[n_val:]]
    print(f"good images: {len(paths)} -> {len(fit_paths)} for memory bank, {len(val_paths)} held out for threshold")

    det = PatchCoreDetector(cfg.backbone, cfg.layers, cfg.image_size, cfg.coreset_ratio,
                            cfg.blur_sigma, device=device)

    t0 = time.time()
    det.fit(make_loader(fit_paths, [0] * len(fit_paths), cfg.image_size, cfg.batch_size, cfg.num_workers))
    print(f"memory bank built in {time.time() - t0:.1f}s, shape {tuple(det.bank.shape)}")

    if val_paths:
        s = det.score_loader(make_loader(val_paths, [0] * len(val_paths), cfg.image_size,
                                         cfg.batch_size, cfg.num_workers), "calibrating").numpy()
        det.threshold = float(s.mean() + cfg.threshold_sigma * s.std())
        det.val_stats = {"mean": s.mean(), "std": s.std(), "max": s.max()}
        print(f"held-out good scores: mean={s.mean():.3f} std={s.std():.3f} max={s.max():.3f}")
        print(f"threshold = mean + {cfg.threshold_sigma}*std = {det.threshold:.3f}")
    else:
        print("Too few images to hold out a validation split; run evaluate.py --save-threshold to set a threshold.")

    from pathlib import Path
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
    det.save(cfg.model_path)
    print(f"saved model -> {cfg.model_path}")


if __name__ == "__main__":
    main()
