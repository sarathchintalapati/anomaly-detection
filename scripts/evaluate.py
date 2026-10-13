"""Evaluate on data/test (good/ vs defect folders), print metrics and save heatmaps.

    python scripts/evaluate.py
    python scripts/evaluate.py --save-threshold     # store the best-F1 threshold in the model file
"""
import _bootstrap  # noqa: F401
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from anomaly import Config, PatchCoreDetector
from anomaly.data import get_test_items, make_loader
from anomaly.utils import get_device, save_visualization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--threshold", type=float, default=None, help="override the threshold stored in the model")
    ap.add_argument("--save-threshold", action="store_true",
                    help="write the best-F1 threshold on the test set into the model file")
    ap.add_argument("--no-vis", action="store_true", help="skip saving heatmap images")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device(cfg.device)
    det = PatchCoreDetector.load(cfg.model_path, device=device)

    paths, labels = get_test_items(cfg.data_root)
    loader = make_loader(paths, labels, det.image_size, cfg.batch_size, cfg.num_workers)
    out_dir = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    thr_for_vis = args.threshold if args.threshold is not None else (det.threshold or 0.0)
    for imgs, lbls, ps in loader:
        s, maps = det.predict(imgs)
        scores.extend(s.tolist())
        if not args.no_vis:
            for i in range(len(ps)):
                save_visualization(ps[i], maps[i].numpy(), float(s[i]), thr_for_vis,
                                   out_dir / "heatmaps" / f"{Path(ps[i]).stem}.png", label=int(lbls[i]))
    scores, y = np.array(scores), np.array(labels)

    n_good, n_bad = int((y == 0).sum()), int((y == 1).sum())
    print(f"\ntest images: {n_good} good, {n_bad} defect")
    if n_good == 0 or n_bad == 0:
        print("Need both good and defect test images for metrics. Scores written to CSV only.")
    else:
        auc = roc_auc_score(y, scores)
        ap_score = average_precision_score(y, scores)
        print(f"image-level ROC-AUC: {auc:.4f}   average precision: {ap_score:.4f}")

        prec, rec, thr = precision_recall_curve(y, scores)
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        best = int(np.argmax(f1s))
        best_thr = float(thr[best])
        print(f"best-F1 threshold on test set: {best_thr:.3f} (F1={f1s[best]:.3f})")

        threshold = args.threshold if args.threshold is not None else det.threshold
        if threshold is None:
            threshold = best_thr
            print("model has no stored threshold; using best-F1 threshold")
        pred = (scores >= threshold).astype(int)
        print(f"\nat threshold {threshold:.3f}:")
        print(f"  precision {precision_score(y, pred, zero_division=0):.3f} | "
              f"recall {recall_score(y, pred, zero_division=0):.3f} | F1 {f1_score(y, pred, zero_division=0):.3f}")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
        print("\nNote: with few defect images these numbers are noisy; the best-F1 threshold is optimistic.")

        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", c="gray")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc.png", dpi=110)
        plt.close()

        plt.figure(figsize=(6, 3.5))
        plt.hist(scores[y == 0], bins=20, alpha=0.7, label="good")
        plt.hist(scores[y == 1], bins=20, alpha=0.7, label="defect")
        plt.axvline(threshold, c="k", ls="--", label="threshold")
        plt.xlabel("anomaly score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "score_hist.png", dpi=110)
        plt.close()

        if args.save_threshold:
            det.threshold = best_thr
            det.save(cfg.model_path)
            print(f"\nsaved threshold {best_thr:.3f} into {cfg.model_path}")

    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score"])
        for p, l, s in zip(paths, labels, scores):
            w.writerow([p, l, f"{s:.5f}"])
    print(f"\nresults written to {out_dir}/")


if __name__ == "__main__":
    main()
