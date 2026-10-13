"""Create a small synthetic dataset to check that the pipeline runs end to end.

    python scripts/make_demo_data.py --root data
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def base_image(rng, size=256):
    y, x = np.mgrid[0:size, 0:size]
    img = 120 + 40 * np.sin(x / 9.0) * np.cos(y / 11.0) + rng.normal(0, 4, (size, size))
    img = np.stack([img, img * 0.9, img * 0.8], -1)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def add_defect(img, rng):
    d = ImageDraw.Draw(img)
    x, y = rng.integers(40, 200, 2)
    if rng.random() < 0.5:
        d.line([x, y, x + rng.integers(20, 50), y + rng.integers(-20, 20)], fill=(20, 20, 20), width=3)
    else:
        d.ellipse([x, y, x + 22, y + 22], fill=(230, 60, 40))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test-good", type=int, default=15)
    ap.add_argument("--n-test-defect", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    root = Path(args.root)
    for sub, n, defect in [("train/good", args.n_train, False), ("test/good", args.n_test_good, False),
                           ("test/defect", args.n_test_defect, True)]:
        (root / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = base_image(rng)
            if defect:
                img = add_defect(img, rng)
            img.save(root / sub / f"{i:03d}.png")
    print(f"demo data written to {root}/")


if __name__ == "__main__":
    main()
