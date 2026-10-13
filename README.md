# Image Anomaly Detection (PatchCore-style)

Detect defects in images when you only have **good (defect-free) images for training** and just a few defect examples.

The model learns what "normal" looks like from good images only, then flags anything that doesn't match, and shows **where** (heatmap).
It is built to run on a laptop (tested design target: MacBook Pro, 8 GB RAM, CPU or Apple GPU via MPS).

## How it works

1. A frozen ImageNet-pretrained ResNet (default `resnet18`) converts each image into a grid of patch features.
2. Patch features from all good images are compressed into a small **memory bank** (greedy coreset).
3. For a new image, each patch is compared with its nearest neighbour in the memory bank. A large distance means "never seen in good images".
4. The image score is the maximum patch distance. If `score >= threshold`, the image is flagged as anomalous.

There is **no gradient training**, so "training" takes a minute or two and needs very little memory. It works well with ~200 images.

## Project layout

```
.
├── config.yaml             # all settings
├── requirements.txt
├── src/anomaly/            # library code (data, model, utils)
├── scripts/
│   ├── train.py            # build memory bank + calibrate threshold
│   ├── evaluate.py         # metrics + heatmaps on the test set
│   ├── predict.py          # score new image(s)
│   └── make_demo_data.py   # synthetic data to test the pipeline
├── data/                   # your images (git-ignored)
├── models/                 # saved model (git-ignored)
└── outputs/                # heatmaps, plots, csv (git-ignored)
```

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the pretrained ResNet weights (needs internet once).

## Data layout

Put your images here (png/jpg/bmp/tif):

```
data/
├── train/good/        # your ~200 defect-free images
└── test/
    ├── good/          # defect-free images NOT used in training
    └── defect/        # images with anomalies (any extra folders here also count as defect)
```

Folder names define the labels: `test/good` = normal, every other folder in `test/` = anomalous.
The data folders are git-ignored; only empty `.gitkeep` placeholders are committed.

## Quick start

```bash
# (optional) sanity check on synthetic data
python scripts/make_demo_data.py

# 1. train: builds the memory bank and sets a threshold
python scripts/train.py

# 2. evaluate on data/test
python scripts/evaluate.py

# 3. predict on new images
python scripts/predict.py path/to/image.png
python scripts/predict.py path/to/folder/
```

Outputs go to `outputs/eval/` (heatmaps, `roc.png`, `score_hist.png`, `scores.csv`) and `outputs/predictions/`.

## Threshold

`train.py` holds out 10% of the good images and sets `threshold = mean + 3 * std` of their scores. This uses no defect images, so it is safe, but may be too strict or too loose.
After checking `evaluate.py` (look at `score_hist.png`), you can:

- tune `threshold_sigma` in `config.yaml` and retrain, or
- run `python scripts/evaluate.py --save-threshold` to store the best-F1 threshold from the test set, or
- override ad hoc with `--threshold 1.8` in `evaluate.py` / `predict.py`.

With only a few defect images, test metrics are noisy and the best-F1 threshold is optimistic. Treat them as a rough guide.

## Tips for 8 GB RAM

- Defaults (`resnet18`, 224 px, batch 8) use roughly 1–2 GB.
- Out of memory? Lower `batch_size` to 2–4 or `coreset_ratio` to 0.05.
- Want more accuracy? Try `backbone: wide_resnet50_2` and `image_size: 256` (slower, more memory).
- Small defects missed? Increase `image_size` (e.g. 320) so they cover more pixels.
- Keep image framing/lighting consistent between good and test images. The method is sensitive to changes in the setup, which is also why it detects defects.
- `device: auto` picks Apple GPU (MPS) when available, otherwise CPU. Set `device: cpu` if you hit MPS issues.

## Pushing to git

```bash
git init
git add .
git commit -m "Initial commit: image anomaly detection"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`.gitignore` excludes images, model files (`*.pt`, `*.pth`, ...), `outputs/`, the virtual environment and Python caches. Run `git status` before the first push to confirm nothing large is staged.
