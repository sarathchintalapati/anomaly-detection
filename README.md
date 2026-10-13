# Image Defect Detection Project

This project explores three different ways to separate good vs bad parts in metal-plate inspection images.

## 1) Original anomaly-detection approach

This is the original PatchCore-style method already described in the project.

### Idea
- Use only good images during training.
- Learn what a normal plate looks like.
- For a new image, compare local patch features to the normal memory bank.
- A large distance indicates an anomaly.
- The image score is computed from the most unusual patch, and the result is shown as a heatmap.

### Typical workflow
1. A pretrained ResNet extracts patch features from each image.
2. Good images are compressed into a memory bank.
3. New images are scored against this memory bank.
4. A threshold decides whether the image is normal or abnormal.

### Best use case
- Suitable when you only have good images and no pixel-wise masks.
- Good for general anomaly detection, not fine defect segmentation.

This is the baseline approach and is the one already described in the existing project README and training pipeline.

---

## 2) Supervised segmentation approach with ground-truth defect masks

This second approach is the segmentation pipeline used to isolate the object, train a model on defect masks, and then use the segmented NG region to decide pass/fail.

### Idea
- Remove the background first to isolate the metal plate.
- Crop to the object bounding box and resize the plate region.
- Train a segmentation model to predict defect pixels inside the plate area.
- Use the ground-truth masks on bad images to supervise learning.
- For each predicted image, compute the NG (defective) percentage inside the plate.
- Use a threshold on the NG percentage to classify the image as good or bad.

### Typical workflow
1. Background removal isolates the main object from the surrounding scene.
2. The plate is cropped and resized to a fixed training size.
3. A U-Net style network is trained with defect masks for bad images.
4. The network predicts defective pixels on the plate interior.
5. The NG area is compared to the total plate area.
6. If the percentage is above a threshold, the image is flagged as bad.

### Best use case
- Best when defect masks are available for training.
- More controlled and more interpretable than purely unsupervised anomaly detection.
- Stronger for known defect types and pixel-level segmentation.

This is the approach implemented in the defect segmentation notebook and associated training/inference scripts.

---

## 3) Canny-edge approach after background removal

This third approach is a simple rule-based method that removes the background, applies Canny edge detection, and then uses a mathematical criterion to decide good vs bad.

### Idea
- Remove the background to isolate the object.
- Convert the segmented object to grayscale.
- Smooth the image slightly to reduce noise.
- Apply Canny edge detection to find the object boundaries and internal edges.
- Compute a simple edge-based metric (for example, edge density, contour length, or number of edge pixels in the object region).
- Compare that metric to a threshold to separate good and bad images.

### Typical workflow
1. Background removal extracts the object mask.
2. The segmented object is normalized and converted to grayscale.
3. Canny edge detection extracts contour information.
4. A simple mathematical rule is applied to the edge map.
5. If the edge metric exceeds or falls below a threshold, the part is classified as good or bad.

### Best use case
- Lightweight and explainable.
- Good when the defect produces a clear boundary or contour change.
- Easier to interpret than a full neural network classifier.

This is the simpler alternative that was explored as a direct edge-based decision rule after segmentation.

---

## Repository contents

```
.
├── anomaly_detection_notebook.ipynb   # original PatchCore-style anomaly detection
├── defect_segmentation.ipynb           # supervised segmentation approach
├── background_removal_canny.ipynb      # background removal + Canny edge pipeline
├── config.yaml                         # configuration for the experiments
├── requirements.txt
├── src/anomaly/                        # shared project code
├── scripts/
│   ├── train.py                        # original anomaly-model training
│   ├── evaluate.py                    # validation and scoring
│   ├── predict.py                     # inference on new images
│   └── make_demo_data.py              # synthetic demo data generator
├── data/                              # project datasets
├── models/                            # trained model weights and outputs
├── outputs/                           # evaluation artifacts and predictions
├── README.md                          # project overview
└── .gitignore
```

---

## Practical summary

The project moved through three increasingly task-specific ideas:

1. Unsupervised anomaly detection: learn normality from good images only.
2. Supervised defect segmentation: train on masks and threshold NG percentage.
3. Edge-based rule: remove background, then detect edges and apply a simple mathematical decision function.

In practice, the first method is the most general unsupervised baseline, the second is the most accurate when defect masks exist, and the third is the simplest and most interpretable engineering rule for rapid inspection.

---

## Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the relevant notebook or script depending on the approach you want to test.

- Original anomaly model: `anomaly_detection_notebook.ipynb`
- Segmentation approach: `defect_segmentation.ipynb`
- Canny edge approach: `background_removal_canny.ipynb`

---

## Notes

- The PatchCore-style approach is good when you have no defect annotations.
- The segmentation approach is better when you have labeled masks and want pixel-level defect localization.
- The Canny approach is a lightweight rule-based alternative focused on contour structure after background removal.
