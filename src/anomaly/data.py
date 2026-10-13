from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def list_images(folder) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], image_size: int):
        self.paths, self.labels = list(paths), list(labels)
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i], str(self.paths[i])


def get_train_paths(data_root) -> List[Path]:
    """Good (defect-free) training images: data/train/good (or data/train)."""
    root = Path(data_root) / "train"
    paths = list_images(root / "good") or list_images(root)
    if not paths:
        raise FileNotFoundError(f"No training images found in {root}/good. See README for the folder layout.")
    return paths


def get_test_items(data_root) -> Tuple[List[Path], List[int]]:
    """data/test/good -> label 0, every other folder in data/test (e.g. defect/) -> label 1."""
    root = Path(data_root) / "test"
    paths, labels = [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        label = 0 if sub.name.lower() == "good" else 1
        for p in list_images(sub):
            paths.append(p)
            labels.append(label)
    if not paths:
        raise FileNotFoundError(f"No test images found in {root}. See README for the folder layout.")
    return paths, labels


def make_loader(paths, labels, image_size, batch_size, num_workers=0) -> DataLoader:
    ds = ImageDataset(paths, labels, image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
