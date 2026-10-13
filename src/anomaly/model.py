"""PatchCore-style anomaly detector.

Training = NO gradient descent. A frozen, ImageNet-pretrained CNN turns every good image into a grid
of patch features; a subset of them (the "memory bank") is stored. At test time every patch of a new
image is compared with its nearest neighbour in the memory bank: patches that look unlike anything
seen in good images get a high score and light up in the anomaly map.
"""
import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "resnet18", layers: Sequence[int] = (2, 3), pretrained: bool = True):
        super().__init__()
        net = torchvision.models.get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.layers = list(layers)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # always stay in eval mode (frozen BatchNorm)
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch embeddings of shape (B, C, H, W)."""
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.layers:
                feats.append(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))  # local aggregation
            if i >= max(self.layers):
                break
        size = feats[0].shape[-2:]
        feats = [feats[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats[1:]]
        return torch.cat(feats, dim=1)


def greedy_coreset(x: torch.Tensor, n: int, device: torch.device, proj_dim: int = 128) -> torch.Tensor:
    """Greedy k-center selection (on a random projection for speed). Returns selected indices."""
    N = x.shape[0]
    if n >= N:
        return torch.arange(N)
    proj = torch.randn(x.shape[1], proj_dim) / math.sqrt(proj_dim)
    z = (x @ proj).to(device)
    idx = torch.empty(n, dtype=torch.long, device=device)
    idx[0] = torch.randint(N, (1,)).item()
    min_d = ((z - z[idx[0]]) ** 2).sum(1)
    for i in tqdm(range(1, n), desc="coreset", leave=False):
        j = torch.argmax(min_d)
        idx[i] = j
        min_d = torch.minimum(min_d, ((z - z[j]) ** 2).sum(1))
    return idx.cpu()


class PatchCoreDetector:
    def __init__(self, backbone="resnet18", layers=(2, 3), image_size=224, coreset_ratio=0.1,
                 blur_sigma=4.0, device=torch.device("cpu"), pretrained=True):
        self.backbone, self.layers = backbone, list(layers)
        self.image_size, self.coreset_ratio, self.blur_sigma = image_size, coreset_ratio, blur_sigma
        self.device = device
        self.extractor = FeatureExtractor(backbone, layers, pretrained).to(device)
        self.bank = None          # (M, C) tensor on device
        self.threshold = None     # float
        self.val_stats = {}

    # ---------------- training ----------------
    @torch.no_grad()
    def fit(self, loader) -> None:
        chunks = []
        for imgs, _, _ in tqdm(loader, desc="extracting features"):
            emb = self.extractor(imgs.to(self.device))                      # B,C,H,W
            chunks.append(emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1]).cpu())
        feats = torch.cat(chunks)
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        print(f"patch features: {len(feats):,} -> keeping {n_keep:,} (coreset {self.coreset_ratio:.0%})")
        keep = greedy_coreset(feats, n_keep, self.device)
        self.bank = feats[keep].to(self.device)

    # ---------------- inference ----------------
    @torch.no_grad()
    def _nn_distance(self, patches: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        out = []
        for i in range(0, patches.shape[0], chunk):
            d = torch.cdist(patches[i:i + chunk], self.bank)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        """imgs: (B,3,H,W) normalized. Returns (scores (B,), anomaly_maps (B,H,W)) on CPU."""
        assert self.bank is not None, "Model is not fitted/loaded."
        emb = self.extractor(imgs.to(self.device))
        B, C, H, W = emb.shape
        d = self._nn_distance(emb.permute(0, 2, 3, 1).reshape(-1, C)).reshape(B, 1, H, W)
        maps = F.interpolate(d, size=imgs.shape[-2:], mode="bilinear", align_corners=False).cpu()
        k = int(4 * self.blur_sigma) * 2 + 1
        maps = TF.gaussian_blur(maps, kernel_size=k, sigma=self.blur_sigma)
        scores = maps.flatten(1).max(dim=1).values
        return scores, maps[:, 0]

    def score_loader(self, loader, desc="scoring") -> torch.Tensor:
        scores = []
        for imgs, _, _ in tqdm(loader, desc=desc, leave=False):
            s, _ = self.predict(imgs)
            scores.append(s)
        return torch.cat(scores)

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        torch.save({
            "bank": self.bank.cpu(),
            "backbone": self.backbone,
            "layers": self.layers,
            "image_size": self.image_size,
            "coreset_ratio": self.coreset_ratio,
            "blur_sigma": self.blur_sigma,
            "threshold": None if self.threshold is None else float(self.threshold),
            "val_stats": {k: float(v) for k, v in self.val_stats.items()},
        }, path)

    @classmethod
    def load(cls, path: str, device=torch.device("cpu")) -> "PatchCoreDetector":
        ck = torch.load(path, map_location="cpu")
        det = cls(ck["backbone"], ck["layers"], ck["image_size"], ck["coreset_ratio"],
                  ck["blur_sigma"], device=device)
        det.bank = ck["bank"].to(device)
        det.threshold = ck["threshold"]
        det.val_stats = ck.get("val_stats", {})
        return det
