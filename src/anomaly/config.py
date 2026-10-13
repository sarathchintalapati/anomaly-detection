from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    data_root: str = "data"
    image_size: int = 224
    batch_size: int = 8
    backbone: str = "resnet18"
    layers: List[int] = field(default_factory=lambda: [2, 3])
    coreset_ratio: float = 0.1
    blur_sigma: float = 4.0
    val_fraction: float = 0.1
    threshold_sigma: float = 3.0
    model_path: str = "models/patchcore.pt"
    output_dir: str = "outputs"
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
