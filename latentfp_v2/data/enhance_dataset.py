from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from latentfp_v2.transforms.synthesis_v2 import synthesize_latent_v2, estimate_ridge_v2


def _list_pngs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.png")])


def _read_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0)


def _random_crop(img: np.ndarray, size: Optional[int]) -> np.ndarray:
    if size is None:
        return img
    h, w = img.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w = img.shape
    y = np.random.randint(0, h - size + 1)
    x = np.random.randint(0, w - size + 1)
    return img[y:y+size, x:x+size]


@dataclass
class EnhanceDatasetCfg:
    root: Path
    crop: Optional[int] = 256
    training: bool = True


class LatentEnhanceDataset(Dataset):
    def __init__(self, root_dir: str | Path, crop: Optional[int] = 256, training: bool = True) -> None:
        self.cfg = EnhanceDatasetCfg(root=Path(root_dir), crop=crop, training=training)
        self.clean_paths = _list_pngs(self.cfg.root)
        if not self.clean_paths:
            raise RuntimeError(f"No PNG found under {self.cfg.root}")

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean = _read_gray01(self.clean_paths[idx])
        if self.cfg.crop is not None:
            clean = _random_crop(clean, self.cfg.crop)

        latent = synthesize_latent_v2(clean)
        ridge = estimate_ridge_v2(clean)

        return {
            "latent": torch.from_numpy(latent).unsqueeze(0).float(),
            "clean": torch.from_numpy(clean).unsqueeze(0).float(),
            "ridge": torch.from_numpy(ridge).unsqueeze(0).float(),
        }


