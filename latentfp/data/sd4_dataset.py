from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from latentfp.transforms.latent_synthesis import synthesize_latent, estimate_ridgemap
from latentfp.transforms.advanced_synthesis import paper_style_latent_synthesis


def _list_pngs(root: Path) -> List[Path]:
	paths: List[Path] = []
	for p in root.rglob("*.png"):
		paths.append(p)
	paths.sort()
	return paths


def _read_grayscale(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise FileNotFoundError(f"Failed to read image: {path}")
	return img.astype(np.float32) / 255.0


def _random_crop(img: np.ndarray, crop_size: Optional[int]) -> np.ndarray:
	if crop_size is None:
		return img
	h, w = img.shape
	if h < crop_size or w < crop_size:
		# Pad reflect to reach crop size
		pad_h = max(0, crop_size - h)
		pad_w = max(0, crop_size - w)
		img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
		h, w = img.shape
		
	y = np.random.randint(0, h - crop_size + 1)
	x = np.random.randint(0, w - crop_size + 1)
	return img[y : y + crop_size, x : x + crop_size]


@dataclass
class SD4DatasetConfig:
	root_dir: Path
	crop_size: Optional[int] = 256
	training: bool = True


class SD4Fingerprints(Dataset):
	def __init__(self, root_dir: str | Path, crop_size: Optional[int] = 256, training: bool = True, 
	             latent_transform=None) -> None:
		self.cfg = SD4DatasetConfig(root_dir=Path(root_dir), crop_size=crop_size, training=training)
		self.image_paths: List[Path] = _list_pngs(self.cfg.root_dir)
		self.latent_transform = latent_transform or synthesize_latent
		if len(self.image_paths) == 0:
			raise RuntimeError(f"No PNG images found under: {self.cfg.root_dir}")

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		path = self.image_paths[idx]
		clean = _read_grayscale(path)
		if self.cfg.training:
			clean = _random_crop(clean, self.cfg.crop_size)
		else:
			if self.cfg.crop_size is not None:
				clean = _random_crop(clean, self.cfg.crop_size)

		latent = self.latent_transform(clean)
		ridge = estimate_ridgemap(clean)

		# Ensure shapes match and are float32
		clean_t = torch.from_numpy(clean).unsqueeze(0).float()
		latent_t = torch.from_numpy(latent).unsqueeze(0).float()
		ridge_t = torch.from_numpy(ridge).unsqueeze(0).float()

		return {
			"latent": latent_t,
			"clean": clean_t,  # 使用 'clean' 而不是 'target' 以与新训练脚本兼容
			"target": clean_t,  # 保持向后兼容
			"ridge": ridge_t,
			"filename": torch.tensor(0),  # placeholder to keep types consistent
		}

# 为了兼容性，添加别名
SD4Dataset = SD4Fingerprints


