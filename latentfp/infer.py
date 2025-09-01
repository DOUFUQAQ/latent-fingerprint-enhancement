from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from latentfp.models.unet import UNetDualHead


def read_grayscale(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise FileNotFoundError(str(path))
	return (img.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)


def save_gray(path: Path, img01: np.ndarray) -> None:
	img = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
	cv2.imwrite(str(path), img)


def load_model(ckpt_path: Path, device: str) -> UNetDualHead:
	ckpt = torch.load(str(ckpt_path), map_location=device)
	# 兼容不同保存格式：{"model": ...} 或 {"generator": ...} 或直接state_dict
	meta = ckpt.get("meta", {"in_channels": 1, "base_channels": 32})
	model = UNetDualHead(in_channels=meta.get("in_channels", 1), base_channels=meta.get("base_channels", 32)).to(device)
	state = None
	if isinstance(ckpt, dict):
		if "model" in ckpt:
			state = ckpt["model"]
		elif "generator" in ckpt:
			state = ckpt["generator"]
		else:
			# 有些检查点直接就是state_dict
			# 检查是否像state_dict一样是{str: tensor}
			if all(isinstance(k, str) for k in ckpt.keys()):
				state = ckpt
	if state is None:
		raise KeyError("Unsupported checkpoint format: expect keys 'model' or 'generator'.")
	model.load_state_dict(state, strict=False)
	model.eval()
	return model


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=Path, required=True)
	parser.add_argument("--ckpt", type=Path, default=Path("checkpoint_unet_dualhead.pt"))
	parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
	args = parser.parse_args()

	args.out_dir.mkdir(parents=True, exist_ok=True)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = load_model(args.ckpt, device)
	img = read_grayscale(args.input)
	t = torch.from_numpy(img).unsqueeze(0).to(device)  # (B=1,1,H,W)
	with torch.no_grad():
		restored, ridge_logits = model(t)
		restored = torch.sigmoid(restored).cpu().numpy()[0, 0]
		ridge = torch.sigmoid(ridge_logits).cpu().numpy()[0, 0]

	# Normalize ridge map for visibility and also provide a binary version
	min_v, max_v = float(ridge.min()), float(ridge.max())
	range_v = max(1e-8, max_v - min_v)
	ridge_norm = (ridge - min_v) / range_v
	# Otsu threshold on 8-bit scaled image
	r8 = (ridge_norm * 255.0).astype(np.uint8)
	_, ridge_bin = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	ridge_bin = (ridge_bin.astype(np.float32) / 255.0)

	save_gray(args.out_dir / "restored.png", restored)
	save_gray(args.out_dir / "ridge.png", ridge_norm)  # visibility enhanced
	save_gray(args.out_dir / "ridge_bin.png", ridge_bin)
	print("Saved outputs to", args.out_dir)


if __name__ == "__main__":
	main()


