from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from latentfp_v2.models.enhance_unet import EnhanceUNetDualHead


def read_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0)[None, ...]


def save_gray(path: Path, img01: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8))


def load_model(ckpt: Path, device: str) -> EnhanceUNetDualHead:
    sd = torch.load(str(ckpt), map_location=device)
    if isinstance(sd, dict) and "generator" in sd:
        sd = sd["generator"]
    model = EnhanceUNetDualHead(1, 32).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoint_enhance_v2_last.pt"))
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_enhanced_v2"))
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, device)

    img = read_gray01(args.input)
    t = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        rec, rid_log = model(t)
        rec = torch.sigmoid(rec).cpu().numpy()[0, 0]
        rid = torch.sigmoid(rid_log).cpu().numpy()[0, 0]

    # 归一化与Otsu二值
    rmin, rmax = float(rid.min()), float(rid.max())
    rnorm = (rid - rmin) / max(1e-6, rmax - rmin)
    r8 = (rnorm * 255.0).astype(np.uint8)
    _, rbin = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rbin = rbin.astype(np.float32) / 255.0

    save_gray(args.out_dir / "restored.png", rec)
    save_gray(args.out_dir / "ridge.png", rnorm)
    save_gray(args.out_dir / "ridge_bin.png", rbin)
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()


