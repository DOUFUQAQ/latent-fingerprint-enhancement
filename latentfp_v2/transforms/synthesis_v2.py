from __future__ import annotations

import random
from typing import Tuple

try:  # OpenCV is optional: the pure-numpy helpers below import without it
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # OpenCV-backed helpers will raise if called
import numpy as np


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = np.clip(img, 0.0, 255.0) / 255.0
    return np.clip(img, 0.0, 1.0)


def multiply_blend(fg01: np.ndarray, bg01: np.ndarray) -> np.ndarray:
    return np.clip(fg01 * bg01, 0.0, 1.0)


def synthesize_latent_v2(clean01: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
    """更稳定的潜指纹合成：复杂背景 + 正片叠底 + 轻度模糊/噪声/对比度退化"""
    if rng is None:
        rng = random
    clean = _to_float01(clean01)
    h, w = clean.shape

    # 背景：Perlin-like 简化（多尺度噪声叠加）
    bg = np.zeros((h, w), dtype=np.float32)
    for k in [64, 32, 16, 8]:
        noise = cv2.resize(np.random.uniform(0.3, 0.9, (max(1, h // k), max(1, w // k))).astype(np.float32), (w, h))
        bg += noise * (0.5 if k >= 32 else 0.25)
    bg = np.clip(bg / bg.max(), 0.2, 1.0)

    latent = multiply_blend(clean, bg)

    # 退化：对比度降低、亮度扰动
    alpha = rng.uniform(0.5, 0.8)
    beta = rng.uniform(-0.1, 0.1)
    latent = np.clip(alpha * latent + beta, 0.0, 1.0)

    # 轻度模糊
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        latent = cv2.GaussianBlur(latent, (k, k), 0)

    # 轻度噪声
    if rng.random() < 0.6:
        sigma = rng.uniform(0.01, 0.03)
        latent = np.clip(latent + np.random.normal(0.0, sigma, latent.shape).astype(np.float32), 0.0, 1.0)

    return latent.astype(np.float32)


def estimate_ridge_v2(clean01: np.ndarray) -> np.ndarray:
    """稳健脊线估计：CLAHE + 自适应阈值 + 细化"""
    img = _to_float01(clean01)
    img8 = (img * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(img8)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    bw = (bw > 0).astype(np.uint8) * 255

    # 细化：Zhang-Suen approx via OpenCV morphology
    prev = np.zeros_like(bw)
    skel = bw.copy()
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(skel, k)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, k)
        temp = cv2.subtract(eroded, opened)
        prev = skel
        skel = cv2.subtract(skel, temp)
        if cv2.countNonZero(prev - skel) == 0:
            break
    return (skel > 0).astype(np.float32)


