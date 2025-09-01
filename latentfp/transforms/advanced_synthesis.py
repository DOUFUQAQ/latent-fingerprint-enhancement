from __future__ import annotations

import random
from typing import Tuple

try:  # OpenCV is optional: the pure-numpy helpers below import without it
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # OpenCV-backed helpers will raise if called
import numpy as np


def _ensure_float01(img: np.ndarray) -> np.ndarray:
	if img.dtype != np.float32:
		img = img.astype(np.float32)
	img = np.clip(img, 0.0, 255.0)
	if img.max() > 1.0:
		img = img / 255.0
	return img


def multiply_blend(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
	"""正片叠底混合模式，如论文中提到的"""
	fg = _ensure_float01(foreground)
	bg = _ensure_float01(background)
	# 正片叠底：result = fg * bg
	return (fg * bg).astype(np.float32)


def create_complex_background(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
	"""创建复杂背景纹理"""
	h, w = size
	
	# 基础噪声纹理
	noise = np.random.uniform(0.3, 0.8, (h, w)).astype(np.float32)
	
	# 添加结构化模式（模拟真实场景中的线条、纹理等）
	y, x = np.ogrid[:h, :w]
	
	# 斜线模式
	if random.random() < 0.4:
		angle = random.uniform(0, np.pi)
		freq = random.uniform(0.02, 0.1)
		pattern = 0.5 + 0.3 * np.sin(freq * (x * np.cos(angle) + y * np.sin(angle)))
		noise = noise * pattern
	
	# 圆形模式
	if random.random() < 0.3:
		cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
		radius = random.uniform(h//8, h//3)
		circle = np.sqrt((x - cx)**2 + (y - cy)**2)
		circle_pattern = 0.4 + 0.4 * np.exp(-circle / radius)
		noise = noise * circle_pattern
	
	# 添加一些随机块状结构
	for _ in range(random.randint(2, 8)):
		x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
		x2, y2 = random.randint(x1, w), random.randint(y1, h)
		intensity = random.uniform(0.2, 0.9)
		noise[y1:y2, x1:x2] *= intensity
	
	return np.clip(noise, 0.0, 1.0)


def paper_style_latent_synthesis(clean_fingerprint: np.ndarray) -> np.ndarray:
	"""根据论文方法合成潜指纹：使用正片叠底与复杂背景融合"""
	clean = _ensure_float01(clean_fingerprint)
	h, w = clean.shape
	
	# 创建复杂背景
	background = create_complex_background((h, w))
	
	# 使用正片叠底混合
	latent_base = multiply_blend(clean, background)
	
	# 添加额外的潜指纹特有的退化
	# 1. 对比度降低
	contrast_factor = random.uniform(0.5, 0.8)
	brightness_offset = random.uniform(-0.1, 0.1)
	latent_base = np.clip(contrast_factor * latent_base + brightness_offset, 0.0, 1.0)
	
	# 2. 轻微模糊
	if random.random() < 0.6:
		kernel_size = random.choice([3, 5])
		latent_base = cv2.GaussianBlur(latent_base, (kernel_size, kernel_size), 0)
	
	# 3. 噪声
	if random.random() < 0.7:
		noise_std = random.uniform(0.02, 0.08)
		noise = np.random.normal(0, noise_std, latent_base.shape).astype(np.float32)
		latent_base = np.clip(latent_base + noise, 0.0, 1.0)
	
	return latent_base.astype(np.float32)
