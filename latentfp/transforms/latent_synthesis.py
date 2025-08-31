from __future__ import annotations

import math
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


def random_gamma(image: np.ndarray, gamma_range: Tuple[float, float] = (0.6, 1.6), p: float = 0.5) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		gamma = random.uniform(*gamma_range)
		image = np.power(np.clip(image, 0.0, 1.0), gamma)
	return image


def random_contrast_brightness(image: np.ndarray, alpha_range=(0.6, 1.4), beta_range=(-0.15, 0.15), p: float = 0.7) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		alpha = random.uniform(*alpha_range)
		beta = random.uniform(*beta_range)
		image = np.clip(alpha * image + beta, 0.0, 1.0)
	return image


def gaussian_blur(image: np.ndarray, k_choices=(3, 5, 7), p: float = 0.5) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		k = random.choice(k_choices)
		image = cv2.GaussianBlur(image, (k, k), 0)
	return image


def motion_blur(image: np.ndarray, max_ksize: int = 9, p: float = 0.3) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		ksize = random.randrange(3, max_ksize + 1, 2)
		angle = random.uniform(0, math.pi)
		kernel = np.zeros((ksize, ksize), dtype=np.float32)
		center = ksize // 2
		sin_a, cos_a = math.sin(angle), math.cos(angle)
		for i in range(ksize):
			x = int(center + (i - center) * cos_a)
			y = int(center + (i - center) * sin_a)
			if 0 <= x < ksize and 0 <= y < ksize:
				kernel[y, x] = 1.0
		kernel = kernel / kernel.sum()
		image = cv2.filter2D(image, -1, kernel)
	return image


def add_gaussian_noise(image: np.ndarray, sigma_range=(0.0, 0.06), p: float = 0.7) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		sigma = random.uniform(*sigma_range)
		noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
		image = np.clip(image + noise, 0.0, 1.0)
	return image


def add_speckle_noise(image: np.ndarray, intensity_range=(0.0, 0.1), p: float = 0.4) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		intensity = random.uniform(*intensity_range)
		noise = image * (np.random.randn(*image.shape).astype(np.float32)) * intensity
		image = np.clip(image + noise, 0.0, 1.0)
	return image


def random_occlusions(image: np.ndarray, num_range=(1, 5), size_frac_range=(0.02, 0.15), p: float = 0.6) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		h, w = image.shape
		num = random.randint(*num_range)
		for _ in range(num):
			fh = int(random.uniform(*size_frac_range) * h)
			fw = int(random.uniform(*size_frac_range) * w)
			y0 = random.randint(0, max(0, h - fh))
			x0 = random.randint(0, max(0, w - fw))
			fill = random.uniform(0.0, 1.0)
			image[y0 : y0 + fh, x0 : x0 + fw] = fill
	return image


def vignette(image: np.ndarray, strength_range=(0.2, 0.6), p: float = 0.3) -> np.ndarray:
	image = _ensure_float01(image)
	if random.random() < p:
		h, w = image.shape
		y, x = np.ogrid[:h, :w]
		cy, cx = h / 2.0, w / 2.0
		dy = (y - cy) / (h / 2.0)
		dx = (x - cx) / (w / 2.0)
		r = np.sqrt(dx * dx + dy * dy)
		strength = random.uniform(*strength_range)
		mask = 1.0 - np.clip(r * strength, 0.0, 1.0)
		image = np.clip(image * mask.astype(np.float32), 0.0, 1.0)
	return image


def synthesize_latent(clean_image: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
	if rng is not None:
		random_state = random.getstate()
		random.setstate(rng.getstate())

	img = _ensure_float01(clean_image)
	img = random_contrast_brightness(img, p=0.9)
	img = random_gamma(img, p=0.7)
	img = gaussian_blur(img, p=0.6)
	img = motion_blur(img, p=0.35)
	img = add_gaussian_noise(img, p=0.8)
	img = add_speckle_noise(img, p=0.5)
	img = random_occlusions(img, p=0.6)
	img = vignette(img, p=0.4)

	# Reduce overall contrast to mimic latent prints
	alpha = random.uniform(0.5, 0.9)
	bias = random.uniform(-0.1, 0.1)
	img = np.clip(alpha * img + bias, 0.0, 1.0)

	if rng is not None:
		random.setstate(random_state)

	return img.astype(np.float32)


def _morphological_thinning(binary: np.ndarray) -> np.ndarray:
	# Zhang-Suen thinning using OpenCV operations
	binary = (binary > 0).astype(np.uint8) * 255
	prev = np.zeros_like(binary)
	skel = binary.copy()
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	# Iterative thinning via morphological operations
	while True:
		eroded = cv2.erode(skel, kernel)
		dilated = cv2.dilate(eroded, kernel)
		temp = cv2.subtract(skel, dilated)
		prev = skel
		skel = eroded
		if cv2.countNonZero(temp) == 0:
			break
	return (prev > 0).astype(np.float32)


def estimate_ridgemap(clean_image: np.ndarray) -> np.ndarray:
	img = _ensure_float01(clean_image)
	img8 = (img * 255.0).astype(np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	img_eq = clahe.apply(img8)
	blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
	bw = cv2.adaptiveThreshold(
		blur,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		15,
		2,
	)
	bw_float = (bw > 0).astype(np.float32)
	skel = _morphological_thinning(bw_float)
	return skel


