"""Reference implementations of the training objectives used by this project.

The training scripts (``latentfp/train.py``, ``latentfp/train_gan.py``,
``latentfp/train_progressive_gan.py``) each carried their own inline copy of
the SSIM / Dice formulas, which made them impossible to check without a GPU.
This module keeps one numpy implementation of the maths - used by the unit
tests and by any post-hoc evaluation - plus an optional torch version that
matches the tensors the training loops already use.

Everything here is float math on ``[0, 1]`` images unless stated otherwise.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np

__all__ = [
    "box_filter",
    "ssim_map",
    "ssim_loss",
    "dice_coefficient",
    "dice_loss",
    "psnr",
    "normalize_weights",
    "combine_losses",
    "torch_available",
    "ssim_loss_torch",
    "dice_loss_torch",
]

# SSIM stabilisers, as in the original Wang et al. formulation with L = 1.
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def box_filter(image: np.ndarray, window: int) -> np.ndarray:
    """Mean filter with ``window`` x ``window`` kernel, stride 1, reflect pad.

    This is the numpy equivalent of ``F.avg_pool2d(image, window, 1,
    window // 2)`` used by the training loops.  Implemented with an integral
    image so it stays O(pixels) regardless of the window size.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"box_filter expects a 2-D array, got {image.shape!r}")
    if window <= 0 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window!r}")

    height, width = image.shape
    radius = window // 2
    padded = np.pad(image, radius, mode="reflect")

    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    total = (
        integral[window : window + height, window : window + width]
        - integral[0:height, window : window + width]
        - integral[window : window + height, 0:width]
        + integral[0:height, 0:width]
    )
    return total / float(window * window)


def ssim_map(pred: np.ndarray, target: np.ndarray, window: int = 11) -> np.ndarray:
    """Per-pixel SSIM map for two equally sized 2-D images."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape!r} vs target {target.shape!r}"
        )

    mu_pred = box_filter(pred, window)
    mu_target = box_filter(target, window)
    sigma_pred = box_filter(pred * pred, window) - mu_pred * mu_pred
    sigma_target = box_filter(target * target, window) - mu_target * mu_target
    sigma_cross = box_filter(pred * target, window) - mu_pred * mu_target

    numerator = (2.0 * mu_pred * mu_target + _C1) * (2.0 * sigma_cross + _C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + _C1) * (
        sigma_pred + sigma_target + _C2
    )
    return numerator / (denominator + 1e-12)


def ssim_loss(pred: np.ndarray, target: np.ndarray, window: int = 11) -> float:
    """``1 - mean(clamp(ssim_map, 0, 1))``, the value minimised in training."""
    ssim_values = np.clip(ssim_map(pred, target, window=window), 0.0, 1.0)
    return float(1.0 - ssim_values.mean())


def dice_coefficient(prob: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Soft Dice overlap of a ``[0, 1]`` probability map and a ``[0, 1]`` mask."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if prob.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prob {prob.shape!r} vs target {target.shape!r}"
        )
    intersection = float((prob * target).sum())
    denominator = float(prob.sum() + target.sum())
    return (2.0 * intersection + eps) / (denominator + eps)


def dice_loss(prob: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """``1 - dice_coefficient``."""
    return 1.0 - dice_coefficient(prob, target, eps=eps)


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB.  ``inf`` for identical images."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape!r} vs target {target.shape!r}"
        )
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range ** 2) / mse))


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Scale ``weights`` so their absolute values sum to 1.

    The training scripts hard-code raw multipliers (``lambda_l1 = 100``,
    ``lambda_perceptual = 10`` ...).  Ratios are what matter, but normalising
    makes runs with different term sets comparable and lets a single term be
    reported as a fraction of the total objective.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    total = float(sum(abs(float(value)) for value in weights.values()))
    if total == 0.0:
        raise ValueError(f"weight magnitudes sum to zero: {dict(weights)!r}")
    return {key: float(value) / total for key, value in weights.items()}


def combine_losses(
    terms: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Weighted sum of the individual loss terms.

    ``weights=None`` means an unweighted sum.  When weights are given the key
    sets must match exactly: a typo in one of the hard-coded lambdas used to
    silently drop a term from the objective instead of failing.
    """
    if not terms:
        raise ValueError("terms must not be empty")

    values = {key: float(value) for key, value in terms.items()}
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"loss term {key!r} is not finite: {value!r}")

    if weights is None:
        return float(sum(values.values()))

    missing = sorted(set(values) - set(weights))
    unknown = sorted(set(weights) - set(values))
    if missing:
        raise KeyError(f"no weight for loss term(s): {missing}")
    if unknown:
        raise KeyError(f"weight(s) given for unknown loss term(s): {unknown}")

    return float(sum(float(weights[key]) * values[key] for key in values))


# --------------------------------------------------------------- torch variants
def torch_available() -> bool:
    """True when ``torch`` can be imported (used to skip GPU-only tests)."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def ssim_loss_torch(pred, target, window_size: int = 11):
    """Torch port of :func:`ssim_loss`, matching the training loops' tensors.

    ``pred`` / ``target`` are ``(N, C, H, W)`` in ``[0, 1]``.  Raises
    ``ImportError`` when torch is unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("torch is required for ssim_loss_torch") from exc

    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, 1, pad)
    mu_target = F.avg_pool2d(target, window_size, 1, pad)
    sigma_pred = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_pred * mu_pred
    sigma_target = F.avg_pool2d(target * target, window_size, 1, pad) - mu_target ** 2
    sigma_cross = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_pred * mu_target

    numerator = (2 * mu_pred * mu_target + _C1) * (2 * sigma_cross + _C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + _C1) * (
        sigma_pred + sigma_target + _C2
    )
    ssim_map_t = numerator / (denominator + 1e-8)
    return (1.0 - ssim_map_t.clamp(0, 1)).mean()


def dice_loss_torch(logits, targets, eps: float = 1e-6):
    """Torch port of :func:`dice_loss`; takes *logits* and applies sigmoid."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("torch is required for dice_loss_torch") from exc

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    sum_probs = probs.sum(dim=(1, 2, 3))
    sum_targets = targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (sum_probs + sum_targets + eps)
    return 1.0 - dice.mean()
