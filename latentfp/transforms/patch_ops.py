"""Sliding-window patch bookkeeping for the sparse-representation enhancer.

Every patch based stage of this project (``repair/gabor_sparse_coding.py``,
``repair/multi_scale_refinement.py``, ``latentfp/transforms/latent_synthesis.py``)
does the same three things:

1. walk the image with a fixed patch size / stride,
2. accumulate each reconstructed patch into an output buffer,
3. divide by the per-pixel hit count so overlapping patches are averaged
   instead of summed.

Steps 1-3 are pure numpy, so they live here and are unit-tested directly
instead of being re-derived inside the OpenCV / scikit-learn code paths.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "iter_patch_positions",
    "count_patch_hits",
    "accumulate_patch",
    "weighted_average",
    "reconstruct_overlapping",
]

Position = Tuple[int, int]


def iter_patch_positions(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Position]:
    """Top-left corners of a full-cover sliding window.

    Only *complete* patches are emitted, i.e. ``y + patch_size <= height``;
    the same convention as ``range(0, h - patch_size + 1, stride)``.  A
    partial border patch would be silently zero-padded, which biases the
    reconstruction near the edges.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride!r}")

    rows = range(0, height - patch_size + 1, stride)
    cols = range(0, width - patch_size + 1, stride)
    return [(y, x) for y in rows for x in cols]


def count_patch_hits(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """Number of patches covering each pixel (the averaging denominator)."""
    hits = np.zeros((height, width), dtype=np.float64)
    for y, x in iter_patch_positions(height, width, patch_size, stride):
        hits[y : y + patch_size, x : x + patch_size] += 1.0
    return hits


def accumulate_patch(
    accumulator: np.ndarray,
    weight: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    patch_weight: float = 1.0,
) -> None:
    """Add ``patch * patch_weight`` into ``accumulator``, in place.

    ``weight`` grows by ``patch_weight`` as well, so ``patch_weight`` behaves
    like a per-patch confidence: the final :func:`weighted_average` is the
    confidence-weighted mean rather than the plain mean.
    """
    patch = np.asarray(patch)
    if patch.ndim != 2:
        raise ValueError(f"patch must be 2-D, got shape {patch.shape!r}")
    patch_h, patch_w = patch.shape
    if top < 0 or left < 0:
        raise ValueError(f"negative offset: top={top}, left={left}")
    if top + patch_h > accumulator.shape[0] or left + patch_w > accumulator.shape[1]:
        raise ValueError(
            f"patch {patch.shape!r} at ({top}, {left}) does not fit into "
            f"buffer {accumulator.shape!r}"
        )
    accumulator[top : top + patch_h, left : left + patch_w] += patch * patch_weight
    weight[top : top + patch_h, left : left + patch_w] += patch_weight


def weighted_average(
    accumulator: np.ndarray,
    weight: np.ndarray,
    fallback: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Divide ``accumulator`` by ``weight``, pixel by pixel.

    Pixels that were never covered by a patch keep whatever ``fallback``
    holds (the untouched input image), matching the behaviour of the
    original ``out = texture.copy(); out[mask] = acc[mask] / (w[mask] + eps)``.
    """
    accumulator = np.asarray(accumulator, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if accumulator.shape != weight.shape:
        raise ValueError(
            f"accumulator {accumulator.shape!r} and weight {weight.shape!r} "
            "must have the same shape"
        )

    covered = weight > 0
    if fallback is None:
        out = np.zeros_like(accumulator)
    else:
        out = np.array(fallback, dtype=np.float64, copy=True)
        if out.shape != accumulator.shape:
            raise ValueError(
                f"fallback {out.shape!r} must match accumulator "
                f"{accumulator.shape!r}"
            )

    out[covered] = accumulator[covered] / (weight[covered] + eps)
    return out


def reconstruct_overlapping(
    patches: List[Tuple[Position, np.ndarray]],
    shape: Tuple[int, int],
    fallback: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Accumulate ``[((top, left), patch), ...]`` and return the average."""
    accumulator = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    for (top, left), patch in patches:
        accumulate_patch(accumulator, weight, patch, top, left)
    return weighted_average(accumulator, weight, fallback=fallback, eps=eps)
