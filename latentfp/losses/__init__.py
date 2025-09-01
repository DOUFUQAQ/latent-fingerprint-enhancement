"""Training objectives.

``metrics`` is numpy-only; ``perceptual_loss`` needs torch + torchvision, so it
is imported lazily/defensively and imported errors are re-raised on use.
"""

from . import metrics

try:  # torch + torchvision are heavy, optional dependencies
    from .perceptual_loss import (
        OrientationFieldLoss,
        TotalVariationLoss,
        VGGPerceptualLoss,
    )
except ImportError:  # pragma: no cover - depends on the environment
    _TORCH_ONLY = ("VGGPerceptualLoss", "OrientationFieldLoss", "TotalVariationLoss")

    def __getattr__(name):  # PEP 562: fail only when actually used
        if name in _TORCH_ONLY:
            raise ImportError(
                f"latentfp.losses.{name} requires torch and torchvision"
            )
        raise AttributeError(name)

    __all__ = ["metrics"]
else:
    __all__ = [
        "metrics",
        "VGGPerceptualLoss",
        "OrientationFieldLoss",
        "TotalVariationLoss",
    ]
