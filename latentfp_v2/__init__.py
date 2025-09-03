"""Second-generation single-stage baseline (plain U-Net regression).

The submodules are imported explicitly; this package deliberately does not
import ``torch`` at import time so that the numpy degradation maths in
``latentfp_v2.transforms.synthesis_v2`` stays usable on a torch-free machine.
"""

__all__ = ["transforms", "data", "models"]
