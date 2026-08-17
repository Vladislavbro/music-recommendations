"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Fix RNG state for python, numpy and (if available) torch.

    Args:
        seed: integer seed.
        deterministic_torch: if True, set torch.backends.cudnn flags for
            deterministic kernels. Has small perf cost; OK for our scale.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
