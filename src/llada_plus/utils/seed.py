"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int, rank: int = 0) -> None:
    """Set Python/NumPy/PyTorch RNG seeds.

    We follow the exact behavior from the original training script:
    - seed is offset by distributed rank
    - enable deterministic cuDNN behavior
    """
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
