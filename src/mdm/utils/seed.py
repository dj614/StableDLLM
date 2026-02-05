"""Random seeding helpers.

The training runner under ``llada.plus`` expects :func:`set_random_seed` to live
in :mod:`mdm.utils.seed`.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int, rank: int = 0, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: Global seed.
        rank: Process rank (added to seed so each rank gets a different stream).
        deterministic: If True, forces deterministic CuDNN behavior.
    """
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
