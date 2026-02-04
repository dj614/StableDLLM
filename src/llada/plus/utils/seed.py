"""Deprecated shim for reproducibility helpers.

Prefer importing from :mod:`mdm.utils.seed`.
"""

from __future__ import annotations

import warnings

from mdm.utils.seed import set_random_seed

warnings.warn(
    "llada.plus.utils.seed is deprecated; use mdm.utils.seed instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["set_random_seed"]
