"""Deprecated shim for HuggingFace mirror helpers.

Prefer importing from :mod:`mdm.utils.hf`.
"""

from __future__ import annotations

import warnings

from mdm.utils.hf import enable_hf_mirror_china, maybe_enable_hf_mirror_china

warnings.warn(
    "llada.utils.hf is deprecated; use mdm.utils.hf instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["enable_hf_mirror_china", "maybe_enable_hf_mirror_china"]
