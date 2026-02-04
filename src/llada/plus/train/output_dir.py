"""Deprecated shim for output directory helpers.

Prefer importing from :mod:`mdm.utils.output_dir`.
"""

from __future__ import annotations

import warnings

from mdm.utils.output_dir import make_output_dir_and_broadcast

warnings.warn(
    "llada.plus.train.output_dir is deprecated; use mdm.utils.output_dir instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["make_output_dir_and_broadcast"]
