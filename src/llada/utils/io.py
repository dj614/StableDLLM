"""Deprecated shim for JSONL helpers.

Prefer importing from :mod:`mdm.utils.io` (or :mod:`mdm.eval.io`).
This module remains for backwards compatibility.
"""

from __future__ import annotations

import warnings

from mdm.utils.io import iter_jsonl, read_jsonl, write_jsonl

warnings.warn(
    "llada.utils.io is deprecated; use mdm.utils.io (or mdm.eval.io) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["iter_jsonl", "read_jsonl", "write_jsonl"]
