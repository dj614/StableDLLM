"""Deprecated shim for GSM8K dataset adapter.

Canonical implementation lives in ``LLaDA/llada/tasks/gsm8k.py``.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.tasks.gsm8k import iter_gsm8k  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.tasks.gsm8k. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.tasks.gsm8k is deprecated; use LLaDA.llada.tasks.gsm8k instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["iter_gsm8k"]
