"""Deprecated shim for answer-extraction helpers.

Canonical implementations live in the task pack under :mod:`LLaDA.llada.eval`.
This module remains for backwards compatibility.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.eval.extract import (  # type: ignore
        extract_boxed_answer,
        extract_gsm8k_hash_answer,
        extract_last_number,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.eval.extract. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.eval.extract is deprecated; use LLaDA.llada.eval.extract instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["extract_gsm8k_hash_answer", "extract_boxed_answer", "extract_last_number"]
