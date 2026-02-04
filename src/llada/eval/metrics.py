"""Deprecated shim for task scoring.

Canonical implementations live in the task pack under :mod:`LLaDA.llada.eval`.
This module remains for backwards compatibility.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.eval.metrics import (  # type: ignore
        ScoreResult,
        score_gsm8k,
        score_hitab,
        score_openscience,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.eval.metrics. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.eval.metrics is deprecated; use LLaDA.llada.eval.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ScoreResult", "score_gsm8k", "score_openscience", "score_hitab"]
