"""Deprecated shim for HiTab dataset adapter.

Canonical implementation lives in ``LLaDA/llada/tasks/hitab.py``.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.tasks.hitab import iter_hitab  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.tasks.hitab. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.tasks.hitab is deprecated; use LLaDA.llada.tasks.hitab instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["iter_hitab"]
