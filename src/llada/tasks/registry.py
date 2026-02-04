"""Deprecated shim for task registry helpers.

Canonical implementation lives in ``LLaDA/llada/tasks/registry.py``.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.tasks.registry import TaskExample, iter_task_examples  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.tasks.registry. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.tasks.registry is deprecated; use LLaDA.llada.tasks.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TaskExample", "iter_task_examples"]
