"""Deprecated shim for OpenScience dataset adapter.

Canonical implementation lives in ``LLaDA/llada/tasks/openscience.py``.
"""

from __future__ import annotations

import warnings

try:
    from LLaDA.llada.tasks.openscience import iter_openscience  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import LLaDA.llada.tasks.openscience. "
        "Make sure you run from repo root with PYTHONPATH including both 'src/' and the repo root ('.')."
    ) from e

warnings.warn(
    "llada.tasks.openscience is deprecated; use LLaDA.llada.tasks.openscience instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["iter_openscience"]
