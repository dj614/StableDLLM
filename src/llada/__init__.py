"""Deprecated compatibility package.

The LLaDA-family integration code has moved to ``modelpacks.llada``.

This shim keeps legacy imports / invocations working, e.g.::

  python -m llada.cli.main infer

New code should use::

  python -m modelpacks.llada.cli.main infer

"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.warn(
    "`llada` is deprecated; use `modelpacks.llada` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Make ``llada.*`` submodules resolve from the new location.
_NEW_ROOT = (Path(__file__).resolve().parent.parent / "modelpacks" / "llada").resolve()
__path__ = [str(_NEW_ROOT)]  # type: ignore

# Keep a minimal public surface for type checkers.
__all__ = ["cli"]
