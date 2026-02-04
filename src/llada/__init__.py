"""llada: legacy LLaDA convenience package.

This repo historically used multiple import layouts (e.g. ``LLaDA.llada``).
The canonical LLaDA *task pack* now lives under ``LLaDA/llada``.

The ``src/llada`` package is kept as a thin compatibility layer so existing
scripts (and older notebooks) keep working.

If you run scripts from repo root, make sure both the repo root and ``src/`` are
on ``PYTHONPATH``, e.g.:

  PYTHONPATH=src:. python -m llada.cli.main --help
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable so `LLaDA.llada` works even when users only
# add `src/` to PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

__all__ = ["cli", "eval", "model", "tasks", "utils", "plus"]
