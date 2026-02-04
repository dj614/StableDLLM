"""Evaluation harness and I/O helpers.

The framework-level evaluation entrypoint lives in :mod:`mdm.eval.harness`.
Task packs should register a :class:`~mdm.tasks.spec.TaskSpec` and implement
``metrics``.
"""

from __future__ import annotations

from mdm.eval.io import iter_jsonl, read_jsonl, write_jsonl
from mdm.eval.harness import evaluate, save_metrics

__all__ = [
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
    "evaluate",
    "save_metrics",
]
