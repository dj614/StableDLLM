"""Shared prediction / ground-truth I/O utilities.

This module defines a minimal JSONL-based interchange format to make evaluation
pluggable across tasks.

Recommended conventions:
  * predictions.jsonl: each line is a dict with at least ``id`` and ``pred``
  * ground_truth.jsonl: each line is a dict with at least ``id`` and ``answer``

Tasks are free to include additional metadata fields.

Implementation lives in :mod:`mdm.utils.io`; this module re-exports it to keep
imports stable.
"""

from __future__ import annotations

from mdm.utils.io import iter_jsonl, read_jsonl, write_jsonl

__all__ = ["iter_jsonl", "read_jsonl", "write_jsonl"]
