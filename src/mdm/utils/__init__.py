"""Small, reusable utilities shared across the MDM framework.

This package should stay task-agnostic.
"""

from .io import iter_jsonl, read_jsonl, write_jsonl

__all__ = ["iter_jsonl", "read_jsonl", "write_jsonl"]
