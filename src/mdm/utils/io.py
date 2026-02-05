"""Lightweight JSONL I/O helpers.

Several modules in this repo import :mod:`mdm.utils.io` (preprocess tools and
evaluation helpers). The refactor
introduced the namespace but the concrete file was missing in this zip.

These helpers intentionally stay dependency-light (stdlib only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union


PathLike = Union[str, Path]


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file (skips empty lines)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Read an entire JSONL file into memory."""
    return list(iter_jsonl(path))


def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of JSON objects to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
