from __future__ import annotations

from typing import Iterator, Optional

from mdm.eval.io import iter_jsonl
from .registry import TaskExample


def iter_hitab(data_jsonl: str, max_samples: Optional[int] = None) -> Iterator[TaskExample]:
    for i, row in enumerate(iter_jsonl(data_jsonl)):
        if max_samples is not None and i >= max_samples:
            break
        prompt = row.get("prompt") or row.get("question") or row.get("input") or ""
        gold_raw = row.get("gold_raw") or row.get("answer") or row.get("output") or row.get("ground_truth") or ""
        yield TaskExample(prompt=prompt, gold_raw=gold_raw, meta={"index": i, "raw": row})
