from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from .registry import TaskExample


def iter_gsm8k(split: str = "test", max_samples: Optional[int] = None) -> Iterator[TaskExample]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in range(n):
        q = ds[i]["question"]
        a = ds[i]["answer"]
        yield TaskExample(prompt=q, gold_raw=a, meta={"index": i})
