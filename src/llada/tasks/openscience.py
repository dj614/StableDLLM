from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from .registry import TaskExample


def iter_openscience(
    split: str = "train",
    start_index: int = 0,
    end_index: int = 0,
    max_samples: Optional[int] = None,
) -> Iterator[TaskExample]:
    ds = load_dataset("nvidia/OpenScienceReasoning-2", split=split)
    if end_index and end_index > start_index:
        ds = ds.select(range(start_index, min(end_index, len(ds))))
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in range(n):
        inp = ds[i]["input"]
        out = ds[i]["output"]
        yield TaskExample(prompt=inp, gold_raw=out, meta={"index": i + start_index})
