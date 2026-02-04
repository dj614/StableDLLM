from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .gsm8k import iter_gsm8k
from .openscience import iter_openscience
from .hitab import iter_hitab


@dataclass
class TaskExample:
    prompt: str
    gold_raw: str
    meta: Dict[str, Any]


def iter_task_examples(
    task: str,
    split: str = "test",
    data_jsonl: str = "",
    start_index: int = 0,
    end_index: int = 0,
    max_samples: Optional[int] = None,
) -> Iterator[TaskExample]:
    task = task.lower()
    if task == "gsm8k":
        yield from iter_gsm8k(split=split, max_samples=max_samples)
        return
    if task == "openscience":
        yield from iter_openscience(split=split, start_index=start_index, end_index=end_index, max_samples=max_samples)
        return
    if task == "hitab":
        if not data_jsonl:
            raise ValueError("--data_jsonl is required for task=hitab")
        yield from iter_hitab(data_jsonl=data_jsonl, max_samples=max_samples)
        return
    raise ValueError(f"Unknown task: {task}")
