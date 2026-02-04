"""Step0 smoke: imports + minimal jsonl round-trip.

This smoke test is intentionally lightweight and does not require GPUs or large
datasets. It only asserts that the newly introduced framework scaffolding is
importable and functional.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from mdm.eval.io import read_jsonl, write_jsonl
from mdm.registry import list_tasks, register_task
from mdm.tasks.spec import BaseTaskSpec


class _DummyTask(BaseTaskSpec):
    """A tiny TaskSpec implementation for smoke testing the registry."""

    name = "_dummy"

    def build_dataset(self, split: str, cfg: Mapping[str, Any]) -> Any:  # noqa: ARG002
        return []

    def collate_fn(self, batch: list[Any]) -> Any:  # noqa: ARG002
        return {}

    def postprocess(self, pred: Any, cfg: Mapping[str, Any]) -> Any:  # noqa: ARG002
        return pred

    def metrics(self, pred_path: str, gt_path: str, cfg: Mapping[str, Any]) -> Mapping[str, float]:  # noqa: ARG002
        # This is a smoke test, not a real metric.
        return {"ok": 1.0}


def main() -> None:
    # Basic jsonl round-trip.
    tmpdir = Path(tempfile.mkdtemp(prefix="mdm_smoke_"))
    p = tmpdir / "predictions.jsonl"
    write_jsonl(p, [{"id": "1", "pred": "hello"}])
    recs = read_jsonl(p)
    assert recs and recs[0]["pred"] == "hello"

    # Registry registration.
    register_task(_DummyTask.name, _DummyTask(), overwrite=True)
    assert _DummyTask.name in list_tasks()

    print("mdm smokeOK: smoke_imports")


if __name__ == "__main__":
    main()
