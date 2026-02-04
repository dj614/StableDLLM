"""Smoke test: LLaDA task pack registration.

This verifies that :mod:`LLaDA.llada.register` can be used as an --auto_import
module for :mod:`mdm.eval.harness`, i.e. the task pack can register tasks
without relying on the mdm legacy bridge being imported directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from mdm.eval.harness import evaluate


def main() -> None:
    # Importing this module should register tasks.
    try:
        import LLaDA.llada.register  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Failed to import LLaDA task pack register: {e}") from e

    tmp = Path("/tmp/mdm_smoke_eval_pack")
    tmp.mkdir(parents=True, exist_ok=True)
    pred = tmp / "pred.jsonl"
    gt = tmp / "gt.jsonl"

    # Harness format: predictions + ground-truth are separate, joined by `id`.
    pred_rows = [
        {"id": 0, "prediction": "#### 3"},
        {"id": 1, "prediction": "#### 4"},
    ]
    gt_rows = [
        {"id": 0, "gold_raw": "#### 3"},
        {"id": 1, "gold_raw": "#### 5"},
    ]
    pred.write_text("\n".join(json.dumps(r) for r in pred_rows) + "\n", encoding="utf-8")
    gt.write_text("\n".join(json.dumps(r) for r in gt_rows) + "\n", encoding="utf-8")

    metrics = evaluate(task_name="llada_gsm8k", pred_path=str(pred), gt_path=str(gt), cfg={})
    assert "accuracy" in metrics
    assert abs(metrics["accuracy"] - 0.5) < 1e-6

    print("LLaDA task pack registration smoke OK")


if __name__ == "__main__":
    main()
