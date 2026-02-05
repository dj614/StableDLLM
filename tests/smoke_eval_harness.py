"""Smoke test for mdm.eval.harness.

This is intentionally lightweight and does not touch model code.
"""

from __future__ import annotations

import json
from pathlib import Path

from mdm.eval.harness import evaluate


def main() -> None:
    # Register the LLaDA task pack and score a tiny dummy file.
    try:
        import LLaDA.llada.register  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Failed to import LLaDA task pack: {e}") from e

    tmp = Path("/tmp/mdm_smoke_eval")
    tmp.mkdir(parents=True, exist_ok=True)
    pred = tmp / "pred.jsonl"
    gt = tmp / "gt.jsonl"

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
    # 1/2 correct
    assert abs(metrics["accuracy"] - 0.5) < 1e-6

    print("mdm eval harness smoke OK")


if __name__ == "__main__":
    main()
