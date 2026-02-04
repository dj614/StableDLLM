"""Smoke test for mdm.eval.harness.

This is intentionally lightweight and does not touch model code.
"""

from __future__ import annotations

import json
from pathlib import Path

from mdm.eval.harness import evaluate


def main() -> None:
    # Register legacy LLaDA tasks (if present) and score a tiny dummy file.
    # This exercises the "bridge" mechanism introduced in Step 4.
    try:
        import mdm.bridges.llada_legacy  # noqa: F401
    except Exception as e:
        raise SystemExit(f"Failed to import legacy bridge: {e}") from e

    tmp = Path("/tmp/mdm_smoke_eval")
    tmp.mkdir(parents=True, exist_ok=True)
    pred = tmp / "pred.jsonl"

    rows = [
        {"prediction": "#### 3", "gold_raw": "#### 3"},
        {"prediction": "#### 4", "gold_raw": "#### 5"},
    ]
    pred.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    metrics = evaluate(task_name="llada_gsm8k", pred_path=str(pred), gt_path="", cfg={})
    assert "accuracy" in metrics
    # 1/2 correct
    assert abs(metrics["accuracy"] - 0.5) < 1e-6

    print("mdm eval harness smoke OK")


if __name__ == "__main__":
    main()
