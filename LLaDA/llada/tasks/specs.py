"""TaskSpec implementations for the LLaDA task pack.

These specs focus on *evaluation*.

They support two common prediction formats:

1) **Legacy**: each prediction record already contains a gold field
   (e.g. ``gold_raw``). This was historically produced by the legacy LLaDA CLI.

2) **Harness**: predictions and ground truth are separate JSONL files. Records
   are joined by ``id`` (preferred) or ``index`` as a fallback.

Step 6 goal: migrate LLaDA-specific dataset/eval adapters out of ``src/llada`` and
into this task pack, so the ``mdm`` framework stays task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from mdm.eval.io import iter_jsonl
from mdm.registry import register_task
from mdm.tasks.spec import BaseTaskSpec

from ..eval.metrics import score_gsm8k, score_hitab, score_openscience


def _row_key(row: Mapping[str, Any]) -> Optional[Any]:
    """Return a stable join key for a pred/gt row."""

    if "id" in row:
        return row.get("id")
    if "index" in row:
        return row.get("index")
    meta = row.get("meta")
    if isinstance(meta, Mapping) and "index" in meta:
        return meta.get("index")
    return None


def _prediction_text(row: Mapping[str, Any]) -> str:
    """Normalize common prediction field names to a single string."""

    for k in ("prediction", "pred_text", "pred", "output", "text"):
        v = row.get(k)
        if isinstance(v, str):
            return v
    return ""


def _gold_text(row: Mapping[str, Any]) -> str:
    """Normalize common gold field names to a single string."""

    for k in ("gold_raw", "answer", "gold", "output", "ground_truth", "label"):
        v = row.get(k)
        if isinstance(v, str):
            return v
    return ""


def _merge_pred_gt(pred_rows: Iterable[Dict[str, Any]], gt_path: str) -> Iterable[Dict[str, Any]]:
    """Merge gold answers into prediction rows when gt_path is provided."""

    if not gt_path:
        return pred_rows

    gt_map: Dict[Any, Dict[str, Any]] = {}
    for gr in iter_jsonl(gt_path):
        if not isinstance(gr, dict):
            continue
        k = _row_key(gr)
        if k is None:
            continue
        gt_map[k] = dict(gr)

    merged: list[Dict[str, Any]] = []
    for pr in pred_rows:
        if not isinstance(pr, dict):
            continue
        r = dict(pr)
        # Ensure canonical fields for scorers.
        r["prediction"] = _prediction_text(r)
        if "gold_raw" not in r or not isinstance(r.get("gold_raw"), str):
            k = _row_key(r)
            if k is not None and k in gt_map:
                r["gold_raw"] = _gold_text(gt_map[k])
        merged.append(r)
    return merged


@dataclass
class _EvalOnlyTask(BaseTaskSpec):
    """Evaluation-only TaskSpec using a legacy scoring function."""

    name: str
    scorer: Any

    def build_dataset(self, split: str, cfg: Mapping[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("Step 6 TaskSpec is eval-only")

    def collate_fn(self, batch: list[Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("Step 6 TaskSpec is eval-only")

    def postprocess(self, pred: Any, cfg: Mapping[str, Any]) -> Any:  # pragma: no cover
        return pred

    def metrics(self, pred_path: str, gt_path: str, cfg: Mapping[str, Any]) -> Mapping[str, float]:
        pred_rows = [dict(r) for r in iter_jsonl(pred_path) if isinstance(r, dict)]
        rows = _merge_pred_gt(pred_rows, gt_path=str(gt_path))
        res = self.scorer(rows)
        return res.to_dict()  # type: ignore[no-any-return]


def register_llada_tasks(prefix: str = "llada_", overwrite: bool = True) -> int:
    """Register LLaDA tasks in :mod:`mdm.registry`.

    Args:
        prefix: Name prefix for registered tasks.
        overwrite: If True, replace any existing registrations (e.g. legacy bridge).

    Returns:
        Number of tasks registered.
    """

    tasks = {
        f"{prefix}gsm8k": score_gsm8k,
        f"{prefix}openscience": score_openscience,
        f"{prefix}hitab": score_hitab,
    }

    n = 0
    for name, scorer in tasks.items():
        register_task(name, _EvalOnlyTask(name=name, scorer=scorer), overwrite=overwrite)
        n += 1
    return n
