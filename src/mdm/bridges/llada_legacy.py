"""Bridge: register *legacy* LLaDA tasks into :mod:`mdm.registry`.

This repo currently ships a LLaDA CLI that writes predictions JSONL where each
record already includes the gold answer ("gold_raw"). The legacy scoring
functions (``llada.eval.metrics``) expect that format.

To keep Step0-4 incremental, this bridge registers simple TaskSpec objects whose
``metrics`` method calls the existing scoring functions.

As of Step 6, the preferred path is to import ``LLaDA.llada.register`` which
registers non-legacy TaskSpecs from the task pack.

Usage
-----

  PYTHONPATH=src python -m mdm.eval.harness \
      --task llada_gsm8k --pred out/pred.jsonl \
      --auto_import mdm.bridges.llada_legacy

"""

from __future__ import annotations

import warnings

warnings.warn("mdm.bridges.llada_legacy is deprecated; prefer LLaDA.llada.register (task pack).", DeprecationWarning, stacklevel=2)


from dataclasses import dataclass
from typing import Any, Mapping

from ..registry import register_task
from ..tasks.spec import BaseTaskSpec
from ..eval.io import iter_jsonl


@dataclass
class _LegacyLLaDATask(BaseTaskSpec):
    name: str
    _scorer: Any

    def build_dataset(self, split: str, cfg: Mapping[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("Legacy bridge task does not implement datasets")

    def collate_fn(self, batch: list[Any]) -> Any:  # pragma: no cover
        raise NotImplementedError("Legacy bridge task does not implement collation")

    def postprocess(self, pred: Any, cfg: Mapping[str, Any]) -> Any:  # pragma: no cover
        return pred

    def metrics(self, pred_path: str, gt_path: str, cfg: Mapping[str, Any]) -> Mapping[str, float]:
        # gt_path intentionally ignored for this legacy format.
        rows = iter_jsonl(pred_path)
        res = self._scorer(rows)
        # score_* returns ScoreResult with to_dict().
        return res.to_dict()  # type: ignore[no-any-return]


def register_llada_legacy_tasks(prefix: str = "llada_", *, overwrite: bool = False) -> None:
    """Register legacy LLaDA tasks.

    This is safe to call multiple times.
    """

    try:
        from llada.eval.metrics import score_gsm8k, score_hitab, score_openscience
    except Exception:  # pragma: no cover
        # LLaDA isn't available in this environment.
        return

    tasks = {
        f"{prefix}gsm8k": score_gsm8k,
        f"{prefix}openscience": score_openscience,
        f"{prefix}hitab": score_hitab,
    }

    for name, scorer in tasks.items():
        try:
            register_task(
                name,
                _LegacyLLaDATask(name=name, _scorer=scorer),
                overwrite=overwrite,
            )
        except ValueError:
            # Task already registered. Keep this bridge idempotent.
            continue


# Convenience: importing this module registers the tasks.
register_llada_legacy_tasks()
