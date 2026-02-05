"""Evaluation harness for the masked-diffusion framework.

This module is intentionally small: it delegates all task-specific logic to a
registered :class:`~mdm.tasks.spec.TaskSpec`.

Why this exists
---------------
Historically, evaluation scripts tend to embed assumptions about dataset fields
and metric logic. The harness provides a stable entry point so that:

* the *framework* (``src/mdm``) stays task-agnostic, and
* *task packs* (e.g., ``LLaDA/``) register a TaskSpec that implements metrics.

The harness can be used as a library function or as a small CLI:

  PYTHONPATH=src:. python -m mdm.eval.harness \
      --task llada_gsm8k --pred path/to/pred.jsonl \
      --auto_import LLaDA.llada.register

Or, using any other task-pack registration module:

  PYTHONPATH=src:. python -m mdm.eval.harness \
      --task llada_gsm8k --pred path/to/pred.jsonl \
      --auto_import LLaDA.llada.register

"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from ..registry import get_task


def _as_dict(cfg: Optional[Mapping[str, Any]]) -> MutableMapping[str, Any]:
    return dict(cfg) if cfg is not None else {}


def evaluate(
    task_name: str,
    pred_path: str,
    gt_path: str = "",
    cfg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, float]:
    """Evaluate predictions with a registered task.

    Args:
        task_name: Name used with :func:`mdm.registry.register_task`.
        pred_path: Path to predictions file.
        gt_path: Path to ground-truth file.
        cfg: Optional config mapping passed through to the task.

    Returns:
        A mapping of metric name -> float.
    """

    task = get_task(task_name)
    return task.metrics(pred_path=str(pred_path), gt_path=str(gt_path), cfg=_as_dict(cfg))


def save_metrics(metrics: Mapping[str, float], out_path: str) -> None:
    """Save metrics as pretty JSON."""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(metrics), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cfg_from_args(args: argparse.Namespace) -> MutableMapping[str, Any]:
    if args.cfg_json:
        p = Path(args.cfg_json)
        return json.loads(p.read_text(encoding="utf-8"))
    if args.cfg:
        return json.loads(args.cfg)
    return {}


def _auto_import(mod_names: list[str]) -> None:
    for m in mod_names:
        importlib.import_module(m)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mdm.eval")
    p.add_argument("--task", type=str, required=True, help="registered task name")
    p.add_argument("--pred", type=str, required=True, help="predictions jsonl")
    p.add_argument("--gt", type=str, default="", help="ground-truth jsonl (optional for some tasks)")
    p.add_argument("--out", type=str, default="", help="optional path to write metrics json")

    p.add_argument(
        "--auto_import",
        type=str,
        nargs="*",
        default=[],
        help=(
            "modules to import before evaluation (useful for task registration), "
            "e.g. LLaDA.llada.register"
        ),
    )

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--cfg_json", type=str, default="", help="path to a JSON config blob")
    grp.add_argument("--cfg", type=str, default="", help="JSON string config blob")

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.auto_import:
        _auto_import(list(args.auto_import))

    cfg = _load_cfg_from_args(args)
    metrics = evaluate(task_name=args.task, pred_path=args.pred, gt_path=args.gt, cfg=cfg)

    if args.out:
        save_metrics(metrics, args.out)
        print(f"Saved metrics: {args.out}")

    print(json.dumps(dict(metrics), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
