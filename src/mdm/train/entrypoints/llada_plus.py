"""Training engine entrypoint: LLaDA+ masked diffusion runner.

This engine uses the implementation under ``src/mdm/engines/llada_plus``.

Step 7 goal: make the *entrypoint* live under ``mdm`` so training can be
configured and dispatched in a task-agnostic way.

Config contract
--------------
The merged config should look like:

    engine: llada_plus
    train:
      seed: 42
      task: gsm8k
      batch_size_per_gpu: 1
      grad_accum: 1
      ... (fields compatible with mdm.engines.llada_plus.cli.train.parse_args)

The runner is invoked with an ``argparse.Namespace`` built from ``train``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping, MutableMapping


def _flatten_train_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    train = cfg.get("train")
    if isinstance(train, Mapping):
        return dict(train)
    # Backward compatible: allow a flat config.
    return dict(cfg)


def _fill_defaults(a: MutableMapping[str, Any]) -> None:
    # Mirror the defaults/normalization in mdm.engines.llada_plus.cli.train.parse_args.
    if a.get("model") is None:
        a["model"] = "llada"
    if a.get("train_mode") is None:
        a["train_mode"] = "Normal"
    if a.get("PPOTS") is None:
        a["PPOTS"] = False
    if a.get("p_model") is None:
        a["p_model"] = "EPR"

    if a.get("eval_strategy") is None:
        a["eval_strategy"] = a.get("save_strategy", "last")
    if a.get("eval_steps") is None:
        a["eval_steps"] = a.get("save_steps", 100)

    if a.get("train_data_path") is None:
        task = a.get("task")
        if task:
            a["train_data_path"] = f"./data/train/{task}.jsonl"

    # Used by PPOTS (IS-on-t) when estimating w(t) statistics.
    if a.get("loss_max") is None:
        a["loss_max"] = 10.0


def _require(a: Mapping[str, Any], key: str) -> None:
    if key not in a or a[key] is None:
        raise ValueError(f"Missing required train config field: {key}")


def train_from_config(cfg: Mapping[str, Any]) -> Any:
    """Dispatch training using the llada_plus engine."""

    # Import lazily so `--dump_config` remains lightweight.
    from mdm.engines.llada_plus.runner import train as _train

    train_cfg = _flatten_train_cfg(cfg)

    # Required fields for the current runner.
    for k in ("seed", "task", "batch_size_per_gpu", "grad_accum"):
        _require(train_cfg, k)

    _fill_defaults(train_cfg)

    # argparse.Namespace is what the runner expects.
    args = argparse.Namespace(**train_cfg)
    return _train(args)
