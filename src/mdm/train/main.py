"""Unified training entrypoint for the MDM framework.

Step 7 introduces a framework-level training CLI.

At this stage we keep configuration semantics minimal (YAML + deep merge +
dotted-key overrides) and dispatch to a training engine selected by config.

Currently supported engines:
- ``llada_plus``: masked-language diffusion runner under
  ``src/mdm/engines/llada_plus``.

Usage examples:

  # Dump the merged config (no training)
  PYTHONPATH=src:. python -m mdm.train --dump_config

  # Run training using the legacy llada_plus runner
  PYTHONPATH=src:. python -m mdm.train \
      --config src/configs/mdm/base/train_llada_plus.yaml \
      --config LLaDA/configs/llada_gsm8k.yaml \
      --set train.seed=1 \
      --auto_import LLaDA.llada.register
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mdm.configs import merge_config_files


def _repo_root() -> Path:
    # .../src/mdm/train/main.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _default_base_config() -> Path:
    return _repo_root() / "src" / "configs" / "mdm" / "train_base.yaml"


def _apply_hf_mirror(china: bool) -> None:
    """Optionally enable the Hugging Face mirror used in the original code."""

    from mdm.utils.hf import maybe_enable_hf_mirror

    maybe_enable_hf_mirror(bool(china))


def _engine_dispatch(engine: str):
    if engine == "llada_plus":
        from mdm.train.entrypoints.llada_plus import train_from_config

        return train_from_config
    raise ValueError(f"Unknown engine: {engine}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MDM training entrypoint")
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Path to a YAML config. Can be specified multiple times (base + overlays). "
            "If omitted, uses src/configs/mdm/train_base.yaml (single-file). For base+overlay layering, use src/configs/mdm/base/train_llada_plus.yaml + LLaDA/configs/llada_*.yaml."
        ),
    )
    ap.add_argument(
        "--set",
        action="append",
        default=None,
        help="Override config values using dotted keys, e.g. --set train.lr=1e-4",
    )
    ap.add_argument(
        "--auto_import",
        action="append",
        default=None,
        help=(
            "Import one or more modules before training (useful for task-pack registrations). "
            "Example: --auto_import LLaDA.llada.register"
        ),
    )
    ap.add_argument(
        "--dump_config",
        action="store_true",
        help="Print the merged config and exit.",
    )

    return ap.parse_args(argv)


def _maybe_auto_import(mods: Optional[Iterable[str]]) -> None:
    if not mods:
        return
    for m in mods:
        __import__(m)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # Ensure the repo root + src are importable when running as a module.
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    _maybe_auto_import(args.auto_import)

    cfg_paths: List[str] = []
    if args.config:
        cfg_paths = list(args.config)
    else:
        cfg_paths = [str(_default_base_config())]

    cfg: Dict[str, Any] = merge_config_files(cfg_paths, overrides=args.set)

    # Mirror toggle can be at top-level or under train.
    china = bool(cfg.get("china") or (isinstance(cfg.get("train"), dict) and cfg["train"].get("china")))
    _apply_hf_mirror(china)

    if args.dump_config:
        import yaml

        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    engine = str(cfg.get("engine") or "llada_plus")
    train_fn = _engine_dispatch(engine)
    train_fn(cfg)


if __name__ == "__main__":
    main()
