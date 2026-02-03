#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Legacy wrapper for OpenScienceReasoning-2 evaluation.

Kept for backward compatibility. Uses the unified CLI:
  python -m LLaDA.llada.cli.main infer/score

Example:
  python LLaDA/evaluate_openscience.py --checkpoint_path /path/to/ckpt --device_ids 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LLaDA.llada.cli.main import main as cli_main  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", type=str, default="", help="finetuned checkpoint path (optional)")
    ap.add_argument("--device_ids", type=int, nargs="+", default=[0], help="gpu ids")
    ap.add_argument("--start_index", type=int, default=5000)
    ap.add_argument("--end_index", type=int, default=6000)
    ap.add_argument("--out_dir", type=str, default="outputs/eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "base" if not args.checkpoint_path else Path(args.checkpoint_path).name
    pred_jsonl = out_dir / f"predictions_openscience_{suffix}.jsonl"
    metrics_json = out_dir / f"predictions_openscience_{suffix}.metrics.json"

    cli_main([
        "infer",
        "--task", "openscience",
        "--split", "train",
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--checkpoint_path", args.checkpoint_path,
        "--device_ids", *map(str, args.device_ids),
        "--batch_size", "8",
        "--temperature", "0",
        "--gen_length", "256",
        "--steps", "256",
        "--block_length", "256",
        "--out_file", str(pred_jsonl),
    ])

    cli_main([
        "score",
        "--task", "openscience",
        "--pred_jsonl", str(pred_jsonl),
        "--out_metrics", str(metrics_json),
    ])


if __name__ == "__main__":
    main()
