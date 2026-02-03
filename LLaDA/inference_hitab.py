#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Legacy wrapper for HiTab inference.

The old version hard-coded local paths. This wrapper delegates to the unified CLI.

Example:
  python LLaDA/inference_hitab.py \
    --checkpoint_path /path/to/ckpt \
    --data_jsonl /path/to/hitab.jsonl \
    --out_file outputs/eval/predictions_hitab.jsonl
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
    ap.add_argument("--device_ids", type=int, nargs="+", default=[0])
    ap.add_argument("--data_jsonl", type=str, required=True, help="HiTab jsonl with at least 'prompt' and gold fields")
    ap.add_argument("--out_file", type=str, default="outputs/eval/predictions_hitab.jsonl")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    cli_main([
        "infer",
        "--task", "hitab",
        "--data_jsonl", args.data_jsonl,
        "--checkpoint_path", args.checkpoint_path,
        "--device_ids", *map(str, args.device_ids),
        "--batch_size", str(args.batch_size),
        "--temperature", str(args.temperature),
        "--gen_length", str(args.gen_length),
        "--steps", str(args.steps),
        "--block_length", str(args.block_length),
        "--out_file", args.out_file,
    ])


if __name__ == "__main__":
    main()
