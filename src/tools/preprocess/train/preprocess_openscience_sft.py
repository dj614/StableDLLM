#!/usr/bin/env python
# coding: utf-8
"""nvidia/OpenScienceReasoning-2 ➜ processed SFT JSONL (model-agnostic).

Streams a subset of the dataset and converts it to the processed JSONL format:
  {"input_ids": [...], "prompt_length": int}

Filtering:
- Keep up to --max_num items
- Drop items whose total length (in tokens) is >= --max_len

The output is tokenizer-dependent. Use the same tokenizer you will train with.

Example:
  PYTHONPATH=src:. python src/tools/preprocess/train/preprocess_openscience_sft.py \
    --out_file ./data/train/openscience.jsonl \
    --tokenizer_path /path/to/your-tokenizer
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mdm.utils.hf import maybe_enable_hf_mirror_china

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from mdm.utils.sft_format import encode_sft_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./data/train/openscience.jsonl")
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer name/path (HF repo id or local path). Must match the tokenizer used in training.",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[DEPRECATED] Alias for --tokenizer_path (kept for backward compatibility).",
    )
    ap.add_argument("--max_num", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--china", action="store_true")
    args = ap.parse_args()

    # Backward-compat: old scripts used --model_path
    if args.model_path:
        args.tokenizer_path = args.model_path

    maybe_enable_hf_mirror_china(args.china)

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print(f"✓ Loading tokenizer: {args.tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)

    print("✓ Loading nvidia/OpenScienceReasoning-2 ...")
    ds = load_dataset(
        "nvidia/OpenScienceReasoning-2",
        split=f"train[:{args.max_num + 2000}]",
    )

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm(ds, desc="streaming-filter"):
            item = encode_sft_pair(ex.get("input", "") or "", ex.get("output", "") or "", tok)
            if len(item["input_ids"]) >= args.max_len:
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.max_num:
                break

    print(f"✓ Done: wrote {kept} records to {args.out_file}")


if __name__ == "__main__":
    main()
