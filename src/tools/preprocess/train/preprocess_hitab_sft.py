#!/usr/bin/env python
# coding: utf-8
"""HiTab JSONL ➜ processed SFT JSONL (model-agnostic).

Input:  JSONL, each line like {"prompt": "...", "response": "..."}
Output: JSONL, each line like {"input_ids": [...], "prompt_length": int}

Notes:
- The output is tokenizer-dependent. Use the same tokenizer you will train with.
- Official HiTab may not include chain-of-thought; prepare your own prompt/response pairs.

Example:
  PYTHONPATH=src:. python src/tools/preprocess/train/preprocess_hitab_sft.py \
    --in_file ./data/raw/hitab_reasoning_sft.jsonl \
    --out_file ./data/train/hitab_reasoning_sft.jsonl \
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

from tqdm import tqdm
from transformers import AutoTokenizer

from mdm.utils.sft_format import encode_sft_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_file",
        type=str,
        default="./data/raw/hitab_reasoning_sft.jsonl",
        help="JSONL that already contains prompt/response pairs",
    )
    ap.add_argument(
        "--out_file",
        type=str,
        default="./data/train/hitab_reasoning_sft.jsonl",
        help="Output processed JSONL (input_ids, prompt_length)",
    )
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
    ap.add_argument("--china", action="store_true", help="Use hf-mirror.com (mainland China mirror).")
    args = ap.parse_args()

    # Backward-compat: old scripts used --model_path
    if args.model_path:
        args.tokenizer_path = args.model_path

    maybe_enable_hf_mirror_china(args.china)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    kept = 0
    with open(args.in_file, "r", encoding="utf8") as f, open(args.out_file, "w", encoding="utf8") as out_f:
        for line in tqdm(f, desc="convert"):
            ex = json.loads(line)
            item = encode_sft_pair(ex["prompt"], ex["response"], tok, strip=False)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ Done: wrote {kept} records -> {args.out_file}")


if __name__ == "__main__":
    main()
