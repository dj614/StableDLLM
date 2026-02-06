#!/usr/bin/env python
# coding: utf-8
"""GSM8K ➜ processed SFT JSONL (model-agnostic).

Converts GSM8K into a JSONL where each line is:
  {"input_ids": [...], "prompt_length": int}

The output is tokenizer-dependent. Use the same tokenizer you will train with.

Example:
  PYTHONPATH=src:. python src/tools/preprocess/train/preprocess_gsm8k_sft.py \
    --out_file ./data/train/gsm8k.jsonl \
    --split train \
    --tokenizer_path GSAI-ML/LLaDA-8B-Instruct
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
    ap.add_argument("--out_file", type=str, default="./data/train/gsm8k.jsonl")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
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

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print(f"✓ Loading tokenizer: {args.tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"✓ Loading GSM8K dataset split={args.split} ...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm(dataset, desc="Converting"):
            item = encode_sft_pair(
                ex.get("question", "") or "",
                ex.get("answer", "") or "",
                tok,
                user_suffix="\n",
                assistant_suffix="\n",
                strip=True,
            )
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ Done: wrote {kept} records to {args.out_file}")


if __name__ == "__main__":
    main()
