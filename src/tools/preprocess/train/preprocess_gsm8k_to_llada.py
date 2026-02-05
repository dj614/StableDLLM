#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mdm.utils.hf import maybe_enable_hf_mirror_china

import os
import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from mdm.utils.sft_format import encode_sft_pair

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./data/train/gsm8k.jsonl")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--china", action="store_true", help="是否使用国内镜像 hf-mirror.com")
    args = ap.parse_args()

    maybe_enable_hf_mirror_china(args.china)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True, 
        trust_remote_code=True
    )

    print(f"✓ 加载 GSM8K 数据集 split={args.split} ...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    with open(args.out_file, "w", encoding="utf8") as out_f:
        kept = 0
        for ex in tqdm(dataset, desc="Converting"):
            item = encode_sft_pair(
                ex["question"],
                ex["answer"],
                tok,
                user_suffix="\n",
                assistant_suffix="\n",
                strip=True,
            )
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条到 {args.out_file}")

if __name__ == "__main__":
    main()
