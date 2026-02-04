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

maybe_enable_hf_mirror_china(sys.argv)

import os
import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from llada.plus.utils.sft_format import encode_sft_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./data/train/openscience.jsonl")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max_num", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--china", action="store_true")
    args = ap.parse_args()

    # 确保输出路径存在
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    print("✓ 流式加载 nvidia/OpenScienceReasoning-2（不下载整个数据集）...")
    stream_ds = load_dataset(
        "nvidia/OpenScienceReasoning-2",
        split=f"train[:{args.max_num + 2000}]",
    )

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm(stream_ds, desc="streaming-filter"):

            item = encode_sft_pair(ex["input"], ex["output"], tok)
            if len(item["input_ids"]) >= args.max_len:
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            kept += 1
            if kept >= args.max_num:
                break

    print(f"✓ 完成！共保存 {kept} 条满足条件的样本到 {args.out_file}")


if __name__ == "__main__":
    main()
