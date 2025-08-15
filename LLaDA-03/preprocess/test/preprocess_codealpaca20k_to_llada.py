#!/usr/bin/env python
# coding: utf-8
"""
CodeAlpaca_20K ➜ 推理数据（不提取答案，直接映射）
输出 JSONL 每行：
  - data_source: "codealpaca20k"
  - prompt: str         (原始 prompt)
  - groundtruth: str    (原始 completion)

用法示例：
  python3 preprocess_codealpaca20k_infer.py \
    --out_file codealpaca20k_test_llada.jsonl \
    --split train \
    --limit 200
"""

import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="/storage/v-mengnijia/LLaDA/data/test/codealpaca20k_test_llada.jsonl",
                    help="输出 JSONL 文件路径")
    ap.add_argument("--split", type=str, default="train",
                    help="数据划分（CodeAlpaca_20K 通常只有 train）")
    ap.add_argument("--limit", type=int, default=None,
                    help="仅处理前 N 条，便于试跑")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print("• 加载数据集 HuggingFaceH4/CodeAlpaca_20K ...")
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=args.split)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as fout:
        for ex in tqdm(ds, desc=f"write infer::{args.split}"):
            if args.limit is not None and kept >= args.limit:
                break

            prompt = (ex.get("prompt") or "").strip()
            comp   = (ex.get("completion") or "").strip()

            item = {
                "data_source": "codealpaca20k",
                "prompt": prompt,
                "groundtruth": comp,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
