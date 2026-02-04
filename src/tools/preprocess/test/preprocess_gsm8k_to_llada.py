#!/usr/bin/env python
# coding: utf-8
"""
GSM8K (main, test) ➜ 推理数据（与 MATH 脚本同风格：规范化、去重、稳健提取）
输出 JSONL 键：
  - data_source: "GSM8K"
  - prompt: str         (question)
  - groundtruth: list[str]  （从 answer 中抽取所有 '#### ' 后的最终答案，已规范化与去重）

用法：
  python3 preprocess_gsm8k_infer_like_math.py \
    --out_file /path/gsm8k_infer_main_test.jsonl \
    --limit 200 \
    --normalize_math true
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

from llada_plus.utils.answer_norm import extract_hash4_answers

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./LLaDA/data/test/gsm8k_test_llada.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，便于试跑")
    ap.add_argument("--normalize_math", type=str, default="true",
                    help="是否做数学规范化（true/false），默认 true")
    args = ap.parse_args()

    normalize_math = str(args.normalize_math).lower() in {"1","true","yes","y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    # GSM8K 的主要数据在 config="main", split="test"
    ds = load_dataset("gsm8k", "main", split="test")

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm(ds, desc="convert gsm8k::main/test"):
            if args.limit is not None and kept >= args.limit:
                break
            q = ex.get("question", "") or ""
            a = ex.get("answer", "") or ""

            gts = extract_hash4_answers(a, normalize_math=normalize_math)

            item = {
                "data_source": "GSM8K",
                "prompt": q,
                "groundtruth": gts,  # list[str]，按出现顺序，已规范化+去重
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
