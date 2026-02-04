#!/usr/bin/env python
# coding: utf-8
"""
MATH test split ➜ 推理数据（更稳健的 boxed 抽取 + 顶层拆分 + 规范化）
输出 JSONL 键：
  - data_source: "MATH"
  - prompt: str
  - groundtruth: list[str]  （多答案已“括号感知拆分”，去重且格式化）

用法示例：
  python3 preprocess_math_infer_fixed_toplvl.py \
    --out_file /path/math_infer_test.jsonl \
    --categories algebra geometry number_theory \
    --limit 200 \
    --split_commas false \
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

from llada.plus.utils.answer_norm import (
    basic_clean,
    dedupe_preserve_order,
    extract_boxed_all,
    normalize_math_expr,
    top_level_split,
)
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_CATEGORIES = [
    "algebra",
    "precalculus",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
]

# ----------- 4) 主流程 -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/test/math_test_llada.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--categories", type=str, nargs="*", default=DEFAULT_CATEGORIES,
                    help=f"MATH 子集列表，默认: {DEFAULT_CATEGORIES}")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    ap.add_argument("--split_commas", type=str, default="true",
                    help="是否用逗号顶层拆分多答案（true/false），默认 true")
    ap.add_argument("--normalize_math", type=str, default="true",
                    help="是否做数学规范化（true/false），默认 true（建议开启）")
    args = ap.parse_args()

    split_commas   = str(args.split_commas).lower()   in {"1","true","yes","y"}
    normalize_math = str(args.normalize_math).lower() in {"1","true","yes","y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for cat in args.categories:
            ds = load_dataset("EleutherAI/hendrycks_math", cat, split="test")
            for ex in tqdm(ds, desc=f"convert test::{cat}"):
                if args.limit is not None and kept >= args.limit:
                    break

                problem  = ex.get("problem", "")
                solution = ex.get("solution", "")

                # 1) 提取所有 \boxed{...}
                raw_chunks = extract_boxed_all(solution)

                # 2) 顶层拆分（不在括号/区间内部切）
                answers = []
                for ch in raw_chunks:
                    parts = top_level_split(ch, split_commas=split_commas)
                    answers.extend([p for p in parts if p])

                # 3) 规范化（先轻量再数学）；再去重
                if normalize_math:
                    answers = [normalize_math_expr(a) for a in answers]
                else:
                    answers = [basic_clean(a) for a in answers]
                answers = dedupe_preserve_order(answers)

                item = {
                    "data_source": "MATH",
                    "prompt": problem,
                    "groundtruth": answers,   # list[str]，已顶层拆分 + 去重 + 规范化
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
            if args.limit is not None and kept >= args.limit:
                break

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
