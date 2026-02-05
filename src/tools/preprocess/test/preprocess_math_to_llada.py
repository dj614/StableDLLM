#!/usr/bin/env python
# coding: utf-8
"""
MATH test split -> inference JSONL (boxed extraction + top-level split + normalization)

Output JSONL keys:
  - data_source: "MATH"
  - prompt: str
  - groundtruth: list[str]
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

from mdm.utils.answer_norm import (
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/test/math_test_llada.jsonl", help="Output JSONL path")
    ap.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help=f"MATH categories (default: {DEFAULT_CATEGORIES})",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    ap.add_argument(
        "--split_commas",
        type=str,
        default="true",
        help="Split top-level answers by comma (true/false).",
    )
    ap.add_argument(
        "--normalize_math",
        type=str,
        default="true",
        help="Normalize math expressions (true/false).",
    )
    args = ap.parse_args()

    split_commas = str(args.split_commas).lower() in {"1", "true", "yes", "y"}
    normalize_math = str(args.normalize_math).lower() in {"1", "true", "yes", "y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for cat in args.categories:
            ds = load_dataset("EleutherAI/hendrycks_math", cat, split="test")
            for ex in tqdm(ds, desc=f"convert test::{cat}"):
                if args.limit is not None and kept >= args.limit:
                    break

                problem = ex.get("problem", "")
                solution = ex.get("solution", "")

                # 1) Extract boxed answers
                raw_chunks = extract_boxed_all(solution)

                # 2) Top-level split (not inside braces)
                answers = []
                for ch in raw_chunks:
                    parts = top_level_split(ch, split_commas=split_commas)
                    answers.extend([p for p in parts if p])

                # 3) Normalize + dedupe
                if normalize_math:
                    answers = [normalize_math_expr(a) for a in answers]
                else:
                    answers = [basic_clean(a) for a in answers]
                answers = dedupe_preserve_order(answers)

                item = {
                    "data_source": "MATH",
                    "prompt": problem,
                    "groundtruth": answers,
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
            if args.limit is not None and kept >= args.limit:
                break

    print(f"Done: wrote {kept} rows -> {args.out_file}")


if __name__ == "__main__":
    main()
