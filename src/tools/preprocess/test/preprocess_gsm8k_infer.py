#!/usr/bin/env python
# coding: utf-8
"""GSM8K (main, test) ➜ inference / evaluation JSONL (model-agnostic).

Outputs JSONL with keys:
  - data_source: "GSM8K"
  - prompt: str
  - groundtruth: list[str]  (answers extracted from the official '#### ' marker,
                            normalized and de-duplicated)

This format is used by evaluation scripts that expect prompts + groundtruths
instead of pre-tokenized input_ids.

Example:
  PYTHONPATH=src:. python src/tools/preprocess/test/preprocess_gsm8k_infer.py \
    --out_file ./data/test/gsm8k_main_test.jsonl \
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

from mdm.utils.answer_norm import extract_hash4_answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./data/test/gsm8k_main_test.jsonl", help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N items (for quick runs)")
    ap.add_argument(
        "--normalize_math",
        type=str,
        default="true",
        help="Whether to normalize math answers (true/false). Default: true",
    )
    args = ap.parse_args()

    normalize_math = str(args.normalize_math).lower() in {"1", "true", "yes", "y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

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
                "groundtruth": gts,
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ Done: wrote {kept} records -> {args.out_file}")


if __name__ == "__main__":
    main()
