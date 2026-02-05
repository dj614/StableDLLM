#!/usr/bin/env python
# coding: utf-8
"""HiTab JSONL ➜ LLaDA SFT 所需格式。

输入:  每行 {"prompt": "...", "response": "..."}（CoT 数据需要自行构造；官方 HiTab 数据仅有 prompt 与 ground-truths，无 CoT responses）
输出:  JSONL （键: input_ids, prompt_length）

用法:
    python3 preprocess_hitab_to_llada.py --in_file ... --out_file ...
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
    ap.add_argument("--in_file",  type=str, default="./LLaDA/hitab_reasoning_sft_str.jsonl", help="包含 CoT responses 的数据集")
    ap.add_argument("--out_file", type=str, default="./LLaDA/data/train/hitab_reasoning_sft_str_processed.jsonl")
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--china", action="store_true", help="是否使用国内镜像 hf-mirror.com")
    args = ap.parse_args()

    maybe_enable_hf_mirror_china(args.china)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    kept = 0
    with open(args.in_file, "r", encoding="utf8") as f, open(
        args.out_file, "w", encoding="utf8"
    ) as out_f:
        for line in tqdm(f, desc="convert"):
            ex = json.loads(line)
            item = encode_sft_pair(ex["prompt"], ex["response"], tok, strip=False)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
