#!/usr/bin/env python
# coding: utf-8
"""
把 MATH (EleutherAI/hendrycks_math) ➜ LLaDA SFT 所需格式
---------------------------------------------------------
输入:  HuggingFace datasets 中的 MATH 子集（problem, solution）
输出:  JSONL (键: input_ids, prompt_length)
用法示例:
    python3 preprocess_math_to_llada.py \
        --out_file /path/math_sft_processed.jsonl \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --categories algebra geometry number_theory \
        --split train \
        --max_len 4096
"""

import argparse, json, os, sys
from typing import List
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

DEFAULT_CATEGORIES = [
    "algebra",
    "precalculus",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
]

def encode_example(problem: str, solution: str, tok) -> dict:
    """
    把一条 MATH 样本编码成 LLaDA 需要的
    input_ids 与 prompt_length。
    """
    user_txt = problem if isinstance(problem, str) else str(problem)
    asst_txt = solution if isinstance(solution, str) else str(solution)

    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + user_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + asst_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    prompt_len = len(user_ids)
    return dict(input_ids=ids, prompt_length=prompt_len)


def load_math_splits(categories: List[str], split: str) -> Dataset:
    """
    逐个子集加载后 concat 成一个 Dataset。
    EleutherAI/hendrycks_math 每个子集独立配置名。
    """
    all_parts = []
    for cat in categories:
        ds = load_dataset("EleutherAI/hendrycks_math", cat, split=split)
        all_parts.append(ds)
    if len(all_parts) == 1:
        return all_parts[0]
    # 延迟导入以避免 datasets 旧版本无 concat_datasets 时报错
    from datasets import concatenate_datasets
    return concatenate_datasets(all_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="/storage/v-mengnijia/LLaDA/math_reasoning_sft_str.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="用于分词的 LLaDA 模型路径")
    ap.add_argument("--categories", type=str, nargs="*", default=DEFAULT_CATEGORIES,
                    help=f"MATH 子集列表，默认: {DEFAULT_CATEGORIES}")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"],
                    help="数据划分（train/test）")
    ap.add_argument("--max_len", type=int, default=None,
                    help="可选：仅保留 prompt tokenized 长度 ≤ max_len 的样本")
    ap.add_argument("--limit", type=int, default=None,
                    help="可选：仅处理前 N 条，方便试跑")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    # tokenizer（与 HiTab 脚本一致）
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )

    # 加载并合并所需子集
    ds = load_math_splits(args.categories, args.split)

    kept, skipped = 0, 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        pbar = tqdm(ds, desc=f"convert ({args.split}, {','.join(args.categories)})")
        for i, ex in enumerate(pbar):
            if args.limit is not None and kept >= args.limit:
                break

            # MATH 字段名：problem, solution
            problem = ex.get("problem", "")
            solution = ex.get("solution", "")

            # 过滤空样本
            if not problem or not solution:
                skipped += 1
                continue

            item = encode_example(problem, solution, tok)

            # 按需基于 prompt 长度做过滤（只看 user_ids 长度）
            if args.max_len is not None and item["prompt_length"] > args.max_len:
                skipped += 1
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 1000 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)

    print(f"✓ 完成: 写入 {kept} 条；跳过 {skipped} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
