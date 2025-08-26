#!/usr/bin/env python
# coding: utf-8
"""
把 HumanEval (openai/openai_humaneval) ➜ LLaDA SFT 所需格式
---------------------------------------------------------
输入:  Huggingface datasets 中的 HumanEval 数据集（prompt, canonical_solution）
输出:  JSONL (键: input_ids, prompt_length)
用法示例:
    python3 preprocess_humaneval_to_llada.py \
        --out_file /path/humaneval_sft_processed.jsonl \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --split test \
        --max_len 4096
"""

import argparse, json, os, sys
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

def encode_example(prompt: str, solution: str, tok) -> dict:
    """
    把一条 HumanEval 样本编码成 LLaDA 需要的
    input_ids 与 prompt_length。
    """
    # 清理和格式化输入
    user_txt = prompt if isinstance(prompt, str) else str(prompt)
    asst_txt = solution if isinstance(solution, str) else str(solution)
    
    # 构建对话格式
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + user_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + asst_txt + SPECIAL["EOS"]

    # 分词
    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    prompt_len = len(user_ids)
    
    return dict(input_ids=ids, prompt_length=prompt_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/humaneval_reasoning_sft_str_processed.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="用于分词的 LLaDA 模型路径")
    ap.add_argument("--split", type=str, default="test", choices=["test"],
                    help="数据划分（HumanEval 只有 test split）")
    ap.add_argument("--max_len", type=int, default=None,
                    help="可选：仅保留 prompt tokenized 长度 ≤ max_len 的样本")
    ap.add_argument("--limit", type=int, default=None,
                    help="可选：仅处理前 N 条，方便试跑")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )

    print(f"✓ 加载 HumanEval 数据集 split={args.split} ...")
    # 加载 HumanEval 数据集
    dataset = load_dataset("openai/openai_humaneval", split=args.split)

    kept, skipped = 0, 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        pbar = tqdm(dataset, desc=f"convert ({args.split})")
        for i, ex in enumerate(pbar):
            if args.limit is not None and kept >= args.limit:
                break

            # HumanEval 字段名：prompt, canonical_solution
            prompt = ex.get("prompt", "")
            solution = ex.get("canonical_solution", "")

            # 过滤空样本
            if not prompt or not solution:
                skipped += 1
                continue

            item = encode_example(prompt, solution, tok)

            # 按需基于 prompt 长度做过滤（只看 user_ids 长度）
            if args.max_len is not None and item["prompt_length"] > args.max_len:
                skipped += 1
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 100 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)

    print(f"✓ 完成: 写入 {kept} 条；跳过 {skipped} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
