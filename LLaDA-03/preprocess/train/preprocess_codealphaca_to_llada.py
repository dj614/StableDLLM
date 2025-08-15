#!/usr/bin/env python
# coding: utf-8
"""
处理 CodeAlpaca_20K 训练数据 -> LLaDA SFT 格式
- 使用 train split
- 键值：input_ids, prompt_length
- prompt = dataset['prompt']，response = dataset['completion']
- 模板格式与之前 HiTab/MATH 处理一致
"""

import argparse
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# LLaDA 模板特殊 token
SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

def encode_llada_example(prompt_txt: str, answer_txt: str, tok):
    """
    把一条 (prompt, answer) 编码成 LLaDA SFT 格式的 input_ids 和 prompt_length
    """
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    return dict(input_ids=ids, prompt_length=len(user_ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/codealpaca20k_sft_processed.jsonl",
                    help="输出 JSONL 文件路径")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="用于分词的 LLaDA 模型路径")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，方便调试")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print("• 加载数据集 HuggingFaceH4/CodeAlpaca_20K ...")
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as fout:
        for ex in tqdm(ds, desc="encode"):
            if args.limit is not None and kept >= args.limit:
                break

            prompt   = ex.get("prompt", "") or ""
            response = ex.get("completion", "") or ""

            item = encode_llada_example(prompt.strip(), response.strip(), tok)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
