#!/usr/bin/env python
# coding: utf-8
"""
把 nvidia/OpenScienceReasoning-2 ➜ LLaDA SFT 所需格式
------------------------------------------------
输入:  HF 数据集 nvidia/OpenScienceReasoning-2 （字段: input & output）
输出:  JSONL （键: input_ids, prompt_length）
过滤: 只保留 prompt+response 总长度 < max_len 的样本
"""

import os, sys

# 提前设置国内镜像（必须在 import transformers/datasets 之前）
if "--china" in sys.argv:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import argparse, json, tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


SPECIAL = dict(
    BOS="<s>",
    EOS="</s>",
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",
)

def encode_example(ex, tok):
    prompt_txt = ex["input"]
    answer_txt = ex["output"]

    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids

    return dict(
        input_ids=ids,
        prompt_length=len(user_ids)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/openscience.jsonl")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max_num", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=4096,
                    help="仅保留 prompt+response token 总长度 < max_len 的样本")
    ap.add_argument("--china", action="store_true", help="使用国内 HF 镜像")
    args = ap.parse_args()

    # 自动创建输出目录
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # 1. 加载 tokenizer
    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )

    # 2. 加载数据集
    print("✓ 加载 nvidia/OpenScienceReasoning-2 数据集 ...")
    dataset = load_dataset("nvidia/OpenScienceReasoning-2", split="train")

    # 3. 逐条处理 + 过滤 + 最多 max_num 条
    print(f"✓ 开始过滤，总长度 < {args.max_len}，最多保留 {args.max_num} 条 ...")
    kept = 0

    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm.tqdm(dataset, desc="convert"):

            item = encode_example(ex, tok)
            total_len = len(item["input_ids"])

            # 长度过滤
            if total_len >= args.max_len:
                continue

            # 保存
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

            # 达到数量上限
            if kept >= args.max_num:
                break

    print(f"✓ 完成: 最终写入 {kept} 条到 {args.out_file}")


if __name__ == "__main__":
    main()
