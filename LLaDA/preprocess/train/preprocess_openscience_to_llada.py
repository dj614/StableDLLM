#!/usr/bin/env python
# coding: utf-8

import os, sys

# >>> 镜像配置（必须在 import datasets/transformers 前） <<<
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
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + ex["input"] + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + ex["output"] + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    return user_ids + asst_ids, len(user_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/openscience.jsonl")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max_num", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--china", action="store_true")
    args = ap.parse_args()

    # 确保输出路径存在
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    print("✓ 流式加载 nvidia/OpenScienceReasoning-2（不下载整个数据集）...")
    stream_ds = load_dataset(
        "nvidia/OpenScienceReasoning-2",
        split="train",
        streaming=True  # <<< 最关键：只流式下载，不加载全量
    )

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm.tqdm(stream_ds, desc="streaming-filter"):

            input_ids, prompt_len = encode_example(ex, tok)

            if len(input_ids) >= args.max_len:
                continue

            out_f.write(json.dumps(
                {"input_ids": input_ids, "prompt_length": prompt_len},
                ensure_ascii=False
            ) + "\n")

            kept += 1
            if kept >= args.max_num:
                break

    print(f"✓ 完成！共保存 {kept} 条满足条件的样本到 {args.out_file}")


if __name__ == "__main__":
    main()
