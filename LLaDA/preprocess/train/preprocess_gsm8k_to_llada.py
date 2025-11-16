#!/usr/bin/env python
# coding: utf-8

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

def enable_hf_mirror(use_china: bool):
    """
    设置 HuggingFace 镜像，加速模型与数据集下载
    """
    if use_china:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def encode_example(question, answer, tok):
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + question.strip() + "\n" + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer.strip() + "\n" + SPECIAL["EOS"]
    
    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    return dict(input_ids=ids, prompt_length=len(user_ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/gsm8k.jsonl")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--china", action="store_true", help="是否使用国内镜像 hf-mirror.com")
    args = ap.parse_args()

    # 启用镜像（模型+数据集）
    enable_hf_mirror(args.china)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True, 
        trust_remote_code=True
    )

    print(f"✓ 加载 GSM8K 数据集 split={args.split} ...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)

    with open(args.out_file, "w", encoding="utf8") as out_f:
        kept = 0
        for ex in tqdm.tqdm(dataset, desc="Converting"):
            item = encode_example(ex["question"], ex["answer"], tok)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条到 {args.out_file}")

if __name__ == "__main__":
    main()
