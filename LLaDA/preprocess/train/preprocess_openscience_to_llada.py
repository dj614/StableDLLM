#!/usr/bin/env python
# coding: utf-8
"""
把 nvidia/OpenScience ➜ LLaDA SFT 所需格式
------------------------------------------------
输入:  HF 数据集 nvidia/OpenScience （字段: input & output）
输出:  JSONL （键: input_ids, prompt_length）
用法:
    python3 preprocess_openscience_to_llada.py
"""

import argparse, json, tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

SPECIAL = dict(
    BOS="<s>",
    EOS="</s>",
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",
)

def hf_url(model_name: str, use_china: bool):
    """根据国内/国外选择 HF 模型下载地址"""
    if use_china:
        return f"https://hf-mirror.com/{model_name}"
    return model_name

def load_hf_dataset(name: str, split: str, use_china: bool):
    """数据集同样支持国内镜像"""
    if use_china:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    return load_dataset(name, split=split)

def encode_example(ex, tok):
    prompt_txt = ex["input"]
    answer_txt = ex["output"]

    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids

    prompt_len = len(user_ids)
    return dict(input_ids=ids, prompt_length=prompt_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file",
                    type=str,
                    default="data/train/openscience.jsonl")
    ap.add_argument("--model_path",
                    type=str,
                    default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max_num",
                    type=int,
                    default=5000)
    ap.add_argument("--china", action="store_true", help="是否使用国内镜像 hf-mirror.com")
    args = ap.parse_args()

    # 1. tok 走镜像
    model_path = hf_url(args.model_path, args.china)
    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )

    # 2. 数据集走镜像
    print("✓ 加载 nvidia/OpenScience 数据集 ...")
    dataset = load_hf_dataset("nvidia/OpenScience", split="train", use_china=args.china)

    # 3. 取前 max_num 条
    dataset = dataset.select(range(args.max_num))

    # 4. 写出
    print(f"✓ 转换中，最多 {args.max_num} 条 ...")
    out_f = open(args.out_file, "w", encoding="utf8")
    kept = 0

    for ex in tqdm.tqdm(dataset, desc="convert"):
        item = encode_example(ex, tok)
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1

    out_f.close()
    print(f"✓ 完成: 写入 {kept} 条到 {args.out_file}")


if __name__ == "__main__":
    main()
