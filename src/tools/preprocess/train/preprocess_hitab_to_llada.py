#!/usr/bin/env python
# coding: utf-8
"""
把 HiTab JSONL ➜ LLaDA SFT 所需格式
------------------------------------------------
输入:  每行 {"prompt": "...", "response": "..."}
输出:  同目录  *.jsonl  （键: input_ids, prompt_length）
用法:
    python3 preprocess_hitab_to_llada.py
"""
import argparse, json, sys, tqdm, os
from transformers import AutoTokenizer

SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

def encode_example(ex, tok):
    prompt_txt   = ex["prompt"]
    answer_txt   = ex["response"]

    user_part  = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part  = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids  = tok(user_part , add_special_tokens=False).input_ids
    asst_ids  = tok(asst_part, add_special_tokens=False).input_ids
    ids       = user_ids + asst_ids
    prompt_len = len(user_ids)
    return dict(input_ids=ids, prompt_length=prompt_len)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  type=str, default="./LLaDA/hitab_reasoning_sft_str.jsonl")
    ap.add_argument("--out_file", type=str, default="./LLaDA/data/train/hitab_reasoning_sft_str_processed.jsonl")
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    out_f = open(args.out_file, "w", encoding="utf8")
    kept, skipped = 0, 0
    with open(args.in_file, "r", encoding="utf8") as f:
        for line in tqdm.tqdm(f, desc="convert"):
            ex = json.loads(line)
            item = encode_example(ex, tok)
            if item is None:
                skipped += 1
                continue
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    out_f.close()
    print(f"✓ 完成: 写入 {kept} 条")

if __name__ == "__main__":
    main()