#!/usr/bin/env python
# coding: utf-8
"""
Databricks Dolly 15k -> LLaDA 训练/推理数据
- 仅保留 category in {"closed_qa", "classification"}
- 前 80% 作为训练数据：输出 input_ids, prompt_length
- 后 20% 作为推理数据：输出 prompt, response（不 tokenize）

用法：
  python3 preprocess_dolly15k_to_llada.py \
    --train_out dolly15k_reasoning_sft_str_processed.jsonl \
    --test_out  dolly15k_test_llada.jsonl \
    --model_path GSAI-ML/LLaDA-8B-Instruct
"""

import argparse, json, os
from typing import Dict
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

KEEP_CATS = {"closed_qa", "classification"}

def build_prompt(instruction: str, context: str) -> str:
    instr = instruction.strip() if isinstance(instruction, str) else ""
    ctx   = context.strip() if isinstance(context, str) else ""
    # instruction + 可选 context（空则不拼）
    core = instr if not ctx else f"{instr}\n\n{ctx}"
    return core

def encode_llada_example(prompt_txt: str, answer_txt: str, tok) -> Dict:
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    return dict(input_ids=ids, prompt_length=len(user_ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_out", type=str, default="data/train/dolly15k_reasoning_sft_str_processed.jsonl")
    ap.add_argument("--test_out",  type=str, default="data/test/dolly15k_test_llada.jsonl")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    for p in [args.train_out, args.test_out]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    print("• Loading dataset databricks/databricks-dolly-15k ...")
    ds = load_dataset("databricks/databricks-dolly-15k")["train"]  # 全量在 train split

    # 过滤类别
    ds = ds.filter(lambda ex: (ex.get("category") or "") in KEEP_CATS)

    n = len(ds)
    train_n = int(n * 0.8)
    assert train_n > 0 and train_n < n, f"bad split sizes: n={n}, train_n={train_n}"

    ds_train = ds.select(range(train_n))
    ds_test  = ds.select(range(train_n, n))

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    # 写训练文件（tokenize）
    kept = 0
    with open(args.train_out, "w", encoding="utf8") as fout:
        for ex in tqdm(ds_train, desc="encode train"):
            instr   = ex.get("instruction", "") or ""
            context = ex.get("context", "") or ""
            resp    = ex.get("response", "") or ""

            prompt_txt = build_prompt(instr, context)
            item = encode_llada_example(prompt_txt, resp, tok)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"✓ 训练集写入 {kept} 条 -> {args.train_out}")

    # 写测试文件（不 tokenize，只保留 prompt/response）
    kept = 0
    with open(args.test_out, "w", encoding="utf8") as fout:
        for ex in tqdm(ds_test, desc="write test"):
            instr   = ex.get("instruction", "") or ""
            context = ex.get("context", "") or ""
            resp    = ex.get("response", "") or ""

            prompt_txt = build_prompt(instr, context)
            item = dict(prompt=prompt_txt, response=resp)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"✓ 推理集写入 {kept} 条 -> {args.test_out}")

if __name__ == "__main__":
    main()
