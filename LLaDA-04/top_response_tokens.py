#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
top_response_tokens.py

统计数据集中 RESPONSE 部分最常出现的 token，
并额外打印 id 59(\), 32289(boxed), 90({) 的出现次数。
"""

import argparse, json, collections
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
        default="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl",
        help="JSONL 文件路径；字段需有 input_ids, prompt_length")
    ap.add_argument("--topk", type=int, default=50,
        help="输出出现次数最多的前 K 个 token")
    ap.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
        help="HuggingFace tokenizer 名称或本地路径，用于把 id 反解成 token")
    # --- 可选：把目标稀疏 id 列成参数 ---
    ap.add_argument("--watch_ids", type=int, nargs='*',
        default=[2262],
        help="另外关心的 token id;[59,795,32289,90,28504,7684,92] for hitab ;[2262] for gsm8k")
    return ap.parse_args()

def main():
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"✗ data 文件不存在: {args.data}")

    counter = collections.Counter()

    # ---------- 1. 读取并统计 ----------
    with open(args.data, "r", encoding="utf-8") as f:
        for ln in f:
            ex = json.loads(ln)
            ids = ex["input_ids"]
            p_len = ex.get("prompt_length", 0)
            counter.update(ids[p_len:])   # 只计 response

    # ---------- 2. 打印 watch_ids ----------  ### NEW ###
    if args.watch_ids:
        print("\n=== Occurrence of specified ids in RESPONSE ===")
        if args.tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                args.tokenizer, use_fast=True, trust_remote_code=True
            )
        for tid in args.watch_ids:
            cnt = counter.get(tid, 0)
            if args.tokenizer:
                token_txt = tok.convert_ids_to_tokens(tid, skip_special_tokens=False)
                print(f"id {tid:7d} ({token_txt}): {cnt}")
            else:
                print(f"id {tid:7d}: {cnt}")
        print("-" * 45)

    # ---------- 3. 打印 top‑K ----------
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = locals().get("tok") or AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True, trust_remote_code=True
        )
        def fmt(item):
            tid, cnt = item
            token_text = tok.convert_ids_to_tokens(tid, skip_special_tokens=False)
            return tid, token_text, cnt
        records = [fmt(it) for it in counter.most_common(args.topk)]
        print(f"\nTop-{args.topk} tokens in RESPONSE:")
        print(f"{'rank':>4} | {'id':>7} | {'count':>9} | token")
        print("-"*45)
        for i,(tid,txt,cnt) in enumerate(records,1):
            print(f"{i:4d} | {tid:7d} | {cnt:9d} | {txt}")
    else:
        records = counter.most_common(args.topk)
        print(f"\nTop-{args.topk} token ids in RESPONSE (no decoding):")
        print(f"{'rank':>4} | {'id':>7} | {'count':>9}")
        print("-"*27)
        for i,(tid,cnt) in enumerate(records,1):
            print(f"{i:4d} | {tid:7d} | {cnt:9d}")

if __name__ == "__main__":
    main()
