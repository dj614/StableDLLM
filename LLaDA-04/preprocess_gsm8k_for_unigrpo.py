#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from datasets import load_dataset

def extract_gsm8k_answer(answer_str: str) -> str:
    """
    GSM8K answers are typically formatted like:

        "#### 5"

    or sometimes with additional explanation. We grab whatever follows '####'
    up to the end of line.
    """
    # look for a line starting with ####
    m = re.search(r"^####\s*(.+)$", answer_str, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # fallback: take the first non-empty line
    for line in answer_str.splitlines():
        line = line.strip()
        if line:
            return line
    return answer_str.strip()


def convert_gsm8k(split: str, output_path: str):
    # load just the requested split
    ds = load_dataset("openai/gsm8k", "main", split=split)

    with open(output_path, "w", encoding="utf-8") as fout:
        for ex in ds:
            question = ex["question"].strip()
            raw_answer = ex["answer"].strip()

            # extract only the numeric/boxed part after '####'
            clean_answer = extract_gsm8k_answer(raw_answer)

            obj = {
                "prompt": question,
                "groundtruth": [clean_answer]
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump GSM8K split to JSONL with prompt+groundtruth"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="which split to process (train/validation/test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_train_unigrpo.jsonl",
        help="path to write the new JSONL file"
    )
    args = parser.parse_args()
    convert_gsm8k(args.split, args.output)
    print(f"Wrote {args.split} → {args.output}")
