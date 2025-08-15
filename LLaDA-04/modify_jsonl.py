#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse


def modify_record(record):
    """
    对单条记录进行修改：
    1) prompt 字段中替换提示语
    2) response 字段中将 \\boxed{...} 换为 ####...
    """
    # 1. 修改 prompt 中的提示语
    record['prompt'] = record['prompt'].replace(
        'Please reason step by step, and put your final answer within \\boxed{}',
        'Please reason step by step, and put your final answer after ####'
    )

    # 2. 修改 response：匹配 \\boxed{内容}，替换为 ####内容
    record['response'] = re.sub(r'\\boxed\{(.*?)\}', r'####\1', record['response'])

    return record


def process_file(input_path, output_path):
    """
    读取 input_path 中的每一行 JSONL，进行字段修改后写入 output_path
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data = modify_record(data)
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量修改 JSONL 文件中 prompt 和 response 的格式"
    )
    parser.add_argument(
        "--input", "-i",
        default="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str.jsonl",
        help="输入 JSONL 文件路径（默认会处理 hitab_reasoning_sft_str.jsonl）"
    )
    parser.add_argument(
        "--output", "-o",
        default="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_modified.jsonl",
        help="输出 JSONL 文件路径（默认输出 hitab_reasoning_sft_str_modified.jsonl）"
    )
    args = parser.parse_args()
    process_file(args.input, args.output)
    print(f"已完成：{args.input} → {args.output}")
