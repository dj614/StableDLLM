#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import sys

def process_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}", file=sys.stderr)
                continue

            gt = obj.get('groundtruth', None)
            if isinstance(gt, str):
                # Try JSON parse first
                try:
                    parsed = json.loads(gt)
                except json.JSONDecodeError:
                    # Fallback to Python literal eval (e.g. single quotes, non-JSON)
                    try:
                        parsed = eval(gt, {}, {})
                    except Exception as e:
                        print(f"Could not parse groundtruth on line: {e}", file=sys.stderr)
                        parsed = []
                # Ensure we have a list
                if not isinstance(parsed, list):
                    parsed = [parsed]
                obj['groundtruth'] = parsed

            # Write back as JSON, preserving unicode
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert string-encoded groundtruth fields into real lists."
    )
    parser.add_argument("input", default="hitab_reasoning_sft_str.jsonl")
    parser.add_argument("output", default="hitab_reasoning_sft_str_unigrpo.jsonl")
    args = parser.parse_args()
    process_file(args.input, args.output)
    print(f"Done! Processed file written to {args.output}")