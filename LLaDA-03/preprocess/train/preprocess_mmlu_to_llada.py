#!/usr/bin/env python
# coding: utf-8
"""
把 MMLU (cais/mmlu) ➜ LLaDA SFT 所需格式
---------------------------------------------------------
输入:  HuggingFace datasets 中的 MMLU 数据集（question, choices, answer）
输出:  JSONL (键: input_ids, prompt_length)
用法示例:
    python3 preprocess_mmlu_to_llada.py \
        --out_file /path/mmlu_sft_processed.jsonl \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --subjects algebra anatomy astronomy \
        --split test \
        --max_len 4096
"""

import argparse, json, os, sys
from typing import List
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

# MMLU 包含的所有学科
ALL_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy',
    'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
    'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
    'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging',
    'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
    'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies', 'sociology',
    'us_foreign_policy', 'virology', 'world_religions'
]

# 默认使用的学科（数学、科学、计算机相关）
DEFAULT_SUBJECTS = [
    'abstract_algebra', 'college_mathematics', 'elementary_mathematics',
    'high_school_mathematics', 'high_school_statistics', 'college_computer_science',
    'high_school_computer_science', 'machine_learning', 'computer_security'
]


def format_question_with_choices(question: str, choices: List[str]) -> str:
    """
    将问题和选项格式化成标准的多选题格式。
    """
    formatted = question.strip()
    if not formatted.endswith('?'):
        formatted += "\n\nOptions:"
    else:
        formatted += "\n\nOptions:"
    
    for i, choice in enumerate(choices):
        formatted += f"\n{chr(65 + i)}. {choice}"  # A, B, C, D
    
    formatted += "\n\nAnswer:"
    return formatted


def encode_example(question: str, choices: List[str], answer: str, tok) -> dict:
    """
    把一条 MMLU 样本编码成 LLaDA 需要的
    input_ids 与 prompt_length。
    """
    # 格式化问题
    formatted_question = format_question_with_choices(question, choices)
    
    # 答案处理：MMLU 的答案是数字索引，转换为字母
    if answer.isdigit():
        answer_letter = chr(65 + int(answer))  # 0->A, 1->B, 2->C, 3->D
    else:
        answer_letter = answer.upper()  # 如果已经是字母，直接使用
    
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + formatted_question + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_letter + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    prompt_len = len(user_ids)
    return dict(input_ids=ids, prompt_length=prompt_len)


def load_mmlu_splits(subjects: List[str], split: str) -> Dataset:
    """
    逐个学科加载后 concat 成一个 Dataset。
    cais/mmlu 每个学科是独立的子集。
    """
    all_parts = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
            print(f"✓ 加载学科: {subject} ({len(ds)} 样本)")
            all_parts.append(ds)
        except Exception as e:
            print(f"⚠️  跳过学科 {subject}: {e}")
            continue
    
    if len(all_parts) == 0:
        raise ValueError(f"没有成功加载任何学科数据，请检查学科名称和分割名称")
    
    if len(all_parts) == 1:
        return all_parts[0]
    
    # 延迟导入以避免 datasets 旧版本无 concat_datasets 时报错
    from datasets import concatenate_datasets
    return concatenate_datasets(all_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/mmlu_reasoning_sft_str.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="用于分词的 LLaDA 模型路径")
    ap.add_argument("--subjects", type=str, nargs="*", default=DEFAULT_SUBJECTS,
                    help=f"MMLU 学科列表，默认: {DEFAULT_SUBJECTS[:3]}... (共{len(DEFAULT_SUBJECTS)}个)")
    ap.add_argument("--split", type=str, default="test", choices=["dev", "test"],
                    help="数据划分（dev/test，MMLU 没有train split）")
    ap.add_argument("--max_len", type=int, default=None,
                    help="可选：仅保留 prompt tokenized 长度 ≤ max_len 的样本")
    ap.add_argument("--limit", type=int, default=None,
                    help="可选：仅处理前 N 条，方便试跑")
    ap.add_argument("--use_all_subjects", action="store_true",
                    help="使用所有 57 个学科而不是默认的数学/计算机相关学科")
    args = ap.parse_args()

    # 如果指定使用所有学科
    if args.use_all_subjects:
        subjects_to_use = ALL_SUBJECTS
        print(f"✓ 将处理所有 {len(ALL_SUBJECTS)} 个学科")
    else:
        subjects_to_use = args.subjects
        print(f"✓ 将处理 {len(subjects_to_use)} 个学科: {subjects_to_use}")

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    # tokenizer（与其他脚本一致）
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )

    # 加载并合并所需学科
    print(f"✓ 开始加载 MMLU 数据集...")
    ds = load_mmlu_splits(subjects_to_use, args.split)
    print(f"✓ 总共加载 {len(ds)} 条数据")

    kept, skipped = 0, 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        pbar = tqdm(ds, desc=f"convert ({args.split}, {len(subjects_to_use)} subjects)")
        for i, ex in enumerate(pbar):
            if args.limit is not None and kept >= args.limit:
                break

            # MMLU 字段名：question, choices, answer
            question = ex.get("question", "")
            choices = ex.get("choices", [])
            answer = str(ex.get("answer", ""))

            # 过滤无效样本
            if not question or not choices or len(choices) != 4:
                skipped += 1
                continue

            try:
                item = encode_example(question, choices, answer, tok)
            except Exception as e:
                print(f"⚠️  编码失败: {e}")
                skipped += 1
                continue

            # 按需基于 prompt 长度做过滤（只看 user_ids 长度）
            if args.max_len is not None and item["prompt_length"] > args.max_len:
                skipped += 1
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 1000 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)

    print(f"✓ 完成: 写入 {kept} 条；跳过 {skipped} 条 -> {args.out_file}")


if __name__ == "__main__":
    main()