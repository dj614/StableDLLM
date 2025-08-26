#!/usr/bin/env python
# coding: utf-8
"""
把 GPQA (Idavidrein/gpqa) ➜ LLaDA SFT 所需格式
---------------------------------------------------------
输入:  Huggingface datasets 中的 GPQA 数据集（多选题格式）
输出:  JSONL (键: input_ids, prompt_length)
用法示例:
    # 默认处理 gpqa_main 配置
    python3 preprocess_gpqa_to_llada.py
    
    # 处理不同的配置
    python3 preprocess_gpqa_to_llada.py --config gpqa_extended
    python3 preprocess_gpqa_to_llada.py --config gpqa_diamond
    python3 preprocess_gpqa_to_llada.py --config gpqa_experts
    
    # 指定输出文件和其他参数
    python3 preprocess_gpqa_to_llada.py \
        --config gpqa_main \
        --out_file /path/gpqa_sft_processed.jsonl \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --max_len 4096
"""

import argparse, json, os, sys
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

def format_multiple_choice_question(question: str, choices: list) -> str:
    """
    将多选题格式化为标准的问题格式
    """
    formatted_question = question.strip()
    if not formatted_question.endswith('?'):
        formatted_question += "\n\nPlease select the correct answer from the following choices:"
    else:
        formatted_question += "\n\nChoices:"
    
    # 添加选择选项 (A), (B), (C), (D)
    choice_labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            formatted_question += f"\n({choice_labels[i]}) {choice.strip()}"
    
    return formatted_question

def get_answer_letter(correct_answer_index: int) -> str:
    """
    根据正确答案的索引返回对应的字母
    """
    choice_labels = ['A', 'B', 'C', 'D']
    if 0 <= correct_answer_index < len(choice_labels):
        return choice_labels[correct_answer_index]
    return 'A'  # 默认返回 A

def encode_example(question: str, correct_answer: str, incorrect_answers: list, tok) -> dict:
    """
    把一条 GPQA 样本编码成 LLaDA 需要的
    input_ids 与 prompt_length。
    """
    # 构建所有选择选项
    choices = [correct_answer] + incorrect_answers
    # 随机打乱选择顺序但记住正确答案的新位置
    import random
    choice_pairs = [(choice, i == 0) for i, choice in enumerate(choices)]
    random.shuffle(choice_pairs)
    
    # 找到正确答案的新位置
    correct_index = 0
    shuffled_choices = []
    for i, (choice, is_correct) in enumerate(choice_pairs):
        shuffled_choices.append(choice)
        if is_correct:
            correct_index = i
    
    # 格式化问题和选择
    formatted_question = format_multiple_choice_question(question, shuffled_choices)
    
    # 获取答案字母
    answer_letter = get_answer_letter(correct_index)
    answer_text = f"The correct answer is ({answer_letter})."
    
    # 构建对话格式
    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + formatted_question + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_text + SPECIAL["EOS"]

    # 分词
    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    prompt_len = len(user_ids)
    
    return dict(input_ids=ids, prompt_length=prompt_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/train/gpqa_reasoning_sft_str_processed.jsonl",
                    help="输出 JSONL 路径")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                    help="用于分词的 LLaDA 模型路径")
    ap.add_argument("--config", type=str, default="gpqa_main", 
                    choices=["gpqa_extended", "gpqa_main", "gpqa_diamond", "gpqa_experts"],
                    help="GPQA 数据集配置")
    ap.add_argument("--split", type=str, default="train", 
                    help="数据划分，可能的值取决于数据集")
    ap.add_argument("--max_len", type=int, default=None,
                    help="可选：仅保留 prompt tokenized 长度 ≤ max_len 的样本")
    ap.add_argument("--limit", type=int, default=None,
                    help="可选：仅处理前 N 条，方便试跑")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print("✓ 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )

    print(f"✓ 加载 GPQA 数据集 config={args.config} split={args.split} ...")
    try:
        # 加载 GPQA 数据集
        dataset = load_dataset("Idavidrein/gpqa", args.config, split=args.split)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("尝试加载不同的 split...")
        # 尝试其他可能的 split
        try:
            dataset_dict = load_dataset("Idavidrein/gpqa", args.config)
            print(f"数据集的 splits: {list(dataset_dict.keys())}")
            # 使用第一个可用的 split
            if dataset_dict:
                split_name = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_name]
                print(f"使用 split: {split_name}")
                args.split = split_name  # 更新 split 名称用于显示
            else:
                raise ValueError("无法加载数据集")
        except Exception as e2:
            print(f"最终加载失败: {e2}")
            sys.exit(1)

    kept, skipped = 0, 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        pbar = tqdm(dataset, desc=f"convert ({args.split})")
        for i, ex in enumerate(pbar):
            if args.limit is not None and kept >= args.limit:
                break

            # GPQA 字段名
            question = ex.get("Question", "")
            correct_answer = ex.get("Correct Answer", "")
            incorrect_answers = [
                ex.get("Incorrect Answer 1", ""),
                ex.get("Incorrect Answer 2", ""), 
                ex.get("Incorrect Answer 3", "")
            ]
            
            # 过滤空答案
            incorrect_answers = [ans for ans in incorrect_answers if ans.strip()]

            # 过滤无效样本
            if not question or not correct_answer or len(incorrect_answers) < 3:
                skipped += 1
                if i < 5:  # 只显示前5个跳过的样本
                    print(f"跳过样本 {i}: question='{question[:50]}...', correct_answer='{correct_answer}', incorrect_count={len(incorrect_answers)}")
                continue

            try:
                item = encode_example(question, correct_answer, incorrect_answers, tok)
            except Exception as e:
                print(f"编码样本 {i} 失败: {e}")
                skipped += 1
                continue

            # 按需基于 prompt 长度做过滤
            if args.max_len is not None and item["prompt_length"] > args.max_len:
                skipped += 1
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 100 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)

    print(f"✓ 完成: 写入 {kept} 条；跳过 {skipped} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
