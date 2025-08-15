import json
import re
from transformers import AutoTokenizer
from is_coord_token import is_coord_token

"""
输入：processed i.e. tokenized dataset，包含 input_ids（tokenized input+output）及prompt_length（input_ids[:prompt_length]=prompt）
输出：统计值
"""

# —— 配置 —— #
TOKENIZER_NAME = "GSAI-ML/LLaDA-8B-Instruct"
# /storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl
# /storage/v-mengnijia/LLaDA/codexglue_reasoning_sft_str_processed.jsonl
JSONL_PATH     = "/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl"
# hitab-html
# codexglue-json
fmt            = "hitab-html"

# —— 加载 tokenizer —— #
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# —— 统计初始化 —— #
max_input_len    = 0
max_prompt_len   = 0
total_input_len  = 0
total_prompt_len = 0
total_struct     = 0
total_nonstruct  = 0
struct_symbols   = set()
n                = 0

# —— 遍历文件 —— #
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        ids = rec['input_ids']
        pl  = rec['prompt_length']
        max_input_len    = max(max_input_len, len(ids))
        max_prompt_len   = max(max_prompt_len, pl)
        total_input_len  += len(ids)
        total_prompt_len += pl
        # 转回 tokens
        tokens = tokenizer.convert_ids_to_tokens(ids[:pl])
        # 统计本条中的结构化/非结构化
        cnt_struct = 0
        for t in tokens:
            if is_coord_token(t, fmt):
                cnt_struct += 1
                struct_symbols.add(t)
        total_struct    += cnt_struct
        total_nonstruct += (pl - cnt_struct)
        n += 1

# —— 计算并打印 —— #
print(f"共处理记录数：{n}")
print(f"最长 input_ids 长度：{max_input_len}")
print(f"最长 prompt_length：{max_prompt_len}")
print(f"平均 input_ids 长度：{total_input_len/n:.2f}")
print(f"平均 prompt_length：{total_prompt_len/n:.2f}")
print(f"平均结构化符号数：{total_struct/n:.2f}")
print(f"平均非结构化符号数：{total_nonstruct/n:.2f}")
print(f"出现过的所有结构化符号数量：{len(struct_symbols)}")
print(f"\n出现过的所有结构化符号：{sorted(struct_symbols)}")
