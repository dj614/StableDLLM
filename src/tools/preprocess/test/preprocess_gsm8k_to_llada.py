#!/usr/bin/env python
# coding: utf-8
"""
GSM8K (main, test) ➜ 推理数据（与 MATH 脚本同风格：规范化、去重、稳健提取）
输出 JSONL 键：
  - data_source: "GSM8K"
  - prompt: str         (question)
  - groundtruth: list[str]  （从 answer 中抽取所有 '#### ' 后的最终答案，已规范化与去重）

用法：
  python3 preprocess_gsm8k_infer_like_math.py \
    --out_file /path/gsm8k_infer_main_test.jsonl \
    --limit 200 \
    --normalize_math true
"""

import argparse, json, os, re
from typing import List
from datasets import load_dataset
from tqdm import tqdm

# ========== 正则与规范化工具 ==========
# 匹配形如：... #### 42\n ；捕获 #### 后到行末（不含换行）
HASH4_REGEX = re.compile(r"####\s*([^\n\r]+)")

LATEX_LEFT_RIGHT = re.compile(r"\\left\s*|\\right\s*")
LATEX_TEXT = re.compile(r"\\text\s*\{([^{}]*)\}")
LATEX_SPACES = re.compile(r"\\[ ,;:!]")
FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")

def basic_clean(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # 去外层 $...$
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_math_expr(s: str) -> str:
    """与 MATH 版一致的轻量规范化（不做 / 或 , 的拆分）。"""
    if not s:
        return ""
    s = basic_clean(s)

    # 去 \left \right
    s = LATEX_LEFT_RIGHT.sub("", s)

    # 展开 \text{...}
    while True:
        ns = LATEX_TEXT.sub(lambda m: m.group(1), s)
        if ns == s:
            break
        s = ns

    # \frac{a}{b} -> a/b  （递归替换）
    while True:
        ns = FRAC.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", s)
        if ns == s:
            break
        s = ns

    # \sqrt{x} -> sqrt(x)
    s = SQRT.sub(lambda m: f"sqrt({m.group(1).strip()})", s)

    # 常见符号
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/").replace(r"\pm", "±")
    s = s.replace(r"\$", "$")

    # 幂：x^{2} -> x**(2), x^2 -> x**2
    s = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^\s*([A-Za-z0-9]+)", r"**\1", s)

    # 去 LaTeX 空白控制
    s = LATEX_SPACES.sub("", s)

    # 补齐 .5 -> 0.5
    s = re.sub(r"(?<!\d)\.(\d+)", r"0.\1", s)

    # 最终压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def extract_final_answers(answer_text: str, normalize: bool=True) -> List[str]:
    """抓取 answer 中所有 '#### ...' 的内容；按需规范化并去重。"""
    if not answer_text:
        return []
    raw = [m.strip() for m in HASH4_REGEX.findall(answer_text)]
    if normalize:
        raw = [normalize_math_expr(x) for x in raw]
    else:
        raw = [basic_clean(x) for x in raw]
    return dedupe_preserve_order(raw)

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./LLaDA/data/test/gsm8k_test_llada.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，便于试跑")
    ap.add_argument("--normalize_math", type=str, default="true",
                    help="是否做数学规范化（true/false），默认 true")
    args = ap.parse_args()

    normalize_math = str(args.normalize_math).lower() in {"1","true","yes","y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    # GSM8K 的主要数据在 config="main", split="test"
    ds = load_dataset("gsm8k", "main", split="test")

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for ex in tqdm(ds, desc="convert gsm8k::main/test"):
            if args.limit is not None and kept >= args.limit:
                break
            q = ex.get("question", "") or ""
            a = ex.get("answer", "") or ""

            gts = extract_final_answers(a, normalize=normalize_math)

            item = {
                "data_source": "GSM8K",
                "prompt": q,
                "groundtruth": gts,  # list[str]，按出现顺序，已规范化+去重
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
