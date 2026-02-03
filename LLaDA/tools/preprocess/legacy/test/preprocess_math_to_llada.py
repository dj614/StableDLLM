#!/usr/bin/env python
# coding: utf-8
"""
MATH test split ➜ 推理数据（更稳健的 boxed 抽取 + 顶层拆分 + 规范化）
输出 JSONL 键：
  - data_source: "MATH"
  - prompt: str
  - groundtruth: list[str]  （多答案已“括号感知拆分”，去重且格式化）

用法示例：
  python3 preprocess_math_infer_fixed_toplvl.py \
    --out_file /path/math_infer_test.jsonl \
    --categories algebra geometry number_theory \
    --limit 200 \
    --split_commas false \
    --normalize_math true
"""

import argparse, json, os, re
from typing import List
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_CATEGORIES = [
    "algebra",
    "precalculus",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
]

# ----------- 1) 精确提取 \boxed{...}，用栈处理嵌套 -----------
def extract_boxed_all(text: str) -> List[str]:
    if not text:
        return []
    s = text
    out = []
    i, n = 0, len(s)
    key = r"\boxed"
    while i < n:
        j = s.find(key, i)
        if j < 0:
            break
        k = j + len(key)
        # 跳过空白
        while k < n and s[k].isspace():
            k += 1
        if k >= n or s[k] != "{":
            i = k
            continue
        # 从 k 开始匹配成对花括号
        depth = 0
        start = k
        p = k
        while p < n:
            ch = s[p]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # 内容在 (start+1 .. p-1)
                    out.append(s[start+1:p].strip())
                    i = p + 1
                    break
            p += 1
        else:
            # 未闭合，退出
            break
    return out

# ----------- 2) 顶层拆分（只在括号/中括号/大括号深度为0时切）-----------
def top_level_split(s: str, split_commas: bool=False) -> List[str]:
    """
    仅在 () [] {} 深度都为0 时，按如下分隔符切分：
      - 'or'（词边界）
      - 'and'（词边界）
      - ';'
      - （可选）','  —— 但不会切开区间 [a,b] 或函数参数，因为它们处于 [] 或 () 内，深度>0
    """
    if not s:
        return []
    n = len(s)
    i = 0
    parts = []
    buf = []
    dep_par, dep_bra, dep_cur = 0, 0, 0

    def is_word_char(c):
        return c.isalpha()

    def flush():
        token = "".join(buf).strip()
        if token:
            parts.append(token)
        buf.clear()

    while i < n:
        ch = s[i]

        # 更新深度
        if ch == "(":
            dep_par += 1
        elif ch == ")":
            dep_par = max(0, dep_par - 1)
        elif ch == "[":
            dep_bra += 1
        elif ch == "]":
            dep_bra = max(0, dep_bra - 1)
        elif ch == "{":
            dep_cur += 1
        elif ch == "}":
            dep_cur = max(0, dep_cur - 1)

        at_top = (dep_par == 0 and dep_bra == 0 and dep_cur == 0)

        # 顶层分号
        if at_top and ch == ";":
            flush()
            i += 1
            continue

        # 顶层逗号（可选）
        if at_top and split_commas and ch == ",":
            flush()
            i += 1
            continue

        # 顶层 'or' / 'and' （词边界）
        if at_top and ch.isalpha():
            # 预读单词
            j = i
            while j < n and is_word_char(s[j]):
                j += 1
            word = s[i:j]
            prev = s[i-1] if i-1 >= 0 else " "
            nxt  = s[j]   if j < n else " "
            if word in ("or", "and") and (not is_word_char(prev)) and (not is_word_char(nxt)):
                flush()
                i = j
                continue
            # 否则正常累积
            buf.append(ch)
            i += 1
            continue

        # 常规累积
        buf.append(ch)
        i += 1

    flush()
    return [p for p in parts if p]

# ----------- 3) 规范化：轻量 + 数学表达式 -----------
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
    # 收紧空白
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_math_expr(s: str) -> str:
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

    # \frac{a}{b} -> a/b  （反复直到收敛）
    while True:
        ns = FRAC.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", s)
        if ns == s:
            break
        s = ns

    # \sqrt{x} -> sqrt(x)
    s = SQRT.sub(lambda m: f"sqrt({m.group(1).strip()})", s)

    # 常见符号
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/").replace(r"\pm", "±")
    s = s.replace(r"\$", "$")  # 转义美元为字面符号

    # 幂：x^{2} -> x**(2), x^2 -> x**2
    s = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^\s*([A-Za-z0-9]+)", r"**\1", s)

    # 去 LaTeX 空白控制
    s = LATEX_SPACES.sub("", s)

    # 补全 .5 -> 0.5
    s = re.sub(r"(?<!\d)\.(\d+)", r"0.\1", s)

    # 最终压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip()
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

# ----------- 4) 主流程 -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="data/test/math_test_llada.jsonl", help="输出 JSONL 路径")
    ap.add_argument("--categories", type=str, nargs="*", default=DEFAULT_CATEGORIES,
                    help=f"MATH 子集列表，默认: {DEFAULT_CATEGORIES}")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    ap.add_argument("--split_commas", type=str, default="true",
                    help="是否用逗号顶层拆分多答案（true/false），默认 true")
    ap.add_argument("--normalize_math", type=str, default="true",
                    help="是否做数学规范化（true/false），默认 true（建议开启）")
    args = ap.parse_args()

    split_commas   = str(args.split_commas).lower()   in {"1","true","yes","y"}
    normalize_math = str(args.normalize_math).lower() in {"1","true","yes","y"}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    kept = 0
    with open(args.out_file, "w", encoding="utf8") as out_f:
        for cat in args.categories:
            ds = load_dataset("EleutherAI/hendrycks_math", cat, split="test")
            for ex in tqdm(ds, desc=f"convert test::{cat}"):
                if args.limit is not None and kept >= args.limit:
                    break

                problem  = ex.get("problem", "")
                solution = ex.get("solution", "")

                # 1) 提取所有 \boxed{...}
                raw_chunks = extract_boxed_all(solution)

                # 2) 顶层拆分（不在括号/区间内部切）
                answers = []
                for ch in raw_chunks:
                    parts = top_level_split(ch, split_commas=split_commas)
                    answers.extend([p for p in parts if p])

                # 3) 规范化（先轻量再数学）；再去重
                if normalize_math:
                    answers = [normalize_math_expr(a) for a in answers]
                else:
                    answers = [basic_clean(a) for a in answers]
                answers = dedupe_preserve_order(answers)

                item = {
                    "data_source": "MATH",
                    "prompt": problem,
                    "groundtruth": answers,   # list[str]，已顶层拆分 + 去重 + 规范化
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
            if args.limit is not None and kept >= args.limit:
                break

    print(f"✓ 完成: 写入 {kept} 条 -> {args.out_file}")

if __name__ == "__main__":
    main()
