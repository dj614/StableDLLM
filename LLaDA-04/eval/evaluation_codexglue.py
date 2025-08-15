#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

# —— 配置路径 —— #
TEST_JSONL  = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/llada_code2text/test.jsonl")
PRED_JSONL  = Path("/storage/v-mengnijia/LLaDA/eval/data/selected_core_codexglue(9)/checkpoint-epoch5/predictions_temp0.0_gen128_steps64_block4.jsonl")
EVAL_SCRIPT = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/evaluator/evaluator.py")

# 中间文件
REF_TXT     = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/evaluator/reference.txt")
PRED_TXT    = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/evaluator/predictions.txt")
REF_CLEAN   = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/evaluator/reference_clean.txt")
PRED_CLEAN  = Path("/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/evaluator/predictions_clean.txt")

def build_ref_and_pred():
    # 1) 生成 reference.txt 和 predictions.txt
    with open(TEST_JSONL,'r',encoding='utf-8') as f_test, \
         open(PRED_JSONL,'r',encoding='utf-8') as f_pred, \
         open(REF_TXT,'w',encoding='utf-8') as f_ref, \
         open(PRED_TXT,'w',encoding='utf-8') as f_pt:
        for i, (tline, pline) in enumerate(zip(f_test, f_pred)):
            gold = json.loads(tline).get("response", "").replace("\n"," ").strip()
            pred = json.loads(pline).get("prediction", "").replace("\n"," ").strip()
            f_ref.write(f"{i}\t{gold}\n")
            f_pt.write(f"{i}\t{pred}\n")
    print(f"→ Wrote {REF_TXT} and {PRED_TXT}")

def debug_file(path: Path):
    # 打印所有不含 \t 的行
    bad = []
    for i, row in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if '\t' not in row:
            bad.append((i, row))
    if bad:
        print(f"=== Malformed lines in {path.name} ===")
        for i, row in bad:
            print(f"  Line {i}: {row!r}")
        print(f"=== end of bad lines ({len(bad)} total) ===\n")
    else:
        print(f"No malformed lines in {path.name}.\n")

def clean_file(src: Path, dst: Path):
    # 过滤掉所有不包含 \t 的行
    with open(src,'r',encoding='utf-8') as fin, open(dst,'w',encoding='utf-8') as fout:
        for row in fin:
            if '\t' in row:
                fout.write(row)
    print(f"→ Cleaned file written to {dst}")

def run_evaluator():
    # 4) 调用 evaluator.py
    cmd = f"cat {PRED_CLEAN} | python3 {EVAL_SCRIPT} {REF_CLEAN}"
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 步骤 1-2
    build_ref_and_pred()
    debug_file(REF_TXT)
    debug_file(PRED_TXT)

    # 步骤 3：清洗
    clean_file(REF_TXT, REF_CLEAN)
    clean_file(PRED_TXT, PRED_CLEAN)

    # 步骤 4：评测
    print("→ Running evaluator …")
    run_evaluator()

if __name__ == "__main__":
    main()
