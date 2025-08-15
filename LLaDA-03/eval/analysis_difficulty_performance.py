#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis_compare_models.py

通用化处理任意多组预测数据，按 prompt token 长度分箱后
绘制 Accuracy, Precision, F1, Prediction%, Extract Rate 多模型对比折线图，并输出 CSV。
"""
import json
import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import re

# --------- Evaluation logic from evaluation.py ---------
def normalize_answer(ans):
    if isinstance(ans, str):
        s = re.sub(r'\\text\s*\{|\\frac\s*\{', '', ans)
        s = s.replace('}', '').replace('\\', '').replace('%', '').strip().lower()
        return s
    return ans


def loose_match(pred, gt, tol=1e-2):
    p = normalize_answer(pred)
    g = normalize_answer(gt)
    try:
        return abs(float(p) - float(g)) < tol
    except:
        return p == g


def f1_score(correct, pd_num, gt_num):
    prec = correct / pd_num if pd_num else 0
    rec  = correct / gt_num if gt_num else 0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0


def cal_metric(pds, gts):
    pd_num = len(pds)
    gt_num = len(gts)
    correct = sum(1 for pd in pds if any(loose_match(pd, gt) for gt in gts))
    if not pds and not gts:
        return (0, 0, 0)
    if not pds:
        return (0, 0, gt_num)
    if not gts:
        return (0, pd_num, 0)
    return (correct, pd_num, gt_num)


def label_choose(results):
    best, best_f1 = results[0], f1_score(*results[0])
    for r in results:
        f = f1_score(*r)
        if f > best_f1:
            best, best_f1 = r, f
    return best


def calculate_sample(data):
    pds = data.get('prediction', [])
    raw = data.get('groundtruth', [])
    if raw and isinstance(raw[0], list):
        cands = raw
    else:
        cands = [raw]
    parsed = []
    for g in cands:
        if len(g) == 1 and isinstance(g[0], str) and ',' in g[0]:
            parsed.append([x.strip() for x in g[0].split(',')])
        else:
            parsed.append(g)
    return [cal_metric(pds, g) for g in parsed]
# -------------------------------------------------------
# python3 /storage/v-mengnijia/LLaDA/eval/analysis_difficulty_performance.py --max_data=1000 --max_prompt_length=3584 --trunc_what="long" --pred=/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_vanilla_config4_20250630.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE1.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE2.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask1.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask2.jsonl --label=Vanilla CoRE1 CoRE2 RespMask1 RespMask2

def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple models by length bins")
    p.add_argument('--max_data', type=int, default=None,
                   help='仅处理前 N 条数据')
    p.add_argument('--pred', nargs='+', required=True,
                   help='预测 JSONL 路径列表，例如 --pred p1.jsonl p2.jsonl ...')
    p.add_argument('--label', nargs='+', required=True,
                   help='对应的标签列表，用于图例，例如 --label vanilla RespMask CoRE ...')
    p.add_argument('--data_file', default="/storage/v-mengnijia/LLaDA/data.jsonl",
                   help='原始 data.jsonl，含 prompt')
    p.add_argument('--tokenizer_name', default='GSAI-ML/LLaDA-8B-Instruct',
                   help='Tokenizer 名称或路径')
    p.add_argument('--num_bins', type=int, default=10,
                   help='等宽分箱数，默认10')
    p.add_argument('--output_prefix', type=str, default='compare',
                   help='输出文件名前缀')
    p.add_argument('--max_prompt_length', type=int, default=1e10,
                   help='用于长短截断的阈值')
    p.add_argument('--trunc_what', type=str, default="long",
                   choices=["long", "short"],
                   help='截断哪部分："long" (<阈值) 或 "short" (>阈值)')
    args = p.parse_args()
    if len(args.pred) != len(args.label):
        p.error("--pred 和 --label 的数量必须一致")
    return args


def load_predictions(pred_file, max_data):
    arr = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_data is not None and i >= max_data:
                break
            arr.append(json.loads(line))
    return arr


def load_data(preds, data_file, max_data):
    merged = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_data is not None and i >= max_data:
                break
            entry = json.loads(line)
            merged.append({
                'prompt': entry.get('prompt', ''),
                'prediction': preds[i].get('prediction', []),
                'groundtruth': preds[i].get('groundtruth', [])
            })
    return merged


def compute_metrics_by_bin(all_data, bins, bin_ids):
    stats = {b: [] for b in range(len(bins)-1)}
    for idx, d in enumerate(all_data):
        b = bin_ids[idx]
        if 0 <= b < len(bins)-1:
            stats[b].append(idx)
    recs = []
    for b, idx_list in stats.items():
        if not idx_list:
            recs.append(None)
            continue
        total_c = total_pd = total_gt = extracted = 0
        for i in idx_list:
            d = all_data[i]
            if d['prediction']:
                extracted += 1
            c, pd, gt = label_choose(calculate_sample(d))
            total_c += c; total_pd += pd; total_gt += gt
        acc   = total_c / total_gt if total_gt else 0
        prec  = total_c / total_pd if total_pd else 0
        f1    = f1_score(total_c, total_pd, total_gt)
        predp = total_pd / total_gt if total_gt else 0
        ext_r = extracted / len(idx_list)
        recs.append({'acc': acc, 'prec': prec, 'f1': f1,
                     'predp': predp, 'extr': ext_r})
    return recs


def compute_overall_metrics(data_list):
    total_c = total_pd = total_gt = extracted = 0
    for d in data_list:
        c, pd, gt = label_choose(calculate_sample(d))
        total_c  += c; total_pd += pd; total_gt += gt
        if d['prediction']:
            extracted += 1
    acc   = total_c / total_gt if total_gt else 0
    prec  = total_c / total_pd if total_pd else 0
    f1    = f1_score(total_c, total_pd, total_gt)
    predp = total_pd / total_gt if total_gt else 0
    extr  = extracted / len(data_list) if data_list else 0
    return {'acc': acc, 'prec': prec, 'f1': f1,
            'predp': predp, 'extr': extr}


def main():
    args = parse_args()
    # 加载多组预测和对应数据
    all_preds = [load_predictions(p, args.max_data) for p in args.pred]
    all_data  = [load_data(preds, args.data_file, args.max_data) for preds in all_preds]

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # 计算 prompt 长度（以第一个数据集为准）
    data0   = all_data[0]
    lengths = [len(tok(d['prompt']).input_ids) for d in data0]

    # 长短截断抽样
    if args.trunc_what == 'long':
        idxs = [i for i, L in enumerate(lengths) if L < args.max_prompt_length]
    else:
        idxs = [i for i, L in enumerate(lengths) if L > args.max_prompt_length]
    filtered_data = [[d for i, d in enumerate(dl) if i in idxs] for dl in all_data]

    # 打印整体指标
    print(f"Prompt length {('<' if args.trunc_what=='long' else '>')} {args.max_prompt_length} 时，各模型指标：")
    overall = [compute_overall_metrics(dl) for dl in filtered_data]
    for label, metrics in zip(args.label, overall):
        print(f"{label}: {metrics}")

    # 分箱并计算各模型指标
    bins    = np.linspace(min(lengths), max(lengths), args.num_bins+1)
    bin_ids = np.digitize(lengths, bins) - 1
    recs_list = [compute_metrics_by_bin(dl, bins, bin_ids) for dl in all_data]
    mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    # 构建 DataFrame
    data = {'mid': mids}
    metrics = ['acc','prec','f1','predp','extr']
    for label, recs in zip(args.label, recs_list):
        for m in metrics:
            data[f'{m}_{label}'] = [ (r[m] if r else np.nan) for r in recs ]
    df = pd.DataFrame(data)
    csv_out = f"{args.output_prefix}_metrics_by_length.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved CSV: {csv_out}")

    # 绘制对比曲线
    ylabel_map = {'acc':'Accuracy','prec':'Precision',
                  'f1':'F1-score','predp':'Prediction%','extr':'Extract Rate'}
    for m in metrics:
        plt.figure()
        for label in args.label:
            plt.plot(df['mid'], df[f'{m}_{label}'], marker='o', label=label)
        plt.xlabel('Prompt token length')
        plt.ylabel(ylabel_map[m])
        plt.title(f"{ylabel_map[m]} vs Prompt Length")
        plt.legend()
        plt.tight_layout()
        png_out = f"{args.output_prefix}_{m}_compare.png"
        plt.savefig(png_out)
        print(f"Saved plot: {png_out}")

if __name__ == '__main__':
    main()
