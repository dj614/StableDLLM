#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis_compare_models.py

通用化处理任意多组预测数据，按 prompt token 长度分箱后
绘制 vanilla, CoRE 平均值, RespMask 平均值 三曲线和输出 CSV。
"""

# python3 /storage/v-mengnijia/LLaDA/eval/analysis_difficulty_performance.py --max_prompt_length 8192 --trunc_what long --pred /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_vanilla_config4_20250630.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE1-newICT.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE2-newICT.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE3-newICT.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE4-newICT.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask1.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask2.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask3.jsonl /storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask4.jsonl --label Vanilla CoRE1 CoRE2 CoRE3 CoRE4 RespMask1 RespMask2 RespMask3 RespMask4 --do_plot

import json
import argparse
import numpy as np
import pandas as pd
import pandas as pandas_mod
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import re

# --------- Evaluation logic (unchanged) ---------
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
    pd_count = len(pds)
    gt_count = len(gts)
    correct = sum(1 for pd in pds if any(loose_match(pd, gt) for gt in gts))
    if not pds and not gts:
        return (0, 0, 0)
    if not pds:
        return (0, 0, gt_count)
    if not gts:
        return (0, pd_count, 0)
    return (correct, pd_count, gt_count)

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
# ------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple models by length bins and plot averages")
    p.add_argument('--do_plot', action='store_true', help='是否绘制图表')
    p.add_argument('--max_data', type=int, default=None, help='仅处理前 N 条数据')
    p.add_argument('--pred', nargs='+', required=True, help='预测 JSONL 路径列表')
    p.add_argument('--label', nargs='+', required=True, help='对应的标签列表')
    p.add_argument('--data_file', default="/storage/v-mengnijia/LLaDA/data.jsonl", help='原始 data.jsonl，含 prompt')
    p.add_argument('--tokenizer_name', default='GSAI-ML/LLaDA-8B-Instruct', help='Tokenizer 名称或路径')
    p.add_argument('--num_bins', type=int, default=20, help='等宽分箱数，默认10')
    p.add_argument('--output_prefix', type=str, default='compare', help='输出文件名前缀')
    p.add_argument('--max_prompt_length', type=int, default=1e10, help='用于截断的 prompt 长度阈值')
    p.add_argument('--trunc_what', type=str, default="long", choices=["long","short"], help='截断哪部分')
    args = p.parse_args()
    if len(args.pred) != len(args.label):
        p.error("--pred 和 --label 的数量必须一致")
    return args


def load_predictions(pred_file, max_data):
    arr = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_data is not None and i >= max_data: break
            arr.append(json.loads(line))
    return arr


def load_data(preds, data_file, max_data):
    merged = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_data is not None and i >= max_data: break
            if i >= len(preds): break
            entry = json.loads(line)
            merged.append({'prompt': entry.get('prompt',''),
                           'prediction': preds[i].get('prediction',[]),
                           'groundtruth': preds[i].get('groundtruth',[])})
    return merged


def compute_overall_metrics(data_list):
    total_c = total_pd = total_gt = extracted = 0
    for d in data_list:
        c, pd_num, gt_num = label_choose(calculate_sample(d))
        total_c += c; total_pd += pd_num; total_gt += gt_num
        if d['prediction']: extracted += 1
    acc = total_c/total_gt if total_gt else 0
    prec= total_c/total_pd if total_pd else 0
    f1= f1_score(total_c, total_pd, total_gt)
    predp= total_pd/total_gt if total_gt else 0
    extr= extracted/len(data_list) if data_list else 0
    return {'acc':acc,'prec':prec,'f1':f1,'predp':predp,'extr':extr}


def main():
    args = parse_args()
    all_preds = [load_predictions(p, args.max_data) for p in args.pred]
    all_data  = [load_data(preds, args.data_file, args.max_data) for preds in all_preds]

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    data0 = all_data[0]
    lengths = [len(tok(d['prompt']).input_ids) for d in data0]
    if args.trunc_what == 'long':
        idxs = [i for i,L in enumerate(lengths) if L < args.max_prompt_length]
    else:
        idxs = [i for i,L in enumerate(lengths) if L > args.max_prompt_length]
    filtered = [[d for i,d in enumerate(dl) if i in idxs] for dl in all_data]

    # 输出整体指标
    print(f"Prompt length {('<' if args.trunc_what=='long' else '>')} {args.max_prompt_length} 时，各模型指标：")
    overall = [compute_overall_metrics(dl) for dl in filtered]
    for label,metrics in zip(args.label,overall):
        print(f"{label}: {metrics}")

    # 计算并打印 CoRE 和 RespMask 平均值
    core_metrics = [m for l, m in zip(args.label, overall) if l.lower().startswith('core')]
    if core_metrics:
        avg_core = {k: np.mean([cm[k] for cm in core_metrics]) for k in core_metrics[0]}
        print(f"CoRE_avg: {avg_core}")
    resp_metrics = [m for l, m in zip(args.label, overall) if l.lower().startswith('respmask')]
    if resp_metrics:
        avg_resp = {k: np.mean([rm[k] for rm in resp_metrics]) for k in resp_metrics[0]}
        print(f"RespMask_avg: {avg_resp}")
    
    # 分箱并保存 CSV
    bins = np.linspace(min(lengths), max(lengths), args.num_bins+1)
    bin_ids = np.digitize(lengths, bins)-1
    recs_all = []
    for dl in all_data:
        stats = {b:[] for b in range(len(bins)-1)}
        for idx,d in enumerate(dl):
            b = bin_ids[idx]
            if 0<=b<len(bins)-1: stats[b].append(idx)
        recs = []
        for idx_list in stats.values():
            if not idx_list:
                recs.append(None)
                continue
            totc=totpd=totgt=extr=0
            for i in idx_list:
                d = dl[i]
                if d['prediction']: extr += 1
                c, pd_num, gt_num = label_choose(calculate_sample(d))
                totc += c; totpd += pd_num; totgt += gt_num
            f1v = f1_score(totc, totpd, totgt)
            recs.append({'acc': totc/totgt if totgt else 0,
                         'prec': totc/totpd if totpd else 0,
                         'f1':   f1v,
                         'predp': totpd/totgt if totgt else 0,
                         'extr': extr/len(idx_list)})
        recs_all.append(recs)

    mids = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    data_dict = {'mid': mids}
    metrics = ['acc','prec','f1','predp','extr']
    for label,recs in zip(args.label,recs_all):
        for m in metrics:
            data_dict[f'{m}_{label}'] = [(r[m] if r else np.nan) for r in recs]

    df = pandas_mod.DataFrame(data_dict)
    df.to_csv(f"{args.output_prefix}_metrics_by_length.csv", index=False)
    print(f"Saved CSV: {args.output_prefix}_metrics_by_length.csv")

    if args.do_plot:
        ylabel_map = {'acc':'Accuracy','prec':'Precision','f1':'F1-score','predp':'Prediction%','extr':'Extract Rate'}
        vanilla_label = args.label[0]
        core_labels = [l for l in args.label if l.lower().startswith('core')]
        resp_labels = [l for l in args.label if l.lower().startswith('respmask')]
        for m in metrics:
            plt.figure()
            plt.plot(df['mid'], df[f'{m}_{vanilla_label}'], marker='o', label=vanilla_label)
            core_cols = [f'{m}_{l}' for l in core_labels]
            plt.plot(df['mid'], df[core_cols].mean(axis=1), marker='s', label='CoRE_avg')
            resp_cols = [f'{m}_{l}' for l in resp_labels]
            plt.plot(df['mid'], df[resp_cols].mean(axis=1), marker='^', label='RespMask_avg')
            plt.xlabel('Prompt token length')
            plt.ylabel(ylabel_map[m])
            plt.title(f"{ylabel_map[m]} vs Prompt Length")
            plt.legend()
            plt.tight_layout()
            out = f"{args.output_prefix}_{m}_compare_avg.png"
            plt.savefig(out)
            print(f"Saved plot: {out}")

if __name__ == '__main__':
    main()
