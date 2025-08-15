#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grad_snr_check_llada.py  (v5, key‑name fixed)

功能：
  • 统计 baseline ↔ MirrorMask 在 response 区域的梯度均值 / 方差 / SNR
  • 支持多个 common id （--common_ids）
  • 支持 rare 子 token 合并或拆开 (--rare_merge / --no-rare_merge)
  • 兼容 LLaDA：手动 logits→CE→backward

用法示例：
python grad_snr_check_llada.py \
  --data /storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl \
  --steps 100 --bs 1 \
  --common_ids "268 341 301 296 297 352 300 468 3742 259"
"""

import argparse, json, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

MASK_TOKEN_ID = 126336  # same as training

# ========= CLI =========
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--seq_len", type=int, default=8192)

ap.add_argument("--rare_toks", type=str, default="\\,boxed,{")
ap.add_argument("--rare_ids", type=int, nargs='*', default=None)
ap.add_argument("--rare_merge", action="store_true", default=True)
ap.add_argument("--no-rare_merge", dest="rare_merge", action="store_false")

ap.add_argument("--common_ids", type=str, default=".")
args = ap.parse_args()

# ========= Load tokenizer / model =========
device = "cuda:1"
tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device)
model.train()
if getattr(model.config, "is_decoder", None):
    model.config.is_decoder = False  # ensure loss computation allowed

# ========= helper: parse token spec =========
def parse_ids(spec: str):
    parts = [p for seg in spec.split(",") for p in seg.split()]
    ids = []
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            tid = tok.convert_tokens_to_ids(p)
            if tid == tok.unk_token_id:
                raise ValueError(f"✗ token «{p}» not found in vocab")
            ids.append(tid)
    return ids

RARE_IDS = args.rare_ids if args.rare_ids else parse_ids(args.rare_toks)
COMMON_IDS = parse_ids(args.common_ids)
if not COMMON_IDS:
    raise ValueError("common_ids list is empty")

# ========= dataset =========
class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_len):
        self.samples = []
        for ln in open(path, encoding="utf-8"):
            ex = json.loads(ln)
            if len(ex["input_ids"]) <= max_len:
                ids = torch.tensor(ex["input_ids"], dtype=torch.long)
                pl  = ex.get("prompt_length", 0)
                self.samples.append((ids, pl))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    ids, pls = zip(*batch)
    pad = tok.pad_token_id or tok.eos_token_id
    ids_pad = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
    return ids_pad, torch.tensor(pls, dtype=torch.long)

ds = JsonlDataset(args.data, args.seq_len)
dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, collate_fn=collate_fn)

# ========= stats =========
def mk(): return {"n":0,"m":0.,"s":0.}
def upd(st,x):
    st["n"]+=1; d=x-st["m"]; st["m"]+=d/st["n"]; st["s"]+=d*(x-st["m"])
def fin(st):
    var = st["s"]/(st["n"]-1) if st["n"]>1 else 0.
    return st["m"], math.sqrt(var)

names_common = [f"common_id{id}" for id in COMMON_IDS]
names_rare   = ["rare"] if args.rare_merge else [f"rare_id{id}" for id in RARE_IDS]
stats = {f"{phase}_{name}": mk()
         for phase in ("bl","mm")
         for name  in (["common_agg"]+names_common+names_rare)}

# ========= helpers =========
def bernoulli(shape,p,device):
    return (torch.rand(shape,device=device)<p).long()

def mask_resp(ids, plens, raw_mask):
    seq = torch.arange(ids.size(1), device=ids.device)[None,:]
    eligible = seq >= plens[:,None]
    final = (raw_mask & eligible).bool()
    noisy = ids.clone()
    noisy[final] = MASK_TOKEN_ID
    return noisy, final

def grad_step(noisy, final_mask, gold_ids):
    if not final_mask.any():
        model.zero_grad(set_to_none=True)
        zero_common = {cid:0. for cid in COMMON_IDS}
        if args.rare_merge:
            zero_rare = {"rare":0.}
        else:
            zero_rare = {f"rare_id{rid}":0. for rid in RARE_IDS}
        return 0., zero_common, zero_rare

    logits = model(noisy).logits
    loss = F.cross_entropy(logits[final_mask], gold_ids[final_mask])
    loss.backward()

    G = model.get_input_embeddings().weight.grad.detach()
    common_each = {cid: G[cid].norm().item() for cid in COMMON_IDS}
    common_agg = math.sqrt(sum(v*v for v in common_each.values()))
    if args.rare_merge:
        rare_each = {"rare": math.sqrt(G[RARE_IDS].pow(2).sum().item())}
    else:
        rare_each = {f"rare_id{rid}": G[rid].norm().item() for rid in RARE_IDS}

    model.zero_grad(set_to_none=True)
    return common_agg, common_each, rare_each

# ========= main loop =========
for step, (ids, plens) in enumerate(dl):
    if step >= args.steps: break
    ids, plens = ids.to(device), plens.to(device)
    t = random.random()

    # ----- baseline -----
    m0 = bernoulli(ids.shape, t, device)
    noisy0, fm0 = mask_resp(ids, plens, m0)
    ca, ce, re = grad_step(noisy0, fm0, ids)
    upd(stats["bl_common_agg"], ca)
    for cid, v in ce.items():
        upd(stats[f"bl_common_id{cid}"], v)
    for k, v in re.items():
        upd(stats[f"bl_{k}"], v)

    # ----- MirrorMask -----
    U = torch.rand_like(ids.float())
    m1 = (U < t).long()
    m2 = (U > 1 - t).long()
    # branch 1
    noisy1, fm1 = mask_resp(ids, plens, m1)
    ca1, ce1, re1 = grad_step(noisy1, fm1, ids)
    # branch 2
    noisy2, fm2 = mask_resp(ids, plens, m2)
    ca2, ce2, re2 = grad_step(noisy2, fm2, ids)
    upd(stats["mm_common_agg"], 0.5*(ca1+ca2))
    for cid in COMMON_IDS:
        upd(stats[f"mm_common_id{cid}"], 0.5*(ce1[cid]+ce2[cid]))
    if args.rare_merge:
        upd(stats["mm_rare"], 0.5*(re1["rare"]+re2["rare"]))
    else:
        for rid in RARE_IDS:
            key=f"rare_id{rid}"
            upd(stats[f"mm_{key}"], 0.5*(re1[key]+re2[key]))

# ===== 统计结果汇总到行式 DataFrame =====
import pandas as pd

def id2tok(i):
    txt = tok.convert_ids_to_tokens(i, skip_special_tokens=False)
    return txt.lstrip("Ġ▁") if txt else "<unk>"

rows = []
for phase in ("bl", "mm"):                           # baseline / MirrorMask
    for key, st in stats.items():
        if not key.startswith(f"{phase}_"):
            continue
        mean, sd = fin(st)
        snr = mean / (sd + 1e-12)
        token_key = key[len(phase)+1:]               # 去掉 "bl_" 或 "mm_"
        # 把 id → 可读 token
        if "_id" in token_key:
            tok_id = int(token_key.split("id")[1])
            token_disp = f"{tok_id}({id2tok(tok_id)})"
        else:
            token_disp = token_key
        rows.append({
            "token":  token_disp,
            "method": "baseline" if phase=="bl" else "mirrormask",
            "mean":   mean,
            "std":    sd,
            "snr":    snr
        })

df_long = pd.DataFrame(rows)
# ===== pivot 成多级列，更直观对比 =====
df_pivot = (
    df_long
    .pivot_table(index="token", columns="method", values=["mean","std","snr"])
    .sort_index()
)

print("\n== Pivot 形式 ==")
print(df_pivot)