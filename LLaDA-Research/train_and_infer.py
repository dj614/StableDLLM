#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_and_infer.py

1) 先执行训练，输出目录由 --output_dir 决定（默认会根据参数自动生成）。
2) 训练结束后，自动找到最后一个 epoch 的 checkpoint（即 checkpoint-epoch{args.epochs}）。
3) 接着加载该 checkpoint，执行推理并将结果写入 JSONL 文件。
"""
import csv, os
import argparse, json, math, time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import wandb
import random, numpy as np
import gc, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm.auto import tqdm

from MMaDA.models import MMadaModelLM
import sys
sys.path.append("/storage/v-mengnijia/dkv_cache")
from dkv_cache.models.modeling_llada_dkv_cache_decode import LLaDAModelLM
from dkv_cache.generation_utils.llada_dkv_cache_decode import generate
from is_coord_token import is_coord_token

MASK_TOKEN_ID = 126336

def _finite(x, name):
    if not torch.isfinite(x).all():
        # 打印一次就够，避免刷屏
        bad = (~torch.isfinite(x)).float().mean().item()
        print(f"[NaNGuard] {name}: non-finite ratio={bad:.6f}")
        return torch.nan_to_num(x, nan=0.0, posinf=args.loss_max, neginf=0.0)
    return x

def clip_loss(x: torch.Tensor, max_val: float = None):
    """若设置了 loss_max，则逐样本裁剪到 [0, max_val]。"""
    if max_val is None:
        return x
    return torch.clamp(x, max=max_val)

def set_random_seed(seed: int, rank: int = 0):
    """
    Seed all relevant RNGs for reproducibility, with an optional per-process offset.
    """
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False
    torch.use_deterministic_algorithms(True)

def make_output_dir_and_broadcast(args, accelerator, gb):
    # output_dir default
    is_flag = "IS" if args.IS else "noIS"
    if accelerator.is_main_process and args.output_dir is None:
        tag = datetime.now().strftime("%y%m%d_%H%M%S")
        # mask tag
        if args.mask_mode in ("CoRE", "mixed"): maskM = f"{args.mask_mode}_{args.m}"
        else: maskM = args.mask_mode
        # train mode + importance sampling tag
        if args.train_mode == "Normal":
            trainM_base = args.train_mode
        elif args.train_mode == "MultiSample":
            trainM_base = f"{args.train_mode}_ns{args.num_samples}"
        elif args.train_mode == "MirrorMask":
            trainM_base = f"{args.train_mode}_dedup{args.mm_dedup}"
        trainM = f"{trainM_base}_{is_flag}"
        # mask ratio mode
        if args.mask_ratio_mode == "random": mask_ratio_mode = "random"
        else: mask_ratio_mode = f"stratified_{args.mask_strata_bins}"
        # EMA
        if args.EMA:
            emaM = f"EMA_bins{args.num_bins}_blr{args.baseline_lr}"
        else: emaM = None
        # assemble
        args.output_dir = (
            f"/storage/result/checkpoints/LLaDA/"
            f"seed{args.seed}_{args.model}_{args.task}_{trainM}_{maskM}_{emaM}_{mask_ratio_mode}_"
            f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb}_"
            f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"
            f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"
        )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def init_ema_control_variate(num_bins: int, device: torch.device = None, dtype=torch.float32, baseline_init: float = 0.0):
    """
    Initialize EMA control variate structures:
    - baseline: Tensor of shape (num_bins,) initialized to baseline_init
    - stats: dict with EMA stats for optimal c calculation
    - stats_lr: default to None (will fallback to baseline_lr)
    """
    device = device or torch.device('cpu')
    baseline = torch.full((num_bins,), baseline_init, device=device, dtype=dtype)
    stats = {
        'mu_L': torch.zeros(num_bins, device=device, dtype=dtype),
        'mu_H': torch.zeros(num_bins, device=device, dtype=dtype),
        'M_LH': torch.zeros(num_bins, device=device, dtype=dtype),
        'M_HH': torch.zeros(num_bins, device=device, dtype=dtype),
    }
    return baseline, stats

def ema_control_variate_optimal_c(
    sample_losses: torch.Tensor,
    baseline: torch.Tensor,
    bins: torch.Tensor,
    baseline_lr: float,
    stats: dict,
    args: object,
    stats_lr: float = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply EMA control variate with optional online optimal c per bin:
      - If args.EMA_Optimized is True, estimate c via EMA stats; otherwise use c=1
      - Subtract c * baseline for each sample
      - Update baseline via EMA
      - Update running stats (mu_L, mu_H, M_LH, M_HH) via EMA only if optimized
    """
    if stats_lr is None:
        stats_lr = baseline_lr
    # Retrieve per-sample values
    baseline_vals = baseline[bins]            # H_i
    L = sample_losses
    # Default coefficient c=1
    c_per_bin = torch.ones_like(baseline_vals)
    if getattr(args, 'EMA_Optimized', False):
        # Online update of stats per bin
        mu_L = stats['mu_L']
        mu_H = stats['mu_H']
        M_LH = stats['M_LH']
        M_HH = stats['M_HH']
        # Gather current stats for affected bins
        cur_mu_L = mu_L[bins]
        cur_mu_H = mu_H[bins]
        cur_M_LH = M_LH[bins]
        cur_M_HH = M_HH[bins]
        with torch.no_grad():
            new_mu_L = mu_L.clone()
            new_mu_H = mu_H.clone()
            new_M_LH = M_LH.clone()
            new_M_HH = M_HH.clone()
            # EMA updates
            new_mu_L[bins] = (1 - stats_lr) * cur_mu_L + stats_lr * L
            new_mu_H[bins] = (1 - stats_lr) * cur_mu_H + stats_lr * baseline_vals
            new_M_LH[bins] = (1 - stats_lr) * cur_M_LH + stats_lr * (L * baseline_vals)
            new_M_HH[bins] = (1 - stats_lr) * cur_M_HH + stats_lr * (baseline_vals * baseline_vals)
            stats['mu_L'] = new_mu_L
            stats['mu_H'] = new_mu_H
            stats['M_LH'] = new_M_LH
            stats['M_HH'] = new_M_HH
        # Estimate optimal c
        cov = stats['M_LH'] - stats['mu_L'] * stats['mu_H']
        var = stats['M_HH'] - stats['mu_H'].pow(2)
        c_all = cov / (var + eps)
        c_per_bin = c_all[bins]
    # Adjust losses
    adjusted = L - c_per_bin * baseline_vals
    # Update baseline via EMA (always)
    with torch.no_grad():
        baseline[bins] = (1 - baseline_lr) * baseline_vals + baseline_lr * L.detach()
    return adjusted.mean()

class LLaDADataset(Dataset):
    def __init__(self, jsonl_path: str, max_len: int):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf8") as f:
            for ln in f:
                ex = json.loads(ln)
                if len(ex["input_ids"]) <= max_len:
                    self.samples.append(ex)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ex = self.samples[idx]
        return {
            "input_ids":     torch.tensor(ex["input_ids"], dtype=torch.long),
            "prompt_length": ex["prompt_length"]
        }

def collate_fn(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids, prompt_lens = [], []
    for x in batch:
        ids = x["input_ids"]
        pad = torch.full((max_len - ids.size(0),), pad_id, dtype=torch.long)
        input_ids.append(torch.cat([ids, pad], dim=0))
        prompt_lens.append(x["prompt_length"])
    return {
        "input_ids":       torch.stack(input_ids),
        "prompt_lengths":  torch.tensor(prompt_lens, dtype=torch.long)
    }

def forward_process(batch_ids: torch.Tensor,
                    prompt_lens: torch.Tensor,
                    mask_mode: str,
                    coord_format: str,
                    tokenizer,
                    m: float,
                    eps: float = 1e-3,
                    fixed_t: torch.Tensor = None,
                    use_IS: bool = False,
                    rare_ids: list = None,
                    delta: float = 0.2):
    """
    Generate noised inputs and mask probabilities, supporting optional importance sampling.
    Returns:
      noisy: Tensor of shape [B, L]
      p_mask_used: Tensor of mask probabilities used for dividing losses
      eligible: Tensor mask of eligible positions
      t: Tensor of noise levels
    """
    B, L = batch_ids.shape
    device = batch_ids.device
    # Sample noise levels
    t = fixed_t if fixed_t is not None else torch.rand(B, device=device)
    # Original mask probability per position
    p_mask = ((1 - eps) * t[:, None] + eps).repeat(1, L)
    # Determine eligible positions
    seq = torch.arange(L, device=device)[None, :]
    resp_pos = seq >= prompt_lens[:, None]
    coord_pos = torch.zeros_like(resp_pos)
    if mask_mode == "CoRE":
        for b in range(B):
            toks = tokenizer.convert_ids_to_tokens(batch_ids[b, :prompt_lens[b]].tolist())
            mask_in = torch.tensor([is_coord_token(tok, coord_format) for tok in toks], device=device)
            coord_pos[b, :prompt_lens[b]] = mask_in
        eligible = resp_pos | coord_pos
        non_coord = (seq < prompt_lens[:, None]) & ~coord_pos
        extra = (torch.rand_like(p_mask) < m) & non_coord
        eligible |= extra
    elif mask_mode == "mixed":
        non_resp = seq < prompt_lens[:, None]
        extra = (torch.rand_like(p_mask) < m) & non_resp
        eligible = resp_pos | extra
    else:
        eligible = resp_pos
    # Importance Sampling: adjust mask probability
    if use_IS and rare_ids is not None and len(rare_ids) > 0:
        # Create tensor for rare_ids matching
        rare_tensor = torch.tensor(rare_ids, device=device)
        # Identify positions of rare tokens
        rare_pos = (batch_ids.unsqueeze(-1) == rare_tensor.view(1, 1, -1)).any(dim=-1)
        # Increase probability for rare positions
        q_mask = p_mask.clone()
        q_mask[rare_pos] = torch.minimum(q_mask[rare_pos] + delta, torch.ones_like(q_mask[rare_pos]))
        used_p = q_mask
    else:
        used_p = p_mask
    # Generate noisy inputs
    noisy = batch_ids.clone()
    mask_here = (torch.rand_like(p_mask) < used_p) & eligible
    noisy[mask_here] = MASK_TOKEN_ID
    # Ensure positions not eligible have probability 1 (no division by zero later)
    used_p = used_p.masked_fill(~eligible, 1.0)
    is_weight = p_mask.div(used_p)
    return noisy, used_p, eligible, t, is_weight

def train(args):
    # create accelerator with a global seed
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.grad_accum
    )
    rank = accelerator.process_index
    set_random_seed(args.seed, rank)

    # define device and gb
    device = accelerator.device
    gb = args.batch_size * args.grad_accum * accelerator.num_processes

    # prepare stratified sampler buffer
    assert args.mask_ratio_mode in ("random", "stratified")
    if args.mask_ratio_mode == "stratified":
        if args.mask_strata_bins is None:
            num_strata = int(math.sqrt(gb))
        else:
            num_strata = args.mask_strata_bins
        base = gb // num_strata
        rem  = gb - base * num_strata
        mask_t_buffer = None
        mask_t_idx = 0
    else:
        num_strata = mask_t_buffer = mask_t_idx = None

    # initialize ema stats
    baseline, stats = init_ema_control_variate(args.num_bins, device=device, dtype=torch.float32)

    # tokenizer
    if args.model == "llada":
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", padding_side="right", use_fast=True, trust_remote_code=True)
    elif args.model == "mmada":
        tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", padding_side="right", use_fast=True, trust_remote_code=True)

    # dataloader
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    g_data = torch.Generator().manual_seed(args.seed + rank)
    ds = LLaDADataset(args.train_data_path, args.max_len)
    train_n = int(len(ds)*args.train_ratio)
    eval_n  = len(ds) - train_n
    train_ds, eval_ds = torch.utils.data.random_split(ds, [train_n, eval_n], generator=g_data)
    if eval_n == 0:
        if accelerator.is_main_process:
            print("⚠️ args.train_ratio == 1.0；跳过 eval")
        do_evaluation = False
    else:
        do_evaluation = True
    if accelerator.num_processes > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=args.seed
        )
    else:
        train_sampler = RandomSampler(
            train_ds,
            generator=torch.Generator()
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id)
    )
    eval_loader  = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, pad_id),
        generator=g_data
    )

    # model
    if args.model == "llada":
        model = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)
    elif args.model == "mmada":
        model = MMadaModelLM.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", trust_remote_code=True, torch_dtype=torch.bfloat16)
    if getattr(model.config, "is_decoder", None):
        model.config.is_decoder = False
    
    # optimizer 
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr, weight_decay=0.1)

    # prepare via accelerator
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    # scheduler
    total_steps = math.ceil(len(train_loader)*args.epochs/args.grad_accum)
    if args.lr_scheduler_type=="warmup_hold_decay":
        hold = total_steps - int(args.decay_ratio*total_steps) - args.warmup_steps
        hold = max(hold,0)
        dec_steps = int(args.decay_ratio*total_steps)
        final_factor = args.final_lr/args.lr
        scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=args.warmup_steps),
                ConstantLR(optimizer, factor=1.0, total_iters=hold),
                LinearLR(optimizer, start_factor=1.0, end_factor=final_factor, total_iters=dec_steps)
            ],
            milestones=[args.warmup_steps, args.warmup_steps+hold]
        )
    else:
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    scheduler = accelerator.prepare(scheduler)

    # wandb init
    if accelerator.is_main_process:
        wandb.init(project=f"{args.model}_sft_{args.task}", config=vars(args))
        code_path = os.path.join(os.path.dirname(__file__), "is_coord_token.py")
        wandb.save(code_path)
        print(f"◎ 全局 batch = {args.batch_size} × grad_accum {args.grad_accum} × processes {accelerator.num_processes} = {gb}")
        print(f"★ 总数据量 {len(ds)}，训练 {train_n}，评估 {eval_n}，步骤 {total_steps}")
    
    # define logging_steps
    if args.logging_steps is None:
        args.logging_steps = max(1, int((len(ds) * args.epochs) / args.batch_size / 100))
    
    def evaluate(step_idx: int):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, device=device, dtype=torch.long)
        pbar = tqdm(eval_loader, desc=f"Eval@{step_idx}", disable=not accelerator.is_main_process)
        with torch.no_grad():
            for batch in pbar:
                eids = batch["input_ids"].to(device)
                eplen = batch["prompt_lengths"].to(device)
                noisy, p_mask, eligible, _, is_weight = forward_process(
                    eids, eplen, mask_mode=args.mask_mode,
                    coord_format=args.coord_format, tokenizer=tokenizer, m=args.m
                )
                seq_idx = torch.arange(eids.shape[1], device=device)[None, :]
                mask_prompt = seq_idx < eplen[:, None]
                noisy[mask_prompt] = eids[mask_prompt]
                logits = model(noisy).logits
                mask_tok = noisy == MASK_TOKEN_ID
                if not mask_tok.any(): continue
                iw = is_weight.masked_fill(~eligible, 1.0)
                ce = F.cross_entropy(logits[mask_tok], eids[mask_tok], reduction="none") / p_mask[mask_tok]  * iw[mask_tok]
                loss_batch = torch.zeros(eids.size(0), device=device)
                loss_batch.scatter_add_(0, mask_tok.nonzero(as_tuple=True)[0], ce)
                denom = eligible.sum(dim=1).clamp(min=1)
                total_loss += (loss_batch/denom).sum()
                total_tokens += eids.size(0)
                if accelerator.is_main_process:
                    pbar.set_postfix(loss=(total_loss/total_tokens).item())
        total_loss = accelerator.reduce(total_loss, "sum")
        total_tokens = accelerator.reduce(total_tokens, "sum")
        if accelerator.is_main_process:
            wandb.log({"eval/loss": (total_loss/total_tokens).item()}, step=step_idx)
        model.train()
        accelerator.wait_for_everyone()

    # register grad norm hooks for common / rare ids
    if args.compare_tok_grads:
        common_ids = args.common_ids
        rare_ids   = args.rare_ids
        common_grad_sum = [0.0] * len(common_ids)
        rare_grad_sum   = [0.0] * len(rare_ids)
        step_counter    = [0]
        emb_weight = model.get_input_embeddings().weight
        def emb_weight_grad_hook(grad):
            step_counter[0] += 1
            with torch.no_grad():
                for i, tid in enumerate(common_ids):
                    common_grad_sum[i] += grad[tid].abs().sum().item()
                for i, tid in enumerate(rare_ids):
                    rare_grad_sum[i]   += grad[tid].abs().sum().item()
        if emb_weight.requires_grad:
            hook_handle = emb_weight.register_hook(emb_weight_grad_hook)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "Embedding 权重 requires_grad=False，已跳过注册 compare_tok_grads 钩子。"
            )
            hook_handle = None

    # initialize before training
    model.train()
    if args.hetero_t_in_l and accelerator.is_main_process:
        hetero_ts: list = []
        hetero_ls: list = []
    if do_evaluation: evaluate(0)
    update_step = 0
    start = time.time()

    # train loop
    for epoch in range(args.epochs):
        # shuffle within epochs
        sampler = train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        elif hasattr(sampler, "generator"):
            sampler.generator.manual_seed(args.seed + epoch)
        # tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        for step_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            plen = batch["prompt_lengths"].to(device)
            # get t for current micro-batch
            if args.mask_ratio_mode == "stratified":
                if mask_t_idx == 0:
                    g = torch.Generator(device=device)
                    g.manual_seed(args.seed + update_step + rank)
                    extra = torch.randperm(num_strata, generator=g, device=device)[:rem].tolist()
                    parts = []
                    for i in range(num_strata):
                        cnt = base + (1 if i in extra else 0)
                        low, high = i/num_strata, (i+1)/num_strata
                        parts.append(
                            torch.rand(cnt, device=device, generator=g) * (high - low)
                            + low
                        )
                    mask_t_buffer = torch.cat(parts, dim=0)
                    mask_t_buffer = mask_t_buffer[torch.randperm(gb, generator=g, device=device)]
                fixed_t = mask_t_buffer[mask_t_idx : mask_t_idx + args.batch_size]
                mask_t_idx += args.batch_size
                if mask_t_idx >= gb:
                    mask_t_idx = 0
            else:
                fixed_t = None
            # ---------------------------------
            # ---------- MirrorMask -----------
            # ---------------------------------
            if args.train_mode == "MirrorMask":
                s1 = torch.zeros(ids.size(0), device=device)
                s2 = torch.zeros(ids.size(0), device=device)
                # 1) 先生成 p_mask, eligible, t
                _, p_mask, eligible, t, is_weight = forward_process(
                    ids, plen, args.mask_mode, args.coord_format,
                    tokenizer, args.m, fixed_t=fixed_t,
                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                )
                if not eligible.any(): continue
                iw = is_weight.masked_fill(~eligible, 1.0)
                # 2) 同一组 U 决定两侧 mask
                U = torch.rand_like(p_mask)
                mask1 = (U < p_mask) & eligible
                mask2 = (U > (1 - p_mask)) & eligible
                # ---------- [DEDUP] 预备：并集概率与交集/差集 ----------
                if args.mm_dedup:
                    p_union = (2.0 * p_mask).clamp(max=1.0)
                    p_union = p_union.masked_fill(~eligible, 1.0)
                    both    = mask1 & mask2
                    only1   = mask1 & (~mask2)
                    only2   = mask2 & (~mask1)
                # —— 2 次前向 —— #
                noisy1 = ids.clone()
                noisy1[mask1] = MASK_TOKEN_ID
                noisy1[:, :plen.max()] = ids[:, :plen.max()]
                noisy2 = ids.clone()
                noisy2[mask2] = MASK_TOKEN_ID
                noisy2[:, :plen.max()] = ids[:, :plen.max()]
                logits12 = model(torch.cat([noisy1, noisy2], dim=0)).logits
                B = ids.size(0)
                logits1, logits2 = logits12[:B], logits12[B:]
                loss_b1 = torch.zeros(B, device=device, dtype=torch.float32)
                loss_b2 = torch.zeros(B, device=device, dtype=torch.float32)
                s1 = torch.zeros(B, device=device, dtype=torch.float32)
                s2 = torch.zeros(B, device=device, dtype=torch.float32)
                # 第一次反向
                mask_tok1 = noisy1 == MASK_TOKEN_ID
                if mask_tok1.any():
                    ce1_base = F.cross_entropy(
                        logits1[mask_tok1], ids[mask_tok1], reduction='none'
                    )
                    if args.mm_dedup:
                        # --- 统一 dtype 到 ce1_base.dtype ---
                        dtype = ce1_base.dtype
                        p_union_f = p_union.to(dtype).detach()
                        iw_f      = iw.to(dtype).detach()
                        # 构造权重（float32）
                        w1_full = torch.zeros_like(p_mask, dtype=dtype)
                        w1_full[only1] = 2.0 / p_union_f[only1]
                        w1_full[both]  = 1.0 / p_union_f[both]
                        w1 = w1_full[mask_tok1] * iw_f[mask_tok1]
                        ce1 = ce1_base * w1
                    else:
                        # 非去重分支也统一 dtype，避免 p_mask 为 bf16
                        denom = p_mask[mask_tok1].to(ce1_base.dtype)
                        ce1 = (ce1_base / denom) * iw[mask_tok1].to(ce1_base.dtype)
                    ce1 = _finite(ce1, "ce1")
                    loss_b1 = torch.zeros(ids.size(0), device=device, dtype=ce1.dtype)
                    loss_b1.scatter_add_(0, mask_tok1.nonzero(as_tuple=True)[0], ce1)
                    loss_b1 = _finite(loss_b1, "loss_b1")
                    s1 = clip_loss(
                        loss_b1 / eligible.sum(dim=1).clamp(min=1),
                        args.loss_max
                    )
                # —— 第二次反向 —— #
                mask_tok2 = noisy2 == MASK_TOKEN_ID
                if mask_tok2.any():
                    ce2_base = F.cross_entropy(
                        logits2[mask_tok2], ids[mask_tok2], reduction='none'
                    )
                    if args.mm_dedup:
                        dtype = ce2_base.dtype
                        p_union_f = p_union.to(dtype).detach()
                        iw_f      = iw.to(dtype).detach()
                        w2_full = torch.zeros_like(p_mask, dtype=dtype)
                        w2_full[only2] = 2.0 / p_union_f[only2]
                        w2_full[both]  = 1.0 / p_union_f[both]
                        w2 = w2_full[mask_tok2] * iw_f[mask_tok2]
                        ce2 = ce2_base * w2
                    else:
                        denom = p_mask[mask_tok2].to(ce2_base.dtype)
                        ce2 = (ce2_base / denom) * iw[mask_tok2].to(ce2_base.dtype)
                    ce2 = _finite(ce2, "ce2")
                    loss_b2 = torch.zeros(ids.size(0), device=device, dtype=ce2.dtype)
                    loss_b2.scatter_add_(0, mask_tok2.nonzero(as_tuple=True)[0], ce2)
                    loss_b2 = _finite(loss_b2, "loss_b2")
                    s2 = clip_loss(
                        loss_b2 / eligible.sum(dim=1).clamp(min=1),
                        args.loss_max
                    )
                if (mask_tok1.any() or mask_tok2.any()):
                    total_loss = 0.5 * (s1.mean() + s2.mean())
                    accelerator.backward(total_loss)
                else:
                    continue
                # —— 计算整体 loss 供日志 & EMA 更新 —— #
                if args.mm_dedup:
                    s_merged = 0.5 * (loss_b1 + loss_b2) / eligible.sum(dim=1).clamp(min=1)
                    s_bar = clip_loss(_finite(s_merged, "s_merged"), args.loss_max)
                else:
                    s_bar = 0.5 * (s1 + s2)
                mean_s1 = s1.mean()
                mean_s2 = s2.mean()
                loss    = (mean_s1 + mean_s2) * 0.5
                # —— 记录 hetero 数据 —— #
                if args.hetero_t_in_l and accelerator.is_main_process:
                    hetero_ts += t.detach().cpu().tolist()
                    hetero_ls += s_bar.detach().cpu().tolist()
                if args.EMA:
                    bins = (t * (args.num_bins - 1)).round().long()
                    _ = ema_control_variate_optimal_c(s_bar, baseline, bins, args.baseline_lr, stats, args)
            
            # ---------------------------------
            # ---------- MultiSample ----------
            # ---------------------------------
            elif args.train_mode == "MultiSample":
                # —— 对每个样本单独 backward，避免一次性保留所有前向图 —— #
                fixed_t = torch.rand(ids.size(0), device=device)
                bins = (fixed_t * (args.num_bins - 1)).round().long() if args.EMA else None
                total_loss = None
                for _ in range(args.num_samples):
                    noisy, p_mask, eligible, _, is_weight = forward_process(
                        ids, plen, args.mask_mode, args.coord_format,
                        tokenizer, args.m, fixed_t=fixed_t,
                        use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                    )
                    if not eligible.any(): continue
                    iw = is_weight.masked_fill(~eligible, 1.0)
                    seq = torch.arange(ids.size(1), device=device)[None, :]
                    noisy[seq < plen[:, None]] = ids[seq < plen[:, None]]
                    logits = model(noisy).logits
                    mask_tok = noisy == MASK_TOKEN_ID
                    if not mask_tok.any(): continue
                    ce = F.cross_entropy(logits[mask_tok], ids[mask_tok], reduction='none') / p_mask[mask_tok] * iw[mask_tok]
                    loss_b = torch.zeros(ids.size(0), device=device)
                    loss_b.scatter_add_(0, mask_tok.nonzero(as_tuple=True)[0], ce)
                    s_vec = clip_loss(loss_b / eligible.sum(dim=1).clamp(min=1), args.loss_max)
                    # —— 记录 hetero 数据 —— #
                    if args.hetero_t_in_l and accelerator.is_main_process:
                        hetero_ts += fixed_t.detach().cpu().tolist()
                        hetero_ls += s_vec.detach().cpu().tolist()
                    if args.EMA:
                        single_loss = ema_control_variate_optimal_c(s_vec, baseline, bins, args.baseline_lr, stats, args)
                    else:
                        single_loss = s_vec.mean()
                    # 分样本累加梯度
                    accelerator.backward(single_loss / args.num_samples)
                    total_loss = single_loss.item() if total_loss is None else total_loss + single_loss.item()
                loss = torch.tensor(total_loss, device=device) if total_loss is not None else torch.tensor(0.0, device=device)

            # ---------------------------------
            # ------------ Normal -------------
            # ---------------------------------
            elif args.train_mode == "Normal":
                noisy, p_mask, eligible, t, is_weight = forward_process(
                    ids, plen, args.mask_mode, args.coord_format,
                    tokenizer, args.m, fixed_t=fixed_t,
                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                )
                if not eligible.any(): continue
                iw = is_weight.masked_fill(~eligible, 1.0)
                seq=torch.arange(ids.size(1),device=device)[None,:]; noisy[seq<plen[:,None]]=ids[seq<plen[:,None]]
                logits=model(noisy).logits; mask_tok=noisy==MASK_TOKEN_ID
                if not mask_tok.any(): continue
                ce=F.cross_entropy(logits[mask_tok],ids[mask_tok],reduction='none')/p_mask[mask_tok] * iw[mask_tok]
                loss_b=torch.zeros(ids.size(0),device=device); loss_b.scatter_add_(0,mask_tok.nonzero(as_tuple=True)[0],ce)
                loss=(loss_b/eligible.sum(dim=1).clamp(min=1))
                loss_vec=clip_loss(loss, args.loss_max)
                # —— 记录 hetero 数据 —— #
                if args.hetero_t_in_l and accelerator.is_main_process:
                    hetero_ts += t.detach().cpu().tolist()
                    hetero_ls += loss_vec.detach().cpu().tolist()
                if args.EMA:
                    bins = (t * (args.num_bins-1)).round().long()
                    loss = ema_control_variate_optimal_c(loss_vec, baseline, bins, args.baseline_lr, stats, args)
                else:
                    loss = loss_vec.mean()
                accelerator.backward(loss)

            # log to wandb
            if accelerator.is_main_process and step_idx % args.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[-1],
                    "train/sec": (time.time()-start)/args.logging_steps
                }, step=update_step)
                start = time.time()

            # grad accum
            boundary = (step_idx+1) % accelerator.gradient_accumulation_steps == 0 or ((step_idx + 1) == len(pbar))
            if boundary:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if accelerator.is_main_process and args.save_strategy == "steps" and update_step % args.save_steps == 0:
                    output_dir = make_output_dir_and_broadcast(args, accelerator, gb)
                    ckpt = Path(args.output_dir)/f"checkpoint-update{update_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ckpt, is_main_process=True,save_function=accelerator.save,safe_serialization=False)
                    tokenizer.save_pretrained(ckpt)

            if do_evaluation and args.eval_strategy == "steps" and update_step % args.eval_steps == 0:
                evaluate(update_step)

        if do_evaluation and args.eval_strategy == "epoch":
            evaluate(update_step)
        if accelerator.is_main_process and args.save_strategy == "epoch":
            output_dir = make_output_dir_and_broadcast(args, accelerator, gb)
            ckpt = Path(args.output_dir)/f"checkpoint-epoch{epoch+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt, is_main_process=True,save_function=accelerator.save,safe_serialization=False)
            tokenizer.save_pretrained(ckpt)

    if accelerator.is_main_process and args.save_strategy == "last":
        output_dir = make_output_dir_and_broadcast(args, accelerator, gb)
        ckpt = Path(args.output_dir) / f"checkpoint-epoch{args.epochs}"
        ckpt.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            ckpt,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=False
        )
        tokenizer.save_pretrained(ckpt)

    accelerator.print("🎉 training finished")
    if accelerator.is_main_process:
        wandb.finish()
    output_dir = make_output_dir_and_broadcast(args, accelerator, gb)

    # save grad norm for common / rare ids
    if args.compare_tok_grads and step_counter[0] > 0:
        common_avgs = [s / step_counter[0] for s in common_grad_sum]
        rare_avgs   = [s / step_counter[0] for s in rare_grad_sum]
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "grad_norms.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token_id","avg_grad","category"])
            for tid, avg in zip(common_ids, common_avgs):
                writer.writerow([tid, f"{avg:.6e}", "common"])
            for tid, avg in zip(rare_ids, rare_avgs):
                writer.writerow([tid, f"{avg:.6e}", "rare"])
        print(f"\n✅ Saved grad norms to {csv_path}\n")
        hook_handle.remove()

    # plot hetero of t in l
    if args.hetero_t_in_l and accelerator.is_main_process:
        plt.figure()
        plt.scatter(hetero_ts, hetero_ls, s=1)
        plt.xlabel("t (noise level)")
        plt.ylabel("sample loss")
        plt.title("Loss vs t (heteroscedasticity)")
        fig_path = Path(args.output_dir) / "hetero_t_loss.png"
        plt.savefig(fig_path)
        print(f"🎨 Saved hetero plot to {fig_path}")

    return output_dir, device

def read_batches(path, bs):
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(json.loads(line))
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:
            yield buf

def inference(output_dir: Path, device, args):
    """
    推理阶段：使用最后一个 epoch 的 checkpoint（checkpoint-epoch{args.epochs}）
    """
    # 1) 模型路由
    model_to_pretrained_path = {
        "llada": "GSAI-ML/LLaDA-8B-Instruct",
        "mmada": "Gen-Verse/MMaDA-8B-MixCoT",
    }
    if args.model not in model_to_pretrained_path:
        raise ValueError(f"未知的 model: {args.model}. 合法取值: {list(model_to_pretrained_path.keys())}")

    model_name = model_to_pretrained_path[args.model]
    ckpt_dir  = output_dir / f"checkpoint-epoch{args.epochs}"

    # 2) 超参
    batch_size   = args.batch_size_infer
    max_data     = args.max_data
    temp         = args.temp
    gen_length   = args.gen_length
    steps        = args.steps
    block_length = args.block_length

    # 3) 输出路径
    base_output = Path("/storage/v-mengnijia/LLaDA/eval/data")
    suffix = Path(*ckpt_dir.parts[-2:])
    output_path = base_output / suffix / f"predictions_{args.model}_{args.task}_temp{temp}_gen{gen_length}_steps{steps}_block{block_length}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4) tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 按模型类型加载
    if args.model == "llada":
        # 假设你本地有对应类
        model = LLaDAModelLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    elif args.model == "mmada":
        model = MMadaModelLM.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"不支持的 model: {args.model}")

    model.to(device).eval()

    # 5) 估算进度
    total = sum(1 for _ in open(args.infer_data_path, encoding="utf-8"))
    if max_data is not None:
        total = min(total, max_data)
    progress = tqdm(total=total, desc=f"Infer {args.task}", unit="ex")

    processed = 0

    # 6) 开始推理
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch in read_batches(args.infer_data_path, batch_size):

            # ===== LLaDA 批量推理 =====
            if args.model == "llada":
                if max_data is not None and processed >= max_data:
                    break

                # 批量构造对话；ms 是“批量对话”的列表
                ms = [[{"role": "user", "content": item["prompt"]}] for item in batch]

                # 一次性应用 chat template
                prompts = tokenizer.apply_chat_template(ms, add_generation_prompt=True, tokenize=False)
                enc = tokenizer(
                    prompts,
                    padding="longest",
                    return_tensors="pt",
                    truncation=False
                )
                input_ids = enc["input_ids"].to(device)

                # 批量生成
                outs = generate(
                    model,
                    tokenizer,
                    input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temp,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    enable_cache=True,
                    cache_reloading_step=8,
                )

                # 批量解码
                answers = tokenizer.batch_decode(
                    outs[:, input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                # 写文件
                for item, ans in zip(batch, answers):
                    item["prediction"] = ans
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    processed += 1
                    progress.update(1)

                continue  # 本批已完成，进入下一批

            # ===== MMaDA 单条推理 =====
            elif args.model == "mmada":
                for item in batch:
                    if max_data is not None and processed >= max_data:
                        break

                    prompt_text = item["prompt"]
                    chat = [{"role": "user", "content": prompt_text}]
                    formatted = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                    ids = tokenizer(formatted)["input_ids"]
                    input_ids = torch.tensor(ids, device=device).unsqueeze(0)

                    out = generate(
                        model,
                        input_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temp,
                        cfg_scale=0.0,
                        remasking="low_confidence"
                    )
                    answer = tokenizer.batch_decode(
                        out[:, input_ids.shape[1]:], skip_special_tokens=True
                    )[0]

                    groundtruth = item.get("response") or item.get("groundtruth")
                    sample = {
                        "prompt": prompt_text,
                        "groundtruth": groundtruth,
                        "prediction": answer
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    processed += 1
                    progress.update(1)

                if max_data is not None and processed >= max_data:
                    break
        # 循环结束
    progress.close()
    print(f"✔ 推理完成：处理 {processed} 条样本，结果保存在 {output_path}")
    return output_path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="全局随机种子")
    ap.add_argument("--task", type=str, choices=["codealpaca20k", "dolly15k", "gsm8k", "hitab", "math"], required=True)
    ap.add_argument("--model", type=str, choices=["llada", "mmada"], required=True)
    # 推理相关
    ap.add_argument("--do_infer", action="store_true", help="开启推理")
    ap.add_argument("--no_infer", action="store_false", dest="do_infer", help="关闭推理")
    ap.add_argument("--infer_data_path", type=str, default=None)
    ap.add_argument("--max_data", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gen_length", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--block_length", type=int, default=None)
    ap.add_argument("--batch_size_infer", type=int, default=None)
    # 训练相关
    ap.add_argument("--train_data_path", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--coord_format", type=str, default="hitab-html")
    ap.add_argument("--mask_mode", type=str, choices=["RespMask","CoRE","mixed"], default="RespMask")
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--train_mode", type=str, choices=["Normal", "MirrorMask", "MultiSample"], default="Normal")
    ap.add_argument("--mm_dedup", action="store_true", help="use union-normalized dedup counting for MirrorMask")
    ap.add_argument("--num_samples", type=int, default=8, help="Number of xt samples per x0,t to forward and average loss")
    ap.add_argument("--EMA", action="store_true", help="开启 EMA 控制变量")
    ap.add_argument("--EMA_Optimized", action="store_true", help='Enable online estimation of optimal c per bin')
    ap.add_argument("--num_bins", type=int, default=10, help="Baseline 分段数 T（将 t∈[0,1] 量化为 0…T-1）")
    ap.add_argument("--baseline_lr", type=float, default=0.01, help="Baseline 指数移动平均学习率 η")
    ap.add_argument("--mask_ratio_mode", choices=["random", "stratified"], default="random", help="‘random’: U[0,1]；‘stratified’: 将[0,1]分成bins后分层采样")
    ap.add_argument("--mask_strata_bins", type=int, default=None, help="分层采样的 bin 数，默认为 sqrt(global_batch)")
    ap.add_argument("--IS", action="store_true", help="Enable importance sampling on rare token masks")
    ap.add_argument("--delta", type=float, default=0.2, help="Δ increment for rare token mask probability")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--loss_max", type=float, default=10, help="样本级 loss 的最大值；None 表示不裁剪")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr_scheduler_type", type=str,
                    choices=["constant","constant_with_warmup","linear","cosine","cosine_with_restarts",
                             "polynomial","inverse_sqrt","reduce_lr_on_plateau","cosine_with_min_lr",
                             "warmup_hold_decay"],
                    default="linear",
                    )
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--decay_ratio", type=float, default=0.1)
    ap.add_argument("--final_lr", type=float, default=2.7e-6)
    ap.add_argument("--eval_strategy", type=str, choices=["epoch","steps","no"], default="epoch")
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_strategy", type=str, choices=["epoch","steps", "last"], default="last")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--logging_steps", type=int, default=None)
    ap.add_argument("--compare_tok_grads", action="store_true", help="对比 common / rare token 的梯度幅度")
    ap.add_argument("--common_ids", type=str, default=None)
    ap.add_argument("--rare_ids", type=str, default=None)
    ap.add_argument("--hetero_t_in_l", action="store_true", default=True, help="在训练结束后画出 loss vs t 的散点图，用于展示 heteroscedasticity")
    # 59, 795: \\, G\\
    # 32289: boxed
    # 90: {
    # 28054, 7684: }</, }.
    # 2262: ####

    args = ap.parse_args()

    # 根据 task 类型设置训练数据地址
    task_to_train_data = {
        "codealpaca20k": "/storage/v-mengnijia/LLaDA/data/sft/codealpaca20k_sft_processed.jsonl",
        "dolly15k":      "/storage/v-mengnijia/LLaDA/data/sft/dolly15k_reasoning_sft_str_processed.jsonl",
        "gsm8k":         "/storage/v-mengnijia/LLaDA/data/sft/gsm8k_reasoning_sft_str_processed.jsonl",
        "hitab":         "/storage/v-mengnijia/LLaDA/data/sft/hitab_reasoning_sft_str_processed.jsonl",
        "math":          "/storage/v-mengnijia/LLaDA/data/sft/math_reasoning_sft_str_processed.jsonl",
    }
    if args.train_data_path is None:
        args.train_data_path = task_to_train_data.get(args.task)
        if args.train_data_path is None:
            raise ValueError(f"Unknown task '{args.task}' for training data path.")
    # 根据 task 类型设置推理数据地址
    task_to_infer_data = {
        "codealpaca20k": "/storage/v-mengnijia/LLaDA/data/test/codealpaca20k_test_llada.jsonl",
        "dolly15k":      "/storage/v-mengnijia/LLaDA/data/test/dolly15k_test_llada.jsonl",
        "gsm8k":         "/storage/v-mengnijia/LLaDA/data/test/gsm8k_test_llada.jsonl",
        "hitab":         "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "math":          "/storage/v-mengnijia/LLaDA/data/test/math_test_llada.jsonl",
    }
    if args.infer_data_path is None:
        args.infer_data_path = task_to_infer_data.get(args.task)
        if args.infer_data_path is None:
            raise ValueError(f"Unknown task '{args.task}' for infer data path.")
    # 根据 task 类型设置推理参数
    task_to_infer_params = {
        "codealpaca20k": (256, 128, 32, 4),
        "dolly15k":      (128, 128, 32, 8),
        "gsm8k":         (128, 128, 32, 8),
        "hitab":         (512, 256, 16, 2),
        "math":          (256, 128, 32, 4),
    }
    if getattr(args, "do_infer", False):
        if (args.gen_length is None) or (args.steps is None) or (args.block_length is None) or (args.batch_size_infer is None):
            fours = task_to_infer_params.get(args.task)
            if fours is None:
                raise ValueError(f"Unknown task '{args.task}' for infer defaults.")
            args.gen_length, args.steps, args.block_length, args.batch_size_infer = fours
    # 根据 task 类型设置 common / rare ids
    def _parse_id_list(val):
        """接受 str/list/None，统一返回 List[int]。空返回 []。"""
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        s = str(val).strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if args.common_ids is None:
        if args.task == "hitab":
            args.common_ids = "13,268,11,220,198,16,15,17,341,477,20,19,18,24,301,296,21,297,352,22"
        elif args.task == "gsm8k":
            args.common_ids = "15,220,16,17,198,20,19,28,18,5284,21,29,373,13,23,27,2983,3585,268,9"
        else:
            args.common_ids = ""  # 其它任务默认空列表
    if args.rare_ids is None:
        if args.task == "hitab":
            args.rare_ids = "59,795,32289,90,28504,7684,92"
        elif args.task == "gsm8k":
            args.rare_ids = "2262"
        else:
            args.rare_ids = ""  # 其它任务默认空列表
    args.common_ids = _parse_id_list(args.common_ids)
    args.rare_ids   = _parse_id_list(args.rare_ids)
    # 根据 save strategy 设置 eval strategy
    if args.eval_strategy is None:
        args.eval_strategy = args.save_strategy
    if args.eval_steps is None:
        args.eval_steps = args.save_steps
    # 根据 task 类型设置训练数据地址
    task_to_train_data = {
        "codealpaca20k": "/storage/v-mengnijia/LLaDA/data/sft/codealpaca20k_sft_processed.jsonl",
        "dolly15k":      "/storage/v-mengnijia/LLaDA/data/sft/dolly15k_reasoning_sft_str_processed.jsonl",
        "gsm8k":         "/storage/v-mengnijia/LLaDA/data/sft/gsm8k_reasoning_sft_str_processed.jsonl",
        "hitab":         "/storage/v-mengnijia/LLaDA/data/sft/hitab_reasoning_sft_str_processed.jsonl",
        "math":          "/storage/v-mengnijia/LLaDA/data/sft/math_reasoning_sft_str_processed.jsonl",
    }
    if args.train_data_path is None:
        args.train_data_path = task_to_train_data.get(args.task)
        if args.train_data_path is None:
            raise ValueError(f"Unknown task '{args.task}' for training data path.")
    return args

if __name__ == "__main__":
    args = parse_args()
    out_dir, device = train(args)
    if args.do_infer:
        try:
            import accelerate
            accelerate.utils.release_memory()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        output_path = inference(out_dir, device, args)
