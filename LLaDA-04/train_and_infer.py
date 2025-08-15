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
from datasets import load_dataset

import wandb
from generate import generate
from is_coord_token import is_coord_token

# mask token id
MASK_TOKEN_ID = 126336

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
    # 确保 cuDNN 的确定性
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
        # train mode tag
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
        if args.pretrained_path.startswith("GSAI-ML/LLaDA-8B-Instruct"):
            args.output_dir = (
                f"/storage/result/checkpoints/LLaDA/"
                f"seed{args.seed}_instruct_{args.task}_{trainM}_{maskM}_{emaM}_{mask_ratio_mode}_"
                f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb}_"
                f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"
                f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"
            )
        else:
            args.output_dir = (
                f"/storage/result/checkpoints/LLaDA/"
                f"seed{args.seed}_base_{args.task}_{trainM}_{maskM}_{emaM}_{mask_ratio_mode}_"
                f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb}_"
                f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"
                f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"
            )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# === Utility functions ===
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

# === Dataset & collate ===
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

# === Masking logic with Importance Sampling support ===
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
    # 构造 Accelerator，并让它做一次全局 seeding
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.grad_accum
    )

    rank = accelerator.process_index
    set_random_seed(args.seed, rank)

    device = accelerator.device
    gb = args.batch_size * args.grad_accum * accelerator.num_processes

    # —— 准备分层采样 buffer —— #
    assert args.mask_ratio_mode in ("random", "stratified")
    if args.mask_ratio_mode == "stratified":
        # 确定 bins 数目
        if args.mask_strata_bins is None:
            num_strata = int(math.sqrt(gb))
        else:
            num_strata = args.mask_strata_bins
        # 计算每层基础样本数 & 余数，多余样本后面随机分配
        base = gb // num_strata
        rem  = gb - base * num_strata
        # 分层采样用的 buffer 和索引
        mask_t_buffer = None
        mask_t_idx = 0
    else:
        # random 模式不需要 buffer
        num_strata = mask_t_buffer = mask_t_idx = None

    # 用于 EMA 模式
    baseline, stats = init_ema_control_variate(args.num_bins, device=device, dtype=torch.float32)

    output_dir = make_output_dir_and_broadcast(args, accelerator, gb)

    # tokenizer、dataset、loader
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, padding_side="right", use_fast=True, trust_remote_code=True)
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
        # 单卡模式用 RandomSampler + 自有 Generator
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

    # model、optimizer、scheduler
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if getattr(model.config, "is_decoder", None):
        model.config.is_decoder = False
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr, weight_decay=0.1)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

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
        wandb.init(project="llada_sft_hitab", config=vars(args))
        # —— 把 is_coord_token.py 也保存到 WandB run 的 files 里 —— #
        code_path = os.path.join(os.path.dirname(__file__), "is_coord_token.py")
        wandb.save(code_path)
        print(f"◎ 全局 batch = {args.batch_size} × grad_accum {args.grad_accum} × processes {accelerator.num_processes} = {gb}")
        print(f"★ 总数据量 {len(ds)}，训练 {train_n}，评估 {eval_n}，步骤 {total_steps}")
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

    if args.compare_tok_grads:
        common_ids = args.common_ids
        rare_ids   = args.rare_ids
        # 用 Python list 来累加，避免再动 torch.Tensor shape
        common_grad_sum = [0.0] * len(common_ids)
        rare_grad_sum   = [0.0] * len(rare_ids)
        step_counter    = [0]  # 用 list 做“mutable int”

        emb_weight = model.get_input_embeddings().weight

        def emb_weight_grad_hook(grad):
            # 每次 backward 到 embedding weight 时触发
            step_counter[0] += 1
            with torch.no_grad():
                # grad shape = [vocab_size, embed_dim]
                for i, tid in enumerate(common_ids):
                    common_grad_sum[i] += grad[tid].abs().sum().item()
                for i, tid in enumerate(rare_ids):
                    rare_grad_sum[i]   += grad[tid].abs().sum().item()

        # 仅在 embedding 可求导时才注册参数梯度钩子
        if emb_weight.requires_grad:
            hook_handle = emb_weight.register_hook(emb_weight_grad_hook)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "Embedding 权重 requires_grad=False，已跳过注册 compare_tok_grads 钩子。"
            )
            hook_handle = None

    # train loop
    model.train()
    if args.hetero_t_in_l and accelerator.is_main_process:
        hetero_ts: list = []
        hetero_ls: list = []
    if do_evaluation: evaluate(0)
    update_step = 0
    start = time.time()

    for epoch in range(args.epochs):
        # —— 不同模式统一 shuffle —— #
        sampler = train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            # DistributedSampler：内部用 seed + epoch 保证可复现
            sampler.set_epoch(epoch)
        elif hasattr(sampler, "generator"):
            # RandomSampler：显式对每个 epoch 重新 seed
            sampler.generator.manual_seed(args.seed + epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        for step_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            plen = batch["prompt_lengths"].to(device)

            # —— 取本 micro‑batch 的 t —— #
            if args.mask_ratio_mode == "stratified":
                # 在每个更新（global step）周期开始时重建 buffer
                if mask_t_idx == 0:
                    # 随机种子用于额外样本分配和打乱
                    g = torch.Generator(device=device)
                    g.manual_seed(args.seed + update_step + rank)
                    # 从 num_strata 中挑 rem 个 stratum 多加一个样本
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
                    # 打散顺序
                    mask_t_buffer = mask_t_buffer[torch.randperm(gb, generator=g, device=device)]
                # 切出 batch_size 个
                fixed_t = mask_t_buffer[mask_t_idx : mask_t_idx + args.batch_size]
                mask_t_idx += args.batch_size
                if mask_t_idx >= gb:
                    mask_t_idx = 0
            else:
                fixed_t = None

            # ------ train mode branches ------
            # 仅剩 MirrorMask 分支；是否做 EMA 由 args.EMA 决定
            if args.train_mode == "MirrorMask":
                s1 = torch.zeros(ids.size(0), device=device)
                s2 = torch.zeros(ids.size(0), device=device)
                # 1) 先生成 p_mask, eligible, t
                _, p_mask, eligible, t, is_weight = forward_process(
                    ids, plen, args.mask_mode, args.coord_format,
                    tokenizer, args.m, fixed_t=fixed_t,
                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                )
                if not eligible.any():
                    continue
                iw = is_weight.masked_fill(~eligible, 1.0)

                # 2) 同一组 U 决定两侧 mask（MirrorMask）
                U = torch.rand_like(p_mask)
                mask1 = (U < p_mask) & eligible
                mask2 = (U > (1 - p_mask)) & eligible

                # ---------- [DEDUP] 预备：并集概率与交集/差集 ----------
                if args.mm_dedup:
                    # 并集包含概率 π_union = min(2p, 1)
                    p_union = (2.0 * p_mask).clamp(max=1.0)
                    # 数值稳健：非 eligible 位置设为 1，避免无意义的除法被索引到
                    p_union = p_union.masked_fill(~eligible, 1.0)
                    both    = mask1 & mask2
                    only1   = mask1 & (~mask2)
                    only2   = mask2 & (~mask1)

                # —— 第一次前向 & backward (mask1) —— #
                noisy1 = ids.clone()
                noisy1[mask1] = MASK_TOKEN_ID
                noisy1[:, :plen.max()] = ids[:, :plen.max()]  # 保留 prompt 部分
                logits1 = model(noisy1).logits
                mask_tok1 = noisy1 == MASK_TOKEN_ID
                if mask_tok1.any():
                    # 基础 CE（逐 token），先不做 1/p 缩放，统一放到权重里处理
                    ce1_base = F.cross_entropy(
                        logits1[mask_tok1], ids[mask_tok1], reduction='none'
                    )

                    if args.mm_dedup:
                        # [DEDUP] 按 union HT 权重：
                        #  - 出现一次：权重 = 2 / π_union
                        #  - 出现两次：两边各给 1 / π_union（合计 1 / π_union）
                        w1_full = torch.zeros_like(p_mask, dtype=ce1_base.dtype)
                        w1_full[only1] = 2.0 / p_union[only1]
                        w1_full[both]  = 1.0 / p_union[both]
                        # 提取与 mask_tok1 对齐的权重并乘上重要性权重 iw
                        w1 = w1_full[mask_tok1] * iw[mask_tok1]
                        ce1 = ce1_base * w1
                    else:
                        # 兼容原逻辑：单次 HT 权重 = 1 / p（再乘 iw）
                        ce1 = (ce1_base / p_mask[mask_tok1]) * iw[mask_tok1]

                    # 按 batch 聚合到样本级
                    loss_b1 = torch.zeros(ids.size(0), device=device, dtype=ce1.dtype)
                    loss_b1.scatter_add_(0, mask_tok1.nonzero(as_tuple=True)[0], ce1)
                    s1 = clip_loss(
                        loss_b1 / eligible.sum(dim=1).clamp(min=1),
                        args.loss_max
                    )
                    accelerator.backward(s1.mean() * 0.5)   # 维持你的 0.5 缩放

                # —— 第二次前向 & backward (mask2) —— #
                noisy2 = ids.clone()
                noisy2[mask2] = MASK_TOKEN_ID
                noisy2[:, :plen.max()] = ids[:, :plen.max()]
                logits2 = model(noisy2).logits
                mask_tok2 = noisy2 == MASK_TOKEN_ID
                if mask_tok2.any():
                    ce2_base = F.cross_entropy(
                        logits2[mask_tok2], ids[mask_tok2], reduction='none'
                    )

                    if args.mm_dedup:
                        w2_full = torch.zeros_like(p_mask, dtype=ce2_base.dtype)
                        w2_full[only2] = 2.0 / p_union[only2]
                        w2_full[both]  = 1.0 / p_union[both]
                        w2 = w2_full[mask_tok2] * iw[mask_tok2]
                        ce2 = ce2_base * w2
                    else:
                        ce2 = (ce2_base / p_mask[mask_tok2]) * iw[mask_tok2]

                    loss_b2 = torch.zeros(ids.size(0), device=device, dtype=ce2.dtype)
                    loss_b2.scatter_add_(0, mask_tok2.nonzero(as_tuple=True)[0], ce2)
                    s2 = clip_loss(
                        loss_b2 / eligible.sum(dim=1).clamp(min=1),
                        args.loss_max
                    )
                    accelerator.backward(s2.mean() * 0.5)

                # —— 计算整体 loss 供日志 & EMA 更新 —— #
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

                # —— 清理，尽早释放显存 —— #
                torch.cuda.empty_cache()

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
                    # 立即释放
                    # del noisy, logits, mask_tok, ce, loss_b, s_vec, single_loss
                    torch.cuda.empty_cache()
                loss = torch.tensor(total_loss, device=device) if total_loss is not None else torch.tensor(0.0, device=device)

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
                if args.EMA:                                                                     # ←← 新增
                    bins = (t * (args.num_bins-1)).round().long()
                    loss = ema_control_variate_optimal_c(loss_vec, baseline, bins, args.baseline_lr, stats, args)
                else:
                    loss = loss_vec.mean()
                accelerator.backward(loss)
            # ---------------------------------

            boundary = (step_idx+1) % accelerator.gradient_accumulation_steps == 0 or ((step_idx + 1) == len(pbar))
            if accelerator.is_main_process and step_idx % args.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[-1],
                    "train/sec": (time.time()-start)/args.logging_steps
                }, step=update_step)
                start = time.time()
            
            if boundary:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if accelerator.is_main_process and args.save_strategy == "steps" and update_step % args.save_steps == 0:
                    ckpt = Path(args.output_dir)/f"checkpoint-update{update_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(model).save_pretrained(ckpt, is_main_process=True,save_function=accelerator.save,safe_serialization=False)
                    tokenizer.save_pretrained(ckpt)

            if do_evaluation and args.eval_strategy == "steps" and update_step % args.eval_steps == 0:
                evaluate(update_step)

        if do_evaluation and args.eval_strategy == "epoch":
            evaluate(update_step)
        if accelerator.is_main_process and args.save_strategy == "epoch":
            ckpt = Path(args.output_dir)/f"checkpoint-epoch{epoch+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt, is_main_process=True,save_function=accelerator.save,safe_serialization=False)
            tokenizer.save_pretrained(ckpt)

    # -------- 只在最后一次保存 --------
    if accelerator.is_main_process and args.save_strategy == "last":
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

    # -------- 训练完之后 --------
    if args.compare_tok_grads and step_counter[0] > 0:
        common_avgs = [s / step_counter[0] for s in common_grad_sum]
        rare_avgs   = [s / step_counter[0] for s in rare_grad_sum]

        # 2) write to CSV
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

        # 移除钩子，避免意外 side‑effect
        hook_handle.remove()

    # —— 训练结束后画图 —— #
    if args.hetero_t_in_l and accelerator.is_main_process:
        plt.figure()
        plt.scatter(hetero_ts, hetero_ls, s=1)
        plt.xlabel("t (noise level)")
        plt.ylabel("sample loss")
        plt.title("Loss vs t (heteroscedasticity)")
        fig_path = Path(args.output_dir) / "hetero_t_loss.png"
        plt.savefig(fig_path)
        print(f"🎨 Saved hetero plot to {fig_path}")

    return output_dir, device  # 返回训练输出目录

def read_batches(args, batch_size: int):
    """
    Yield batches of data items based on args.task.
    Supports file-based tasks (e.g., 'hitab') and Hugging Face datasets.

    Yields full batches of size 'batch_size', and a final smaller batch
    containing any remainder items if 'len(data)' is not divisible by 'batch_size'.

    Args:
        args: argument namespace with task-specific parameters
        batch_size: number of items per batch
    """
    task = args.task
    max_data = args.max_data

    if task == "hitab":
        path = Path(args.infer_hitab_path)
        buffer = []
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_data is not None and count >= max_data:
                    break
                buffer.append(json.loads(line))
                count += 1
                if len(buffer) == batch_size:
                    yield buffer
                    buffer = []
        # Yield final remainder batch if any
        if buffer:
            yield buffer

    else:
        # Generic dataset loader for any Hugging Face dataset
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split
        )
        if max_data is not None:
            dataset = dataset.select(range(max_data))

        items = [dict(zip(dataset.keys(), vals)) for vals in zip(*dataset.values())]
        total_items = len(items)
        # Iterate in steps of batch_size
        for i in range(0, total_items, batch_size):
            batch = items[i : i + batch_size]
            if batch:
                yield batch

def inference(output_dir:Path, device, args):
    """
    推理阶段：自动使用最后一个 epoch 的 checkpoint。
    """
    # 构造 checkpoint 路径
    MODEL_NAME   = args.pretrained_path
    ckpt         = output_dir / f"checkpoint-epoch{args.epochs}"
    print(f"▶ Loading checkpoint from {ckpt}")

    # Hyperparameters
    BATCH_SIZE   = 16
    MAX_DATA     = args.max_data
    TEMP         = args.temp
    GEN_LENGTH   = args.gen_length
    STEPS        = args.steps
    BLOCK_LENGTH = args.block_length

    # Output directory
    BASE_OUT = Path(args.output_base) if args.output_base else Path("./eval_results")
    suffix = Path(*ckpt.parts[-2:])
    OUT_DIR = BASE_OUT / suffix
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"predictions_{args.task}"
        f"_temp{TEMP}_gen{GEN_LENGTH}"
        f"_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"
    )
    out_file = OUT_DIR / filename

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype="auto")
    model.eval().to(device)

    # Determine total for progress bar
    if args.task == "hitab":
        total = sum(1 for _ in open(args.infer_hitab_path, encoding="utf-8"))
    else:
        ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split
        )
        total = len(ds)
    if MAX_DATA is not None:
        total = min(total, MAX_DATA)
    prog = tqdm(total=total, desc=f"Infer {args.task}", unit="ex")

    # Start Inferencing ...
    with open(out_file, "w", encoding="utf-8") as fout:
        processed = 0
        for batch in read_batches(args, BATCH_SIZE):
            for item in batch:
                if MAX_DATA is not None and processed >= MAX_DATA:
                    break

                # Extract input text
                if args.task == "hitab":
                    text = item["prompt"]
                else:
                    text = item.get(args.input_field)

                # Prepare generation prompt
                chat_input = [{"role": "user", "content": text}]
                prompt = tokenizer.apply_chat_template(
                    chat_input,
                    add_generation_prompt=True,
                    tokenize=False
                )
                input_ids = torch.tensor(
                    tokenizer(prompt)["input_ids"]
                ).to(device).unsqueeze(0)

                # Generate
                out_ids = generate(
                    model,
                    input_ids,
                    steps=STEPS,
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH,
                    temperature=TEMP,
                    cfg_scale=0.0,
                    remasking='low_confidence'
                )
                prediction = tokenizer.batch_decode(
                    out_ids[:, input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]

                # Build result dict
                if args.task == "hitab":
                    gt = item.pop("response", item.pop("groundtruth"))
                    result = {"prompt": item["prompt"], "groundtruth": gt, "prediction": prediction}
                else:
                    result = {
                        args.input_field: item.get(args.input_field),
                        args.output_field: item.get(args.output_field),
                        "prediction": prediction
                    }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
                prog.update(1)

            if MAX_DATA is not None and processed >= MAX_DATA:
                break

    prog.close()
    print(f"✔ Finished {args.task} inference: {processed} samples saved to {out_file}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="全局随机种子")
    ap.add_argument("--task", type=str, choices=["hitab", "gsm8k"], required=True)
    # 推理相关
    ap.add_argument("--do_infer", action="store_true", help="开启推理")
    ap.add_argument("--no_infer", action="store_false", dest="do_infer", help="关闭推理")
    ap.add_argument("--infer_hitab_path", type=str, default="/storage/v-mengnijia/LLaDA/data.jsonl")
    ap.add_argument("--dataset_name", type=str, default="openai/gsm8k",
                    help="Hugging Face dataset name for non-file tasks")
    ap.add_argument("--dataset_config", type=str, default="main",
                    help="Config name for the dataset")
    ap.add_argument("--dataset_split", type=str, default="test",
                    help="Split name for the dataset")
    ap.add_argument("--input_field", type=str, default="question",
                    help="Field name to use as input text for dataset tasks")
    ap.add_argument("--output_field", type=str, default="answer",
                    help="Field name to use as ground truth for dataset tasks")
    ap.add_argument("--max_data", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gen_length", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--block_length", type=int, default=None)
    ap.add_argument("--batch_size_infer", type=int, default=2)
    ap.add_argument("--output_base", type=str, default=None,
                    help="Base directory for inference outputs")
    # 训练相关
    ap.add_argument("--pretrained_path", type=str, choices=["GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-8B-Base"], default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--train_data_path", type=str, default="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--coord_format", type=str, choices=["hitab-html", "codexglue-json", "cora_pubmed"], default=None)
    ap.add_argument("--mask_mode", type=str, choices=["RespMask","CoRE","mixed"], default="RespMask")
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--train_mode", type=str, choices=["Normal", "MirrorMask", "MultiSample"], default="Normal")
    ap.add_argument("--num_samples", type=int, default=8, help="Number of xt samples per x0,t to forward and average loss")
    ap.add_argument("--EMA", action="store_true", help="开启 EMA 控制变量")
    ap.add_argument("--EMA_Optimized", action="store_true", help='Enable online estimation of optimal c per bin')
    ap.add_argument("--num_bins", type=int, default=10, help="Baseline 分段数 T（将 t∈[0,1] 量化为 0…T-1）")
    ap.add_argument("--baseline_lr", type=float, default=0.01, help="Baseline 指数移动平均学习率 η")
    ap.add_argument("--mask_ratio_mode", choices=["random", "stratified"], default="random", help="‘random’: U[0,1]；‘stratified’: 将[0,1]分成bins后分层采样")
    ap.add_argument("--mask_strata_bins", type=int, default=None, help="分层采样的 bin 数，必须整除 global batch size，默认为 sqrt(global_batch)")
    ap.add_argument("--IS", action="store_true", help="Enable importance sampling on rare token masks")
    ap.add_argument("--delta", type=float, default=0.2, help="Δ increment for rare token mask probability")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--loss_max", type=float, default=10, help="样本级 loss 的最大值；None 表示不裁剪")
    ap.add_argument("--train_ratio", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr_scheduler_type", type=str,
                    choices=["constant","constant_with_warmup","linear","cosine","cosine_with_restarts",
                             "polynomial","inverse_sqrt","reduce_lr_on_plateau","cosine_with_min_lr",
                             "warmup_hold_decay"],
                    default="constant",
                    )
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup_steps", type=int, default=35)
    ap.add_argument("--decay_ratio", type=float, default=0.1)
    ap.add_argument("--final_lr", type=float, default=2.7e-6)
    ap.add_argument("--eval_strategy", type=str, choices=["epoch","steps","no"], default="epoch")
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_strategy", type=str, choices=["epoch","steps", "last"], default="last")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--logging_steps", type=int, default=None)
    ap.add_argument("--compare_tok_grads", action="store_true", default=True, help="对比 common / rare token 的梯度幅度")
    ap.add_argument("--common_ids", type=str, default=None)
    ap.add_argument("--rare_ids", type=str, default=None)
    ap.add_argument("--hetero_t_in_l", action="store_true", default=True, help="在训练结束后画出 loss vs t 的散点图，用于展示 heteroscedasticity")
    ap.add_argument("--mm_dedup", action="store_true", default=True, help="use union-normalized dedup counting for MirrorMask")

    # 59, 795: \\, G\\
    # 32289: boxed
    # 90: {
    # 28054, 7684: }</, }.
    # 2262: ####

    args = ap.parse_args()
    if args.do_infer and args.gen_length is None:
        if args.task == "hitab": args.gen_length = 512
        elif args.task == "gsm8k": args.gen_length = 128
    if args.do_infer and args.steps is None:
        if args.task == "hitab": args.steps = 256
        elif args.task == "gsm8k": args.steps = 128
    if args.do_infer and args.block_length is None:
        if args.task == "hitab": args.block_length = 16
        elif args.task == "gsm8k": args.block_length = 32
    if args.common_ids is None:
        if args.task == "hitab": args.common_ids = "13,268,11,220,198,16,15,17,341,477,20,19,18,24,301,296,21,297,352,22"
        elif args.task == "gsm8k": args.common_ids = "15,220,16,17,198,20,19,28,18,5284,21,29,373,13,23,27,2983,3585,268,9"
    if args.rare_ids is None:
        if args.task == "hitab": args.rare_ids = "59,795,32289,90,28504,7684,92"
        elif args.task == "gsm8k": args.rare_ids = "2262"
    if args.mask_mode == "CoRE" and args.coord_format is None:
        ap.error("--coord_format is required when --mask_mode=CoRE")
    if args.train_mode == "MultiSample" and args.num_samples == 1:
        print(">>>>>Warning: args.train_mode == MultiSample but args.num_samples == 1; Depreciate to args.train_mode == Normal")
    if args.eval_strategy is None:
        args.eval_strategy = args.save_strategy
    if args.eval_steps is None:
        args.eval_steps = args.save_steps
    args.common_ids = [int(x) for x in args.common_ids.split(",")]  # ⇽⇽ new
    args.rare_ids   = [int(x) for x in args.rare_ids.split(",")]    # ⇽⇽ new
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
        inference(out_dir, device, args)
