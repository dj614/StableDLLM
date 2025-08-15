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

import gc, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm.auto import tqdm

import wandb
from generate import generate
from is_coord_token import is_coord_token

# mask token id

MASK_TOKEN_ID = 126336

# === Utility functions ===

def ema_control_variate(sample_losses: torch.Tensor,

                         baseline: torch.Tensor,

                         bins: torch.Tensor,

                         baseline_lr: float) -> torch.Tensor:

    """

    Apply EMA control variate: subtract baseline and update it.

    Returns the mean adjusted loss.

    """

    # Retrieve baseline values for each sample

    baseline_vals = baseline[bins]

    # Adjust losses

    adjusted = sample_losses - baseline_vals

    # Update baseline with EMA

    with torch.no_grad():

        baseline[bins] = (1 - baseline_lr) * baseline[bins] + baseline_lr * sample_losses.detach()

    # Return scalar loss

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

    return noisy, used_p, eligible, t



def train(args):

    # accelerator

    accelerator = Accelerator(

        mixed_precision="bf16",

        log_with="wandb",

        gradient_accumulation_steps=args.grad_accum

    )

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

        samples_per_stratum = base

        # 分层采样用的 buffer 和索引

        mask_t_buffer = None

        mask_t_idx = 0

    else:

        # random 模式不需要 buffer

        num_strata = samples_per_stratum = mask_t_buffer = mask_t_idx = None



    # 用于 EMA 模式：存储 baseline 平均值

    b_baseline   = torch.zeros(args.num_bins, device=device, dtype=torch.float32)

    # 用于 SimpleAverage 模式：存储累计平均值和计数

    mu_baseline  = torch.zeros(args.num_bins, device=device, dtype=torch.float32)

    mu_counts    = torch.zeros(args.num_bins, device=device, dtype=torch.long)



    # output_dir default

    is_flag = "IS" if args.IS else "noIS"

    if accelerator.is_main_process and args.output_dir is None:

        tag = datetime.now().strftime("%y%m%d_%H%M%S")

        # mask tag

        if args.mask_mode in ("CoRE", "mixed"): maskM = f"{args.mask_mode}_{args.m}"

        else: maskM = args.mask_mode

        # train mode tag

        if args.train_mode in ("Normal","RB","JackKnife","SimpleAverage","Antithetic","MirrorMask","SemiAnalytic"):

            trainM_base = args.train_mode

        elif args.train_mode in ("MirrorMask_with_EMA","Antithetic_with_EMA","EMA"):

            trainM_base = f"{args.train_mode}_bins{args.num_bins}_blr{args.baseline_lr}"

        else:

            trainM_base = f"{args.train_mode}_ns{args.num_samples}"

        trainM = f"{trainM_base}_{is_flag}"

        # mask ratio mode

        if args.mask_ratio_mode == "random": mask_ratio_mode = "random"

        else: mask_ratio_mode = f"stratified_{args.mask_strata_bins or gb//2}"

        # assemble

        if args.pretrained_path.startswith("GSAI-ML/LLaDA-8B-Instruct"):

            args.output_dir = (

                f"/storage/result/checkpoints/LLaDA/"

                f"instruct_{args.task}_{trainM}_{maskM}_{mask_ratio_mode}_"

                f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb}_"

                f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"

                f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"

            )

        else:

            args.output_dir = (

                f"/storage/result/checkpoints/LLaDA/"

                f"base_{args.task}_{trainM}_{maskM}_{mask_ratio_mode}_"

                f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb}_"

                f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"

                f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"

            )

    args.output_dir = broadcast_object_list([args.output_dir])[0]

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    # tokenizer、dataset、loader

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, padding_side="right", use_fast=True, trust_remote_code=True)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds = LLaDADataset(args.train_data_path, args.max_len)

    train_n = int(len(ds)*args.train_ratio)

    eval_n  = len(ds) - train_n

    train_ds, eval_ds = torch.utils.data.random_split(ds, [train_n, eval_n], generator=torch.Generator().manual_seed(42))

    if eval_n == 0:

        if accelerator.is_main_process:

            print("⚠️ args.train_ratio == 1.0；跳过 eval")

        do_evaluation = False

    else:

        do_evaluation = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: collate_fn(x,pad_id))

    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x: collate_fn(x,pad_id))



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

                noisy, p_mask, eligible, _ = forward_process(

                    eids, eplen, mask_mode=args.mask_mode,

                    coord_format=args.coord_format, tokenizer=tokenizer, m=args.m

                )

                seq_idx = torch.arange(eids.shape[1], device=device)[None, :]

                mask_prompt = seq_idx < eplen[:, None]

                noisy[mask_prompt] = eids[mask_prompt]

                logits = model(noisy).logits

                mask_tok = noisy == MASK_TOKEN_ID

                if not mask_tok.any(): continue

                ce = F.cross_entropy(logits[mask_tok], eids[mask_tok], reduction="none") / p_mask[mask_tok]

                loss_batch = torch.zeros(eids.size(0), device=device)

                loss_batch.scatter_add_(0, mask_tok.nonzero(as_tuple=False)[:,0], ce)

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



        # 注册参数梯度钩子

        hook_handle = emb_weight.register_hook(emb_weight_grad_hook)



    # train loop

    model.train()

    if do_evaluation: evaluate(0)

    update_step = 0

    start = time.time()



    for epoch in range(args.epochs):

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

                    g.manual_seed(40)

                    # 从 num_strata 中挑 rem 个 stratum 多加一个样本

                    extra = torch.randperm(num_strata, generator=g, device=device)[:rem].tolist()

                    parts = []

                    for i in range(num_strata):

                        cnt = base + (1 if i in extra else 0)

                        low, high = i/num_strata, (i+1)/num_strata

                        parts.append(

                            torch.rand(cnt, device=device) * (high - low)

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

            if args.train_mode == "EMA":

                noisy, p_mask, eligible, t = forward_process(

                    ids, plen, args.mask_mode, args.coord_format,

                    tokenizer, args.m, fixed_t=fixed_t,

                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta

                )

                if not eligible.any(): continue

                # restore prompt

                seq = torch.arange(ids.size(1), device=device)[None,:]

                noisy[seq < plen[:,None]] = ids[seq < plen[:,None]]

                logits = model(noisy).logits

                mask_tok = noisy == MASK_TOKEN_ID

                if not mask_tok.any(): continue

                ce = F.cross_entropy(logits[mask_tok], ids[mask_tok], reduction='none') / p_mask[mask_tok]

                loss_b = torch.zeros(ids.size(0), device=device)

                loss_b.scatter_add_(0, mask_tok.nonzero(as_tuple=False)[:,0], ce)

                sample_losses = loss_b / eligible.sum(dim=1).clamp(min=1)

                bins = (t * (args.num_bins-1)).round().long()

                loss = ema_control_variate(sample_losses, b_baseline, bins, args.baseline_lr)

                accelerator.backward(loss)
            
            elif args.train_mode in ("MirrorMask","MirrorMask_with_EMA"):

                _, p_mask, eligible, t = forward_process(

                    ids, plen, args.mask_mode, args.coord_format,

                    tokenizer, args.m, fixed_t=fixed_t,

                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta

                )

                if not eligible.any(): continue

                U=torch.rand_like(p_mask)

                mask1=(U<p_mask)&eligible

                mask2=(U>(1-p_mask))&eligible

                s1=s2=None; total_loss=torch.tensor(0.0,device=device)

                for mask,holder in ((mask1,'s1'),(mask2,'s2')):

                    noisy=ids.clone(); noisy[mask]=MASK_TOKEN_ID

                    seq=torch.arange(ids.size(1),device=device)[None,:]

                    noisy[seq<plen[:,None]]=ids[seq<plen[:,None]]

                    logits=model(noisy).logits; mask_tok=noisy==MASK_TOKEN_ID

                    if mask_tok.any():

                        ce=F.cross_entropy(logits[mask_tok],ids[mask_tok],reduction='none')/p_mask[mask_tok]

                        loss_b=torch.zeros(ids.size(0),device=device); loss_b.scatter_add_(0,mask_tok.nonzero(as_tuple=False)[:,0],ce)

                        s=loss_b/eligible.sum(dim=1).clamp(min=1)

                        if args.train_mode=="MirrorMask":

                            total_loss+=(s*0.5).mean(); accelerator.backward((s*0.5).mean())

                        else:

                            if holder=='s1':s1=s

                            else:s2=s

                    del noisy,logits,mask_tok,ce,loss_b

                if args.train_mode=="MirrorMask_with_EMA" and s1 is not None and s2 is not None:

                    s_bar=0.5*(s1+s2);bins=(t*(args.num_bins-1)).round().long()

                    loss=ema_control_variate(s_bar,b_baseline,bins,args.baseline_lr)

                    accelerator.backward(loss)

                    loss=loss

                elif args.train_mode=="MirrorMask": loss=total_loss

            elif args.train_mode == "MultiSample":

                fixed_t=torch.rand(ids.size(0),device=device)

                total_ms=0.0

                for _ in range(args.num_samples):

                    noisy,p_mask,eligible,_=forward_process(ids,plen,args.mask_mode,args.coord_format,tokenizer,args.m,fixed_t=fixed_t,use_IS=args.IS,rare_ids=args.rare_ids,delta=args.delta)

                    if not eligible.any(): continue

                    seq=torch.arange(ids.size(1),device=device)[None,:]; noisy[seq<plen[:,None]]=ids[seq<plen[:,None]]

                    logits=model(noisy).logits; mask_tok=noisy==MASK_TOKEN_ID

                    if not mask_tok.any(): continue

                    ce=F.cross_entropy(logits[mask_tok],ids[mask_tok],reduction='none')/p_mask[mask_tok]

                    loss_b=torch.zeros(ids.size(0),device=device); loss_b.scatter_add_(0,mask_tok.nonzero(as_tuple=False)[:,0],ce)

                    loss_i=(loss_b/eligible.sum(dim=1).clamp(min=1)).mean()/args.num_samples

                    accelerator.backward(loss_i); total_ms+=loss_i.item()

                    del noisy, p_mask, eligible, logits, mask_tok, ce, loss_b, loss_i

                loss=torch.tensor(total_ms/args.num_samples,device=device)



            elif args.train_mode == "Normal":

                noisy, p_mask, eligible, t = forward_process(

                    ids, plen, args.mask_mode, args.coord_format,

                    tokenizer, args.m, fixed_t=fixed_t,

                    use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta

                )

                if not eligible.any(): continue

                seq=torch.arange(ids.size(1),device=device)[None,:]; noisy[seq<plen[:,None]]=ids[seq<plen[:,None]]

                logits=model(noisy).logits; mask_tok=noisy==MASK_TOKEN_ID

                if not mask_tok.any(): continue

                ce=F.cross_entropy(logits[mask_tok],ids[mask_tok],reduction='none')/p_mask[mask_tok]

                loss_b=torch.zeros(ids.size(0),device=device); loss_b.scatter_add_(0,mask_tok.nonzero(as_tuple=False)[:,0],ce)

                loss=(loss_b/eligible.sum(dim=1).clamp(min=1)).mean(); accelerator.backward(loss)

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



    return output_dir, device  # 返回训练输出目录



def inference(output_dir:Path, device, args):

    """

    推理阶段：自动使用最后一个 epoch 的 checkpoint。

    """

    # 构造 checkpoint 路径

    ckpt = output_dir / f"checkpoint-epoch{args.epochs}"

    print(f"▶ Loading checkpoint from {ckpt}")



    # 推理超参

    infer_data   = args.infer_data_path

    BATCH_SIZE   = 16

    MAX_DATA     = args.max_data

    temperature  = args.temp

    GEN_LENGTH   = args.gen_length

    STEPS        = args.steps

    BLOCK_LENGTH = args.block_length

    BASE_OUT     = Path("/storage/v-mengnijia/LLaDA/eval/data")

    suffix       = Path(*ckpt.parts[-2:])

    OUT_PATH     = BASE_OUT/suffix/f"predictions_temp{temperature}_gen{GEN_LENGTH}_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device       = device



    # load model/tokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

    model     = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype="auto")

    model.eval().to(device)



    # 2. 读取数据集并按 batch_size 分块

    def read_batches(path, bs):

        with open(path, "r", encoding="utf-8") as f:

            buf = []

            for line in f:

                buf.append(json.loads(line))

                if len(buf) == bs:

                    yield buf

                    buf = []

            if buf:                              # 处理最后一个不足 batch 的残包

                yield buf



    # 统计总条数，并截断

    total = sum(1 for _ in open(infer_data, encoding="utf-8"))

    if MAX_DATA is not None:

        total = min(total, MAX_DATA)

    prog = tqdm(total=total, desc="Infer", unit="ex")



    processed = 0  # 已处理样本计数



    with open(OUT_PATH, "w", encoding="utf-8") as fout:

        for batch in read_batches(infer_data, BATCH_SIZE):

            for i, item in enumerate(batch):

                if MAX_DATA is not None and processed >= MAX_DATA:

                    break

                prompt = item["prompt"]

                m = [{"role": "user", "content": prompt}, ]

                prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

                input_ids = tokenizer(prompt)['input_ids']

                input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

                out = generate(model, input_ids, steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH, temperature=temperature, cfg_scale=0., remasking='low_confidence')

                ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

                # 写回 jsonl：保留 prompt / ground_truth，新增 prediction

                orig = batch[i]

                try:

                    gt = orig.pop("response")

                except:

                    gt = orig.pop("groundtruth")

                new_sample = {

                    "prompt":      orig["prompt"],

                    "groundtruth": gt,   # 把 response 拿出来当 groundtruth

                    "prediction":  ans

                }

                fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

               

                processed += 1

                prog.update(1)

           

            if MAX_DATA is not None and processed >= MAX_DATA:

                break



    prog.close()

    print(f"✔ 推理完成，实际处理 {processed} 条样本，结果保存在 {OUT_PATH}")



def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, required=True)

    # 推理相关

    ap.add_argument("--do_infer", action="store_true", help="开启推理")

    ap.add_argument("--no_infer", action="store_false", dest="do_infer", help="关闭推理")

    ap.add_argument("--infer_data_path", type=str, default=None) # /storage/v-mengnijia/LLaDA/data.jsonl, /storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/llada_code2text/test.jsonl

    ap.add_argument("--max_data", type=int, default=None)

    ap.add_argument("--temp", type=float, default=0.0)

    ap.add_argument("--gen_length", type=int)

    ap.add_argument("--steps", type=int)

    ap.add_argument("--block_length", type=int)

    # 训练相关

    ap.add_argument("--pretrained_path", type=str, choices=["GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-8B-Base"], default="GSAI-ML/LLaDA-8B-Instruct")

    ap.add_argument("--train_data_path", type=str, default="/storage/v-mengnijia/LLaDA/data/sft/hitab_reasoning_sft_str_processed.jsonl")

    ap.add_argument("--max_len", type=int, default=4096)

    ap.add_argument("--coord_format", type=str, choices=["hitab-html", "codexglue-json", "cora_pubmed"], default=None)

    ap.add_argument("--mask_mode", type=str, choices=["RespMask","CoRE","mixed"], default="RespMask")

    ap.add_argument("--m", type=float, default=0.0)

    ap.add_argument("--train_mode", type=str, choices=["Normal", "EMA", "SimpleAverage", "Antithetic", "Antithetic_with_EMA", "MirrorMask", "RB", "JackKnife", "MirrorMask_with_EMA", "SemiAnalytic", "MultiSample"], default="MultiSample")

    ap.add_argument("--num_samples", type=int, default=8, help="Number of xt samples per x0,t to forward and average loss")

    ap.add_argument("--num_bins", type=int, default=10, help="Baseline 分段数 T（将 t∈[0,1] 量化为 0…T-1）")

    ap.add_argument("--baseline_lr", type=float, default=0.01, help="Baseline 指数移动平均学习率 η")

    ap.add_argument("--mask_ratio_mode", choices=["random", "stratified"], default="random", help="‘random’: U[0,1]；‘stratified’: 将[0,1]分成bins后分层采样")

    ap.add_argument("--mask_strata_bins", type=int, default=None, help="分层采样的 bin 数，必须整除 global batch size，默认为 sqrt(global_batch)")

    ap.add_argument("--IS", action="store_true", help="Enable importance sampling on rare token masks")

    ap.add_argument("--delta", type=float, default=0.2, help="Δ increment for rare token mask probability")

    ap.add_argument("--epochs", type=int, default=5)

    ap.add_argument("--train_ratio", type=float, default=0.9)

    ap.add_argument("--batch_size", type=int, default=2)

    ap.add_argument("--grad_accum", type=int, default=16)

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

    # >>> 在 parse_args() 末尾 ap.add_argument(...) 下面插入

    ap.add_argument("--compare_tok_grads", action="store_true", help="对比 common / rare token 的梯度幅度")

    ap.add_argument("--common_ids", type=str, default="268,341,301,296,297,352,300,468,3742,259")

    ap.add_argument("--rare_ids", type=str, default="59,795,32289,90,28504,7684")

    # 59, 795: \\, G\\

    # 32289: boxed

    # 90: {

    # 28054, 7684: }</, }.



    args = ap.parse_args()

    if args.mask_mode == "CoRE" and args.coord_format is None:

        ap.error("--coord_format is required when --mask_mode=CoRE")

    if args.train_mode == "MultiSample" and args.num_samples == 1:

        print(">>>>>Warning: args.train_mode == MultiSample but args.num_samples == 1; Depreciate to args.train_mode == Normal")

    if args.eval_strategy is None:

        args.eval_strategy = args.save_strategy

    if args.eval_steps is None:

        args.eval_steps = args.save_steps

    # 如果开启了推理，则这四个参数都必须传

    if args.do_infer:

        missing = []

        for name in ("infer_data_path", "gen_length", "steps", "block_length"):

            if getattr(args, name) is None:

                missing.append(f"--{name}")

        if missing:

            ap.error(f"{', '.join(missing)} is required when --do_infer is set")

    args.common_ids = [int(x) for x in args.common_ids.split(",")]  # ⇽⇽ new

    args.rare_ids   = [int(x) for x in args.rare_ids.split(",")]    # ⇽⇽ new

    return args





if __name__=="__main__":

    args = parse_args()

    # 1) 训练

    out_dir, device = train(args)

    # 2) 推理

    if args.do_infer:

        # ---------- 释放训练显存 ----------

        try:

            import accelerate

            accelerate.utils.release_memory()   # accelerate ≥ 0.27

        except Exception:

            pass

        torch.cuda.empty_cache()

        gc.collect()

        torch.cuda.empty_cache()

        inference(out_dir, device, args)

