"""Training runner extracted from the original monolithic script.

This module intentionally keeps behavior identical to `LLaDA/rebuttal.py`:
  - Accelerate + DeepSpeedCPUAdam
  - same masking diffusion process and losses
  - optional importance sampling over diffusion time t
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm.auto import tqdm
import wandb

from mdm.engines.llada_plus.data import LLaDADataset, collate_fn
from mdm.engines.llada_plus.diffusion import MASK_TOKEN_ID, forward_process
from mdm.engines.llada_plus.importance_sampling import evaluate_over_x0, fit_p_of_t
from mdm.engines.llada_plus.losses import batched_loss_for_backpropagate
from mdm.utils.seed import set_random_seed
from mdm.utils.output_dir import make_output_dir_and_broadcast


def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.grad_accum,
    )
    rank = accelerator.process_index
    set_random_seed(args.seed, rank)

    device = accelerator.device
    gb = args.batch_size_per_gpu * args.grad_accum * accelerator.num_processes

    model_path = "GSAI-ML/LLaDA-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    g_data = torch.Generator().manual_seed(args.seed)

    # dataset & dataloader
    ds = LLaDADataset(args.train_data_path, args.max_len)
    train_n = int(len(ds) * args.train_ratio)
    eval_n = len(ds) - train_n
    train_ds, eval_ds = torch.utils.data.random_split(ds, [train_n, eval_n], generator=g_data)
    do_evaluation = eval_n != 0
    if not do_evaluation:
        accelerator.print("args.train_ratio == 1.0; skipping eval")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if getattr(model.config, "is_decoder", None):
        model.config.is_decoder = False

    # optimizer
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr, weight_decay=0.1)

    # prepare via accelerator
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    # scheduler
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    # wandb init
    if accelerator.is_main_process:
        wandb.init(project=f"llada_{args.task}", config=vars(args))
    accelerator.print(
        f"global batch = {args.batch_size_per_gpu} x grad_accum {args.grad_accum} x "
        f"processes {accelerator.num_processes} = {gb}"
    )
    accelerator.print(f"data counts: train {train_n}, eval {eval_n}, steps {total_steps}")

    # output dir
    output_dir = make_output_dir_and_broadcast(args, accelerator)

    # importance sampling over t
    if args.PPOTS:
        model.eval()
        start_time = time.time()
        wi, gi, vi, t_values = evaluate_over_x0(
            model=model,
            NUM_SAMPLES_X0=args.num_samples_x0,
            NUM_SAMPLES_T=args.num_samples_t,
            NUM_SAMPLES_XT=args.num_samples_xt,
            args=args,
            device=device,
            pad_id=pad_id,
        )
        res = fit_p_of_t(
            wi=wi,
            gi=gi,
            vi=vi,
            t_values=t_values,
            args=args,
            fit_mode=args.p_model,
            plot=accelerator.is_main_process,
        )
        end_time = time.time()
        accelerator.print("=== Experiment Config & Time ===")
        accelerator.print(f"num_samples_x0 = {args.num_samples_x0}")
        accelerator.print(f"num_samples_t  = {args.num_samples_t}")
        accelerator.print(f"num_samples_xt = {args.num_samples_xt}")
        accelerator.print(f"num_starts     = {args.n_starts}")
        accelerator.print(f"Total time     = {end_time - start_time:.2f} sec")
        model.train()

        t_grid = torch.tensor(res.t_values, device=device, dtype=torch.float32)
        p_grid = torch.tensor(res.p_on_grid, device=device, dtype=torch.float32)
        p_grid = p_grid / torch.trapz(p_grid, t_grid).clamp(min=1e-12)
        dt = t_grid[1:] - t_grid[:-1]
        cumsum_area = torch.cumsum(0.5 * (p_grid[:-1] + p_grid[1:]) * dt, dim=0)
        cdf = torch.cat([torch.zeros(1, device=device), cumsum_area], dim=0)
        cdf = cdf / cdf[-1].clamp(min=1e-12)

        def sample_t_from_p(batch_size_per_gpu: int):
            u = torch.rand(batch_size_per_gpu, device=device)
            idx = torch.searchsorted(cdf, u, right=True)
            idx = torch.clamp(idx, 1, t_grid.numel() - 1)
            t0, t1 = t_grid[idx - 1], t_grid[idx]
            c0, c1 = cdf[idx - 1], cdf[idx]
            r = (u - c0) / (c1 - c0 + 1e-12)
            t_s = t0 + r * (t1 - t0)
            p0, p1 = p_grid[idx - 1], p_grid[idx]
            p_s = p0 + r * (p1 - p0)
            iw = (1.0 / p_s.clamp(min=1e-12)).detach()
            if hasattr(args, "max_is_weight") and args.max_is_weight is not None:
                iw = iw.clamp_max(float(args.max_is_weight))
            return t_s, iw

    else:
        sample_t_from_p = None

    # eval loop
    def evaluate(step_idx: int):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        pbar = tqdm(eval_loader, desc=f"Eval@{step_idx}", disable=not accelerator.is_main_process)
        with torch.no_grad():
            for batch in pbar:
                eids = batch["input_ids"].to(device)
                eam = batch["attention_mask"].to(device)
                elbls = batch["labels"].to(device)

                fixed_t = torch.rand(eids.shape[0], device=device)
                iw_t = torch.ones(eids.shape[0], device=device)
                if args.PPOTS:
                    fixed_t, iw_t = sample_t_from_p(eids.shape[0])

                p_mask, iw_t, noisy1, _, eligible = forward_process(
                    eids,
                    eam,
                    elbls,
                    args.train_mode,
                    fixed_t=fixed_t,
                    iw_t=iw_t,
                )
                if not (noisy1 == MASK_TOKEN_ID).any():
                    continue

                loss = batched_loss_for_backpropagate(
                    eids,
                    noisy1,
                    model,
                    p_mask,
                    iw_t,
                    eligible,
                    train=False,
                    pad_id=pad_id,
                    attn_mask=eam,
                )
                loss_world = accelerator.gather(loss.detach()).mean()
                total_loss += loss_world
                num_batches += 1
                if accelerator.is_main_process:
                    pbar.set_postfix(loss=loss_world.item())
        if num_batches > 0 and accelerator.is_main_process:
            avg_loss = (total_loss / num_batches).item()
            wandb.log({"eval/loss": avg_loss}, step=step_idx)
        model.train()
        accelerator.wait_for_everyone()

    # training
    if do_evaluation:
        evaluate(0)

    update_step = 0
    start = time.time()
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )
        for step_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            fixed_t = torch.rand(ids.shape[0], device=device)
            iw_t = torch.ones(ids.shape[0], device=device)
            if args.PPOTS:
                fixed_t, iw_t = sample_t_from_p(ids.shape[0])

            p_mask, iw_t, noisy1, noisy2, eligible = forward_process(
                ids,
                am,
                lbls,
                args.train_mode,
                fixed_t=fixed_t,
                iw_t=iw_t,
            )
            has_mask1 = (noisy1 == MASK_TOKEN_ID).any()
            has_mask2 = (noisy2 == MASK_TOKEN_ID).any() if noisy2 is not None else False

            if args.train_mode == "Normal":
                if not has_mask1:
                    continue
            elif args.train_mode in ["MIRROR"]:
                if not (has_mask1 or has_mask2):
                    continue

            if args.train_mode == "Normal":
                loss = batched_loss_for_backpropagate(
                    ids,
                    noisy1,
                    model,
                    p_mask,
                    iw_t,
                    eligible,
                    train=True,
                    debug={},
                    pad_id=pad_id,
                    attn_mask=am,
                )

            elif args.train_mode == "MIRROR":
                loss1 = batched_loss_for_backpropagate(
                    ids,
                    noisy1,
                    model,
                    p_mask,
                    iw_t,
                    eligible,
                    train=True,
                    debug={},
                    pad_id=pad_id,
                    attn_mask=am,
                )
                loss2 = batched_loss_for_backpropagate(
                    ids,
                    noisy2,
                    model,
                    p_mask,
                    iw_t,
                    eligible,
                    train=True,
                    debug={},
                    pad_id=pad_id,
                    attn_mask=am,
                )
                loss = 0.5 * (loss1 + loss2)

            accelerator.backward(loss)

            if accelerator.is_main_process and update_step % args.logging_steps == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[-1],
                        "train/sec": (time.time() - start) / args.logging_steps,
                    },
                    step=update_step,
                )
                start = time.time()

            boundary = (
                (step_idx + 1) % accelerator.gradient_accumulation_steps == 0
                or ((step_idx + 1) == len(pbar))
            )
            if boundary:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

        if do_evaluation and args.eval_strategy == "epoch":
            evaluate(update_step)

    if accelerator.is_main_process and args.save_strategy == "last":
        ckpt = Path(args.output_dir) / f"checkpoint-epoch{args.epochs}"
        ckpt.mkdir(parents=True, exist_ok=True)
        state_dict = accelerator.get_state_dict(model)
        accelerator.unwrap_model(model).save_pretrained(
            ckpt,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(ckpt)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

    return output_dir, device
