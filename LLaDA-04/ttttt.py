#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_and_infer.py

1) 先执行训练，输出目录由 --output_dir 决定（默认会根据参数自动生成）。
2) 训练结束后，自动找到最后一个 epoch 的 checkpoint（即 checkpoint-epoch{args.epochs}）。
3) 接着加载该 checkpoint，执行推理并将结果写入 JSONL 文件。
"""

import re, argparse, json, math, time, wandb
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from deepspeed.ops.adam import DeepSpeedCPUAdam
# from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from tqdm.auto import tqdm
from generate import generate

# mask token id
MASK_TOKEN_ID = 126336

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

# === Masking logic ===
CODEX_STRUCTURED = {
    "#", "else", "class", "==", "try", "while", "<", "except", "*", "if",
    "for", "with", "!=", "'''", "raise", "return", "def", "from", '"',
    '"""', "::", "as", "..", "/", "Ċ", "import", "{", "...", "}", "'", "+", "@", "-", ",", "[", "(", ":", '"', ")", "=", "]"
}

def is_coord_token(tok: str, fmt: str) -> bool:
    """
    判断 token tok（去除 'Ġ' 前缀后）是否为结构化符号。
    对 codexglue-json 分支仅做硬过滤：tok_clean 必须完全落在 CODEX_STRUCTURED 中。
    """
    tok_clean = tok.lstrip('Ġ')
    assert fmt == 'codexglue-json'
    if tok_clean in CODEX_STRUCTURED:
        return True
    if set(tok) == {'Ġ'} and len(tok) >= 2:
        return True
    return False

def forward_process(batch_ids, prompt_lens, train_mode, coord_format, tokenizer, m, eps=1e-3):
    B, L = batch_ids.shape
    device = batch_ids.device
    t = torch.rand(B, device=device)
    p_mask = ((1 - eps) * t[:, None] + eps).repeat(1, L)
    seq = torch.arange(L, device=device)[None, :]
    resp_pos = seq >= prompt_lens[:, None]
    coord_pos = torch.zeros_like(resp_pos)
    if train_mode == "CoRE":
        # always mask responses + coords + m-fraction of non-coord prompt
        for b in range(B):
            toks = tokenizer.convert_ids_to_tokens(batch_ids[b,:prompt_lens[b]].tolist())
            mask_in = torch.tensor([is_coord_token(tok, coord_format) for tok in toks], device=device)
            coord_pos[b, :prompt_lens[b]] = mask_in
        eligible = resp_pos | coord_pos
        non_coord = (seq < prompt_lens[:,None]) & ~coord_pos
        extra = (torch.rand_like(p_mask) < m) & non_coord
        eligible |= extra
    elif train_mode == "mixed":
        # mask responses always, plus m-fraction of all prompt tokens
        non_resp = seq < prompt_lens[:, None]
        extra = (torch.rand_like(p_mask) < m) & non_resp
        eligible = resp_pos | extra
    else:
        # default to RespMask
        eligible = resp_pos
    noisy = batch_ids.clone()
    mask_here = (torch.rand_like(p_mask) < p_mask) & eligible
    noisy[mask_here] = MASK_TOKEN_ID
    p_mask = p_mask.masked_fill(~eligible, 1.0)
    return noisy, p_mask, eligible

def train(args):
    # accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.grad_accum
    )
    device = accelerator.device
    gb = args.batch_size * args.grad_accum * accelerator.num_processes

    # output_dir default
    if accelerator.is_main_process and args.output_dir is None:
        tag = datetime.now().strftime("%y%m%d")
        if args.train_mode == "CoRE":
            mode = f"CoRE_{args.coord_format}_{args.m}"
        elif args.train_mode == "mixed":
            mode = f"mixed_{args.m}"
        else:
            mode = args.train_mode
        args.output_dir = (
            f"/storage/result/checkpoints/LLaDA/"
            f"instruct_hitab_full_sft_{mode}_{args.train_type}_max_len{args.max_len}_"
            f"epoch{args.epochs}_bs{gb}_"
            f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"
            f"warmup{args.warmup_steps}_{tag}"
        )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer、dataset、loader
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, padding_side="right", use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    ds = LLaDADataset(args.data_file, args.max_len)
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
    # 如果选择 LoRA 微调，则在这里 wrap 一层
    if args.train_type == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,               # LoRA rank，可根据需要调整
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
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
        print(f"◎ 全局 batch = {args.batch_size} × grad_accum {args.grad_accum} × processes {accelerator.num_processes} = {gb}")
        print(f"★ 总数据量 {len(ds)}，训练 {train_n}，评估 {eval_n}，步骤 {total_steps}")
    
    def evaluate(step_idx: int):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, device=device, dtype=torch.long)
        pbar = tqdm(eval_loader, desc=f"Eval@{step_idx}", disable=not accelerator.is_main_process)
        with torch.no_grad():
            for batch in pbar:
                eids = batch["input_ids"].to(device)
                eplen = batch["prompt_lengths"].to(device)
                noisy, p_mask, eligible = forward_process(
                    eids, eplen, train_mode=args.train_mode,
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
            noisy, p_mask, eligible = forward_process(
                ids, plen, train_mode=args.train_mode,
                coord_format=args.coord_format, tokenizer=tokenizer, m=args.m
            )
            seq_idx = torch.arange(ids.shape[1], device=device)[None, :]
            mask_prompt = seq_idx < plen[:, None]
            noisy[mask_prompt] = ids[mask_prompt]
            logits = model(noisy).logits
            mask_tok = noisy == MASK_TOKEN_ID
            if not mask_tok.any(): continue
            ce = F.cross_entropy(logits[mask_tok], ids[mask_tok], reduction='none') / p_mask[mask_tok]
            loss_batch = torch.zeros(ids.size(0), device=device)
            loss_batch.scatter_add_(0, mask_tok.nonzero(as_tuple=False)[:,0], ce)
            denom = eligible.sum(dim=1).clamp(min=1)
            loss = (loss_batch/denom).mean()

            accelerator.backward(loss)
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

    accelerator.print("🎉 training finished")
    if accelerator.is_main_process:
        wandb.finish()

    return output_dir  # 返回训练输出目录

def inference(output_dir: Path, args):
    """
    推理阶段：自动使用最后一个 epoch 的 checkpoint。
    """
    # 构造 checkpoint 路径
    ckpt = output_dir / f"checkpoint-epoch{args.epochs}"
    print(f"▶ Loading checkpoint from {ckpt}")

    # 推理超参
    DATA_PATH    = args.data_path
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
    device       = torch.device("cuda")

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
    total = sum(1 for _ in open(DATA_PATH, encoding="utf-8"))
    if MAX_DATA is not None:
        total = min(total, MAX_DATA)
    prog = tqdm(total=total, desc="Infer", unit="ex")

    processed = 0  # 已处理样本计数

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for batch in read_batches(DATA_PATH, BATCH_SIZE):
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
    # 推理相关
    ap.add_argument("--do_infer", type=bool, default=True)
    ap.add_argument("--data_path", type=str, required=True) # /storage/v-mengnijia/LLaDA/data.jsonl, /storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/llada_code2text/test.jsonl
    ap.add_argument("--max_data", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gen_length", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--block_length", type=int, required=True)
    # 训练相关
    ap.add_argument("--train_type", type=str, choices=["lora", "full-sft"], default="full-sft")
    ap.add_argument("--lora_r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--lora_dropout", type=float, default=None)
    ap.add_argument("--train_mode", type=str, choices=["RespMask","CoRE","mixed"], default="RespMask")
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--coord_format", type=str, choices=["hitab-html", "codexglue-json"], default=None)
    ap.add_argument("--data_file", type=str, default="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl")
    ap.add_argument("--train_ratio", type=float, default=1.0)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--pretrained_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--training_steps", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--logging_steps", type=int, default=None)
    ap.add_argument("--lr_scheduler_type", type=str,
                    choices=["constant","constant_with_warmup","linear","cosine","cosine_with_restarts",
                             "polynomial","inverse_sqrt","reduce_lr_on_plateau","cosine_with_min_lr",
                             "warmup_hold_decay"],
                    default="constant",
                    )
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--decay_ratio", type=float, default=0.1)
    ap.add_argument("--final_lr", type=float, default=2.7e-6)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--train_strategy", type=str, choices=["epoch", "steps"], default="epoch")
    ap.add_argument("--eval_strategy", type=str, choices=["epoch","steps","no"], default=None)
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_strategy", type=str, choices=["epoch","steps"], default="epoch")
    ap.add_argument("--save_steps", type=int, default=100)
    args = ap.parse_args()
    if args.train_type == "lora":
        if args.lora_r is None: args.lora_r = 8
        if args.lora_alpha is None: args.lora_alpha = 16
        if args.lora_dropout is None: args.lora_dropout = 0.05
    if args.train_mode == "CoRE" and args.coord_format is None:
        ap.error("--coord_format is required when --train_mode=CoRE")
    if args.logging_steps is None:
        args.logging_steps = int(320/(args.batch_size * args.grad_accum))
    if args.eval_strategy is None:
        args.eval_strategy = args.save_strategy
    if args.eval_steps is None:
        args.eval_steps = args.save_steps
    return args

if __name__=="__main__":
    args = parse_args()
    # 1) 训练
    out_dir = train(args)
    # 2) 推理
    if args.do_infer: inference(out_dir, args)
