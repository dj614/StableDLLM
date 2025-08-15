import re
import json
import math
import random
from typing import List, Any
import time
from datetime import datetime
from pathlib import Path

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm.auto import tqdm
import wandb

from generate import generate

# mask token id
MASK_TOKEN_ID = 126336

SPECIAL = dict(
    BOS="<s>",               # tokenizer.bos_token
    EOS="</s>",              # tokenizer.eos_token
    START_USER="<start_id>user<end_id>\n",
    START_ASSIST="<start_id>assistant<end_id>\n",
    EOT="<eot_id>",          # end-of-turn
)

def make_output_dir_and_broadcast(args, accelerator, gb=None):
    if accelerator.is_main_process and args.output_dir is None:
        tag = datetime.now().strftime("%y%m%d_%H%M%S")
        is_flag = f"IS{args.delta}" if args.IS else "noIS"
        trainM = args.train_mode
        emaM = "noEMA"
        if args.EMA:
            emaM = f"EMA_Opt_bins{args.num_bins}_blr{args.baseline_lr}" if args.EMA_Optimized else f"EMA_bins{args.num_bins}_blr{args.baseline_lr}"
        args.output_dir = (
            f"/storage/result/checkpoints/LLaDA/"
            f"instruct_{args.task}_{trainM}_{emaM}_{is_flag}_"
            f"train_ratio{args.train_ratio}_epoch{args.epochs}_bs{gb or args.batch_size}_"
            f"lr_sched_{args.lr_scheduler_type}_lr{args.lr}_"
            f"warmup{args.warmup_steps}_max_len{args.max_len}_{tag}"
        )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return


def clip_loss(x: torch.Tensor, max_val: float = None):
    return x if max_val is None else torch.clamp(x, max=max_val)


def init_ema_control_variate(num_bins: int, device: torch.device = None, dtype=torch.float32, baseline_init: float = 0.0):
    device = device or torch.device('cpu')
    baseline = torch.full((num_bins,), baseline_init, device=device, dtype=dtype)
    stats = {k: torch.zeros(num_bins, device=device, dtype=dtype) for k in ['mu_L','mu_H','M_LH','M_HH']}
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
    stats_lr = stats_lr or baseline_lr
    baseline_vals = baseline[bins].detach()
    L = sample_losses
    c_per_bin = torch.ones_like(baseline_vals)
    if args.EMA_Optimized:
        mu_L, mu_H = stats['mu_L'], stats['mu_H']
        M_LH, M_HH = stats['M_LH'], stats['M_HH']
        cur_L, cur_H = mu_L[bins], mu_H[bins]
        cur_LH, cur_HH = M_LH[bins], M_HH[bins]
        with torch.no_grad():
            new_mu_L = mu_L.clone(); new_mu_H = mu_H.clone()
            new_M_LH = M_LH.clone(); new_M_HH = M_HH.clone()
            new_mu_L[bins] = (1 - stats_lr)*cur_L + stats_lr*L
            new_mu_H[bins] = (1 - stats_lr)*cur_H + stats_lr*baseline_vals
            new_M_LH[bins] = (1 - stats_lr)*cur_LH + stats_lr*(L*baseline_vals)
            new_M_HH[bins] = (1 - stats_lr)*cur_HH + stats_lr*(baseline_vals*baseline_vals)
            stats.update({'mu_L': new_mu_L,'mu_H': new_mu_H,'M_LH': new_M_LH,'M_HH': new_M_HH})
        cov = stats['M_LH'] - stats['mu_L']*stats['mu_H']
        var = stats['M_HH'] - stats['mu_H'].pow(2)
        c_per_bin = cov[bins] / (var[bins] + eps)
    adjusted = L - c_per_bin*baseline_vals
    with torch.no_grad():
        baseline.data[bins] = (1 - baseline_lr)*baseline_vals + baseline_lr*L.detach()
    return adjusted.mean()


def encode_example(ex, tok):
    prompt_txt = ex.get("prompt", "")
    # for supervised SFT you may have a reference 'response' field
    answer_txt = ex.get("response", "")

    user_part = SPECIAL["BOS"] + SPECIAL["START_USER"] + prompt_txt + SPECIAL["EOT"]
    asst_part = SPECIAL["START_ASSIST"] + answer_txt + SPECIAL["EOS"]

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    prompt_lens = len(user_ids)
    return {"prompt_ids": ids, "prompt_lens": prompt_lens, 'groundtruth':ex.get('groundtruth',[])}


class PromptDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int):
        """
        Each example is statically encoded from 'prompt' and 'response' fields.
        prompt_ids = [BOS user prompt EOT assistant response EOS]
        prompt_lens indexes the split point.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        with open(jsonl_path, 'r', encoding='utf8') as f:
            for ln in f:
                ex = json.loads(ln)
                rec = encode_example(ex, tokenizer)
                # drop if total length exceeds max_len
                if rec["prompt_lens"] <= max_len:
                    self.examples.append(rec)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        rec = self.examples[idx]
        return {
            'prompt_ids': torch.tensor(rec['prompt_ids'], dtype=torch.long),
            'prompt_lens': rec['prompt_lens'],
            'groundtruth': rec["groundtruth"]
        }


def collate_fn(batch, pad_id):
    """
    Pads 'prompt_ids' to the longest sequence in batch.
    Returns:
      - prompt_ids: LongTensor[B, L]
      - prompt_lens: LongTensor[B]
    """
    lengths = [b['prompt_ids'].size(0) for b in batch]
    prompt_lens = torch.tensor([b['prompt_lens'] for b in batch], dtype=torch.long)
    max_len = max(lengths)
    gts, padded_ids = [], []
    for b in batch:
        gts.append(b['groundtruth'])
        seq = b['prompt_ids']
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long, device=seq.device)
            seq = torch.cat([seq, pad_tensor], dim=0)
        padded_ids.append(seq)
    prompt_ids = torch.stack(padded_ids, dim=0)
    return {
        'prompt_ids': prompt_ids,
        'prompt_lens': prompt_lens,
        "groundtruth": gts
    }


def forward_process(ids, prompt_lens, eps=1e-3, fixed_t=None, use_IS=False, rare_ids=None, delta=0.2):
    B,L = ids.shape
    dev = ids.device
    t   = fixed_t if fixed_t is not None else torch.rand(B, device=dev)
    p   = ((1 - eps) * t[:,None] + eps).repeat(1,L)
    seq = torch.arange(L, device=dev)[None,:]
    eligible = seq >= prompt_lens[:,None]
    if use_IS and rare_ids:
        rare = torch.tensor(rare_ids, device=dev)
        pos  = (ids.unsqueeze(-1) == rare.view(1, 1, -1)).any(-1)
        q    = p.clone()
        q[pos] = torch.minimum(q[pos] + delta, torch.ones_like(q[pos]))
        used_p = q
    else: 
        used_p = p
    noisy = ids.clone()
    mask = (torch.rand_like(p) < used_p) & eligible
    noisy[mask] = MASK_TOKEN_ID
    used_p = used_p.masked_fill(~eligible, 1.0)
    is_weight = p.div(used_p)
    return noisy, used_p, eligible, t, is_weight


def estimate_log_probs(model, inp, labels, mask):
    logits = model(inp).logits
    lp     = F.log_softmax(logits, dim=-1)
    sel    = lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    mask_sum = mask.sum(dim=1).clamp(min=1)
    return ((sel * mask).sum(dim=1) / mask_sum)


def extract_predictions(texts:List[str]) -> List[List[str]]:
    pat = r"\\\\boxed\{(.*?)\}";res=[]
    for t in texts:
        ms  = re.findall(pat, t, flags=re.DOTALL)
        ans = set()
        for m in ms:
            a = m.strip()
            w = re.match(r"\\\\text\s*\{(.*?)\}", a, flags=re.DOTALL)
            if w: a = w.group(1)
            ans.add(a.replace('\\%','%').replace('\\','').rstrip('.'))
        res.append(list(ans))
    return res


def extract_groundtruths(o:Any) -> List[str]:
    if not isinstance(o,list) or not o: return []
    out = []
    for e in o:
        out.extend(e if isinstance(e,list) else [e])
    return out


def compare_results(preds:List[List[str]], gts:List[str]) -> List[List[bool]]:
    gt = set(gts)
    return [[p in gt for p in ps] for ps in preds]


def compute_reward(comps:List[str], gts:Any)->torch.Tensor:
    ps = extract_predictions(comps)
    gt = extract_groundtruths(gts)
    mt = compare_results(ps,gt)
    r  = []
    for p, m in zip(ps, mt):
        if not p: r.append(-1.0)
        else: r.append(0.0 if len(gt)==0 else sum(m)/len(gt))
    return torch.tensor(r)


def compute_advantage(rewards:torch.Tensor, eps=1e-8)->torch.Tensor:
    m = rewards.mean()
    s = rewards.std(unbiased=True)
    return (rewards - m) / (s + eps)

def unigrpo_train(args):
    accelerator = Accelerator(mixed_precision='bf16',log_with='wandb')
    dev = accelerator.device
    gb = args.batch_size * accelerator.num_processes

    # intialize EMA stats
    baseline, stats = init_ema_control_variate(args.num_bins, device=dev)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, use_fast=True, trust_remote_code=True)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    # dataloader
    ds = PromptDataset(args.train_data, tokenizer, args.max_len)
    n1 = int(len(ds) * args.train_ratio)
    n2 = len(ds) - n1
    tr, ev = torch.utils.data.random_split(ds, [n1,n2], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x:collate_fn(x, pad))
    ev_loader = DataLoader(ev, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=lambda x:collate_fn(x, pad))

    # load models
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    ref   = AutoModelForCausalLM.from_pretrained(args.ref_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    old   = AutoModelForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    for m in (model, ref, old):
        if getattr(m.config, 'is_decoder', None):
            m.config.is_decoder=False
    
    # optimizer
    optim = DeepSpeedCPUAdam(model.parameters(), lr=args.lr, weight_decay=0.1)

    # scheduler
    total = math.ceil(len(tr_loader) * args.epochs)
    sched = get_scheduler(args.lr_scheduler_type, optim, num_warmup_steps=args.warmup_steps, num_training_steps=total)

    # prepare via accelerator
    model, optim, tr_loader, ev_loader, sched = accelerator.prepare(model, optim, tr_loader, ev_loader, sched)
    ref   = ref.to(dev)
    old   = old.to(dev)

    # initialize wandb
    if accelerator.is_main_process:
        wandb.init(project=f'llada_sft_{args.task}',config=vars(args))
        args.logging_steps = args.logging_steps or max(1, (len(ds) * args.epochs) // args.batch_size // 100)
    
    print(f"训练数据：{n1}")
    # training loop
    model.train()
    update_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        old.load_state_dict(accelerator.unwrap_model(model).state_dict())
        pbar = tqdm(tr_loader,desc=f'Epoch {epoch+1}/{args.epochs}',disable=not accelerator.is_main_process)
        for step_idx, batch in enumerate(pbar):
            prompt_ids = batch["prompt_ids"]
            gts        = batch['groundtruth']
            pls        = batch['prompt_lens'].to(dev)
            # sample G completions for each prompt
            all_comps, all_comp_ids = [], []
            for i in prompt_ids:
                comps, ids_list = [], []
                for _ in range(args.G):
                    comp = generate(old, i.unsqueeze(0), steps=args.steps, gen_length=args.gen_length,
                                             block_length=args.block_length, temperature=args.temperature,
                                             cfg_scale=args.cfg_scale, remasking=args.remasking,
                                             return_logprobs=False)
                    ids_list.append(comp)
                    comps.append(tokenizer.batch_decode(comp, skip_special_tokens=True)[0])
                all_comp_ids.append(ids_list)
                all_comps.append(comps)
            # compute rewards and advantages
            all_rew   = [compute_reward(comps, gt).to(dev) for comps, gt in zip(all_comps, gts)]
            all_adv   = [compute_advantage(r) for r in all_rew]
            # update gradients
            if args.train_mode == 'Normal':
                # 先采样一组 mu 个 t
                r1 = random.random(); t1 = math.floor(r1 * args.T)
                steps = [t1] + [math.floor(n/(args.mu-1)*(args.T-t1)+t1) for n in range(1, args.mu)]
                for t in steps:
                    total_loss = 0.0
                    count = 0
                    # 对每个 prompt 和它的 G 个 completion
                    for i, (comps, advs) in enumerate(zip(all_comps, all_adv)):
                        pl = torch.tensor([pls[i].item()], device=dev)
                        for j, A in enumerate(advs):
                            comp_ids = all_comp_ids[i][j]
                            gen_ids  = comp_ids[0, pl: ]
                            base_ids = torch.cat([prompt_ids[i, :pl], gen_ids], dim=0).unsqueeze(0)
                            # 固定 t, 对这个 completion 计算一次 loss
                            noisy, _, eligible, _, is_weight = forward_process(
                                base_ids, pl,
                                fixed_t=torch.tensor([t/args.T], device=dev),
                                use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                            )
                            mask = noisy == MASK_TOKEN_ID
                            lp_t = estimate_log_probs(model, noisy, base_ids, mask)
                            with torch.no_grad():
                                lp_o = estimate_log_probs(old, noisy, base_ids, mask)
                            ratio = torch.exp(lp_t - lp_o)
                            clipped = torch.clamp(ratio, 1-args.clip_epsilon, 1+args.clip_epsilon)
                            iw = is_weight.masked_fill(~eligible, 1.0)
                            surrogate = torch.min(ratio * A * iw, clipped * A * iw)
                            policy_loss = -surrogate.mean()
                            # KL penalty
                            with torch.no_grad():
                                ref_logits = ref(noisy).logits
                            logp = F.log_softmax(model(noisy).logits, dim=-1)
                            logpref = F.log_softmax(ref_logits, dim=-1)
                            kl_per_token = (F.softmax(model(noisy).logits, dim=-1) * (logp - logpref)).sum(dim=-1)
                            avg_kl = (kl_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                            kl_loss = args.beta * avg_kl.mean()
                            L = policy_loss + kl_loss
                            # EMA or clip
                            if args.EMA:
                                bin_idx = torch.full((1,), int((t/args.T)*(args.num_bins-1)), device=dev)
                                loss = ema_control_variate_optimal_c(
                                    L.to(baseline.dtype).unsqueeze(0),
                                    baseline, bin_idx, args.baseline_lr, stats, args
                                )
                            else:
                                loss = clip_loss(L, args.loss_max)
                            total_loss += loss
                            count += 1
                    # 平均后一次性更新
                    avg_loss = total_loss / count
                    accelerator.backward(avg_loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                    update_step += 1

            # train_mode branch: MirrorMask（按 timestep 聚合所有 completions 再更新）
            elif args.train_mode == 'MirrorMask':
                M = (args.mu + 1) // 2
                # sample mu/2 个 timestep
                t0 = random.randint(0, args.T - 1)
                steps = [t0] + [math.floor(n/(M-1)*(args.T - t0) + t0) for n in range(1, M)]
                for t in steps:
                    total_loss = 0.0
                    count = 0
                    # 对每个 prompt 和它的 G 个 completion
                    for i, (comps, advs) in enumerate(zip(all_comps, all_adv)):
                        pl = torch.tensor([pls[i].item()], device=dev)
                        for j, A in enumerate(advs):
                            comp_ids = all_comp_ids[i][j]
                            gen_ids  = comp_ids[0, pl: ]
                            base_ids = torch.cat([prompt_ids[i, :pl], gen_ids], dim=0).unsqueeze(0)
                            # 先生成 p_mask, eligible, t
                            _, p_mask, eligible, t, is_weight = forward_process(
                                base_ids, pl,
                                fixed_t=torch.tensor([t/args.T], device=dev),
                                use_IS=args.IS, rare_ids=args.rare_ids, delta=args.delta
                            )
                            if not eligible.any():
                                continue
                            iw = is_weight.masked_fill(~eligible, 1.0)
                            # 随机 U 决定两侧 mask
                            U = torch.rand_like(p_mask)
                            mask1 = (U < p_mask) & eligible
                            mask2 = (U > (1 - p_mask)) & eligible
                            # 生成第一个 noisy & mask
                            noisy1 = base_ids.clone()
                            noisy1[mask1] = MASK_TOKEN_ID
                            noisy1[:, :pl.max()] = base_ids[:, :pl.max()]  # 保留 prompt 部分
                            logits1 = model(noisy1).logits
                            mask1 = noisy1 == MASK_TOKEN_ID
                            # 生成第二个 noisy & mask
                            noisy2 = base_ids.clone()
                            noisy2[mask2] = MASK_TOKEN_ID
                            noisy2[:, :pl.max()] = base_ids[:, :pl.max()]  # 保留 prompt 部分
                            logits2 = model(noisy2).logits
                            mask2 = noisy2 == MASK_TOKEN_ID
                            # 计算第一个 ratio
                            lp_t1 = estimate_log_probs(model, noisy1, base_ids, mask1)
                            with torch.no_grad():
                                lp_o1 = estimate_log_probs(old, noisy1, base_ids, mask1)
                            ratio1 = torch.exp(lp_t1 - lp_o1)
                            clipped1 = torch.clamp(ratio1, 1-args.clip_epsilon, 1+args.clip_epsilon)
                            # 计算第二个 ratio
                            lp_t2 = estimate_log_probs(model, noisy2, base_ids, mask2)
                            with torch.no_grad():
                                lp_o2 = estimate_log_probs(old, noisy2, base_ids, mask2)
                            ratio2 = torch.exp(lp_t2 - lp_o2)
                            clipped2 = torch.clamp(ratio2, 1-args.clip_epsilon, 1+args.clip_epsilon)
                            # 计算重要性采样加权 surrogate
                            iw = is_weight.masked_fill(~eligible, 1.0)
                            surrogate1 = torch.min(ratio1 * A * iw, clipped1 * A * iw)
                            policy_loss1 = -surrogate1.mean()
                            surrogate2 = torch.min(ratio2 * A * iw, clipped2 * A * iw)
                            policy_loss2 = -surrogate2.mean()
                            # KL penalty 1
                            with torch.no_grad():
                                ref_logits1 = ref(noisy1).logits
                            logp1 = F.log_softmax(model(noisy1).logits, dim=-1)
                            logpref1 = F.log_softmax(ref_logits1, dim=-1)
                            kl_per_token1 = (F.softmax(model(noisy1).logits, dim=-1) * (logp1 - logpref1)).sum(dim=-1)
                            avg_kl1 = (kl_per_token1 * mask1).sum(dim=1) / mask1.sum(dim=1).clamp(min=1)
                            kl_loss1 = args.beta * avg_kl1.mean()
                            L1 = policy_loss1 + kl_loss1
                            # KL penalty 2
                            with torch.no_grad():
                                ref_logits2 = ref(noisy2).logits
                            logp2 = F.log_softmax(model(noisy2).logits, dim=-1)
                            logpref2 = F.log_softmax(ref_logits2, dim=-1)
                            kl_per_token2 = (F.softmax(model(noisy2).logits, dim=-1) * (logp2 - logpref2)).sum(dim=-1)
                            avg_kl2 = (kl_per_token2 * mask2).sum(dim=1) / mask2.sum(dim=1).clamp(min=1)
                            kl_loss2 = args.beta * avg_kl2.mean()
                            L2 = policy_loss2 + kl_loss2
                            # EMA or clip
                            if args.EMA:
                                bin_idx = torch.full((1,), int((t/args.T)*(args.num_bins-1)), device=dev)
                                loss1 = ema_control_variate_optimal_c(
                                    L1.to(baseline.dtype).unsqueeze(0),
                                    baseline, bin_idx, args.baseline_lr, stats, args
                                )
                                loss2 = ema_control_variate_optimal_c(
                                    L2.to(baseline.dtype).unsqueeze(0),
                                    baseline, bin_idx, args.baseline_lr, stats, args
                                )
                            else:
                                loss1 = clip_loss(L1, args.loss_max)
                                loss2 = clip_loss(L2, args.loss_max)
                            total_loss += loss1 + loss2
                            count += 1 # not 2!
                    # 平均后一次性更新
                    avg_loss = total_loss / count
                    accelerator.backward(avg_loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                    update_step += 1

            # log to wandb
            if accelerator.is_main_process and step_idx % args.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": sched.get_last_lr()[-1],
                    "train/sec": (time.time()-start)/args.logging_steps
                }, step=update_step)
                start = time.time()

    # save the model at the end
    if accelerator.is_main_process:
        make_output_dir_and_broadcast(args,accelerator, gb)
        ckpt = Path(args.output_dir) / f"checkpoint-epoch{args.epochs}"
        ckpt.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            ckpt,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=False
        )
        tokenizer.save_pretrained(ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--rare_ids", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--ref_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--mu", type=int, default=16)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cfg_scale", type=float, default=0.)
    parser.add_argument("--remasking", type=str, default='random')
    parser.add_argument("--train_mode", type=str, choices=["Normal", "MirrorMask"], default="Normal")
    parser.add_argument("--IS", action="store_true")
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--EMA", action="store_true")
    parser.add_argument("--EMA_Optimized", action="store_true")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--baseline_lr", type=float, default=0.01)
    parser.add_argument("--loss_max", type=float, default=10)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO‐style clipping ε")
    parser.add_argument("--beta", type=float, default=0.01, help="weight for KL‐divergence regularization")

    args = parser.parse_args()
    if args.rare_ids is None:
        args.rare_ids = "59,795,32289,90,28504,7684,92" if args.task == "hitab" else "2262"
    args.rare_ids = [int(x) for x in args.rare_ids.split(",")]  
    if args.train_data is None:
        if args.task == "hitab": args.train_data = "hitab_reasoning_sft_str_unigrpo.jsonl"
        elif args.task == "gsm8k": args.train_data = "gsm8k_train_unigrpo.jsonl"
    if args.gen_length is None:
        if args.task == "hitab": args.gen_length = 512
        elif args.task == "gsm8k": args.gen_length = 128
    if args.steps is None:
        if args.task == "hitab": args.steps = 256
        elif args.task == "gsm8k": args.steps = 128
    if args.block_length is None:
        if args.task == "hitab": args.block_length = 16
        elif args.task == "gsm8k": args.block_length = 32

    unigrpo_train(args)
