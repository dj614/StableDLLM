#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

if "--china" in sys.argv:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import argparse, json, math, time
from typing import Callable, Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

import wandb
import random, numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm.auto import tqdm

# --- compatibility for older NumPy (no _np_trapz) ---
try:
    _np_trapz = np.trapezoid   # newer NumPy
except AttributeError:
    _np_trapz = np.trapz       # older NumPy fallback

MASK_TOKEN_ID = 126336

def set_random_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_output_dir_and_broadcast(args, accelerator, gb):
    if accelerator.is_main_process and args.output_dir is None:
        args.output_dir = (
            f"/root/workspace/checkpoints/"
            f"seed{args.seed}_{args.model}_{args.task}_"
            f"ppots_mirror_plus"
        )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =========================
# Dataset & Collate
# =========================

class LLaDADataset(Dataset):
    """
    读取 JSONL，每行至少包含:
      - "input_ids": List[int]
      - "prompt_length": int
    padding / attention_mask / labels 在 collate_fn 中统一构造
    """
    def __init__(self, jsonl_path: str, max_len: int):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf8") as f:
            for ln in f:
                ex = json.loads(ln)
                if len(ex["input_ids"]) <= max_len:
                    self.samples.append(ex)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "prompt_length": int(ex["prompt_length"]),
        }


def collate_fn(batch, pad_id: int):
    """
    统一构造:
      - input_ids: [B, L_max]
      - attention_mask: [B, L_max]  (1=有效token, 0=pad)
      - labels: [B, L_max]
        * 前 prompt_length 个位置 -> -100
        * pad 部分 -> -100
        * 其余 continuation + non-pad -> 对应 token id
    """
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids_list = []
    attention_masks = []
    labels_list = []

    for x in batch:
        ids = x["input_ids"]
        L = ids.size(0)
        prompt_len = int(x["prompt_length"])

        pad_len = max_len - L
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
            ids_padded = torch.cat([ids, pad_tensor], dim=0)
            attn = torch.cat(
                [torch.ones(L, dtype=torch.long),
                 torch.zeros(pad_len, dtype=torch.long)],
                dim=0
            )
        else:
            ids_padded = ids
            attn = torch.ones(L, dtype=torch.long)

        labels = ids_padded.clone()

        # 前 prompt_len 视为 prompt，不计 loss
        eff_prompt_len = min(prompt_len, max_len)
        labels[:eff_prompt_len] = -100

        # pad 部分也不计 loss
        if pad_len > 0:
            labels[L:] = -100

        input_ids_list.append(ids_padded)
        attention_masks.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels_list),
    }


# =========================
# Forward noise process
# =========================

def forward_process(
    batch_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_mode: str,
    fixed_t: torch.Tensor,
    iw_t: torch.Tensor,
    eps: float = 1e-3,
):
    """
    统一使用:
      - labels != -100 作为可预测 token（eligible）
      - labels == -100 或 pad( attention_mask=0 ) 的地方一律不加噪
    """
    B, L = batch_ids.shape
    device = batch_ids.device

    eligible_pos = (labels != -100) & (attention_mask == 1)  # [B,L]

    # mask 概率：p_mask
    p_mask_base = (1 - eps) * fixed_t[:, None] + eps
    p_mask = p_mask_base.expand(-1, L)

    # 采样噪声
    u = torch.rand_like(p_mask)

    noisy1 = batch_ids.clone()
    mask_here1 = (u < p_mask) & eligible_pos
    noisy1[mask_here1] = MASK_TOKEN_ID

    noisy2 = None
    if train_mode in ["MirrorMask", "mirror_plus"]:
        noisy2 = batch_ids.clone()
        mask_here2 = (u > (1 - p_mask)) & eligible_pos
        noisy2[mask_here2] = MASK_TOKEN_ID

    eligible_count = eligible_pos.sum(dim=1).clamp(min=1)
    return p_mask, iw_t, noisy1, noisy2, eligible_count


# =========================
# Loss for training / eval
# =========================

def batched_loss_for_backpropagate(
    ids: torch.Tensor,
    noisy: torch.Tensor,
    model,
    p_mask: torch.Tensor,
    iw_t: torch.Tensor,
    eligible: torch.Tensor,
    train: bool = True,
    debug: Optional[dict] = None,
    pad_id: int = 0,
    return_sample_losses: bool = False,
    loss_clip_max: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    """
    ids/noisy: [B, L]
    p_mask:    [B, L]
    iw_t:      [B]
    eligible:  [B]  (每行可预测 token 数)
    """
    device = ids.device
    with torch.set_grad_enabled(train):
        mask_tok = (noisy == MASK_TOKEN_ID)
        rows_with_mask = mask_tok.any(dim=1)
        if not mask_tok.any():
            if return_sample_losses:
                zeros = torch.zeros(ids.size(0), device=device, dtype=torch.float32)
                return (torch.zeros((), device=device, dtype=torch.float32), zeros)
            return torch.zeros((), device=device, dtype=torch.float32)

        labels = ids.masked_fill(~mask_tok, -100)
        _am = attn_mask if attn_mask is not None else (noisy != pad_id).long()
        logits = model(noisy, attention_mask=_am).logits

        ce_all = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none"
        ).view_as(labels)

        ce_tok = ce_all[mask_tok]
        if ce_tok.numel() == 0:
            if return_sample_losses:
                zeros = torch.zeros(ids.size(0), device=device, dtype=torch.float32)
                return (torch.zeros((), device=device, dtype=torch.float32), zeros)
            return torch.zeros((), device=device, dtype=torch.float32)

        p_m = p_mask[mask_tok]
        row_idx = mask_tok.nonzero(as_tuple=True)[0]
        iw_samp = iw_t[row_idx]

        ce_weight = (ce_tok / p_m.clamp_min(1e-12)) * iw_samp

        loss_b = torch.zeros(ids.size(0), device=device, dtype=ce_weight.dtype)
        loss_b.scatter_add_(0, row_idx, ce_weight)

        denom = eligible.clamp(min=1).to(loss_b.dtype)

        if rows_with_mask.any():
            loss_scalar = (loss_b[rows_with_mask] / denom[rows_with_mask]).mean()
        else:
            loss_scalar = torch.zeros((), device=device, dtype=loss_b.dtype)

        # 样本级 loss（无 IS 权重）
        ce_no_tiw = (ce_tok / p_m.clamp_min(1e-12))
        loss_b_no_tiw = torch.zeros(ids.size(0), device=device, dtype=ce_no_tiw.dtype)
        loss_b_no_tiw.scatter_add_(0, row_idx, ce_no_tiw)
        per_sample_L = (loss_b_no_tiw / denom).to(torch.float32)
        if loss_clip_max is not None and math.isfinite(loss_clip_max):
            per_sample_L = per_sample_L.clamp_max(loss_clip_max)

        if debug is not None:
            debug["count/masked_tokens"] = float(mask_tok.sum().item())
            for name, x in [("ce_all", ce_all), ("loss_b", loss_b)]:
                debug[f"nan/{name}"] = float(torch.isnan(x).float().mean().item())
                debug[f"inf/{name}"] = float(torch.isinf(x).float().mean().item())

    if return_sample_losses:
        # 返回 scalar loss + 每个样本的 loss（按 eligible 平均）
        return (loss_scalar if train else loss_scalar.detach()), per_sample_L

    return loss_scalar if train else loss_scalar.detach()


# =========================
# Utility: sample from jsonl
# =========================

def sample_multi_from_jsonl(path: str, k: int) -> List[dict]:
    reservoir: List[Optional[str]] = [None] * k
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if n < k:
                reservoir[n] = line
            else:
                j = random.randrange(n + 1)
                if j < k:
                    reservoir[j] = line
            n += 1
    if n == 0:
        raise RuntimeError("Empty JSONL file")
    out = [json.loads(s) for s in reservoir if s is not None]
    if len(out) < k:
        return out
    return out


# =========================
# 多噪声 t-grid 评估 (for IS-on-t)
# =========================

@torch.no_grad()
def batched_losses_for_many_noisy(
    args,
    model,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    t_grid: torch.Tensor,
    k_per_t: int,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ids: [1, L]
    attention_mask: [1, L]
    labels: [1, L]
    t_grid: [T]
    """
    device_ = ids.device
    B1, L = ids.shape
    assert B1 == 1
    T = t_grid.numel()

    prompt_pos = (labels == -100) & (attention_mask == 1)
    eligible_1L = (labels != -100) & (attention_mask == 1)
    num_elig = int(eligible_1L.sum().item())
    if num_elig == 0:
        mean_t = np.full(T, np.nan, dtype=float)
        var_t = np.full(T, np.nan, dtype=float)
        return mean_t, var_t

    p_mask_T1 = ((1 - eps) * t_grid[:, None] + eps).to(device_)
    p_mask_TL = p_mask_T1.repeat(1, L)
    p_mask_TKL = p_mask_TL.repeat_interleave(k_per_t, dim=0)
    eligible_TKL = eligible_1L.repeat(T * k_per_t, 1)

    bern = torch.rand_like(p_mask_TKL)
    mask_here = (bern < p_mask_TKL) & eligible_TKL

    base = ids.repeat(T * k_per_t, 1)
    noisy = base.clone()
    noisy[mask_here] = MASK_TOKEN_ID

    used_p = p_mask_TKL.masked_fill(~eligible_TKL, 1.0)

    tokens_per_row = L
    rows_per_chunk = max(1, int(args.max_tokens_per_forward // max(1, tokens_per_row)))
    n_rows = noisy.size(0)

    losses_row = torch.zeros((n_rows,), device=device_, dtype=torch.float32)
    prompt_TKL = prompt_pos.repeat(T * k_per_t, 1)

    for i in range(0, n_rows, rows_per_chunk):
        j = min(n_rows, i + rows_per_chunk)
        noisy_chunk = noisy[i:j].clone()
        noisy_chunk[prompt_TKL[i:j]] = base[i:j][prompt_TKL[i:j]]
        am_chunk = attention_mask.repeat(j - i, 1)
        logits = model(noisy_chunk, attention_mask=am_chunk).logits  # [R, L, V]
        mask_chunk = mask_here[i:j]                                  # [R, L]

        if mask_chunk.any():
            # 展平 logits/base/mask
            flat_logits = logits.view(-1, logits.size(-1))           # [R*L, V]
            flat_base   = base[i:j].reshape(-1)                      # [R*L]
            flat_mask   = mask_chunk.view(-1)                        # [R*L]
            flat_used_p = used_p[i:j].reshape(-1)                    # [R*L]

            # 只取 masked token 计算 CE
            ce = F.cross_entropy(
                flat_logits[flat_mask],      # [num_masked, V]
                flat_base[flat_mask],        # [num_masked]
                reduction="none"
            )
            used_p_chunk = flat_used_p[flat_mask]                    # [num_masked]
            ce = ce / used_p_chunk                                   # importance correction

            row_idx = mask_chunk.nonzero(as_tuple=True)[0]
            loss_b = torch.zeros(j - i, device=device_)
            loss_b.scatter_add_(0, row_idx, ce)
            denom = eligible_TKL[i:j].sum(dim=1).clamp(min=1).float()

            rows_with = mask_chunk.any(dim=1)
            if rows_with.any():
                loss_vec = (loss_b[rows_with] / denom[rows_with]).clamp_max(args.loss_max)
                chunk = losses_row[i:j]
                chunk[rows_with] = loss_vec
                losses_row[i:j] = chunk

    losses_row = losses_row.view(T, k_per_t).float().cpu().numpy()

    vals = losses_row
    count = np.full(T, k_per_t, dtype=int)
    sum_ = vals.sum(axis=1)
    sum2 = (vals * vals).sum(axis=1)

    mean_per_t = np.full(T, np.nan, dtype=float)
    mask_mean = count > 0
    mean_per_t[mask_mean] = (sum_[mask_mean] / count[mask_mean]).astype(float)

    var_per_t = np.full(T, np.nan, dtype=float)
    mask_var = count >= 2
    var_per_t[mask_var] = (
        (sum2[mask_var] - (sum_[mask_var] ** 2) / count[mask_var]) / (count[mask_var] - 1)
    ).astype(float)

    return mean_per_t, var_per_t


def evaluate_over_x0(
    model,
    NUM_SAMPLES_X0: int,
    NUM_SAMPLES_T: int,
    NUM_SAMPLES_XT: int,
    args,
    device,
    pad_id: int,
):
    """
    从 train_data_path 中抽样若干 x0，
    使用统一的 schema 构造:
      - input_ids
      - attention_mask
      - labels (由 prompt_length 构造)
    然后跑 t-grid noisy loss -> wi, gi, vi 等。
    """
    t_values = torch.linspace(
        1 / NUM_SAMPLES_T,
        1 - 1 / NUM_SAMPLES_T,
        NUM_SAMPLES_T - 1,
        device=device
    )  # [T]

    mean_stack, var_stack = [], []

    x0_list = sample_multi_from_jsonl(args.train_data_path, NUM_SAMPLES_X0)

    for x0 in x0_list:
        ids_list = x0["input_ids"]
        prompt_len = int(x0.get("prompt_length", len(ids_list)))

        ids = torch.tensor(ids_list, dtype=torch.long, device=device).unsqueeze(0)  # [1,L]
        L = ids.size(1)

        # attention_mask: 全为 1（没有额外 pad）
        am = torch.ones(1, L, dtype=torch.long, device=device)

        # labels: prompt 部分 -100，其余为 token；无 pad
        labels = ids.clone()
        eff_pl = min(prompt_len, L)
        labels[:, :eff_pl] = -100

        mean_t, var_t = batched_losses_for_many_noisy(
            args, model, ids, am, labels, t_values, k_per_t=NUM_SAMPLES_XT, eps=1e-3
        )
        mean_stack.append(mean_t)
        var_stack.append(var_t)

    mean_stack = np.stack(mean_stack, axis=0)
    var_stack = np.stack(var_stack, axis=0)
    avg_mean = np.nanmean(mean_stack, axis=0)
    avg_var = np.nanmean(var_stack, axis=0)
    result = np.sqrt(avg_var + avg_mean ** 2)
    return result, avg_mean, avg_var, t_values.detach().cpu().numpy()


# =========================
# 拟合 p(t)
# =========================

@dataclass
class FitResult:
    mode: str
    params: Dict[str, float]
    p_func: Callable[[np.ndarray], np.ndarray]
    t_values: np.ndarray
    p_on_grid: np.ndarray


def _safe_to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x, dtype=float).reshape(-1)
    return x


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def _kl_trapz(p: np.ndarray, q: np.ndarray, t: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    log_ratio = np.log(p) - np.log(q)
    return float(_np_trapz(p * log_ratio, t))


def _fit_powerU_w_KL(t: np.ndarray, w_raw: np.ndarray,
                     args, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.clip(np.asarray(t, float), 1e-6, 1 - 1e-6)
    w_raw = np.maximum(np.asarray(w_raw, float), 0.0)

    Zw = _np_trapz(w_raw, t)
    if not np.isfinite(Zw) or Zw <= 0:
        raise ValueError("wi has non-positive integral on t grid.")
    w_pdf = w_raw / Zw

    def unpack(z):
        a = _softplus(z[0]) + 1e-9
        b = _softplus(z[1]) + 1e-9
        A = _softplus(z[2]) + 1e-9
        r = 1.0 + _softplus(z[3])
        q = 1.0 + _softplus(z[4])
        k = _softplus(z[5])
        m = 1.0 + _softplus(z[6])
        return a, b, A, r, q, k, m

    def build_pdf(z):
        a, b, A, r, q, k, m = unpack(z)
        vhat = a * (t ** r) + b * ((1 - t) ** q)
        ghat = A * np.exp(k * (t ** m))
        pdf = np.sqrt(vhat + ghat ** 2)
        Z = _np_trapz(pdf, t)
        if not np.isfinite(Z) or Z <= 0:
            return None, None, None, None
        return pdf / Z, (a, b, A, r, q, k, m), ghat, vhat

    def loss(z):
        p_pdf, _, _, _ = build_pdf(z)
        if p_pdf is None:
            return 1e6
        return _kl_trapz(w_pdf, p_pdf, t)

    best = (None, np.inf)
    for _ in range(args.n_starts):
        z0 = np.array([
            -0.3, 0.4, -1.0, 0.3, 0.2, -0.2, 0.2
        ], dtype=float) + rng.normal(scale=0.6, size=7)
        res = minimize(loss, z0, method="L-BFGS-B")
        if np.isfinite(res.fun) and res.fun < best[1]:
            best = (res.x, res.fun)

    z_star = best[0]
    p_pdf, params_tuple, g_on_grid, v_on_grid = build_pdf(z_star)
    a, b, A, r, q, k, m = params_tuple
    params = {
        "a": float(a), "b": float(b), "A": float(A),
        "r": float(r), "q": float(q), "k": float(k), "m": float(m)
    }

    def g_func(x):
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return A * np.exp(k * (x ** m))

    def v_func(x):
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return a * (x ** r) + b * ((1 - x) ** q)

    return params, p_pdf, g_on_grid, v_on_grid, g_func, v_func


def fit_p_of_t(
    wi, gi, vi, t_values,
    args,
    fit_mode,
    plot: bool = True
) -> FitResult:
    t = _safe_to_numpy(t_values)
    w_raw = _safe_to_numpy(wi)
    assert t.shape == w_raw.shape, "Input shapes must match."

    valid = np.isfinite(t) & np.isfinite(w_raw)
    t, w_raw = t[valid], w_raw[valid]
    assert t.size >= 5, "Too few valid t points to fit p(t)."

    t_clip = np.clip(t, 1e-6, 1 - 1e-6)

    if fit_mode == "insight_powerU":
        params_fit, p_pdf_grid, g_grid, v_grid, g_func, v_func = _fit_powerU_w_KL(t_clip, w_raw, args)
        family = "insight_powerU"
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    lam = float(args.mix_uniform)
    lam = 0.0 if lam < 0 else (0.999999 if lam >= 1.0 else lam)
    p_grid = (1.0 - lam) * p_pdf_grid + lam * 1.0
    Z_grid = _np_trapz(p_grid, t_clip)
    p_grid = p_grid / (Z_grid if Z_grid > 0 else 1.0)

    def p_func0(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return np.interp(x, t_clip, p_grid)

    if plot:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) wᵢ 与 fitted p(t)
        plt.figure()
        w_vis = np.maximum(_safe_to_numpy(wi), 0.0)
        Zw = _np_trapz(w_vis, t_clip)
        if Zw > 0 and np.isfinite(Zw):
            w_vis = w_vis / Zw
        plt.plot(t_clip, w_vis, marker='o', label="wᵢ")
        plt.plot(t_clip, p_grid, label=f"fitted p(t) [{family}]")
        plt.title("wᵢ and fitted p(t) vs t")
        plt.xlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"w(t)_p(t)_vs_t_{family}.png", dpi=180)
        plt.close()

        # 2) gᵢ vs t + fitted g(t)
        gi_vis = _safe_to_numpy(gi)
        plt.figure()
        plt.plot(t_clip, gi_vis, marker='o', label="gᵢ")
        plt.plot(t_clip, g_grid, label="fitted g(t)")
        plt.title("gᵢ vs t")
        plt.xlabel("t")
        plt.ylabel("gᵢ / g(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"gi_vs_t_{family}.png", dpi=180)
        plt.close()

        # 3) vᵢ vs t + fitted v(t)
        vi_vis = _safe_to_numpy(vi)
        plt.figure()
        plt.plot(t_clip, vi_vis, marker='o', label="vᵢ")
        plt.plot(t_clip, v_grid, label="fitted v(t)")
        plt.title("vᵢ vs t")
        plt.xlabel("t")
        plt.ylabel("vᵢ / v(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"vi_vs_t_{family}.png", dpi=180)
        plt.close()

    return FitResult(
        mode=family,
        params={**params_fit, "mix_uniform": lam},
        p_func=p_func0,
        t_values=t_clip,
        p_on_grid=p_grid,
    )


# =========================
# Train loop
# =========================

def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        gradient_accumulation_steps=args.grad_accum
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
        trust_remote_code=True
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    g_data = torch.Generator().manual_seed(args.seed)

    # dataset & dataloader
    ds = LLaDADataset(args.train_data_path, args.max_len)
    train_n = int(len(ds) * args.train_ratio)
    eval_n = len(ds) - train_n
    train_ds, eval_ds = torch.utils.data.random_split(ds, [train_n, eval_n], generator=g_data)
    if eval_n == 0:
        accelerator.print("⚠️ args.train_ratio == 1.0；跳过 eval")
        do_evaluation = False
    else:
        do_evaluation = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id)
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, pad_id)
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if getattr(model.config, "is_decoder", None):
        model.config.is_decoder = False

    # optimizer
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr, weight_decay=0.1)

    # prepare via accelerator
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    # scheduler
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    # wandb init
    if accelerator.is_main_process:
        wandb.init(project=f"llada_mirror_plus_{args.task}", config=vars(args))
    accelerator.print(
        f"◎ 全局 batch = {args.batch_size_per_gpu} × grad_accum {args.grad_accum} × "
        f"processes {accelerator.num_processes} = {gb}"
    )
    accelerator.print(f"★ 总数据量 训练 {train_n}，评估 {eval_n}，步骤 {total_steps}")

    # output dir
    output_dir = make_output_dir_and_broadcast(args, accelerator, gb)

    # importance sampling over t
    if args.IS_on_t:
        model.eval()
        start_time = time.time()
        wi, gi, vi, t_values = evaluate_over_x0(
            model=model,
            NUM_SAMPLES_X0=args.num_samples_x0,
            NUM_SAMPLES_T=args.num_samples_t,
            NUM_SAMPLES_XT=args.num_samples_xt,
            args=args,
            device=device,
            pad_id=pad_id
        )
        res = fit_p_of_t(
            wi=wi, gi=gi, vi=vi, t_values=t_values,
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

        t_grid = torch.tensor(res.t_values, device=device, dtype=torch.float32)  # [N]
        p_grid = torch.tensor(res.p_on_grid, device=device, dtype=torch.float32)  # [N]
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
            return t_s, (1.0 / p_s.clamp(min=1e-12)).detach()
    else:
        sample_t_from_p = None

    # eval loop
    def evaluate(step_idx: int):
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        pbar = tqdm(
            eval_loader,
            desc=f"Eval@{step_idx}",
            disable=not accelerator.is_main_process
        )
        with torch.no_grad():
            for batch in pbar:
                eids = batch["input_ids"].to(device)
                eam = batch["attention_mask"].to(device)
                elbls = batch["labels"].to(device)

                fixed_t, iw_t = torch.rand(eids.shape[0], device=device), torch.ones(eids.shape[0], device=device)
                if args.IS_on_t:
                    fixed_t, iw_t = sample_t_from_p(eids.shape[0])

                p_mask, iw_t, noisy1, _, eligible = forward_process(
                    eids, eam, elbls, args.train_mode,
                    fixed_t=fixed_t, iw_t=iw_t
                )
                has_mask_eval = (noisy1 == MASK_TOKEN_ID).any()
                if not has_mask_eval:
                    continue

                loss = batched_loss_for_backpropagate(
                    eids, noisy1, model, p_mask, iw_t, eligible,
                    train=False, pad_id=pad_id, attn_mask=eam
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
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process
        )
        for step_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            fixed_t, iw_t = torch.rand(ids.shape[0], device=device), torch.ones(ids.shape[0], device=device)
            if args.IS_on_t:
                fixed_t, iw_t = sample_t_from_p(ids.shape[0])

            p_mask, iw_t, noisy1, noisy2, eligible = forward_process(
                ids, am, lbls, args.train_mode, fixed_t=fixed_t, iw_t=iw_t
            )
            has_mask1 = (noisy1 == MASK_TOKEN_ID).any()
            has_mask2 = (noisy2 == MASK_TOKEN_ID).any() if noisy2 is not None else False

            if args.train_mode == "Normal":
                if not has_mask1:
                    continue
            elif args.train_mode == "MirrorMask":
                if not (has_mask1 or has_mask2):
                    continue
            elif args.train_mode == "mirror_plus":
                if not (has_mask1 or has_mask2):
                    continue

            if args.train_mode == "Normal":
                # 只 noisy1
                loss = batched_loss_for_backpropagate(
                    ids, noisy1, model, p_mask, iw_t, eligible,
                    train=True, debug={}, pad_id=pad_id,
                    return_sample_losses=False,
                    loss_clip_max=getattr(args, "loss_max", None),
                    attn_mask=am
                )

            elif args.train_mode == "MirrorMask":
                # 始终 noisy1 + noisy2 取平均
                dbg1, dbg2 = {}, {}
                loss1 = batched_loss_for_backpropagate(
                    ids, noisy1, model, p_mask, iw_t, eligible,
                    train=True, debug=dbg1, pad_id=pad_id,
                    return_sample_losses=False,
                    loss_clip_max=getattr(args, "loss_max", None),
                    attn_mask=am
                )
                loss2 = batched_loss_for_backpropagate(
                    ids, noisy2, model, p_mask, iw_t, eligible,
                    train=True, debug=dbg2, pad_id=pad_id,
                    return_sample_losses=False,
                    loss_clip_max=getattr(args, "loss_max", None),
                    attn_mask=am
                )
                loss = 0.5 * (loss1 + loss2)

            elif args.train_mode == "mirror_plus":
                # per-sample 判断是否使用两个方向
                use_two_mask = (fixed_t < 0.9)          # [B] bool

                # 1) noisy1 的 per-sample loss：所有样本都要算
                _, loss1_per = batched_loss_for_backpropagate(
                    ids, noisy1, model, p_mask, iw_t, eligible,
                    train=True, debug={}, pad_id=pad_id,
                    return_sample_losses=True,
                    loss_clip_max=getattr(args, "loss_max", None),
                    attn_mask=am
                )  # [B]

                # 2) noisy2 的 per-sample loss：只对 t_i < 0.9 的样本算
                loss2_per = torch.zeros_like(loss1_per)
                if noisy2 is not None and use_two_mask.any():
                    # 需要用两个 noisy 的样本下标
                    idx_two = use_two_mask.nonzero(as_tuple=True)[0]  # [B_two]

                    _, loss2_sub = batched_loss_for_backpropagate(
                        ids[idx_two], noisy2[idx_two], model,
                        p_mask[idx_two], iw_t[idx_two], eligible[idx_two],
                        train=True, debug={}, pad_id=pad_id,
                        return_sample_losses=True,
                        loss_clip_max=getattr(args, "loss_max", None),
                        attn_mask=am[idx_two]
                    )  # [B_two]

                    # 把子 batch 的 loss 填回到对应位置
                    loss2_per[idx_two] = loss2_sub

                # 3) 按样本合成最终 loss_i
                #    如果 t_i < 0.9：平均 noisy1 & noisy2
                #    如果 t_i >= 0.9：只用 noisy1
                final_per_sample_loss = torch.where(
                    use_two_mask,
                    0.5 * (loss1_per + loss2_per),  # t < 0.9
                    loss1_per                       # t >= 0.9
                )

                # batch loss = 所有样本 loss 的平均
                loss = final_per_sample_loss.mean()

            accelerator.backward(loss)

            if accelerator.is_main_process and update_step % args.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[-1],
                    "train/sec": (time.time() - start) / args.logging_steps
                }, step=update_step)
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
            safe_serialization=True
        )
        tokenizer.save_pretrained(ckpt)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

    return output_dir, device


# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--china", action="store_true", help="是否使用国内镜像 hf-mirror.com")
    ap.add_argument("--seed", type=int, required=True, help="全局随机种子")
    ap.add_argument("--task", type=str, choices=["openscience", "gsm8k", "hitab"], required=True)
    ap.add_argument("--model", type=str, choices=["llada"], default="llada")
    ap.add_argument("--train_mode", type=str, choices=["Normal", "MirrorMask", "mirror_plus"], default="Normal")
    ap.add_argument("--IS_on_t", action="store_true")
    ap.add_argument("--p_model", type=str, choices=["insight_powerU"], default="insight_powerU", help="选择 p(t) 的参数化族")
    ap.add_argument("--max_tokens_per_forward", type=int, default=60000)
    ap.add_argument("--max_is_weight", type=float, default=1e6)
    ap.add_argument("--n_starts", type=int, default=20)
    ap.add_argument("--num_samples_x0", type=int, default=10)
    ap.add_argument("--num_samples_t", type=int, default=70)
    ap.add_argument("--num_samples_xt", type=int, default=10)
    ap.add_argument("--mix_uniform", type=float, default=0.0)
    ap.add_argument("--train_data_path", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--loss_max", type=float, default=10.0, help="样本级 loss 的最大值；None 表示不裁剪")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--batch_size_per_gpu", type=int, required=True)
    ap.add_argument("--grad_accum", type=int, required=True)
    ap.add_argument("--lr_scheduler_type", type=str,
                    choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                    default="linear")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--eval_strategy", type=str, choices=["epoch", "steps", "no"], default="epoch")
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_strategy", type=str, choices=["epoch", "steps", "last"], default="last")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--logging_steps", type=int, default=5)

    args = ap.parse_args()

    if args.eval_strategy is None:
        args.eval_strategy = args.save_strategy
    if args.eval_steps is None:
        args.eval_steps = args.save_steps
    
    if args.train_data_path is None:
        args.train_data_path = f"/root/workspace/data/train/{args.task}.jsonl"
    return args


if __name__ == "__main__":
    args = parse_args()
    out_dir, device = train(args)
