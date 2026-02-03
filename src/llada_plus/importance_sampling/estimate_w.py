"""Estimate w(t) (and g(t), v(t) proxies) from noisy-loss statistics.

This module is used when `--IS_on_t` is enabled.

It is a direct extraction from the original `LLaDA/rebuttal.py`:
  - batched_losses_for_many_noisy
  - evaluate_over_x0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from llada_plus.data import sample_multi_from_jsonl
from llada_plus.diffusion.masking import MASK_TOKEN_ID


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
    """For a single x0, compute mean/var of noisy-loss for each t in t_grid.

    Args:
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
            flat_logits = logits.view(-1, logits.size(-1))
            flat_base = base[i:j].reshape(-1)
            flat_mask = mask_chunk.view(-1)
            flat_used_p = used_p[i:j].reshape(-1)

            ce = F.cross_entropy(
                flat_logits[flat_mask],
                flat_base[flat_mask],
                reduction="none",
            )
            used_p_chunk = flat_used_p[flat_mask]
            ce = ce / used_p_chunk

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
    *,
    model,
    NUM_SAMPLES_X0: int,
    NUM_SAMPLES_T: int,
    NUM_SAMPLES_XT: int,
    args,
    device,
    pad_id: int,
):
    """Estimate w(t), g(t), v(t) over a t-grid by sampling x0 from train JSONL."""
    t_values = torch.linspace(
        1 / NUM_SAMPLES_T,
        1 - 1 / NUM_SAMPLES_T,
        NUM_SAMPLES_T - 1,
        device=device,
    )

    mean_stack, var_stack = [], []
    x0_list = sample_multi_from_jsonl(args.train_data_path, NUM_SAMPLES_X0)

    for x0 in x0_list:
        ids_list = x0["input_ids"]
        prompt_len = int(x0.get("prompt_length", len(ids_list)))

        ids = torch.tensor(ids_list, dtype=torch.long, device=device).unsqueeze(0)
        L = ids.size(1)

        am = torch.ones(1, L, dtype=torch.long, device=device)

        labels = ids.clone()
        eff_pl = min(prompt_len, L)
        labels[:, :eff_pl] = -100

        mean_t, var_t = batched_losses_for_many_noisy(
            args,
            model,
            ids,
            am,
            labels,
            t_values,
            k_per_t=NUM_SAMPLES_XT,
            eps=1e-3,
        )
        mean_stack.append(mean_t)
        var_stack.append(var_t)

    mean_stack = np.stack(mean_stack, axis=0)
    var_stack = np.stack(var_stack, axis=0)
    avg_mean = np.nanmean(mean_stack, axis=0)
    avg_var = np.nanmean(var_stack, axis=0)
    wi = np.sqrt(avg_var + avg_mean**2)
    return wi, avg_mean, avg_var, t_values.detach().cpu().numpy()
