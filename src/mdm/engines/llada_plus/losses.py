"""Masked-token loss helpers for LLaDA+ training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .diffusion import MASK_TOKEN_ID


def batched_loss_for_backpropagate(
    input_ids: torch.Tensor,
    noisy_ids: torch.Tensor,
    model,
    p_mask: torch.Tensor,
    iw_t: torch.Tensor,
    eligible: torch.Tensor,
    *,
    train: bool,
    pad_id: int,
    attn_mask: Optional[torch.Tensor] = None,
    debug: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute masked-token cross-entropy loss with diffusion weighting."""
    logits = model(noisy_ids, attention_mask=attn_mask).logits  # [B, L, V]

    mask_tok = (noisy_ids == MASK_TOKEN_ID) & eligible
    if not mask_tok.any():
        return torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)

    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = input_ids.view(-1)
    flat_mask = mask_tok.view(-1)
    flat_p = p_mask.view(-1).clamp(min=1e-12)

    ce = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], reduction="none")
    ce = ce / flat_p[flat_mask]

    row_idx = mask_tok.nonzero(as_tuple=True)[0]
    loss_per_row = torch.zeros(input_ids.size(0), device=input_ids.device)
    loss_per_row.scatter_add_(0, row_idx, ce)

    denom = eligible.sum(dim=1).clamp(min=1).float()
    loss_per_row = loss_per_row / denom

    iw = iw_t.to(device=input_ids.device, dtype=torch.float32)
    if iw.ndim == 2 and iw.size(1) == 1:
        iw = iw.view(-1)
    loss_per_row = loss_per_row * iw

    if debug is not None:
        debug["masked_frac"] = float(mask_tok.sum().item()) / float(mask_tok.numel())
        debug["avg_p_mask"] = float(p_mask.mean().item())

    return loss_per_row.mean()
