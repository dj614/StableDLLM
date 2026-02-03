"""Masked-token cross entropy loss with p_mask correction.

This is a direct extraction from the original `rebuttal.py`.

Key semantics:
  - Only masked tokens (noisy == MASK_TOKEN_ID) contribute to loss.
  - Per-token CE is divided by p_mask to correct the masking probability.
  - Optional per-sample losses can be returned for mirror_plus mixing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from llada_plus.diffusion.masking import MASK_TOKEN_ID


def batched_loss_for_backpropagate(
    ids: torch.Tensor,
    noisy: torch.Tensor,
    model,
    p_mask: torch.Tensor,
    iw_t: torch.Tensor,
    eligible: torch.Tensor,
    *,
    train: bool = True,
    debug: Optional[dict] = None,
    pad_id: int = 0,
    return_sample_losses: bool = False,
    loss_clip_max: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the masked-token loss.

    Args:
        ids/noisy: [B, L]
        p_mask: [B, L]
        iw_t: [B]
        eligible: [B] number of eligible tokens per sample
        train: enables grad
        return_sample_losses: if True, also returns per-sample loss [B]
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
            reduction="none",
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

        # Sample-level loss (no iw_t) for mirror_plus mixing.
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
        return (loss_scalar if train else loss_scalar.detach()), per_sample_L

    return loss_scalar if train else loss_scalar.detach()
