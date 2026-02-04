"""Masked-token cross entropy loss with p_mask correction.

This is a direct extraction from the original `rebuttal.py`.

Key semantics:
  - Only masked tokens (noisy == MASK_TOKEN_ID) contribute to loss.
  - Per-token CE is divided by p_mask to correct the masking probability.
  - Optional importance-sampling weight iw_t(t) is applied at the sample level.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from llada.plus.diffusion.masking import MASK_TOKEN_ID


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
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the masked-token loss.

    Args:
        ids/noisy: [B, L]
        p_mask: [B, L]
        iw_t: [B]
        eligible: [B] number of eligible tokens per sample
        train: enables grad
        debug: optional dict to fill with NaN/Inf stats
        pad_id: padding token id
        attn_mask: optional attention mask overriding (noisy != pad_id)
    """
    device = ids.device
    with torch.set_grad_enabled(train):
        mask_tok = (noisy == MASK_TOKEN_ID)
        rows_with_mask = mask_tok.any(dim=1)

        if not mask_tok.any():
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
            return torch.zeros((), device=device, dtype=torch.float32)

        p_m = p_mask[mask_tok]
        row_idx = mask_tok.nonzero(as_tuple=True)[0]
        iw_samp = iw_t[row_idx]

        # Token loss corrected by p_mask, then re-weighted by iw_t for importance sampling over t.
        ce_weight = (ce_tok / p_m.clamp_min(1e-12)) * iw_samp

        loss_b = torch.zeros(ids.size(0), device=device, dtype=ce_weight.dtype)
        loss_b.scatter_add_(0, row_idx, ce_weight)

        denom = eligible.clamp(min=1).to(loss_b.dtype)

        if rows_with_mask.any():
            loss_scalar = (loss_b[rows_with_mask] / denom[rows_with_mask]).mean()
        else:
            loss_scalar = torch.zeros((), device=device, dtype=loss_b.dtype)

        if debug is not None:
            debug["count/masked_tokens"] = float(mask_tok.sum().item())
            debug["nan/ce_all"] = float(torch.isnan(ce_all).float().mean().item())
            debug["inf/ce_all"] = float(torch.isinf(ce_all).float().mean().item())
            debug["nan/loss_b"] = float(torch.isnan(loss_b).float().mean().item())
            debug["inf/loss_b"] = float(torch.isinf(loss_b).float().mean().item())

    return loss_scalar if train else loss_scalar.detach()
