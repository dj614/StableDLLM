"""Masking diffusion helpers for LLaDA+ training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

MASK_TOKEN_ID = 126336


def forward_process(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    train_mode: str = "Normal",
    *,
    fixed_t: Optional[torch.Tensor] = None,
    iw_t: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Apply the LLaDA masking diffusion process.

    Returns:
      p_mask: per-token masking probabilities [B, L]
      iw_t: per-sample importance weight [B]
      noisy1: masked input_ids [B, L]
      noisy2: optional second masked copy for MIRROR mode (else None)
      eligible: boolean mask of eligible tokens (answer tokens, non-padding)
    """
    device = input_ids.device
    B, L = input_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones((B, L), device=device, dtype=torch.long)

    if fixed_t is None:
        t = torch.rand((B,), device=device)
    else:
        t = fixed_t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.repeat(B)

    p_mask = ((1.0 - float(eps)) * t + float(eps)).clamp(min=float(eps), max=1.0)
    p_mask = p_mask[:, None].repeat(1, L)

    eligible = (labels != -100) & (attention_mask == 1)

    bern1 = torch.rand((B, L), device=device)
    mask1 = (bern1 < p_mask) & eligible
    noisy1 = input_ids.clone()
    noisy1[mask1] = MASK_TOKEN_ID

    noisy2 = None
    if train_mode == "MIRROR":
        bern2 = torch.rand((B, L), device=device)
        mask2 = (bern2 < p_mask) & eligible
        noisy2 = input_ids.clone()
        noisy2[mask2] = MASK_TOKEN_ID

    if iw_t is None:
        iw_t = torch.ones((B,), device=device)
    else:
        iw_t = iw_t.to(device=device, dtype=torch.float32)
        if iw_t.ndim == 0:
            iw_t = iw_t.repeat(B)
        elif iw_t.ndim == 2 and iw_t.size(1) == 1:
            iw_t = iw_t.view(-1)

    return p_mask, iw_t, noisy1, noisy2, eligible
