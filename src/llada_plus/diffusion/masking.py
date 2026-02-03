"""Masking-based forward diffusion process used by LLaDA."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# Keep the same special token id as the original implementation.
MASK_TOKEN_ID = 126336


def forward_process(
    batch_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    train_mode: str,
    fixed_t: torch.Tensor,
    iw_t: torch.Tensor,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Apply forward noising (masking) to a batch.

    Rules are identical to the original script:
      - eligible positions are those where labels != -100 and attention_mask == 1
      - we never noise prompt tokens (labels == -100) or padding tokens (attention_mask == 0)
      - p_mask = (1-eps)*t + eps, broadcast to sequence length
      - Normal mode returns only noisy1
      - MirrorMask/mirror_plus also returns a mirrored noisy2
    """
    B, L = batch_ids.shape
    device = batch_ids.device

    eligible_pos = (labels != -100) & (attention_mask == 1)  # [B, L]

    p_mask_base = (1 - eps) * fixed_t[:, None] + eps
    p_mask = p_mask_base.expand(-1, L)

    u = torch.rand_like(p_mask)

    noisy1 = batch_ids.clone()
    mask_here1 = (u < p_mask) & eligible_pos
    noisy1[mask_here1] = MASK_TOKEN_ID

    noisy2: Optional[torch.Tensor] = None
    if train_mode in ["MirrorMask", "mirror_plus"]:
        noisy2 = batch_ids.clone()
        mask_here2 = (u > (1 - p_mask)) & eligible_pos
        noisy2[mask_here2] = MASK_TOKEN_ID

    eligible_count = eligible_pos.sum(dim=1).clamp(min=1)
    return p_mask, iw_t, noisy1, noisy2, eligible_count
