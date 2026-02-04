"""Sampling utilities for diffusion time `t`.

This module provides a thin wrapper used by the training runner:
  - uniform sampling of t
  - sampling from a fitted p(t) on a grid (importance sampling)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class TSampler:
    """Callable wrapper returning (t, iw_t) given a batch size."""

    sample: Callable[[int], Tuple[torch.Tensor, torch.Tensor]]


def uniform_t_sampler(device: torch.device) -> TSampler:
    def _sample(bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand(bs, device=device)
        iw = torch.ones(bs, device=device)
        return t, iw

    return TSampler(sample=_sample)


def grid_pdf_t_sampler(t_grid: torch.Tensor, p_grid: torch.Tensor, *, max_is_weight: float = 1e6) -> TSampler:
    """Create a sampler for t based on a pdf defined on a grid.

    Args:
        t_grid: strictly increasing [N] tensor in (0,1)
        p_grid: non-negative [N] tensor; will be normalized by trapezoid rule
        max_is_weight: clamp for the importance weight 1/p(t)
    """
    if t_grid.ndim != 1 or p_grid.ndim != 1:
        raise ValueError("t_grid and p_grid must be 1D tensors")
    if t_grid.numel() < 2:
        raise ValueError("t_grid must have at least 2 points")

    p_grid = p_grid / torch.trapz(p_grid, t_grid).clamp(min=1e-12)
    dt = t_grid[1:] - t_grid[:-1]
    cumsum_area = torch.cumsum(0.5 * (p_grid[:-1] + p_grid[1:]) * dt, dim=0)
    cdf = torch.cat([torch.zeros(1, device=t_grid.device), cumsum_area], dim=0)
    cdf = cdf / cdf[-1].clamp(min=1e-12)

    def _sample(bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.rand(bs, device=t_grid.device)
        idx = torch.searchsorted(cdf, u, right=True)
        idx = torch.clamp(idx, 1, t_grid.numel() - 1)
        t0, t1 = t_grid[idx - 1], t_grid[idx]
        c0, c1 = cdf[idx - 1], cdf[idx]
        r = (u - c0) / (c1 - c0 + 1e-12)
        t_s = t0 + r * (t1 - t0)
        p0, p1 = p_grid[idx - 1], p_grid[idx]
        p_s = p0 + r * (p1 - p0)
        iw = (1.0 / p_s.clamp(min=1e-12)).clamp_max(float(max_is_weight)).detach()
        return t_s, iw

    return TSampler(sample=_sample)
