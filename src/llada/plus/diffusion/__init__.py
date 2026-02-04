"""Diffusion/noise process utilities."""

from .masking import MASK_TOKEN_ID, forward_process

__all__ = [
    "MASK_TOKEN_ID",
    "forward_process",
]
