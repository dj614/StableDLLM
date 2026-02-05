"""Hugging Face hub environment helpers.

The original scripts optionally used the hf-mirror.com endpoint when running
in restricted networks. Several entrypoints in this repo import helpers from
``mdm.utils.hf``.

This module keeps behavior minimal: set a small set of environment variables.
"""

from __future__ import annotations

import os


def enable_hf_mirror_china() -> None:
    """Enable the Hugging Face mirror used by the original LLaDA scripts."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
    # Disable hf_transfer by default (often blocked on mirrored endpoints).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def maybe_enable_hf_mirror_china(enabled: bool) -> None:
    if enabled:
        enable_hf_mirror_china()


# Backward-compatible aliases used by mdm.train.*
def enable_hf_mirror() -> None:
    enable_hf_mirror_china()


def maybe_enable_hf_mirror(enabled: bool) -> None:
    maybe_enable_hf_mirror_china(enabled)
