"""HuggingFace helper utilities.

Some scripts in this repo support a `--china` flag to use the hf-mirror.com
endpoints (useful when the default HF CDN is slow/unavailable).
"""

from __future__ import annotations

import os
from typing import Iterable


def enable_hf_mirror_china() -> None:
    """Enable HuggingFace mirror endpoints commonly used in China.

    This matches the env vars historically set in the legacy preprocessing scripts.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
    # Avoid hf_transfer when using mirrors.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def maybe_enable_hf_mirror_china(argv: Iterable[str]) -> None:
    """Enable HF mirror if '--china' is present in argv."""
    if any(a == "--china" for a in argv):
        enable_hf_mirror_china()
