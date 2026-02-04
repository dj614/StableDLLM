from __future__ import annotations

import re
from typing import Optional


_GSM8K_HASH_RE = re.compile(r"####\s*([-+]?\d+)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_LAST_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def extract_gsm8k_hash_answer(text: str | None) -> Optional[str]:
    """Extract GSM8K answer formatted as '#### 1234'."""
    if not isinstance(text, str):
        return None
    m = _GSM8K_HASH_RE.search(text)
    return m.group(1).strip() if m else None


def extract_boxed_answer(text: str | None) -> Optional[str]:
    """Extract answer inside \boxed{...}."""
    if not isinstance(text, str):
        return None
    m = _BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def extract_last_number(text: str | None) -> Optional[str]:
    """Fallback: return the last number-like token in text."""
    if not isinstance(text, str):
        return None
    ms = list(_LAST_NUMBER_RE.finditer(text))
    if not ms:
        return None
    return ms[-1].group(1).strip()
