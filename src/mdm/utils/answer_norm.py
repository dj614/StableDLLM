"""Answer normalization helpers used by preprocessing scripts."""

from __future__ import annotations

import re
from typing import Iterable, List


_HASH4_RE = re.compile(r"####\s*(.+)")


def basic_clean(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _extract_braced(text: str, start: int) -> tuple[str | None, int | None]:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return None, None


def extract_boxed_all(text: str) -> List[str]:
    """Extract all contents inside \\boxed{...} (or \\fbox{...})."""
    s = text or ""
    out: List[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\boxed{", i) or s.startswith("\\fbox{", i):
            prefix = "\\boxed{" if s.startswith("\\boxed{", i) else "\\fbox{"
            start = i + len(prefix)
            content, end = _extract_braced(s, start)
            if content is not None:
                out.append(content)
                i = end if end is not None else start
                continue
        i += 1
    if not out and s:
        out = [s]
    return out


def normalize_math_expr(text: str) -> str:
    s = basic_clean(text)
    if not s:
        return ""
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("~", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def top_level_split(text: str, *, split_commas: bool = True) -> List[str]:
    if not split_commas:
        return [basic_clean(text)]
    s = text or ""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = basic_clean("".join(buf))
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = basic_clean("".join(buf))
    if tail:
        parts.append(tail)
    return parts


def extract_hash4_answers(answer: str, *, normalize_math: bool = True) -> List[str]:
    """Extract answers after '####' markers (GSM8K-style)."""
    s = answer or ""
    matches = _HASH4_RE.findall(s)
    if not matches:
        matches = [s]
    cleaned: List[str] = []
    for m in matches:
        val = normalize_math_expr(m) if normalize_math else basic_clean(m)
        if val:
            cleaned.append(val)
    return dedupe_preserve_order(cleaned)
