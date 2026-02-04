"""Answer extraction + lightweight normalization for evaluation/preprocess.

This repo has a few preprocessing scripts for inference-time datasets (e.g. MATH,
GSM8K) that need robust answer extraction and mild formatting normalization.
Keeping the logic here avoids copy/paste drift across scripts.

Notes
-----
These utilities are intentionally heuristic and *not* a symbolic algebra system.
They are designed to:
  - extract common final-answer markers (\\boxed{...}, '#### ...')
  - split multi-answer outputs at top level (not inside brackets)
  - do small LaTeX-to-text normalizations to reduce superficial mismatches
"""

from __future__ import annotations

import re
from typing import Iterable, List


# ---------- GSM8K '#### ' final answer ----------
HASH4_REGEX = re.compile(r"####\s*([^\n\r]+)")


def extract_hash4_answers(text: str, *, normalize_math: bool = True) -> List[str]:
    """Extract all occurrences of '#### ...' from GSM8K-style solutions."""
    if not text:
        return []
    raw = [m.strip() for m in HASH4_REGEX.findall(text)]
    if normalize_math:
        raw = [normalize_math_expr(x) for x in raw]
    else:
        raw = [basic_clean(x) for x in raw]
    return dedupe_preserve_order(raw)


# ---------- Robust \boxed{...} extraction with a brace stack ----------
def extract_boxed_all(text: str) -> List[str]:
    """Extract all '\\boxed{...}' contents, handling nested braces."""
    if not text:
        return []
    s = text
    out: List[str] = []
    i, n = 0, len(s)
    key = r"\\boxed"
    while i < n:
        j = s.find(key, i)
        if j < 0:
            break
        k = j + len(key)
        while k < n and s[k].isspace():
            k += 1
        if k >= n or s[k] != "{":
            i = k
            continue

        depth = 0
        start = k
        p = k
        while p < n:
            ch = s[p]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[start + 1 : p].strip())
                    i = p + 1
                    break
            p += 1
        else:
            break
    return out


# ---------- Top-level splitting (outside any brackets) ----------
def top_level_split(s: str, *, split_commas: bool = False) -> List[str]:
    """Split answers only at top level (outside () [] {}).

    Splits on:
      - ';'
      - word-boundary 'or'/'and'
      - optionally ','
    """
    if not s:
        return []
    n = len(s)
    i = 0
    parts: List[str] = []
    buf: List[str] = []
    dep_par = dep_bra = dep_cur = 0

    def is_word_char(c: str) -> bool:
        return c.isalpha()

    def flush() -> None:
        token = "".join(buf).strip()
        if token:
            parts.append(token)
        buf.clear()

    while i < n:
        ch = s[i]
        if ch == "(":
            dep_par += 1
        elif ch == ")":
            dep_par = max(0, dep_par - 1)
        elif ch == "[":
            dep_bra += 1
        elif ch == "]":
            dep_bra = max(0, dep_bra - 1)
        elif ch == "{":
            dep_cur += 1
        elif ch == "}":
            dep_cur = max(0, dep_cur - 1)

        at_top = (dep_par == 0 and dep_bra == 0 and dep_cur == 0)

        if at_top and ch == ";":
            flush()
            i += 1
            continue
        if at_top and split_commas and ch == ",":
            flush()
            i += 1
            continue

        if at_top and ch.isalpha():
            j = i
            while j < n and is_word_char(s[j]):
                j += 1
            word = s[i:j]
            prev = s[i - 1] if i - 1 >= 0 else " "
            nxt = s[j] if j < n else " "
            if word in ("or", "and") and (not is_word_char(prev)) and (not is_word_char(nxt)):
                flush()
                i = j
                continue
            buf.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush()
    return [p for p in parts if p]


# ---------- Normalization ----------
LATEX_LEFT_RIGHT = re.compile(r"\\left\s*|\\right\s*")
LATEX_TEXT = re.compile(r"\\text\s*\{([^{}]*)\}")
LATEX_SPACES = re.compile(r"\\[ ,;:!]")
FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")


def basic_clean(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_math_expr(s: str) -> str:
    """Lightweight LaTeX-ish normalization (heuristic)."""
    if not s:
        return ""
    s = basic_clean(s)

    s = LATEX_LEFT_RIGHT.sub("", s)

    while True:
        ns = LATEX_TEXT.sub(lambda m: m.group(1), s)
        if ns == s:
            break
        s = ns

    while True:
        ns = FRAC.sub(lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", s)
        if ns == s:
            break
        s = ns

    s = SQRT.sub(lambda m: f"sqrt({m.group(1).strip()})", s)

    s = (
        s.replace(r"\\cdot", "*")
        .replace(r"\\times", "*")
        .replace(r"\\div", "/")
        .replace(r"\\pm", "±")
    )
    s = s.replace(r"\\$", "$")

    s = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^\s*([A-Za-z0-9]+)", r"**\1", s)

    s = LATEX_SPACES.sub("", s)
    s = re.sub(r"(?<!\d)\.(\d+)", r"0.\1", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = (x or "").strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out
