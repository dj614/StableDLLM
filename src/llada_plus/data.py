"""JSONL dataset utilities for LLaDA+ training.

Expected JSONL schema (per line)
-------------------------------
Each line should be a JSON object with:

  - "input_ids": a list of token ids (prompt + completion)
  - "prompt_length": an int indicating how many prefix tokens belong to the
    prompt/context (these will be excluded from the masked-token loss)

The rest of the training pipeline (masking diffusion + masked CE) relies on:
  - labels == -100 to mark *non-eligible* positions (prompt tokens and padding)
  - attention_mask == 0 to mark padding

This module is dependency-light (stdlib + torch only) so it can be used in small
debug smoke tests without requiring Transformers/Accelerate.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


def _as_int_list(x: Any) -> List[int]:
    if not isinstance(x, list):
        raise TypeError(f"input_ids must be a list[int], got {type(x)}")
    out: List[int] = []
    for i, v in enumerate(x):
        try:
            out.append(int(v))
        except Exception as e:  # pragma: no cover
            raise TypeError(f"input_ids[{i}] is not int-like: {v!r}") from e
    return out


def _normalize_record(rec: Dict[str, Any], *, max_len: Optional[int]) -> Dict[str, Any]:
    """Validate/normalize one JSONL record."""
    # Accept a couple of common aliases, but keep behavior explicit.
    ids_raw = rec.get("input_ids", None)
    if ids_raw is None:
        ids_raw = rec.get("ids", None)
    if ids_raw is None:
        raise KeyError(
            "JSONL record missing 'input_ids'. Expected keys like: input_ids, prompt_length. "
            f"Got keys: {sorted(rec.keys())}"
        )

    input_ids = _as_int_list(ids_raw)

    pl = rec.get("prompt_length", None)
    if pl is None:
        pl = rec.get("prompt_len", None)
    if pl is None:
        pl = rec.get("prompt_ids_len", None)
    if pl is None:
        # Default to 0 (pretraining-style). We do NOT default to len(input_ids),
        # because that would make all tokens ineligible and training would stall.
        pl = 0

    try:
        prompt_length = int(pl)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"prompt_length must be int-like, got {pl!r}") from e

    if prompt_length < 0:
        prompt_length = 0

    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]

    if not input_ids:
        raise ValueError("input_ids is empty after normalization/truncation")

    if prompt_length > len(input_ids):
        prompt_length = len(input_ids)

    return {
        "input_ids": input_ids,
        "prompt_length": prompt_length,
    }


def _build_jsonl_offsets(path: str) -> List[int]:
    """Return byte offsets for each non-empty line in a JSONL file."""
    offsets: List[int] = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


@dataclass
class LLaDADataset(torch.utils.data.Dataset):
    """Memory-light JSONL dataset (stores only line offsets).

    Args:
        jsonl_path: Path to the processed jsonl.
        max_len: Truncate sequences to this length.
    """

    jsonl_path: str
    max_len: int

    def __post_init__(self) -> None:
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)
        self._offsets = _build_jsonl_offsets(self.jsonl_path)
        if not self._offsets:
            raise ValueError(f"No non-empty lines found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self._offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        try:
            rec = json.loads(line.decode("utf-8"))
        except Exception as e:
            raise ValueError(
                f"Invalid JSON on line index {idx} (byte offset {off}) in {self.jsonl_path}"
            ) from e
        return _normalize_record(rec, max_len=self.max_len)


def collate_fn(examples: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad batch and create {input_ids, attention_mask, labels}.

    - `labels` is `input_ids` with prompt tokens and padding set to -100.
    - `attention_mask` is 1 for real tokens, 0 for padding.
    """
    if len(examples) == 0:
        raise ValueError("Empty batch")

    lengths = [len(ex["input_ids"]) for ex in examples]
    max_len = max(lengths)

    B = len(examples)
    input_ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, ex in enumerate(examples):
        ids = ex["input_ids"]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, :L] = input_ids[i, :L]

        pl = int(ex.get("prompt_length", 0))
        if pl < 0:
            pl = 0
        if pl > L:
            pl = L
        if pl > 0:
            labels[i, :pl] = -100

        # Padding positions are already -100 and attention_mask==0.

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def sample_multi_from_jsonl(
    jsonl_path: str,
    n: int,
    *,
    seed: Optional[int] = None,
    max_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reservoir-sample `n` records from a JSONL without loading it fully.

    Returns normalized records with keys: input_ids, prompt_length.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)

    reservoir: List[Dict[str, Any]] = []
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            try:
                norm = _normalize_record(rec, max_len=max_len)
            except Exception:
                continue

            if len(reservoir) < n:
                reservoir.append(norm)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = norm

    if len(reservoir) < n:
        raise ValueError(
            f"Requested {n} samples from {jsonl_path}, but only got {len(reservoir)} valid records"
        )

    return reservoir
