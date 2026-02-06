"""Utilities shared by LLaDA and MMaDA scripts.

LLaDA (LLaDA-8B-*) and MMaDA (MMaDA-8B-*) share the same *architecture*, but
MMaDA is trained with a different tokenizer. This means hard-coding special
token IDs (especially the diffusion [MASK] token) is fragile.

These helpers resolve special IDs from the tokenizer/model config, and provide
simple model/tokenizer loading wrappers.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

# Checkpoint defaults (can be overridden via CLI args).
DEFAULT_LLADA_INSTRUCT = os.environ.get("LLADA_MODEL_NAME_OR_PATH", "GSAI-ML/LLaDA-8B-Instruct")
DEFAULT_LLADA_BASE = os.environ.get("LLADA_BASE_MODEL_NAME_OR_PATH", "GSAI-ML/LLaDA-8B-Base")
DEFAULT_MMADA_MIXCOT = os.environ.get("MMADA_MODEL_NAME_OR_PATH", "Gen-Verse/MMaDA-8B-MixCoT")


def pick_device(device: Optional[str] = None) -> str:
    """Pick a sensible default device."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_dtype(dtype: str) -> torch.dtype:
    d = (dtype or "bf16").lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16", "half"}:
        return torch.float16
    if d in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device: Optional[str] = None,
    dtype: str = "bf16",
    trust_remote_code: bool = True,
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    """Load model + tokenizer with common defaults."""
    dev = pick_device(device)
    torch_dtype = parse_dtype(dtype)

    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model = model.to(dev).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer, dev


def _try_get(obj, attr: str):
    return getattr(obj, attr, None) if obj is not None else None


def resolve_mask_id(
    tokenizer,
    model: Optional[torch.nn.Module] = None,
    *,
    override: Optional[int] = None,
    fallback: int = 126336,
) -> int:
    """Resolve the diffusion [MASK] token id.

    Prefer tokenizer/model config; fall back to the original LLaDA reserved id.
    """
    if override is not None:
        return int(override)

    for src in (tokenizer, _try_get(model, "config")):
        val = _try_get(src, "mask_token_id")
        if val is not None:
            return int(val)

    # Some tokenizers do not expose mask_token_id but still contain a "[MASK]" token.
    try:
        tok = _try_get(tokenizer, "mask_token") or "[MASK]"
        val = tokenizer.convert_tokens_to_ids(tok)
        unk = _try_get(tokenizer, "unk_token_id")
        if val is not None and (unk is None or val != unk):
            return int(val)
    except Exception:
        pass

    return int(fallback)


def resolve_pad_id(
    tokenizer,
    model: Optional[torch.nn.Module] = None,
    *,
    override: Optional[int] = None,
    fallback: Optional[int] = 126081,
) -> Optional[int]:
    """Resolve pad token id (used for stripping padding in chat loop)."""
    if override is not None:
        return int(override)

    for src in (tokenizer, _try_get(model, "config")):
        val = _try_get(src, "pad_token_id")
        if val is not None:
            return int(val)
    return int(fallback) if fallback is not None else None


def resolve_eos_id(
    tokenizer,
    model: Optional[torch.nn.Module] = None,
    *,
    override: Optional[int] = None,
) -> Optional[int]:
    if override is not None:
        return int(override)

    for src in (tokenizer, _try_get(model, "config")):
        val = _try_get(src, "eos_token_id")
        if val is not None:
            return int(val)
    return None


def format_user_prompt(
    tokenizer,
    user_text: str,
    *,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """Format a single user message into a prompt string.

    If the tokenizer provides `apply_chat_template`, use it; otherwise return the raw text.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    return user_text
