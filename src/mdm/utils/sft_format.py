"""Simple SFT JSONL formatter utilities.

Produces records with:
  - input_ids: concatenated prompt + response token ids
  - prompt_length: number of prompt tokens (excluded from masked-token loss)
"""

from __future__ import annotations

from typing import Dict


def encode_sft_pair(
    prompt: str,
    response: str,
    tokenizer,
    *,
    user_suffix: str = "",
    assistant_suffix: str = "",
    strip: bool = True,
) -> Dict[str, object]:
    """Encode a prompt/response pair into the JSONL training format."""
    if strip:
        prompt = (prompt or "").strip()
        response = (response or "").strip()
    else:
        prompt = prompt or ""
        response = response or ""

    prompt_text = f"{prompt}{user_suffix}"
    response_text = f"{response}{assistant_suffix}"

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "prompt_length": len(prompt_ids),
    }
