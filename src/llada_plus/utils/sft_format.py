"""Chat-format helpers for preparing SFT-style JSONL for LLaDA/LLaDA+.

Several preprocessing scripts in this repo build `input_ids` by explicitly
concatenating a user turn and an assistant turn with model-specific special
tokens. This module centralizes that logic to avoid copy/paste divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SpecialTokens:
    # These match the chat template used by GSAI-ML/LLaDA-8B-Instruct.
    BOS: str = "<s>"
    EOS: str = "</s>"
    START_USER: str = "<start_id>user<end_id>\n"
    START_ASSIST: str = "<start_id>assistant<end_id>\n"
    EOT: str = "<eot_id>"


SPECIAL = SpecialTokens()


def encode_sft_pair(
    prompt: str,
    answer: str,
    tok: Any,
    *,
    user_suffix: str = "",
    assistant_suffix: str = "",
    strip: bool = True,
) -> Dict[str, Any]:
    """Encode a single (prompt, answer) pair into {input_ids, prompt_length}.

    Args:
        prompt: user content.
        answer: assistant content.
        tok: HF tokenizer.
        user_suffix: appended to user content before EOT (e.g. "\n").
        assistant_suffix: appended to assistant content before EOS (e.g. "\n").
        strip: whether to .strip() prompt/answer.
    """
    p = prompt.strip() if strip else prompt
    a = answer.strip() if strip else answer

    user_part = SPECIAL.BOS + SPECIAL.START_USER + p + user_suffix + SPECIAL.EOT
    asst_part = SPECIAL.START_ASSIST + a + assistant_suffix + SPECIAL.EOS

    user_ids = tok(user_part, add_special_tokens=False).input_ids
    asst_ids = tok(asst_part, add_special_tokens=False).input_ids
    ids = user_ids + asst_ids
    return {"input_ids": ids, "prompt_length": len(user_ids)}
