from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    tokenizer: any
    model: torch.nn.Module
    device: torch.device
    device_ids: List[int]


def load_tokenizer_and_model(
    model_name: str,
    checkpoint_path: str = "",
    device_ids: Optional[List[int]] = None,
) -> LoadedModel:
    device_ids = device_ids or [0]
    load_path = checkpoint_path if checkpoint_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available() and len(device_ids) > 1:
        model = torch.nn.DataParallel(base_model, device_ids=device_ids).to(device)
    else:
        model = base_model.to(device)

    model.eval()
    return LoadedModel(tokenizer=tokenizer, model=model, device=device, device_ids=device_ids)
