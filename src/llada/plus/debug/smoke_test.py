"""End-to-end smoke test for the *local* LLaDA+ training components.

This script generates a tiny processed JSONL dataset and then runs:

  Dataset -> collate_fn -> forward_process -> masked CE loss

using a small PyTorch "toy" language model, so it can be executed without
Transformers/Accelerate/DeepSpeed or downloading any HuggingFace checkpoint.

Run (from repo root):

  PYTHONPATH=src python -m llada.plus.debug.smoke_test
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llada.plus.data import LLaDADataset, collate_fn, sample_multi_from_jsonl
from core.diffusion import forward_process
from core.losses import batched_loss_for_backpropagate


class ToyLM(torch.nn.Module):
    """A tiny LM with the same call signature as HuggingFace CausalLM.

    Notes:
      - We mod input_ids by vocab_size so very large ids (e.g. MASK_TOKEN_ID)
        won't crash the embedding lookup.
      - Targets (labels) are expected to be within [0, vocab_size).
    """

    def __init__(self, vocab_size: int = 512, hidden: int = 64):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb = torch.nn.Embedding(self.vocab_size, int(hidden))
        self.lm_head = torch.nn.Linear(int(hidden), self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        ids = torch.remainder(input_ids, self.vocab_size)
        h = self.emb(ids)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})


def _write_dummy_jsonl(path: Path, *, num_examples: int, seq_len: int, vocab_size: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(int(num_examples)):
            L = rng.randint(max(4, seq_len // 2), int(seq_len))
            # Ensure there are eligible tokens (prompt_length < L)
            prompt_length = rng.randint(0, max(0, L - 2))
            input_ids = [rng.randrange(int(vocab_size)) for _ in range(L)]
            f.write(json.dumps({"input_ids": input_ids, "prompt_length": prompt_length}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Use an existing processed jsonl; if omitted, a temp one is created.",
    )
    ap.add_argument("--num_examples", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.jsonl is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="llada_plus_smoke_"))
        jsonl_path = tmpdir / "dummy.jsonl"
        _write_dummy_jsonl(
            jsonl_path,
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        print(f"[smoke] wrote dummy jsonl: {jsonl_path}")
    else:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(str(jsonl_path))
        print(f"[smoke] using jsonl: {jsonl_path}")

    ds = LLaDADataset(str(jsonl_path), max_len=args.max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id=0),
    )

    model = ToyLM(vocab_size=args.vocab_size).to(args.device)

    batch = next(iter(loader))
    ids = batch["input_ids"].to(args.device)
    am = batch["attention_mask"].to(args.device)
    lbls = batch["labels"].to(args.device)

    # Force a moderate masking ratio so we almost surely get masked positions.
    fixed_t = torch.full((ids.size(0),), 0.5, device=args.device)
    iw_t = torch.ones(ids.size(0), device=args.device)

    p_mask, iw_t, noisy1, _, eligible = forward_process(
        ids,
        am,
        lbls,
        train_mode="Normal",
        fixed_t=fixed_t,
        iw_t=iw_t,
    )

    loss = batched_loss_for_backpropagate(
        ids,
        noisy1,
        model,
        p_mask,
        iw_t,
        eligible,
        train=True,
        pad_id=0,
        attn_mask=am,
    )
    loss.backward()
    print(f"[smoke] loss={loss.item():.6f} (backward ok)")

    # Also exercise PPOTS helper.
    k = min(5, len(ds))
    samples = sample_multi_from_jsonl(str(jsonl_path), n=k, seed=args.seed, max_len=args.max_len)
    assert len(samples) == k and isinstance(samples[0]["input_ids"], list)
    print(f"[smoke] sample_multi_from_jsonl ok: got {len(samples)} examples")

    print("[smoke] SUCCESS")


if __name__ == "__main__":
    main()
