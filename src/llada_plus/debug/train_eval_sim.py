"""Simulate the *training loop* and the *inference/eval loop* end-to-end on a tiny JSONL.

This is meant for debugging the data plumbing + masking diffusion + loss + forward pass
without requiring Transformers/Accelerate/DeepSpeed or downloading any checkpoints.

It follows the same high-level structure as `llada_plus.train.runner.train`:
  1) load processed JSONL -> split train/eval
  2) DataLoader + collate_fn
  3) forward_process (masking diffusion)
  4) masked-token CE loss
  5) evaluation (masked-loss + masked-token accuracy)

Run (from repo root):

  PYTHONPATH=src python -m llada_plus.debug.train_eval_sim

You can also point it to your processed jsonl:

  PYTHONPATH=src python -m llada_plus.debug.train_eval_sim --jsonl ./data/train/gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from llada_plus.data import LLaDADataset, collate_fn
from llada_plus.diffusion import MASK_TOKEN_ID, forward_process
from llada_plus.losses import batched_loss_for_backpropagate


class ToyLM(torch.nn.Module):
    """A tiny LM with a HuggingFace-like forward() signature."""

    def __init__(self, vocab_size: int = 512, hidden: int = 64):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb = torch.nn.Embedding(self.vocab_size, int(hidden))
        self.lm_head = torch.nn.Linear(int(hidden), self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # Make very large ids (e.g. MASK_TOKEN_ID) safe for embedding lookup.
        ids = torch.remainder(input_ids, self.vocab_size)
        h = self.emb(ids)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits})


def _write_dummy_jsonl(
    path: Path,
    *,
    num_examples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(int(num_examples)):
            L = rng.randint(max(8, seq_len // 2), int(seq_len))
            prompt_length = rng.randint(0, max(0, L - 2))
            input_ids = [rng.randrange(int(vocab_size)) for _ in range(L)]
            f.write(json.dumps({"input_ids": input_ids, "prompt_length": prompt_length}) + "\n")


@dataclass
class EvalStats:
    loss_sum: float = 0.0
    loss_batches: int = 0
    masked_correct: int = 0
    masked_total: int = 0


@torch.no_grad()
def _eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pad_id: int,
    train_mode: str,
    fixed_t: float,
) -> EvalStats:
    model.eval()
    stats = EvalStats()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        B = ids.size(0)
        t = torch.full((B,), float(fixed_t), device=device)
        iw = torch.ones((B,), device=device)

        p_mask, iw, noisy1, _, eligible = forward_process(
            ids,
            am,
            lbls,
            train_mode=train_mode,
            fixed_t=t,
            iw_t=iw,
        )

        if not (noisy1 == MASK_TOKEN_ID).any():
            # No masked positions -> skip (same as main runner)
            continue

        loss = batched_loss_for_backpropagate(
            ids,
            noisy1,
            model,
            p_mask,
            iw,
            eligible,
            train=False,
            pad_id=pad_id,
            attn_mask=am,
        )
        stats.loss_sum += float(loss.item())
        stats.loss_batches += 1

        # "Inference" sanity check: greedy-predict masked positions and compute accuracy.
        mask_tok = (noisy1 == MASK_TOKEN_ID) & (lbls != -100) & (am == 1)
        if mask_tok.any():
            logits = model(noisy1, attention_mask=am).logits  # [B, L, V]
            pred = torch.argmax(logits, dim=-1)  # [B, L]
            gt = torch.remainder(ids, getattr(model, "vocab_size", 512))
            correct = (pred == gt) & mask_tok
            stats.masked_correct += int(correct.sum().item())
            stats.masked_total += int(mask_tok.sum().item())

    model.train()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default=None, help="Processed jsonl path. If omitted, a dummy one is generated.")
    ap.add_argument("--num_examples", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--train_mode", type=str, choices=["Normal", "MIRROR"], default="Normal")
    ap.add_argument(
        "--fixed_t",
        type=float,
        default=0.5,
        help="Use a fixed masking ratio t in (0,1). 0.5 makes it likely to have masked tokens every batch.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.jsonl is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="llada_plus_train_eval_sim_"))
        jsonl_path = tmpdir / "dummy.jsonl"
        _write_dummy_jsonl(
            jsonl_path,
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        print(f"[sim] wrote dummy jsonl: {jsonl_path}")
    else:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(str(jsonl_path))
        print(f"[sim] using jsonl: {jsonl_path}")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    ds = LLaDADataset(str(jsonl_path), max_len=int(args.max_len))
    train_n = int(len(ds) * float(args.train_ratio))
    eval_n = len(ds) - train_n
    g = torch.Generator().manual_seed(int(args.seed))
    train_ds, eval_ds = torch.utils.data.random_split(ds, [train_n, eval_n], generator=g)

    pad_id = 0
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id=pad_id),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, pad_id=pad_id),
    )

    model = ToyLM(vocab_size=int(args.vocab_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    # --- Sanity: run an eval before training ---
    if eval_n > 0:
        stats0 = _eval_one_epoch(
            model,
            eval_loader,
            device=device,
            pad_id=pad_id,
            train_mode=args.train_mode,
            fixed_t=float(args.fixed_t),
        )
        avg0 = stats0.loss_sum / max(1, stats0.loss_batches)
        acc0 = (stats0.masked_correct / stats0.masked_total) if stats0.masked_total > 0 else 0.0
        print(f"[sim] eval@init: loss={avg0:.6f} | masked_acc={acc0:.4f} | masked_tokens={stats0.masked_total}")

    # --- Train ---
    for epoch in range(int(args.epochs)):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            B = ids.size(0)
            t = torch.full((B,), float(args.fixed_t), device=device)
            iw = torch.ones((B,), device=device)

            p_mask, iw, noisy1, noisy2, eligible = forward_process(
                ids,
                am,
                lbls,
                train_mode=args.train_mode,
                fixed_t=t,
                iw_t=iw,
            )

            has_mask1 = (noisy1 == MASK_TOKEN_ID).any()
            has_mask2 = (noisy2 == MASK_TOKEN_ID).any() if noisy2 is not None else False
            if args.train_mode == "Normal" and not has_mask1:
                continue
            if args.train_mode == "MIRROR" and not (has_mask1 or has_mask2):
                continue

            if args.train_mode == "Normal":
                loss = batched_loss_for_backpropagate(
                    ids,
                    noisy1,
                    model,
                    p_mask,
                    iw,
                    eligible,
                    train=True,
                    pad_id=pad_id,
                    attn_mask=am,
                )
            else:
                loss1 = batched_loss_for_backpropagate(
                    ids,
                    noisy1,
                    model,
                    p_mask,
                    iw,
                    eligible,
                    train=True,
                    pad_id=pad_id,
                    attn_mask=am,
                )
                if has_mask2 and noisy2 is not None:
                    loss2 = batched_loss_for_backpropagate(
                        ids,
                        noisy2,
                        model,
                        p_mask,
                        iw,
                        eligible,
                        train=True,
                        pad_id=pad_id,
                        attn_mask=am,
                    )
                    loss = 0.5 * (loss1 + loss2)
                else:
                    loss = loss1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            steps += 1
            if steps % 20 == 0:
                print(f"[sim] epoch {epoch+1} step {steps}: train_loss={running/steps:.6f}")

        if steps == 0:
            raise RuntimeError(
                "No training steps produced masked tokens. "
                "Try increasing --fixed_t (e.g. 0.5) or check your jsonl prompt_length/labels logic."
            )

        print(f"[sim] epoch {epoch+1} done: train_loss={running/steps:.6f} (steps={steps})")

        if eval_n > 0:
            stats = _eval_one_epoch(
                model,
                eval_loader,
                device=device,
                pad_id=pad_id,
                train_mode=args.train_mode,
                fixed_t=float(args.fixed_t),
            )
            avg = stats.loss_sum / max(1, stats.loss_batches)
            acc = (stats.masked_correct / stats.masked_total) if stats.masked_total > 0 else 0.0
            print(f"[sim] eval@epoch{epoch+1}: loss={avg:.6f} | masked_acc={acc:.4f} | masked_tokens={stats.masked_total}")

    print("[sim] SUCCESS")


if __name__ == "__main__":
    main()
