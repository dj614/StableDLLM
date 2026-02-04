from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..model.load import load_tokenizer_and_model
from ..tasks.registry import iter_task_examples
from ..utils.io import iter_jsonl, write_jsonl
from ..eval.metrics import score_gsm8k, score_hitab, score_openscience

try:
    # The original sampler lives in the top-level `LLaDA/` package in this repo.
    from LLaDA.generate import generate  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for alternative layouts where `generate` is importable directly.
    from generate import generate  # type: ignore


DEFAULT_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


def _ensure_repo_paths():
    # placeholder for future path utilities
    return


def cmd_infer(args: argparse.Namespace) -> None:
    loaded = load_tokenizer_and_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device_ids=args.device_ids,
    )
    tok = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    batch_gold: List[str] = []

    def flush_batch():
        nonlocal batch_prompts, batch_meta, batch_gold, rows
        if not batch_prompts:
            return
        prompts = []
        for p in batch_prompts:
            msgs = [{"role": "user", "content": p}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)
        encoded = tok(prompts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel) and input_ids.size(0) < len(loaded.device_ids):
                out = generate(
                    model.module,
                    input_ids.to(f"cuda:{loaded.device_ids[0]}") if device.type == "cuda" else input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )
            else:
                out = generate(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )

        decoded = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        for pred_text, meta, gold_raw, raw_prompt in zip(decoded, batch_meta, batch_gold, batch_prompts):
            rows.append({
                "task": args.task,
                "prompt": raw_prompt,
                "gold_raw": gold_raw,
                "prediction": pred_text,
                "meta": meta,
            })

        batch_prompts, batch_meta, batch_gold = [], [], []

    for ex in iter_task_examples(
        task=args.task,
        split=args.split,
        data_jsonl=args.data_jsonl,
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
    ):
        batch_prompts.append(ex.prompt)
        batch_meta.append(ex.meta)
        batch_gold.append(ex.gold_raw)
        if len(batch_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()
    write_jsonl(out_path, rows)
    print(f"Saved predictions: {out_path} (n={len(rows)})")


def cmd_score(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(args.pred_jsonl))
    task = args.task.lower()
    if task == "gsm8k":
        res = score_gsm8k(rows)
    elif task == "openscience":
        res = score_openscience(rows)
    elif task == "hitab":
        res = score_hitab(rows)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    metrics = res.to_dict()
    out_path = Path(args.out_metrics) if args.out_metrics else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics: {out_path}")
    print(json.dumps(metrics, indent=2))


def cmd_preprocess(args: argparse.Namespace) -> None:
    # Minimal placeholder: keep legacy preprocess under tools/ for now.
    raise SystemExit(
        "preprocess has been moved to tools/preprocess/legacy/ for this repo. "
        "If you need a unified preprocessing pipeline, extend llada/cli/main.py."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llada")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer", help="run inference and write predictions jsonl")
    pi.add_argument("--task", type=str, required=True, choices=["gsm8k", "openscience", "hitab"])
    pi.add_argument("--split", type=str, default="test")
    pi.add_argument("--data_jsonl", type=str, default="", help="for hitab: path to jsonl with prompt + gold")
    pi.add_argument("--start_index", type=int, default=5000, help="openscience start index (default matches legacy)")
    pi.add_argument("--end_index", type=int, default=6000, help="openscience end index (default matches legacy)")
    pi.add_argument("--max_samples", type=int, default=None)

    pi.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    pi.add_argument("--checkpoint_path", type=str, default="")
    pi.add_argument("--device_ids", type=int, nargs="+", default=[0])

    pi.add_argument("--batch_size", type=int, default=16)
    pi.add_argument("--temperature", type=float, default=0.0)
    pi.add_argument("--gen_length", type=int, default=128)
    pi.add_argument("--steps", type=int, default=128)
    pi.add_argument("--block_length", type=int, default=32)
    pi.add_argument("--cfg_scale", type=float, default=0.0)
    pi.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    pi.add_argument("--mask_id", type=int, default=126336)
    pi.add_argument("--out_file", type=str, required=True)
    pi.set_defaults(func=cmd_infer)

    ps = sub.add_parser("score", help="score an existing predictions jsonl")
    ps.add_argument("--task", type=str, required=True, choices=["gsm8k", "openscience", "hitab"])
    ps.add_argument("--pred_jsonl", type=str, required=True)
    ps.add_argument("--out_metrics", type=str, default="")
    ps.set_defaults(func=cmd_score)

    pp = sub.add_parser("preprocess", help="(placeholder) see tools/preprocess/legacy/")
    pp.set_defaults(func=cmd_preprocess)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
