"""CLI argument parsing for the training runner.

This file exists so that executable scripts remain thin wrappers.
"""

from __future__ import annotations

import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--china", action="store_true", help="enable hf-mirror.com endpoint")
    ap.add_argument("--seed", type=int, required=True, help="global random seed")
    ap.add_argument("--task", type=str, choices=["openscience", "gsm8k", "hitab"], required=True)
    ap.add_argument("--model", type=str, choices=["llada"], default="llada")
    ap.add_argument("--train_mode", type=str, choices=["Normal", "MIRROR"], default="Normal")
    ap.add_argument("--PPOTS", action="store_true")
    ap.add_argument("--p_model", type=str, choices=["EPR"], default="EPR", help="select p(t) parameterization")
    ap.add_argument("--max_tokens_per_forward", type=int, default=60000)
    ap.add_argument("--max_is_weight", type=float, default=1e6)
    ap.add_argument("--n_starts", type=int, default=20)
    ap.add_argument("--num_samples_x0", type=int, default=10)
    ap.add_argument("--num_samples_t", type=int, default=30)
    ap.add_argument("--num_samples_xt", type=int, default=10)
    ap.add_argument("--mix_uniform", type=float, default=0.0)
    ap.add_argument(
        "--loss_max",
        type=float,
        default=10.0,
        help="(PPOTS) sample-level loss clamp used when estimating w(t)",
    )
    ap.add_argument("--train_data_path", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--batch_size_per_gpu", type=int, required=True)
    ap.add_argument("--grad_accum", type=int, required=True)
    ap.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        default="linear",
    )
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--eval_strategy", type=str, choices=["epoch", "steps", "no"], default="epoch")
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_strategy", type=str, choices=["epoch", "steps", "last"], default="last")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--logging_steps", type=int, default=5)

    args = ap.parse_args()

    if args.eval_strategy is None:
        args.eval_strategy = args.save_strategy
    if args.eval_steps is None:
        args.eval_steps = args.save_steps

    if args.train_data_path is None:
        args.train_data_path = f"./data/train/{args.task}.jsonl"
    return args
