#!/usr/bin/env bash
set -euo pipefail

# Minimal reproducible GSM8K eval using the unified CLI.
# Usage:
#   bash scripts/run_gsm8k_eval.sh /path/to/checkpoint 0 1 2 3
#
# If checkpoint is empty, base model is used.

CKPT="${1:-}"
shift || true

# Remaining args are device ids, e.g. 0 1 2 3
DEVICE_IDS=("$@")
if [ "${#DEVICE_IDS[@]}" -eq 0 ]; then
  DEVICE_IDS=(0)
fi

OUT_DIR="outputs/eval"
mkdir -p "${OUT_DIR}"

SUFFIX="base"
if [ -n "${CKPT}" ]; then
  # take last 2 path components as a readable suffix
  SUFFIX="$(basename "$(dirname "${CKPT}")")/$(basename "${CKPT}")"
  SUFFIX="${SUFFIX//\//_}"
fi

PRED_JSONL="${OUT_DIR}/predictions_gsm8k_${SUFFIX}.jsonl"
METRICS_JSON="${OUT_DIR}/predictions_gsm8k_${SUFFIX}.metrics.json"

PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
  python -m llada.cli.main infer \
    --task gsm8k --split test \
    --checkpoint_path "${CKPT}" \
    --device_ids "${DEVICE_IDS[@]}" \
    --batch_size 16 --temperature 0 \
    --gen_length 128 --steps 128 --block_length 32 \
    --out_file "${PRED_JSONL}"

PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" \
  python -m llada.cli.main score \
    --task gsm8k \
    --pred_jsonl "${PRED_JSONL}" \
    --out_metrics "${METRICS_JSON}"

echo "Predictions: ${PRED_JSONL}"
echo "Metrics:     ${METRICS_JSON}"
