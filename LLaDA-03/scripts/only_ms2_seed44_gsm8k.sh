#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ========== 启动并运行 train_and_infer.py ==========
accelerate launch \
  --config_file accelerate_ds.yaml \
  train_and_infer.py \
  --seed 44 \
  --task gsm8k \
  --do_infer \
  --pretrained_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data_path /storage/v-mengnijia/LLaDA/gsm8k_reasoning_sft_str_processed.jsonl \
  --max_len 4096 \
  --mask_mode RespMask \
  --train_mode MultiSample \
  --num_samples 2 \
  --mask_ratio_mode random \
  --epochs 5 \
  --train_ratio 0.9 \
  --batch_size 16 \
  --grad_accum 2 \
  --lr_scheduler_type linear \
  --lr 5e-5 \
  --warmup_steps 0 \
  --eval_strategy epoch \
  --save_strategy last \
  --compare_tok_grads
