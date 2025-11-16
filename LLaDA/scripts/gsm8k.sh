#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8


accelerate launch \
  --config_file LLaDA/accelerate_ds.yaml \
  --main_process_port 29500 \
  LLaDA/rebuttal.py \
  --seed 42 \
  --task gsm8k \
  --train_mode mirror_plus \
  --IS_on_t \
  --batch_size_per_gpu 8 \
  --grad_accum 2

accelerate launch \
  --config_file LLaDA/accelerate_ds.yaml \
  --main_process_port 29500 \
  LLaDA/rebuttal.py \
  --seed 731 \
  --task gsm8k \
  --train_mode mirror_plus \
  --IS_on_t \
  --batch_size_per_gpu 8 \
  --grad_accum 2

accelerate launch \
  --config_file LLaDA/accelerate_ds.yaml \
  --main_process_port 29500 \
  LLaDA/rebuttal.py \
  --seed 20231 \
  --task gsm8k \
  --train_mode mirror_plus \
  --IS_on_t \
  --batch_size_per_gpu 8 \
  --grad_accum 2