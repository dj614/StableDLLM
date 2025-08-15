#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

accelerate launch \
  --config_file /storage/v-mengnijia/LLaDA/accelerate_ds.yaml \
  --main_process_port 29501 \
  /storage/v-mengnijia/LLaDA/before_seeding.py \
  --task hitab \
  --no_infer \
  --train_mode MirrorMask
  