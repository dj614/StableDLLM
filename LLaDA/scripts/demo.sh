#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"""
accelerate launch \
  --config_file /storage/v-mengnijia/LLaDA/accelerate_ds.yaml \
  /storage/v-mengnijia/LLaDA/rebuttal.py \
  --seed XXX \ (42, 731, 20231)
  --task XXX \ (openscience, gsm8k)
  --train_mode mirror_plus \
  --IS_on_t \
  --batch_size_per_gpu XXX\
  --grad_accum XXX
"""

"""
batch_size_per_gpu * grad_accum * num_gpus_per_machine = 32
"""