#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ========== 启动并运行 train_and_infer.py ==========
# ========== output_dir, logging_steps, common_ids, rare_ids 交给默认 ==========
accelerate launch \
  --config_file /storage/v-mengnijia/LLaDA/accelerate_ds.yaml \
  --main_process_port 29501 \
  /storage/v-mengnijia/LLaDA/train_and_infer.py \
  --seed 42 \
  --task hitab \
  --model llada \
  --do_infer \
  --train_mode MirrorMask \
  --epochs 5 \
  --loss_max 10 \
  --train_ratio 0.9 \
  --batch_size 1 \
  --grad_accum 32 \
  --hetero_t_in_l
