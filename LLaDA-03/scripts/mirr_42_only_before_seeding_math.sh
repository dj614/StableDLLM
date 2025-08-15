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
  --do_infer \
  --infer_data_path /storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl \
  --gen_length 256 \
  --block_length 32 \
  --steps 128 \
  --temp 0.0 \
  --pretrained_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data_path /storage/v-mengnijia/LLaDA/data/sft/hitab_reasoning_sft_str_preprocessed.jsonl \
  --train_mode MirrorMask \
  --epochs 5 \
  --train_ratio 0.9 \
  --batch_size 2 \
  --grad_accum 16 
