#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ========== 启动并运行 train_and_infer.py ==========
# ========== output_dir, logging_steps, common_ids, rare_ids 交给默认 ==========
accelerate launch \
  --config_file accelerate_ds.yaml \
  train_and_infer.py \
  --seed 44 \
  --task hitab \
  --no_infer \
  --max_data 1412 \
  --temp 0.0 \
  --gen_length 512 \
  --steps 256 \
  --block_length 16 \
  --batch_size_infer 2 \
  --pretrained_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data_path /storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl \
  --max_len 4096 \
  --coord_format hitab-html \
  --mask_mode RespMask \
  --m 0.0 \
  --train_mode Normal \
  --mask_ratio_mode random \
  --epochs 5 \
  --train_ratio 0.9 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr_scheduler_type linear \
  --lr 5e-5 \
  --warmup_steps 0 \
  --decay_ratio 0.1 \
  --final_lr 2.7e-6 \
  --eval_strategy epoch \
  --eval_steps 100 \
  --save_strategy last \
  --save_steps 100 \
  --compare_tok_grads \
  --hetero_t_in_l
