#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ========== 启动并运行 train_and_infer.py ==========
# /storage/result/checkpoints/LLaDA/seed42_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_182237/checkpoint-epoch5
# ========== output_dir, logging_steps, common_ids, rare_ids 交给默认 ==========
accelerate launch \
  --config_file accelerate_ds.yaml \
  --main_process_port 29501 \
  train_and_infer.py \
  --seed 42 \
  --task hitab \
  --pretrained_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data_path /storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl \
  --max_len 4096 \
  --coord_format hitab-html \
  --mask_mode RespMask \
  --m 0.0 \
  --train_mode MirrorMask \
  --IS \
  --epochs 5 \
  --train_ratio 0.9 \
  --batch_size 1 \
  --grad_accum 32 \
  --lr_scheduler_type linear \
  --lr 5e-5 \
  --warmup_steps 0 \
  --decay_ratio 0.1 \
  --final_lr 2.7e-6 \
  --eval_strategy epoch \
  --eval_steps 100 \
  --save_strategy last \
  --save_steps 100 \
  --compare_tok_grads
