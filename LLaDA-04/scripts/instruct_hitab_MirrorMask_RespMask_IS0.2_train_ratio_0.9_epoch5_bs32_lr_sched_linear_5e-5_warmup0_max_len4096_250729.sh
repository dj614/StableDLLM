#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

# ========== 启动并运行 train_ema_baseline.py ==========
accelerate launch \
  --main_process_port=29501 \
  --config_file accelerate_ds.yaml \
  train_ema_baseline.py \
  --task hitab \
  --do_infer \
  --infer_data_path /storage/v-mengnijia/LLaDA/data.jsonl \
  --max_data 1412 \
  --temp 0.0 \
  --gen_length 512 \
  --steps 256 \
  --block_length 16 \
  --pretrained_path GSAI-ML/LLaDA-8B-Instruct \
  --train_data_path /storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_processed.jsonl \
  --max_len 4096 \
  --coord_format hitab-html \
  --mask_mode RespMask \
  --m 0.0 \
  --train_mode MirrorMask \
  --num_samples 8 \
  --num_bins 10 \
  --baseline_lr 0.01 \
  --mask_ratio_mode random \
  --mask_strata_bins 16 \
  --IS \
  --delta=0.2 \
  --compare_tok_grads \
  --common_ids 268,341,301,296,297,352,300,468,3742,259 \
  --rare_ids 59,795,32289,90,28504,7684 \
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
  --save_strategy last \
  --save_steps 100
