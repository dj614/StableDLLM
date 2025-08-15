#!/usr/bin/env bash

# ========== 环境变量 ==========
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ========== 启动并运行 train_and_infer.py ==========
# /storage/result/checkpoints/LLaDA/noseed_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250807_071252/checkpoint-epoch5
# ========== output_dir, logging_steps, common_ids, rare_ids 交给默认 ==========
accelerate launch \
  --config_file accelerate_ds.yaml \
  --main_process_port 29501 \
  train_and_infer_no_seed.py \
  --task hitab \
  --do_infer \
  --train_mode MirrorMask \
  --IS \
  --delta=0.2 \
  --batch_size 1 \
  --grad_accum 32 \
  --lr_scheduler_type linear \
  --lr 5e-5 \
  --compare_tok_grads
