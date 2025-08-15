#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

accelerate launch --main_process_port=29501 --config_file accelerate_ds.yaml   train_ema_baseline.py --task="gsm8k" --train_mode="MirrorMask" --mask_mode="RespMask" --train_data_path="/storage/v-mengnijia/LLaDA/gsm8k_reasoning_sft_str_processed.jsonl" --train_ratio=0.9 --pretrained_path="GSAI-ML/LLaDA-8B-Instruct" --epochs=5 --batch_size=16 --grad_accum=2 --lr_scheduler_type="linear" --lr=5e-5 --max_len=4096 --save_strategy="last" --no_infer