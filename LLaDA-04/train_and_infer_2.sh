#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

accelerate launch --main_process_port=29501 --config_file accelerate_ds.yaml   ttttt.py --pretrained_path="GSAI-ML/LLaDA-8B-Base" --data_path="/storage/v-mengnijia/LLaDA/CodeXGLUE/Code-Text/code-to-text/llada_code2text/test.jsonl" --gen_length=128 --steps=64 --block_length=4 --train_mode="RespMask" --m=0.0 --data_file="/storage/v-mengnijia/LLaDA/codexglue_reasoning_sft_str_processed.jsonl" --coord_format="codexglue-json" --epochs=5 --batch_size=2 --grad_accum=16 --logging_steps=10 --lr_scheduler_type="linear" --lr=2e-5 --max_len=4096 --output_dir="llada-base-respmask-codexglue"