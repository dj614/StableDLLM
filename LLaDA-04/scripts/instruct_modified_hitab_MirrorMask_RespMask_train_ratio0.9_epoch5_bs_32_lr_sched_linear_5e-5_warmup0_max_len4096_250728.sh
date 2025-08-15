#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16777216
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

accelerate launch --config_file accelerate_ds.yaml   train_ema_baseline.py --task="hitab_modified" --do_infer --infer_data_path="/storage/v-mengnijia/LLaDA/data.jsonl" --gen_length=512 --steps=256 --block_length=16 --train_mode="MirrorMask" --mask_mode="RespMask" --coord_format="hitab-html" --train_data_path="/storage/v-mengnijia/LLaDA/hitab_reasoning_sft_str_modified_processed.jsonl" --train_ratio=0.9 --pretrained_path="GSAI-ML/LLaDA-8B-Instruct" --epochs=5 --batch_size=2 --grad_accum=16 --lr_scheduler_type="linear" --lr=5e-5 --max_len=4096 --save_strategy="last" --compare_tok_grads --rare_ids="2262,71307"