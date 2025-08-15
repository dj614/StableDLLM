from tqdm.auto import tqdm
from pathlib import Path
import json
import torch
import argparse

from dkv_cache.models.modeling_llada_dkv_cache_decode import LLaDAModelLM
from generation_utils.llada_dkv_cache_decode import generate
from transformers import AutoTokenizer

# python3 inference_hitab.py --task="hitab" --ckpt="/storage/result/checkpoints/LLaDA/seed43_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_121459/checkpoint-epoch5" --device="cuda:1"
# python3 inference_hitab.py --task="hitab" --ckpt="/storage/result/checkpoints/LLaDA/seed42_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_062335/checkpoint-epoch5" --device="cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# ------------ 可自行修改的超参 ------------
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
ckpt             = args.ckpt
DATA_PATH        = "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl"
BATCH_SIZE       = 2       # 每 GPU 同时处理几条 prompt
MAX_DATA         = None
TEMP             = 0.
GEN_LENGTH       = 512
STEPS            = 256
BLOCK_LENGTH     = 16
BASE_OUTPUT      = Path("/storage/v-mengnijia/LLaDA/eval/data")
if ckpt:
    suffix       = Path(*Path(ckpt).parts[-2:])
    OUTPUT_PATH  = BASE_OUTPUT / suffix / f"predictions_{args.task}_temp{TEMP}_gen{GEN_LENGTH}_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"
else:
    OUTPUT_PATH  = BASE_OUTPUT / f"predictions_{args.task}_temp{TEMP}_gen{GEN_LENGTH}_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
device    = torch.device(args.device)
# --------------------------------------

# 1. 加载 tokenizer / model
load_path = ckpt if ckpt else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(
    load_path,
    trust_remote_code=True,
    torch_dtype="auto"
).eval().to(device)

# 2. 读取数据集并按 batch_size 分块
def read_batches(path, bs):
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(json.loads(line))
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:
            yield buf

total_samples = sum(1 for _ in open(DATA_PATH, encoding="utf-8"))
if MAX_DATA is not None:
    total_samples = min(total_samples, MAX_DATA)
progress = tqdm(total=total_samples, desc="Samples", unit="example")

processed = 0

# 3. 批量推理 + 保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for batch in read_batches(DATA_PATH, BATCH_SIZE):
        if MAX_DATA is not None and processed >= MAX_DATA:
            break

        # -- build a list of chat‐messages for this batch
        ms = []
        for item in batch:
            ms.append([{"role": "user", "content": item["prompt"]}])

        # -- apply your chat‐template in one go, then tokenize with padding
        prompts = tokenizer.apply_chat_template(ms, add_generation_prompt=True, tokenize=False)
        enc = tokenizer(
            prompts,
            padding_side="left",
            padding="longest",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)

        # -- single batch call to generate()
        outs = generate(
            model,
            tokenizer,
            input_ids,
            steps=STEPS,
            gen_length=GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            temperature=TEMP,
            cfg_scale=0.0,
            remasking="low_confidence",
            enable_cache=True,
            cache_reloading_step=8,
        )

        # -- decode all at once
        answers = tokenizer.batch_decode(
            outs[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # -- write each to disk
        for item, ans in zip(batch, answers):
            item["prediction"] = ans
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            processed += 1
            progress.update(1)

    progress.close()
    print(f"✔ All done! 结果已保存到 {OUTPUT_PATH}")
