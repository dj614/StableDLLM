from tqdm.auto import tqdm
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generate import generate

# ------------ 可自行修改的超参 ------------
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
CHECKPOINT_PATH  = "/storage/result/checkpoints/LLaDA/seed43_instruct_gsm8k_Normal_IS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_052124/checkpoint-epoch5"
DATASET_NAME     = "openai/gsm8k"
DATASET_CONFIG   = "main"
SPLIT            = "test"
BATCH_SIZE       = 16
MAX_DATA         = None
TEMP             = 0.
GEN_LENGTH       = 256
STEPS            = 256
BLOCK_LENGTH     = 16
BASE_OUTPUT      = Path("/storage/v-mengnijia/LLaDA/eval/data")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:])
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_gsm8k_{SPLIT}_temp_{TEMP}_gen{GEN_LENGTH}_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
device    = torch.device("cuda:1")
# --------------------------------------

# 1. 加载 tokenizer / model
load_path = CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True, torch_dtype="auto")
model.eval().to(device)

# 2. 加载 GSM8K 数据集
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
if MAX_DATA is not None:
    dataset = dataset.select(range(MAX_DATA))
progress = tqdm(total=len(dataset), desc="Samples", unit="example")

# 3. 推理 + 保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i: i + BATCH_SIZE]
        batch_items = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        for item in batch_items:
            question = item["question"]
            m = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
            out = generate(model, input_ids, steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH, temperature=TEMP, cfg_scale=0., remasking="low_confidence")
            ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            sample = {
                "question": item["question"],
                "answer": item["answer"],
                "prediction": ans
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            progress.update(1)

progress.close()
print(f"✔ All done! 结果已保存到 {OUTPUT_PATH}")
