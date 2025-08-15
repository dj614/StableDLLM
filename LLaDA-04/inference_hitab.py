from tqdm.auto import tqdm
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate import generate

# ------------ 可自行修改的超参 ------------
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
CHECKPOINT_PATH  = "/storage/result/checkpoints/LLaDA/seed43_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_101810/checkpoint-epoch5"
DATA_PATH        = "/storage/v-mengnijia/LLaDA/data.jsonl"
BATCH_SIZE       = 16       # 每 GPU 同时处理几条 prompt
MAX_DATA         = None
TASK             = "hitab"
TEMP             = 0.
GEN_LENGTH       = 512
STEPS            = 256
BLOCK_LENGTH     = 16
BASE_OUTPUT      = Path("/storage/v-mengnijia/LLaDA/eval/data")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:])
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_{TASK}_temp{TEMP}_gen{GEN_LENGTH}_steps{STEPS}_block{BLOCK_LENGTH}.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
device    = torch.device("cuda:1")
# --------------------------------------

# 1. 加载 tokenizer / model
load_path = CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True, torch_dtype="auto")
model.eval().to(device)

# 2. 读取数据集并按 batch_size 分块
def read_batches(path, bs):
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(json.loads(line))
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:                              # 处理最后一个不足 batch 的残包
            yield buf

total_samples = sum(1 for _ in open(DATA_PATH, encoding="utf-8"))  # 1412
if MAX_DATA is not None:
    total_samples = min(total_samples, MAX_DATA)
progress = tqdm(total=total_samples, desc="Samples", unit="example")

processed = 0  # 已处理样本计数

# 3. 推理 + 保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for batch in read_batches(DATA_PATH, BATCH_SIZE):
        for i, item in enumerate(batch):
            if MAX_DATA is not None and processed >= MAX_DATA:
                break
            prompt = item["prompt"]
            m = [{"role": "user", "content": prompt}, ]
            prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
            out = generate(model, input_ids, steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH, temperature=TEMP, cfg_scale=0., remasking='low_confidence')
            ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            # 写回 jsonl：保留 prompt / ground_truth，新增 prediction
            sample = batch[i]
            sample["prediction"] = ans
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            processed += 1
            progress.update(1)
        
        if MAX_DATA is not None and processed >= MAX_DATA:
            break

progress.close()
print(f"✔ All done! 结果已保存到 {OUTPUT_PATH}")