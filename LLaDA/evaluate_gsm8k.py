from tqdm.auto import tqdm 
from pathlib import Path
import json
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generate import generate

# ------------ 解析命令行参数 ------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="", help="path to finetuned checkpoint (optional)")
parser.add_argument("--device_ids", type=int, nargs="+", default=[0,1,2,3,4,5,6,7],
                    help="gpu ids for DataParallel, e.g. --device_ids 0 1 2 3")
args = parser.parse_args()

# ------------ 可修改的超参 ------------
CHECKPOINT_PATH  = args.checkpoint_path
DEVICE_IDS       = args.device_ids

# ----------- 不需要修改的超参 ----------
BATCH_SIZE       = 16
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME     = "openai/gsm8k"
DATASET_CONFIG   = "main"
SPLIT            = "test"
TEMP             = 0.
GEN_LENGTH       = 128
STEPS            = 128
BLOCK_LENGTH     = 32
BASE_OUTPUT      = Path("/root/workspace/data/eval")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:]) if CHECKPOINT_PATH else Path("base")
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_gsm8k_128_128_32.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ================= 工具函数 =================
def extract_hash_answer(text):
    """提取 #### 后的整数答案"""
    if not isinstance(text, str):
        return None
    m = re.search(r"####\s*([-+]?\d+)", text)
    return m.group(1).strip() if m else None

# ================= 1. 加载模型 =================
load_path = CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

print(f"Loading model on GPUs: {DEVICE_IDS} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    load_path,
    trust_remote_code=True,
    torch_dtype="auto",
)

model = torch.nn.DataParallel(base_model, device_ids=DEVICE_IDS)
model.eval().cuda()

# ================= 2. 加载数据 =================
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
progress = tqdm(total=len(dataset), desc="Samples", unit="example")

correct = 0
total = 0

# ================= 3. 推理 + 保存 =================
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]  # dict of lists
        questions = batch["question"]
        gold_raws = batch["answer"]

        # ====== 构造 batch prompts ======
        prompts = []
        for q in questions:
            msgs = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)

        # ====== tokenize batch ======
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        # ====== 多卡推理 ======
        with torch.no_grad():
            curr_bs = encoded["input_ids"].size(0)
            n_gpus = len(DEVICE_IDS)
            if curr_bs < n_gpus:
                # 最后一个小 batch：不用 DataParallel，单卡跑
                out = generate(
                    model.module,  # 取出真实模型
                    encoded["input_ids"].to("cuda:0"),
                    steps=STEPS,
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH,
                    temperature=TEMP,
                    cfg_scale=0.,
                    remasking="low_confidence"
                )
            else:
                # 正常 batch：多卡 DP 跑
                out = generate(
                    model,
                    encoded["input_ids"],
                    steps=STEPS,
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH,
                    temperature=TEMP,
                    cfg_scale=0.,
                    remasking="low_confidence"
                )

        decoded = tokenizer.batch_decode(
            out[:, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # ====== 逐条处理结果 ======
        for q, gold_raw, pred_text in zip(questions, gold_raws, decoded):
            gold_answer = extract_hash_answer(gold_raw)
            pred_answer = extract_hash_answer(pred_text)

            # pred 提取不到 => 直接错；gold 提取不到也判错（按你原逻辑）
            is_correct = (gold_answer is not None) and (pred_answer == gold_answer)

            total += 1
            correct += int(is_correct)

            fout.write(json.dumps({
                "question": q,
                "gold_answer": gold_answer,
                "gold_raw": gold_raw,
                "prediction": pred_text,
                "pred_answer": pred_answer,
                "correct": is_correct
            }, ensure_ascii=False) + "\n")

            progress.update(1)

progress.close()
acc = correct / total if total > 0 else 0
print(f"✔ 评测完成，共 {total} 条，正确 {correct} 条，Accuracy = {acc:.4f}")
print(f"✔ 结果已保存到 {OUTPUT_PATH}")
