from tqdm.auto import tqdm 
from pathlib import Path
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from generate import generate

# ------------ 可修改的超参 ------------
CHECKPOINT_PATH  = ""
device           = torch.device("cuda:0")

# ----------- 不需要修改的超参 ----------
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME     = "openai/gsm8k"
DATASET_CONFIG   = "main"
SPLIT            = "test"
BATCH_SIZE       = 16
TEMP             = 0.
GEN_LENGTH       = 128
STEPS            = 128
BLOCK_LENGTH     = 32
BASE_OUTPUT      = Path("LLaDA/eval/data")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:])
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_gsm8k_128_128_32.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 解析标准答案：从 "#### 18" 提取数字 18
def extract_gold_answer(answer_text):
    m = re.search(r"####\s*([-+]?\d+)", answer_text)
    if m:
        return m.group(1).strip()
    return None

# 解析模型预测答案：从生成内容中提取第一个整数
def extract_pred_answer(pred_text):
    m = re.search(r"####\s*([-+]?\d+)", pred_text)
    if m:
        return m.group(1).strip()
    return None   # 严格模式：解析不出 → 算错

# 1. 加载 tokenizer / model
load_path = CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True, torch_dtype="auto")
model.eval().to(device)

# 2. 加载 GSM8K 数据集
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)
progress = tqdm(total=len(dataset), desc="Samples", unit="example")

correct = 0
total = 0

# 3. 推理 + 保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i: i + BATCH_SIZE]
        batch_items = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        for item in batch_items:
            question = item["question"]
            gold_answer_text = item["answer"]
            gold_answer = extract_gold_answer(gold_answer_text)

            # 构建 prompt
            m = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # 推理
            out = generate(
                model,
                input_ids,
                steps=STEPS,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=TEMP,
                cfg_scale=0.,
                remasking="low_confidence"
            )
            pred_text = tokenizer.batch_decode(
                out[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            pred_answer = extract_pred_answer(pred_text)

            # 严格模式：无法解析 or 不相等 → 错
            is_correct = (gold_answer is not None) and (pred_answer == gold_answer)

            total += 1
            correct += int(is_correct)

            fout.write(json.dumps({
                "question": question,
                "gold_answer": gold_answer,
                "gold_raw": gold_answer_text,
                "prediction": pred_text,
                "pred_answer": pred_answer,
                "correct": is_correct
            }, ensure_ascii=False) + "\n")

            progress.update(1)

progress.close()

acc = correct / total
print(f"✔ 评测完成，共 {total} 条，正确 {correct} 条，Accuracy = {acc:.4f}")
print(f"✔ 结果已保存到 {OUTPUT_PATH}")
