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
DATASET_NAME     = "nvidia/OpenScience"
SPLIT            = "train"     # openscience 本身无 test
START_INDEX      = 5000        # 第 5001 条（0-based）
END_INDEX        = 6000        # 第 6000 条（不含）
BATCH_SIZE       = 4
TEMP             = 0.
GEN_LENGTH       = 256
STEPS            = 256
BLOCK_LENGTH     = 256
BASE_OUTPUT      = Path("LLaDA/eval/data")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:])
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_openscience_256_256_256.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 解析 \boxed{...} 里的答案
def extract_boxed_answer(text):
    m = re.search(r"\\boxed\{([A-Za-z0-9\+\-\.\, ]+)\}", text)
    if m:
        return m.group(1).strip()
    return None

# 从模型生成内容中解析答案（可能生成类似 "The answer is D."）
def extract_pred_answer(pred):
    # 尝试找字母答案
    m = re.search(r"\\boxed\{([A-Za-z0-9\+\-\.\, ]+)\}", pred)
    if m:
        return m.group(1).strip()
    # 失败则返回全文，让你 debug
    return None

# 1. 加载 tokenizer / model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True, torch_dtype="auto")
model.eval().to(device)

# 2. 加载 openscience
dataset = load_dataset(DATASET_NAME, split=SPLIT)
dataset = dataset.select(range(START_INDEX, END_INDEX))

print(f"✔ Loaded OpenScience samples: {len(dataset)}")

# 检查 keys
assert "input" in dataset[0] and "output" in dataset[0], "OpenScience 格式不符合 input/output！"

progress = tqdm(total=len(dataset), desc="Samples", unit="example")

correct = 0
total = 0

# 3. 推理并保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i: i + BATCH_SIZE]
        for item in batch:

            question = item["input"]
            gold_cot_output = item["output"]
            gold_answer = extract_boxed_answer(gold_cot_output)

            # 构造 prompt
            m = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # 推理
            out = generate(
                model, input_ids,
                steps=STEPS,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=TEMP,
                cfg_scale=0.,
                remasking="low_confidence"
            )
            pred_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            pred_answer = extract_pred_answer(pred_text)

            # 对比答案
            is_correct = (gold_answer == pred_answer)

            total += 1
            correct += int(is_correct)

            fout.write(json.dumps({
                "input": question,
                "gold_output": gold_cot_output,
                "gold_answer": gold_answer,
                "prediction": pred_text,
                "pred_answer": pred_answer,
                "correct": is_correct
            }, ensure_ascii=False) + "\n")

            progress.update(1)

progress.close()

acc = correct / total
print(f"✔ 完成推理！共 {total} 条，正确 {correct} 条，Accuracy = {acc:.4f}")
print(f"✔ 结果已保存至：{OUTPUT_PATH}")
