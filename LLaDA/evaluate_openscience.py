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
BATCH_SIZE       = 4
MODEL_NAME       = "GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME     = "nvidia/OpenScienceReasoning-2"
SPLIT            = "train"
START_INDEX      = 7000
END_INDEX        = 8000
TEMP             = 0.
GEN_LENGTH       = 256
STEPS            = 256
BLOCK_LENGTH     = 256
BASE_OUTPUT      = Path("/root/workspace/data/eval")
suffix           = Path(*Path(CHECKPOINT_PATH).parts[-2:])
OUTPUT_PATH      = BASE_OUTPUT / suffix / f"predictions_openscience_256_256_256.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============== 工具函数 ======================
def extract_boxed_answer(text):
    # 匹配 \boxed{...}
    if not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{([A-Za-z0-9\+\-\.\, ]+)\}", text)
    return m.group(1).strip() if m else None

# ============== 1. 加载 tokenizer / model ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading model on GPUs: {DEVICE_IDS} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    torch_dtype="auto"
)

model = torch.nn.DataParallel(base_model, device_ids=DEVICE_IDS)
model.eval().cuda()

# ============== 2. 加载数据 ======================
dataset = load_dataset(DATASET_NAME, split=SPLIT)
dataset = dataset.select(range(START_INDEX, END_INDEX))
assert "input" in dataset[0] and "output" in dataset[0]
print(f"✔ Loaded OpenScience samples: {len(dataset)}")

progress = tqdm(total=len(dataset), desc="Samples", unit="example")
correct = 0
total = 0

# ============== 3. 推理 ======================
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]   # 注意：这里 batch 是 dict of lists

        inputs = batch["input"]
        outputs = batch["output"]

        # ======= 构造 batch 输入 =======
        prompts = []
        for inp in inputs:
            m = [{"role": "user", "content": inp}]
            p = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            prompts.append(p)

        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

        # ======= 多卡推理 =======
        with torch.no_grad():
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

        # ======= 处理 batch 内每个样本 =======
        for inp, gold_cot_output, pred_text in zip(inputs, outputs, decoded):
            gold_answer = extract_boxed_answer(gold_cot_output)
            pred_answer = extract_boxed_answer(pred_text)

            is_correct = (gold_answer == pred_answer)
            total += 1
            correct += int(is_correct)

            fout.write(json.dumps({
                "input": inp,
                "gold_output": gold_cot_output,
                "gold_answer": gold_answer,
                "prediction": pred_text,
                "pred_answer": pred_answer,
                "correct": is_correct
            }, ensure_ascii=False) + "\n")

            progress.update(1)

progress.close()
acc = correct / total if total > 0 else 0
print(f"✔ 完成推理！共 {total} 条，正确 {correct} 条，Accuracy = {acc:.4f}")
print(f"✔ 结果已保存至：{OUTPUT_PATH}")
