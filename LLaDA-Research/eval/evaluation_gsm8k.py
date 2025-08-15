import json
import re
import pandas as pd

def normalize_answer(ans):
    if not isinstance(ans, str):
        return ans
    ans = ans.strip().lower()
    ans = ans.replace(",", "")  # 去除千位分隔符
    ans = ans.replace("$", "")  # 去除美元符号
    ans = ans.replace("\\%", "").replace("%", "").replace("\\", "")
    ans = re.sub(r"[^\d\.\-+]", "", ans)  # 保留数字、小数点、负号
    return ans

def loose_match(pred, gt, tol=1e-3):
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < tol
    except:
        return pred_norm == gt_norm

def extract_final_answer(text):
    if not isinstance(text, str):
        return ""

    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\$?([-+]?\d[\d,]*(?:\.\d+)?)\$?",
        r"<<[^=]*=([^>]+)>>",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            value = matches[-1].strip()
            return normalize_answer(value)

    # fallback：最后一行中的最后一个数字
    lines = text.strip().splitlines()
    for line in reversed(lines):
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", line)
        if nums:
            return normalize_answer(nums[-1])
    return ""

def f1_score(correct, pd_num, gt_num):
    precision = correct / pd_num if pd_num else 0
    recall = correct / gt_num if gt_num else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

def evaluate_gsm8k(jsonl_path, output_jsonl="combined_predictions_groundtruth_gsm8k_normal.jsonl", max_data=None):
    total_correct = total_pd = total_gt = 0
    sample_count = extracted_sample_count = 0
    combined_records = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_data is not None and sample_count >= max_data:
                break
            sample = json.loads(line)
            sample_count += 1

            pred_raw = sample.get("prediction", "")
            gt_raw = sample.get("answer", "")

            pred = extract_final_answer(pred_raw)
            gt = extract_final_answer(gt_raw)

            if pred:
                extracted_sample_count += 1

            is_match = loose_match(pred, gt)
            if is_match:
                total_correct += 1

            total_pd += 1
            total_gt += 1

            # ✅ 只保留 prediction 和 groundtruth（已归一化）
            combined_records.append({
                "prediction": pred,
                "groundtruth": gt,
            })

    # 写入简化版 JSONL 文件
    with open(output_jsonl, 'w', encoding='utf-8') as fw:
        for record in combined_records:
            json.dump(record, fw, ensure_ascii=False)
            fw.write('\n')

    acc = total_correct / total_gt if total_gt else 0
    prec = total_correct / total_pd if total_pd else 0
    f1 = f1_score(total_correct, total_pd, total_gt)
    return {
        "correct": total_correct,
        "predicted": total_pd,
        "groundtruth": total_gt,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "prediction%": round(total_pd / total_gt if total_gt else 0, 4),
        "extract_rate": round(extracted_sample_count / sample_count, 4),
    }

if __name__ == "__main__":
    tasks = [
        "only_ms2_seed44", "only_ms2_seed45"
    ]
    rows = []
    for task in tasks:
        if task == "only_ms2_seed44":
            input_file = "/storage/v-mengnijia/LLaDA/eval/data/seed44_instruct_gsm8k_MultiSample_ns2_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250731_045711/checkpoint-epoch5/predictions_gsm8k_test.jsonl"
            output_file = "/storage/v-mengnijia/LLaDA/eval/combined_predictions_groundtruth_gsm8k_only_ms2_seed44.jsonl"
            rows.append(evaluate_gsm8k(input_file, output_file))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
