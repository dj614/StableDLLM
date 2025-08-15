import json
import re
import pandas as pd

def normalize_answer(ans):
    """Clean and normalize an answer for loose matching."""
    if isinstance(ans, str):
        # Remove LaTeX \text{}, \frac{}, and similar patterns
        ans = re.sub(r'\\text\s*\{', '', ans)
        ans = re.sub(r'\\frac\s*\{', '', ans)
        ans = re.sub(r'\}', '', ans)
        ans = ans.replace("\\%", "").replace("%", "").replace("\\", "")
        ans = ans.strip().lower()
    return ans

def loose_match(pred, gt, tol=1e-2):
    """Loose match between prediction and groundtruth element."""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < tol
    except:
        # fallback to string comparison
        return pred_norm == gt_norm

def label_choose(results):
    """Select the best label set with highest F1."""
    best_label = results[0]
    best_f1 = f1_score(*best_label)
    for r in results:
        f1 = f1_score(*r)
        if f1 > best_f1:
            best_label = r
            best_f1 = f1
    return best_label

def f1_score(correct, pd_num, gt_num):
    precision = correct / pd_num if pd_num != 0 else 0
    recall = correct / gt_num if gt_num != 0 else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

def cal_metric(pds, gts):
    pd_num = len(pds)
    gt_num = len(gts)

    # Handle empty edge cases first
    if pd_num == 0 and gt_num == 0:
        return (0, 0, 0)
    if pd_num == 0:
        return (0, 0, gt_num)
    if gt_num == 0:
        return (0, pd_num, 0)

    # 每个 ground truth 只能被匹配一次
    matched = [False] * gt_num
    correct = 0

    for pd in pds:
        for j, gt in enumerate(gts):
            if not matched[j] and loose_match(pd, gt):
                matched[j] = True   # 占用这个 gt
                correct += 1
                break                # 该预测只匹配一个 gt，就退出

    return (correct, pd_num, gt_num)


def calculate(data):
    predictions = data.get("prediction", [])
    raw_gts = data.get("groundtruth", [])

    if raw_gts and isinstance(raw_gts[0], list):
        candidate_lists = raw_gts
    else:
        candidate_lists = [raw_gts]

    parsed_groundtruths = []
    for gts in candidate_lists:
        if len(gts) == 1 and isinstance(gts[0], str) and "," in gts[0]:
            parsed = [x.strip() for x in gts[0].split(",")]
        else:
            parsed = gts
        parsed_groundtruths.append(parsed)

    return [cal_metric(predictions, gts) for gts in parsed_groundtruths]

def evaluate(output_jsonl_file, task_name, max_data=None):   # ★ ② 新增 task_name 参数
    # 统计量初始化
    total_correct = total_pd = total_gt = 0
    sample_count = extracted_sample_count = 0

    with open(output_jsonl_file, 'r', encoding="utf-8") as f:
        for line in f:
            if max_data is not None and sample_count >= max_data:
                break
            data = json.loads(line)
            sample_count += 1
            if data.get("prediction"):
                extracted_sample_count += 1
            best = label_choose(calculate(data))
            total_correct += best[0]; total_pd += best[1]; total_gt += best[2]

    # 汇总指标
    acc = total_correct / total_gt if total_gt else 0
    prec = total_correct / total_pd if total_pd else 0
    f1   = f1_score(total_correct, total_pd, total_gt)
    return {                               # ★ ③ 改为返回字典
        "task"          : task_name,
        "correct"       : total_correct,
        "predicted"     : total_pd,
        "groundtruth"   : total_gt,
        "accuracy"      : round(acc, 4),
        "precision"     : round(prec, 4),
        "f1"            : round(f1, 4),
        "prediction%"   : round(total_pd/total_gt if total_gt else 0, 4),
        "extract_rate"  : round(extracted_sample_count/sample_count, 4),
    }

def evaluate_avg_task(sub_tasks, result_file_map, avg_task_name, max_data):
    sub_results = [evaluate(result_file_map[task], task, max_data) for task in sub_tasks]
    avg_result = sub_results[0].copy()
    for key in avg_result:
        if key != "task":
            avg_result[key] = sum(r[key] for r in sub_results) / len(sub_results)
    avg_result["task"] = avg_task_name
    return avg_result

if __name__ == '__main__':
    tasks = [
            # "vanilla",
            #  "respmask_best", "respmask_1", "respmask_2", "respmask_3", "respmask_4", 
            # "respmask_avg",
            "standard_32_1", "standard_32_2", 
            # "mirr_only_32_1e-4_42",
            #  "core_best", "core_1", "core_2", "core_3", "core_4", "core_avg",
            #  "simpleaverage", 
            #  "ema-100-0.01", "ema-10-0.1", 
            # "ema-10-0.01", 
            # "ema-10-0.01-42",
            #  "antithetic", 
            #  "semianalytic", 
            #  "strat_16_mirror", 
             "mirr_strat_6_0", "mirr_strat_6_1", "mirr_strat_6_2", 
            #  "mirr_strat_46",
             "mirror_is", 
            #  "mirr_is_42",
            #  "mirrormask", 
             "mirrormask_1", "mirrormask_2", "mirrormask_3", "mirrormask_4", 
            #  "mirrormask_avg",
            #  "mirr_only_16_5e-5_42",
            #  "mirr_is_ema_opt_10_0.02_42",
            #  "multisample2", "multisample2_1", "multisample2_2", "multisample2_3", "multisample2_4",
            #  "best",
            # "mirr_ema_42", "mirr_is_ema_42", "mirr_is_ema_strat_42",
            "multisample8",
            # "noseed_mirr_is_ema", "noseed_mirr_ema"
            ]
    max_data = None

    rows = []
    for task in tasks:
        if task == "standard_32_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_32_1.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "standard_32_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_32_2.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_only_16_5e-5_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_only_16_5e-5_42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_is_ema_opt_10_0.02_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_opt_10_0.02_42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_only_32_1e-4_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_only_32_1e-4_42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "noseed_mirr_is_ema":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_is_ema.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "noseed_mirr_ema":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_ema.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_strat_46":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_strat_46.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_is_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_ema_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_ema_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_is_ema_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_is_ema_strat_42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_strat_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "best":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_best_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "ema-10-0.01-42":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ema_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "multisample2_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed42.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "multisample2_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed43.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirror_is":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_is_mirrormask.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_strat_6_0":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirrormask.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_strat_6_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_1.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirr_strat_6_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_2.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "vanilla":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_vanilla_config4_20250630.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "mirror_mod_hitab":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_modified_mirrormask.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "strat_16_mirror":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_stratified_16_mirrormask.jsonl"
            rows.append(evaluate(result_file, task, max_data))
        elif task == "respmask_avg":
            sub_tasks = ["respmask_best", "respmask_1", "respmask_2", "respmask_3", "respmask_4"]
            result_file_map = {
                "respmask_best": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_RespMask_config4_20250708.jsonl",
                "respmask_1": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask1.jsonl",
                "respmask_2": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask2.jsonl",
                "respmask_3": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask3.jsonl",
                "respmask_4": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask4.jsonl",
            }
            rows.append(evaluate_avg_task(sub_tasks, result_file_map, "respmask_avg", max_data))

        elif task == "core_avg":
            sub_tasks = ["core_best", "core_1", "core_2", "core_3", "core_4"]
            result_file_map = {
                "core_best": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_OldCoRE_config4_20250708.jsonl",
                "core_1": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE1.jsonl",
                "core_2": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE2.jsonl",
                "core_3": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE3.jsonl",
                "core_4": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE4.jsonl",
            }
            rows.append(evaluate_avg_task(sub_tasks, result_file_map, "core_avg", max_data))
            
        elif task == "mirrormask_avg":
            sub_tasks = ["mirrormask", "mirrormask_1", "mirrormask_2", "mirrormask_3", "mirrormask_4"]
            result_file_map = {
                "mirrormask":   "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_00.jsonl",
                "mirrormask_1": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_11.jsonl",
                "mirrormask_2": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_22.jsonl",
                "mirrormask_3": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_33.jsonl",
                "mirrormask_4": "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_44.jsonl"
            }
            rows.append(evaluate_avg_task(sub_tasks, result_file_map, "mirrormask_avg", max_data))

        elif task == "simpleaverage":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_simpleaverage.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "ema-100-0.01":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "ema-10-0.1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_10_0.1.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "ema-10-0.01":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_10_0.01.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "antithetic":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_antithetic.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "semianalytic":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_semianalytic.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "mirrormask":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_00.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "mirrormask_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_11.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "mirrormask_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_22.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "mirrormask_3":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_33.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "mirrormask_4":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_44.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "multisample2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_multisample2.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "multisample8":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_multisample8.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "respmask_best":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_RespMask_config4_20250708.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "respmask_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask1.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "respmask_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask2.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "respmask_3":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask3.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "respmask_4":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldRespMask4.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "core_best":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_OldCoRE_config4_20250708.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "core_1":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE1.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "core_2":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE2.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "core_3":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE3.jsonl"
            rows.append(evaluate(result_file, task, max_data))

        elif task == "core_4":
            result_file = "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_OldCoRE4.jsonl"
            rows.append(evaluate(result_file, task, max_data))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))