import json 
import re

tasks = [
    "standard_32_1",
    "standard_32_2",
    ]

# 输入文件路径映射

task_paths = {
    "standard_32_1": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed43_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_101810/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_32_1.jsonl",
    ),
    "standard_32_2": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed44_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_101832/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_32_2.jsonl",
    ),
    "mirr_only_16_5e-5_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_noIS_RespMask_None_random_train_ratio0.9_epoch3_bs16_lr_sched_linear_lr5e-05_warmup0_max_len4096_250808_144352/checkpoint-epoch3/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_only_16_5e-5_42.jsonl",
    ),
    "mirr_is_ema_opt_10_0.02_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_EMA_bins10_blr0.02_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250808_145316/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_opt_10_0.02_42.jsonl",
    ),
    "mirr_only_32_1e-4_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_noIS_RespMask_None_random_train_ratio0.9_epoch3_bs32_lr_sched_linear_lr0.0001_warmup0_max_len4096_250808_100011/checkpoint-epoch3/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_only_32_1e-4_42.jsonl",
    ),
    "noseed_mirr_is_ema": (
        "/storage/v-mengnijia/LLaDA/eval/data/noseed_instruct_hitab_MirrorMask_IS_RespMask_EMA_Opt_bins10_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250807_071723/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_is_ema.jsonl",
    ),
    "noseed_mirr_ema": (
        "/storage/v-mengnijia/LLaDA/eval/data/noseed_instruct_hitab_MirrorMask_noIS_RespMask_EMA_Opt_bins10_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250807_071733/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_ema.jsonl",
    ),
    "mirr_strat_46": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed46_instruct_hitab_MirrorMask_noIS_RespMask_None_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_182228/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_strat_46.jsonl",
    ),
    "mirr_is_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_182237/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_42.jsonl",
    ),
    "mirr_ema_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_noIS_RespMask_EMA_bins6_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_075947/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_ema_seed42.jsonl",
    ),
    "mirr_is_ema_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_EMA_bins6_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_071537/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_seed42.jsonl",
    ),
    "mirr_is_ema_strat_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_EMA_bins6_blr0.01_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250805_135547/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_strat_seed42.jsonl",
    ),
    "best_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_with_EMA_bins10_blr0.01_IS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250803_071200/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_best_seed42.jsonl",
    ),
    "only_ema_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_EMA_bins10_blr0.01_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_160754/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ema_seed42.jsonl",
    ),
    "only_ms2_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MultiSample_ns2_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_090157/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed42.jsonl",
    ),
    "only_ms2_seed43": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed43_instruct_hitab_MultiSample_ns2_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_090516/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed43.jsonl",
    ),
    "strat_6_mir_0": (
        "eval/data/instruct_hitab_MirrorMask_noIS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_091540/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirrormask.jsonl",
    ),
    "strat_6_mir_1": (
        "eval/data/instruct_hitab_MirrorMask_noIS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_091934/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_1.jsonl",
    ),
    "strat_6_mir_2": (
        "eval/data/instruct_hitab_MirrorMask_noIS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_092123/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_2.jsonl",
    ),
    "mir_is_0": (
        "eval/data/instruct_hitab_MirrorMask_IS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_091006/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_is_mirrormask.jsonl",
    ),
    "vanilla": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_llada_instruct_vanilla_config4_20250630.jsonl",
    ),
    "strat_16_mirror_hitab": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_stratified_16_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250728_090010/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_stratified_16_mirrormask.jsonl",
    ),
    "mirror_mod_hitab": (
        "eval/data/instruct_hitab_modified_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250728_061746/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_modified_mirrormask.jsonl",
    ),
    "ema-10-0.01": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_EMA_bins10_blr0.01_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len8192_250726_213500/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_10_0.01.jsonl",
    ),
    "mirrormask_0": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250724/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_00.jsonl",
    ),
    "mirrormask_1": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_1/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_11.jsonl",
    ),
    "mirrormask_2": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_2/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_22.jsonl",
    ),
    "mirrormask_3": (
        "/storage/v-mengnijia/LLaDA/eval/data/hitab_mirrormask_3/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_33.jsonl",
    ),
    "mirrormask_4": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_44.jsonl",
    ),
    # "respmask_0": (
    #     "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
    #     "/storage/v-mengnijia/LLaDA/data.jsonl",
    #     "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_0.jsonl",
    # ),
    # "respmask_1": (
    #     "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
    #     "/storage/v-mengnijia/LLaDA/data.jsonl",
    #     "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_1.jsonl",
    # ),
    # "respmask_2": (
    #     "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
    #     "/storage/v-mengnijia/LLaDA/data.jsonl",
    #     "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_2.jsonl",
    # ),
    # "respmask_3": (
    #     "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
    #     "/storage/v-mengnijia/LLaDA/data.jsonl",
    #     "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_3.jsonl",
    # ),
    # "respmask_4": (
    #     "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725_4/predictions_config4.jsonl",
    #     "/storage/v-mengnijia/LLaDA/data.jsonl",
    #     "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_4.jsonl",
    # ),
}


def extract_boxed_answers(response_text):
    """
    Extracts substrings placed inside \boxed{...} in a *looser* manner, handling:
    - \text{...} wrappers
    - trailing % or .
    - surrounding whitespace and escape characters
    - partial latex wrapping in numeric/textual outputs
    
    Returns:
        List of cleaned extracted answers (strings).
    """
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, response_text, flags=re.DOTALL)

    cleaned_answers = []
    for m in matches:
        ans = m.strip()

        # Remove LaTeX \text{...} if present
        text_wrap_match = re.match(r"\\text\s*\{(.*?)\}", ans, flags=re.DOTALL)
        if text_wrap_match:
            ans = text_wrap_match.group(1).strip()

        # Remove LaTeX percentage and escape backslashes
        ans = ans.replace("\\%", "%").replace("\\", "").strip()

        # Remove trailing periods
        ans = ans.rstrip(".")

        cleaned_answers.append(ans)

    return cleaned_answers

# 提取 output 中的 range
def extract_output_ranges(output_list):
    """
    **extract answers from groundtruths
    """
    if isinstance(output_list, list) and output_list:
        return [[entry for entry in output_list]]
    return [[]]

for task in tasks:
    if task not in task_paths:
        print(f"⚠️  Unknown task '{task}', skipping.")
        continue

    input_file_1, input_file_2, output_file = task_paths[task]
    predictions, groundtruths = [], []

    # read predictions
    with open(input_file_1, 'r', encoding='utf-8') as inf:
        for line in inf:
            data = json.loads(line)
            preds = extract_boxed_answers(data.get("prediction", ""))
            predictions.append({"prediction": preds})

    # read groundtruths
    with open(input_file_2, 'r', encoding='utf-8') as inf:
        for line in inf:
            data = json.loads(line)
            gts = extract_output_ranges(data.get("groundtruth", []))
            groundtruths.append({"groundtruth": gts})

    # merge up to shortest length
    n = min(len(predictions), len(groundtruths))
    with open(output_file, 'w', encoding='utf-8') as outf:
        for i in range(n):
            combined = {**predictions[i], **groundtruths[i]}
            outf.write(json.dumps(combined, ensure_ascii=False) + "\n")

    print(f"✅ 任务 '{task}' 完成；结果已保存到 {output_file}")