import json 
import re

tasks = ["standard_32_0", "mirr_newis_43"]

# 输入文件路径映射
task_paths = {
    "mirr_newis_43": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed43_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_121459/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_newis_43.jsonl",
    ),
    "standard_32_0": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_062335/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_32_0.jsonl",
    ),
    "standard_16_0": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_Normal_noIS_RespMask_None_random_train_ratio0.9_epoch5_bs16_lr_sched_linear_lr5e-05_warmup0_max_len4096_250809_062103/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_standard_16_0.jsonl",
    ),
    "ema_opt_6_0.01_42_only": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_Normal_noIS_RespMask_EMA_bins6_blr0.01_random_train_ratio0.9_lossmax10_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250808_080513/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_opt_6_0.01_42_only.jsonl",
    ),
    "ema_opt_10_0.01_42_only": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_Normal_noIS_RespMask_EMA_Opt_bins10_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250808_041027/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_opt_10_0.01_42_only.jsonl",
    ),
    "noseed_mirr_is": (
        "/storage/v-mengnijia/LLaDA/eval/data/noseed_instruct_hitab_MirrorMask_IS_RespMask_None_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250807_071252/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_is.jsonl",
    ),
    "noseed_mirr_strat": (
        "/storage/v-mengnijia/LLaDA/eval/data/noseed_instruct_hitab_MirrorMask_noIS_RespMask_None_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250807_071318/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_noseed_mirr_strat.jsonl",
    ),
    "mirr_is_ema_warmup_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_EMA_bins6_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup35_max_len4096_250806_183755/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_is_ema_warmup_42.jsonl",
    ),
    "mirr_ema_warmup_42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_noIS_RespMask_EMA_bins6_blr0.01_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup35_max_len4096_250806_182700/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirr_ema_warmup_42.jsonl",
    ),
    "vanilla_1024_256_16": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_256_16.jsonl",
    ),
    "vanilla_1024_1024_64": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps1024_block64.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_1024_64.jsonl",
    ),
    "vanilla_1024_1024_128": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps1024_block128.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_1024_128.jsonl",
    ),
    "vanilla_1024_1024_256": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps1024_block256.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_1024_256.jsonl",
    ),
    "vanilla_1024_1024_512": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps1024_block512.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_1024_512.jsonl",
    ),
    "vanilla_1024_512_64": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps512_block64.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_512_64.jsonl",
    ),
    "vanilla_1024_256_32": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps256_block32.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_256_32.jsonl",
    ),
    "vanilla_1024_256_64": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps256_block64.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_256_64.jsonl",
    ),
    "vanilla_1024_128": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps512_block128.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024_512_128.jsonl",
    ),
    "vanilla_1024_16": (
        "/storage/v-mengnijia/LLaDA/eval/data/predictions_hitab_temp0.0_gen1024_steps512_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_vanilla_1024.jsonl",
    ),
    "mirror_ema_strat_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_noIS_RespMask_EMA_bins6_blr0.01_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250806_075630/checkpoint-epoch5/predictions_hitab_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirror_ema_strat_seed42.jsonl",
    ),
    "mirror_is_ema_strat_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed42_instruct_hitab_MirrorMask_IS_RespMask_EMA_bins6_blr0.01_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250805_135547/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirror_is_ema_strat_seed42.jsonl",
    ),
    "mirror_is_strat_seed42": (
        "/storage/v-mengnijia/LLaDA/eval/data/LLaDA/seed42_instruct_hitab_MirrorMask_IS_RespMask_None_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250805_115433/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirror_is_strat_seed42.jsonl",
    ),
    "only_ms2_seed44": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed44_instruct_hitab_MultiSample_ns2_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_152039/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed44.jsonl",
    ),
    "only_ms2_seed45": (
        "/storage/v-mengnijia/LLaDA/eval/data/seed45_instruct_hitab_MultiSample_ns2_noIS_RespMask_random_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250730_152047/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_only_ms2_seed45.jsonl",
    ),
    "strat_6_mir_1": (
        "eval/data/instruct_hitab_MirrorMask_noIS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_091934/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_1.jsonl",
    ),
    "strat_6_mir_2": (
        "eval/data/instruct_hitab_MirrorMask_noIS_RespMask_stratified_6_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250729_092123/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_strat_6_mirror_2.jsonl",
    ),
    "RB": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_RB_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250728_053606/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_rb.jsonl",
    ),
    "JK": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_JackKnife_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len8192_250728_054621/checkpoint-epoch5/predictions_temp0.0_gen512_steps256_block16.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_jk.jsonl",
    ),
    "mirrormask_1": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250726_052614/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_1.jsonl",
    ),
    "mirrormask_2": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_MirrorMask_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len4096_250725/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_mirrormask_2.jsonl",
    ),
    "ema_10_0_001": (
        "/storage/v-mengnijia/LLaDA/eval/data/instruct_hitab_EMA_bins10_blr0.001_RespMask_train_ratio0.9_epoch5_bs32_lr_sched_linear_lr5e-05_warmup0_max_len8192_250727_060331/checkpoint-epoch5/predictions_config4.jsonl",
        "/storage/v-mengnijia/LLaDA/data/test/hitab_test_llada.jsonl",
        "/storage/v-mengnijia/LLaDA/eval/data/combined_predictions_groundtruth_hitab_ema_10_0_001.jsonl",
    ),
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