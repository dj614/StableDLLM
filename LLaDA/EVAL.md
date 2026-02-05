# Eval

## Usage
Please refer to `eval_llada.sh` for the required dependencies and execution commands.

For LLaDA-Base, below is an example snapshot of conditional generation metrics evaluated with `lm-eval`:

||BBH|GSM8K|Math|HumanEval|MBPP|
|-|-|-|-|-|-|
|`lm-eval`|49.7|70.3|31.4|35.4|40.0|

In addition, we provide ablation studies on the above five metrics with respect to different generation lengths using `lm-eval`.
||BBH|GSM8K|Math|HumanEval|MBPP|
|-|-|-|-|-|-|
|gen_length=1024,steps=1024,block_length=1024|49.7|70.3|31.4|35.4|40.0|
|gen_length=512,steps=512,block_length=512|50.4|70.8|30.9|32.9|39.2|
|gen_length=256,steps=256,block_length=256|45.0|70.0|30.3|32.9|40.2|


## Notes on evaluating chat/instruct models with `lm-eval`

Evaluation results for chat/instruct models can vary depending on chat templates, few-shot formatting, and task implementations.
If your numbers differ substantially, double-check:

- `transformers` / `accelerate` versions
- whether `--apply_chat_template` and `--fewshot_as_multiturn` match the task’s expected format
- whether code-evaluation tasks require `--confirm_run_unsafe_code`

The commands below provide a concrete reference setup for sanity-checking `lm-eval` on a chat model.

Below is an example command sequence we used to test Meta-Llama-3-8B-Instruct:
```
pip install transformers==4.49.0 accelerate==0.34.2
pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .


export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks mmlu_generative,gpqa_main_generative_n_shot,gsm8k \
    --num_fewshot 5 \
    --trust_remote_code \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto:4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --num_fewshot 4 \
    --trust_remote_code \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto:4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks mmlu_pro,arc_challenge_chat \
    --trust_remote_code \
    --apply_chat_template \
    --batch_size auto:4

# For HumanEval and MBPP, using --apply_chat_template leads to significantly lower final results.
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks humaneval_instruct,mbpp \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --batch_size auto:4

```
