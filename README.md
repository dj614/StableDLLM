<h1 align="center">Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models </h1>

<p align="center">
  <a><img 
     src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge&logo=data:image/png;base64,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" 
     alt="Github"></a> 
  <a href='http://arxiv.org/abs/2511.18159'><img src='https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge&logo=arxiv' alt='arXiv PDF'></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <i><b> <img src="https://img.alicdn.com/imgextra/i2/O1CN01FPcQDy1WTPjPX6IH9_!!6000000002789-2-tps-96-96.png" width="16px"  style="vertical-align: middle;"> Qwen Large Model Application Team, Alibaba</b></i>
</p>

**StableDLLM** is the reference codebase for our work on stabilizing masked diffusion model (MDM) post-training. Building on a variance decomposition of MDM training into **masking pattern noise (A)**, **masking rate noise (B)**, and **data noise (C)**, this repo implements practical variance-reduction recipes, most notably **P-POTS** (a Pareto-optimal unbiased masking rate sampler) and **MIRROR** (complementary, negatively correlated masking), alongside the standard training procedure that samples masking rates uniformly. Together, these methods make diffusion language model training (including **LLaDA** and **MMaDA**) substantially more effective and stable. It provides a lightweight “framework layer” for configuration, training, and evaluation—while keeping the upstream [**LLaDA**](https://github.com/ML-GSAI/LLaDA) and [**MMaDA**](https://github.com/Gen-Verse/MMaDA) code vendored and mostly intact.


<p align="center">
  <a href="assets/paper.pdf">
    <img src="assets/paper_p5.png" width="800" />
  </a>
</p>

## 🔥 News

- **[2026.02.09]** We released the codebase.
- **[2026.01.26]** Our paper was accepted to ICLR 2026.
- **[2025.11.22]** We posted our paper on [arXiv:2511.18159](https://arxiv.org/abs/2511.18159)

---

## 💡 Main Results

Compared to standard MDM training, our methods boost accuracy by **7–8%** on complex reasoning tasks while reducing run-to-run variability to **near ARM levels**. This substantially narrows the gap to strong ARM baselines; in most settings, even the best baseline runs still fall below the worst run of our method.

<p align="center">
  <a href="assets/table1.png">
    <img src="assets/table1.png" width="800" />
  </a>
</p>

<p align="center">
  <a href="assets/table2.pdf">
    <img src="assets/table2.png" width="800" />
  </a>
</p>

---

## 🛠️ Installation

### Requirements

- Python **3.10+**
- PyTorch, Transformers, Datasets, Accelerate  
- DeepSpeed (optional), WandB (optional)
- `PyYAML` (used by the MDM config loader)

### Setup (from repo root)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r LLaDA/requirements.txt
pip install pyyaml
```

## 🎯 Quickstart

Below, we use LLaDA to illustrate how to use our repository. MMaDA can be used in the same way.

### 1) Data preprocessing / preparation (example: [**GSM8K**](https://huggingface.co/datasets/openai/gsm8k))

Training uses a processed JSONL format (see **Data format** below). The repo provides preprocessors under `src/tools/preprocess/train/`.

Example for GSM8K:

```bash
PYTHONPATH=src:. python src/tools/preprocess/train/preprocess_gsm8k_sft.py \
  --out_file ./data/train/gsm8k.jsonl \
  --tokenizer_path <YOUR_TOKENIZER>
```

#### Data format

**Training (processed JSONL)**

The training engine (`mdm.engines.llada_plus`) reads a JSONL where each line is a dict:

```json
{"input_ids": [ ... token ids ... ], "prompt_length": 123}
```

- `input_ids`: tokenized full sequence (prompt + answer)
- `prompt_length`: prefix length (tokens) that belong to the prompt. Loss is applied only to tokens after this boundary.

The training dataloader will:
- pad sequences in-batch,
- set `labels = input_ids` but mask out:
  - padding,
  - prompt tokens (`labels[:prompt_length] = -100`).

This format is produced by preprocess scripts under `src/tools/preprocess/train/`.

---

### 2) Register a task (adding a new task)

Tasks are registered via `mdm.registry` with a `TaskSpec`.

- Interface: `src/mdm/tasks/spec.py`
- Registry: `src/mdm/registry.py`

Minimal pattern:

```python
from mdm.registry import register_task
from mdm.tasks.spec import BaseTaskSpec

class MyTask(BaseTaskSpec):
    name = "my_task"

    def build_dataset(self, split, cfg): ...
    def collate_fn(self, batch): ...
    def postprocess(self, pred, cfg): ...
    def metrics(self, pred_path, gt_path, cfg): ...

register_task("my_task", MyTask())
```

Then evaluate with `mdm.eval.harness --task my_task` (see **Evaluation** below).

---

### 3) Configuration

The unified entrypoint (`python -m mdm.train`) merges YAML configs and then dispatches to a training engine.

#### Engine: `llada_plus`

Config keys live under `train.*` (see `src/mdm/train/main.py` and `src/mdm/engines/llada_plus/cli/train.py`).

Common knobs:

- **Data**
  - `train.train_data_path`: processed JSONL path (defaults to `./data/train/{task}.jsonl`)
  - `train.max_len`: max token length
  - `train.epochs`, `train.train_ratio`
- **Optimization**
  - `train.lr`, `train.lr_scheduler_type`, `train.warmup_steps`
  - `train.batch_size_per_gpu`, `train.grad_accum`
- **Eval / checkpoints**
  - `train.eval_strategy`: `epoch` or `steps`
  - `train.save_strategy`: `last`, `epoch`, or `steps`
  - `train.output_dir`: log/ckpt directory (default: `./logs/{task}`)
- **Diffusion / sampling**
  - `train.train_mode`: `Normal` or `MIRROR`
  - `train.PPOTS`: enable IS-on-t training logic
  - `train.p_model`: importance sampling model (`EPR`, `AP`, ...)
  - plus various IS-related sampling counts and caps

To inspect the final merged config without training:

```bash
PYTHONPATH=src:. python -m mdm.train \
  --config src/configs/mdm/base/train_llada_plus.yaml \
  --config LLaDA/configs/llada_gsm8k.yaml \
  --dump_config
```

---

### 4) Train (including multi-GPU / multi-node)

#### Train with the unified MDM entrypoint

The training CLI supports multiple YAML files (base + overlays). A typical run uses:

- base: `src/configs/mdm/base/train_llada_plus.yaml`
- overlay: `LLaDA/configs/llada_gsm8k.yaml` (task-specific settings)

```bash
PYTHONPATH=src:. python -m mdm.train \
  --config src/configs/mdm/base/train_llada_plus.yaml \
  --config LLaDA/configs/llada_gsm8k.yaml \
  --auto_import LLaDA.llada.register \
  --set train.output_dir=./outputs/llada_plus_gsm8k
```

#### Multi-GPU training (Accelerate / DeepSpeed)

The LLaDA+ runner is built on Accelerate. For multi-GPU training, use `accelerate launch`.

Example Accelerate config:

- `src/configs/accelerate/deepspeed_zero2.yaml`

Note: the `deepspeed_config_file` path in that file may need to point to:

- `src/configs/deepspeed/zero2_cpu_offload.json`

Example launch:

```bash
accelerate launch --config_file src/configs/accelerate/deepspeed_zero2.yaml \
  -m mdm.train \
  --config src/configs/mdm/base/train_llada_plus.yaml \
  --config LLaDA/configs/llada_gsm8k.yaml \
  --auto_import LLaDA.llada.register \
  --set train.output_dir=./outputs/llada_plus_gsm8k_ds
```

---

### 5) Inference (legacy `llada` CLI)

A small convenience CLI lives at `src/llada/cli/main.py`:

```bash
PYTHONPATH=src:. python -m llada.cli.main infer \
  --task gsm8k \
  --out_file ./outputs/preds_gsm8k.jsonl \
  --model_name GSAI-ML/LLaDA-8B-Instruct \
  --steps 128 --gen_length 128
```

#### Inference / scoring JSONL

The legacy `llada infer` command writes one dict per sample, e.g.:

```json
{
  "task": "gsm8k",
  "prompt": "...",
  "gold_raw": "...",
  "prediction": "...",
  "meta": {"index": 0}
}
```

The scoring helpers (`llada score`) use robust answer extraction heuristics in `LLaDA/llada/eval/`.

---

### 6) Evaluation

#### Score predictions (legacy `llada` CLI)

```bash
PYTHONPATH=src:. python -m llada.cli.main score \
  --task gsm8k \
  --pred_jsonl ./outputs/preds_gsm8k.jsonl
```

#### Evaluation harness (`mdm.eval`)

For framework-level evaluation (task-pack driven), use:

```bash
PYTHONPATH=src:. python -m mdm.eval.harness \
  --task llada_gsm8k \
  --pred ./outputs/preds_gsm8k.jsonl \
  --auto_import LLaDA.llada.register
```

If your predictions do not already contain gold answers, you can provide a ground-truth JSONL:

```bash
PYTHONPATH=src:. python -m mdm.eval.harness \
  --task llada_gsm8k \
  --pred ./outputs/preds.jsonl \
  --gt ./data/test/gsm8k_gt.jsonl \
  --auto_import LLaDA.llada.register
```

The task pack (`LLaDA/llada/tasks/specs.py`) will merge pred and gt by id (preferred) or index.

## 🧪 Development & tests

Run the lightweight smoke tests:

```bash
bash scripts/smoke.sh
```

Or directly:

```bash
PYTHONPATH=src:. python -m mdm.debug.smoke_imports
PYTHONPATH=src:. python -m mdm.engines.llada_plus.debug.smoke_test
```

Run unit tests:

```bash
pytest -q
```

## 🙏🏻 Acknowledgements & license

The LLaDA/ and MMaDA/ subtrees are vendored from the upstream [**LLaDA**](https://github.com/ML-GSAI/LLaDA) and [**MMaDA**](https://github.com/Gen-Verse/MMaDA) repository. We thank the original authors for their open-source contributions.

## ⭐️ Citation

If you find this work useful, please kindly cite:

```bash
@misc{jia2025bringstabilitytodiffusion,
  title        = {Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models},
  author       = {Mengni Jia, Mengyu Zhou, Yihao Liu, Xiaoxi Jiang, Guanjun Jiang},
  year         = {2025},
  note         = {arXiv preprint arXiv:2511.18159}
}
```