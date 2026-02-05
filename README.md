# Diffusion+ (MDM): Masked-Diffusion Training & LLaDA Utilities

**Diffusion+ (MDM)** is a small, opinionated scaffold for **masked-language diffusion** experiments, centered around the **LLaDA** family of diffusion language models.

It provides a lightweight “framework layer” for configuration, training, and evaluation—while keeping the upstream **LLaDA** code vendored and mostly intact.

> **Links (fill as needed)**  
> - Project page: **[TBD]**  
> - Paper: **[TBD]** (arXiv: **TBD**)  
> - License: see **[Acknowledgements & license](#acknowledgements--license)**

---

## MDM Framework (at a glance)

This repo contains two main pieces:

- **`src/mdm/`**: framework + engine(s)
  - unified training entrypoint (`python -m mdm.train`) with YAML deep-merge + overrides
  - task registry + evaluation harness (`python -m mdm.eval.harness`)
  - an engine implementation for **LLaDA+** training (`mdm.engines.llada_plus`)
- **`LLaDA/`**: vendored upstream LLaDA repo (tasks, metrics, original sampler, configs, etc.)  
  See `LLaDA/README.md` for upstream-specific notes.

**Why this exists:** training/evaluation scripts for diffusion LMs often hard-code dataset fields and ad-hoc conventions. This repo separates:

- **framework plumbing** (config/registry/eval harness) in `src/mdm/`
- **task packs** (datasets/metrics/adapters) in `LLaDA/llada/`

---

## 🔥 News

- **[YYYY.MM.DD]** TBD — initial public release.
- **[YYYY.MM.DD]** TBD — paper/code update.

(Replace with your real milestones.)

---

## Repository layout

