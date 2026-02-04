#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the Step0-5 refactor scaffolding.
#
# This script sets PYTHONPATH to include both:
#   - src/     (framework + legacy llada package)
#   - repo root (LLaDA task pack)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import mdm
import mdm.registry
from mdm.tasks.spec import TaskSpec, BaseTaskSpec
from mdm.eval.io import read_jsonl, write_jsonl
from mdm.eval.harness import evaluate

assert mdm.__version__
assert hasattr(mdm.registry, "register_task")
assert TaskSpec is not None
assert BaseTaskSpec is not None
assert callable(read_jsonl) and callable(write_jsonl)
assert callable(evaluate)

print("mdm scaffolding import OK")
PY

python tests/smoke_eval_harness.py
python tests/smoke_llada_taskpack_register.py
python tests/smoke_train_entrypoint.py
python tests/smoke_train_config_overlay.py

echo "smoke OK"
