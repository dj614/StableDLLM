#!/usr/bin/env bash
set -euo pipefail

# Step0: minimal smoke test for the newly introduced MDM scaffolding.
# Usage (from repo root):
#   PYTHONPATH=src bash src/scripts/smoke_mdm_imports.sh

python -m mdm.debug.smoke_imports

echo "mdm smoke: OK"
