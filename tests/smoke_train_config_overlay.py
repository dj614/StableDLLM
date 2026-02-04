"""Smoke test: base+overlay config layering.

Step 8 introduces a layered config layout:
- Base configs under src/configs
- Task overlays under LLaDA/configs

This test verifies that mdm.train can merge base + overlay and dump the result
without importing heavy training dependencies.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    from mdm.train.main import main as train_main

    root = _repo_root()
    base = root / "src" / "configs" / "mdm" / "base" / "train_llada_plus.yaml"
    overlay = root / "LLaDA" / "configs" / "llada_gsm8k.yaml"

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        train_main(["--dump_config", "--config", str(base), "--config", str(overlay)])
    finally:
        sys.stdout = old

    out = buf.getvalue()
    assert "engine" in out
    assert "train" in out
    # Overlay should set the task.
    assert "task: gsm8k" in out

    print("mdm.train base+overlay --dump_config smoke OK")


if __name__ == "__main__":
    main()
