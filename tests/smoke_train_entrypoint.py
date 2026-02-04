"""Smoke test: mdm training entrypoint.

This test verifies that the framework-level training CLI can:
- locate the default base config
- merge configs
- run in --dump_config mode without importing heavy training dependencies

We intentionally do NOT start training.
"""

from __future__ import annotations

import io
import sys


def main() -> None:
    from mdm.train.main import main as train_main

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        train_main(["--dump_config"])
    finally:
        sys.stdout = old

    out = buf.getvalue()
    assert "engine" in out
    assert "train" in out

    print("mdm.train --dump_config smoke OK")


if __name__ == "__main__":
    main()
