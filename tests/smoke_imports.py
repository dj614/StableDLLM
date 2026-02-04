"""Minimal smoke test (no external deps).

Run with:
  PYTHONPATH=src python tests/smoke_imports.py
"""

from __future__ import annotations


def main() -> None:
    import mdm
    import mdm.registry
    from mdm.eval.io import read_jsonl, write_jsonl
    from mdm.tasks.spec import BaseTaskSpec, TaskSpec

    assert mdm.__version__
    assert callable(mdm.registry.register_task)
    assert callable(read_jsonl)
    assert callable(write_jsonl)
    assert TaskSpec is not None
    assert BaseTaskSpec is not None

    print("OK")


if __name__ == "__main__":
    main()
