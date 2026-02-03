"""LLaDA utilities: clean CLI + evaluation for inference-only workflows.

Compatibility shim: the implementation has been moved under `LLaDA/src/llada/`.
This file keeps import paths stable, e.g.:

  python -m LLaDA.llada.cli.main infer ...

"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Make `LLaDA.llada` a pkgutil-style namespace package.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_impl = Path(__file__).resolve().parent.parent / "src" / "llada"
if _impl.is_dir():
    # Allow submodules (cli/eval/model/...) to live under LLaDA/src/llada.
    __path__.append(str(_impl))  # type: ignore[attr-defined]

__all__ = ["cli", "eval", "model", "tasks", "utils"]
