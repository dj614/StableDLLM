"""Task registration for the LLaDA task pack.

After Step 6, LLaDA-specific evaluation adapters live in this package, so
importing :mod:`LLaDA.llada.register` registers *non-legacy* TaskSpec objects.

Example:

  PYTHONPATH=src:. python -m mdm.eval.harness \
      --task llada_gsm8k --pred out/predictions.jsonl \
      --gt out/ground_truth.jsonl \
      --auto_import LLaDA.llada.register

"""

from __future__ import annotations

from typing import Optional


def register_all(prefix: str = "llada_", overwrite: bool = True) -> Optional[int]:
    """Register all LLaDA tasks into :mod:`mdm.registry`.

    Returns the number of tasks registered if available, otherwise ``None``.
    """

    try:
        from .tasks.specs import register_llada_tasks

        return register_llada_tasks(prefix=prefix, overwrite=overwrite)
    except Exception:
        # If the framework isn't on the path, keep import safe.
        return None


# Convenience: importing this module registers tasks.
register_all()
