"""LLaDA task pack.

This package is intentionally lightweight at Step 5 of the refactor:

* The *framework* lives under :mod:`src.mdm`.
* This package provides a place for LLaDA-specific task specs, dataset adapters,
  and metric logic.

At this stage, we provide a registration entry point that registers the
task-pack TaskSpec implementations directly.

Import :mod:`LLaDA.llada.register` (or call :func:`LLaDA.llada.register.register_all`)
so the tasks become available via :func:`mdm.registry.get_task`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.0"
