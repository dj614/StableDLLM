"""Training engine entrypoints.

Each engine exposes a ``train_from_config(cfg)`` function.

Step 7 starts with a single engine (``llada_plus``) which wraps the runner
under :mod:`mdm.engines.llada_plus`.
"""

from __future__ import annotations

__all__ = []
