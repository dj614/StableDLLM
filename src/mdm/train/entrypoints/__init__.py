"""Training engine entrypoints.

Each engine exposes a ``train_from_config(cfg)`` function.

Step 7 starts with a single engine (``llada_plus``) which wraps the existing
runner under :mod:`llada.plus`.
"""

from __future__ import annotations

__all__ = []
