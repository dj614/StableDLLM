"""llada.plus: training utilities extracted from the original monolithic script.

This package is intentionally lightweight and import-safe:
- Core algorithmic components live under subpackages (data/diffusion/losses/...).
- Training orchestration lives under llada.plus.train.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
