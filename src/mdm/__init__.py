"""Masked Diffusion Model (MDM) framework package.

This package is intentionally task-agnostic.

Task- or dataset-specific logic should live outside of this package (e.g., under
task packs such as ``LLaDA/``) and connect to the framework via the registry +
TaskSpec interfaces.

Step 1 of the refactor introduces the ``mdm`` namespace as a stable home for
shareable components.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]


# Semantic version for the framework namespace (independent of the repo version).
__version__ = "0.1.0"
