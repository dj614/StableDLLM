"""Configuration utilities for the MDM framework.

Step 7 introduces a framework-level training entrypoint that needs a lightweight
configuration mechanism.

Design goals:
- Avoid a heavyweight config system (Hydra/OmegaConf) at this stage.
- Support multiple YAML files merged in order (base + overlays).
- Support simple dotted-key overrides.

Public API:
- :func:`mdm.configs.merge_config_files`
"""

from __future__ import annotations

from .loader import merge_config_files

__all__ = ["merge_config_files"]
