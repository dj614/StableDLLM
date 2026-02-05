"""Bridge modules that help migrate existing task code into the mdm framework.

Bridges are *optional*. They typically register legacy tasks into
:mod:`mdm.registry` without forcing the legacy code to change immediately.

Examples:
  * Deprecated bridges can live here to ease migration from legacy task packs.
"""

from __future__ import annotations
