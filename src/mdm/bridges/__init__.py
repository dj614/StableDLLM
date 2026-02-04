"""Bridge modules that help migrate existing task code into the mdm framework.

Bridges are *optional*. They typically register legacy tasks into
:mod:`mdm.registry` without forcing the legacy code to change immediately.

Examples:
  * :mod:`mdm.bridges.llada_legacy` registers LLaDA tasks that already contain
    gold answers inside the predictions JSONL.
"""

from __future__ import annotations
