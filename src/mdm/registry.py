"""Global registries for MDM framework components.

This module intentionally keeps a very small surface area:

* Register / retrieve TaskSpec implementations by name.

Later steps may extend this registry to models, samplers, schedulers, etc.
"""

from __future__ import annotations

from typing import Dict, List

from mdm.tasks.spec import TaskSpec


_TASK_REGISTRY: Dict[str, TaskSpec] = {}


def register_task(name: str, spec: TaskSpec, *, overwrite: bool = False) -> None:
    """Register a task spec under a stable string name.

    Args:
        name: Task identifier, e.g. "llada_gsm8k".
        spec: Object implementing the TaskSpec protocol.
        overwrite: If True, allow overwriting an existing registration.

    Raises:
        ValueError: If the name is already registered and overwrite is False.
    """

    if (not overwrite) and (name in _TASK_REGISTRY):
        raise ValueError(
            f"Task '{name}' is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _TASK_REGISTRY[name] = spec


def get_task(name: str) -> TaskSpec:
    """Retrieve a registered task spec."""

    try:
        return _TASK_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(_TASK_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from e


def list_tasks() -> List[str]:
    """Return all registered task names (sorted)."""

    return sorted(_TASK_REGISTRY.keys())
