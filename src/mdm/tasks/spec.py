"""TaskSpec: a small interface boundary between the framework and task packs.

The framework should never hard-code dataset fields, prompting rules, tokenizer
logic or task-specific metrics. Instead, those are provided by a TaskSpec.

This is intentionally minimal so tasks can be integrated incrementally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class TaskSpec(Protocol):
    """Protocol for task packs.

    A concrete task spec typically lives in a task pack directory such as
    ``LLaDA/`` and registers itself via ``mdm.registry.register_task``.
    """

    # A stable identifier for the task spec (optional but recommended).
    name: str

    def build_dataset(self, split: str, cfg: Mapping[str, Any]) -> Any:
        """Build a dataset or datamodule for the given split."""

    def collate_fn(self, batch: list[Any]) -> Any:
        """Convert a list of dataset items into a model-ready batch."""

    def postprocess(self, pred: Any, cfg: Mapping[str, Any]) -> Any:
        """Task-specific post-processing of model outputs."""

    def metrics(self, pred_path: str, gt_path: str, cfg: Mapping[str, Any]) -> Mapping[str, float]:
        """Compute metrics for saved predictions vs ground truth."""


class BaseTaskSpec(ABC):
    """Optional ABC base-class for TaskSpec implementations."""

    name: str

    @abstractmethod
    def build_dataset(self, split: str, cfg: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, batch: list[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, pred: Any, cfg: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def metrics(self, pred_path: str, gt_path: str, cfg: Mapping[str, Any]) -> Mapping[str, float]:
        raise NotImplementedError
