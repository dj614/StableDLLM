"""Loss functions."""

from .masked_ce import batched_loss_for_backpropagate

__all__ = ["batched_loss_for_backpropagate"]
