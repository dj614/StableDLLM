"""Importance sampling helpers for LLaDA+ training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .estimate_w import evaluate_over_x0


@dataclass
class FitResult:
    t_values: np.ndarray
    p_on_grid: np.ndarray


def fit_p_of_t(
    *,
    wi: np.ndarray,
    gi: np.ndarray,
    vi: np.ndarray,
    t_values: np.ndarray,
    args: Any,
    fit_mode: str = "EPR",
    plot: bool = False,
) -> FitResult:
    """Fit a sampling distribution p(t) over a fixed grid.

    This is a lightweight fallback that treats w(t) as the unnormalized density
    (with optional uniform mixing). More advanced fitting can be added later.
    """
    t = np.asarray(t_values, dtype=float)
    w = np.asarray(wi, dtype=float)
    w = np.clip(w, 1e-12, None)

    mix = float(getattr(args, "mix_uniform", 0.0) or 0.0)
    if mix > 0.0:
        w = (1.0 - mix) * w + mix * np.ones_like(w)

    if plot:
        try:  # pragma: no cover - optional debug visualization
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(t, w, label=f"p(t) [{fit_mode}]")
            plt.xlabel("t")
            plt.ylabel("unnormalized density")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return FitResult(t_values=t, p_on_grid=w)


__all__ = ["evaluate_over_x0", "fit_p_of_t", "FitResult"]
