"""Fit a sampling distribution p(t) from estimated weights w(t).

This is extracted from the original `LLaDA/rebuttal.py` with minimal changes:
  - small refactors for readability
  - keep exact math and plotting outputs

Currently supported family:
  - EPR
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from scipy.optimize import minimize


# --- compatibility for older NumPy (no trapezoid) ---
try:
    _np_trapz = np.trapezoid  # newer NumPy
except AttributeError:  # pragma: no cover
    _np_trapz = np.trapz


@dataclass
class FitResult:
    mode: str
    params: Dict[str, float]
    p_func: Callable[[np.ndarray], np.ndarray]
    t_values: np.ndarray
    p_on_grid: np.ndarray


def _safe_to_numpy(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x, dtype=float).reshape(-1)
    return x


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def _kl_trapz(p: np.ndarray, q: np.ndarray, t: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    log_ratio = np.log(p) - np.log(q)
    return float(_np_trapz(p * log_ratio, t))


def _fit_EPR_w_KL(t: np.ndarray, w_raw: np.ndarray, args, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.clip(np.asarray(t, float), 1e-6, 1 - 1e-6)
    w_raw = np.maximum(np.asarray(w_raw, float), 0.0)

    Zw = _np_trapz(w_raw, t)
    if not np.isfinite(Zw) or Zw <= 0:
        raise ValueError("wi has non-positive integral on t grid.")
    w_pdf = w_raw / Zw

    def unpack(z):
        a = _softplus(z[0]) + 1e-9
        b = _softplus(z[1]) + 1e-9
        A = _softplus(z[2]) + 1e-9
        r = 1.0 + _softplus(z[3])
        q = 1.0 + _softplus(z[4])
        k = _softplus(z[5])
        m = 1.0 + _softplus(z[6])
        return a, b, A, r, q, k, m

    def build_pdf(z):
        a, b, A, r, q, k, m = unpack(z)
        vhat = a * (t**r) + b * ((1 - t) ** q)
        ghat = A * np.exp(k * (t**m))
        pdf = np.sqrt(vhat + ghat**2)
        Z = _np_trapz(pdf, t)
        if not np.isfinite(Z) or Z <= 0:
            return None, None, None, None
        return pdf / Z, (a, b, A, r, q, k, m), ghat, vhat

    def loss(z):
        p_pdf, _, _, _ = build_pdf(z)
        if p_pdf is None:
            return 1e6
        return _kl_trapz(w_pdf, p_pdf, t)

    best = (None, np.inf)
    for _ in range(args.n_starts):
        z0 = np.array([-0.3, 0.4, -1.0, 0.3, 0.2, -0.2, 0.2], dtype=float) + rng.normal(scale=0.6, size=7)
        res = minimize(loss, z0, method="L-BFGS-B")
        if np.isfinite(res.fun) and res.fun < best[1]:
            best = (res.x, res.fun)

    z_star = best[0]
    p_pdf, params_tuple, g_on_grid, v_on_grid = build_pdf(z_star)
    a, b, A, r, q, k, m = params_tuple
    params = {
        "a": float(a),
        "b": float(b),
        "A": float(A),
        "r": float(r),
        "q": float(q),
        "k": float(k),
        "m": float(m),
    }

    def g_func(x):
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return A * np.exp(k * (x**m))

    def v_func(x):
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return a * (x**r) + b * ((1 - x) ** q)

    return params, p_pdf, g_on_grid, v_on_grid, g_func, v_func


def fit_p_of_t(
    *,
    wi,
    gi,
    vi,
    t_values,
    args,
    fit_mode: str,
    plot: bool = True,
) -> FitResult:
    t = _safe_to_numpy(t_values)
    w_raw = _safe_to_numpy(wi)
    if t.shape != w_raw.shape:
        raise ValueError("Input shapes must match")

    valid = np.isfinite(t) & np.isfinite(w_raw)
    t, w_raw = t[valid], w_raw[valid]
    if t.size < 5:
        raise ValueError("Too few valid t points to fit p(t).")

    t_clip = np.clip(t, 1e-6, 1 - 1e-6)

    if fit_mode == "EPR":
        params_fit, p_pdf_grid, g_grid, v_grid, _, _ = _fit_EPR_w_KL(t_clip, w_raw, args)
        family = "EPR"
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    lam = float(args.mix_uniform)
    lam = 0.0 if lam < 0 else (0.999999 if lam >= 1.0 else lam)
    p_grid = (1.0 - lam) * p_pdf_grid + lam * 1.0
    Z_grid = _np_trapz(p_grid, t_clip)
    p_grid = p_grid / (Z_grid if Z_grid > 0 else 1.0)

    def p_func0(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, float), 1e-9, 1 - 1e-9)
        return np.interp(x, t_clip, p_grid)

    if plot:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) wᵢ and fitted p(t)
        plt.figure()
        w_vis = np.maximum(_safe_to_numpy(wi), 0.0)
        Zw = _np_trapz(w_vis, t_clip)
        if Zw > 0 and np.isfinite(Zw):
            w_vis = w_vis / Zw
        plt.plot(t_clip, w_vis, marker="o", label="wᵢ")
        plt.plot(t_clip, p_grid, label=f"fitted p(t) [{family}]")
        plt.title("wᵢ and fitted p(t) vs t")
        plt.xlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"w(t)_p(t)_vs_t_{family}.png", dpi=180)
        plt.close()

        # 2) gᵢ vs t + fitted g(t)
        gi_vis = _safe_to_numpy(gi)
        plt.figure()
        plt.plot(t_clip, gi_vis, marker="o", label="gᵢ")
        plt.plot(t_clip, g_grid, label="fitted g(t)")
        plt.title("gᵢ vs t")
        plt.xlabel("t")
        plt.ylabel("gᵢ / g(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"gi_vs_t_{family}.png", dpi=180)
        plt.close()

        # 3) vᵢ vs t + fitted v(t)
        vi_vis = _safe_to_numpy(vi)
        plt.figure()
        plt.plot(t_clip, vi_vis, marker="o", label="vᵢ")
        plt.plot(t_clip, v_grid, label="fitted v(t)")
        plt.title("vᵢ vs t")
        plt.xlabel("t")
        plt.ylabel("vᵢ / v(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"vi_vs_t_{family}.png", dpi=180)
        plt.close()

    return FitResult(
        mode=family,
        params={**params_fit, "mix_uniform": lam},
        p_func=p_func0,
        t_values=t_clip,
        p_on_grid=p_grid,
    )
