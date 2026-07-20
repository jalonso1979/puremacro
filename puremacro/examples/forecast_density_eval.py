"""Density-forecast evaluation: PIT, CRPS, KLIC for two competing models.

We fit two GARCH(1,1) specifications on a synthetic return series — a
correctly-specified GARCH(1,1) and a deliberately-misspecified
constant-volatility model — then compare them via:

  - PIT histogram (should be uniform under correct calibration)
  - mean CRPS (lower is better)
  - Amisano-Giacomini KLIC weighted likelihood-ratio test

Run:
    python -m puremacro.examples.forecast_density_eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..garch.fit import garch11_fit
from ..forecast.density import (
    pit, pit_uniformity_test,
    crps_gaussian,
    klic_amisano_giacomini,
)


def _simulate_garch(T: int = 1500, omega: float = 0.05,
                    a: float = 0.10, b: float = 0.85,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(T)
    eps = np.empty(T)
    sigma2[0] = omega / max(1.0 - a - b, 1e-3)
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, T):
        sigma2[t] = omega + a * eps[t - 1] ** 2 + b * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return eps


def run_eval() -> dict:
    eps = _simulate_garch()
    n_train = 1000
    train, test = eps[:n_train], eps[n_train:]

    # Model 1: GARCH(1,1) (correct).
    g1 = garch11_fit(pd.Series(train))
    sigma1_test = np.empty(len(test))
    sigma2_t = float(g1.sigma.iloc[-1]) ** 2
    eps_prev = train[-1]
    for t in range(len(test)):
        sigma2_t = g1.omega + g1.alpha * eps_prev ** 2 + g1.beta * sigma2_t
        sigma1_test[t] = np.sqrt(max(sigma2_t, 1e-12))
        eps_prev = test[t]

    # Model 2: constant volatility (misspecified).
    sigma2_const = float(np.var(train))
    sigma2_test = np.full(len(test), np.sqrt(sigma2_const))

    # PIT under each model.
    pit1 = pit(test, lambda y, t: norm.cdf(y / sigma1_test[t]))
    pit2 = pit(test, lambda y, t: norm.cdf(y / sigma2_test[t]))
    ks1 = pit_uniformity_test(pit1)
    ks2 = pit_uniformity_test(pit2)

    crps1 = crps_gaussian(test, np.zeros(len(test)), sigma1_test).mean()
    crps2 = crps_gaussian(test, np.zeros(len(test)), sigma2_test).mean()

    # Log-likelihood under each model.
    ll1 = norm.logpdf(test, scale=sigma1_test)
    ll2 = norm.logpdf(test, scale=sigma2_test)
    klic = klic_amisano_giacomini(ll1, ll2, h=1)

    return {
        "test": test,
        "sigma1": sigma1_test,
        "sigma2": sigma2_test,
        "pit1": pit1, "pit2": pit2,
        "ks1": ks1, "ks2": ks2,
        "crps1": float(crps1), "crps2": float(crps2),
        "klic": klic,
    }


def main() -> None:
    out = run_eval()
    print("Density-forecast evaluation: GARCH(1,1) vs constant-vol")
    print(f"  Test sample size: {len(out['test'])}")
    print()
    print(f"  PIT KS (GARCH):      stat={out['ks1']['stat']:.3f}, "
          f"p={out['ks1']['p_value']:.3f}   (high p ⇒ calibrated)")
    print(f"  PIT KS (const-vol):  stat={out['ks2']['stat']:.3f}, "
          f"p={out['ks2']['p_value']:.3f}")
    print()
    print(f"  Mean CRPS (GARCH):     {out['crps1']:.4f}")
    print(f"  Mean CRPS (const-vol): {out['crps2']:.4f}   "
          f"(lower is better)")
    print()
    print(f"  Amisano-Giacomini KLIC (GARCH vs const-vol): "
          f"stat={out['klic']['stat']:.2f}, p={out['klic']['p_value']:.4f}")
    print(f"  → positive stat with small p ⇒ GARCH significantly better")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
        for ax, pits, label in zip(axes, [out["pit1"], out["pit2"]],
                                    ["GARCH", "constant-vol"]):
            ax.hist(pits, bins=15, color="0.4", edgecolor="0.0")
            ax.axhline(len(pits) / 15, color="0.0", lw=0.6, ls="--")
            ax.set_title(f"PIT histogram — {label}")
            ax.set_xlabel("PIT value")
        axes[0].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "forecast_density_eval.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'forecast_density_eval.png'}")


if __name__ == "__main__":
    main()
