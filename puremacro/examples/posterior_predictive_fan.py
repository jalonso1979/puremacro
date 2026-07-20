"""Posterior-predictive fan chart from BVAR Gibbs draws.

Pipeline:
  1. ``minnesota_gibbs`` → posterior draws of (A, Σ, intercept).
  2. ``simulate_var_paths`` → one path per draw.
  3. ``fan_chart_quantiles`` + ``plot_fan_from_paths`` → render the fan.

This is the cleanest end-to-end demonstration of how the new Tier 4
``posterior`` module ties together with the Tier 2 BVAR Gibbs.

Run:
    python -m puremacro.examples.posterior_predictive_fan
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.bvar import minnesota_gibbs
from ..posterior import simulate_var_paths, fan_chart_quantiles


def _simulate(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.7, 0.10, 0.0],
                   [0.05, 0.6, 0.10],
                   [0.0, 0.05, 0.55]])
    Sigma = np.array([[1.0, 0.3, 0.1],
                       [0.3, 1.0, 0.2],
                       [0.1, 0.2, 1.0]]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return Y


def run_demo(H: int = 12, n_draws: int = 600) -> dict:
    Y = _simulate()
    g = minnesota_gibbs(Y, p=1, n_draws=n_draws, burn=300,
                        rng=np.random.default_rng(0))
    paths = simulate_var_paths(
        g["A_draws"], g["Sigma_draws"], g["intercept_draws"],
        y_init=Y[-1:], horizon=H,
        rng=np.random.default_rng(1),
    )
    quants = fan_chart_quantiles(paths, levels=(0.10, 0.30, 0.50,
                                                  0.70, 0.90))
    return {"Y": Y, "paths": paths, "quants": quants, "H": H}


def main() -> None:
    out = run_demo()
    paths = out["paths"]; q = out["quants"]; H = out["H"]
    print("Posterior-predictive fan chart from Minnesota BVAR Gibbs")
    print(f"  T training = {len(out['Y'])}, posterior draws = {paths.shape[0]}")
    print(f"  Forecast horizons = 0..{H}, variables = {paths.shape[2]}")
    print()
    print("Variable 0, predictive median + 80% band:")
    for h in range(H + 1):
        med = q[2, h, 0]
        lo, hi = q[0, h, 0], q[4, h, 0]
        print(f"  h={h:>2d}:  {med:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        from ..plot import plot_fan_from_paths
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
        for i, ax in enumerate(axes):
            plot_fan_from_paths(paths, var_index=i,
                                 levels=(0.50, 0.80, 0.95),
                                 title=f"y_{i}", ax=ax)
        fig.suptitle("BVAR posterior-predictive fan charts (median + 50/80/95%)")
        fig.tight_layout()
        fig.savefig(out_dir / "posterior_predictive_fan.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'posterior_predictive_fan.png'}")


if __name__ == "__main__":
    main()
