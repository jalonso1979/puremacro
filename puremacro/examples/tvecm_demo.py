"""Threshold cointegration / TVECM demo.

Simulate a cointegrated pair where the speed of error correction
differs depending on the sign of the cointegration error: large
positive deviations from equilibrium snap back fast (high regime),
while small or negative deviations correct slowly (low regime). The
TVECM should recover this asymmetry.

Run:
    python -m puremacro.examples.tvecm_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.regime.tvecm import tvecm_fit


def _simulate(T: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T); y = np.zeros(T)
    e = np.zeros(T)
    for t in range(1, T):
        x[t] = x[t - 1] + rng.standard_normal() * 0.5
        # cointegration error
        e[t] = y[t - 1] - 2.0 * x[t - 1]
        if e[t] > 0.5:
            adj = -0.9 * e[t]   # fast adjustment in high regime
        else:
            adj = -0.1 * e[t]   # slow adjustment otherwise
        y[t] = y[t - 1] + adj + 2.0 * (x[t] - x[t - 1]) + rng.standard_normal() * 0.3
    return np.column_stack([y, x])


def run_demo() -> dict:
    Y = _simulate()
    res = tvecm_fit(Y, p=2)
    return {"Y": Y, "res": res}


def main() -> None:
    out = run_demo()
    res = out["res"]
    print("Threshold VECM (Hansen-Seo, simplified) demo")
    print(f"  Cointegrating vector β = {np.round(res.beta, 3).tolist()}  "
          f"(true [1, -2])")
    print(f"  Threshold γ̂ = {res.threshold:+.3f}")
    print(f"  Regime sizes: low = {res.n_low}, high = {res.n_high}")
    print()
    print("Adjustment speeds (true: low ≈ -0.1, high ≈ -0.9 in equation y):")
    print(f"  α_low  = {np.round(res.alpha_low, 3).tolist()}")
    print(f"  α_high = {np.round(res.alpha_high, 3).tolist()}")
    print()
    print(f"  Total RSS at the optimum: {res.rss_total:.3f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        Y = out["Y"]
        # Cointegrating residual.
        ec = Y @ res.beta
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(ec, color="0.4", lw=0.7, label="ec_t = β'·y_t")
        ax.axhline(res.threshold, color="0.0", lw=0.7, ls="--",
                   label=f"threshold γ̂={res.threshold:+.3f}")
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("cointegration error")
        ax.set_title("Threshold VECM: where the regime switches")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "tvecm.png", dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'tvecm.png'}")


if __name__ == "__main__":
    main()
