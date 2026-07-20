"""Self-exciting threshold VAR: regime-dependent dynamics.

Simulate a 2-variable VAR whose persistence and volatility differ
above vs. below a threshold on the first variable's lag. Fit
``tvar_fit`` and check that the recovered threshold and per-regime
coefficients match the truth.

Run:
    python -m puremacro.examples.tvar_threshold_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.regime import tvar_fit


def _simulate(T: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, 2))
    A_low = np.array([[0.3, 0.0], [0.05, 0.4]])
    A_high = np.array([[0.7, 0.10], [0.05, 0.6]])
    mu_low = np.array([0.4, 0.0])
    mu_high = np.array([-0.3, 0.0])
    sd_low, sd_high = 0.4, 0.9
    threshold_true = 0.0
    for t in range(1, T):
        if Y[t - 1, 0] <= threshold_true:
            Y[t] = mu_low + A_low @ Y[t - 1] + sd_low * rng.standard_normal(2)
        else:
            Y[t] = mu_high + A_high @ Y[t - 1] + sd_high * rng.standard_normal(2)
    return Y, threshold_true


def run_demo() -> dict:
    Y, c_true = _simulate()
    fit = tvar_fit(Y, threshold_var_idx=0, p=1)
    return {"Y": Y, "fit": fit, "threshold_true": c_true}


def main() -> None:
    out = run_demo()
    fit = out["fit"]
    print("Self-exciting threshold VAR — Tsay (1998)")
    print(f"  T = {len(out['Y'])} obs")
    print(f"  Estimated threshold: {fit.threshold:+.3f}  "
          f"(true {out['threshold_true']:+.3f})")
    print(f"  Estimated delay:     d = {fit.delay}  (true 1)")
    print(f"  Regime sizes: low = {fit.n_low}, high = {fit.n_high}")
    print()
    print("  Low regime intercept:", np.round(fit.intercept_low, 3))
    print("  Low regime A:")
    print(np.round(fit.A_low.reshape(2, 2), 3))
    print("  High regime intercept:", np.round(fit.intercept_high, 3))
    print("  High regime A:")
    print(np.round(fit.A_high.reshape(2, 2), 3))
    print()
    print("  Low-regime σ (diag of Sigma):",
          np.round(np.sqrt(np.diag(fit.Sigma_low)), 3))
    print("  High-regime σ (diag of Sigma):",
          np.round(np.sqrt(np.diag(fit.Sigma_high)), 3))

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        Y = out["Y"]
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        below = Y[1:, 0] <= fit.threshold
        idx = np.arange(1, len(Y))
        ax.plot(idx, Y[1:, 0], color="0.6", lw=0.5)
        ax.scatter(idx[below], Y[1:, 0][below], s=4, color="0.0",
                   label=f"below threshold ({fit.n_low})")
        ax.scatter(idx[~below], Y[1:, 0][~below], s=4, color="0.5",
                   label=f"above threshold ({fit.n_high})")
        ax.axhline(fit.threshold, ls="--", color="0.0", lw=0.6,
                   label=f"threshold = {fit.threshold:+.3f}")
        ax.set_xlabel("t"); ax.set_ylabel("y_0")
        ax.set_title("TVAR regime classification by threshold on y_0(t-1)")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "tvar_threshold.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'tvar_threshold.png'}")


if __name__ == "__main__":
    main()
