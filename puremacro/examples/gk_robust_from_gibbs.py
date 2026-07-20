"""Posterior-aware sign-restriction bands: Giacomini-Kitagawa 2021
fed by ``minnesota_gibbs`` posterior draws.

The Tier 1 ``gk_robust_bands`` uses a recursive bootstrap to summarise
VAR uncertainty. The proper Bayesian way is to feed the posterior
draws of (A, Σ) directly. This example contrasts the two on the same
DGP — they should agree in shape, but the Gibbs-fed version is the
GK-2021-correct construction.

Run:
    python -m puremacro.examples.gk_robust_from_gibbs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.bvar import minnesota_gibbs
from ..var.identify.sign_robust import gk_robust_bands, gk_robust_bands_from_gibbs


def _simulate(T: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.10, 0.05],
                   [-0.05, 0.5, 0.10],
                   [0.0, 0.10, 0.55]])
    Sigma = np.array([[1.0, 0.4, 0.1],
                       [0.4, 1.0, 0.2],
                       [0.1, 0.2, 1.0]]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return Y


def run_demo(H: int = 8) -> dict:
    Y = _simulate()
    restrictions = {0: [+1, +1, 0], 1: [+1, +1, 0]}
    boot = gk_robust_bands(Y, p=2, horizon=H, restrictions=restrictions,
                            n_var_draws=60, n_q_per_draw=300, ci=0.9, seed=0)
    g = minnesota_gibbs(Y, p=2, n_draws=200, burn=200,
                        rng=np.random.default_rng(0))
    bayes = gk_robust_bands_from_gibbs(g["A_draws"], g["Sigma_draws"],
                                        horizon=H, restrictions=restrictions,
                                        n_q_per_draw=300, ci=0.9, seed=0)
    return {"H": H, "boot": boot, "bayes": bayes}


def main() -> None:
    out = run_demo()
    H = out["H"]
    boot = out["boot"]; bay = out["bayes"]
    print("GK robust bands: bootstrap (Tier 1) vs Gibbs-posterior (Tier 5)")
    print(f"  Mean Q-acceptances per draw — bootstrap: "
          f"{boot.n_accepted_per_draw.mean():.1f}  "
          f"Gibbs: {bay.n_accepted_per_draw.mean():.1f}")
    print()
    print("Variable 2 (unrestricted), response to shock 0:")
    print("    h    boot median   boot band            bayes median   bayes band")
    for h in range(H + 1):
        print(f"  {h:>3d}    "
              f"{boot.irf_median[h, 2]:+.3f}      "
              f"[{boot.irf_lower[h, 2]:+.3f}, "
              f"{boot.irf_upper[h, 2]:+.3f}]   "
              f"{bay.irf_median[h, 2]:+.3f}        "
              f"[{bay.irf_lower[h, 2]:+.3f}, "
              f"{bay.irf_upper[h, 2]:+.3f}]")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
        h_arr = np.arange(H + 1)
        for i, ax in enumerate(axes):
            ax.fill_between(h_arr, boot.irf_lower[:, i],
                             boot.irf_upper[:, i],
                             color="0.85", label="bootstrap")
            ax.fill_between(h_arr, bay.irf_lower[:, i],
                             bay.irf_upper[:, i],
                             color="0.55", alpha=0.7, label="Gibbs")
            ax.plot(h_arr, bay.irf_median[:, i], color="0.0", lw=1.0)
            ax.set_xlabel("Horizon h")
            ax.set_title(f"y_{i}")
            if i == 0:
                ax.set_ylabel("Response to shock 0")
                ax.legend(frameon=False, loc="best")
        fig.suptitle("GK robust bands: bootstrap vs Gibbs posterior")
        fig.tight_layout()
        fig.savefig(out_dir / "gk_robust_from_gibbs.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'gk_robust_from_gibbs.png'}")


if __name__ == "__main__":
    main()
