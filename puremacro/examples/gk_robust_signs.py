"""Giacomini-Kitagawa (2021) robust bands vs RWZ posterior bands
on a sign-restricted SVAR.

A standard sign-restriction SVAR (RWZ / Uhlig) reports posterior
quantiles of the IRF across Haar-uniform Q draws. That mixes
information from the data with the *uniform-on-Q* prior, which is not
identifying. Giacomini-Kitagawa 2021 separate the two: at each VAR
draw, the *identified set* is the (min, max) of the IRF across all Q
that satisfy the sign restrictions. This example contrasts the
two approaches on the same DGP.

Run:
    python -m puremacro.examples.gk_robust_signs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..var.identify.sign import sign_restriction_svar
from ..var.identify.sign_robust import gk_robust_bands


def _simulate(T: int = 250, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.6,  0.10, 0.05],
        [-0.05, 0.5,  0.10],
        [0.0,   0.10, 0.55],
    ])
    Sigma = np.array([
        [1.0, 0.4, 0.1],
        [0.4, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return Y


def run_demo(H: int = 8) -> dict:
    Y = _simulate()
    # Sign restrictions on shock 1: y0 up, y1 up at h=0 and h=1; y2 unrestricted.
    restrictions = {0: [+1, +1, 0], 1: [+1, +1, 0]}
    rwz = sign_restriction_svar(Y, p=2, horizon=H, restrictions=restrictions,
                                n_draws=2000, ci=0.9, seed=0)
    gk = gk_robust_bands(Y, p=2, horizon=H, restrictions=restrictions,
                        shock_idx=0, n_var_draws=80, n_q_per_draw=400,
                        ci=0.9, seed=0)
    return {"rwz_median": rwz.irf_median, "rwz_lo": rwz.irf_lower,
            "rwz_hi": rwz.irf_upper, "gk": gk, "H": H}


def main() -> None:
    out = run_demo()
    H = out["H"]
    rwz_med = out["rwz_median"][:, :, 0]   # response of vars i to shock 0
    rwz_lo = out["rwz_lo"][:, :, 0]
    rwz_hi = out["rwz_hi"][:, :, 0]
    gk_med = out["gk"].irf_median
    gk_lo = out["gk"].irf_lower
    gk_hi = out["gk"].irf_upper
    print("Sign restrictions: RWZ posterior bands vs GK robust identified-set bands")
    print(f"  Mean accepted Q draws per VAR: {out['gk'].n_accepted_per_draw.mean():.1f}")
    print()
    print("Variable 2 (unrestricted), response to shock 0:")
    print("    h   RWZ median   RWZ band       GK median   GK identified set")
    for h in range(H + 1):
        print(f"  {h:>3d}   {rwz_med[h, 2]:+.3f}     "
              f"[{rwz_lo[h, 2]:+.3f}, {rwz_hi[h, 2]:+.3f}]   "
              f"{gk_med[h, 2]:+.3f}     "
              f"[{gk_lo[h, 2]:+.3f}, {gk_hi[h, 2]:+.3f}]")
    print()
    print("Note: the identified set is typically *wider* than the RWZ posterior")
    print("band — it reflects only what the data + sign restrictions identify.")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
        h_arr = np.arange(H + 1)
        for i, ax in enumerate(axes):
            ax.fill_between(h_arr, gk_lo[:, i], gk_hi[:, i],
                            color="0.85", label="GK identified set")
            ax.fill_between(h_arr, rwz_lo[:, i], rwz_hi[:, i],
                            color="0.55", alpha=0.7, label="RWZ 90% posterior")
            ax.plot(h_arr, rwz_med[:, i], color="0.0", lw=1.0)
            ax.axhline(0.0, color="0.5", lw=0.4)
            ax.set_xlabel("Horizon h")
            ax.set_title(f"y_{i}")
            if i == 0:
                ax.set_ylabel("Response to shock 0")
                ax.legend(frameon=False, loc="best")
        fig.suptitle("Sign-restriction SVAR: RWZ vs GK robust bands")
        fig.tight_layout()
        fig.savefig(out_dir / "gk_robust_signs.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'gk_robust_signs.png'}")


if __name__ == "__main__":
    main()
