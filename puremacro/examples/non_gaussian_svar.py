"""SVAR identification by non-Gaussianity (Lanne-Meitz-Saikkonen 2017).

We simulate a 2-variable VAR driven by Laplace shocks (heavy-tailed,
strongly non-Gaussian) and compare:
  - Cholesky identification (depends on variable ordering),
  - non-Gaussian identification via FastICA on the residuals.

The latter recovers the structural impact matrix up to permutation /
sign without imposing any sign or zero restrictions, simply by exploiting
the independence + non-Gaussianity of the structural shocks.

Run:
    python -m puremacro.examples.non_gaussian_svar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.estimate import estimate_var
from ..var.identify.cholesky import cholesky_factor
from ..var.identify import non_gaussian_svar
from ..var.irf import irf as compute_irf


def _simulate(T: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.10], [0.05, 0.4]])
    B0_true = np.array([[1.0, 0.5], [0.3, 1.0]])
    shocks = rng.laplace(scale=0.7, size=(T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B0_true @ shocks[t]
    return Y, A, B0_true


def run_demo() -> dict:
    Y, A_true, B0_true = _simulate()
    A_list, _, Sigma, _, _ = estimate_var(Y, p=1)
    B0_chol = cholesky_factor(Sigma)
    irf_chol = compute_irf(A_list, B0_chol, horizon=8)
    out_ng = non_gaussian_svar(Y, p=1, horizon=8, seed=0)
    return {"Y": Y, "B0_true": B0_true,
            "B0_chol": B0_chol, "irf_chol": irf_chol,
            "ng": out_ng}


def main() -> None:
    out = run_demo()
    print("Non-Gaussian SVAR (LMS 2017) vs Cholesky on Laplace-shock VAR")
    print()
    print("  True B0 (up to column permutation & sign):")
    print(np.round(out["B0_true"], 3))
    print()
    print("  Cholesky B0 (depends on variable ordering):")
    print(np.round(out["B0_chol"], 3))
    print()
    print("  Non-Gaussian B0 (columns ordered by descending |excess kurt|):")
    print(np.round(out["ng"].B0, 3))
    print(f"  Excess kurtosis of recovered shocks: "
          f"{np.round(out['ng'].kurtosis, 2).tolist()}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        irf_chol = out["irf_chol"]
        irf_ng = out["ng"].irf
        H = irf_chol.shape[0]
        h_arr = np.arange(H)
        fig, axes = plt.subplots(2, 2, figsize=(8, 4.5),
                                  sharex=True, sharey="row")
        for i in range(2):
            for j in range(2):
                ax = axes[i, j]
                ax.plot(h_arr, irf_chol[:, i, j], color="0.5", lw=1.0,
                        label="Cholesky")
                ax.plot(h_arr, irf_ng[:, i, j], color="0.0", lw=1.0,
                        label="Non-Gaussian")
                ax.axhline(0, color="0.6", lw=0.4)
                ax.set_title(f"y_{i} ← shock {j}")
                if (i, j) == (0, 0):
                    ax.legend(frameon=False, loc="best", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("h")
        fig.suptitle("IRFs: Cholesky vs Non-Gaussian SVAR")
        fig.tight_layout()
        fig.savefig(out_dir / "non_gaussian_svar.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'non_gaussian_svar.png'}")


if __name__ == "__main__":
    main()
