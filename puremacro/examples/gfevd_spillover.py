"""Diebold-Yilmaz spillover with GFEVD: order-invariance demo.

Cholesky-based FEVD depends on the column ordering of the panel:
re-order the variables and the spillover index changes. The
Pesaran-Shin (1998) generalised FEVD is order-invariant — which is
what Diebold-Yilmaz (2012) actually use, but earlier implementations
(including the original ``puremacro`` code) defaulted to Cholesky.

This demo prints the spillover index under both schemes for two
distinct orderings of the same panel. Cholesky should change; GFEVD
should not.

Run:
    python -m puremacro.examples.gfevd_spillover
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..connectedness.diebold_yilmaz import spillover_index


def _simulate(T: int = 600, seed: int = 0) -> pd.DataFrame:
    """Simulate a 4-variable VAR(1) with strongly *correlated* shocks
    (off-diagonal Sigma). This is what makes Cholesky's variable
    ordering bite — with diagonal Sigma, Cholesky and GFEVD give the
    same numbers regardless of order."""
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.5, 0.10, 0.05, 0.0],
        [0.10, 0.4, 0.10, 0.05],
        [0.05, 0.10, 0.5, 0.10],
        [0.0,  0.05, 0.10, 0.45],
    ])
    Sigma = np.array([
        [1.0, 0.6, 0.4, 0.2],
        [0.6, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0],
    ])
    L = np.linalg.cholesky(Sigma)
    n = 4
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return pd.DataFrame(Y, columns=list("ABCD"))


def run_demo() -> dict:
    df = _simulate()
    df_perm = df[["D", "C", "B", "A"]]  # reverse order

    chol_native = spillover_index(df, var_lags=2, fevd_horizon=12,
                                   identification="cholesky")
    chol_perm = spillover_index(df_perm, var_lags=2, fevd_horizon=12,
                                identification="cholesky")
    gfev_native = spillover_index(df, var_lags=2, fevd_horizon=12,
                                   identification="gfevd")
    gfev_perm = spillover_index(df_perm, var_lags=2, fevd_horizon=12,
                                identification="gfevd")
    return {
        "chol_native": chol_native, "chol_perm": chol_perm,
        "gfev_native": gfev_native, "gfev_perm": gfev_perm,
    }


def main() -> None:
    out = run_demo()
    print("Diebold-Yilmaz spillover index — order-invariance check")
    print(f"  Cholesky FEVD,  order ABCD : {out['chol_native']['total']:.2f} %")
    print(f"  Cholesky FEVD,  order DCBA : {out['chol_perm']['total']:.2f} %  "
          f"(should differ)")
    print(f"  GFEVD,          order ABCD : {out['gfev_native']['total']:.2f} %")
    print(f"  GFEVD,          order DCBA : {out['gfev_perm']['total']:.2f} %  "
          f"(should match)")
    delta_chol = abs(out['chol_native']['total'] - out['chol_perm']['total'])
    delta_gfev = abs(out['gfev_native']['total'] - out['gfev_perm']['total'])
    print()
    print(f"  |Cholesky delta| = {delta_chol:.3f}    "
          f"|GFEVD delta| = {delta_gfev:.3f}")


if __name__ == "__main__":
    main()
