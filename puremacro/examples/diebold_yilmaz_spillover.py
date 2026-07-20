"""Diebold-Yilmaz (2012) volatility-spillover index on a synthetic 4-variable VAR.

We simulate a stable VAR(1) with off-diagonal coefficients so each
variable both feeds and absorbs shocks from its neighbours. The total
spillover index, directional to/from measures, and pairwise spillover
matrix are computed from the Cholesky FEVD at horizon 12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..connectedness.diebold_yilmaz import spillover_index


def _simulate_var4(T: int = 500, seed: int = 0, burn: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = np.array([
        [0.5, 0.1, 0.0, 0.05],
        [0.1, 0.4, 0.1, 0.00],
        [0.0, 0.1, 0.5, 0.10],
        [0.05, 0.0, 0.1, 0.45],
    ])
    n = A.shape[0]
    eps = rng.standard_normal((T + burn, n))
    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = A @ Y[t - 1] + eps[t]
    return pd.DataFrame(Y[burn:], columns=[f"v{i}" for i in range(n)])


def run_diebold_yilmaz(
    T: int = 500,
    seed: int = 0,
    var_lags: int = 2,
    fevd_horizon: int = 12,
) -> dict:
    """Compute spillover measures on the synthetic 4-variable VAR DGP."""
    df = _simulate_var4(T=T, seed=seed)
    out = spillover_index(
        df, var_lags=var_lags, fevd_horizon=fevd_horizon,
    )
    out["data"] = df
    return out


def main() -> None:
    out = run_diebold_yilmaz()
    print("Diebold-Yilmaz spillover index on synthetic 4-variable VAR")
    print(f"  Total spillover index:  {out['total']:.2f} %")
    print(f"  Directional 'to':       {out['directional_to']}")
    print(f"  Directional 'from':     {out['directional_from']}")
    print(f"  Net spillover:          {out['net']}")
    print(f"  Pairwise matrix shape:  {out['pairwise_matrix'].shape}")


if __name__ == "__main__":
    main()
