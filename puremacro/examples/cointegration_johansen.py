"""Engle-Granger and Johansen cointegration on a 3-variable simulated system.

We construct a 3-variable I(1) system driven by one common stochastic
trend (cointegrating rank r=1). Engle-Granger should reject no
cointegration on the y0 vs y1 pair, and Johansen's trace test should
identify rank r=1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..var.vecm import engle_granger, fit_vecm, johansen


def _simulate_cointegrated(T: int = 400, seed: int = 0) -> pd.DataFrame:
    """Generate three I(1) series sharing one common stochastic trend."""
    rng = np.random.default_rng(seed)
    # Common random-walk trend
    eta = rng.standard_normal(T)
    tau = np.cumsum(eta)
    # Idiosyncratic stationary noise
    e0 = 0.4 * rng.standard_normal(T)
    e1 = 0.4 * rng.standard_normal(T)
    e2 = 0.4 * rng.standard_normal(T)
    y0 = tau + e0
    y1 = 0.7 * tau + e1
    y2 = -0.5 * tau + e2
    return pd.DataFrame({"y0": y0, "y1": y1, "y2": y2})


def run_johansen(
    T: int = 400,
    seed: int = 0,
    p: int = 2,
) -> dict:
    """Run Engle-Granger + Johansen + VECM on the simulated DGP."""
    df = _simulate_cointegrated(T=T, seed=seed)

    eg = engle_granger(df["y0"].values, df["y1"].values)
    joh = johansen(df, p=p)
    vecm = fit_vecm(df, p=p, r=1)

    return {
        "engle_granger": eg,
        "johansen": joh,
        "vecm": vecm,
        "data": df,
    }


def main() -> None:
    out = run_johansen()
    eg = out["engle_granger"]
    joh = out["johansen"]
    print("Cointegration on simulated 3-variable system (1 common trend)")
    print(f"  Engle-Granger beta_hat:    {eg['cointegrating_beta']:+.3f}")
    print(f"  Engle-Granger ADF t-stat:  {eg['adf_t']:+.3f}")
    print(f"  Johansen eigenvalues:      {joh['eigenvalues']}")
    print(f"  Johansen trace stats:      {joh['trace_stats']}")
    print(f"  Johansen 5% crit values:   {joh['crit_values']}")
    print(f"  Johansen estimated rank:   r_hat = {joh['r_hat']}")
    print(f"  VECM Pi shape:             {out['vecm']['Pi'].shape}")


if __name__ == "__main__":
    main()
