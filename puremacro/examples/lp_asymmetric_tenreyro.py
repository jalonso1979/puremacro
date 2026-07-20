"""Tenreyro-Thwaites (2016)-style asymmetric local projection on synthetic data.

We simulate a DGP where positive shocks to ``x`` propagate strongly into
``y`` while negative shocks are partially absorbed (smaller |response|).
Running ``lp_asymmetric`` recovers two separate IRF columns ``beta_pos``
and ``beta_neg`` whose magnitudes differ by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..lp.asymmetric import lp_asymmetric


def _simulate(T: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    # Asymmetric loading: positive x -> 1.0 * x; negative x -> 0.3 * x
    eps = 0.5 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(5, T):
        x_pos_lag = max(x[t - 1], 0.0) + 0.6 * max(x[t - 2], 0.0)
        x_neg_lag = min(x[t - 1], 0.0) + 0.6 * min(x[t - 2], 0.0)
        y[t] = 0.5 * y[t - 1] + 1.0 * x_pos_lag + 0.3 * x_neg_lag + eps[t]
    df = pd.DataFrame({"y": y, "x": x})
    df.index = pd.RangeIndex(T)
    return df


def run_asymmetric_lp(
    T: int = 500,
    seed: int = 0,
    horizons=range(0, 9),
) -> dict:
    """Estimate the asymmetric LP. Returns the IRF DataFrame."""
    df = _simulate(T=T, seed=seed)
    irf = lp_asymmetric(
        df, y="y", x="x", horizons=horizons, n_lags=2, alpha=0.10,
    )
    return {
        "irf_pos": irf[["h", "beta_pos", "se_pos", "lo_pos", "hi_pos"]],
        "irf_neg": irf[["h", "beta_neg", "se_neg", "lo_neg", "hi_neg"]],
        "irf_full": irf,
        "data": df,
    }


def main() -> None:
    out = run_asymmetric_lp()
    irf = out["irf_full"]
    print("Tenreyro-Thwaites asymmetric LP on synthetic asymmetric DGP")
    cols = ["h", "beta_pos", "lo_pos", "hi_pos", "beta_neg", "lo_neg", "hi_neg"]
    print(irf[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
