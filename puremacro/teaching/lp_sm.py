"""Single-country local projections (Jordà 2005) via statsmodels OLS with HAC SE."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def lp_ols_hac(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.32,
) -> pd.DataFrame:
    """Run y_{t+h} = α_h + β_h x_t + Σ γ_l (controls_{t-l}) + u_{t+h} for each h.

    Returns DataFrame with columns h, beta, se, t, lo, hi.
    SE use Newey-West with bandwidth = h+1 (Plagborg-Møller & Wolf 2021).
    ``alpha`` is the two-sided significance level (0.32 → 68% bands).
    """
    horizons = list(horizons)
    ctl = list(controls or [])
    rows = []
    for h in horizons:
        tmp = df.copy()
        tmp["__y"] = tmp[y].shift(-h)
        reg_cols = [x] + ctl
        for l in range(1, n_lags + 1):
            for c in [x, y] + ctl:
                tmp[f"__{c}_L{l}"] = tmp[c].shift(l)
                reg_cols.append(f"__{c}_L{l}")
        tmp = tmp.dropna(subset=["__y"] + reg_cols)
        X = sm.add_constant(tmp[reg_cols], has_constant="add")
        y_vec = tmp["__y"].astype(float)
        model = sm.OLS(y_vec, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h + 1, 1)})
        b = model.params[x]
        s = model.bse[x]
        from scipy.stats import norm
        z = abs(norm.ppf(alpha / 2))
        rows.append({"h": h, "beta": b, "se": s, "t": b / s,
                     "lo": b - z * s, "hi": b + z * s})
    return pd.DataFrame(rows)
