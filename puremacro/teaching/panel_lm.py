"""Panel local projections via linearmodels.PanelOLS with Driscoll-Kraay SE."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def _prepare(df_wide: pd.DataFrame, y: str, x: str, n_lags: int,
             controls: Sequence[str]) -> pd.DataFrame:
    # Build a sorted (entity, time) DataFrame with lags pre-computed.
    df = df_wide.copy()
    df.index = df.index.set_names(["entity", "time"])
    df = df.sort_index()
    for l in range(1, n_lags + 1):
        for c in [x, y] + list(controls):
            df[f"{c}_L{l}"] = df.groupby(level="entity")[c].shift(l)
    return df


def panel_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.32,
    time_effects: bool = False,
) -> pd.DataFrame:
    """Panel LP with country FE (and optional time FE) + Driscoll-Kraay SE.

    Defaults to **entity-only FE** because uncertainty proxies are
    largely common across countries — time FE would absorb the very
    aggregate variation we're measuring (Great Recession, COVID, etc.).
    Pass ``time_effects=True`` only when the proxy has substantial
    cross-country idiosyncratic variation that survives time-demeaning.

    Estimates y_{i,t+h} = α_i (+ λ_t) + β_h x_{it} + Σ_l γ_l Z_{i,t-l} + u_{i,t+h}.
    """
    from scipy.stats import norm

    horizons = list(horizons)
    ctl = list(controls)
    df = _prepare(df_wide, y, x, n_lags, ctl)
    reg_cols = [x] + ctl + [f"{c}_L{l}" for l in range(1, n_lags + 1) for c in [x, y] + ctl]

    rows = []
    for h in horizons:
        tmp = df.copy()
        tmp["__y"] = tmp.groupby(level="entity")[y].shift(-h)
        cols = ["__y"] + reg_cols
        tmp = tmp.dropna(subset=cols)
        n_entities = tmp.index.get_level_values("entity").nunique()
        if len(tmp) < 10 or n_entities < 2:
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "lo": np.nan,
                         "hi": np.nan, "n_obs": 0})
            continue
        mod = PanelOLS(
            dependent=tmp["__y"],
            exog=tmp[reg_cols],
            entity_effects=True,
            time_effects=time_effects,
            drop_absorbed=True,
        )
        res = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=max(h + 1, 1))
        b = res.params[x]
        s = res.std_errors[x]
        z = abs(norm.ppf(alpha / 2))
        rows.append({"h": h, "beta": b, "se": s, "lo": b - z * s, "hi": b + z * s,
                     "n_obs": int(res.nobs)})
    return pd.DataFrame(rows)
