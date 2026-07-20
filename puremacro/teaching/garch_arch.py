"""AR(1)-GARCH(1,1) volatility proxy via the ``arch`` package."""
from __future__ import annotations

import pandas as pd
from arch import arch_model


def fit_ar_garch11(series: pd.Series, rescale: bool = False) -> pd.Series:
    """Fit AR(1) mean + GARCH(1,1) vol; return conditional σ_t with the series' index."""
    s = series.dropna().astype(float)
    if len(s) < 50:
        raise ValueError(f"need ≥ 50 obs to fit AR(1)-GARCH(1,1); got {len(s)}")
    res = arch_model(s, mean="AR", lags=1, vol="GARCH", p=1, q=1, rescale=rescale).fit(disp="off")
    sigma: pd.Series = pd.Series(res.conditional_volatility).dropna()
    # Re-index to the tail of the input series so σ_t lines up with y_t.
    sigma.index = s.index[-len(sigma):]
    # Reindex to the full input support with forward-fill for the dropped AR burn-in
    # (so downstream callers can join by date without holes).
    sigma = sigma.reindex(s.index).bfill()
    return sigma.rename(f"sigma({series.name or 'x'})")
