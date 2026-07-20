"""Panel LP with two-way FE + Driscoll-Kraay (1998) HAC SE.

Thin wrapper over the shared engine in
:mod:`puremacro.lp._panel_helpers`. Driscoll-Kraay is robust to
heteroskedasticity, autocorrelation, and cross-sectional dependence;
default lag truncation L = ⌊4·(T/100)^(2/9)⌋ on the within-projected
design.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ._panel_helpers import _focal_dk_se, panel_lp_horizon_loop


def panel_lp_dk(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
) -> pd.DataFrame:
    """Two-way FE panel LP with Driscoll-Kraay HAC SE.

    Returns DataFrame with columns ``[h, beta, se, t, lo, hi]``.
    """
    return panel_lp_horizon_loop(
        df_wide,
        y=y, x=x,
        horizons=horizons,
        n_lags=n_lags,
        controls=controls,
        alpha=alpha,
        entity_level=entity_level,
        time_level=time_level,
        se_fn=_focal_dk_se,
    )


__all__ = ["panel_lp_dk"]
