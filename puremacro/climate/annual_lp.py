"""Annual panel LP for climate (CDD, HDD) shocks.

Delegates to ``puremacro.lp.panel_lp_dk`` (Driscoll-Kraay HAC SE) twice
— once for the CDD shock with HDD as a control, once for HDD with CDD
as a control. This partials out the joint heating/cooling comovement
so each reported IRF is the *partial* response to that shock.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ..lp.panel_dk import panel_lp_dk


def climate_annual_lp(
    panel: pd.DataFrame,
    *,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    region_col: str = "region",
    year_col: str = "year",
    alpha: float = 0.10,
) -> dict:
    """Run two panel LPs (CDD shock, HDD shock) on the same panel.

    The input ``panel`` is long-form with one row per (region, year) and
    must contain ``response``, ``cdd_col``, ``hdd_col``, and all of
    ``controls``. Internally converted to the MultiIndex shape that
    ``panel_lp_dk`` expects.

    Returns
    -------
    dict with keys ``'cdd'`` and ``'hdd'``. Each value is the long
    DataFrame returned by ``panel_lp_dk`` with columns
    ``[h, beta, se, t, lo, hi]``.
    """
    horizons = list(horizons)
    controls = list(controls)
    needed_cols = {response, cdd_col, hdd_col, region_col, year_col} | set(controls)
    missing = needed_cols - set(panel.columns)
    if missing:
        raise KeyError(f"climate_annual_lp: panel missing columns {sorted(missing)}")
    # Build MultiIndex (entity, time) DataFrame as panel_lp_dk expects.
    wide = (
        panel[list(needed_cols)]
        .dropna()
        .copy()
        .set_index([region_col, year_col])
        .sort_index()
    )
    wide.index.names = ["code", "date"]
    cdd_result = panel_lp_dk(
        wide, y=response, x=cdd_col,
        horizons=horizons, n_lags=n_lags,
        controls=[hdd_col] + controls,
        alpha=alpha,
        entity_level="code", time_level="date",
    )
    hdd_result = panel_lp_dk(
        wide, y=response, x=hdd_col,
        horizons=horizons, n_lags=n_lags,
        controls=[cdd_col] + controls,
        alpha=alpha,
        entity_level="code", time_level="date",
    )
    return {"cdd": cdd_result, "hdd": hdd_result}


__all__ = ["climate_annual_lp"]
