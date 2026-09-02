"""OECD nominal exchange rates (LCU per USD), monthly.

Dataflow: ``OECD.SDD.STES,DSD_KEI@DF_KEI`` (7 dimensions:
``REF_AREA . FREQ . MEASURE . UNIT_MEASURE . ACTIVITY . ADJUSTMENT . TRANSFORMATION``)
with ``MEASURE=CC`` ("Nominal exchange rates") and
``UNIT_MEASURE=XDC_USD`` ("National currency per US dollar"), monthly.

Returns long-form schema: ``code, date, variable=xrate_usd_per_local_m,
value (= LCU/USD; e.g. ~150 for JPY), sa_source, source``.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

_EMPTY = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])


def fetch_xrate_monthly(
    codes: Iterable[str] | None = None,
    *,
    start_period: str = "1995",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch monthly LCU-per-USD nominal exchange rates for the given codes."""
    code_key = "+".join(c.upper() for c in codes) if codes else ""
    # OECD.SDD.STES,DSD_KEI@DF_KEI has 7 dimensions:
    # REF_AREA . FREQ . MEASURE . UNIT_MEASURE . ACTIVITY . ADJUSTMENT . TRANSFORMATION
    # MEASURE=CC = "Nominal exchange rates"; UNIT_MEASURE=XDC_USD = LCU/USD.
    key = f"{code_key}.M.CC.XDC_USD..."
    from ._oecd_sdmx import get_sdmx_csv
    df = get_sdmx_csv("OECD.SDD.STES,DSD_KEI@DF_KEI,", key, start_period,
                      refresh=refresh)
    if df.empty:
        return _EMPTY.copy()
    if df.empty or "OBS_VALUE" not in df.columns:
        return _EMPTY.copy()

    df = df.dropna(subset=["OBS_VALUE"]).copy()
    df = df[pd.to_numeric(df["OBS_VALUE"], errors="coerce") > 0]
    if df.empty:
        return _EMPTY.copy()
    df = df.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
    try:
        dates = pd.to_datetime(df["TIME_PERIOD"].astype(str) + "-01")
    except Exception:
        return _EMPTY.copy()
    out = pd.DataFrame({
        "code": df["REF_AREA"].values,
        "date": dates.values,
        "variable": "xrate_usd_per_local_m",
        "value": pd.to_numeric(df["OBS_VALUE"], errors="coerce").astype(float).values,
        "sa_source": "none",
        "source": "OECD:DSD_KEI@DF_KEI:CC:XDC_USD",
    })
    return out


__all__ = ["fetch_xrate_monthly"]
