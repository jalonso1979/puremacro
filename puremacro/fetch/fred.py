"""FRED fetcher, backed by the official fredapi client.

FRED_API_KEY must be set in the environment. Register free at
https://fred.stlouisfed.org/docs/api/api_key.html.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # fredapi is a [data]-extra side-channel dep — lazy at runtime
    from fredapi import Fred

# Vertical-slice US series. All SA where applicable (FRED series selected with SA already applied).
US_SERIES_MAP = {
    "log_gdp_real":          ("GDPC1",    "log"),     # Real GDP, SA, quarterly
    "log_gfcf_real":         ("GPDIC1",   "log"),     # Real GPDI (investment), SA, quarterly
    "log_hours":             ("HOANBS",   "log"),     # Nonfarm business hours, SA, quarterly
    "log_emp":               ("PAYEMS",   "log"),     # All employees, total nonfarm, SA, monthly
    "urate":                 ("UNRATE",   "level"),   # Civilian unemployment rate, SA, monthly
    "log_cpi":               ("CPIAUCSL", "log"),     # CPI-U, SA, monthly
    "short_rate":            ("FEDFUNDS", "level"),   # Effective Federal Funds Rate, monthly
    "log_cons_real":         ("PCECC96",  "log"),     # Real PCE, SA, quarterly — for KPR RBC comparison
    "log_deflator":          ("GDPDEF",   "log"),     # GDP deflator, SA, quarterly
    "vix":                   ("VIXCLS",   "level"),   # VIX, daily — aggregate to M/Q at panel build
    "log_ip":                ("INDPRO",   "log"),     # Industrial production, SA, monthly
    "log_ip_capgoods":       ("IPBUSEQ",  "log"),     # IP: Business equipment, SA, monthly
    "log_hours_manuf":       ("AWHMAN",   "log"),     # Avg weekly hours, manufacturing, SA, monthly
    "log_nonres_biz_inv_ind": ("NEWORDER", "log"),    # New orders nondef. cap. goods ex air, SA, monthly
    "log_wages_real":        ("COMPRNFB", "log"),     # Real hourly compensation, nonfarm business, SA, quarterly
    "log_income_real":       ("DSPIC96",  "log"),     # Real disposable personal income, SA, monthly
}


def _client(api_key: str | None = None) -> "Fred":
    from fredapi import Fred  # lazy import behind the guard (see module docstring)
    from puremacro import credentials
    key = credentials.require("fred", explicit=api_key)
    return Fred(api_key=key)


def fetch_series(fred_id: str, *, variable: str, code: str = "USA", refresh: bool = False) -> pd.DataFrame:
    """Fetch one FRED series in long format. ``refresh`` is advisory — fredapi caches internally."""
    s = _client().get_series(fred_id)
    s.index = pd.to_datetime(s.index)
    df = s.reset_index()
    df.columns = ["date", "raw_value"]
    transform = US_SERIES_MAP.get(variable, (fred_id, "level"))[1]
    if transform == "log":
        df["value"] = np.log(df["raw_value"].astype(float))
    else:
        df["value"] = df["raw_value"].astype(float)
    df["code"] = code
    df["variable"] = variable
    df["sa_source"] = "fred"
    df["source"] = f"FRED:{fred_id}"
    return df[["code", "date", "variable", "value", "sa_source", "source"]].dropna(subset=["value"])


def fetch_all_us(refresh: bool = False) -> pd.DataFrame:
    frames = []
    for var, (fred_id, _) in US_SERIES_MAP.items():
        try:
            frames.append(fetch_series(fred_id, variable=var, code="USA", refresh=refresh))
        except Exception as e:  # noqa: BLE001
            print(f"[fred] {var} ({fred_id}) failed: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "date", "variable", "value", "sa_source", "source"]
    )
