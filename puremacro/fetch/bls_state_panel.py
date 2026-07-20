"""State-panel fetchers via FRED CSV (mirrors BLS data; no quota).

Three fetchers wrap ``puremacro.fetch._classic.fetch_fred`` for state-
level series:

  - ``iter_state_urate_q`` — monthly SA state urate from FRED ``{ST}UR``,
    averaged to quarterly.
  - ``iter_state_employment_q`` — monthly SA state total nonfarm
    employment from FRED ``{ST}NA``, averaged to quarterly, returned
    as log(thousands).
  - ``iter_state_participation_q`` — monthly SA state LFPR from FRED
    ``LBSSA{FIPS}``, averaged to quarterly.

(Slice 6a's bls_state_panel originally hit the BLS API directly but
the 25-req/day free-tier limit was a hard blocker. FRED CSV is the
canonical channel — no API key required when puremacro's _safe_urlopen
sends a non-Mozilla UA.)

Output records are 5-tuples ``(state_code, qdate, value, source_url,
metadata)``. State universe: 50 states + DC = 51 cross-sectional units.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ._classic import fetch_fred


_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# 2-digit FIPS codes for 50 states + DC.
_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def _monthly_series_to_quarterly(s: pd.Series) -> dict[pd.Timestamp, float]:
    """Average monthly values within each quarter."""
    if s is None or s.empty:
        return {}
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    s.index = pd.to_datetime(s.index)
    q = s.resample("QE").mean()
    return {ts: float(v) for ts, v in q.dropna().items()}


def iter_state_urate_q(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, qdate, urate_pct, source_url, metadata) per state-quarter.

    FRED series: ``{ST}UR`` (monthly SA state unemployment rate).
    """
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        if st not in _FIPS:
            continue
        sid = f"{st}UR"
        try:
            series = fetch_fred(sid)
        except Exception:
            continue
        q = _monthly_series_to_quarterly(series)
        url = _FRED_BASE + sid
        for qdate, val in sorted(q.items()):
            yield (st, qdate, val, url, {
                "series_id": sid, "freq": "Q",
                "source": "FRED (BLS LAUS mirror)",
                "measure": "urate_pct_sa",
            })


def iter_state_employment_q(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, qdate, log_emp_thousands, source_url, metadata).

    FRED series: ``{ST}NA`` (monthly SA state nonfarm employment, thousands).
    Value is log(thousands).
    """
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        if st not in _FIPS:
            continue
        sid = f"{st}NA"
        try:
            series = fetch_fred(sid)
        except Exception:
            continue
        q = _monthly_series_to_quarterly(series)
        url = _FRED_BASE + sid
        for qdate, emp in sorted(q.items()):
            if emp <= 0:
                continue
            yield (st, qdate, math.log(emp), url, {
                "series_id": sid, "freq": "Q",
                "source": "FRED (BLS CES mirror)",
                "measure": "log_total_nonfarm_sa_thousands",
            })


def iter_state_participation_q(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, qdate, lfpr_pct, source_url, metadata).

    FRED series: ``LBSSA{FIPS}`` (monthly SA state LFPR).
    """
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        fips = _FIPS.get(st)
        if fips is None:
            continue
        sid = f"LBSSA{fips}"
        try:
            series = fetch_fred(sid)
        except Exception:
            continue
        q = _monthly_series_to_quarterly(series)
        url = _FRED_BASE + sid
        for qdate, val in sorted(q.items()):
            yield (st, qdate, val, url, {
                "series_id": sid, "freq": "Q",
                "source": "FRED (BLS LAUS mirror)",
                "measure": "lfpr_pct_sa",
            })


# Backward-compat alias for the prior annual fetcher name:
iter_state_participation_a = iter_state_participation_q

__all__ = [
    "iter_state_urate_q",
    "iter_state_employment_q",
    "iter_state_participation_q",
    "iter_state_participation_a",
]
