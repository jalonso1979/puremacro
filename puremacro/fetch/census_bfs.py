"""Census Business Formation Statistics (BFS) — state-level monthly.

The BFS publishes the count of *Business Applications* (BA) and the count
of *Business Formations* projected within 4 / 8 quarters (BF4Q, BF8Q),
both at the U.S. and state level. Series begin in 2004-07 and are
released roughly two weeks after the reference month.

We pull the canonical state-level BA series via FRED (which mirrors the
Census release):

    BABATOTALSA{ST}   — Total BA, SA, monthly  (canonical entry signal)
    BABATOTNSA{ST}    — Total BA, NSA, monthly (fallback if SA missing)

The High-Propensity BA and BF4Q series (BAHBA, BFRWDI) are published by
Census but are NOT reliably mirrored to FRED at state granularity as of
2026 — caller must download those CSVs manually from
https://www.census.gov/econ/bfs/data/ if needed.

Output: long-form DataFrame with columns
``[state, date, variable, value, sa_source, source]`` matching the
convention in `src/fetch/epu_states.py`.

Without a FRED API key this raises a clear error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from puremacro import credentials

logger = logging.getLogger(__name__)

_OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "us_states" / "state_bfs_monthly.csv"

# 50 states + DC (51 series).
_STATES = (
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL",
    "GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
)

# (FRED series suffix, output variable name, sa_source tag).
# Order matters: SA primary, NSA fallback inside fetch().
_SERIES = (
    ("BABATOTALSA",  "bfs_ba_total_sa",  "FRED-SA"),
    ("BABATOTALNSA", "bfs_ba_total_nsa", "FRED-NSA"),
)


def fetch(refresh: bool = False, *, api_key: str | None = None,
          states: Iterable[str] = _STATES) -> pd.DataFrame:
    """Return tidy long-form BFS panel — state, date, variable, value.

    Caches to ``data/raw/us_states/state_bfs_monthly.csv``. Set ``refresh=True``
    to re-pull all 153 series from FRED.
    """
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    if _OUT.exists() and not refresh:
        return pd.read_csv(_OUT, parse_dates=["date"])

    try:
        from fredapi import Fred
    except ImportError as e:
        raise RuntimeError(
            "BFS fetcher needs fredapi: pip install fredapi. "
            "Or download CSVs manually from https://www.census.gov/econ/bfs/data/"
        ) from e

    key = credentials.require("fred", explicit=api_key)

    import time
    fred = Fred(api_key=key)
    rows = []
    for st in states:
        for prefix, var, sa in _SERIES:
            sid = f"{prefix}{st}"
            try:
                s = fred.get_series(sid).dropna()
            except Exception as e:
                logger.warning("BFS: %s missing on FRED (%s)", sid, e)
                continue
            if s.empty:
                continue
            rows.append(pd.DataFrame({
                "state": st, "date": s.index, "variable": var,
                "value": s.values.astype(float),
                "sa_source": sa, "source": f"FRED:{sid}",
            }))
            time.sleep(0.05)  # gentle rate-limit guard (FRED limit ≈ 120/min)
    if not rows:
        raise RuntimeError(
            "Pulled 0 BFS series. Check FRED status and series IDs: "
            f"{[p+'{ST}' for p,_,_ in _SERIES]}"
        )
    out = pd.concat(rows, ignore_index=True)
    out["log_value"] = np.log(out["value"].replace(0, np.nan))
    out = out.sort_values(["state", "variable", "date"]).reset_index(drop=True)
    out.to_csv(_OUT, index=False)
    return out


def fetch_us(refresh: bool = False, *, api_key: str | None = None) -> pd.DataFrame:
    """National BFS aggregates (US totals) for context."""
    try:
        from fredapi import Fred
    except ImportError as e:
        raise RuntimeError("BFS fetcher needs fredapi") from e
    key = credentials.require("fred", explicit=api_key)
    fred = Fred(api_key=key)
    rows = []
    for prefix, var, sa in _SERIES:
        sid = f"{prefix}US"
        try:
            s = fred.get_series(sid).dropna()
        except Exception:
            continue
        rows.append(pd.DataFrame({
            "code": "USA", "date": s.index, "variable": var,
            "value": s.values.astype(float),
            "sa_source": sa, "source": f"FRED:{sid}",
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
