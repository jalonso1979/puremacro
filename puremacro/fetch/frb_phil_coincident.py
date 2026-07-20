"""FRB Philadelphia state coincident indexes (Crone-Clayton-Matheson).

FRED IDs: {ST}PHCI — 50 states + DC, monthly, 1979-01+. Seasonally adjusted.

Returns tidy long-form (state, date, variable, value, sa_source, source).
Imports fredapi lazily inside the function so this module is importable
in environments where fredapi isn't installed (e.g., offline test runners).
"""
from __future__ import annotations

import pandas as pd


_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL",
    "GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]


def _default_fred():
    """Lazy import so this module is importable without fredapi installed."""
    from dotenv import load_dotenv
    from fredapi import Fred as _Fred
    load_dotenv()
    return _Fred


# Expose Fred as a module-level name so tests can monkeypatch it.
Fred = _default_fred if False else None


def _get_fred_cls():
    global Fred
    if Fred is None:
        Fred = _default_fred()
    return Fred


def fetch_state_coincident(api_key: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return FRB Phil state coincident indexes as tidy long-form frame.

    Note: DC is not covered by the FRB Philadelphia state coincident index
    program (DCPHCI does not exist on FRED), so the loop silently skips
    any {ST}PHCI series that returns 400 Bad Request. Other transient
    failures also skip the offending state rather than abort the whole
    fetch.
    """
    from puremacro import credentials
    FredCls = _get_fred_cls()
    key = credentials.require("fred", explicit=api_key)
    fred = FredCls(api_key=key)
    rows = []
    skipped: list[tuple[str, str]] = []
    for st in _STATES:
        try:
            s = fred.get_series(f"{st}PHCI")
        except Exception as e:  # noqa: BLE001 — fredapi raises ValueError on 400
            skipped.append((st, str(e)[:120]))
            continue
        # Accept pandas Series
        try:
            s = s.dropna()
        except AttributeError:
            s = pd.Series(s)
        for dt, v in s.items():
            rows.append({
                "state": st,
                "date": pd.Timestamp(dt),
                "variable": "state_coincident",
                "value": float(v),
                "sa_source": "SA",
                "source": "philadelphiafed.org",
            })
    if skipped:
        print(f"[frb_phil_coincident] skipped {len(skipped)} state(s): "
              + ", ".join(st for st, _ in skipped))
    out = pd.DataFrame(rows).sort_values(["state", "date"]).reset_index(drop=True)
    return out[["state", "date", "variable", "value", "sa_source", "source"]]


if __name__ == "__main__":
    df = fetch_state_coincident(refresh=True)
    print(f"{len(df):,} rows, {df['state'].nunique()} states")
