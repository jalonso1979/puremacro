"""Fetch US state-level nonfarm payroll employment from FRED.

Series ID pattern: ``{ST}NAN`` — All Employees, Total Nonfarm (thousands),
Seasonally Adjusted, monthly, one series per state (2-letter postal code).

Also fetches state-level private nonfarm employment (``{ST}NAP``) and
average weekly hours in manufacturing (``SMU{fips}..`` pattern) where
available, but the primary deliverable is total payroll employment.

Output is a tidy CSV ``data/raw/us_states/state_employment_monthly.csv``
with columns ``[state, date, log_emp]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 50 states + DC (51 series). Skip territories.
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "us_states" / "state_employment_monthly.csv"
OUT_INC = Path(__file__).resolve().parents[2] / "data" / "raw" / "us_states" / "state_income_quarterly.csv"


def _client(api_key: str | None = None):
    from puremacro import credentials
    from fredapi import Fred
    key = credentials.require("fred", explicit=api_key)
    return Fred(api_key=key)


def fetch_state_employment(refresh: bool = False) -> pd.DataFrame:
    """Fetch monthly employment for each state and return tidy long frame.

    Caches to ``data/raw/us_states/state_employment_monthly.csv``.
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not refresh:
        return pd.read_csv(OUT, parse_dates=["date"])

    fred = _client()
    rows = []
    for st in STATES:
        s = fred.get_series(f"{st}NAN")  # SA, thousands of persons
        s = s.dropna()
        df = pd.DataFrame({"state": st, "date": s.index, "emp": s.values})
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["log_emp"] = np.log(out["emp"].astype(float))
    out.to_csv(OUT, index=False)
    return out


def fetch_state_income(refresh: bool = False) -> pd.DataFrame:
    """Fetch quarterly state-level income from FRED (BEA SQ1).

    Two series per state:
      - ``{ST}OTOT`` — total personal income, SAAR, million USD, quarterly.
      - ``{ST}WTOT`` — wage and salary disbursements, SAAR, million USD, quarterly
        (1990+ only; older states may be missing).

    Caches to ``data/raw/us_states/state_income_quarterly.csv`` with columns
    ``[state, date, pi, wage, log_pi, log_wage]``.
    """
    OUT_INC.parent.mkdir(parents=True, exist_ok=True)
    if OUT_INC.exists() and not refresh:
        return pd.read_csv(OUT_INC, parse_dates=["date"])

    fred = _client()
    rows = []
    for st in STATES:
        try:
            pi = fred.get_series(f"{st}OTOT").dropna()
        except Exception:
            pi = pd.Series(dtype=float)
        try:
            wage = fred.get_series(f"{st}WTOT").dropna()
        except Exception:
            wage = pd.Series(dtype=float)
        idx = pi.index.union(wage.index)
        df = pd.DataFrame({"state": st, "date": idx})
        df["pi"] = pi.reindex(idx).values
        df["wage"] = wage.reindex(idx).values
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["log_pi"] = np.log(out["pi"].astype(float))
    out["log_wage"] = np.log(out["wage"].astype(float))
    out.to_csv(OUT_INC, index=False)
    return out


# --- GARCH-σ on state employment growth -----------------------------------
# `arch` is lazy-imported inside fetch_state_employment_garch_sigma below;
# see ARCHITECTURE.md for the Pyodide contract.

OUT_GARCH = Path(__file__).resolve().parents[2] / "data" / "processed" / "state_employment_garch_sigma.csv"


def fetch_state_employment_garch_sigma(refresh: bool = False) -> pd.DataFrame:
    """Return AR(1)-GARCH(1,1) conditional σ of monthly Δlog emp per state.

    Output: tidy long-form with variable='state_emp_garch'.
    """
    from arch import arch_model as _arch_model  # lazy: Pyodide contract

    OUT_GARCH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GARCH.exists() and not refresh:
        return pd.read_csv(OUT_GARCH, parse_dates=["date"])

    emp = fetch_state_employment(refresh=False).sort_values(["state", "date"]).copy()
    rows = []
    for st, sub in emp.groupby("state"):
        sub = sub.sort_values("date").reset_index(drop=True)
        growth = 100 * np.log(sub["emp"].astype(float)).diff()
        growth.index = pd.DatetimeIndex(sub["date"])
        growth = growth.dropna()
        if len(growth) < 60:
            continue
        try:
            fit = _arch_model(growth, mean="AR", lags=1, vol="GARCH", p=1, q=1,
                              dist="normal", rescale=False).fit(disp="off")
            sigma = pd.Series(fit.conditional_volatility).dropna()
        except Exception:
            continue
        for d, s in zip(sigma.index, sigma.values):
            rows.append({
                "state": st, "date": pd.Timestamp(d),
                "variable": "state_emp_garch", "value": float(s),
                "sa_source": "SA", "source": "derived_from_FRED",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GARCH, index=False)
    return out


if __name__ == "__main__":
    df_e = fetch_state_employment(refresh=True)
    print(df_e.groupby("state")["date"].agg(["min", "max", "count"]).head())
    print(f"Employment: {len(df_e):,} rows, {df_e['state'].nunique()} states → {OUT}\n")

    df_i = fetch_state_income(refresh=True)
    print(df_i.groupby("state")[["pi", "wage"]].count().head())
    print(f"Income: {len(df_i):,} rows, {df_i['state'].nunique()} states → {OUT_INC}")
