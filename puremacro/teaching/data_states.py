"""State-level monthly panel loader for T4.

Joins:
  - state nonfarm employment (from fred_states.fetch_state_employment)
  - state quarterly income and wages (forward-filled to monthly)
  - state-EPU (Baker-Bloom-Davis-Kost)
  - state coincident index -> derived rolling-sigma of Dlog (state_cond_vol)
  - state employment GARCH-sigma

Output: DataFrame with MultiIndex (state, date) and numeric columns.
All upstream fetchers are wrapped in _fetch_* helpers for monkeypatching.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "state_monthly_uncertainty_panel.csv"


# Lazy-imported upstream fetchers via indirection so tests can monkeypatch.
def _fetch_emp() -> pd.DataFrame:
    from ..fetch import fred_states
    return fred_states.fetch_state_employment()


def _fetch_income_q() -> pd.DataFrame:
    from ..fetch import fred_states
    return fred_states.fetch_state_income()


def _fetch_phci() -> pd.DataFrame:
    from ..fetch import frb_phil_coincident
    return frb_phil_coincident.fetch_state_coincident()


def _fetch_state_epu() -> pd.DataFrame:
    from ..fetch import epu_states
    return epu_states.fetch()


def _fetch_state_emp_garch() -> pd.DataFrame:
    from ..fetch import fred_states
    return fred_states.fetch_state_employment_garch_sigma()


def _rolling_sigma_from_tidy(
    tidy: pd.DataFrame, *, window: int = 24, min_periods: int = 12
) -> pd.DataFrame:
    """Compute rolling sigma of Dlog(value) per state, using only lagged data."""
    tidy = tidy.sort_values(["state", "date"]).copy()
    tidy["dlog"] = tidy.groupby("state")["value"].transform(
        lambda s: np.log(s).diff()
    )
    tidy["rolling_sigma"] = tidy.groupby("state")["dlog"].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    return tidy[["state", "date", "rolling_sigma"]]


def load_us_states(start: str = "1985-01", end: str = "2024-12") -> pd.DataFrame:
    """Return state monthly panel with MultiIndex (state, date)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # 1. employment (monthly, non-tidy)
    emp = _fetch_emp().copy()
    emp["date"] = pd.to_datetime(emp["date"])
    emp = emp[["state", "date", "log_emp"]]

    # 2. income + wages (quarterly, non-tidy)
    inc = _fetch_income_q().copy()
    inc["date"] = pd.to_datetime(inc["date"])
    inc["log_pi_real"] = inc["log_pi"]
    inc["log_wage_real"] = inc["log_wage"]
    inc_q = inc[["state", "date", "log_pi_real", "log_wage_real"]]

    # 3. state-EPU (tidy -> wide-by-variable)
    se = _fetch_state_epu().rename(columns={"value": "state_epu"})
    se = se[["state", "date", "state_epu"]]

    # 4. PHCI -> rolling sigma
    ph = _fetch_phci()
    ph_sigma = _rolling_sigma_from_tidy(ph)
    ph_sigma = ph_sigma.rename(columns={"rolling_sigma": "state_cond_vol"})

    # 5. state employment GARCH-sigma
    eg = _fetch_state_emp_garch().rename(columns={"value": "state_emp_garch"})
    eg = eg[["state", "date", "state_emp_garch"]]

    # Union of monthly dates
    dates = pd.date_range(start_ts, end_ts, freq="MS")
    states = sorted(set(emp["state"]).union(se["state"]).union(ph["state"]))
    grid = pd.MultiIndex.from_product(
        [states, dates], names=["state", "date"]
    ).to_frame(index=False)

    merged = (grid
        .merge(emp, on=["state", "date"], how="left")
        .merge(se, on=["state", "date"], how="left")
        .merge(ph_sigma, on=["state", "date"], how="left")
        .merge(eg, on=["state", "date"], how="left")
    )

    # Merge quarterly income + forward-fill within each state.
    inc_q = inc_q.sort_values(["state", "date"])
    merged = merged.sort_values(["state", "date"]).reset_index(drop=True)
    merged = merged.merge(inc_q, on=["state", "date"], how="left")
    merged[["log_pi_real", "log_wage_real"]] = (
        merged.groupby("state")[["log_pi_real", "log_wage_real"]].ffill()
    )

    out = merged.set_index(["state", "date"]).sort_index()

    # Cache
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_CACHE)
    return out
