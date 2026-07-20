"""Thin wrapper: estimate Smets-Wouters (2007) via the generic
puremacro.dsge.estimate_dsge engine.

The model-specific bits — bundled-data loading, OBSERVED_VARS
validation, the fixed (non-estimated) calibrated parameters, and the
initial-params construction from SW07_POSTERIOR_MODE + SW07_SHOCK_STDS —
live here. Everything else (mode refinement, Hessian, proposal-cov
construction, MH chains) is in puremacro.dsge.estimate.
"""
from __future__ import annotations

import importlib.resources
from typing import Optional

import pandas as pd

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import PRIORS


# Calibrated (NOT estimated) SW07 parameters; merged into every observation_eq
# call. These appear in SW07_POSTERIOR_MODE but NOT in PRIORS.
_FIXED_PARAMS = {
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
}


def _load_bundled_data() -> pd.DataFrame:
    pkg = importlib.resources.files("puremacro.dsge")
    return pd.read_csv(
        pkg / "_sw07_data.csv",
        comment="#", parse_dates=["date"], index_col="date",
    )


def _validate_data(df: pd.DataFrame) -> None:
    missing = set(OBSERVED_VARS) - set(df.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(df) < 50:
        raise ValueError(f"data has only {len(df)} obs; need >= 50")
    if df[list(OBSERVED_VARS)].isna().any().any():
        raise ValueError("data contains NaN values")


def estimate_sw07(
    data: Optional[pd.DataFrame] = None,
    *,
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk MH.

    Thin wrapper over :func:`puremacro.dsge.estimate.estimate_dsge`.

    Parameters
    ----------
    data : DataFrame with columns OBSERVED_VARS; if None, loads the
        bundled 1966Q1-2004Q4 US dataset (156 quarterly obs × 7 cols).
    n_draws, n_chains, burn_in, seed : MCMC controls.
    """
    df = _load_bundled_data() if data is None else data.copy()
    _validate_data(df)
    initial_params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    return estimate_dsge(
        df,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=list(OBSERVED_VARS),
        initial_params=initial_params,
        fixed_params=_FIXED_PARAMS,
        model_name="SW07",
        n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
    )


__all__ = ["estimate_sw07"]
