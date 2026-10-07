"""Thin wrapper: estimate Smets-Wouters (2007) via the generic
puremacro.dsge.estimate_dsge engine.

The model-specific bits — bundled-data loading, OBSERVED_VARS
validation, the fixed (non-estimated) calibrated parameters, and the
starting values of the mode search — live here. Everything else (mode
refinement, Hessian, proposal-cov construction, MH chains) is in
puremacro.dsge.estimate.

Since 4.6.0 the mode search starts from the ``.mod`` file's estimated_params
initial values (``SW07_MOD_INITIAL_VALUES``) rather than from the rounded
Table 1a/1b Mode column: two entries of that column lie outside the prior
support, and from the clipped start the default 100-iteration optimiser
stopped short of the mode with a warning. ``dataset="authors"`` selects the
authors' own 1947Q3-2004Q4 series, and ``presample``/``lik_init`` are
forwarded to :func:`puremacro.dsge.estimate.estimate_dsge`.
"""
from __future__ import annotations

import importlib.resources
from typing import Optional

import pandas as pd

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_data import load_sw07_data
from puremacro.dsge.sw07_priors import PRIORS, SW07_MOD_INITIAL_VALUES


# Calibrated (NOT estimated) SW07 parameters; merged into every observation_eq
# call. These appear in SW07_POSTERIOR_MODE but NOT in PRIORS.
_FIXED_PARAMS = {
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
}


def _load_bundled_data(dataset: str = "fred") -> pd.DataFrame:
    """The bundled SW07 observables, 1966Q1-2004Q4 (156 quarters).

    Built by ``tools/build_sw07_data.py`` from FRED following the SW07 data
    appendix (ECB WP 722, printed p. 47): per-capita real GDP, consumption
    and investment growth and real-wage growth in percent (100 x log
    differences), ``log_hours`` = 100 x log of per-capita hours (NFB average
    hours x civilian employment / population 16+, demeaned), ``infl`` = 100 x
    log difference of the GDP deflator, ``ffr`` = federal funds rate / 4. The
    index is the first day of each quarter.
    

    ``dataset="authors"`` returns the authors' 1947Q3-2004Q4 series instead
    (``_sw07_usmodel_data.csv``); see :func:`puremacro.dsge.sw07_data.load_sw07_data`,
    which keeps the ``"1966Q1"`` labels. Here the index is the first day of each
    quarter, as it has been since the dataset was bundled.
    """
    df = load_sw07_data(dataset)
    idx = pd.PeriodIndex(df.index, freq="Q").to_timestamp()
    df.index = pd.DatetimeIndex(idx.to_numpy(), name="date")
    return df


def _validate_data(df: pd.DataFrame) -> None:
    missing = set(OBSERVED_VARS) - set(df.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(df) < 50:
        raise ValueError(f"data has only {len(df)} obs; need >= 50")
    if df[list(OBSERVED_VARS)].isna().any().any():
        raise ValueError("data contains NaN values")


def _start_values(start) -> dict[str, float]:
    """Starting values of the mode search for ``estimate_sw07(start=...)``."""
    if isinstance(start, dict):
        return dict(start)
    if start == "mod":
        return dict(SW07_MOD_INITIAL_VALUES)
    if start == "table1":
        return {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    raise ValueError(f"start must be 'mod', 'table1' or a dict of parameter values, got {start!r}")


def estimate_sw07(
    data: Optional[pd.DataFrame] = None,
    *,
    dataset: str = "fred",
    start="mod",
    presample: int = 0,
    lik_init: str = "stationary",
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk MH.

    Thin wrapper over :func:`puremacro.dsge.estimate.estimate_dsge`.

    Parameters
    ----------
    data : DataFrame with columns OBSERVED_VARS; if None, loads the bundled
        dataset named by ``dataset``. User data must use the paper's units:
        growth rates and hours in 100 x log, inflation and the interest rate
        in quarterly percent.
    dataset : {"fred", "authors"}
        Bundled series to use when ``data`` is None: puremacro's FRED rebuild
        (1966Q1-2004Q4, 156 quarters; the default and the basis of the
        replication fixture) or the authors' own series (1947Q3-2004Q4; see
        :func:`puremacro.dsge.sw07_data.load_sw07_data`). New in 4.6.0.
    start : {"mod", "table1"} or dict
        Starting values of the mode search: the ``.mod`` file's
        estimated_params initial values (default since 4.6.0), the rounded
        Table 1a/1b Mode column (the default through 4.5.0, two entries of
        which are clipped into the prior support), or explicit values.
    presample, lik_init
        Forwarded to :func:`estimate_dsge` (Dynare's options of the same
        names). New in 4.6.0.
    n_draws, n_chains, burn_in, seed : MCMC controls.
    """
    df = _load_bundled_data(dataset) if data is None else data.copy()
    _validate_data(df)
    return estimate_dsge(
        df,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=list(OBSERVED_VARS),
        initial_params=_start_values(start),
        fixed_params=_FIXED_PARAMS,
        model_name="SW07",
        n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
        presample=presample, lik_init=lik_init,
    )


__all__ = ["estimate_sw07"]
