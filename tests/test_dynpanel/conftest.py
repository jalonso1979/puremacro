"""Shared simulation helpers for puremacro.dynpanel tests."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def simulate_dynamic_panel(
    N: int = 100,
    T: int = 6,
    rho: float = 0.5,
    beta_exog: float | None = None,
    sigma_eps: float = 1.0,
    sigma_alpha: float = 1.0,
    burnin: int = 50,
    seed: int = 0,
) -> dict:
    """Simulate a dynamic linear panel:

        y_{i,t} = ρ · y_{i,t-1} + β · x_{i,t} + α_i + ε_{i,t}

    where ``α_i ~ N(0, σ_α^2)``, ``ε_{i,t} ~ N(0, σ_ε^2)``, and ``x_{i,t}``
    is a strictly exogenous regressor when ``beta_exog`` is not None.

    Returns a dict with:
        ``y, panel_id, time_id`` long-format arrays of length N*T
        ``x`` exogenous regressor as a (N*T, 1) matrix or None
        ``rho_true, beta_true`` the true parameters
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, sigma_alpha, size=N)
    if beta_exog is not None:
        # x_{i,t} ~ N(0, 1), iid across i, t
        x_full = rng.normal(0.0, 1.0, size=(N, burnin + T))
    else:
        x_full = np.zeros((N, burnin + T))

    y_full = np.zeros((N, burnin + T))
    eps_full = rng.normal(0.0, sigma_eps, size=(N, burnin + T))
    # initial condition: stationary mean
    y_full[:, 0] = alpha / max(1.0 - rho, 0.05) + eps_full[:, 0]
    for t in range(1, burnin + T):
        contrib = rho * y_full[:, t - 1] + alpha + eps_full[:, t]
        if beta_exog is not None:
            contrib = contrib + beta_exog * x_full[:, t]
        y_full[:, t] = contrib

    # take last T columns as the observed sample
    y_obs = y_full[:, -T:]
    x_obs = x_full[:, -T:] if beta_exog is not None else None

    # long-format
    panel_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N).astype(np.int64)
    y = y_obs.ravel()
    if x_obs is not None:
        x = x_obs.ravel()[:, None]
    else:
        x = None

    return {
        "y": y,
        "panel_id": panel_id,
        "time_id": time_id,
        "x": x,
        "rho_true": rho,
        "beta_true": beta_exog,
        "N": N,
        "T": T,
    }


# ---------------------------------------------------------------------
# Stata's abdata (Arellano-Bond 1991 employment panel), for the
# [XT] xtabond replication tests. The data are NOT bundled: point
# PUREMACRO_ABDATA at a copy of https://www.stata-press.com/data/r19/abdata.dta
# (or a CSV with columns id, year, n, w, k, ys), or drop abdata.dta /
# abdata.csv into tests/fixtures/. Tests skip when no copy is found.
# ---------------------------------------------------------------------
_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _find_abdata() -> Path | None:
    env = os.environ.get("PUREMACRO_ABDATA")
    candidates = [Path(env)] if env else []
    candidates += [_FIXTURES / "abdata.dta", _FIXTURES / "abdata.csv"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_abdata_xtabond_design() -> dict | None:
    """Return the design of Stata's xtabond Examples 1-5 on abdata.

    ``xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984 year, lags(2)``:
    ``X_exog`` holds, in Stata's coefficient order, w, L.w, k, L.k, L2.k,
    ys, L.ys, L2.ys, yr1980..yr1984, year (lags taken within firm and only
    across consecutive years). Values are converted to float64 exactly as
    Stata does when it computes with float-stored data.
    Returns None when no copy of abdata is available.
    """
    import pandas as pd

    path = _find_abdata()
    if path is None:
        return None
    if path.suffix.lower() == ".dta":
        df = pd.read_stata(path)
    else:
        df = pd.read_csv(path)
    needed = {"id", "year", "n", "w", "k", "ys"}
    if needed - set(df.columns):
        return None
    df = df.sort_values(["id", "year"]).reset_index(drop=True)
    for c in ("id", "year", "n", "w", "k", "ys"):
        df[c] = df[c].astype(np.float64)
    g = df.groupby("id")

    def lag(col: str, L: int):
        s = g[col].shift(L)
        consecutive = (df["year"] - g["year"].shift(L)) == L
        return s.where(consecutive)

    year = df["year"]
    X = np.column_stack(
        [
            df["w"], lag("w", 1),
            df["k"], lag("k", 1), lag("k", 2),
            df["ys"], lag("ys", 1), lag("ys", 2),
            *[(year == yr).astype(float) for yr in range(1980, 1985)],
            year,
        ]
    ).astype(float)
    names = [
        "L1.n", "L2.n", "w", "L1.w", "k", "L1.k", "L2.k",
        "ys", "L1.ys", "L2.ys",
        "yr1980", "yr1981", "yr1982", "yr1983", "yr1984", "year",
    ]
    return {
        "y": df["n"].to_numpy(),
        "panel_id": df["id"].to_numpy().astype(np.int64),
        "time_id": year.to_numpy().astype(np.int64),
        "X_exog": X,
        "names": names,
        "path": str(path),
    }


@pytest.fixture(scope="session")
def abdata_xtabond():
    design = load_abdata_xtabond_design()
    if design is None:
        pytest.skip(
            "abdata not available: set PUREMACRO_ABDATA to a copy of "
            "https://www.stata-press.com/data/r19/abdata.dta (not bundled)."
        )
    return design


@pytest.fixture
def small_panel():
    """Small simulated panel for smoke tests."""
    return simulate_dynamic_panel(N=50, T=6, rho=0.5, seed=42)


@pytest.fixture
def medium_panel():
    """Medium simulated panel for routine estimator tests."""
    return simulate_dynamic_panel(N=150, T=8, rho=0.5, seed=42)
