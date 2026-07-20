"""Shared simulation helpers for puremacro.dynpanel tests."""
from __future__ import annotations

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


@pytest.fixture
def small_panel():
    """Small simulated panel for smoke tests."""
    return simulate_dynamic_panel(N=50, T=6, rho=0.5, seed=42)


@pytest.fixture
def medium_panel():
    """Medium simulated panel for routine estimator tests."""
    return simulate_dynamic_panel(N=150, T=8, rho=0.5, seed=42)
