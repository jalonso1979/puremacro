"""Tests for puremacro.dsge.sw07_observation."""
import importlib.resources

import numpy as np
import pandas as pd
import pytest


def test_make_state_space_returns_StateSpaceModel():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    from puremacro.state_space import StateSpaceModel
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert isinstance(ssm, StateSpaceModel)


def test_make_state_space_shapes():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    n_state = ssm.T.shape[0]
    n_obs = ssm.Z.shape[0]
    assert n_state == 44
    assert n_obs == 7


def test_make_state_space_q_psd():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    eigs = np.linalg.eigvalsh(ssm.Q)
    assert (eigs >= -1e-10).all()


def test_make_state_space_measurement_intercept_d_finite():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert np.isfinite(ssm.d).all()
    assert ssm.d.shape == (7,)


def test_make_state_space_h_small_ridge():
    """H (measurement-error cov) is a small positive ridge."""
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert ssm.H.shape == (7, 7)
    assert (np.diag(ssm.H) > 0).all()
    assert (np.diag(ssm.H) < 1e-4).all()


def test_make_state_space_log_likelihood_finite_at_mode():
    """End-to-end: Kalman filter on the bundled data at SW07_POSTERIOR_MODE
    must produce a finite log-likelihood."""
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    from puremacro.state_space import kalman_filter

    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)

    pkg = importlib.resources.files("puremacro.dsge")
    df = pd.read_csv(pkg / "_sw07_data.csv", comment="#", parse_dates=["date"], index_col="date")
    y = df[[
        "gdp_growth", "cons_growth", "inv_growth",
        "wage_growth", "log_hours", "infl", "ffr",
    ]].to_numpy()

    out = kalman_filter(y, ssm)
    assert np.isfinite(out["loglik"])
