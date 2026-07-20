"""Smoke tests for puremacro.dsge.estimate_sw07 — runs with small n_draws."""
import time

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_estimate_sw07_tiny_runs_clean():
    """n_draws=500, n_chains=1, burn_in=200 — runs without error in <600s."""
    from puremacro.dsge import estimate_sw07, SW07PosteriorResult
    from puremacro.dsge.sw07_priors import param_names

    t0 = time.time()
    result = estimate_sw07(n_draws=500, n_chains=1, burn_in=200, seed=0)
    elapsed = time.time() - t0

    assert isinstance(result, SW07PosteriorResult)
    n_params = len(param_names())
    assert result.draws.shape == (1, 500, n_params)
    assert result.log_posterior_trace.shape == (1, 500)
    assert len(result.accept_rates) == 1
    assert 0.05 <= result.accept_rates[0] <= 0.65
    assert np.isfinite(result.draws).all()
    assert elapsed < 600


@pytest.mark.slow
def test_estimate_sw07_summary_dataframe():
    """summary() returns a DataFrame with the expected columns + per-param rows."""
    from puremacro.dsge import estimate_sw07
    from puremacro.dsge.sw07_priors import param_names

    result = estimate_sw07(n_draws=300, n_chains=1, burn_in=100, seed=1)
    s = result.summary()
    assert list(s.columns) == ["mean", "std", "q5", "q50", "q95", "mode"]
    assert list(s.index) == list(param_names())
    assert np.isfinite(s.values).all()


def test_estimate_sw07_with_user_data_runs():
    """Synthetic user data (length 60) should run the pipeline without error."""
    from puremacro.dsge import estimate_sw07
    from puremacro.dsge.sw07_observation import OBSERVED_VARS

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        0.1 * rng.standard_normal((60, 7)),
        columns=list(OBSERVED_VARS),
        index=pd.date_range("2000-01-01", periods=60, freq="QE"),
    )
    result = estimate_sw07(data=df, n_draws=200, n_chains=1, burn_in=50, seed=2)
    assert result.data_n_obs == 60
