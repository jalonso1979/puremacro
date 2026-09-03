"""Tests for FertilitySolution.irf and FertilitySolution.fevd."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _solve():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return solve_fertility()


def test_irf_returns_long_dataframe_with_var_columns():
    from puremacro.dsge.fertility_adj_costs import VAR_NAMES
    sol = _solve()
    irf = sol.irf("ea", horizon=10)
    assert isinstance(irf, pd.DataFrame)
    assert irf.shape == (11, 12)
    assert tuple(irf.columns) == VAR_NAMES


def test_irf_horizon_zero_returns_impact_only():
    sol = _solve()
    irf = sol.irf("ea", horizon=0)
    assert len(irf) == 1


def test_irf_unknown_shock_name_raises():
    sol = _solve()
    with pytest.raises(ValueError, match="unknown shock"):
        sol.irf("bad_shock_name", horizon=4)


# ---------------------------------------------------------------------------
# Regression test for the IRF control timing (fixed after 1.9.0).
# ---------------------------------------------------------------------------
def test_irf_controls_read_the_lagged_state():
    """`_compute_irf` must reproduce the recursion `solve_fertility` builds.

    The solver partitions its policy matrix as

        x_t = G x_{t-1} + N eps_t        (states)
        y_t = F x_{t-1} + L eps_t        (controls)

    so a control at t reads the LAGGED state. The IRF loop advanced the state
    and *then* applied F, which evaluates F at x_t rather than x_{t-1}: the
    reported y at horizon h was exactly the correct y at horizon h + 1, for
    every h >= 1. The states were unaffected, so the paths still looked
    entirely reasonable — and every existing test in this file passed both
    before and after the fix, because none of them pinned the timing.

    Note `L != F @ N` in this model (max difference 0.28), so the two readings
    are genuinely different objects and the h = 0 row cannot disambiguate them.
    """
    import numpy as np

    sol = _solve()
    horizon = 6
    for s_idx, shock in enumerate(sol.shock_names):
        sigma_key = {"ea": "sigmaa", "ep": "sigmap", "en": "sigman"}[shock]
        sigma = sol.params.get(sigma_key, 1.0)

        x_prev = np.zeros(sol.G.shape[0])
        expected = []
        for h in range(horizon + 1):
            eps = sigma if h == 0 else 0.0
            x_t = sol.G @ x_prev + sol.N[:, s_idx] * eps
            y_t = sol.F @ x_prev + sol.L[:, s_idx] * eps
            expected.append(np.concatenate([x_t, y_t]))
            x_prev = x_t

        got = sol.irf(shock, horizon=horizon).to_numpy()
        np.testing.assert_allclose(
            got, np.array(expected), atol=1e-10,
            err_msg=(
                f"IRF for shock {shock!r} departs from x_t = G x_{{t-1}} + N eps, "
                f"y_t = F x_{{t-1}} + L eps"
            ),
        )
