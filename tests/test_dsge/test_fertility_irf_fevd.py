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
