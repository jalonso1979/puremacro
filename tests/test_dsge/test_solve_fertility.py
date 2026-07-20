"""Tests for puremacro.dsge.fertility_adj_costs.solve_fertility."""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _solve():
    """Helper: run solve_fertility, suppressing the BGP over-determined warning."""
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return solve_fertility()


def test_solve_fertility_returns_dataclass():
    from puremacro.dsge._results import FertilitySolution
    sol = _solve()
    assert isinstance(sol, FertilitySolution)
    assert sol.var_names[:5] == ("a", "mun", "ph", "k", "n")
    assert sol.shock_names == ("ea", "ep", "en")


def test_solve_fertility_blanchard_kahn_satisfied():
    sol = _solve()
    assert sol.klein_solution.eu == (1, 1), (
        f"Blanchard-Kahn failed: eu={sol.klein_solution.eu}"
    )


def test_solve_fertility_g_matrix_stable_eigenvalues():
    sol = _solve()
    eigvals = np.linalg.eigvals(sol.G)
    max_modulus = float(np.max(np.abs(eigvals)))
    assert max_modulus < 1.0 - 1e-6, (
        f"G is not contractive: max|eig|={max_modulus:.4f}"
    )


def test_productivity_shock_positive_output_response():
    sol = _solve()
    irf = sol.irf("ea", horizon=8)
    assert irf["y"].iloc[0] > 0, (
        "impact response of y to productivity shock should be positive"
    )
    assert irf["y"].iloc[4] > 0, (
        "horizon-4 response of y to productivity shock should still be positive"
    )


def test_mortality_shock_negative_fertility_response():
    sol = _solve()
    irf = sol.irf("ep", horizon=8)
    n_path = irf["n"].iloc[:5].to_numpy()
    assert np.any(n_path < 0), (
        f"expected at least one negative entry in n's IRF to mortality shock; got {n_path}"
    )
