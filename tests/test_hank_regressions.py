"""Regression tests for the 2.3.1 fixes to the non-linear sequence-space HANK
solver (audit findings C7, C47, C48 and the r4 majors).

Before 2.3.1 the Broyden solver failed for monetary shocks of 200 bp and above
at the documented default horizon (300) and returned a -200%-of-GDP path with
``converged=False`` and no warning; ``linear_path`` and the starting Jacobian
came from hard-coded heuristic formulas, so the "non-linear minus linear"
difference was mostly Jacobian error; a zero shock produced a non-zero path;
``simulate_targeted_transfer`` accepted an unused ``beta`` argument and its
"deciles" were asset-grid bins.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from puremacro.models.hank_sequence_space import (
    fake_news_algorithm,
    simulate_targeted_transfer,
    solve_hank_sequence_space,
    solve_nonlinear_transition,
)


@pytest.fixture(scope="module")
def ss():
    return solve_hank_sequence_space()


def _rate_shock(bp: float, horizon: int) -> np.ndarray:
    return (bp / 10_000.0) * (0.7 ** np.arange(horizon))


@pytest.mark.parametrize("bp", [500.0, -500.0, 200.0])
def test_large_shocks_converge_at_documented_defaults(ss, bp):
    """+/-500 bp and +200 bp converge with horizon=300, max_iter=100, tol=1e-6 in < 5 s."""
    t0 = time.perf_counter()
    res = solve_nonlinear_transition(ss, _rate_shock(bp, 300), shock_var="r", horizon=300)
    elapsed = time.perf_counter() - t0
    assert res.converged
    assert res.iterations < 100
    assert float(np.max(np.abs(res.residuals))) < 1e-6
    assert elapsed < 5.0
    # the path is a sensible fraction of GDP, not the old -200%
    assert float(np.max(np.abs(res.nonlinear_path))) < 0.5


def test_zero_shock_is_an_exact_fixed_point(ss):
    res = solve_nonlinear_transition(ss, np.zeros(60), shock_var="r", horizon=60)
    assert res.converged
    assert res.iterations == 0
    assert float(np.max(np.abs(res.nonlinear_path))) < 1e-10
    assert float(np.max(np.abs(res.residuals))) < 1e-9


def test_nonlinear_path_converges_to_linear_path_for_small_shocks(ss):
    """linear_path must be the linearisation of the solved model: the impact
    response per unit of shock agrees with the non-linear one to 2% at 1e-4."""
    small = 1e-4 * (0.7 ** np.arange(80))
    res = solve_nonlinear_transition(ss, small, shock_var="r", horizon=80)
    assert res.converged
    ratio = res.nonlinear_path[0] / res.linear_path[0]
    assert abs(ratio - 1.0) < 0.02
    # whole path, not just impact
    scale = float(np.max(np.abs(res.linear_path)))
    assert float(np.max(np.abs(res.nonlinear_path - res.linear_path))) < 0.03 * scale


def test_backtracking_never_accepts_a_norm_increase(ss):
    res = solve_nonlinear_transition(ss, _rate_shock(500.0, 120), shock_var="r", horizon=120)
    hist = np.asarray(res.norm_history, dtype=float)
    assert hist.size >= 1
    assert np.all(np.diff(hist) <= 1e-12)


def test_fake_news_identity_and_genuine_jacobian(ss):
    fn = fake_news_algorithm(T=30, ss_model=ss)
    J, F = np.asarray(fn.jacobian), np.asarray(fn.fake_news)
    # J[t, s] = J[t-1, s-1] + F[t, s]
    np.testing.assert_allclose(J[1:, 1:], J[:-1, :-1] + F[1:, 1:], atol=1e-10)
    np.testing.assert_allclose(J[:, 0], F[:, 0], atol=1e-12)
    # a marginal-propensity-to-consume sized impact entry, not the old 0.22 heuristic
    assert 0.02 < J[0, 0] < 0.15


def test_targeted_transfer_uses_wealth_deciles_and_model_dynamics(ss):
    res_all = simulate_targeted_transfer(ss_model=ss, target="all", amount=1.0, T=20)
    table = res_all.decile_incidence            # DataFrame: Transfer / Consumption / Decile_MPC per decile
    inc = np.asarray(table["Transfer"], dtype=float)
    assert inc.size == 10
    assert np.all(inc > 0.0)
    np.testing.assert_allclose(inc.sum(), 1.0, atol=1e-6)
    # a universal transfer is spread evenly across mass-defined deciles
    assert float(np.max(np.abs(inc - 0.1))) < 0.02
    res_poor = simulate_targeted_transfer(ss_model=ss, target="borrowers", amount=1.0, T=20)
    assert res_poor.impact_mpc > res_all.impact_mpc
    # the removed `beta` argument is rejected rather than silently ignored
    with pytest.raises(TypeError):
        simulate_targeted_transfer(ss_model=ss, target="all", beta=0.9)
