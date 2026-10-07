"""Regression tests for the MITSHOCK fix in puremacro.vfi.continuous_mit_shock.

Through 4.3.0, ``continuous_mit_shock``:

1. treated a transitory shock (persistence < 1) as permanent whenever
   Z_(T-1) (or beta_(T-1)/beta) differed from 1 by more than about 1e-5, and
   solved a terminal steady state at Z_(T-1). For T=40 and shock_size=0.05
   the model switched at persistence ~0.806; at persistence 0.92 the economy
   converged to a steady state with 0.19% higher TFP. No warning was issued.
2. passed the relative path to ``solve_continuous_transition``, which reads
   any TFP or discount-factor path with an entry above 0.5 as levels, so
   ``shock_size=0.6`` gave Z_0 = 0.6 instead of 1.6.

A transitory shock now always keeps the initial steady state as its terminal
condition and warns when it has not died out by T-1; a permanent shock needs
persistence=1.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.vfi.continuous_distribution import solve_aiyagari_continuous
from puremacro.vfi.continuous_transition import (
    _mit_horizon_needed,
    continuous_mit_shock,
    solve_continuous_transition,
)

_DOCS_ECONOMY = dict(beta=0.96, gamma=2.0, alpha=0.36, delta=0.08, N_k=100, n_z=3, max_evals=20)
T, S = 40, 0.05
# The old switch: |Z_(T-1) - 1| > 1e-6 + 1e-5 (numpy.isclose), i.e. rho > 0.80578 here.
OLD_THRESHOLD = (1.1e-5 / S) ** (1.0 / (T - 1))


@pytest.fixture(scope="module")
def ss():
    """The n_z=3 economy of docs/vfi_continuous_transition.md."""
    return solve_aiyagari_continuous(**_DOCS_ECONOMY)


def _quiet(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return continuous_mit_shock(**kw)


# ---------------------------------------------------------------------------
# 1. Continuity in the persistence across the old threshold
# ---------------------------------------------------------------------------

def test_irf_is_continuous_in_persistence_across_the_old_threshold(ss):
    rhos = [0.8050, 0.8055, 0.8060, 0.8065]
    assert rhos[1] < OLD_THRESHOLD < rhos[2]
    res = [
        continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=r, horizon=T, tol=1e-9, max_iter=200)
        for r in rhos
    ]
    for r in res:
        assert r.converged
        assert r.metadata["mit_shock"]["terminal_condition"] == "initial_steady_state"
        assert r.metadata["terminal_steady_state"] is None  # pre-fix: solved at Z_(T-1) for rho >= 0.806
    # Equal steps in rho: the step that crosses the old threshold must match its
    # neighbours up to curvature. Pre-fix the crossing step of r_39 was 4.6e-7
    # against 2.1e-6 on either side (ratio ~0.3 of max|d1|); post-fix 0.004.
    for name in ("r_path", "K_s_path"):
        p = [getattr(r, name) for r in res]
        d = [p[i + 1] - p[i] for i in range(3)]
        scale = float(np.max(np.abs(d[0])))
        assert scale > 0.0
        assert np.max(np.abs(d[1] - d[0])) < 0.05 * scale, name
        assert np.max(np.abs(d[2] - d[1])) < 0.05 * scale, name


def test_below_the_old_threshold_nothing_changes(ss):
    # continuous_mit_shock is bit-identical to the direct call, which keeps the
    # initial steady state for a shock that has died out.
    a = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T)
    b = solve_continuous_transition(ss, shock_path=S * 0.8 ** np.arange(T, dtype=np.float64), horizon=T)
    assert b.metadata["terminal_steady_state"] is None
    np.testing.assert_array_equal(a.r_path, b.r_path)
    np.testing.assert_array_equal(a.K_s_path, b.K_s_path)
    assert a.iterations == b.iterations


# ---------------------------------------------------------------------------
# 2. Transitory shocks return to the initial steady state
# ---------------------------------------------------------------------------

def test_transitory_shock_keeps_the_initial_steady_state_and_warns(ss):
    with pytest.warns(RuntimeWarning, match="has not died out") as rec:
        res = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.92, horizon=T, tol=1e-9, max_iter=200)
    assert rec[0].filename == __file__
    assert "horizon=84" in str(rec[0].message)
    assert res.converged
    assert res.metadata["terminal_steady_state"] is None  # pre-fix: a steady state at Z = 1.0019
    mit = res.metadata["mit_shock"]
    assert mit["terminal_condition"] == "initial_steady_state"
    assert not mit["permanent"] and mit["truncated"]
    assert mit["shock_at_last_date"] == pytest.approx(S * 0.92 ** (T - 1))
    assert mit["remaining_share"] == pytest.approx(0.92 ** (T - 1))
    assert mit["horizon_needed"] == 84
    assert mit["capital_gap_last"] == pytest.approx(res.K_s_path[-1] - ss.K)

    # The same path solved against the initial steady state, stated explicitly.
    path = 1.0 + S * 0.92 ** np.arange(T)
    ref = solve_continuous_transition(ss, terminal_steady_state=ss, shock_path=path, horizon=T, tol=1e-9, max_iter=200)
    np.testing.assert_allclose(res.K_s_path, ref.K_s_path, atol=1e-8)
    np.testing.assert_allclose(res.r_path, ref.r_path, atol=1e-9)
    # What the old code solved instead: a permanent TFP gain of 0.19%.
    old = solve_continuous_transition(ss, shock_path=path, horizon=T, tol=1e-9, max_iter=200)
    assert old.metadata["terminal_steady_state"]["Z"] == pytest.approx(1.0 + S * 0.92 ** (T - 1))
    assert np.max(np.abs(res.K_s_path - old.K_s_path)) > 1e-2


def test_long_horizon_transitory_shock_returns_without_warning(ss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.92, horizon=150)
    mit = res.metadata["mit_shock"]
    assert res.converged and not mit["truncated"] and mit["terminal_condition"] == "initial_steady_state"
    assert abs(res.K_s_path[-1] - ss.K) < 2e-3
    assert abs(res.r_path[-1] - ss.r) < 1e-4


def test_nearly_permanent_shock_is_still_transitory(ss):
    near = _quiet(steady_state=ss, shock_type="tfp", shock_size=S, persistence=0.999, horizon=30, max_iter=60)
    assert near.metadata["mit_shock"]["terminal_condition"] == "initial_steady_state"
    assert near.metadata["terminal_steady_state"] is None
    perm = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=1.0, horizon=30)
    assert perm.metadata["mit_shock"]["terminal_condition"] == "solved_steady_state"
    assert perm.metadata["mit_shock"]["permanent"] and not perm.metadata["mit_shock"]["truncated"]
    assert perm.metadata["terminal_steady_state"]["Z"] == pytest.approx(1.0 + S)


def test_transitory_beta_shock_keeps_the_initial_steady_state(ss):
    # beta_(T-1)/beta - 1 = 0.01 * 0.9**29 = 4.7e-4: pre-fix a steady state at that beta was solved.
    with pytest.warns(RuntimeWarning, match="discount|beta") as rec:
        res = continuous_mit_shock(ss, shock_type="beta", shock_size=0.01, persistence=0.9, horizon=30, max_iter=40)
    assert any("has not died out" in str(w.message) for w in rec)
    assert res.metadata["terminal_steady_state"] is None
    assert res.metadata["mit_shock"]["terminal_condition"] == "initial_steady_state"
    np.testing.assert_allclose(res.metadata["beta_path"], 0.96 * (1.0 + 0.01 * 0.9 ** np.arange(30)))


def test_user_terminal_steady_state_is_used_as_given(ss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=20, terminal_steady_state=ss)
    assert res.metadata["mit_shock"]["terminal_condition"] == "user"
    assert res.metadata["mit_shock"]["truncated"]  # reported, but the caller chose the terminal state


# ---------------------------------------------------------------------------
# 3. Truncation diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.8, 0.9, 0.92, 0.95, 0.99])
@pytest.mark.parametrize("tol", [1e-6, 1e-3, 0.05])
def test_horizon_needed_is_the_smallest_sufficient_horizon(rho, tol):
    h = _mit_horizon_needed(rho, tol)
    assert rho ** (h - 1) <= tol
    assert h == 1 or rho ** (h - 2) > tol


def test_horizon_needed_edge_cases():
    assert _mit_horizon_needed(0.8, 1e-3) == 32
    assert _mit_horizon_needed(0.0, 1e-3) == 2
    assert _mit_horizon_needed(0.5, 1.0) == 1


def test_truncation_tol_controls_the_warning(ss):
    with pytest.warns(RuntimeWarning, match="horizon=32"):
        continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loose = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=20, truncation_tol=0.05)
        enough = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=32)
        zero = continuous_mit_shock(ss, shock_type="tfp", shock_size=0.0, persistence=0.99, horizon=10)
    assert not loose.metadata["mit_shock"]["truncated"]
    assert not enough.metadata["mit_shock"]["truncated"]
    assert not zero.metadata["mit_shock"]["truncated"] and zero.iterations == 0
    with pytest.raises(ValueError, match="truncation_tol"):
        continuous_mit_shock(ss, horizon=10, truncation_tol=0.0)


def test_permanent_rate_wedge_warns(ss):
    with pytest.warns(RuntimeWarning, match="permanent rate wedge"):
        res = continuous_mit_shock(ss, shock_type="rate", shock_size=0.005, persistence=1.0, horizon=20)
    mit = res.metadata["mit_shock"]
    assert mit["permanent"] and mit["truncated"] and mit["terminal_condition"] == "initial_steady_state"


# ---------------------------------------------------------------------------
# 4. shock_size is always a deviation
# ---------------------------------------------------------------------------

def test_large_tfp_shock_is_not_read_as_a_level(ss):
    up = _quiet(steady_state=ss, shock_type="tfp", shock_size=0.6, persistence=0.8, horizon=10, max_iter=0)
    np.testing.assert_allclose(up.metadata["Z_path"][:3], [1.6, 1.48, 1.384])  # pre-fix 0.6, 0.48, 0.384
    down = _quiet(steady_state=ss, shock_type="tfp", shock_size=-0.6, persistence=1.0, horizon=10, max_iter=0)
    np.testing.assert_allclose(down.metadata["Z_path"], 0.4)
    beta = _quiet(steady_state=ss, shock_type="beta", shock_size=0.6, persistence=0.8, horizon=10, max_iter=0)
    np.testing.assert_allclose(beta.metadata["beta_path"][:2], [0.96 * 1.6, 0.96 * 1.48])  # pre-fix 0.6, 0.48
    with pytest.raises(ValueError, match="above -1"):
        continuous_mit_shock(ss, shock_type="tfp", shock_size=-1.0, horizon=10)
    with pytest.raises(ValueError, match="Invalid shock_type"):
        continuous_mit_shock(ss, shock_type="money", horizon=10)


def test_dict_steady_state_is_resolved(ss):
    res = continuous_mit_shock(dict(_DOCS_ECONOMY), shock_type="tfp", shock_size=S, persistence=0.8, horizon=T)
    ref = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T)
    np.testing.assert_array_equal(res.r_path, ref.r_path)
    with pytest.raises(TypeError, match="steady_state"):
        continuous_mit_shock({"r_ss": 0.03}, horizon=5)


# ---------------------------------------------------------------------------
# 5. damping (documented: Broyden uses it only in the line-search fallback)
# ---------------------------------------------------------------------------

def test_damping_has_no_effect_on_broyden_with_backtracking(ss):
    a = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T, damping=0.2)
    b = continuous_mit_shock(ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T, damping=0.8)
    np.testing.assert_array_equal(a.r_path, b.r_path)
    assert a.iterations == b.iterations
    # Shooting uses it at every step.
    s2 = _quiet(steady_state=ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T, solver="shooting", damping=0.2, max_iter=60)
    s8 = _quiet(steady_state=ss, shock_type="tfp", shock_size=S, persistence=0.8, horizon=T, solver="shooting", damping=0.8, max_iter=60)
    assert s2.converged and not s8.converged
