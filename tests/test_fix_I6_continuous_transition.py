"""Regression tests for the AIYCONT follow-up in puremacro.vfi.continuous_transition.

Through 4.3.0, ``solve_continuous_transition``:

1. solved its internal terminal steady state with a hidden 500-iteration EGM
   cap and Brent xtol=1e-6, so at Z_term = 1 it did not reproduce the initial
   steady state (r* 0.03941949 against 0.03941451 in the n_z=3 docs economy,
   consumption policy 0.09 away);
2. forwarded a steady-state dict to ``solve_aiyagari_continuous(**dict)``, so
   a dict describing an already solved state (grids, distribution, prices)
   was replaced by a freshly solved default economy;
3. read beta, gamma, alpha and delta from its own keywords with defaults
   0.96, 2, 0.36, 0.08 instead of from the initial steady state;
4. built the discount-factor path of ``shock_var="beta"`` but never used it,
   so a discount-factor shock returned the steady state.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from puremacro.vfi.continuous_distribution import solve_aiyagari_continuous
from puremacro.vfi.continuous_transition import (
    _solve_terminal_steady_state,
    continuous_mit_shock,
    solve_continuous_transition,
)
from puremacro.vfi.discretize import markov_stationary

_DOCS_ECONOMY = dict(beta=0.96, gamma=2.0, alpha=0.36, delta=0.08, N_k=100, n_z=3, max_evals=20)


@pytest.fixture(scope="module")
def docs_ss():
    """The n_z=3 economy of docs/vfi_continuous_transition.md."""
    return solve_aiyagari_continuous(**_DOCS_ECONOMY)


@pytest.fixture(scope="module")
def patient_ss():
    """A steady state solved with beta=0.95, away from the old 0.96 default."""
    return solve_aiyagari_continuous(beta=0.95, N_k=100, n_z=3, max_evals=30)


# ---------------------------------------------------------------------------
# 1. Terminal steady state uses the solve_aiyagari_continuous controls
# ---------------------------------------------------------------------------

def _terminal(ss, Z, **controls):
    hh = ss.household_solution
    P, z = np.asarray(hh.P_z), np.asarray(hh.z_grid)
    L = float(np.sum(markov_stationary(P) * z))
    K_hist = np.asarray(hh.a_grid)
    a_dense = float(K_hist[-1]) * np.linspace(0.0, 1.0, len(hh.policy_c)) ** 1.5
    p = ss.metadata["params"]
    return _solve_terminal_steady_state(
        ss, Z, p["beta"], p["gamma"], p["alpha"], p["delta"], L, K_hist, a_dense, P, z, **controls
    )


def test_terminal_at_unit_tfp_reproduces_the_initial_steady_state(docs_ss):
    r, w, K, c, pdf, diag = _terminal(docs_ss, 1.0)
    # Pre-fix: r gap 4.98e-6 and max|c_term - c_init| = 0.090.
    assert diag["converged"]
    assert abs(r - docs_ss.r) < 1e-12
    assert abs(K - docs_ss.K) < 1e-9
    assert np.max(np.abs(c - docs_ss.household_solution.policy_c)) < 1e-12
    assert diag["egm_iterations"] > 500  # the old hidden cap
    assert diag["xtol"] == 1e-8 and diag["egm_tol"] == 1e-8 and diag["egm_max_iter"] == 10_000


def test_terminal_egm_cap_is_reported(docs_ss):
    _, _, _, _, _, diag = _terminal(docs_ss, 1.05, egm_max_iter=50)
    assert not diag["converged"]
    assert not diag["egm_converged"]
    assert diag["egm_cap_hits"] == diag["household_solves"]
    assert any("egm_max_iter=50" in reason for reason in diag["nonconvergence_reasons"])


def test_permanent_shock_terminal_controls_and_warning(docs_ss):
    res = continuous_mit_shock(docs_ss, shock_type="tfp", shock_size=0.05, persistence=1.0, horizon=30)
    diag = res.metadata["terminal_steady_state"]
    assert diag is not None and diag["converged"]
    # Inherited from the initial steady state's metadata.
    assert diag["xtol"] == docs_ss.metadata["xtol"]
    assert diag["egm_tol"] == docs_ss.metadata["egm_tol"]
    assert diag["Z"] == pytest.approx(1.05)

    with pytest.warns(RuntimeWarning, match="terminal steady state") as rec:
        bad = continuous_mit_shock(
            docs_ss, shock_type="tfp", shock_size=0.05, persistence=1.0, horizon=30, egm_max_iter=50
        )
    assert rec[0].filename == __file__
    assert not bad.converged
    assert bad.metadata["terminal_steady_state"]["egm_max_iter"] == 50


def test_invalid_terminal_controls_raise(docs_ss):
    with pytest.raises(ValueError, match="egm_max_iter"):
        solve_continuous_transition(docs_ss, horizon=5, egm_max_iter=2.5)
    with pytest.raises(ValueError, match="xtol"):
        solve_continuous_transition(docs_ss, horizon=5, xtol=0.0)


# ---------------------------------------------------------------------------
# 2. Steady-state dicts are configurations only
# ---------------------------------------------------------------------------

def test_solved_state_dict_raises_type_error(docs_ss):
    solved_like = {
        "k_grid": np.linspace(0.1, 15.0, 40),
        "pdf_ss": np.full((40, 2), 1.0 / 80),
        "r_ss": 0.035,
        "alpha": 0.36,
    }
    with pytest.raises(TypeError, match=r"'k_grid', 'pdf_ss', 'r_ss'.*AiyagariContinuousEquilibrium"):
        solve_continuous_transition(solved_like, horizon=5)
    with pytest.raises(TypeError, match="terminal_steady_state"):
        solve_continuous_transition(docs_ss, terminal_steady_state={"r_ss": 0.03}, horizon=5)


def test_config_dict_equals_passing_the_solved_equilibrium(docs_ss):
    shock = 0.02 * 0.8 ** np.arange(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_dict = solve_continuous_transition(dict(_DOCS_ECONOMY), shock_path=shock, horizon=20)
    from_ss = solve_continuous_transition(docs_ss, shock_path=shock, horizon=20)
    np.testing.assert_array_equal(from_dict.r_path, from_ss.r_path)
    np.testing.assert_array_equal(from_dict.K_s_path, from_ss.K_s_path)


# ---------------------------------------------------------------------------
# 3. Structural parameters come from the initial steady state
# ---------------------------------------------------------------------------

def test_parameters_are_read_from_the_steady_state(patient_ss):
    # Pre-fix, beta=0.96 dynamics made the beta=0.95 steady state non-stationary.
    res = solve_continuous_transition(patient_ss, shock_path=None, horizon=30)
    assert res.metadata["params"]["beta"] == 0.95
    assert res.converged and res.iterations == 0
    assert np.max(np.abs(res.r_path - patient_ss.r)) < 1e-12

    # Pre-fix, a TFP shock on this steady state drifted to the 0.96 economy's
    # rate (r_{T-1} 0.0401 against r* 0.0501).
    shock = continuous_mit_shock(patient_ss, shock_type="tfp", shock_size=0.02, persistence=0.8, horizon=40)
    assert shock.converged
    assert abs(shock.r_path[-1] - patient_ss.r) < 1e-3


def test_contradicting_structural_keyword_raises(patient_ss):
    with pytest.raises(ValueError, match="beta=0.96 contradicts"):
        solve_continuous_transition(patient_ss, horizon=5, beta=0.96)
    # The recorded value itself is accepted.
    res = solve_continuous_transition(patient_ss, horizon=5, beta=0.95)
    assert res.converged


def test_steady_state_without_recorded_params_uses_keywords_or_warns(docs_ss):
    bare = dataclasses.replace(docs_ss, metadata={})
    with pytest.warns(UserWarning, match="does not record beta, gamma, alpha, delta") as rec:
        solve_continuous_transition(bare, horizon=5)
    assert rec[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = solve_continuous_transition(bare, horizon=5, beta=0.96, gamma=2.0, alpha=0.36, delta=0.08)
    assert res.converged and res.iterations == 0


def test_unknown_keyword_emits_future_warning(docs_ss):
    with pytest.warns(FutureWarning, match="'egm_tolerance'") as rec:
        solve_continuous_transition(docs_ss, horizon=5, egm_tolerance=1e-6)
    assert rec[0].filename == __file__


# ---------------------------------------------------------------------------
# 4. Discount-factor shocks enter the Euler equation
# ---------------------------------------------------------------------------

def test_beta_shock_moves_the_economy(docs_ss):
    T = 40
    res = continuous_mit_shock(
        docs_ss, shock_type="beta", shock_size=0.01, persistence=0.8, horizon=T, max_iter=40
    )
    # Pre-fix: iterations == 0 and max|K_t - K*| = 6e-7.
    assert res.converged
    assert res.iterations > 0
    assert res.max_residual < 1e-4
    np.testing.assert_allclose(res.metadata["beta_path"], 0.96 * (1.0 + 0.01 * 0.8 ** np.arange(T)))
    # More patient households save more: capital rises and the rate falls.
    assert np.all(res.K_s_path[1:] > docs_ss.K + 1e-3)
    assert np.all(res.r_path[1:] < docs_ss.r)


def test_zero_beta_shock_is_the_steady_state(docs_ss):
    res = solve_continuous_transition(docs_ss, shock_path=np.zeros(20), shock_var="beta", horizon=20)
    assert res.converged and res.iterations == 0
    assert np.max(np.abs(res.r_path - docs_ss.r)) < 1e-12


def test_permanent_beta_shock_solves_the_terminal_steady_state(docs_ss):
    res = continuous_mit_shock(docs_ss, shock_type="beta", shock_size=0.005, persistence=1.0, horizon=30)
    diag = res.metadata["terminal_steady_state"]
    assert diag is not None and diag["converged"]
    assert diag["beta"] == pytest.approx(0.96 * 1.005)
    assert diag["Z"] == 1.0
    # A more patient economy has a lower steady-state rate.
    assert diag["r"] < docs_ss.r
