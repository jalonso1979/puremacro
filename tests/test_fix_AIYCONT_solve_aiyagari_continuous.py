"""Regression tests for AIYCONT: honest convergence in ``solve_aiyagari_continuous``.

The household problem is solved by the endogenous grid method (Carroll 2006),
iterating the Coleman operator T on the consumption policy until
``sup |c_n - c_(n-1)| < egm_tol``. Before the fix the loop stopped after a
fixed, hidden 500 iterations, ``solver`` and unknown keyword arguments were
ignored, and ``converged`` was the literal ``True``.

The oracle for the household fixed point is the fixed-point property itself:
a converged policy satisfies ``||T(c) - c||_inf -> 0``. ``_coleman_step`` below
is written independently of the library loop from the Euler equation
``u'(c) = beta (1 + r) E[u'(c')]`` and the budget ``a' = (1 + r) a + w z - c``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.vfi import AiyagariContinuousModel, solve_aiyagari_continuous, tauchen
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    continuous_stationary_distribution,
    continuous_stationary_equilibrium,
)
from puremacro.vfi.discretize import markov_stationary

# The worked example of docs/vfi_continuous_equilibrium.md, section 4.
DOCS_EXAMPLE = dict(beta=0.96, gamma=2.0, alpha=0.36, delta=0.08, N_k=150, n_z=3, max_evals=25)
# A slow contraction: beta (1 + r) is close to one near r* = 1/beta - 1.
SLOW = dict(beta=0.995, gamma=1.0, rho_z=0.9, sigma_z=0.1, a_max=400.0, n_a=1200, N_k=600)
SMALL = dict(n_z=5, a_max=30.0, N_k=150)


def _prices(r, alpha, delta):
    kl = ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))
    return kl, (1.0 - alpha) * kl**alpha


def _coleman_step(c, r, w, beta, gamma, a, z, P):
    """One application of the Coleman operator T on c (n_a x n_z), borrowing limit 0."""
    marg = beta * (1.0 + r) * (P @ (c ** (-gamma)).T).T  # E[u'(c(a', z')) | z]
    c_today = marg ** (-1.0 / gamma)
    out = np.empty_like(c)
    for m in range(len(z)):
        a_today = (c_today[:, m] + a - w * z[m]) / (1.0 + r)
        cm = np.interp(a, a_today, c_today[:, m])
        constrained = a < a_today[0]
        cm[constrained] = (1.0 + r) * a[constrained] + w * z[m]
        out[:, m] = cm
    return out


def _grids(cfg):
    beta = cfg.get("beta", 0.96)
    n_z = cfg.get("n_z", 5)
    log_z, P = tauchen(n_z, cfg.get("rho_z", 0.9), cfg.get("sigma_z", 0.2))
    a = cfg.get("a_max", 30.0) * np.linspace(0.0, 1.0, cfg.get("n_a", 150)) ** 1.5
    return beta, np.exp(log_z), P, a


def _true_excess(r, cfg, tol=1e-11, max_iter=400_000):
    """K^s - K^d at r with the household fixed point iterated to ``tol``."""
    beta, z, P, a = _grids(cfg)
    gamma, alpha, delta = cfg.get("gamma", 2.0), cfg.get("alpha", 0.36), cfg.get("delta", 0.08)
    a_max, N_k = cfg.get("a_max", 30.0), cfg.get("N_k", 1000)
    kl, w = _prices(r, alpha, delta)
    c = r * a[:, None] + w * z[None, :]
    for _ in range(max_iter):
        c_next = _coleman_step(c, r, w, beta, gamma, a, z, P)
        step = np.max(np.abs(c_next - c))
        c = c_next
        if step < tol:
            break
    k_hist = np.linspace(0.0, a_max, N_k)
    a_next = np.maximum((1.0 + r) * a[:, None] + w * z[None, :] - c, 0.0)
    policy = np.column_stack([np.interp(k_hist, a, a_next[:, m]) for m in range(len(z))])
    dist = continuous_stationary_distribution(policy, k_hist, shock_transition=P, shock_grid=z)
    L = float(np.sum(markov_stationary(P) * z))
    return float(dist.mean()) - L * kl


def _fixed_point_residual(eq, cfg):
    beta, z, P, a = _grids(cfg)
    hh = eq.household_solution
    c = hh.policy_c
    Tc = _coleman_step(c, eq.r, hh.w, beta, cfg.get("gamma", 2.0), a, z, P)
    return float(np.max(np.abs(Tc - c)))


# ---------------------------------------------------------------------------
# 1. The EGM loop has an exposed cap and tolerance and reports hitting the cap
# ---------------------------------------------------------------------------


def test_slow_contraction_household_policy_is_a_fixed_point():
    """beta = 0.995: the old 500-iteration cap left ||T(c) - c|| = 8.8e-4 at r*."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eq = solve_aiyagari_continuous(**SLOW)
    assert eq.converged
    assert eq.metadata["egm_converged"]
    assert 500 < eq.metadata["egm_iterations"] < eq.metadata["egm_max_iter"]
    assert eq.metadata["egm_residual"] < 1e-8
    assert _fixed_point_residual(eq, SLOW) < 1e-8
    # The market clears with the converged household policy, not only with the
    # policy the solver returned.
    assert abs(_true_excess(eq.r, SLOW)) < 1e-4


def test_old_cap_is_reported_not_hidden():
    """egm_max_iter=500 reproduces the old stopping point and says so."""
    with pytest.warns(RuntimeWarning, match="egm_max_iter=500"):
        eq = solve_aiyagari_continuous(egm_max_iter=500, **SLOW)
    assert not eq.converged
    assert not eq.metadata["egm_converged"]
    assert eq.metadata["egm_iterations"] == 500
    assert eq.metadata["egm_residual"] > 1e-8
    assert _fixed_point_residual(eq, SLOW) > 1e-6


def test_docs_example_converges_honestly():
    """The docs example: the old cap bound at r*, true excess 1.6e-2 against a reported -2.5e-6."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eq = solve_aiyagari_continuous(**DOCS_EXAMPLE)
    assert eq.converged
    assert eq.metadata["egm_converged"]
    assert eq.metadata["dist_converged"]
    assert abs(eq.capital_market_clearing_error) < 1e-4
    assert _fixed_point_residual(eq, DOCS_EXAMPLE) < 1e-8
    assert abs(_true_excess(eq.r, DOCS_EXAMPLE)) < 1e-4


def test_egm_tol_is_honoured():
    loose = solve_aiyagari_continuous(egm_tol=1e-4, **SMALL)
    tight = solve_aiyagari_continuous(egm_tol=1e-10, **SMALL)
    assert loose.metadata["egm_tol"] == 1e-4 and tight.metadata["egm_tol"] == 1e-10
    assert loose.metadata["egm_iterations"] < tight.metadata["egm_iterations"]
    assert loose.metadata["egm_residual"] < 1e-4
    assert tight.metadata["egm_residual"] < 1e-10


@pytest.mark.parametrize(
    "bad",
    [
        dict(egm_tol=0.0),
        dict(egm_tol=-1e-8),
        dict(egm_tol=None),
        dict(egm_max_iter=0),
        dict(egm_max_iter=2.5),
        dict(egm_max_iter=float("inf")),
        dict(egm_max_iter=float("nan")),
        dict(egm_max_iter=None),
        dict(tol_ge=0.0),
        dict(tol_ge=float("nan")),
    ],
)
def test_invalid_tolerances_raise(bad):
    with pytest.raises(ValueError):
        solve_aiyagari_continuous(**SMALL, **bad)


# ---------------------------------------------------------------------------
# 2. converged combines the household, the distribution and market clearing
# ---------------------------------------------------------------------------


def test_unconverged_distribution_is_not_converged():
    with pytest.warns(RuntimeWarning, match="distribution"):
        eq = solve_aiyagari_continuous(
            n_z=3, a_max=30.0, N_k=300, dist_options={"method": "power", "max_iter": 20}
        )
    assert not eq.distribution.converged
    assert not eq.metadata["dist_converged"]
    assert not eq.converged


def test_uncleared_market_is_not_converged():
    """xtol=5e-3 leaves |K^s - K^d| far above the documented 1e-4."""
    with pytest.warns(RuntimeWarning, match="tol_ge"):
        eq = solve_aiyagari_continuous(xtol=5e-3, **SMALL)
    assert abs(eq.capital_market_clearing_error) > 1e-4
    assert not eq.metadata["clearing_ok"]
    assert not eq.converged
    # The same solve judged against a looser clearing tolerance.
    loose = solve_aiyagari_continuous(xtol=5e-3, tol_ge=1e3, **SMALL)
    assert loose.metadata["clearing_ok"]
    assert loose.converged == (loose.metadata["egm_converged"] and loose.metadata["dist_converged"])


def test_summary_reports_the_honest_flag():
    with pytest.warns(RuntimeWarning):
        eq = solve_aiyagari_continuous(xtol=5e-3, **SMALL)
    assert eq.summary().loc["Converged", "Value"] == "False"


# ---------------------------------------------------------------------------
# 3. solver and **kwargs are no longer silently ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["collocation", "fem", "no_such_solver"])
def test_unimplemented_solver_raises(name):
    with pytest.raises(ValueError, match="endogenous grid"):
        solve_aiyagari_continuous(solver=name, **SMALL)
    with pytest.raises(ValueError, match="endogenous grid"):
        AiyagariContinuousModel(n_z=5).solve(solver=name, N_k=150)


def test_egm_solver_names_are_equivalent():
    base = solve_aiyagari_continuous(**SMALL)
    for name in ("EGM", " auto ", None):
        eq = solve_aiyagari_continuous(solver=name, **SMALL)
        assert eq.r == base.r
        assert eq.metadata["solver"] == "egm"


def test_unknown_keyword_warns_and_names_it():
    base = solve_aiyagari_continuous(**SMALL)
    with pytest.warns(FutureWarning, match=r"'tol'.*ignored"):
        eq = solve_aiyagari_continuous(tol=1e-4, **SMALL)
    assert eq.r == base.r


def test_egm_options_pass_through_model_wrapper():
    with pytest.warns(RuntimeWarning, match="egm_max_iter=3") as rec:
        eq = AiyagariContinuousModel(n_z=5).solve(N_k=150, egm_max_iter=3)
    assert eq.metadata["egm_max_iter"] == 3
    assert not eq.converged
    # The warning is attributed to the caller, not to the wrapper inside puremacro.
    assert [w.filename for w in rec if w.category is RuntimeWarning] == [__file__]


def test_unknown_keyword_warning_points_at_the_caller():
    with pytest.warns(FutureWarning, match="'tol'") as rec:
        AiyagariContinuousModel(n_z=5).solve(N_k=150, tol=1e-4)
    assert [w.filename for w in rec if w.category is FutureWarning] == [__file__]


# ---------------------------------------------------------------------------
# 3b. The wrappers' default xtol meets the default clearing tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: AiyagariContinuousModel(n_z=3).solve(N_k=150),
        lambda: AiyagariContinuousEquilibrium.solve(n_z=3, N_k=150),
    ],
    ids=["AiyagariContinuousModel.solve", "AiyagariContinuousEquilibrium.solve"],
)
def test_wrapper_defaults_clear_the_market(call):
    """n_z=3: the old wrapper default xtol=1e-6 left |K^s - K^d| = 5.8e-4 > tol_ge = 1e-4."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eq = call()
    assert eq.metadata["xtol"] == 1e-8
    assert eq.converged
    assert eq.metadata["clearing_ok"]
    assert abs(eq.capital_market_clearing_error) < 1e-4
    cfg = dict(n_z=3, N_k=150)
    assert abs(_true_excess(eq.r, cfg)) < 1e-4


def test_old_wrapper_xtol_is_reported_as_not_cleared():
    with pytest.warns(RuntimeWarning, match="tighten xtol"):
        eq = AiyagariContinuousModel(n_z=3).solve(N_k=150, xtol=1e-6)
    assert abs(eq.capital_market_clearing_error) > 1e-4
    assert not eq.converged


# ---------------------------------------------------------------------------
# 4. The generic wrapper reports an unconverged distribution too
# ---------------------------------------------------------------------------


def test_continuous_stationary_equilibrium_reports_distribution_failure():
    k_hist = np.linspace(0.0, 20.0, 300)

    def build(r):
        return lambda k: (1.0 + r) * 0.5 * k + 1.0

    def residual(r, sol, dist, prob):
        return dist.mean() - 3.0 / (1.0 + r)

    ok = continuous_stationary_equilibrium(build, residual, (0.01, 0.50), asset_grid=k_hist)
    assert ok.converged
    with pytest.warns(RuntimeWarning, match="distribution"):
        bad = continuous_stationary_equilibrium(
            build, residual, (0.01, 0.50), asset_grid=k_hist,
            dist_options={"method": "power", "max_iter": 10},
        )
    assert not bad.distribution.converged
    assert not bad.converged
