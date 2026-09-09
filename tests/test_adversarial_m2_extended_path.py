"""Adversarial stress harness and empirical challenge suite for puremacro 2.9.0 M2 (Extended Path).

Empirically tests the Fair & Taylor (1983) Extended Path algorithm across 6 stress dimensions:
1. Non-linear RBC model under high shock volatility (sigma = 0.05, 0.10) and disaster shock.
2. Extreme and degenerate horizon values (T_H = 1, 2, 3, 50, 200) and terminal error decay.
3. Near-explosive persistence (rho = 0.999) under standard and forced-failure regimes (strict=True vs strict=False).
4. Zero-shock steady-state pinning across 50 periods on multiple distinct models (New Keynesian, RBC, Hansen 1985 Indivisible Labor) asserting max deviation <= 1e-12.
5. Linear model invariance against state-space simulation with non-zero initial states and correlated shocks, asserting max |y_EP - y_lin| <= 1e-10.
6. Input validation, dimension mismatch error guards, and boundary parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, extended_path, ExtendedPathResult


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def nonlinear_rbc_model():
    """Canonical non-linear Hansen/KPR Real Business Cycle model."""
    params = {
        "alpha": 0.33,
        "beta": 0.99,
        "delta": 0.025,
        "rho": 0.90,
    }
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_ss = (r_ss / params["alpha"]) ** (1.0 / (params["alpha"] - 1.0))
    y_ss = k_ss ** params["alpha"]
    c_ss = y_ss - params["delta"] * k_ss
    steady_state = {"c": c_ss, "k": k_ss, "y": y_ss, "a": 0.0}
    variables = ["c", "k", "y", "a"]
    shocks = ["e_a"]

    def nonlinear_rbc_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - np.exp(curr.a) * (lag.k ** p.alpha),
            curr.y - curr.c - curr.k + (1.0 - p.delta) * lag.k,
            1.0 / curr.c - p.beta * (1.0 / lead.c) * (
                p.alpha * np.exp(lead.a) * (curr.k ** (p.alpha - 1.0)) + 1.0 - p.delta
            ),
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    m = build_dynare(
        nonlinear_rbc_eqs,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )
    return {
        "model": m,
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


@pytest.fixture
def linear_nk_model():
    """Canonical 3-equation New Keynesian model with Taylor rule."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_a": 0.6,
    }
    variables = ["y", "pi", "r", "a"]
    shocks = ["e_d", "e_m"]
    steady_state = {v: 0.0 for v in variables}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma + curr.a),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
            curr.a - (p.rho_a * lag.a + shocks_v.e_d),
        ]

    m = build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    return {
        "model": m,
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


@pytest.fixture
def hansen_1985_model():
    """Hansen (1985) Real Business Cycle model with indivisible labor."""
    beta = 0.99
    delta = 0.025
    theta = 0.36
    rho = 0.95
    h_ss = 1.0 / 3.0

    r_ss = 1.0 / beta - (1.0 - delta)
    k_over_h = (theta / r_ss) ** (1.0 / (1.0 - theta))
    k_ss = h_ss * k_over_h
    y_ss = (k_ss ** theta) * (h_ss ** (1.0 - theta))
    c_ss = y_ss - delta * k_ss
    i_ss = delta * k_ss
    A_param = (1.0 - theta) * y_ss / (c_ss * h_ss)

    params = {
        "beta": beta,
        "delta": delta,
        "theta": theta,
        "rho": rho,
        "A": A_param,
    }
    steady_state = {
        "c": c_ss,
        "k": k_ss,
        "h": h_ss,
        "y": y_ss,
        "i": i_ss,
        "a": 0.0,
    }
    variables = ["c", "k", "h", "y", "i", "a"]
    shocks = ["e_a"]

    def hansen_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - np.exp(curr.a) * (lag.k ** p.theta) * (curr.h ** (1.0 - p.theta)),
            curr.y - curr.c - curr.i,
            curr.k - ((1.0 - p.delta) * lag.k + curr.i),
            1.0 / curr.c - p.beta * (1.0 / lead.c) * (p.theta * lead.y / curr.k + 1.0 - p.delta),
            p.A * curr.c - (1.0 - p.theta) * curr.y / curr.h,
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    m = build_dynare(
        hansen_eqs,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )
    return {
        "model": m,
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


@pytest.fixture
def linear_rbc_model():
    """Log-linearized Hansen/KPR Real Business Cycle model."""
    params = {
        "alpha": 0.33,
        "beta": 0.99,
        "delta": 0.025,
        "rho": 0.90,
    }
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_y_ratio = params["alpha"] / r_ss
    i_y_ratio = params["delta"] * k_y_ratio
    c_y_ratio = 1.0 - i_y_ratio
    variables = ["c", "k", "y", "a"]
    shocks = ["e_a"]
    steady_state = {"c": 0.0, "k": 0.0, "y": 0.0, "a": 0.0}

    def rbc_loglin_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (curr.a + p.alpha * lag.k),
            curr.y - (c_y_ratio * curr.c + i_y_ratio * (curr.k - (1.0 - p.delta) * lag.k) / p.delta),
            curr.c - (lead.c - (1.0 - p.beta * (1.0 - p.delta)) * (lead.y - curr.k)),
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    m = build_dynare(
        rbc_loglin_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )
    return {
        "model": m,
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


# ===========================================================================
# 1. High Shock Volatility & Disaster Stress Tests
# ===========================================================================

@pytest.mark.parametrize("sigma", [0.05, 0.10])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_adversarial_nonlinear_rbc_high_volatility(nonlinear_rbc_model, sigma, seed):
    """Stress-test non-linear RBC under 5x and 10x canonical shock volatility."""
    m = nonlinear_rbc_model["model"]
    res = extended_path(m, periods=100, horizon=100, sigma=sigma, seed=seed, strict=True)

    assert res.converged is True
    assert len(res.path) == 100
    assert len(res.shocks) == 100
    assert res.residual_norm < 1e-8

    # Ensure macroeconomic variables remain economically strictly positive
    assert (res.path["c"] > 0.0).all(), "Consumption violated positivity constraint"
    assert (res.path["k"] > 0.0).all(), "Capital violated positivity constraint"
    assert (res.path["y"] > 0.0).all(), "Output violated positivity constraint"

    # Solver iterations per period must remain efficient (<= 15 iterations)
    assert max(res.iterations) <= 15


def test_adversarial_nonlinear_rbc_disaster_shock(nonlinear_rbc_model):
    """Stress-test non-linear RBC with a -5 sigma disaster productivity drop."""
    m = nonlinear_rbc_model["model"]
    T = 60
    shocks = np.zeros((T, 1))
    # Plant a -5 sigma shock at t=5 (5 * 0.01 = -0.05 TFP plunge)
    shocks[5, 0] = -0.05

    res = extended_path(m, periods=T, horizon=100, shocks=shocks, strict=True)

    assert res.converged is True
    assert res.residual_norm < 1e-8
    # Consumption and capital must absorb shock without becoming negative or exploding
    assert res.path.loc[6, "y"] < res.path.loc[5, "y"]
    assert res.path["c"].min() > 0.5 * nonlinear_rbc_model["steady_state"]["c"]
    assert res.path["k"].min() > 0.5 * nonlinear_rbc_model["steady_state"]["k"]
    # Path must recover back toward steady state by T=60
    final_diff = abs(res.path.iloc[-1]["a"] - nonlinear_rbc_model["steady_state"]["a"])
    assert final_diff < 1e-2


# ===========================================================================
# 2. Extreme & Boundary Horizon Tests
# ===========================================================================

@pytest.mark.parametrize("horizon", [1, 2, 3, 200])
def test_adversarial_extreme_horizons(nonlinear_rbc_model, horizon):
    """Stress-test extreme horizon values: boundary cases T_H=1, 2, 3 and deep T_H=200."""
    m = nonlinear_rbc_model["model"]
    res = extended_path(m, periods=30, horizon=horizon, sigma=0.01, seed=42, strict=True)

    assert res.converged is True
    assert len(res.path) == 30
    assert res.residual_norm < 1e-8

    if horizon == 200:
        # Deep horizon must reach near-machine-precision terminal boundary
        assert res.terminal_error <= 1e-12


def test_adversarial_horizon_monotonic_terminal_decay(nonlinear_rbc_model):
    """Verify that expanding forward horizon suppresses policy truncation error against deep horizon."""
    m = nonlinear_rbc_model["model"]
    shocks = np.zeros((1, 1))
    shocks[0, 0] = 0.02  # Initial persistent shock

    # Deep horizon reference solution (H=300)
    res_ref = extended_path(m, periods=1, horizon=300, shocks=shocks, strict=True)
    y_ref = res_ref.path.iloc[0].to_numpy()

    horizons = [20, 50, 100, 200]
    truncation_errors = []

    for H in horizons:
        res = extended_path(m, periods=1, horizon=H, shocks=shocks, strict=True)
        y_H = res.path.iloc[0].to_numpy()
        truncation_errors.append(np.max(np.abs(y_H - y_ref)))

    # Truncation error must decrease monotonically as horizon expands
    for i in range(len(truncation_errors) - 1):
        assert truncation_errors[i] > truncation_errors[i + 1]

    # At H=200, truncation error must be below 1e-8
    assert truncation_errors[-1] <= 1e-8


# ===========================================================================
# 3. Near-Explosive Persistence & Error Guard Handling
# ===========================================================================

def test_adversarial_near_explosive_persistence_convergence():
    """Verify extended_path converges on near-explosive shock persistence (rho = 0.999)."""
    params = {"alpha": 0.33, "beta": 0.99, "delta": 0.025, "rho": 0.999}
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_ss = (r_ss / params["alpha"]) ** (1.0 / (params["alpha"] - 1.0))
    y_ss = k_ss ** params["alpha"]
    c_ss = y_ss - params["delta"] * k_ss
    steady_state = {"c": c_ss, "k": k_ss, "y": y_ss, "a": 0.0}
    variables = ["c", "k", "y", "a"]
    shocks = ["e_a"]

    def rbc_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - np.exp(curr.a) * (lag.k ** p.alpha),
            curr.y - curr.c - curr.k + (1.0 - p.delta) * lag.k,
            1.0 / curr.c - p.beta * (1.0 / lead.c) * (
                p.alpha * np.exp(lead.a) * (curr.k ** (p.alpha - 1.0)) + 1.0 - p.delta
            ),
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    m = build_dynare(
        rbc_eqs,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )

    res = extended_path(m, periods=25, horizon=150, sigma=0.01, seed=42, strict=True)
    assert res.converged is True
    assert res.residual_norm < 1e-8
    assert max(res.iterations) <= 10


def test_adversarial_solver_failure_strict_vs_nonstrict():
    """Verify graceful handling of non-convergence: strict=False marks flag; strict=True raises RuntimeError."""
    params = {"alpha": 0.33, "beta": 0.99, "delta": 0.025, "rho": 0.90}
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_ss = (r_ss / params["alpha"]) ** (1.0 / (params["alpha"] - 1.0))
    y_ss = k_ss ** params["alpha"]
    c_ss = y_ss - params["delta"] * k_ss
    steady_state = {"c": c_ss, "k": k_ss, "y": y_ss, "a": 0.0}

    def rbc_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - np.exp(curr.a) * (lag.k ** p.alpha),
            curr.y - curr.c - curr.k + (1.0 - p.delta) * lag.k,
            1.0 / curr.c - p.beta * (1.0 / lead.c) * (
                p.alpha * np.exp(lead.a) * (curr.k ** (p.alpha - 1.0)) + 1.0 - p.delta
            ),
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    m = build_dynare(
        rbc_eqs,
        variables=["c", "k", "y", "a"],
        shocks=["e_a"],
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )

    # Forcing max_iter=1 on non-linear RBC guarantees non-convergence
    res_nonstrict = extended_path(m, periods=5, horizon=50, sigma=0.01, seed=42, max_iter=1, strict=False)
    assert res_nonstrict.converged is False
    assert len(res_nonstrict.path) == 5

    # With strict=True, it must raise RuntimeError with informative diagnostics
    with pytest.raises(RuntimeError, match="Extended path solver failed to converge at period 1"):
        extended_path(m, periods=5, horizon=50, sigma=0.01, seed=42, max_iter=1, strict=True)


# ===========================================================================
# 4. Zero-Shock Pinnedness Across Multiple Distinct Models
# ===========================================================================

def test_adversarial_zero_shock_pinnedness_new_keynesian(linear_nk_model):
    """Verify zero-shock simulation stays pinned to steady state <= 1e-12 on New Keynesian model."""
    m = linear_nk_model["model"]
    ss = linear_nk_model["steady_state"]
    vars_list = linear_nk_model["variables"]

    res = extended_path(m, periods=50, horizon=50, sigma=0.0)
    assert res.converged is True

    max_dev = max(np.max(np.abs(res.path[v] - ss[v])) for v in vars_list)
    assert max_dev <= 1e-12, f"NK zero-shock max deviation {max_dev:.2e} exceeded 1e-12"


def test_adversarial_zero_shock_pinnedness_nonlinear_rbc(nonlinear_rbc_model):
    """Verify zero-shock simulation stays pinned to steady state <= 1e-12 on Non-linear RBC model."""
    m = nonlinear_rbc_model["model"]
    ss = nonlinear_rbc_model["steady_state"]
    vars_list = nonlinear_rbc_model["variables"]

    res = extended_path(m, periods=50, horizon=50, sigma=0.0)
    assert res.converged is True

    max_dev = max(np.max(np.abs(res.path[v] - ss[v])) for v in vars_list)
    assert max_dev <= 1e-12, f"RBC zero-shock max deviation {max_dev:.2e} exceeded 1e-12"


def test_adversarial_zero_shock_pinnedness_hansen_1985(hansen_1985_model):
    """Verify zero-shock simulation stays pinned to steady state <= 1e-12 on Hansen (1985) model."""
    m = hansen_1985_model["model"]
    ss = hansen_1985_model["steady_state"]
    vars_list = hansen_1985_model["variables"]

    res = extended_path(m, periods=50, horizon=50, sigma=0.0)
    assert res.converged is True

    max_dev = max(np.max(np.abs(res.path[v] - ss[v])) for v in vars_list)
    assert max_dev <= 1e-12, f"Hansen zero-shock max deviation {max_dev:.2e} exceeded 1e-12"


# ===========================================================================
# 5. Linear Invariance with Non-Zero Initial States & Correlated Shocks
# ===========================================================================

def test_adversarial_linear_nk_invariance_correlated_shocks_nonzero_state(linear_nk_model):
    """Verify linear NK extended path matches state-space simulation with correlated shocks and y_init."""
    m = linear_nk_model["model"]
    vars_list = linear_nk_model["variables"]
    T = 60

    # Non-zero initial state across all variables
    y_init = {"y": 0.025, "pi": -0.015, "r": 0.010, "a": 0.035}
    init_arr = np.array([y_init[v] for v in vars_list])

    # Correlated bivariate shock innovations
    rng = np.random.default_rng(2026)
    cov = np.array([[1.0, 0.7], [0.7, 1.2]]) * (0.01 ** 2)
    shocks_corr = rng.multivariate_normal([0.0, 0.0], cov, size=T)

    sim_lin = m.simulate(periods=T, shocks=shocks_corr, initial_state=init_arr, burn=0)
    res_ep = extended_path(m, periods=T, horizon=150, shocks=shocks_corr, y_init=y_init, strict=True)

    assert res_ep.converged is True
    diff = np.max(np.abs(res_ep.path.to_numpy() - sim_lin.to_numpy()))
    assert diff <= 1e-10, f"Linear NK invariance max diff {diff:.2e} exceeded 1e-10"


def test_adversarial_linear_rbc_invariance_persistent_state(linear_rbc_model):
    """Verify linear RBC extended path matches state-space simulation with persistent state and y_init."""
    m = linear_rbc_model["model"]
    vars_list = linear_rbc_model["variables"]
    T = 50

    # Non-zero initial state in capital accumulation
    y_init = {"c": 0.015, "k": 0.050, "y": 0.020, "a": 0.030}
    init_arr = np.array([y_init[v] for v in vars_list])

    rng = np.random.default_rng(777)
    shocks_in = rng.normal(0, 0.01, size=(T, 1))

    sim_lin = m.simulate(periods=T, shocks=shocks_in, initial_state=init_arr, burn=0)
    # Forward horizon T_H=350 suppresses capital persistence truncation error to machine precision
    res_ep = extended_path(m, periods=T, horizon=350, shocks=shocks_in, y_init=y_init, strict=True)

    assert res_ep.converged is True
    diff = np.max(np.abs(res_ep.path.to_numpy() - sim_lin.to_numpy()))
    assert diff <= 1e-10, f"Linear RBC invariance max diff {diff:.2e} exceeded 1e-10"


# ===========================================================================
# 6. Input Validation & Error Guard Coverage
# ===========================================================================

def test_adversarial_input_validation_guards(linear_nk_model):
    """Verify strict input validation guards on extended_path arguments."""
    m = linear_nk_model["model"]

    with pytest.raises(ValueError, match="periods must be at least 1"):
        extended_path(m, periods=0)

    with pytest.raises(ValueError, match="horizon must be at least 1"):
        extended_path(m, horizon=0)

    with pytest.raises(ValueError, match="burn must be non-negative"):
        extended_path(m, burn=-1)

    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        extended_path(m, max_iter=0)

    with pytest.raises(ValueError, match="tol must be a positive finite number"):
        extended_path(m, tol=-1.0)

    # Initial state dimension mismatch
    with pytest.raises(ValueError, match="y_init dimension"):
        extended_path(m, y_init=[0.0, 0.0])

    # Shocks column dimension mismatch
    with pytest.raises(ValueError, match="shocks column dimension"):
        extended_path(m, shocks=np.zeros((50, 5)))

    # Shocks row count insufficient
    with pytest.raises(ValueError, match="shocks row count"):
        extended_path(m, periods=50, shocks=np.zeros((20, 2)))
