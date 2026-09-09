"""Empirical adversarial stress test suite for puremacro 2.9.0 Tier 3 Milestone 2.

Authored by m2_challenger_2 to empirically challenge:
1. Warm-start vs cold-start trajectory equivalence (absence of numerical hysteresis or history contamination).
2. Input validation and error guard robustness:
   - Shape mismatches: (periods, n_shocks + 1), (periods - 5, n_shocks), 3D arrays
   - Non-positive parameters: periods <= 0, horizon <= 0, burn < 0, max_iter < 1, tol <= 0 / nan / inf
   - max_iter = 1 behavior on linear (exact 1-step solve) vs non-linear (convergence failure & strict guard)
3. Presentation methods of ExtendedPathResult under extreme boundary edge cases:
   - Single period T=1
   - Single endogenous variable n_v=1
   - Compound boundary T=1 and n_v=1
   - Custom Matplotlib axes injection
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, extended_path, ExtendedPathResult
from puremacro.dsge.perfect_foresight import solve_perfect_foresight


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rbc_nonlinear_model():
    """Non-linear Hansen RBC model for empirical stress testing."""
    params = {"alpha": 0.33, "beta": 0.99, "delta": 0.025, "rho": 0.90}
    r_ss = 1.0 / params["beta"] - (1.0 - params["delta"])
    k_ss = (r_ss / params["alpha"]) ** (1.0 / (params["alpha"] - 1.0))
    y_ss = k_ss ** params["alpha"]
    c_ss = y_ss - params["delta"] * k_ss
    steady_state = {"c": c_ss, "k": k_ss, "y": y_ss, "a": 0.0}

    def nonlinear_rbc_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - np.exp(curr.a) * (lag.k ** p.alpha),
            curr.y - curr.c - curr.k + (1.0 - p.delta) * lag.k,
            1.0 / curr.c - p.beta * (1.0 / lead.c) * (
                p.alpha * np.exp(lead.a) * (curr.k ** (p.alpha - 1.0)) + 1.0 - p.delta
            ),
            curr.a - (p.rho * lag.a + shocks_v.e_a),
        ]

    return build_dynare(
        nonlinear_rbc_eqs,
        variables=["c", "k", "y", "a"],
        shocks=["e_a"],
        params=params,
        steady_state=steady_state,
        check_steady_state=True,
        strict=False,
    )


@pytest.fixture
def nk_linear_model():
    """3-equation linear New Keynesian model with Taylor rule."""
    params = {
        "beta": 0.99, "sigma": 1.0, "kappa": 0.15,
        "phi_pi": 1.5, "phi_y": 0.25, "rho_r": 0.7, "rho_a": 0.6,
    }
    def nk_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma + curr.a),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
            curr.a - (p.rho_a * lag.a + shocks_v.e_d),
        ]

    return build_dynare(
        nk_eqs,
        variables=["y", "pi", "r", "a"],
        shocks=["e_d", "e_m"],
        params=params,
        steady_state={"y": 0.0, "pi": 0.0, "r": 0.0, "a": 0.0},
        check_steady_state=False,
        strict=False,
    )


# ============================================================================
# 1. Warm-Start vs Cold-Start Equivalence & Hysteresis Stress
# ============================================================================

def test_warm_start_vs_cold_start_trajectory_equivalence(rbc_nonlinear_model):
    """Verify warm-start from shifted trajectory reaches identical solution to cold-start.

    Ensures zero numerical hysteresis or path contamination.
    """
    m = rbc_nonlinear_model
    T = 60
    H = 40
    rng = np.random.default_rng(2026)
    shocks = rng.normal(0, 0.01, (T, 1))

    # Standard warm-start extended path
    res_warm = extended_path(m, periods=T, horizon=H, shocks=shocks, tol=1e-8)

    # Reference cold-start engine (resets initial guess to steady state at every t)
    v_names = tuple(m.variables)
    s_names = tuple(m.shocks)
    y_ss_arr = np.array([float(m.steady_state.get(v, 0.0)) for v in v_names], dtype=float)
    dyn_eqs = m._dynare_equations
    p_dict = dict(m._params or {})
    from puremacro.dsge.build import _Vec
    p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), what="parameter")
    v_list = list(v_names)
    s_list = list(s_names)

    def eq_fn(yp, yc, yl, eps):
        yp_v = _Vec(v_list, yp, what="variable")
        yc_v = _Vec(v_list, yc, what="variable")
        yl_v = _Vec(v_list, yl, what="variable")
        eps_v = _Vec(s_list, np.atleast_1d(eps), what="shock")
        return dyn_eqs(yp_v, yc_v, yl_v, eps_v, p_vec)

    path_cold = np.zeros((T, len(v_names)), dtype=float)
    current_state = y_ss_arr.copy()
    cold_iters = []

    for t in range(T):
        u_t = shocks[t]
        forward_exo = np.zeros((H, len(s_names)), dtype=float)
        forward_exo[0] = u_t
        cold_guess = np.tile(y_ss_arr, (H, 1))

        res_t = solve_perfect_foresight(
            model_or_equations=None,
            y_init=current_state,
            y_ss=y_ss_arr,
            exogenous_path=forward_exo,
            periods=H,
            initial_path=cold_guess,
            tol=1e-8,
            max_iter=50,
            variable_names=v_names,
            equations_fn=eq_fn,
        )
        y_t = res_t.path.iloc[0].to_numpy()
        path_cold[t] = y_t
        current_state = y_t.copy()
        cold_iters.append(int(res_t.iterations))

    # Maximum deviation must be bounded by solver tolerance
    diff = np.max(np.abs(res_warm.path.to_numpy() - path_cold))
    assert diff <= 1e-8, f"Warm vs cold start deviated by {diff:.4e} > 1e-8"

    # Verify warm-start is at least as fast or faster than cold-start
    assert sum(res_warm.iterations) <= sum(cold_iters)


def test_impulse_recovery_zero_hysteresis(rbc_nonlinear_model):
    """Verify system returns to steady state after large impulse without hysteresis."""
    m = rbc_nonlinear_model
    T = 50
    H = 40
    shocks = np.zeros((T, 1))
    shocks[0, 0] = 0.05  # Large 5% shock at t=0

    res = extended_path(m, periods=T, horizon=H, shocks=shocks, tol=1e-10)
    ss_arr = np.array([float(m.steady_state[v]) for v in m.variables])

    # In RBC, capital accumulates after TFP shock, peaking around t=14, then decays monotonically
    dist_peak = np.linalg.norm(res.path.iloc[14].to_numpy() - ss_arr)
    dist_t25 = np.linalg.norm(res.path.iloc[25].to_numpy() - ss_arr)
    dist_t45 = np.linalg.norm(res.path.iloc[45].to_numpy() - ss_arr)

    assert dist_t25 < dist_peak
    assert dist_t45 < dist_t25
    assert res.converged is True
    assert res.terminal_error < 1e-5


# ============================================================================
# 2. Input Validation and Error Handling
# ============================================================================

def test_shock_shape_mismatches_raise_value_error(nk_linear_model):
    """Verify shock matrix with invalid row or column dimensions raises ValueError."""
    m = nk_linear_model
    # Extra columns: (20, 3) when model has 2 shocks
    with pytest.raises(ValueError, match="shocks column dimension"):
        extended_path(m, periods=20, shocks=np.zeros((20, 3)))

    # Fewer rows: (15, 2) when periods=20
    with pytest.raises(ValueError, match="shocks row count .* is less than periods"):
        extended_path(m, periods=20, shocks=np.zeros((15, 2)))

    # 3D array
    with pytest.raises((ValueError, IndexError)):
        extended_path(m, periods=20, shocks=np.zeros((20, 2, 1)))


@pytest.mark.parametrize("param_override,err_match", [
    ({"periods": 0}, "periods must be at least 1"),
    ({"periods": -5}, "periods must be at least 1"),
    ({"horizon": 0}, "horizon must be at least 1"),
    ({"horizon": -10}, "horizon must be at least 1"),
    ({"burn": -1}, "burn must be non-negative"),
    ({"max_iter": 0}, "max_iter must be an integer >= 1"),
    ({"max_iter": -3}, "max_iter must be an integer >= 1"),
    ({"tol": 0.0}, "tol must be a positive finite number"),
    ({"tol": -1.0}, "tol must be a positive finite number"),
    ({"tol": float("nan")}, "tol must be a positive finite number"),
    ({"tol": float("inf")}, "tol must be a positive finite number"),
])
def test_boundary_parameter_validation_guards(nk_linear_model, param_override, err_match):
    """Verify all parameter boundaries reject invalid inputs with ValueError."""
    with pytest.raises(ValueError, match=err_match):
        extended_path(nk_linear_model, **param_override)


def test_max_iter_1_linear_vs_nonlinear(nk_linear_model, rbc_nonlinear_model):
    """Verify max_iter=1 converges on affine linear model but fails on non-linear."""
    # Linear model: stacked system is linear in Y, reaches machine zero in 1 Newton step
    res_lin = extended_path(nk_linear_model, periods=10, max_iter=1, strict=True)
    assert res_lin.converged is True
    assert all(it == 1 for it in res_lin.iterations)

    # Non-linear model: 1 step cannot reach 1e-8 residual
    shocks = np.array([[0.05], *[[0.0]] * 4])
    res_nl_nonstrict = extended_path(rbc_nonlinear_model, periods=5, shocks=shocks, max_iter=1, strict=False)
    assert res_nl_nonstrict.converged is False

    with pytest.raises(RuntimeError, match="failed to converge"):
        extended_path(rbc_nonlinear_model, periods=5, shocks=shocks, max_iter=1, strict=True)


# ============================================================================
# 3. Presentation Methods Under Extreme Edge Cases
# ============================================================================

def test_presentation_single_period_t1(rbc_nonlinear_model):
    """Verify all presentation methods function on single period T=1 simulation."""
    res = extended_path(rbc_nonlinear_model, periods=1, horizon=20, sigma=0.01, seed=42)
    assert len(res.path) == 1
    assert res.periods == 1

    # Text and DataFrame summaries
    sum_txt = res.summary()
    assert isinstance(sum_txt, str)
    assert "CONVERGED" in sum_txt

    sum_df = res.summary(as_dataframe=True)
    assert isinstance(sum_df, pd.DataFrame)
    assert "initial" in sum_df.columns
    assert "terminal" in sum_df.columns

    # Single-axes and subplots plots
    fig1 = res.plot()
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = res.plot(subplots=True)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # Export formats
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()


def test_presentation_single_variable_nv1():
    """Verify presentation methods on single endogenous variable (n_v=1)."""
    res = ExtendedPathResult(
        path=pd.DataFrame({"x": [1.0, 1.1, 1.05]}, index=pd.RangeIndex(1, 4, name="t")),
        shocks=pd.DataFrame({"e": [0.1, -0.05, 0.0]}, index=pd.RangeIndex(1, 4, name="t")),
        converged=True,
        iterations=[2, 1, 1],
        residual_norm=1e-9,
        terminal_error=1e-10,
    )
    assert res.n_vars == 1

    # summary
    assert isinstance(res.summary(), str)
    assert isinstance(res.summary(as_dataframe=True), pd.DataFrame)

    # plot with subplots=True gracefully falls back to single plot when n_vars == 1
    fig_sub = res.plot(subplots=True)
    assert isinstance(fig_sub, plt.Figure)
    plt.close(fig_sub)


def test_presentation_compound_boundary_t1_nv1():
    """Verify presentation methods when both T=1 and n_v=1."""
    res = ExtendedPathResult(
        path=pd.DataFrame({"y": [2.5]}, index=pd.RangeIndex(1, 2, name="t")),
        shocks=pd.DataFrame({"e": [0.0]}, index=pd.RangeIndex(1, 2, name="t")),
        converged=True,
        iterations=[1],
        residual_norm=0.0,
        terminal_error=0.0,
    )
    assert res.periods == 1
    assert res.n_vars == 1

    assert isinstance(res.summary(), str)
    assert isinstance(res.to_frame(), pd.DataFrame)

    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_presentation_custom_axes_injection(rbc_nonlinear_model):
    """Verify caller-supplied Matplotlib axes is modified and returned directly."""
    res = extended_path(rbc_nonlinear_model, periods=10, horizon=20, sigma=0.01, seed=42)
    fig, custom_ax = plt.subplots(figsize=(6, 3))

    out_ax = res.plot(variables=["c", "k"], ax=custom_ax, alpha=0.7)
    assert out_ax is custom_ax
    plt.close(fig)
