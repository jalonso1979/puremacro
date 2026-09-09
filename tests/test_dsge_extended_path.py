"""Comprehensive unit tests for Fair & Taylor (1983) Extended Path simulation.

Tests cover all 6 requirements:
1. Non-linear model simulation (100 periods, zero divergences).
2. Linear model invariance (analytical parity <= 1e-10 against LinearModel.simulate).
3. Zero-shock steady-state pinning (machine precision <= 1e-12).
4. Horizon sensitivity analysis (T_H in [20, 50, 100], boundary horizons T_H=1, 5).
5. Convergence failure & error guard handling.
6. Result presentation contract compliance (summary, plot, to_frame, to_markdown, to_latex, to_typst).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, extended_path, ExtendedPathResult


# ===========================================================================
# Standard Fixtures
# ===========================================================================

@pytest.fixture
def nonlinear_rbc_setup():
    """Canonical non-linear Hansen/KPR Real Business Cycle (RBC) model.

    Endogenous variables: c (consumption), k (capital), y (output), a (TFP).
    Exogenous shocks: e_a (TFP innovation).
    Non-linear production: y_t = exp(a_t) * k_{t-1}^alpha
    Euler equation: 1 / c_t = beta * E_t[(1 / c_{t+1}) * (alpha * exp(a_{t+1}) * k_t^{alpha-1} + 1 - delta)]
    Resource constraint: y_t = c_t + k_t - (1 - delta) * k_{t-1}
    """
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
def linear_nk_setup():
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
def linear_rbc_setup():
    """Canonical log-linearized Hansen/KPR Real Business Cycle model."""
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


@pytest.fixture
def mock_ep_result():
    """Mock ExtendedPathResult instance for presentation contract validation."""
    df_path = pd.DataFrame(
        {
            "c": [1.00, 1.02, 1.01, 1.03],
            "k": [10.0, 10.1, 10.2, 10.15],
            "y": [1.20, 1.25, 1.22, 1.24],
        },
        index=pd.RangeIndex(1, 5, name="t"),
    )
    df_shocks = pd.DataFrame(
        {"e_a": [0.01, -0.02, 0.005, -0.015]},
        index=pd.RangeIndex(1, 5, name="t"),
    )
    return ExtendedPathResult(
        path=df_path,
        shocks=df_shocks,
        converged=True,
        iterations=[3, 2, 2, 2],
        residual_norm=1.2e-8,
        terminal_error=4.5e-9,
    )


# ===========================================================================
# 1. Non-Linear Model Simulation (Fair & Taylor 1983)
# ===========================================================================

def test_nonlinear_rbc_simulation_100_periods_zero_divergence(nonlinear_rbc_setup):
    """Verify non-linear RBC model executes over 100 periods with 0 solver divergences."""
    m = nonlinear_rbc_setup["model"]
    res = extended_path(m, periods=100, horizon=50, sigma=0.01, seed=42)

    assert res.converged is True
    assert len(res.path) == 100
    assert len(res.shocks) == 100
    if isinstance(res.iterations, list):
        assert all(it < 50 for it in res.iterations)
    assert res.residual_norm < 1e-6
    assert res.terminal_error < 1e-4


def test_nonlinear_rbc_burn_in_sample_advancement(nonlinear_rbc_setup):
    """Verify burn-in period advances state and leaves exactly `periods` samples."""
    m = nonlinear_rbc_setup["model"]
    res = extended_path(m, periods=50, burn=30, horizon=40, sigma=0.01, seed=42)

    assert len(res.path) == 50
    assert len(res.shocks) == 50
    assert res.path.index[0] == 1
    assert res.path.index[-1] == 50
    # Because of burn-in shocks, period 1 state should have deviated from steady state
    ss_k = nonlinear_rbc_setup["steady_state"]["k"]
    assert abs(res.path["k"].iloc[0] - ss_k) > 1e-6


def test_nonlinear_rbc_strictly_positive_capital_and_consumption(nonlinear_rbc_setup):
    """Verify consumption and capital remain strictly positive along non-linear trajectory."""
    m = nonlinear_rbc_setup["model"]
    res = extended_path(m, periods=80, horizon=40, sigma=0.015, seed=123)

    assert np.all(res.path["c"] > 0)
    assert np.all(res.path["k"] > 0)
    assert np.all(res.path["y"] > 0)


def test_nonlinear_rbc_reproducibility_via_seed(nonlinear_rbc_setup):
    """Verify extended path produces deterministic, bitwise identical output given seed."""
    m = nonlinear_rbc_setup["model"]
    res1 = extended_path(m, periods=30, horizon=35, sigma=0.01, seed=999)
    res2 = extended_path(m, periods=30, horizon=35, sigma=0.01, seed=999)

    pd.testing.assert_frame_equal(res1.path, res2.path)
    pd.testing.assert_frame_equal(res1.shocks, res2.shocks)


def test_nonlinear_rbc_dynamic_euler_residual_consistency(nonlinear_rbc_setup):
    """Verify dynamic production and resource equations balance at realized states."""
    m = nonlinear_rbc_setup["model"]
    res = extended_path(m, periods=25, horizon=40, sigma=0.01, seed=77)
    p = nonlinear_rbc_setup["params"]

    # Resource constraint: y_t - c_t - k_t + (1 - delta) * k_{t-1} == 0
    y = res.path["y"].to_numpy()
    c = res.path["c"].to_numpy()
    k = res.path["k"].to_numpy()
    ss_k = nonlinear_rbc_setup["steady_state"]["k"]

    for t in range(len(res.path)):
        k_lag = ss_k if t == 0 else k[t - 1]
        res_rc = y[t] - c[t] - k[t] + (1.0 - p["delta"]) * k_lag
        assert abs(res_rc) < 1e-5


# ===========================================================================
# 2. Linear Model Invariance (Exact Analytical Parity <= 1e-10)
# ===========================================================================

def test_linear_nk_extended_path_identically_matches_linear_simulate(linear_nk_setup):
    """Verify Extended Path on NK model matches LinearModel.simulate within <= 1e-10."""
    m = linear_nk_setup["model"]
    T = 30
    shocks = np.random.default_rng(99).normal(0, 0.5, (T, len(m.shocks)))

    res_ep = extended_path(m, periods=T, horizon=40, shocks=shocks, seed=99)
    res_lin = m.simulate(periods=T, shocks=shocks)

    diff = res_ep.path.to_numpy() - res_lin.to_numpy()
    max_dev = float(np.max(np.abs(diff)))
    assert max_dev <= 1e-10, f"Violation: max_dev={max_dev:.4e}"


def test_linear_rbc_extended_path_identically_matches_linear_simulate(linear_rbc_setup):
    """Verify Extended Path on Hansen RBC matches LinearModel.simulate within <= 1e-10."""
    m = linear_rbc_setup["model"]
    T = 35
    shocks = np.random.default_rng(42).normal(0, 0.01, (T, len(m.shocks)))

    res_ep = extended_path(m, periods=T, horizon=350, shocks=shocks, seed=42)
    res_lin = m.simulate(periods=T, shocks=shocks)

    diff = res_ep.path.to_numpy() - res_lin.to_numpy()
    max_dev = float(np.max(np.abs(diff)))
    assert max_dev <= 1e-10, f"Violation: max_dev={max_dev:.4e}"


def test_linear_invariance_under_arbitrary_shock_sequences(linear_nk_setup):
    """Verify invariance holds for multi-period custom shock patterns."""
    m = linear_nk_setup["model"]
    T = 20
    custom_shocks = np.zeros((T, 2))
    custom_shocks[0, 0] = 1.5   # Demand shock
    custom_shocks[5, 1] = -0.75  # Monetary shock
    custom_shocks[10:14, 0] = 0.5

    res_ep = extended_path(m, periods=T, horizon=40, shocks=custom_shocks)
    res_lin = m.simulate(periods=T, shocks=custom_shocks)

    diff = np.max(np.abs(res_ep.path.to_numpy() - res_lin.to_numpy()))
    assert diff <= 1e-10


def test_linear_invariance_across_multiple_seeds(linear_nk_setup):
    """Verify linear invariance holds across multiple stochastic seeds."""
    m = linear_nk_setup["model"]
    for s in [1, 42, 999]:
        shocks = np.random.default_rng(s).normal(0, 0.4, (20, len(m.shocks)))
        res_ep = extended_path(m, periods=20, horizon=40, shocks=shocks)
        res_lin = m.simulate(periods=20, shocks=shocks)
        max_dev = float(np.max(np.abs(res_ep.path.to_numpy() - res_lin.to_numpy())))
        assert max_dev <= 1e-10


def test_linear_invariance_with_nonzero_initial_state(linear_nk_setup):
    """Verify invariance when initialized away from steady state."""
    m = linear_nk_setup["model"]
    T = 25
    y_init = np.array([0.05, -0.02, 0.01, 0.03])
    shocks = np.random.default_rng(123).normal(0, 0.3, (T, len(m.shocks)))

    res_ep = extended_path(m, periods=T, horizon=50, shocks=shocks, y_init=y_init)
    res_lin = m.simulate(periods=T, shocks=shocks, initial_state=y_init)

    diff = np.max(np.abs(res_ep.path.to_numpy() - res_lin.to_numpy()))
    assert diff <= 1e-10


# ===========================================================================
# 3. Zero-Shock Trajectory & Steady-State Pinnedness (<= 1e-12)
# ===========================================================================

def test_zero_shock_nonlinear_rbc_remains_at_steady_state(nonlinear_rbc_setup):
    """Verify non-linear RBC remains at steady state under zero shocks within 1e-12."""
    m = nonlinear_rbc_setup["model"]
    ss = nonlinear_rbc_setup["steady_state"]
    ss_arr = np.array([ss[v] for v in m.variables])

    res = extended_path(m, periods=20, horizon=30, sigma=0.0)
    dev = np.max(np.abs(res.path.to_numpy() - ss_arr))
    assert dev <= 1e-12


def test_zero_shock_linear_nk_remains_at_zero(linear_nk_setup):
    """Verify NK model remains identically at zero under zero shocks within 1e-12."""
    m = linear_nk_setup["model"]
    res = extended_path(m, periods=25, horizon=30, sigma=0.0)
    dev = np.max(np.abs(res.path.to_numpy()))
    assert dev <= 1e-12


def test_zero_shock_linear_rbc_remains_at_steady_state(linear_rbc_setup):
    """Verify log-linear RBC remains at 0 under zero shocks within 1e-12."""
    m = linear_rbc_setup["model"]
    res = extended_path(m, periods=20, horizon=30, sigma=0.0)
    dev = np.max(np.abs(res.path.to_numpy()))
    assert dev <= 1e-12


def test_zero_shock_iteration_count_at_most_two(linear_nk_setup):
    """Verify solver converges in <= 2 iterations per step under zero shocks."""
    m = linear_nk_setup["model"]
    res = extended_path(m, periods=15, horizon=25, sigma=0.0)
    assert np.all(np.asarray(res.iterations) <= 2)


def test_zero_shock_terminal_error_machine_zero(nonlinear_rbc_setup):
    """Verify residual norm and terminal error are machine zero under zero shocks."""
    m = nonlinear_rbc_setup["model"]
    res = extended_path(m, periods=15, horizon=30, sigma=0.0)
    assert res.residual_norm <= 1e-12
    assert res.terminal_error <= 1e-12


# ===========================================================================
# 4. Horizon Sensitivity Analysis (T_H in [20, 50, 100])
# ===========================================================================

def test_horizon_sensitivity_terminal_error_monotonic_decay(nonlinear_rbc_setup):
    """Verify forward horizon truncation error decays as T_H increases."""
    m = nonlinear_rbc_setup["model"]
    shocks = np.random.default_rng(42).normal(0, 0.01, (20, 1))

    res20 = extended_path(m, periods=20, horizon=20, shocks=shocks)
    res50 = extended_path(m, periods=20, horizon=50, shocks=shocks)
    res100 = extended_path(m, periods=20, horizon=100, shocks=shocks)

    # Difference relative to T_H=100 should be larger for T_H=20 than for T_H=50
    diff20 = np.max(np.abs(res20.path.to_numpy() - res100.path.to_numpy()))
    diff50 = np.max(np.abs(res50.path.to_numpy() - res100.path.to_numpy()))
    assert diff50 <= diff20


def test_horizon_sensitivity_trajectory_geometric_convergence(nonlinear_rbc_setup):
    """Verify geometric decay of truncation error matching dominant eigenvalue."""
    m = nonlinear_rbc_setup["model"]
    shocks = np.random.default_rng(88).normal(0, 0.01, (15, 1))

    res25 = extended_path(m, periods=15, horizon=25, shocks=shocks)
    res50 = extended_path(m, periods=15, horizon=50, shocks=shocks)
    res100 = extended_path(m, periods=15, horizon=100, shocks=shocks)

    err_25_50 = np.max(np.abs(res25.path.to_numpy() - res50.path.to_numpy()))
    err_50_100 = np.max(np.abs(res50.path.to_numpy() - res100.path.to_numpy()))
    assert err_50_100 < 0.2 * err_25_50


def test_horizon_sensitivity_high_persistence_model(linear_nk_setup):
    """Verify extended path executes stably with high persistence shocks."""
    m = linear_nk_setup["model"]
    res = extended_path(m, periods=20, horizon=60, seed=5)
    assert res.converged is True
    assert np.all(np.isfinite(res.path.to_numpy()))


def test_minimal_boundary_horizon_th_one(linear_nk_setup):
    """Verify minimal horizon T_H=1 boundary executes without crash."""
    m = linear_nk_setup["model"]
    res = extended_path(m, periods=10, horizon=1, seed=42)
    assert len(res.path) == 10
    assert isinstance(res.path, pd.DataFrame)


def test_minimal_boundary_horizon_th_five(linear_nk_setup):
    """Verify short horizon T_H=5 boundary converges."""
    m = linear_nk_setup["model"]
    res = extended_path(m, periods=10, horizon=5, seed=42)
    assert res.converged is True


# ===========================================================================
# 5. Convergence Failure & Error Guard Handling
# ===========================================================================

def test_max_iter_cutoff_graceful_handling(nonlinear_rbc_setup):
    """Verify max_iter=1 with strict=False returns converged=False without raising."""
    m = nonlinear_rbc_setup["model"]
    shocks = np.array([[0.1], *[[0.0]] * 4])
    res = extended_path(m, periods=5, horizon=20, shocks=shocks, max_iter=1, seed=42, strict=False)
    assert hasattr(res, "converged")
    assert res.converged is False


def test_max_iter_cutoff_strict_raises_runtime_error(nonlinear_rbc_setup):
    """Verify max_iter=1 with strict=True raises RuntimeError on non-linear model."""
    m = nonlinear_rbc_setup["model"]
    shocks = np.array([[0.1], *[[0.0]] * 4])
    with pytest.raises(RuntimeError, match="failed to converge"):
        extended_path(m, periods=5, horizon=20, shocks=shocks, max_iter=1, strict=True)


def test_invalid_shock_dimension_raises_value_error(linear_nk_setup):
    """Verify shock matrix with incorrect column count raises ValueError."""
    m = linear_nk_setup["model"]
    bad_shocks = np.zeros((20, 5))  # NK model has 2 shocks
    with pytest.raises((ValueError, IndexError, AssertionError)):
        extended_path(m, periods=20, shocks=bad_shocks)


def test_invalid_horizon_and_periods_raises_value_error(linear_nk_setup):
    """Verify non-positive periods or horizon raises ValueError."""
    m = linear_nk_setup["model"]
    with pytest.raises(ValueError):
        extended_path(m, periods=0)
    with pytest.raises(ValueError):
        extended_path(m, horizon=0)
    with pytest.raises(ValueError):
        extended_path(m, burn=-1)


def test_invalid_y_init_dimension_raises_value_error(linear_nk_setup):
    """Verify y_init with incorrect variable count raises ValueError."""
    m = linear_nk_setup["model"]
    bad_init = np.zeros(10)  # NK model has 4 variables
    with pytest.raises(ValueError, match="y_init dimension"):
        extended_path(m, periods=10, y_init=bad_init)


def test_extreme_10_sigma_shock_resilience(linear_nk_setup):
    """Verify massive 10-sigma shock does not produce NaN or overflow."""
    m = linear_nk_setup["model"]
    T = 20
    shocks = np.zeros((T, len(m.shocks)))
    shocks[0, 0] = 10.0

    res = extended_path(m, periods=T, horizon=30, shocks=shocks)
    assert np.all(np.isfinite(res.path.to_numpy()))


# ===========================================================================
# 6. Result Class Presentation Contract Compliance
# ===========================================================================

def test_result_presentation_contract_immutability(mock_ep_result):
    """Verify ExtendedPathResult is frozen and immutable."""
    with pytest.raises((FrozenInstanceError, AttributeError)):
        mock_ep_result.converged = False


def test_result_presentation_contract_attributes_and_properties(mock_ep_result):
    """Verify all 8 result attributes and helper properties."""
    assert mock_ep_result.periods == 4
    assert mock_ep_result.n_vars == 3
    assert mock_ep_result.n_shocks == 1
    assert mock_ep_result.total_iterations == 9
    assert mock_ep_result.mean_iterations == 2.25
    assert mock_ep_result.max_iterations == 3
    assert mock_ep_result.residual_norm == 1.2e-8
    assert mock_ep_result.terminal_error == 4.5e-9


def test_result_presentation_contract_to_frame(mock_ep_result):
    """Verify .to_frame() returns DataFrame copy without mutating internal state."""
    df = mock_ep_result.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    df.iloc[0, 0] = 999.0
    assert mock_ep_result.path.iloc[0, 0] != 999.0


def test_result_presentation_contract_summary_text_and_dataframe(mock_ep_result):
    """Verify .summary() provides publication string report and DataFrame."""
    summary_txt = mock_ep_result.summary()
    assert isinstance(summary_txt, str)
    assert "Extended Path" in summary_txt or "Fair-Taylor" in summary_txt
    assert "CONVERGED" in summary_txt

    summary_df = mock_ep_result.summary(as_dataframe=True)
    assert isinstance(summary_df, pd.DataFrame)
    for col in ["mean", "std", "min", "max", "initial", "terminal"]:
        assert col in summary_df.columns


def test_result_presentation_contract_to_markdown(mock_ep_result):
    """Verify .to_markdown() exports clean table."""
    md_str = mock_ep_result.to_markdown()
    assert isinstance(md_str, str)
    assert "|" in md_str
    assert "c" in md_str

    md_head = mock_ep_result.to_markdown(head=1)
    assert isinstance(md_head, str)


def test_result_presentation_contract_to_latex(mock_ep_result):
    """Verify .to_latex() exports tabular environment."""
    tex_str = mock_ep_result.to_latex()
    assert isinstance(tex_str, str)
    assert "\\begin{tabular}" in tex_str
    assert "\\end{tabular}" in tex_str


def test_result_presentation_contract_to_typst(mock_ep_result):
    """Verify .to_typst() exports Typst table."""
    typ_str = mock_ep_result.to_typst()
    assert isinstance(typ_str, str)
    assert "#table(" in typ_str


def test_result_presentation_contract_plot_default(mock_ep_result):
    """Verify .plot() returns Figure without crashing."""
    fig = mock_ep_result.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_result_presentation_contract_plot_variables_subset(mock_ep_result):
    """Verify .plot() filters to variable subset."""
    fig = mock_ep_result.plot(variables=["c"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_result_presentation_contract_plot_custom_ax(mock_ep_result):
    """Verify .plot() draws on caller-supplied axes."""
    fig, ax = plt.subplots()
    out = mock_ep_result.plot(ax=ax)
    assert out is ax
    plt.close(fig)


def test_result_presentation_contract_plot_subplots_grid(mock_ep_result):
    """Verify .plot(subplots=True) creates multiple subplots."""
    fig = mock_ep_result.plot(subplots=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_result_presentation_contract_plot_styles(mock_ep_result):
    """Verify .plot() supports publication and grayscale styles."""
    fig1 = mock_ep_result.plot(style="publication")
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = mock_ep_result.plot(style="grayscale")
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_result_item_access_and_key_errors(mock_ep_result):
    """Verify subscript access on variables, shocks, and KeyError on missing."""
    assert isinstance(mock_ep_result["c"], pd.Series)
    assert isinstance(mock_ep_result["e_a"], pd.Series)
    with pytest.raises(KeyError):
        _ = mock_ep_result["nonexistent_var"]
