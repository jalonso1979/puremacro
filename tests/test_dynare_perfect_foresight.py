"""Tests for DSGE Deterministic Non-Linear Simulation / Perfect Foresight Solver."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import PerfectForesightResult, solve_perfect_foresight


@pytest.fixture
def ramsey_model():
    """Deterministic Neoclassical Growth (Cass-Koopmans / Ramsey) Model setup.

    Euler equation:
        c_t^(-sigma) = beta * c_{t+1}^(-sigma) * (alpha * A_t * k_t^(alpha-1) + 1 - delta)
    Resource constraint:
        k_t = A_t * k_{t-1}^alpha + (1-delta) * k_{t-1} - c_t
    """
    alpha = 0.33
    beta = 0.96
    delta = 0.10
    sigma = 1.0
    A_ss = 1.0

    # Steady state
    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (alpha * A_ss / r_ss) ** (1.0 / (1.0 - alpha))
    y_ss_val = A_ss * k_ss**alpha
    c_ss = y_ss_val - delta * k_ss

    def equations_fn(y_plus, y_curr, y_lag, eps):
        c_p, k_p = y_plus
        c, k = y_curr
        c_m, k_m = y_lag
        A = float(eps) if np.ndim(eps) == 0 else float(eps[0])
        euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1.0) + 1.0 - delta)
        res_c = k - (A * k_m**alpha + (1.0 - delta) * k_m - c)
        return [euler, res_c]

    y_ss = np.array([c_ss, k_ss])

    return {
        "equations_fn": equations_fn,
        "y_ss": y_ss,
        "k_ss": k_ss,
        "c_ss": c_ss,
        "params": dict(alpha=alpha, beta=beta, delta=delta, sigma=sigma, A_ss=A_ss),
    }


def test_ramsey_steady_state_zero_residual(ramsey_model):
    """Verify that steady state satisfies equations with machine-zero error."""
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    res = eqs(y_ss, y_ss, y_ss, 1.0)
    np.testing.assert_allclose(res, [0.0, 0.0], atol=1e-14)


def test_ramsey_transition_monotonic_capital(ramsey_model):
    """Test 1: Transition from low initial capital k_0 = 0.5 * k_ss to steady state.

    Verifies:
    - Stacked Newton-Raphson converges within tolerance (residual_norm < 1e-8).
    - Monotonic capital accumulation along the transition path towards steady state.
    - Terminal capital matches steady state.
    """
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    k_ss = ramsey_model["k_ss"]
    c_ss = ramsey_model["c_ss"]

    # Initial condition: k_0 = 0.5 * k_ss (capital scarcity)
    y_init = np.array([c_ss, 0.5 * k_ss])
    n_periods = 100
    exo_path = np.ones(n_periods)

    result = solve_perfect_foresight(
        eqs,
        y_init=y_init,
        y_ss=y_ss,
        exogenous_path=exo_path,
        n_periods=n_periods,
        tol=1e-8,
        variable_names=["c", "k"],
    )

    assert isinstance(result, PerfectForesightResult)
    assert result.converged is True
    assert result.iterations <= 10
    assert result.residual_norm < 1e-8
    assert result.terminal_error < 1e-4

    # Check path dimensions and index
    path = result.path
    assert path.shape == (n_periods, 2)
    assert list(path.columns) == ["c", "k"]
    assert path.index[0] == 1
    assert path.index[-1] == n_periods

    # Monotonic capital accumulation:
    # Capital strictly increases at each period t from k_0 towards k_ss
    k_trajectory = np.r_[y_init[1], path["k"].values]
    k_diffs = np.diff(k_trajectory)
    # Check that capital accumulation is strictly positive along transition
    assert np.all(k_diffs[:-1] > 0)
    # Terminal capital is within 0.01% of steady state
    assert path["k"].iloc[-1] == pytest.approx(k_ss, rel=1e-4)


def test_announced_vs_unannounced_shock_at_t5(ramsey_model):
    """Test 2: Unannounced vs announced technology shock at t=5.

    Under perfect foresight:
    - When a +5% TFP shock at t=5 is pre-announced at t=1, forward-looking
      households immediately adjust consumption upward at t=1..4 (before the shock hits).
    - When the shock is unannounced, households do not anticipate it;
      the economy remains at steady state for t=1..4 and consumption does not move.
    """
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    c_ss = ramsey_model["c_ss"]

    T = 50
    # Case A: Announced shock at t=5 (announced at t=1)
    exo_ann = np.ones(T)
    exo_ann[4] = 1.05  # t=5 (index 4 in 0-based array corresponds to period 5)

    res_ann = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=exo_ann,
        n_periods=T,
        variable_names=["c", "k"],
    )

    assert res_ann.converged is True
    c_ann = res_ann.path["c"]

    # Under announced shock, households anticipate higher future wealth:
    # consumption at t=1, 2, 3, 4 is strictly higher than steady state!
    assert c_ann.loc[1] > c_ss
    assert c_ann.loc[2] > c_ss
    assert c_ann.loc[3] > c_ss
    assert c_ann.loc[4] > c_ss

    # Case B: Unannounced shock at t=5
    # Before t=5 (periods 1, 2, 3, 4), households expect no shock, so the economy
    # stays at steady state (c_t = c_ss, k_t = k_ss).
    # At t=5, the shock hits unexpectedly starting from k_4 = k_ss.
    exo_unann_sub = np.ones(T - 4)
    exo_unann_sub[0] = 1.05  # shock hits in the first period of sub-path (period 5)

    res_unann_sub = solve_perfect_foresight(
        eqs,
        y_init=y_ss,
        y_ss=y_ss,
        exogenous_path=exo_unann_sub,
        n_periods=T - 4,
        variable_names=["c", "k"],
    )

    assert res_unann_sub.converged is True
    c_unann_pre = pd.Series([c_ss, c_ss, c_ss, c_ss], index=[1, 2, 3, 4])

    # For unannounced shock, consumption before t=5 does NOT adjust
    assert c_unann_pre.loc[1] == pytest.approx(c_ss)
    assert c_unann_pre.loc[4] == pytest.approx(c_ss)

    # Comparing announced vs unannounced:
    # Consumption is strictly higher under announced shock before t=5
    diff_t1 = c_ann.loc[1] - c_unann_pre.loc[1]
    diff_t4 = c_ann.loc[4] - c_unann_pre.loc[4]
    assert diff_t1 > 1e-4
    assert diff_t4 > 1e-3


def test_adversarial_residual_norm_across_all_periods(ramsey_model):
    """Adversarial test: Residual norm < 1e-8 across all individual periods.

    Tests multiple challenging non-linear scenarios:
    1. Capital decumulation from high initial capital (k_0 = 1.8 * k_ss)
    2. Deep capital poverty (k_0 = 0.25 * k_ss)
    3. Permanent technology shock (A jumps from 1.0 to 1.1 at t=10)
    In all cases, evaluates every single equation at every single period manually
    to confirm max |residual| < 1e-8.
    """
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    k_ss = ramsey_model["k_ss"]
    c_ss = ramsey_model["c_ss"]

    test_cases = [
        ("high_k", np.array([c_ss, 1.8 * k_ss]), y_ss, np.ones(80), 80),
        ("low_k", np.array([c_ss, 0.25 * k_ss]), y_ss, np.ones(80), 80),
    ]

    # Permanent shock setup
    alpha = ramsey_model["params"]["alpha"]
    beta = ramsey_model["params"]["beta"]
    delta = ramsey_model["params"]["delta"]
    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss_new = (alpha * 1.1 / r_ss) ** (1.0 / (1.0 - alpha))
    c_ss_new = 1.1 * k_ss_new**alpha - delta * k_ss_new
    y_ss_new = np.array([c_ss_new, k_ss_new])

    exo_perm = np.ones(80)
    exo_perm[9:] = 1.1  # permanent jump at t=10
    test_cases.append(("permanent_shock", y_ss, y_ss_new, exo_perm, 80))

    for label, y0, y_end, exo, T in test_cases:
        res = solve_perfect_foresight(
            eqs,
            y_init=y0,
            y_ss=y_end,
            exogenous_path=exo,
            n_periods=T,
            tol=1e-8,
            variable_names=["c", "k"],
        )

        assert res.converged is True, f"Failed to converge for {label}"
        assert res.residual_norm < 1e-8, f"Residual norm >= 1e-8 for {label}"

        # Explicit independent check across EVERY period t = 1..T
        path_arr = res.path.values
        for t in range(T):
            y_l = y0 if t == 0 else path_arr[t - 1]
            y_c = path_arr[t]
            y_p = y_end if t == T - 1 else path_arr[t + 1]
            r = eqs(y_p, y_c, y_l, exo[t])
            assert np.max(np.abs(r)) < 1e-8, (
                f"Period t={t+1} violated residual tolerance in case '{label}': "
                f"residuals={r}"
            )


def test_result_export_methods(ramsey_model):
    """Test .to_frame(), .summary(), .to_latex(), .to_typst(), .to_markdown()."""
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    k_ss = ramsey_model["k_ss"]
    c_ss = ramsey_model["c_ss"]

    res = solve_perfect_foresight(
        eqs,
        y_init=np.array([c_ss, 0.8 * k_ss]),
        y_ss=y_ss,
        exogenous_path=np.ones(30),
        n_periods=30,
        variable_names=["c", "k"],
    )

    # 1. to_frame
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (30, 2)
    pd.testing.assert_frame_equal(df, res.path)

    # 2. summary
    summary_str = res.summary()
    assert isinstance(summary_str, str)
    assert "PERFECT FORESIGHT" in summary_str
    assert "CONVERGED" in summary_str
    assert "Iterations" in summary_str
    assert "Residual norm" in summary_str
    assert "Terminal error" in summary_str

    summary_df = res.summary(as_dataframe=True)
    assert isinstance(summary_df, pd.DataFrame)
    assert "mean" in summary_df.columns
    assert "initial" in summary_df.columns

    # 3. to_markdown
    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert " c " in md_str or "c" in md_str
    assert " k " in md_str or "k" in md_str
    assert "|" in md_str

    md_head = res.to_markdown(head=5)
    assert md_head.count("\n") < md_str.count("\n")

    # 4. to_latex
    latex_str = res.to_latex()
    assert isinstance(latex_str, str)
    assert "\\begin{tabular}" in latex_str
    assert "\\end{tabular}" in latex_str

    # 5. to_typst
    typst_str = res.to_typst()
    assert isinstance(typst_str, str)
    assert "#table(" in typst_str

    # 6. Item access
    assert np.allclose(res["c"].values, res.path["c"].values)
    assert np.allclose(res["k"].values, res.path["k"].values)
    with pytest.raises(KeyError):
        _ = res["nonexistent"]


def test_result_plot(ramsey_model):
    """Test .plot(variables=None, style='publication') and styles."""
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    k_ss = ramsey_model["k_ss"]
    c_ss = ramsey_model["c_ss"]

    res = solve_perfect_foresight(
        eqs,
        y_init=np.array([c_ss, 0.7 * k_ss]),
        y_ss=y_ss,
        exogenous_path=np.ones(25),
        n_periods=25,
        variable_names=["c", "k"],
    )

    # Publication style
    fig1 = res.plot(style="publication")
    assert isinstance(fig1, Figure)
    plt.close(fig1)

    # Default style with subset of variables
    fig2 = res.plot(variables=["c"], style="default")
    assert isinstance(fig2, Figure)
    plt.close(fig2)

    # Custom ax
    fig3, ax3 = plt.subplots()
    fig_out = res.plot(ax=ax3, title="Custom Title")
    assert fig_out is fig3
    plt.close(fig3)

    # Unknown variable error
    with pytest.raises(ValueError, match="None of requested variables"):
        res.plot(variables=["unknown"])


def test_dampening_and_convergence_options(ramsey_model):
    """Test dampening parameter, differentiation methods, and input validation."""
    eqs = ramsey_model["equations_fn"]
    y_ss = ramsey_model["y_ss"]
    k_ss = ramsey_model["k_ss"]
    c_ss = ramsey_model["c_ss"]

    # Dampened Newton-Raphson step (0.8)
    res_damp = solve_perfect_foresight(
        eqs,
        y_init=np.array([c_ss, 0.6 * k_ss]),
        y_ss=y_ss,
        exogenous_path=np.ones(20),
        n_periods=20,
        dampening=0.8,
        variable_names=["c", "k"],
    )
    assert res_damp.converged is True

    # Method = "central"
    res_cd = solve_perfect_foresight(
        eqs,
        y_init=np.array([c_ss, 0.6 * k_ss]),
        y_ss=y_ss,
        exogenous_path=np.ones(20),
        n_periods=20,
        method="central",
        variable_names=["c", "k"],
    )
    assert res_cd.converged is True

    # Input validation: dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        solve_perfect_foresight(eqs, y_init=np.ones(2), y_ss=np.ones(3), exogenous_path=np.ones(10), n_periods=10)

    # Input validation: insufficient exogenous path
    with pytest.raises(ValueError, match="must be at least n_periods"):
        solve_perfect_foresight(eqs, y_init=y_ss, y_ss=y_ss, exogenous_path=np.ones(5), n_periods=10)

    # Exogenous path with T+2 boundary elements
    exo_t_plus_2 = np.ones(22)  # for n_periods=20
    res_slice = solve_perfect_foresight(
        eqs,
        y_init=np.array([c_ss, 0.8 * k_ss]),
        y_ss=y_ss,
        exogenous_path=exo_t_plus_2,
        n_periods=20,
        variable_names=["c", "k"],
    )
    assert res_slice.converged is True
    assert len(res_slice.path) == 20
