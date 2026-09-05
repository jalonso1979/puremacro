"""Tests for Sequence-Space HANK solver and Non-Linear Transition Dynamics (Auclert et al. 2021)."""
from __future__ import annotations

import time
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.models import (
    SequenceSpaceHANKResult,
    FakeNewsResult,
    FiscalTransferResult,
    NonlinearHANKResult,
    solve_hank_sequence_space,
    fake_news_algorithm,
    simulate_targeted_transfer,
    solve_nonlinear_transition,
)


def test_solve_nonlinear_transition_large_monetary_shock():
    """Test non-linear transition under large monetary shock converging to ||H||_inf < 1e-6 in < 5s."""
    horizon = 300
    shock_seq = 0.01 * (0.7 ** np.arange(horizon))  # 100 bps monetary policy shock

    t0 = time.perf_counter()
    res = solve_nonlinear_transition(
        ss_model=None,
        shock_seq=shock_seq,
        shock_var="r",
        horizon=horizon,
        max_iter=100,
        tol=1e-6,
        backtracking=True,
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(res, NonlinearHANKResult)
    assert res.converged is True
    max_residual = float(np.max(np.abs(res.residuals)))
    assert max_residual < 1e-6, f"Residual {max_residual:.2e} >= 1e-6"
    assert elapsed < 5.0, f"Elapsed time {elapsed:.2f}s >= 5.0s"

    assert len(res.U) == horizon
    assert len(res.linear_path) == horizon
    assert len(res.nonlinear_path) == horizon
    assert len(res.residuals) == horizon
    assert len(res.irf_consumption_nonlinear) == horizon
    assert len(res.irf_rate_nonlinear) == horizon
    assert len(res.irf_inflation_nonlinear) == horizon

    # Monetary contraction should reduce output and consumption initially
    assert res.nonlinear_path[0] < 0.0
    assert res.irf_consumption_nonlinear[0] < 0.0
    assert res.irf_rate_nonlinear[0] > 0.0
    assert 0 < res.iterations <= 100


def test_solve_nonlinear_transition_large_fiscal_shock():
    """Test non-linear transition under large fiscal spending shock converging in < 5s."""
    horizon = 300
    shock_seq = 0.01 * (0.7 ** np.arange(horizon))  # 1% GDP spending shock

    t0 = time.perf_counter()
    res = solve_nonlinear_transition(
        ss_model=None,
        shock_seq=shock_seq,
        shock_var="G",
        horizon=horizon,
        max_iter=100,
        tol=1e-6,
        backtracking=True,
    )
    elapsed = time.perf_counter() - t0

    assert isinstance(res, NonlinearHANKResult)
    assert res.converged is True
    max_residual = float(np.max(np.abs(res.residuals)))
    assert max_residual < 1e-6, f"Residual {max_residual:.2e} >= 1e-6"
    assert elapsed < 5.0, f"Elapsed time {elapsed:.2f}s >= 5.0s"

    # Fiscal expansion stimulates output on impact and clears goods market Y = C + G
    assert res.nonlinear_path[0] > 0.0
    assert np.isclose(res.nonlinear_path[0], res.irf_consumption_nonlinear[0] + shock_seq[0], atol=1e-5)


def test_nonlinear_asymmetry():
    """Verify non-linear asymmetry: positive vs negative shocks produce distinct nonlinear responses."""
    horizon = 100
    shock_pos = +0.01 * (0.7 ** np.arange(horizon))
    shock_neg = -0.01 * (0.7 ** np.arange(horizon))

    # Pre-solve steady state to speed up comparison
    ss = solve_hank_sequence_space(T=40, n_a=50)

    res_pos = solve_nonlinear_transition(ss, shock_pos, shock_var="r", horizon=horizon)
    res_neg = solve_nonlinear_transition(ss, shock_neg, shock_var="r", horizon=horizon)

    assert res_pos.converged is True
    assert res_neg.converged is True

    # In linear model, response is symmetric: pos + neg == 0
    linear_sum = res_pos.linear_path + res_neg.linear_path
    assert np.allclose(linear_sum, 0.0, atol=1e-10)

    # In non-linear model, borrowing constraints and precautionary concave savings induce asymmetry
    nonlin_sum = res_pos.nonlinear_path + res_neg.nonlinear_path
    max_asymmetry = float(np.max(np.abs(nonlin_sum)))
    assert max_asymmetry > 1e-5, f"Non-linear asymmetry too small: {max_asymmetry:.2e}"


def test_nonlinear_hank_result_presentation_suite():
    """Test standard presentation interface: .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()."""
    horizon = 40
    shock_seq = 0.005 * (0.7 ** np.arange(horizon))
    res = solve_nonlinear_transition(shock_seq=shock_seq, horizon=horizon)

    # 1. Summary
    summary = res.summary()
    assert "Non-Linear Sequence-Space HANK Transition Dynamics" in summary
    assert "Broyden Solver Status" in summary
    assert "CONVERGED" in summary
    assert f"{horizon} quarters" in summary

    # 2. DataFrame
    df = res.to_frame()
    assert df.shape == (horizon, 9)
    for col in ["Output_Linear", "Output_Nonlinear", "Consumption_Linear", "Consumption_Nonlinear", "Residual"]:
        assert col in df.columns

    # 3. Formats
    md = res.to_markdown()
    assert "|" in md
    assert "Output_Nonlinear" in md

    tex = res.to_latex()
    assert "\\begin{tabular}" in tex or "\\toprule" in tex

    typ = res.to_typst()
    assert "#table" in typ


def test_nonlinear_hank_plot_4_panel():
    """Test 4-panel comparison plot (Y, C, r, H)."""
    horizon = 40
    res = solve_nonlinear_transition(horizon=horizon)

    # Publication style
    fig = res.plot(style="publication")
    assert fig is not None
    assert len(fig.axes) == 4
    plt.close(fig)

    # Grayscale style
    fig_gray = res.plot(style="grayscale")
    assert fig_gray is not None
    assert len(fig_gray.axes) == 4
    plt.close(fig_gray)


def test_sequence_space_model_method_integration():
    """Test calling .solve_nonlinear directly on SequenceSpaceHANKResult."""
    ss = solve_hank_sequence_space(T=30, n_a=30)
    res = ss.solve_nonlinear(horizon=40, max_iter=50)

    assert isinstance(res, NonlinearHANKResult)
    assert res.converged is True
    assert len(res.U) == 40


def test_solve_nonlinear_transition_invalid_inputs():
    """Test error handling on invalid inputs."""
    with pytest.raises(ValueError, match="Unknown shock_var"):
        solve_nonlinear_transition(shock_var="invalid_shock")

    with pytest.raises(TypeError, match="ss_model must be SequenceSpaceHANKResult"):
        solve_nonlinear_transition(ss_model="invalid_model_type")


def test_solve_hank_sequence_space_basic():
    """Verify base SequenceSpaceHANKResult functionality."""
    res = solve_hank_sequence_space(T=30, n_a=30)

    assert isinstance(res, SequenceSpaceHANKResult)
    assert len(res.irf_output) == 30
    assert len(res.irf_consumption) == 30
    assert len(res.irf_inflation) == 30
    assert len(res.irf_rate) == 30
    assert res.jacobian_c_r.shape == (30, 30)
    assert res.jacobian_c_y.shape == (30, 30)
    assert 0.01 <= res.steady_state_mpc <= 0.8
    assert len(res.mpc_distribution) == 10
    assert res.irf_output[0] < 0.0


def test_fake_news_algorithm():
    """Verify fake news algorithm Proposition 1 identity."""
    res = solve_hank_sequence_space(T=25, n_a=25)
    fn = res.fake_news()

    assert isinstance(fn, FakeNewsResult)
    assert fn.horizon == 25
    assert fn.jacobian.shape == (25, 25)
    assert fn.fake_news.shape == (25, 25)

    for t in range(1, 25):
        for s in range(1, 25):
            expected = fn.jacobian[t - 1, s - 1] + fn.fake_news[t, s]
            assert np.isclose(fn.jacobian[t, s], expected, atol=1e-12)


def test_simulate_targeted_transfer():
    """Verify targeted fiscal transfer simulations."""
    res = solve_hank_sequence_space(T=30, n_a=30)
    trans_borrowers = res.simulate_transfer(target="borrowers", amount=1.0)
    assert isinstance(trans_borrowers, FiscalTransferResult)
    assert trans_borrowers.impact_mpc > 0.0

    trans_unconstrained = res.simulate_transfer(target="unconstrained", amount=1.0)
    assert trans_borrowers.impact_mpc > 1.5 * trans_unconstrained.impact_mpc
