"""Comprehensive unit tests for Waggoner & Zha (1999) Conditional Forecasting.

Tests cover:
1. Target path enforcement across forecast horizons.
2. Multi-target, multi-shock inversion.
3. Unconstrained intermediate periods (None/NaN targets).
4. Analytical credible intervals (single and multi-confidence levels).
5. Stochastic simulation paths (Monte Carlo conditional paths).
6. Inversion methods: covariance-weighted vs minimum-energy.
7. Underdetermined/conflicting restrictions error handling (exact vs least-squares).
8. Zero-loading shock uninvertibility error detection.
9. Initial state deviation x0 propagation.
10. Presentation contract: summary, plot, to_frame, to_markdown, to_latex, to_typst.
11. Immutability: FrozenInstanceError on mutation.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build_dynare,
    conditional_forecast,
    ConditionalForecastResult,
)


@pytest.fixture
def nk_model():
    """Canonical 3-equation New Keynesian model."""
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

    return build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )


def test_conditional_forecast_exact_target_path_enforcement(nk_model):
    """Verify target interest rate path is enforced exactly over 4 quarters."""
    target_r = [0.02, 0.02, 0.015, 0.01]
    res = conditional_forecast(
        nk_model,
        target_paths={"r": target_r},
        controlled_shocks=["e_m"],
        horizon=6,
    )

    assert isinstance(res, ConditionalForecastResult)
    assert len(res.forecast) == 6
    assert len(res.shocks) == 6
    np.testing.assert_allclose(res.forecast["r"].iloc[:4].to_numpy(), target_r, atol=1e-10)
    # Uncontrolled shock e_d must remain exactly zero
    assert np.all(res.shocks["e_d"] == 0.0)


def test_conditional_forecast_multiple_targets_multiple_shocks(nk_model):
    """Verify simultaneous targets on output gap and interest rate using both shocks."""
    target_y = [0.01, 0.015]
    target_r = [0.005, 0.008]
    res = conditional_forecast(
        nk_model,
        target_paths={"y": target_y, "r": target_r},
        controlled_shocks=["e_d", "e_m"],
        horizon=4,
    )

    np.testing.assert_allclose(res.forecast["y"].iloc[:2].to_numpy(), target_y, atol=1e-10)
    np.testing.assert_allclose(res.forecast["r"].iloc[:2].to_numpy(), target_r, atol=1e-10)
    # Beyond target periods, shocks should be zero
    assert np.all(res.shocks.iloc[2:] == 0.0)


def test_conditional_forecast_unconstrained_intermediate_periods(nk_model):
    """Verify paths with unconstrained intermediate periods (None/NaN) are preserved."""
    target_r = [0.02, None, 0.015]
    res = conditional_forecast(
        nk_model,
        target_paths={"r": target_r},
        controlled_shocks=["e_m"],
        horizon=5,
    )

    assert abs(res.forecast["r"].iloc[0] - 0.02) < 1e-10
    assert abs(res.forecast["r"].iloc[2] - 0.015) < 1e-10
    # Period 2 is unconstrained, so r should be determined by model dynamics
    assert np.isfinite(res.forecast["r"].iloc[1])


def test_conditional_forecast_analytical_credible_intervals(nk_model):
    """Verify calculation of analytical forecast credible bands."""
    res = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02, 0.02]},
        controlled_shocks=["e_m"],
        horizon=8,
        ci=(0.68, 0.95),
    )

    assert "lower_68" in res.bands
    assert "upper_68" in res.bands
    assert "lower_95" in res.bands
    assert "upper_95" in res.bands

    # 95% band must be wider than 68% band
    diff_95 = res.bands["upper_95"]["r"] - res.bands["lower_95"]["r"]
    diff_68 = res.bands["upper_68"]["r"] - res.bands["lower_68"]["r"]
    assert np.all(diff_95 >= diff_68)


def test_conditional_forecast_stochastic_simulation_paths(nk_model):
    """Verify generation of Monte Carlo stochastic simulation paths."""
    res = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02, 0.02]},
        controlled_shocks=["e_m"],
        horizon=4,
        n_sims=30,
        seed=42,
    )

    assert res.simulations is not None
    assert res.simulations.shape == (30, 4, 4)
    # Target constraints must hold across all stochastic simulations
    r_idx = list(nk_model.variables).index("r")
    np.testing.assert_allclose(res.simulations[:, :2, r_idx], 0.02, atol=1e-8)


def test_conditional_forecast_inversion_methods(nk_model):
    """Verify covariance_weighted and minimum_energy inversion methods."""
    res_cov = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02]},
        controlled_shocks=["e_d", "e_m"],
        method="covariance_weighted",
    )
    res_me = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02]},
        controlled_shocks=["e_d", "e_m"],
        method="minimum_energy",
    )

    assert abs(res_cov.forecast["r"].iloc[0] - 0.02) < 1e-10
    assert abs(res_me.forecast["r"].iloc[0] - 0.02) < 1e-10


def test_conditional_forecast_underdetermined_error_handling(nk_model):
    """Verify exact=True raises ValueError on overdetermined/conflicting restrictions."""
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        conditional_forecast(
            nk_model,
            target_paths={"y": [0.01, 0.02], "pi": [0.03, 0.04], "r": [0.05, 0.06]},
            controlled_shocks=["e_m"],
            exact=True,
            horizon=3,
        )


def test_conditional_forecast_least_squares_fallback(nk_model):
    """Verify exact=False emits warning and solves via least squares."""
    with pytest.warns(UserWarning):
        res = conditional_forecast(
            nk_model,
            target_paths={"y": [0.01, 0.02], "pi": [0.03, 0.04], "r": [0.05, 0.06]},
            controlled_shocks=["e_m"],
            exact=False,
            horizon=3,
        )
    assert isinstance(res, ConditionalForecastResult)
    assert len(res.forecast) == 3


def test_conditional_forecast_uninvertible_shock_raises(nk_model):
    """Verify controlled shock with zero impact raises ValueError."""
    # Shock e_m has zero loading on a
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        conditional_forecast(
            nk_model,
            target_paths={"a": [0.05]},
            controlled_shocks=["e_m"],
            exact=True,
        )


def test_conditional_forecast_initial_state_deviation(nk_model):
    """Verify propagation of non-zero initial state x0."""
    x0 = {"r": 0.01, "a": 0.02}
    res = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02, 0.02]},
        controlled_shocks=["e_m"],
        x0=x0,
        horizon=4,
    )
    assert abs(res.forecast["r"].iloc[0] - 0.02) < 1e-10
    assert abs(res.forecast["r"].iloc[1] - 0.02) < 1e-10


def test_conditional_forecast_presentation_contract(nk_model):
    """Verify presentation dataclass contract compliance."""
    res = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02, 0.02]},
        controlled_shocks=["e_m"],
        horizon=4,
        ci=0.90,
    )

    summary = res.summary()
    assert isinstance(summary, str)
    assert "Conditional Forecast" in summary
    assert "Controlled Shocks" in summary

    frame = res.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 4

    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md

    latex = res.to_latex()
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert isinstance(typst, str)
    assert "#table(" in typst

    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_conditional_forecast_immutability(nk_model):
    """Verify ConditionalForecastResult is frozen."""
    res = conditional_forecast(
        nk_model,
        target_paths={"r": [0.02]},
        controlled_shocks=["e_m"],
        horizon=2,
    )
    with pytest.raises(FrozenInstanceError):
        res.forecast = None
