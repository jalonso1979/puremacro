"""Unit tests for 2nd-order DSGE perturbation with pruning (Kim et al. 2008)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    canonical_growth_2nd_order,
    PrunedDSGESolution,
    PrunedSimulationResult,
)


def test_canonical_growth_solution_properties():
    sol = canonical_growth_2nd_order()
    assert isinstance(sol, PrunedDSGESolution)
    assert sol.n_states == 2
    assert sol.n_controls == 1
    assert sol.n_shocks == 1
    assert sol.state_names == ("k", "z")
    assert sol.control_names == ("c",)
    assert sol.shock_names == ("eps",)
    assert sol.is_stable is True

    # Eigenvalues inside unit circle
    eigs = np.abs(sol.eigenvalues)
    assert np.all(eigs < 1.0)


def test_pruned_simulation_stability_and_decomposition():
    sol = canonical_growth_2nd_order()
    sim = sol.simulate(periods=300, seed=42, burn=50)

    assert isinstance(sim, PrunedSimulationResult)
    assert len(sim.states) == 300
    assert len(sim.controls) == 300

    # Test decomposition x = x1 + x2
    diff_x = (sim.states - (sim.states_1st + sim.states_2nd)).to_numpy()
    np.testing.assert_allclose(diff_x, 0.0, atol=1e-12)

    diff_y = (sim.controls - (sim.controls_1st + sim.controls_2nd)).to_numpy()
    np.testing.assert_allclose(diff_y, 0.0, atol=1e-12)

    # DataFrame and summary exports
    frame = sim.to_frame()
    assert list(frame.columns) == ["k", "z", "c"]

    summary = sim.summary()
    assert "Pruned DSGE Simulation Summary" in summary
    assert "Periods Simulated : 300" in summary


def test_pruning_prevents_explosive_trajectories():
    """Verify Kim et al. (2008) core result: raw 2nd order explodes while pruned stays bounded."""
    # Model with unstable quadratic threshold at x = 0.25 (2*(1-0.9)/0.8)
    G = np.array([[0.9]])
    N = np.array([[1.0]])
    F = np.array([[1.0]])
    L = np.array([[0.0]])
    H_xx = np.array([[0.8]])
    H_ss = np.array([0.0])
    G_xx = np.array([[0.0]])
    G_ss = np.array([0.0])

    sol = PrunedDSGESolution(
        G=G, N=N, F=F, L=L,
        H_xx=H_xx, H_sigmasigma=H_ss,
        G_xx=G_xx, G_sigmasigma=G_ss,
        state_names=("x",), control_names=("y",), shock_names=("e",),
    )

    # Shock e_1 = 0.4 pushes state beyond unstable manifold threshold (0.25)
    shocks = np.zeros((100, 1))
    shocks[1] = 0.4

    # Pruned simulation remains stationary and decays back to 0
    sim_pruned = sol.simulate(periods=50, shocks=shocks, burn=0)
    assert np.all(np.isfinite(sim_pruned.states.to_numpy()))
    assert sim_pruned.states["x"].max() < 1.0
    assert sim_pruned.states["x"].iloc[-1] < 0.05

    # Raw unpruned simulation explodes into multi-trillion values
    raw_x, raw_y = sol.simulate_raw(periods=50, shocks=shocks, burn=0)
    has_exploded = np.any(np.isnan(raw_x)) or np.nanmax(np.abs(raw_x)) > 1e6
    assert has_exploded


def test_girf_asymmetry_and_dynamics():
    sol = canonical_growth_2nd_order()

    # Positive vs negative shock. The growth model's consumption response to
    # a persistent TFP shock is hump-shaped (it keeps rising for decades as
    # capital accumulates), so mean reversion is checked at a long horizon.
    girf_pos = sol.girf("eps", size=+2.0, horizon=400)
    girf_neg = sol.girf("eps", size=-2.0, horizon=400)

    assert len(girf_pos) == 401
    assert "k" in girf_pos.columns
    assert "c" in girf_pos.columns

    # In a non-linear 2nd-order model, girf_pos != -girf_neg (asymmetric response)
    sum_paths = girf_pos["c"].to_numpy() + girf_neg["c"].to_numpy()
    # The quadratic curvature ensures that sum_paths is strictly non-zero
    assert np.max(np.abs(sum_paths)) > 1e-5

    # Convergence back toward 0 at long horizon
    assert abs(girf_pos["c"].iloc[-1]) < abs(girf_pos["c"].iloc[1])


def test_stochastic_steady_state_ergodic_mean():
    sol = canonical_growth_2nd_order()
    means = sol.stochastic_steady_state(sigma=1.0)

    assert "states" in means
    assert "controls" in means
    s_means = means["states"]
    c_means = means["controls"]

    assert "k" in s_means
    assert "z" in s_means
    assert "c" in c_means

    # Exogenous technology has zero mean
    assert s_means["z"] == pytest.approx(0.0, abs=1e-12)

    # Precautionary wealth accumulation pushes capital ergodic mean positive
    assert s_means["k"] > 0.0


def test_pruning_validation_errors():
    sol = canonical_growth_2nd_order()

    with pytest.raises(ValueError, match="unknown shock"):
        sol.girf("nonexistent_shock")

    with pytest.raises(ValueError, match="incompatible with total_t"):
        sol.simulate(periods=100, shocks=np.zeros((50, 1)))

    # Explosive G
    bad_G = sol.G * 2.0
    bad_sol = PrunedDSGESolution(
        G=bad_G,
        N=sol.N,
        F=sol.F,
        L=sol.L,
        H_xx=sol.H_xx,
        H_sigmasigma=sol.H_sigmasigma,
        G_xx=sol.G_xx,
        G_sigmasigma=sol.G_sigmasigma,
        state_names=sol.state_names,
        control_names=sol.control_names,
        shock_names=sol.shock_names,
    )
    assert bad_sol.is_stable is False
    with pytest.raises(ValueError, match="eigenvalues outside the unit circle"):
        bad_sol.simulate(periods=50)
