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


def test_simulate_rejects_sigma_with_explicit_shocks():
    """sigma scales the risk correction but never a supplied innovation array.

    Silently accepting both shifted the reported mean by
    (sigma^2 - 1) (I - G)^-1 0.5 H_sigmasigma while leaving the paths' shock
    content untouched, so the combination is refused.
    """
    sol = canonical_growth_2nd_order()
    eps = np.zeros((60, 1))
    eps[1] = 0.01

    with pytest.raises(ValueError, match="cannot be combined with an explicit"):
        sol.simulate(periods=50, shocks=eps, burn=0, sigma=3.0)
    with pytest.raises(ValueError, match="cannot be combined with an explicit"):
        sol.simulate_raw(periods=50, shocks=eps, burn=0, sigma=3.0)

    # sigma = 1.0 (the default) is still accepted with an explicit path.
    sim = sol.simulate(periods=50, shocks=eps, burn=0)
    assert len(sim.states) == 50


def test_stoch_simul_simulated_and_theoretical_means_share_a_scale():
    """Both Mean columns of one report must be levels (Dynare's convention).

    Before the fix the simulated Mean was a deviation from the deterministic
    steady state while the theoretical Mean was a level, so the same row of the
    same report showed e.g. k = -0.015 next to k = 21.44.
    """
    sol = canonical_growth_2nd_order()
    res = sol.stoch_simul(order=2, irf=0, periods=40_000, seed=7, burn=200)

    theo = res.theoretical_moments.moments["Mean"]
    sim = res.simulated_moments["Mean"]
    ss = sol.steady_state

    for v in res.variable_names:
        # Monte-Carlo tolerance: 6 i.i.d. standard errors widened by a factor
        # sqrt(40) for the serial correlation of these very persistent series.
        # On the old (deviation) scale the gap was a whole steady state.
        se = 40.0 * float(res.simulated_moments["Std.Dev."][v]) / np.sqrt(40_000) + 1e-9
        assert abs(float(sim[v]) - float(theo[v])) < se, v

    # k and c have non-trivial steady states, so the shift is not a no-op:
    # before the fix these simulated means were ~0 rather than ~ss.
    for v in ("k", "c"):
        assert abs(float(ss[v])) > 0.5
        assert abs(float(sim[v]) - float(ss[v])) < 0.1 * abs(float(ss[v]))


def test_stoch_simul_higher_moments_are_shift_invariant():
    """Only the Mean column moved: the dispersion moments are unchanged."""
    sol = canonical_growth_2nd_order()
    res = sol.stoch_simul(order=2, irf=0, periods=2_000, seed=3, burn=100)
    sim_dev = sol.simulate(periods=2_000, seed=3, burn=100)
    dev = pd.concat([sim_dev.states, sim_dev.controls], axis=1)[list(res.variable_names)]

    for col, expected in (
        ("Std.Dev.", dev.std(axis=0)),
        ("Variance", dev.var(axis=0)),
        ("Skewness", dev.skew(axis=0)),
        ("Kurtosis", dev.kurtosis(axis=0)),
    ):
        np.testing.assert_allclose(
            res.simulated_moments[col].to_numpy(),
            expected.reindex(list(res.variable_names)).to_numpy(),
            rtol=0.0, atol=0.0, err_msg=col,
        )


def test_girf_is_independent_of_sigma():
    """The risk correction cancels between the shocked and baseline paths.

    Pins the documented behaviour of the ``sigma`` keyword of girf/irf: it is
    a signature-parity no-op, not a risk-adjustment knob.
    """
    sol = canonical_growth_2nd_order()
    # 0.5 * ghs2 * 1000**2 would be O(50) on k if it did not cancel.
    assert np.abs(sol.H_sigmasigma).max() > 0.0
    base = sol.girf("eps", size=0.01, horizon=6, sigma=0.0)
    for s in (1.0, 10.0, 1000.0):
        np.testing.assert_allclose(
            sol.girf("eps", size=0.01, horizon=6, sigma=s).to_numpy(),
            base.to_numpy(),
            rtol=1e-7, atol=1e-12,
        )
