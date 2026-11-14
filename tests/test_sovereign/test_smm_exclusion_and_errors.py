"""Tests for sovereign SMM exclusion/reentry stationary moments and error handling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from puremacro.sovereign import FiscalSovereignResult, solve_fiscal_sovereign_default
from puremacro.sovereign._fiscal_paths import MOMENTS, draw_uniforms, ergodic_moments, simulate_paths
from puremacro.sovereign.smm import (
    _compute_stationary_or_simulated_moments,
    _stationary_access_exclusion_dist,
    calibrate_fiscal_sovereign_smm,
)
from puremacro.structural.bridge import MomentTargets


def _make_toy_fiscal_result(
    *,
    theta: float = 0.282,
    p_def_mid_low: float = 0.25,
    p_trans: np.ndarray | None = None,
    policy_index: np.ndarray | None = None,
    default_prob: np.ndarray | None = None,
    spread: np.ndarray | None = None,
    g_policy: np.ndarray | None = None,
    g_default: np.ndarray | None = None,
    primary_balance: np.ndarray | None = None,
    debt_to_gdp: np.ndarray | None = None,
    y_default_vec: np.ndarray | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> FiscalSovereignResult:
    """Build a self-consistent 3x2 toy FiscalSovereignResult for exact tests."""
    b_grid = np.array([-0.2, 0.0, 0.2], dtype=float)
    y_grid = np.array([0.8, 1.2], dtype=float)
    n_b, n_y = len(b_grid), len(y_grid)

    if p_trans is None:
        p_trans = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    if policy_index is None:
        # b=-0.2 -> b=0; b=0,y_low -> b=-0.2 (asset); b=0,y_high -> b=0.2 (boundary); b=0.2 -> b=0
        policy_index = np.array([[1, 1], [0, 2], [1, 1]], dtype=int)
    if default_prob is None:
        default_prob = np.array([[0.0, 0.0], [p_def_mid_low, 0.0], [0.0, 0.0]], dtype=float)
    if spread is None:
        spread = np.array([[0.005, 0.002], [0.015, 0.006], [0.030, 0.012]], dtype=float)
    if g_policy is None:
        g_policy = np.array([[0.12, 0.24], [0.14, 0.28], [0.10, 0.20]], dtype=float)
    if g_default is None:
        g_default = np.array([0.08, 0.12], dtype=float)
    if primary_balance is None:
        primary_balance = np.array([[0.04, 0.01], [0.01, -0.02], [0.05, 0.02]], dtype=float)
    if debt_to_gdp is None:
        debt_to_gdp = b_grid[:, None] / y_grid[None, :]
    if y_default_vec is None:
        y_default_vec = np.array([0.75, 1.00], dtype=float)

    policy_b = b_grid[policy_index]
    q_bond = 0.98 / (1.0 + np.where(np.isfinite(spread), np.maximum(spread, 0.0), 0.0))
    metadata: dict[str, Any] = {
        "theta": float(theta),
        "zero_index": 1,
        "y_default_vec": y_default_vec,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return FiscalSovereignResult(
        b_grid=b_grid,
        y_grid=y_grid,
        q_bond=q_bond,
        bond_price=q_bond,
        default_prob=default_prob,
        policy_b=policy_b,
        policy_b_prime=policy_b,
        spread=spread,
        p_trans=p_trans,
        converged=True,
        tau=0.20,
        psi=1.0,
        policy_index=policy_index,
        g_policy=g_policy,
        g_default=g_default,
        primary_balance=primary_balance,
        debt_to_gdp=debt_to_gdp,
        private_consumption_repay=0.7 * y_grid[None, :] - g_policy,
        private_consumption_default=y_default_vec - g_default,
        metadata=metadata,
    )


def _analytical_toy_weights(p_def_mid_low: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form stationary (w_acc, w_exc) for the default 3x2 toy chain (theta > 0)."""
    a1 = 1.0 / (2.0 + 0.5 * p_def_mid_low * ((1.0 / theta) - 2.0))
    a0 = 0.5 * (1.0 - p_def_mid_low) * a1
    a2 = 0.5 * a1
    d_mass = 0.5 * p_def_mid_low * a1
    e_mass = ((1.0 - theta) / theta) * d_mass
    w_acc = np.array(
        [
            [0.5 * a0, 0.5 * a0],
            [0.5 * a1, 0.5 * a1],
            [0.5 * a2, 0.5 * a2],
        ],
        dtype=float,
    )
    w_exc = np.array([0.5 * e_mass, 0.5 * e_mass], dtype=float)
    return w_acc, w_exc


def _solve_fast_equilibrium(theta: float, sigma_eps: float) -> FiscalSovereignResult:
    """Solve a compact fiscal sovereign equilibrium for Monte Carlo comparisons."""
    return solve_fiscal_sovereign_default(
        tau=0.20,
        psi=1.0,
        theta=theta,
        n_b=21,
        n_y=5,
        sigma_eps=sigma_eps,
        policy_search="dense" if sigma_eps > 0.0 else "monotone",
        tol=1e-8,
        max_iter=2000,
    )


def _monte_carlo_mean_and_se(
    sol: FiscalSovereignResult,
    labels: tuple[str, ...],
    *,
    n_seeds: int = 16,
    periods: int = 20_000,
    burn: int = 2_000,
) -> tuple[dict[str, float], dict[str, float]]:
    """Simulate Monte Carlo long-run means and cross-seed standard errors via _fiscal_paths."""
    seeds = list(range(4100, 4100 + n_seeds))
    uniforms = draw_uniforms(seeds, periods)
    zero_idx = int(sol.metadata["zero_index"])
    y0_idx = len(sol.y_grid) // 2
    paths = simulate_paths(sol, zero_idx, y0_idx, uniforms)
    per_path = ergodic_moments(paths, burn=burn)
    means: dict[str, float] = {}
    ses: dict[str, float] = {}
    for lbl in labels:
        vals = np.array([float(d[lbl]) for d in per_path], dtype=float)
        assert np.all(np.isfinite(vals)), f"Monte Carlo {lbl} nonfinite for {per_path}"
        means[lbl] = float(np.mean(vals))
        ses[lbl] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return means, ses


# ---------------------------------------------------------------------------
# (a) Nonfinite moment rejection (no silent 0.0 or dropping bad seeds)
# ---------------------------------------------------------------------------


def test_simulation_all_seeds_absorbing_exclusion_raises_not_zero() -> None:
    """Permanent exclusion (theta=0) after sure default leaves zero post-burn repayment paths."""
    p_def = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=float)
    sol = _make_toy_fiscal_result(theta=0.0, default_prob=p_def)

    with pytest.raises(ValueError, match="nonfinite"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("debt_pct", "spread_mean_pct", "corr_pb_y"),
            use_simulation=True,
            seeds=(1, 2, 3),
            periods=80,
            burn=20,
        )


def test_simulation_partial_nonfinite_seeds_raises_instead_of_dropping() -> None:
    """If even one seed produces NaN/Inf moments, raise rather than dropping that seed."""
    # Draw pattern with theta=0: some seeds enter default during burn-in, others never default.
    p_def = np.array([[0.0, 0.0], [0.02, 0.0], [0.0, 0.0]], dtype=float)
    sol = _make_toy_fiscal_result(theta=0.0, default_prob=p_def)
    seeds = list(range(10, 18))
    u = draw_uniforms(seeds, 120)
    paths = simulate_paths(sol, 1, 1, u)
    per_path = ergodic_moments(paths, burn=60)
    finite_flags = [np.isfinite(d["debt_pct"]) for d in per_path]
    # Confirm we genuinely hit a mixed case: some finite paths, some NaN paths.
    assert any(finite_flags) and not all(finite_flags)

    with pytest.raises(ValueError, match=r"seed\(s\).*nonfinite|is nonfinite"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("debt_pct",),
            use_simulation=True,
            seeds=seeds,
            periods=120,
            burn=60,
        )


def test_simulation_nonfinite_spread_raises_not_zero() -> None:
    """NaN spread visited on equilibrium paths must raise ValueError in simulation mode."""
    bad_spread = np.array([[0.005, 0.002], [np.nan, 0.006], [0.030, 0.012]], dtype=float)
    sol = _make_toy_fiscal_result(theta=0.5, p_def_mid_low=0.1, spread=bad_spread)

    with pytest.raises(ValueError, match="nonfinite"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("spread_mean_pct",),
            use_simulation=True,
            seeds=(101, 102, 103),
            periods=200,
            burn=20,
        )


@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_stationary_zero_repayment_mass_raises(theta: float) -> None:
    """Zero stationary repayment mass rejects repayment moments instead of uniform/zero fallback."""
    # Every access state defaults with probability 1 -> w_rep == 0 everywhere.
    p_def = np.ones((3, 2), dtype=float)
    sol = _make_toy_fiscal_result(theta=theta, default_prob=p_def)

    with pytest.raises(ValueError, match="repayment probability mass is zero"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("debt_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )


def test_stationary_zero_access_mass_at_theta_zero_raises_annual_default() -> None:
    """At theta=0 with positive default hazard, market-access mass drains to zero."""
    p_def = np.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]], dtype=float)
    sol = _make_toy_fiscal_result(theta=0.0, default_prob=p_def)

    with pytest.raises(ValueError, match="market-access probability mass is zero"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("annual_default_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )


@pytest.mark.parametrize(
    ("field_name", "moment_label"),
    [
        ("debt_to_gdp", "debt_pct"),
        ("spread", "spread_mean_pct"),
        ("g_policy", "spending_pct"),
        ("g_default", "spending_pct"),
        ("primary_balance", "corr_pb_y"),
    ],
)
def test_stationary_nonfinite_model_arrays_raise(field_name: str, moment_label: str) -> None:
    """NaN/Inf entries on positive-weight stationary states raise ValueError cleanly."""
    sol = _make_toy_fiscal_result(theta=0.4, p_def_mid_low=0.3)
    bad = np.array(getattr(sol, field_name), dtype=float, copy=True)
    if bad.ndim == 1:
        bad[0] = np.nan
    else:
        # (1, 1) is positive-weight repayment; policy_index[1, 0] == 0 so spread[0, 0] is chosen.
        bad[1, 1] = np.inf
        bad[0, 0] = np.nan
    sol_bad = replace(sol, **{field_name: bad})

    with pytest.raises(ValueError, match="Nonfinite values"):
        _compute_stationary_or_simulated_moments(
            sol_bad,
            (moment_label,),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )


def test_stationary_zero_variance_corr_pb_y_raises_not_zero() -> None:
    """Constant primary_balance / y has zero variance and must raise ValueError, not return 0.0."""
    y_grid = np.array([0.8, 1.2], dtype=float)
    const_pb = np.broadcast_to(0.05 * y_grid[None, :], (3, 2)).copy()
    sol = _make_toy_fiscal_result(theta=0.5, primary_balance=const_pb)

    with pytest.raises(ValueError, match="corr_pb_y"):
        _compute_stationary_or_simulated_moments(
            sol,
            ("corr_pb_y",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )


def test_stationary_nonconvergence_raises_runtime_error() -> None:
    """Exhausting max_iter in _stationary_access_exclusion_dist raises RuntimeError."""
    sol = _make_toy_fiscal_result(theta=0.2, p_def_mid_low=0.3)
    with pytest.raises(RuntimeError, match="did not converge"):
        _stationary_access_exclusion_dist(
            sol.default_prob,
            sol.policy_index,
            sol.p_trans,
            theta=0.2,
            zero_idx=1,
            tol=1e-14,
            max_iter=2,
        )


def test_calibrate_rejects_unconverged_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """calibrate_fiscal_sovereign_smm must raise RuntimeError when the inner solver does not converge."""
    unconverged = replace(_make_toy_fiscal_result(theta=0.282), converged=False, residual=0.12, iterations=2000)

    def _fake_solver(**_: Any) -> FiscalSovereignResult:
        return unconverged

    monkeypatch.setattr("puremacro.sovereign.smm.solve_fiscal_sovereign_default", _fake_solver)
    targets = MomentTargets(
        values=np.array([5.0]),
        covariance=np.array([[1.0]]),
        labels=("debt_pct",),
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        calibrate_fiscal_sovereign_smm(targets, max_nfev=2)


# ---------------------------------------------------------------------------
# (b) Stationary (access/excluded, b, y) Markov vs analytical & _fiscal_paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.15, 0.282, 0.5, 1.0])
def test_analytical_stationary_exclusion_weights_and_all_moments(theta: float) -> None:
    """Stationary (w_acc, w_exc) and all 7 moments match exact closed form and Monte Carlo."""
    p_def_mid_low = 0.30
    sol = _make_toy_fiscal_result(theta=theta, p_def_mid_low=p_def_mid_low)

    w_acc, w_exc = _stationary_access_exclusion_dist(
        sol.default_prob,
        sol.policy_index,
        sol.p_trans,
        theta=theta,
        zero_idx=1,
    )
    exp_acc, exp_exc = _analytical_toy_weights(p_def_mid_low, theta)
    np.testing.assert_allclose(w_acc, exp_acc, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(w_exc, exp_exc, rtol=1e-11, atol=1e-12)
    assert np.isclose(np.sum(w_acc) + np.sum(w_exc), 1.0, atol=1e-12)

    if theta == 1.0:
        np.testing.assert_allclose(w_exc, 0.0, atol=1e-13)
    else:
        default_flow = float(np.sum(w_acc * sol.default_prob))
        np.testing.assert_allclose(np.sum(w_exc), ((1.0 - theta) / theta) * default_flow, atol=1e-12)

    # Exact closed-form moments from analytical weights:
    w_rep = exp_acc * (1.0 - sol.default_prob)
    w_rep_norm = w_rep / np.sum(w_rep)
    y_mat = np.broadcast_to(sol.y_grid[None, :], sol.default_prob.shape)
    y_idx = np.broadcast_to(np.arange(2)[None, :], sol.default_prob.shape)
    chosen_spread = sol.spread[sol.policy_index, y_idx]
    pb_ratio = sol.primary_balance / y_mat
    ly = np.log(y_mat)
    mean_pb = float(np.sum(w_rep_norm * pb_ratio))
    mean_ly = float(np.sum(w_rep_norm * ly))
    expected = {
        # Note signed debt_to_gdp: b=-0.2 contributes negatively (no max(..., 0) clipping).
        "debt_pct": 100.0 * float(np.sum(w_rep_norm * sol.debt_to_gdp)),
        "spread_mean_pct": 100.0 * float(np.sum(w_rep_norm * chosen_spread)),
        "spread_std_pp": float(
            np.sqrt(np.sum(w_rep_norm * (100.0 * chosen_spread - 100.0 * float(np.sum(w_rep_norm * chosen_spread))) ** 2))
        ),
        "annual_default_pct": 400.0 * float(np.sum(exp_acc * sol.default_prob)) / float(np.sum(exp_acc)),
        "spending_pct": 100.0
        * (
            float(np.sum(w_rep * (sol.g_policy / y_mat)))
            + float(np.sum((exp_exc + np.sum(exp_acc * sol.default_prob, axis=0)) * (sol.g_default / sol.metadata["y_default_vec"])))
        ),
        "corr_pb_y": float(
            np.sum(w_rep_norm * (pb_ratio - mean_pb) * (ly - mean_ly))
            / np.sqrt(np.sum(w_rep_norm * (pb_ratio - mean_pb) ** 2) * np.sum(w_rep_norm * (ly - mean_ly) ** 2))
        ),
        "boundary_choice_share_pct": 100.0 * float(np.sum(w_rep_norm * (sol.policy_index == 2))),
    }

    got = _compute_stationary_or_simulated_moments(
        sol,
        MOMENTS,
        use_simulation=False,
        seeds=(1,),
        periods=10,
        burn=0,
    )
    for lbl, val in zip(MOMENTS, got):
        assert np.isclose(val, expected[lbl], rtol=1e-10, atol=1e-10), f"{lbl}: {val} vs {expected[lbl]}"

    # Compare stationary moments against _fiscal_paths Monte Carlo long-run frequencies.
    mc_means, mc_ses = _monte_carlo_mean_and_se(sol, MOMENTS, n_seeds=12, periods=15_000, burn=1_500)
    for lbl, val in zip(MOMENTS, got):
        tol = max(4.0 * mc_ses[lbl], 0.05)
        assert abs(val - mc_means[lbl]) <= tol, (
            f"theta={theta}, {lbl}: stationary={val:.4f}, mc={mc_means[lbl]:.4f} (se={mc_ses[lbl]:.4f})"
        )


def test_exclusion_state_fixes_unconditional_reentry_bias() -> None:
    """For theta < 1, explicit w_exc fixes the bias caused by unconditional zero_idx reentry."""
    theta = 0.2
    p_def = np.array([[0.0, 0.0], [0.55, 0.10], [0.05, 0.0]], dtype=float)
    p_trans = np.array([[0.90, 0.10], [0.10, 0.90]], dtype=float)
    sol = _make_toy_fiscal_result(
        theta=theta,
        default_prob=p_def,
        p_trans=p_trans,
        g_policy=np.array([[0.24, 0.36], [0.24, 0.36], [0.24, 0.36]], dtype=float),
        g_default=np.array([0.075, 0.10], dtype=float),
    )

    w_acc, w_exc = _stationary_access_exclusion_dist(p_def, sol.policy_index, p_trans, theta=theta, zero_idx=1)
    assert np.sum(w_exc) > 0.30
    assert np.sum(w_acc) < 0.70

    # Compare explicit (w_acc, w_exc) moments against _fiscal_paths Monte Carlo.
    got = _compute_stationary_or_simulated_moments(
        sol,
        ("annual_default_pct", "spending_pct", "debt_pct"),
        use_simulation=False,
        seeds=(1,),
        periods=10,
        burn=0,
    )
    mc_means, mc_ses = _monte_carlo_mean_and_se(
        sol,
        ("annual_default_pct", "spending_pct", "debt_pct"),
        n_seeds=12,
        periods=20_000,
        burn=2_000,
    )
    for idx, lbl in enumerate(("annual_default_pct", "spending_pct", "debt_pct")):
        assert abs(got[idx] - mc_means[lbl]) <= max(4.0 * mc_ses[lbl], 0.1)

    # Naive unconditional zero_idx reentry sets w_exc=0 (equivalent to theta=1 in transition),
    # severely distorting default rate and spending_pct relative to Monte Carlo at theta=0.2.
    naive_theta1 = _compute_stationary_or_simulated_moments(
        replace(sol, metadata={**sol.metadata, "theta": 1.0}),
        ("annual_default_pct", "spending_pct"),
        use_simulation=False,
        seeds=(1,),
        periods=10,
        burn=0,
    )
    assert abs(naive_theta1[0] - mc_means["annual_default_pct"]) > 5.0
    assert abs(naive_theta1[1] - mc_means["spending_pct"]) > 5.0


@pytest.mark.parametrize("sigma_eps", [0.0, 0.005])
@pytest.mark.parametrize("theta", [0.15, 0.282, 0.5, 1.0])
def test_solved_equilibrium_stationary_matches_fiscal_paths_mc(theta: float, sigma_eps: float) -> None:
    """Solved equilibria (discrete & smoothed) match _fiscal_paths Monte Carlo across theta."""
    sol = _solve_fast_equilibrium(theta=theta, sigma_eps=sigma_eps)
    stationary_vals = _compute_stationary_or_simulated_moments(
        sol,
        MOMENTS,
        use_simulation=False,
        seeds=(1,),
        periods=10,
        burn=0,
    )
    mc_means, mc_ses = _monte_carlo_mean_and_se(sol, MOMENTS, n_seeds=12, periods=15_000, burn=1_500)

    assert np.all(np.isfinite(stationary_vals))
    for lbl, stat_val in zip(MOMENTS, stationary_vals):
        # Guard both Monte Carlo sampling variance and near-zero deterministic moments.
        tol = max(4.5 * mc_ses[lbl], 0.10 if lbl != "corr_pb_y" else 0.02)
        assert abs(stat_val - mc_means[lbl]) <= tol, (
            f"theta={theta}, sigma_eps={sigma_eps}, {lbl}: "
            f"stationary={stat_val:.4f}, mc={mc_means[lbl]:.4f}, se={mc_ses[lbl]:.4f}"
        )


def test_theta_zero_reachable_safe_component_and_optional_metadata_compat() -> None:
    """theta=0 stays finite if reachable states never default, and zero_index/y_default_vec stay optional."""
    # From (zero_idx=1, y0=1), chain moves between (1, 1) and (2, 1) where p_def == 0,
    # even though unused state (0, 0) has positive p_def.
    p_def = np.array([[0.8, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    p_trans = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    policy_index = np.array([[1, 1], [1, 2], [1, 1]], dtype=int)
    sol = _make_toy_fiscal_result(
        theta=0.0,
        p_trans=p_trans,
        policy_index=policy_index,
        default_prob=p_def,
    )
    # Remove optional stationary metadata keys when spending_pct is not requested.
    sol_minimal_meta = replace(sol, metadata={"theta": 0.0})
    vals = _compute_stationary_or_simulated_moments(
        sol_minimal_meta,
        ("debt_pct", "spread_mean_pct", "annual_default_pct"),
        use_simulation=False,
        seeds=(1,),
        periods=10,
        burn=0,
    )
    np.testing.assert_allclose(vals, [100.0 * (0.5 * 0.2 / 1.2), 100.0 * 0.009, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# (c) Invalid inputs validation
# ---------------------------------------------------------------------------


def test_compute_moments_rejects_invalid_solution_and_labels() -> None:
    """Validate strict type/shape/metadata/label errors on _compute_stationary_or_simulated_moments."""
    sol = _make_toy_fiscal_result(theta=0.3)

    with pytest.raises(TypeError, match="FiscalSovereignResult"):
        _compute_stationary_or_simulated_moments(
            object(),  # type: ignore[arg-type]
            ("debt_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )

    for bad_labels in ("debt_pct", (), ("",), ("unknown_moment",)):
        with pytest.raises(ValueError):
            _compute_stationary_or_simulated_moments(
                sol,
                bad_labels,  # type: ignore[arg-type]
                use_simulation=False,
                seeds=(1,),
                periods=10,
                burn=0,
            )

    sol_smoothed_debt = replace(
        sol,
        metadata={**sol.metadata, "debt_choice_probabilities": np.ones((3, 2, 3)) / 3.0},
    )
    with pytest.raises(ValueError, match="debt_choice_probabilities"):
        _compute_stationary_or_simulated_moments(
            sol_smoothed_debt, ("debt_pct",), use_simulation=False, seeds=(1,), periods=10, burn=0
        )

    with pytest.raises(ValueError, match="policy_index"):
        _compute_stationary_or_simulated_moments(
            replace(sol, policy_index=np.array([[0, 3], [0, 1], [1, 1]], dtype=int)),
            ("debt_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )

    with pytest.raises(ValueError, match="policy_b is inconsistent"):
        _compute_stationary_or_simulated_moments(
            replace(sol, policy_b=np.zeros((3, 2))),
            ("debt_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )

    with pytest.raises(ValueError, match="p_trans rows"):
        _compute_stationary_or_simulated_moments(
            replace(sol, p_trans=np.array([[0.5, 0.2], [0.5, 0.5]])),
            ("debt_pct",),
            use_simulation=False,
            seeds=(1,),
            periods=10,
            burn=0,
        )

    for bad_meta in (
        {"zero_index": 1},  # missing theta
        {"theta": True, "zero_index": 1},
        {"theta": 1.2, "zero_index": 1},
        {"theta": -0.1, "zero_index": 1},
        {"theta": 0.3, "zero_index": 9},
        {"theta": 0.3, "zero_index": True},
        {"theta": 0.3, "zero_index": 1, "y_default_vec": np.array([-0.5, 1.0])},
    ):
        with pytest.raises(ValueError):
            _compute_stationary_or_simulated_moments(
                replace(sol, metadata=bad_meta),
                ("spending_pct",),
                use_simulation=False,
                seeds=(1,),
                periods=10,
                burn=0,
            )

    # Simulation mode requires zero_index and y_default_vec in metadata.
    with pytest.raises(ValueError, match="zero_index"):
        _compute_stationary_or_simulated_moments(
            replace(sol, metadata={"theta": 0.3, "y_default_vec": np.array([0.75, 1.0])}),
            ("debt_pct",),
            use_simulation=True,
            seeds=(1,),
            periods=20,
            burn=5,
        )


def test_simulation_and_calibration_reject_invalid_arguments() -> None:
    """Invalid simulation args, targets, parameters, and options fail loudly with clear errors."""
    sol = _make_toy_fiscal_result(theta=0.3)

    for bad_sim_kwargs in (
        {"seeds": (), "periods": 20, "burn": 5},
        {"seeds": (-1,), "periods": 20, "burn": 5},
        {"seeds": (True,), "periods": 20, "burn": 5},
        {"seeds": (1,), "periods": 0, "burn": 0},
        {"seeds": (1,), "periods": 10, "burn": 10},
        {"seeds": (1,), "periods": 10, "burn": -1},
    ):
        with pytest.raises(ValueError):
            _compute_stationary_or_simulated_moments(
                sol,
                ("debt_pct",),
                use_simulation=True,
                **bad_sim_kwargs,  # type: ignore[arg-type]
            )

    valid_targets = MomentTargets(
        values=np.array([10.0]),
        covariance=np.array([[1.0]]),
        labels=("debt_pct",),
    )

    with pytest.raises(ValueError, match="MomentTargets"):
        calibrate_fiscal_sovereign_smm("not_targets")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported moment label"):
        calibrate_fiscal_sovereign_smm(
            MomentTargets(values=np.array([1.0]), covariance=np.array([[1.0]]), labels=("bogus",))
        )

    for bad_kw in (
        {"param_names": "beta", "theta0": (0.95,)},
        {"param_names": ("beta", "beta"), "theta0": (0.95, 0.95)},
        {"param_names": ("not_a_solver_param",), "theta0": (1.0,)},
        {"param_names": ("beta", "psi"), "theta0": (0.95,)},
        {"param_names": ("beta",), "theta0": (np.nan,)},
        {"param_names": ("beta",), "theta0": (0.95,), "n_b": 1},
        {"param_names": ("beta",), "theta0": (0.95,), "n_y": True},
        {"param_names": ("beta",), "theta0": (0.95,), "use_simulation": "no"},  # type: ignore[dict-item]
        {"param_names": ("beta",), "theta0": (0.95,), "use_simulation": True, "seeds": ()},
    ):
        with pytest.raises(ValueError):
            calibrate_fiscal_sovereign_smm(valid_targets, bounds=None, **bad_kw)

    with pytest.raises(TypeError, match="FiscalRule"):
        calibrate_fiscal_sovereign_smm(
            valid_targets, param_names=("beta",), theta0=(0.95,), bounds=None, rule="bad"  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="solver_kwargs"):
        calibrate_fiscal_sovereign_smm(
            valid_targets,
            param_names=("beta",),
            theta0=(0.95,),
            bounds=None,
            solver_kwargs=["not_a_mapping"],  # type: ignore[arg-type]
        )
