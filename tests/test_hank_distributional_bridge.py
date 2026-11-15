"""Tests for Empirical Distributional LP <-> One-Asset HANK Incidence Bridge.

Covers:
1. Exact cross-sectional adding-up & accounting conservation (< 1e-10) across
   fiscal closures (balanced_budget, debt_rule, proportional, lump_sum), monetary
   shocks, linear SequenceSpaceHANKResult, nonlinear transitions, and transfers.
2. Fail-closed validation (strict conservation audit, unconverged models,
   incompatible group counts, missing horizon overlap, NaN/Inf/negative SEs,
   non-partitioning weights, invalid tracking/lang).
3. Zero-shock & Ricardian / transfer incidence checks (lump_sum vs proportional
   and cash-on-hand transfers strictly raise low-wealth / HtM consumption more).
4. Empirical scoring against ``estimate_distributional_lp`` (shares) and
   ``estimate_quantile_lp`` (quantiles) plus ``EmpiricalGroupTargets`` / DataFrames:
   matching HANK closure beats mismatched-sign baseline on RMSE/MAE/concordance.
5. Steady-state MPC/HtM calibration metadata, ``SolverDiagnosticRecord`` regime
   checks (interior PASS vs truncated UNSUPPORTED_REGIME), bilingual EN/ES
   outputs (``.summary()``, ``.to_markdown()``, ``.to_dict()``), and Pyodide safety.
"""
from __future__ import annotations

import ast
import dataclasses
from importlib.abc import MetaPathFinder
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from puremacro.lp.distributional import (
    estimate_distributional_lp,
    estimate_quantile_lp,
)
from puremacro.models.hank_distributional_bridge import (
    EmpiricalGroupTargets,
    HANKDistributionalBridgeResult,
    HANKGroupIRFResult,
    bridge_hank_distributional,
    extract_hank_group_irf,
    score_hank_distributional,
    simulate_transfer_group_irf,
    solve_hank_group_irf,
)
from puremacro.models.hank_fiscal import solve_hank_fiscal
from puremacro.models.hank_sequence_space import (
    solve_hank_sequence_space,
    solve_nonlinear_transition,
)


@pytest.fixture(scope="module")
def hank_ss() -> Any:
    """Shared converged one-asset HANK steady state with safe upper asset bound."""
    return solve_hank_sequence_space(
        T=40,
        beta=0.985,
        gamma=1.0,
        r_ss=0.01,
        phi_pi=1.5,
        kappa=0.1,
        shock_magnitude=0.0,
        shock_rho=0.0,
        n_a=100,
        a_max=100.0,
    )


@pytest.fixture(scope="module")
def fiscal_lumpsum_taxcut(hank_ss: Any) -> Any:
    """Balanced-budget lump-sum tax cut (negative G shock) generating positive transfers."""
    T = 40
    shock_seq = -0.02 * (0.7 ** np.arange(T))
    return solve_hank_fiscal(
        ss_model=hank_ss,
        shock_seq=shock_seq,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=T,
        tol=1e-8,
    )


@pytest.fixture(scope="module")
def fiscal_prop_taxcut(hank_ss: Any) -> Any:
    """Balanced-budget proportional tax cut under the identical aggregate budget path."""
    T = 40
    shock_seq = -0.02 * (0.7 ** np.arange(T))
    return solve_hank_fiscal(
        ss_model=hank_ss,
        shock_seq=shock_seq,
        closure="balanced_budget",
        tax_type="proportional",
        horizon=T,
        tol=1e-8,
    )


# ---------------------------------------------------------------------------
# 1. Exact cross-sectional conservation & adding-up (< 1e-10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "closure,tax_type,use_monetary",
    [
        ("balanced_budget", "proportional", False),
        ("balanced_budget", "lump_sum", False),
        ("debt_rule", "proportional", False),
        ("debt_rule", "lump_sum", False),
        ("balanced_budget", "lump_sum", True),
    ],
)
def test_exact_conservation_across_fiscal_and_monetary_closures(
    hank_ss: Any,
    closure: str,
    tax_type: str,
    use_monetary: bool,
) -> None:
    T = 40
    shock_seq: np.ndarray
    mon_seq: np.ndarray | None
    if use_monetary:
        shock_seq = np.zeros(T, dtype=np.float64)
        mon_seq = np.asarray(-0.004 * (0.75 ** np.arange(T)), dtype=np.float64)
    else:
        shock_seq = np.asarray(0.02 * (0.7 ** np.arange(T)), dtype=np.float64)
        mon_seq = None

    res = solve_hank_fiscal(
        ss_model=hank_ss,
        shock_seq=shock_seq,
        monetary_shock_seq=mon_seq,
        closure=closure,
        tax_type=tax_type,
        horizon=T,
        phi_B=0.08,
        tol=1e-8,
    )
    girf = extract_hank_group_irf(res, groups="quintiles", tracking="cohort", conservation_tol=1e-10)

    assert isinstance(girf, HANKGroupIRFResult)
    assert girf.consumption_irf.shape == (T, 5)
    assert girf.asset_irf.shape == (T, 5)
    assert girf.solver_diagnostics.overall_status == "PASS"
    assert girf.convergence_status.converged is True
    assert girf.accuracy_status.is_accurate is True
    assert girf.support_status.is_supported is True

    # Exact population-weighted adding-up against aggregate HANK dC and dA
    b_ss = float(res.steady_state.get("B_ss", res.steady_state_model.government_debt))
    res_da = np.asarray(res.A, dtype=np.float64) - b_ss
    dc_agg_diff = np.max(np.abs(girf.consumption_irf @ girf.group_masses - res.dC))
    da_agg_diff = np.max(np.abs(girf.asset_irf @ girf.group_masses - res_da))
    dshare_c_sum = np.max(np.abs(np.sum(girf.consumption_share_irf, axis=1)))
    dshare_a_sum = np.max(np.abs(np.sum(girf.asset_share_irf, axis=1)))
    mass_sum_err = np.max(np.abs(np.sum(girf.group_masses_path, axis=1) - 1.0))

    assert dc_agg_diff < 1e-10
    assert da_agg_diff < 1e-10
    assert dshare_c_sum < 1e-10
    assert dshare_a_sum < 1e-10
    assert mass_sum_err < 1e-10
    assert girf.conservation_audit["consumption_adding_up_error"] < 1e-10
    assert girf.conservation_audit["consumption_share_adding_up_error"] < 1e-10


@pytest.mark.parametrize(
    "groups,expected_g",
    [
        ("quintiles", 5),
        ("deciles", 10),
        ("quartiles", 4),
        ("terciles", 3),
        ("htm_vs_unconstrained", 2),
        ("income", 2),
        ("wealth_x_income", 4),
        (4, 4),
        ([0.0, 0.1, 0.5, 0.9, 1.0], 4),
    ],
)
def test_group_partition_schemes_and_conservation(
    fiscal_lumpsum_taxcut: Any,
    groups: Any,
    expected_g: int,
) -> None:
    girf = extract_hank_group_irf(
        fiscal_lumpsum_taxcut,
        groups=groups,
        tracking="cohort",
        conservation_tol=1e-10,
    )
    assert len(girf.groups) == expected_g
    assert girf.consumption_irf.shape == (40, expected_g)
    assert np.isclose(np.sum(girf.group_masses), 1.0, atol=1e-12)
    assert girf.conservation_audit["consumption_adding_up_error"] < 1e-10
    assert girf.conservation_audit["consumption_share_adding_up_error"] < 1e-10
    assert girf.conservation_audit["asset_adding_up_error"] < 1e-10


def test_cross_section_and_fixed_bins_tracking_conservation(fiscal_lumpsum_taxcut: Any) -> None:
    for mode in ("cross_section", "fixed_bins"):
        girf = extract_hank_group_irf(
            fiscal_lumpsum_taxcut,
            groups="quintiles",
            tracking=mode,
            conservation_tol=1e-10,
        )
        assert girf.tracking == mode
        assert girf.conservation_audit["consumption_adding_up_error"] < 1e-10
        assert girf.conservation_audit["consumption_share_adding_up_error"] < 1e-10
        assert girf.conservation_audit["asset_adding_up_error"] < 1e-10


def test_linear_and_nonlinear_sequence_space_and_transfer_conservation() -> None:
    ss_lin = solve_hank_sequence_space(
        T=35,
        beta=0.985,
        gamma=1.0,
        r_ss=0.01,
        phi_pi=1.5,
        kappa=0.1,
        shock_magnitude=-0.004,
        shock_rho=0.75,
        n_a=100,
        a_max=100.0,
    )
    girf_lin = extract_hank_group_irf(ss_lin, groups="quintiles", tracking="cohort")
    assert girf_lin.model_kind == "sequence_space_linear"
    assert girf_lin.conservation_audit["consumption_adding_up_error"] < 1e-10
    assert girf_lin.conservation_audit["consumption_share_adding_up_error"] < 1e-10
    assert girf_lin.conservation_audit["asset_adding_up_error"] < 1e-10

    girf_lin_fb = extract_hank_group_irf(ss_lin, groups="quintiles", tracking="fixed_bins")
    assert girf_lin_fb.conservation_audit["consumption_adding_up_error"] < 1e-10
    assert girf_lin_fb.conservation_audit["consumption_share_adding_up_error"] < 1e-10
    assert girf_lin_fb.conservation_audit["asset_adding_up_error"] < 1e-10

    ss_nl = solve_nonlinear_transition(
        T=35,
        beta=0.985,
        gamma=1.0,
        r_ss=0.01,
        phi_pi=1.5,
        kappa=0.1,
        shock_magnitude=-0.004,
        shock_rho=0.75,
        n_a=100,
        a_max=100.0,
        tol=1e-9,
    )
    girf_nl = extract_hank_group_irf(ss_nl, groups="quintiles", tracking="cohort")
    assert girf_nl.model_kind == "sequence_space_nonlinear"
    assert girf_nl.conservation_audit["consumption_adding_up_error"] < 1e-10
    assert girf_nl.conservation_audit["consumption_share_adding_up_error"] < 1e-10
    assert girf_nl.conservation_audit["asset_adding_up_error"] < 1e-10

    for nl_flag in (False, True):
        girf_tr = simulate_transfer_group_irf(
            ss_model=ss_lin,
            target="all",
            amount=0.01,
            horizon=30,
            nonlinear=nl_flag,
            groups="quintiles",
        )
        assert girf_tr.model_kind == "partial_equilibrium_transfer"
        assert girf_tr.conservation_audit["consumption_adding_up_error"] < 1e-10
        assert girf_tr.conservation_audit["consumption_share_adding_up_error"] < 1e-10
        assert girf_tr.conservation_audit["asset_adding_up_error"] < 1e-10


# ---------------------------------------------------------------------------
# 2. Zero-shock, Ricardian & transfer differential incidence checks
# ---------------------------------------------------------------------------


def test_zero_shock_produces_exact_zero_group_irfs(hank_ss: Any) -> None:
    T = 30
    girf = solve_hank_group_irf(
        ss_model=hank_ss,
        shock_seq=np.zeros(T),
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=T,
        groups="quintiles",
    )
    assert np.max(np.abs(girf.consumption_irf)) < 1e-12
    assert np.max(np.abs(girf.consumption_share_irf)) < 1e-12
    assert np.max(np.abs(girf.asset_irf)) < 1e-10
    assert np.max(np.abs(girf.asset_share_irf)) < 1e-10


def test_transfer_incidence_low_liquidity_responds_more_than_high_wealth(
    hank_ss: Any,
    fiscal_lumpsum_taxcut: Any,
    fiscal_prop_taxcut: Any,
) -> None:
    # 1. Partial-equilibrium cash-on-hand uniform transfer: Q1 >> Q5 and HtM >> Unconstrained
    tr_q = simulate_transfer_group_irf(
        ss_model=hank_ss,
        target="all",
        amount=0.01,
        horizon=25,
        groups="quintiles",
    )
    assert tr_q.consumption_irf[0, 0] > 0.0
    assert tr_q.consumption_irf[0, 0] > tr_q.consumption_irf[0, -1]
    assert tr_q.steady_state_group_mpc[0] > tr_q.steady_state_group_mpc[-1]

    tr_htm = simulate_transfer_group_irf(
        ss_model=hank_ss,
        target="all",
        amount=0.01,
        horizon=25,
        groups="htm_vs_unconstrained",
        htm_definition="constrained",
    )
    assert tr_htm.consumption_irf[0, 0] > tr_htm.consumption_irf[0, 1]
    assert tr_htm.steady_state_group_mpc[0] > tr_htm.steady_state_group_mpc[1]

    # 2. GE fiscal lump-sum tax cut vs proportional tax cut:
    # Uniform lump-sum boosts low-wealth / HtM consumption strictly more than Q5 / Unconstrained,
    # and strictly widens the (HtM - Unconstrained) impact incidence relative to proportional taxes.
    ls_q = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")
    assert ls_q.consumption_irf[0, 0] > ls_q.consumption_irf[0, -1] > 0.0

    ls_htm = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="htm_vs_unconstrained")
    pr_htm = extract_hank_group_irf(fiscal_prop_taxcut, groups="htm_vs_unconstrained")
    ls_gap = float(ls_htm.consumption_irf[0, 0] - ls_htm.consumption_irf[0, 1])
    pr_gap = float(pr_htm.consumption_irf[0, 0] - pr_htm.consumption_irf[0, 1])
    assert ls_gap > 0.0
    assert ls_gap > pr_gap

    # 3. Ricardian failure: debt-financed spending expansion cushions Q1 vs balanced budget
    T = 40
    g_expand = 0.02 * (0.7 ** np.arange(T))
    bb_q = solve_hank_group_irf(
        ss_model=hank_ss,
        shock_seq=g_expand,
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=T,
        groups="quintiles",
    )
    dr_q = solve_hank_group_irf(
        ss_model=hank_ss,
        shock_seq=g_expand,
        closure="debt_rule",
        tax_type="lump_sum",
        horizon=T,
        phi_B=0.08,
        groups="quintiles",
    )
    assert dr_q.consumption_irf[0, 0] > bb_q.consumption_irf[0, 0]
    assert dr_q.aggregate_consumption_irf[0] > bb_q.aggregate_consumption_irf[0]


# ---------------------------------------------------------------------------
# 3. Empirical Distributional LP & Quantile LP alignment and scoring
# ---------------------------------------------------------------------------


def _simulate_distributional_share_dgp(
    true_share_irf: np.ndarray,
    ss_shares: np.ndarray,
    *,
    n_obs: int = 320,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a synthetic time-series panel where empirical share LPs recover ``true_share_irf``."""
    rng = np.random.default_rng(seed)
    H, G = true_share_irf.shape
    eps = rng.normal(0.0, 1.0, size=n_obs)

    shares = np.zeros((n_obs, G), dtype=np.float64)
    # Build stationary level shares so s_{t+h, g} - s_{t-1, g} = psi_h * eps_t + zero-sum noise
    base_noise = rng.normal(0.0, 0.00025, size=(n_obs, G))
    base_noise -= base_noise.mean(axis=1, keepdims=True)
    for t in range(n_obs):
        signal = np.zeros(G, dtype=np.float64)
        for k in range(H):
            if t - k >= 0:
                signal += true_share_irf[k] * eps[t - k]
        # Remove small finite-precision drift so exact share sum == 1.0 across bins
        shares[t] = ss_shares + signal + base_noise[t]

    # Exact share normalization check without modifying relative impulse structure
    row_err = np.max(np.abs(shares.sum(axis=1) - 1.0))
    assert row_err < 1e-12

    data: dict[str, Any] = {"eps": eps}
    for j in range(G):
        data[f"Q{j + 1}"] = shares[:, j]
    return pd.DataFrame(data)


def _simulate_quantile_level_dgp(
    true_level_irf: np.ndarray,
    ss_group_c: np.ndarray,
    *,
    n_obs: int = 360,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate a single time-series consumption DGP matching HANK quintile responses."""
    rng = np.random.default_rng(seed)
    H, G = true_level_irf.shape
    assert G == 5
    eps = rng.normal(0.0, 1.0, size=n_obs)
    mean_irf = true_level_irf.mean(axis=1)
    grad_irf = 0.5 * (true_level_irf[:, 0] - true_level_irf[:, -1])
    sig = np.zeros(n_obs, dtype=np.float64)
    scale_mod = np.zeros(n_obs, dtype=np.float64)
    for t in range(n_obs):
        for k in range(H):
            if t - k >= 0:
                sig[t] += mean_irf[k] * eps[t - k]
                scale_mod[t] += grad_irf[k] * eps[t - k]
    z = rng.normal(0.0, 0.015, size=n_obs)
    c_series = float(np.mean(ss_group_c)) + sig + (1.0 - 20.0 * np.clip(scale_mod, -0.025, 0.025)) * z
    return pd.DataFrame({"eps": eps, "c": c_series})


def test_empirical_distributional_lp_bridge_beats_mismatched_closure(
    hank_ss: Any,
    fiscal_lumpsum_taxcut: Any,
) -> None:
    girf_match = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")
    true_share_irf = girf_match.consumption_share_irf[:10]
    df_lp = _simulate_distributional_share_dgp(
        true_share_irf,
        girf_match.steady_state_group_consumption_shares,
        n_obs=360,
        seed=7,
    )

    lp_res = estimate_distributional_lp(
        df_lp,
        shock="eps",
        bin_cols=["Q1", "Q2", "Q3", "Q4", "Q5"],
        horizons=range(0, 10),
        n_lags=1,
        distribution_type="shares",
    )
    assert lp_res.adding_up_max_error < 1e-10

    # Contractionary balanced-budget spending shock (opposite tax/consumption incidence)
    T = 40
    res_mismatch = solve_hank_fiscal(
        ss_model=hank_ss,
        shock_seq=0.02 * (0.7 ** np.arange(T)),
        closure="balanced_budget",
        tax_type="proportional",
        horizon=T,
        tol=1e-8,
    )
    girf_mismatch = extract_hank_group_irf(res_mismatch, groups="quintiles")

    bridge_good = bridge_hank_distributional(girf_match, lp_res)
    bridge_bad = score_hank_distributional(girf_mismatch, lp_res)

    assert isinstance(bridge_good, HANKDistributionalBridgeResult)
    assert bridge_good.response_type == "share"
    assert len(bridge_good.aligned_horizons) == 10
    assert bridge_good.model_irf.shape == (10, 5)
    assert bridge_good.solver_diagnostics.overall_status == "PASS"

    # Matching expansionary lump-sum closure strictly beats mismatched-sign closure
    assert bridge_good.rmse < bridge_bad.rmse
    assert bridge_good.mae < bridge_bad.mae
    assert bridge_good.weighted_rmse < bridge_bad.weighted_rmse
    assert bridge_good.sign_concordance > bridge_bad.sign_concordance
    assert bridge_good.sign_concordance >= 0.80
    assert bridge_good.band_coverage > bridge_bad.band_coverage
    assert bridge_good.mean_trajectory_correlation > 0.70

    # Check gradient & per-group metrics tables and summary tables
    assert bridge_good.gradient_metrics["low_group"] == "Q1"
    assert bridge_good.gradient_metrics["high_group"] == "Q5"
    assert bridge_good.gradient_metrics["gradient_impact_diff_model"] > 0.0
    assert bridge_good.gradient_metrics["gradient_impact_diff_empirical"] > 0.0
    assert set(bridge_good.per_group_metrics.index) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert "steady_state_mpc" in bridge_good.per_group_metrics.columns
    assert "within_band" in bridge_good.aligned_table.columns
    assert "t_stat" in bridge_good.aligned_table.columns
    assert not bridge_good.to_frame("metrics").empty
    assert not bridge_good.to_dataframe("gradient").empty
    assert "\\begin{tabular}" in bridge_good.to_latex(layout="per_group")
    assert "#table(" in bridge_good.to_typst(layout="gradient")


def test_empirical_quantile_lp_bridge_and_targets_formats(
    hank_ss: Any,
    fiscal_lumpsum_taxcut: Any,
) -> None:
    girf_match = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")
    df_q = _simulate_quantile_level_dgp(
        girf_match.consumption_irf[:6],
        girf_match.steady_state_group_consumption,
        n_obs=400,
        seed=99,
    )
    qlp_res = estimate_quantile_lp(
        df_q,
        shock="eps",
        outcome="c",
        quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
        horizons=range(0, 6),
        n_lags=1,
        cov_type="hac",
    )

    # Auto response_type emits informative warning for quantile LPs and compares in levels
    br_auto = bridge_hank_distributional(girf_match, qlp_res)
    assert br_auto.response_type == "level"
    assert any("quantile" in w.lower() for w in br_auto.warnings)
    assert br_auto.model_irf.shape == (6, 5)
    assert br_auto.sign_concordance >= 0.80

    # Contractionary baseline scores worse than expansionary lump-sum
    T = 40
    girf_contra = solve_hank_group_irf(
        ss_model=hank_ss,
        shock_seq=0.02 * (0.7 ** np.arange(T)),
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=T,
        groups="quintiles",
    )
    br_bad = bridge_hank_distributional(girf_contra, qlp_res, response_type="level")
    assert br_auto.rmse < br_bad.rmse
    assert br_auto.sign_concordance > br_bad.sign_concordance

    # Also test EmpiricalGroupTargets, wide DataFrame, long DataFrame, and explicit bin_mapping
    targets = EmpiricalGroupTargets(
        horizons=np.arange(6),
        groups=("Low", "LowerMid", "Mid", "UpperMid", "High"),
        irf_matrix=girf_match.consumption_irf[:6],
        se_matrix=np.full((6, 5), 0.0005),
        distribution_type="levels",
        shock_name="fiscal_transfer",
    )
    br_exact = bridge_hank_distributional(
        girf_match,
        targets,
        bin_mapping={"Low": "Q1", "LowerMid": "Q2", "Mid": "Q3", "UpperMid": "Q4", "High": "Q5"},
        horizons=[0, 1, 2, 3],
    )
    assert br_exact.rmse < 1e-14
    assert np.isclose(br_exact.band_coverage, 1.0)
    assert np.isclose(br_exact.sign_concordance, 1.0)
    assert len(br_exact.aligned_horizons) == 4

    wide_df = pd.DataFrame(
        {
            "Q1": girf_match.consumption_irf[:4, 0],
            "Q2": girf_match.consumption_irf[:4, 1],
            "Q3": girf_match.consumption_irf[:4, 2],
            "Q4": girf_match.consumption_irf[:4, 3],
            "Q5": girf_match.consumption_irf[:4, 4],
            "Q1_se": np.full(4, 0.001),
            "Q2_se": np.full(4, 0.001),
            "Q3_se": np.full(4, 0.001),
            "Q4_se": np.full(4, 0.001),
            "Q5_se": np.full(4, 0.001),
        },
        index=pd.Index([0, 1, 2, 3], name="horizon"),
    )
    br_wide = bridge_hank_distributional(girf_match, wide_df, shock_scale=2.0)
    assert np.isclose(br_wide.shock_scale, 2.0)
    assert br_wide.rmse > 0.0

    long_df = pd.DataFrame(
        [
            {"horizon": h, "bin": f"Q{g + 1}", "beta": float(girf_match.consumption_irf[h, g]), "se": 0.001}
            for h in range(4)
            for g in range(5)
        ]
    )
    br_long = bridge_hank_distributional(girf_match, long_df)
    assert br_long.rmse < 1e-14


# ---------------------------------------------------------------------------
# 4. Calibration metadata, SolverDiagnosticRecord regimes & bilingual outputs
# ---------------------------------------------------------------------------


def test_calibration_metadata_and_diagnostic_regimes(fiscal_lumpsum_taxcut: Any) -> None:
    girf = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")
    cal = girf.calibration
    assert 0.05 < cal["aggregate_mpc"] < 0.30
    assert 0.01 < cal["htm_constrained_share"] < 0.25
    assert cal["htm_bottom_quartile_share"] == 0.25
    assert 0.0 < cal["wealth_gini"] < 1.0
    assert cal["mpc_gradient_low_minus_high"] > 0.0
    assert cal["top_grid_mass"] <= 1e-6
    assert set(cal["group_mpcs"].keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}

    # Truncated upper grid (a_max=20.0) fails closed when allow_unconverged=False,
    # and surfaces UNSUPPORTED_REGIME via SolverDiagnosticRecord when allow_unconverged=True.
    T = 25
    with pytest.raises(ValueError, match="UNSUPPORTED_REGIME"):
        solve_hank_group_irf(
            shock_seq=-0.01 * (0.7 ** np.arange(T)),
            closure="balanced_budget",
            tax_type="lump_sum",
            horizon=T,
            n_a=60,
            a_max=20.0,
            groups="quintiles",
            allow_unconverged=False,
        )
    girf_trunc = solve_hank_group_irf(
        shock_seq=-0.01 * (0.7 ** np.arange(T)),
        closure="balanced_budget",
        tax_type="lump_sum",
        horizon=T,
        n_a=60,
        a_max=20.0,
        groups="quintiles",
        allow_unconverged=True,
    )
    assert girf_trunc.support_status.is_supported is False
    assert girf_trunc.support_status.regime == "boundary_asset_truncation"
    assert girf_trunc.solver_diagnostics.overall_status == "UNSUPPORTED_REGIME"


def test_bilingual_outputs_and_plotting(fiscal_lumpsum_taxcut: Any) -> None:
    girf = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")
    targets = EmpiricalGroupTargets(
        horizons=np.arange(8),
        groups=("Q1", "Q2", "Q3", "Q4", "Q5"),
        irf_matrix=girf.consumption_irf[:8],
        se_matrix=np.full((8, 5), 0.001),
    )
    bridge = bridge_hank_distributional(girf, targets)

    for obj in (girf, bridge):
        s_en = obj.summary(lang="en")
        s_es = obj.summary(lang="es")
        md_en = obj.to_markdown(lang="en")
        md_es = obj.to_markdown(lang="es")
        d_en = obj.to_dict(lang="en")
        d_es = obj.to_dict(lang="es")

        assert "Conservation" in s_en or "Conservación" in s_es
        assert "MPC" in s_en and "MPC" in s_es
        assert "Conservación" in s_es
        assert "Conservación" in md_es
        assert d_en["lang"] == "en"
        assert d_es["lang"] == "es"
        json.dumps(d_en)
        json.dumps(d_es)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig1 = girf.plot(metric="consumption")
    fig2 = bridge.plot(groups=["Q1", "Q5"])
    assert fig1 is not None and fig2 is not None
    plt.close(fig1)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# 5. Fail-closed error handling & input validation
# ---------------------------------------------------------------------------


def test_fail_closed_conservation_and_validation(fiscal_lumpsum_taxcut: Any) -> None:
    # Tamper aggregate dC by 1e-4 to prove fail-closed accounting conservation raises ValueError
    broken = dataclasses.replace(fiscal_lumpsum_taxcut, dC=fiscal_lumpsum_taxcut.dC + 1e-4)
    with pytest.raises(ValueError, match="Cross-sectional accounting conservation failed"):
        extract_hank_group_irf(broken, groups="quintiles", strict_conservation=True)

    flagged = extract_hank_group_irf(broken, groups="quintiles", strict_conservation=False)
    assert flagged.accuracy_status.is_accurate is False
    assert flagged.solver_diagnostics.overall_status == "CONVERGED_INACCURATE"

    # Unconverged result refused unless allow_unconverged=True
    unconv = dataclasses.replace(fiscal_lumpsum_taxcut, converged=False)
    with pytest.raises(ValueError, match="did not converge cleanly"):
        extract_hank_group_irf(unconv, allow_unconverged=False)

    girf = extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="quintiles")

    # Incompatible group count without bin_mapping
    bad_g = EmpiricalGroupTargets(
        horizons=np.arange(4),
        groups=("G1", "G2", "G3"),
        irf_matrix=np.zeros((4, 3)),
    )
    with pytest.raises(ValueError, match="3 bins"):
        bridge_hank_distributional(girf, bad_g)

    # Empty horizon overlap
    bad_h = EmpiricalGroupTargets(
        horizons=np.array([100, 101]),
        groups=("Q1", "Q2", "Q3", "Q4", "Q5"),
        irf_matrix=np.zeros((2, 5)),
    )
    with pytest.raises(ValueError, match="No overlapping horizons"):
        bridge_hank_distributional(girf, bad_h)

    # NaN / Inf / negative SE rejections
    with pytest.raises(ValueError, match="non-finite"):
        EmpiricalGroupTargets(
            horizons=np.arange(3),
            groups=("Q1", "Q2"),
            irf_matrix=np.array([[0.1, np.nan], [0.0, 0.0], [0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="non-finite"):
        EmpiricalGroupTargets(
            horizons=np.arange(3),
            groups=("Q1", "Q2"),
            irf_matrix=np.zeros((3, 2)),
            se_matrix=np.array([[0.1, np.inf], [0.1, 0.1], [0.1, 0.1]]),
        )
    with pytest.raises(ValueError, match="negative"):
        EmpiricalGroupTargets(
            horizons=np.arange(3),
            groups=("Q1", "Q2"),
            irf_matrix=np.zeros((3, 2)),
            se_matrix=np.array([[0.1, -0.01], [0.1, 0.1], [0.1, 0.1]]),
        )

    # Non-partitioning custom weights
    shape_ss = fiscal_lumpsum_taxcut.steady_state_model.policy_c.shape
    with pytest.raises(ValueError, match="sum to 1.0"):
        extract_hank_group_irf(
            fiscal_lumpsum_taxcut,
            groups={"A": np.full(shape_ss, 0.4), "B": np.full(shape_ss, 0.4)},
        )

    # Invalid tracking on linear model / non-quantile group scheme / invalid lang / bad shock_scale
    ss_lin = solve_hank_sequence_space(T=20, n_a=60, a_max=100.0)
    with pytest.raises(ValueError, match="cohort"):
        extract_hank_group_irf(ss_lin, groups="quintiles", tracking="cross_section")
    with pytest.raises(ValueError, match="quantile"):
        extract_hank_group_irf(fiscal_lumpsum_taxcut, groups="htm_vs_unconstrained", tracking="cross_section")
    with pytest.raises(ValueError, match="lang must be 'en' or 'es'"):
        girf.summary(lang="fr")
    with pytest.raises(ValueError, match="shock_scale"):
        bridge_hank_distributional(
            girf,
            EmpiricalGroupTargets(
                horizons=np.arange(4),
                groups=("Q1", "Q2", "Q3", "Q4", "Q5"),
                irf_matrix=np.zeros((4, 5)),
            ),
            shock_scale=0.0,
        )


# ---------------------------------------------------------------------------
# 6. Pyodide import isolation
# ---------------------------------------------------------------------------


_FORBIDDEN_PYODIDE = {
    "statsmodels",
    "linearmodels",
    "arch",
    "bs4",
    "pdfplumber",
    "pypdf",
    "sqlite3",
    "ssl",
    "pyarrow",
    "openpyxl",
}


class _BlockForbiddenPyodide(MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.split(".")[0] in _FORBIDDEN_PYODIDE:
            raise ImportError(f"Blocked forbidden Pyodide import: {fullname}")
        return None


@pytest.mark.mechanism_control
def test_pyodide_import_hook_positive_control() -> None:
    blocker = _BlockForbiddenPyodide()
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "statsmodels" or k.startswith("statsmodels.")}
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ImportError, match="Blocked forbidden Pyodide import"):
            __import__("statsmodels")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_pyodide_top_level_import_isolation() -> None:
    mod_path = Path("puremacro/models/hank_distributional_bridge.py")
    tree = ast.parse(mod_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in _FORBIDDEN_PYODIDE
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in _FORBIDDEN_PYODIDE

    blocker = _BlockForbiddenPyodide()
    sys.meta_path.insert(0, blocker)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "puremacro.models.hank_distributional_bridge"}
    try:
        __import__("puremacro.models.hank_distributional_bridge")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
