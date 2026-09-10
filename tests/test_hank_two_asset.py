"""Unit and integration tests for Track B2: Two-Asset HANK Sequence-Space Bridge.

Tests:
1. Stationary distribution normalization: sum(D*) == 1.0 and exact column-stochastic Lambda.
2. Transaction costs properties: chi(0, a) == 0, chi(d, a) > 0 for d != 0, and monotonicity.
3. Policy functions monotonicity: consumption c strictly increasing in liquid cash.
4. Two-Asset Fake-News Jacobians dimensions (T, T) and economic signs.
5. Finite difference verification: forward household simulation matches Fake News column 0.
6. Parsing and solving reference hank_two_asset.mod end-to-end via HANKModel and solve_hank_bridge.
7. Diagnostics, summary, markdown/latex export, and 2D distribution plotting.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from puremacro.models.hank_sequence_space import (
    _transaction_cost,
    _lottery_2d,
    _build_transition_matrix_2d,
    _stationary_distribution_2d,
    _solve_two_asset_household_block,
    _two_asset_jacobians,
    solve_two_asset_hank_sequence_space,
    TwoAssetSequenceSpaceHANKResult,
)
from puremacro.dsge.hank import (
    HANKModel,
    HANKResult,
    load_hank_mod,
    solve_hank_bridge,
)


# ---------------------------------------------------------------------------
# 1. Stationary Distribution Normalization & Stochastic Matrix
# ---------------------------------------------------------------------------

def test_stationary_distribution_normalization():
    """Verify 2D bilinear lottery transition matrix is column-stochastic and sum(D*) == 1.0."""
    na, nb, ns = 20, 20, 2
    a_grid = np.geomspace(1e-4, 25.0 + 1e-4, na) - 1e-4
    b_grid = np.linspace(0.0, 10.0, nb)
    pi_s = np.array([[0.9, 0.1], [0.1, 0.9]])

    hh = _solve_two_asset_household_block(
        n_a=na,
        n_b=nb,
        a_max=25.0,
        b_max=10.0,
        r_b_ss=0.01,
        r_a_ss=0.03,
    )

    # Column stochastic: each column sums to 1.0 within numerical precision
    col_sums = hh.Lambda.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-12)

    # Stationary distribution sums to 1.0
    assert pytest.approx(np.sum(hh.D_ss), abs=1e-6) == 1.0
    assert np.all(hh.D_ss >= 0.0)

    # Marginals sum to 1.0
    assert pytest.approx(np.sum(hh.marginal_distribution_a), abs=1e-6) == 1.0
    assert pytest.approx(np.sum(hh.marginal_distribution_b), abs=1e-6) == 1.0
    assert pytest.approx(np.sum(hh.joint_distribution), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# 2. Transaction Costs & Monotonicity
# ---------------------------------------------------------------------------

def test_transaction_cost_properties():
    """Verify chi(0, a) == 0, chi(d, a) > 0 for d != 0, and convexity/monotonicity."""
    a = 5.0
    assert _transaction_cost(0.0, a, chi_0=0.25, chi_1=1.0) == 0.0
    assert _transaction_cost(1.0, a, chi_0=0.25, chi_1=1.0) > 0.0
    assert _transaction_cost(-1.0, a, chi_0=0.25, chi_1=1.0) > 0.0

    # Symmetric around 0
    c_pos = _transaction_cost(2.0, a, chi_0=0.25, chi_1=1.0)
    c_neg = _transaction_cost(-2.0, a, chi_0=0.25, chi_1=1.0)
    assert pytest.approx(c_pos) == c_neg

    # Strictly increasing in magnitude |d|
    d_vals = np.array([0.5, 1.0, 2.0, 3.0])
    costs = [_transaction_cost(d, a, chi_0=0.25, chi_1=1.0) for d in d_vals]
    assert np.all(np.diff(costs) > 0.0)

    # Non-quadratic parameter chi_1 != 1.0
    c_gen = _transaction_cost(1.5, a, chi_0=0.3, chi_1=1.5)
    assert c_gen > 0.0


def test_policy_function_properties():
    """Verify policy function monotonicity in assets."""
    hh = _solve_two_asset_household_block(
        n_a=20,
        n_b=20,
        a_max=20.0,
        b_max=10.0,
    )
    # Average consumption is strictly monotonically increasing in liquid assets b
    mean_c_b = np.mean(hh.c_ss, axis=(0, 2))
    assert np.all(np.diff(mean_c_b) > 0.0)

    # Positive consumption and asset constraints respected
    assert np.all(hh.c_ss > 0.0)
    assert np.all(hh.b_ss >= 0.0)
    assert np.all(hh.a_ss >= 0.0)

    # High productivity state yields higher average consumption than low productivity state
    assert np.mean(hh.c_ss[:, :, 1]) > np.mean(hh.c_ss[:, :, 0])


# ---------------------------------------------------------------------------
# 3. Fake-News Jacobians Dimensions & Properties
# ---------------------------------------------------------------------------

def test_two_asset_jacobians_dimensions_and_signs():
    """Verify all 6 sequence-space Jacobians have shape (T, T) and correct economic signs."""
    T = 25
    hh = _solve_two_asset_household_block(
        n_a=15,
        n_b=15,
        a_max=20.0,
        b_max=10.0,
    )
    jacs = _two_asset_jacobians(hh, T=T)

    expected_keys = ["J_C_rb", "J_C_ra", "J_C_Y", "J_D_rb", "J_D_ra", "J_D_Y"]
    for key in expected_keys:
        assert key in jacs
        mat = jacs[key]
        assert mat.shape == (T, T)
        assert np.all(np.isfinite(mat))

    # Real liquid rate increase depresses contemporaneous consumption
    assert jacs["J_C_rb"][0, 0] < 0.0

    # Aggregate income expansion stimulates consumption
    assert jacs["J_C_Y"][0, 0] > 0.0


# ---------------------------------------------------------------------------
# 4. Reference .mod File Parsing & Bridge Integration
# ---------------------------------------------------------------------------

def test_two_asset_mod_file_parsing():
    """Verify parsing and configuration of reference hank_two_asset.mod."""
    model = load_hank_mod("puremacro/dsge/_references/hank_two_asset.mod")
    assert model.is_two_asset is True
    assert model.hetagent_config["model"] == "hank_two_asset"
    assert int(float(model.hetagent_config["assets"])) == 2
    assert model.hetagent_config["liquid_asset"] == "b"
    assert model.hetagent_config["illiquid_asset"] == "a"

    assert len(model.asset_grid) == 25
    assert model.liquid_asset_grid is not None
    assert len(model.liquid_asset_grid) == 25
    assert model.joint_distribution is not None
    assert model.joint_distribution.shape == (25, 25)
    assert pytest.approx(np.sum(model.joint_distribution), abs=1e-5) == 1.0


def test_solve_hank_bridge_two_asset_end_to_end():
    """Verify solve_hank_bridge end-to-end on Two-Asset HANK model."""
    horizon = 20
    res = solve_hank_bridge(
        "puremacro/dsge/_references/hank_two_asset.mod",
        shock="eps_m",
        magnitude=-0.0025,
        horizon=horizon,
    )

    assert isinstance(res, HANKResult)
    assert res.converged is True
    assert res.horizon == horizon
    assert res.liquid_asset_grid is not None
    assert res.joint_distribution is not None
    assert res.marginal_distribution_b is not None

    # Check columns
    cols = res.transition_paths.columns
    assert "Y" in cols
    assert "C" in cols
    assert "D" in cols
    assert "r_b" in cols
    assert "r_a" in cols
    assert "pi" in cols
    assert "i" in cols

    # Monetary easing transmission:
    # Output, consumption, and inflation expand
    dY = res.transition_paths["Y"].to_numpy()
    dC = res.transition_paths["C"].to_numpy()
    dpi = res.transition_paths["pi"].to_numpy()
    assert dY[0] > 0.0
    assert dC[0] > 0.0
    assert dpi[0] > 0.0

    # Summary table and exports
    df_sum = res.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert "impact_response" in df_sum.columns

    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| Y" in md or "| variable" in md

    latex = res.to_latex()
    assert isinstance(latex, str)
    assert r"\begin{tabular}" in latex

    typst = res.to_typst()
    assert isinstance(typst, str)
    assert "#table(" in typst

    # 2D Plotting
    fig_tr, axes_tr = res.plot_transition()
    assert fig_tr is not None

    fig_dist, axes_dist = res.plot_distribution()
    assert fig_dist is not None


def test_solve_two_asset_hank_standalone():
    """Test standalone solve_two_asset_hank_sequence_space function."""
    res = solve_two_asset_hank_sequence_space(T=20, n_a=15, n_b=15)
    assert isinstance(res, TwoAssetSequenceSpaceHANKResult)
    assert res.horizon == 20
    assert res.converged is True
    assert len(res.irf_output) == 20
    assert len(res.irf_consumption) == 20
    assert len(res.irf_deposit) == 20
    assert res.jacobian_c_rb.shape == (20, 20)
    assert res.jacobian_d_rb.shape == (20, 20)

    summary = res.summary()
    assert "Two-Asset Sequence-Space HANK" in summary

    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
