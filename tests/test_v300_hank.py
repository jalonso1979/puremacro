"""Unit and integration tests for Pillar 3: Heterogeneous Agents (HANK) Sequence-Space Bridge.

Tests:
1. Gate 3.1: hetagent_block syntax parsing and AST DAG node validation.
2. Stationary asset distribution and MPC distribution calculation on asset grid.
3. Sequence-space Jacobians (J_C_r, J_C_Y) via Fake-News Algorithm.
4. Gate 3.2: Linear transition simulation for monetary policy shock matches Auclert et al. (2021) baseline.
5. Nonlinear Broyden transition simulation convergence.
6. Gate 3.3: HANKResult methods (.summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), plotting).
7. High-level solve_hank_bridge API end-to-end integration.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._results import HANKResult
from puremacro.dsge.hank import (
    HANKModel,
    load_hank_mod,
    solve_hank_bridge,
)

SAMPLE_HANK_MOD = """
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50;
  a_max = 30.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
"""


# ---------------------------------------------------------------------------
# 1. Gate 3.1: Parser and AST Node Validation Tests
# ---------------------------------------------------------------------------

def test_parse_hetagent_block_syntax():
    """Verify that hetagent_block parses cleanly into ParsedModelDAG with key-values."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    assert dag.hetagent_block is not None
    assert dag.hetagent_block["model"] == "one_asset_hank"
    assert int(float(dag.hetagent_block["n_a"])) == 50
    assert float(dag.hetagent_block["a_max"]) == 30.0
    assert float(dag.hetagent_block["borrowing_limit"]) == 0.0
    assert dag.hetagent_block["grid"] == "hyperbolic"

    # Standard model declarations parsed cleanly alongside
    assert "Y" in dag.variables
    assert "C" in dag.variables
    assert "eps_m" in dag.shocks
    assert float(dag.parameter_values["beta"]) == 0.985
    assert float(dag.parameter_values["kappa"]) == 0.1


def test_load_hank_mod_missing_block_raises():
    """Calling load_hank_mod on standard DSGE .mod without hetagent_block raises informative error."""
    plain_mod = """
    var Y; varexo eps;
    parameters alpha; alpha = 0.33;
    model; Y = eps; end;
    """
    with pytest.raises(ValueError, match="does not contain a hetagent_block"):
        load_hank_mod(plain_mod)


# ---------------------------------------------------------------------------
# 2. Household Steady State & Distributions Tests
# ---------------------------------------------------------------------------

def test_hank_model_steady_state():
    """Verify stationary wealth distribution D*(a) and MPC(a) across asset grid."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    model = HANKModel(dag)

    assert len(model.asset_grid) == 50
    assert model.asset_grid[0] == 0.0
    assert model.asset_grid[-1] == 30.0

    # Stationary wealth distribution
    assert len(model.asset_distribution) == 50
    assert np.all(model.asset_distribution >= 0.0)
    assert pytest.approx(np.sum(model.asset_distribution), abs=1e-5) == 1.0

    # MPC distribution across wealth
    assert model.mpc_distribution is not None
    assert len(model.mpc_distribution) == 50
    assert np.all(model.mpc_distribution > 0.0)
    assert np.all(model.mpc_distribution < 1.0)
    # Borrowing-constrained / low-wealth agents have higher MPC than wealthy agents
    assert model.mpc_distribution[0] > model.mpc_distribution[-1]

    # Aggregate steady-state values
    assert model.steady_state["Y"] > 0.0
    assert model.steady_state["C"] > 0.0
    assert pytest.approx(model.steady_state["r"], abs=1e-6) == 0.01
    assert model.steady_state["pi"] == 0.0


# ---------------------------------------------------------------------------
# 3. Fake-News Sequence-Space Jacobians Tests
# ---------------------------------------------------------------------------

def test_hank_jacobians_fake_news():
    """Verify sequence-space Jacobians J_C_r and J_C_Y computed via Fake-News Algorithm."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    model = HANKModel(dag)

    T = 40
    jacobians = model.compute_jacobians(T=T)
    assert "J_C_r" in jacobians
    assert "J_C_Y" in jacobians

    J_C_r = jacobians["J_C_r"]
    J_C_Y = jacobians["J_C_Y"]

    assert J_C_r.shape == (T, T)
    assert J_C_Y.shape == (T, T)

    # Intertemporal properties:
    # 1. Higher interest rates depress contemporaneous consumption: dC_0 / dr_0 < 0
    assert J_C_r[0, 0] < 0.0

    # 2. Higher income stimulates contemporaneous consumption: dC_0 / dY_0 > 0
    assert J_C_Y[0, 0] > 0.0
    # Average MPC on impact is between 0 and 1
    assert 0.0 < J_C_Y[0, 0] < 1.0

    # 3. Discounting / causality: future shocks have decaying contemporaneous effect
    assert abs(J_C_r[0, -1]) < abs(J_C_r[0, 0])
    assert abs(J_C_Y[0, -1]) < abs(J_C_Y[0, 0])


# ---------------------------------------------------------------------------
# 4. Gate 3.2: General Equilibrium Transition Simulation Tests
# ---------------------------------------------------------------------------

def test_linear_transition_monetary_policy_shock():
    """Simulate monetary policy shock (expansionary, -25bp) under linear SSJ."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    model = HANKModel(dag)

    # Expansionary 25bp monetary policy shock: eps_m = -0.0025
    horizon = 30
    res = model.simulate(
        shock="eps_m",
        magnitude=-0.0025,
        rho=0.5,
        horizon=horizon,
        nonlinear=False,
    )

    assert isinstance(res, HANKResult)
    assert res.converged is True
    assert res.horizon == horizon
    assert res.shock_name == "eps_m"
    assert len(res.transition_paths) == horizon

    # Check Auclert et al. (2021) baseline macroeconomic transmission:
    # Interest rate reduction stimulates consumption and output
    dY = res.transition_paths["Y"].to_numpy()
    dC = res.transition_paths["C"].to_numpy()
    dpi = res.transition_paths["pi"].to_numpy()
    dr = res.transition_paths["r"].to_numpy()

    assert dY[0] > 0.0, "Output should expand on impact after monetary easing"
    assert dC[0] > 0.0, "Consumption should expand on impact after monetary easing"
    assert dpi[0] > 0.0, "Inflation should rise on impact via NKPC"
    assert dr[0] < 0.0, "Real interest rate should fall on impact"

    # Paths return toward steady state at the end of horizon
    assert abs(dY[-1]) < abs(dY[0])
    assert abs(dpi[-1]) < abs(dpi[0])


def test_nonlinear_broyden_transition():
    """Simulate general equilibrium transition using nonlinear Broyden solver."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    model = HANKModel(dag)

    horizon = 20
    res_nl = model.simulate(
        shock="eps_m",
        magnitude=-0.0025,
        rho=0.5,
        horizon=horizon,
        nonlinear=True,
    )

    assert isinstance(res_nl, HANKResult)
    assert res_nl.converged is True
    assert len(res_nl.transition_paths) == horizon

    dY_nl = res_nl.transition_paths["Y"].to_numpy()
    dC_nl = res_nl.transition_paths["C"].to_numpy()
    assert dY_nl[0] > 0.0
    assert dC_nl[0] > 0.0


# ---------------------------------------------------------------------------
# 5. Gate 3.3: HANKResult Diagnostics and Reporting Tests
# ---------------------------------------------------------------------------

def test_hank_result_diagnostics_and_outputs():
    """Verify HANKResult table generation, export formats, and plotting routines."""
    dag = parse_mod_to_dag(SAMPLE_HANK_MOD)
    model = HANKModel(dag)
    res = model.simulate(shock="eps_m", magnitude=-0.0025, horizon=25)

    # 1. Summary table
    df_summary = res.summary()
    assert isinstance(df_summary, pd.DataFrame)
    assert "impact_response" in df_summary.columns
    assert "peak_response" in df_summary.columns
    assert "peak_period" in df_summary.columns
    assert "Y" in df_summary.index
    assert "C" in df_summary.index
    assert df_summary.loc["Y", "impact_response"] > 0.0

    # 2. DataFrame paths
    df_paths = res.to_frame()
    assert isinstance(df_paths, pd.DataFrame)
    assert len(df_paths) == 25
    assert "Y" in df_paths.columns

    # 3. Markdown, LaTeX, Typst representations
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| variable" in md or "| Y" in md

    latex = res.to_latex()
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert isinstance(typst, str)
    assert "#table(" in typst

    # 4. Plotting routines (headless Agg)
    fig_tr, axes_tr = res.plot_transition()
    assert fig_tr is not None
    assert axes_tr is not None

    fig_dist, axes_dist = res.plot_distribution()
    assert fig_dist is not None
    assert axes_dist is not None


# ---------------------------------------------------------------------------
# 6. High-Level Helper Functions Tests
# ---------------------------------------------------------------------------

def test_solve_hank_bridge_helper():
    """Test end-to-end solve_hank_bridge helper function."""
    res = solve_hank_bridge(SAMPLE_HANK_MOD, horizon=20, magnitude=-0.0025)
    assert isinstance(res, HANKResult)
    assert res.horizon == 20
    assert res.converged is True
    assert res.transition_paths["Y"].iloc[0] > 0.0
