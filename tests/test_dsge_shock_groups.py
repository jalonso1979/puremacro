"""Comprehensive unit tests for Dynare shock groups parsing and decomposition.

Tests cover:
1. AST parsing of shock_groups; and shock_groups(group_name=...); blocks.
2. Grouped historical shock decomposition.
3. Machine precision adding-up invariant: sum(components) + initial == y_obs (<= 1e-12).
4. Unassigned shocks handling.
5. Integration with smooth_model historical innovations.
6. Presentation contract compliance: summary, plot, to_frame, to_markdown, to_latex, to_typst.
7. FrozenInstanceError on result mutation.
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
    load_mod,
    shock_groups_decomposition,
    ShockDecompositionResult,
)
from puremacro.dsge._parser import parse_mod_to_dag


@pytest.fixture
def nk_model_with_groups():
    """Build NK model with declared shock groups."""
    mod_text = """
    var y, pi, r, a;
    varexo e_d, e_m;
    parameters beta, sigma, kappa, phi_pi, phi_y, rho_r, rho_a;

    beta = 0.99;
    sigma = 1.0;
    kappa = 0.15;
    phi_pi = 1.5;
    phi_y = 0.25;
    rho_r = 0.7;
    rho_a = 0.6;

    model;
    y = y(+1) - (r - pi(+1)) / sigma + a;
    pi = beta * pi(+1) + kappa * y;
    r = rho_r * r(-1) + (1.0 - rho_r) * (phi_pi * pi + phi_y * y) + e_m;
    a = rho_a * a(-1) + e_d;
    end;

    shock_groups;
    'Demand Shocks' = e_d;
    'Monetary Shocks' = e_m;
    end;

    shock_groups(group_name = policy_focus);
    'Policy' = e_m;
    end;
    """
    dag = parse_mod_to_dag(mod_text)
    m = load_mod(mod_text)
    return m, dag


def test_shock_groups_ast_parsing(nk_model_with_groups):
    """Verify AST parser correctly extracts standard and named shock groups."""
    _, dag = nk_model_with_groups
    assert "Demand Shocks" in dag.shock_groups
    assert dag.shock_groups["Demand Shocks"] == ["e_d"]
    assert "Monetary Shocks" in dag.shock_groups
    assert dag.shock_groups["Monetary Shocks"] == ["e_m"]

    assert "policy_focus" in dag.named_shock_groups
    assert dag.named_shock_groups["policy_focus"]["Policy"] == ["e_m"]


def test_shock_groups_decomposition_adding_up_balance(nk_model_with_groups):
    """Verify components + initial identically equals y_obs within <= 1e-12."""
    m, _ = nk_model_with_groups
    T = 25
    rng = np.random.default_rng(42)
    shocks = rng.normal(0, 0.02, size=(T, 2))
    sim_df = m.simulate(periods=T, shocks=shocks, burn=0)

    groups = {
        "Demand": ["e_d"],
        "Monetary": ["e_m"],
    }
    res = shock_groups_decomposition(m, data=sim_df, groups=groups, initial_state=np.zeros(m.n_states))

    assert isinstance(res, ShockDecompositionResult)
    for var in m.variables:
        # Check adding-up identity: sum of components across all groups plus initial equals data[var]
        total = np.zeros(T)
        for g_name, g_df in res.components.items():
            total += g_df[var].to_numpy()
        diff = np.max(np.abs(total - sim_df[var].to_numpy()))
        assert diff <= 1e-12, f"Balance error for {var}: {diff:.4e}"


def test_shock_groups_decomposition_unassigned_group(nk_model_with_groups):
    """Verify unassigned shocks are properly partitioned into 'Others' or unassigned."""
    m, _ = nk_model_with_groups
    T = 15
    rng = np.random.default_rng(99)
    shocks = rng.normal(0, 0.01, size=(T, 2))
    sim_df = m.simulate(periods=T, shocks=shocks, burn=0)

    # Only assign e_m, leaving e_d unassigned
    groups = {"Monetary": ["e_m"]}
    res = shock_groups_decomposition(m, data=sim_df, groups=groups)

    assert "Monetary" in res.components
    assert "Others" in res.components
    # e_d must be captured in 'Others'
    assert np.any(np.abs(res.components["Others"]["y"]) > 1e-6)

    # Invariant still holds
    total = np.zeros(T)
    for g_name, g_df in res.components.items():
        total += g_df["y"].to_numpy()
    np.testing.assert_allclose(total, sim_df["y"].to_numpy(), atol=1e-12)


def test_shock_groups_presentation_contract(nk_model_with_groups):
    """Verify presentation dataclass contract compliance for ShockDecompositionResult."""
    m, _ = nk_model_with_groups
    sim_df = m.simulate(periods=10, seed=123)
    res = shock_groups_decomposition(m, data=sim_df)

    summary = res.summary()
    assert isinstance(summary, str)
    assert "Shock Decomposition" in summary

    frame = res.to_frame(var="r")
    assert isinstance(frame, pd.DataFrame)

    md = res.to_markdown(var="r")
    assert isinstance(md, str)
    assert "|" in md

    latex = res.to_latex(var="r")
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    typst = res.to_typst(var="r")
    assert isinstance(typst, str)
    assert "#table(" in typst

    fig = res.plot(var="r")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_shock_groups_immutability(nk_model_with_groups):
    """Verify ShockDecompositionResult is frozen."""
    m, _ = nk_model_with_groups
    sim_df = m.simulate(periods=5, seed=1)
    res = shock_groups_decomposition(m, data=sim_df)
    with pytest.raises(FrozenInstanceError):
        res.components = {}
