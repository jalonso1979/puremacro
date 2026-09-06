"""Unit tests for Gertler-Karadi (2011) DSGE with financial frictions and OccBin."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    GK2011_PARAMS,
    GertlerKaradiResult,
    build_gertler_karadi_model,
    solve_gertler_karadi,
    solve_steady_state,
)


# ---------------------------------------------------------------------------
# Test 1: Steady-State Solving & Calibration Checks
# ---------------------------------------------------------------------------

def test_gk_steady_state_calibration():
    """Verify steady-state values match canonical GK (2011) Table 1 targets."""
    ss = solve_steady_state()

    # 1. Steady-state leverage phi = Q*S / N ~ 4.0
    phi_ss = ss["phi"]
    assert 3.8 <= phi_ss <= 4.3, f"Steady-state leverage {phi_ss:.4f} outside target [3.8, 4.3]"

    # 2. Steady-state credit spread R_k - R ~ 100 bps annualized
    spread_ann = ss["spread_ann"]
    assert 95.0 <= spread_ann <= 110.0, f"Annualized spread {spread_ann:.2f} bps outside target [95, 110] bps"

    # 3. Steady-state asset price Q = 1.0, capacity utilization U = 1.0, capital quality xi = 1.0
    assert np.isclose(ss["Q"], 1.0, atol=1e-12)
    assert np.isclose(ss["U"], 1.0, atol=1e-12)
    assert np.isclose(ss["xi"], 1.0, atol=1e-12)
    assert np.isclose(ss["Pi"], 1.0, atol=1e-12)
    assert np.isclose(ss["psi"], 0.0, atol=1e-12)

    # 4. Resource constraint clearing: Y = C + I + G
    res_gap = ss["Y"] - ss["C"] - ss["I"] - ss["G"]
    assert np.isclose(res_gap, 0.0, atol=1e-10), f"Resource constraint gap {res_gap:.2e} != 0"

    # 5. Bank balance sheet clearing: Q*S = B + N
    bs_gap = ss["Q"] * ss["K"] - (ss["B"] + ss["N"])
    assert np.isclose(bs_gap, 0.0, atol=1e-10), f"Bank balance sheet gap {bs_gap:.2e} != 0"

    # 6. Bank net worth renewal: N = N_e + N_n
    nw_gap = ss["N"] - (ss["Ne"] + ss["Nn"])
    assert np.isclose(nw_gap, 0.0, atol=1e-10), f"Net worth renewal gap {nw_gap:.2e} != 0"


# ---------------------------------------------------------------------------
# Test 2: Klein Linear Solver & Capital Quality Shock
# ---------------------------------------------------------------------------

def test_gk_klein_solver_capital_quality_shock():
    """Verify Klein QZ linear solution under capital quality shock.

    Asserts:
    - Credit spread R_k - R spikes significantly (> 100 bps annualized).
    - Bank net worth N contracts sharply (> 15% drop).
    - Output Y and investment I decline.
    - Tobin's Q declines.
    """
    res = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="klein",
    )

    assert isinstance(res, GertlerKaradiResult)
    assert res.solver_method == "klein"
    assert res.binding_periods == 0
    assert len(res.irf) == 40

    # 1. Credit spread spikes on impact
    spread_spike_ann = res.irf["prem"].iloc[0] * 40000.0
    assert spread_spike_ann > 100.0, f"Spread spike {spread_spike_ann:.1f} bps < 100 bps"

    # 2. Bank net worth contracts sharply
    ss_N = res.steady_state["N"]
    n_drop_pct = (res.irf["N"].iloc[0] / ss_N) * 100.0
    assert n_drop_pct < -15.0, f"Net worth drop {n_drop_pct:.2f}% not sharp enough (< -15%)"

    # 3. Output and investment decline
    assert res.irf["Y"].iloc[0] < 0.0, "Output did not decline on impact"
    assert res.irf["I"].min() < 0.0, "Investment did not decline"

    # 4. Asset price Q declines
    assert res.irf["Q"].iloc[0] < 0.0, "Asset price Q did not drop on impact"


# ---------------------------------------------------------------------------
# Test 3: OccBin Solver & Credit Policy Regime Switching
# ---------------------------------------------------------------------------

def test_gk_occbin_credit_policy():
    """Verify OccBin piecewise-linear solution under credit policy intervention.

    Asserts:
    - Piecewise-linear algorithm converges.
    - Public credit intervention binds for positive periods (binding_periods > 0).
    - Government credit ratio psi > 0 during the intervention spell.
    - Credit spread spike is cushioned relative to linear Klein without intervention.
    """
    res_occ = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="occbin",
        constraint_type="credit_policy",
        threshold=0.0025,  # 100 bps annualized spread threshold
    )

    assert res_occ.solver_method == "occbin"
    assert res_occ.occbin_result is not None
    assert res_occ.occbin_result.converged is True
    assert res_occ.binding_periods > 0, "Credit policy constraint never bound"

    # Public credit assistance is positive during binding spell
    psi_binding = res_occ.irf["psi"].iloc[:res_occ.binding_periods]
    assert np.all(psi_binding.values > 0.0), "Credit policy psi was not positive during binding spell"

    # Net worth contracts sharply
    ss_N = res_occ.steady_state["N"]
    n_drop_pct = (res_occ.irf["N"].iloc[0] / ss_N) * 100.0
    assert n_drop_pct < -15.0, f"Net worth drop {n_drop_pct:.2f}% < -15%"

    # Credit policy cushions spread surge relative to Klein
    res_klein = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="klein",
    )
    occ_max_spread = res_occ.irf["prem"].max()
    klein_max_spread = res_klein.irf["prem"].max()
    assert occ_max_spread < klein_max_spread, (
        f"Credit policy failed to mitigate spread: OccBin {occ_max_spread:.4f} >= Klein {klein_max_spread:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Small Shock Matches Linear Klein to Machine Precision
# ---------------------------------------------------------------------------

def test_gk_occbin_small_shock_matches_klein():
    """Verify that under a small shock that never triggers the threshold,
    OccBin matches the pure linear Klein solution to numerical precision.
    """
    small_shock = -0.001
    res_klein = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=small_shock,
        horizon=40,
        method="klein",
    )
    res_occ = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=small_shock,
        horizon=40,
        method="occbin",
        constraint_type="credit_policy",
        threshold=0.0025,
    )

    assert res_occ.binding_periods == 0, "Small shock unexpectedly triggered binding constraint"
    assert res_occ.regimes == [0] * 40

    # Test numerical equivalence across all variables
    np.testing.assert_allclose(res_occ.irf.values, res_klein.irf.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: OccBin Leverage Cap Constraint
# ---------------------------------------------------------------------------

def test_gk_occbin_leverage_cap():
    """Verify OccBin solver under hard regulatory leverage cap."""
    res_cap = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="occbin",
        constraint_type="leverage_cap",
        threshold=0.0,
    )

    assert res_cap.solver_method == "occbin"
    assert res_cap.occbin_result.converged is True
    assert res_cap.binding_periods > 0
    # Leverage deviation is capped at threshold
    assert np.all(res_cap.irf["phi"].iloc[:res_cap.binding_periods].values <= 1e-6)


# ---------------------------------------------------------------------------
# Test 6: Alternative Shocks (TFP & Monetary Policy)
# ---------------------------------------------------------------------------

def test_gk_alternative_shocks():
    """Verify model response to technology (TFP) and monetary policy shocks."""
    # 1. Technology shock (positive 1% TFP innovation)
    # Under Calvo price stickiness (Galí 1999), technology shock lowers inflation
    # and hours worked while increasing consumption.
    res_tfp = solve_gertler_karadi(shock_type="tfp", shock_size=0.01, method="klein")
    assert res_tfp.irf["a"].iloc[0] > 0.0, "Technology level did not increase"
    assert res_tfp.irf["Pi"].iloc[0] < 0.0, "Positive TFP shock did not lower inflation"
    assert res_tfp.irf["C"].iloc[0] > 0.0, "Positive TFP shock did not increase consumption"
    assert res_tfp.irf["L"].iloc[0] < 0.0, "Hours worked did not contract under sticky prices"

    # 2. Contractionary monetary policy shock (positive 25 bps nominal rate shock)
    res_mon = solve_gertler_karadi(shock_type="monetary", shock_size=0.0025, method="klein")
    assert res_mon.irf["Rn"].iloc[0] > 0.0, "Monetary shock did not increase nominal rate"
    assert res_mon.irf["Y"].iloc[0] < 0.0, "Contractionary monetary shock did not depress output"
    assert res_mon.irf["Pi"].iloc[0] < 0.0, "Contractionary monetary shock did not reduce inflation"


# ---------------------------------------------------------------------------
# Test 7: Result Presentation Suite & Subscript Access
# ---------------------------------------------------------------------------

def test_gk_result_presentation_suite():
    """Verify GertlerKaradiResult implements the puremacro presentation contract."""
    res = solve_gertler_karadi(method="occbin", shock_size=-0.05)

    # 1. Subscript access
    y_series = res["Y"]
    assert isinstance(y_series, pd.Series)
    assert len(y_series) == 40
    assert res["solver_method"] == "occbin"

    with pytest.raises(KeyError):
        _ = res["non_existent_variable"]

    # 2. .to_frame()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40

    # 3. .summary()
    summary_text = res.summary()
    assert isinstance(summary_text, str)
    assert "GERTLER-KARADI" in summary_text
    assert "CALIBRATION" in summary_text
    assert "TRAJECTORY" in summary_text

    # 4. .to_markdown()
    md_text = res.to_markdown()
    assert isinstance(md_text, str)
    assert "|" in md_text

    # 5. .to_latex()
    latex_text = res.to_latex()
    assert isinstance(latex_text, str)
    assert "\\begin{tabular}" in latex_text

    # 6. .to_typst()
    typst_text = res.to_typst()
    assert isinstance(typst_text, str)
    assert "#table(" in typst_text

    # 7. .plot()
    fig = res.plot(style="publication")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig_default = res.plot(variables=["Y", "prem", "N"], style="default")
    assert isinstance(fig_default, plt.Figure)
    plt.close(fig_default)


# ---------------------------------------------------------------------------
# Test 8: Custom Parameters & Sensitivity
# ---------------------------------------------------------------------------

def test_gk_custom_parameters():
    """Verify that solve_gertler_karadi correctly accepts custom parameter overrides."""
    custom_params = {
        "beta": 0.985,
        "theta_b": 0.965,
        "lambda_b": 0.35,
    }
    res = solve_gertler_karadi(params=custom_params, method="klein")
    assert res.steady_state["R"] == pytest.approx(1.0 / 0.985, rel=1e-5)
    assert res.params["theta_b"] == 0.965
    assert res.params["lambda_b"] == 0.35


# ---------------------------------------------------------------------------
# Test 9: Adversarial Input Validation
# ---------------------------------------------------------------------------

def test_gk_input_validation():
    """Verify helpful error messages on invalid shock types, methods, or regimes."""
    with pytest.raises(ValueError, match="unknown shock_type"):
        solve_gertler_karadi(shock_type="invalid_shock")

    with pytest.raises(ValueError, match="unknown method"):
        solve_gertler_karadi(method="invalid_solver")

    with pytest.raises(ValueError, match="unknown constraint_type"):
        solve_gertler_karadi(method="occbin", constraint_type="unknown_constraint")


# ---------------------------------------------------------------------------
# Test 10: Shock Aliases and Variable Horizons
# ---------------------------------------------------------------------------

def test_gk_aliases_and_horizons():
    """Verify shock aliases ('xi', 'technology', 'policy') and varying horizons."""
    # Test xi alias
    res_xi = solve_gertler_karadi(shock_type="xi", shock_size=-0.03, horizon=20, method="klein")
    assert len(res_xi.irf) == 20
    assert res_xi.irf["prem"].iloc[0] > 0.0

    # Test technology alias
    res_tech = solve_gertler_karadi(shock_type="technology", shock_size=0.01, horizon=30, method="klein")
    assert len(res_tech.irf) == 30
    assert res_tech.irf["a"].iloc[0] == pytest.approx(0.01, rel=1e-5)

    # Test policy alias
    res_pol = solve_gertler_karadi(shock_type="policy", shock_size=0.005, horizon=15, method="klein")
    assert len(res_pol.irf) == 15
    assert res_pol.irf["Rn"].iloc[0] > 0.0


# ---------------------------------------------------------------------------
# Test 11: LinearModel Architecture & Blanchard-Kahn
# ---------------------------------------------------------------------------

def test_gk_linear_model_properties():
    """Verify build_gertler_karadi_model satisfies Blanchard-Kahn conditions."""
    model = build_gertler_karadi_model()
    assert len(model.variables) == 26
    assert len(model.shocks) == 3
    assert len(model.states) > 0
    assert len(model.controls) > 0

    # Check that generalized eigenvalues correctly satisfy Blanchard-Kahn condition:
    # number of generalized eigenvalues with modulus > 1 equals number of forward-looking controls
    # The engine solves the stacked lead/lag system (lagged copies of the
    # states are the predetermined block, every current variable is
    # non-predetermined), so there are n_states + n_variables generalised
    # eigenvalues and Blanchard-Kahn requires exactly n_variables unstable ones.
    eigs = model.eigenvalues
    assert len(eigs) == model.n_states + len(model.variables)
    n_unstable = np.sum(np.abs(eigs) > 1.0 + 1e-6)
    assert n_unstable == len(model.variables)
    assert model.is_determinate
    assert model.solution.G.shape == (model.n_states, model.n_states)
    assert model.solution.F.shape == (model.n_controls, model.n_states)

