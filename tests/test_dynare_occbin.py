"""Tests for OccBin (Guerrieri & Iacoviello 2015) piecewise-linear solver."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, OccBinConstraint, OccBinResult, solve_occbin


@pytest.fixture
def nk_model_setup():
    """Set up analytical 3-equation New Keynesian model with Taylor rule and ZLB."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.1,
        "phi_pi": 1.5,
        "phi_y": 0.125,
        "rho_g": 0.8,
        "r_ss": 0.01,
    }

    variables = ["y", "pi", "r", "g"]
    shocks = ["eps_r", "eps_g"]
    steady_state = {v: 0.0 for v in variables}

    # Reference regime: standard Taylor rule
    def nk_ref(lead, curr, lag, shocks_v, p):
        return [
            # Dynamic IS
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            # New Keynesian Phillips Curve
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            # Taylor rule with policy shock
            curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
            # Exogenous demand shock process
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Constrained regime: Zero Lower Bound r_t = -r_ss
    def nk_cons(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    ref_model = build_dynare(
        nk_ref,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
    )

    cons_model = build_dynare(
        nk_cons,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
    )

    constraint = OccBinConstraint(
        variable="r",
        threshold=-params["r_ss"],
        operator="<",
    )

    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "ref_model": ref_model,
        "cons_model": cons_model,
        "nk_cons_fn": nk_cons,
        "constraint": constraint,
    }


def test_occbin_constraint_dataclass():
    """Verify OccBinConstraint properties and evaluation logic."""
    c = OccBinConstraint(variable="r", threshold=-0.01, operator="<")
    assert c.variable == "r"
    assert c.threshold == -0.01
    assert c.operator == "<"
    assert c.evaluate(-0.02) is True
    assert c.evaluate(-0.005) is False
    assert c.evaluate_relax(-0.005) is True
    assert c.evaluate_relax(-0.02) is False

    # Test invalid operator
    with pytest.raises(ValueError, match="invalid operator"):
        OccBinConstraint(variable="r", threshold=0.0, operator="!=")


def test_occbin_small_shock_matches_linear(nk_model_setup):
    """Test 1: Small shock that does not hit ZLB matches pure linear model solution exactly."""
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    variables = nk_model_setup["variables"]
    horizon = 40

    # Small demand shock that leaves r > -r_ss
    shock_seq = np.array([0.0, -0.002])
    res = solve_occbin(ref, cons, constraint, shock_sequence=shock_seq, horizon=horizon)

    assert isinstance(res, OccBinResult)
    assert res.converged is True
    assert res.binding_periods == 0
    assert res.iterations == 1
    assert res.regimes == [0] * horizon
    assert len(res.simulated_path) == horizon

    # Pure linear model solution
    dr = ref.decision_rules()
    P_0 = np.zeros((len(variables), len(variables)))
    for s in ref.states:
        idx_s = variables.index(s)
        P_0[:, idx_s] = dr.ghx[s].values

    lin_sim = np.zeros((horizon, len(variables)))
    lin_sim[0] = dr.ghu.values @ shock_seq
    for t in range(1, horizon):
        lin_sim[t] = P_0 @ lin_sim[t - 1]

    # Exact equality down to machine precision
    np.testing.assert_allclose(res.simulated_path.values, lin_sim, atol=1e-12)


def test_occbin_large_negative_demand_shock_hitting_zlb(nk_model_setup):
    """Test 2: Large negative demand shock hitting ZLB for 4 periods.

    Asserts:
    - Constraint holds and binds for exactly 4 periods.
    - Regime transitions smoothly from constrained (1) to reference (0).
    - Nominal rate never drops below the floor (-r_ss).
    """
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    params = nk_model_setup["params"]
    horizon = 40

    # Large demand shock hitting ZLB
    shock_seq = np.array([0.0, -0.020])
    res = solve_occbin(ref, cons, constraint, shock_sequence=shock_seq, horizon=horizon)

    assert res.converged is True
    assert res.binding_periods == 4
    assert res.regimes[:4] == [1, 1, 1, 1]
    assert res.regimes[4:] == [0] * (horizon - 4)

    # 1. Rate never drops below floor
    r_path = res.simulated_path["r"]
    assert np.all(r_path.values >= -params["r_ss"] - 1e-12), "r dropped below ZLB floor"

    # 2. Rate is clamped at floor for exactly 4 periods
    np.testing.assert_allclose(r_path.iloc[:4].values, -params["r_ss"], atol=1e-10)

    # 3. Rate smoothly lifts off above floor at t=5 (index 4)
    assert r_path.iloc[4] > -params["r_ss"] + 1e-5
    assert r_path.iloc[5] > r_path.iloc[4]  # returning toward steady state

    # 4. Shadow rate verification
    assert res.shadow_path is not None
    shadow_r = res.shadow_path["r_shadow"]
    # Shadow rate is below floor during binding spell
    assert np.all(shadow_r.iloc[:4] < -params["r_ss"])
    # Shadow rate lifts above floor at lift-off
    assert shadow_r.iloc[4] >= -params["r_ss"]


def test_occbin_monotonicity_adversarial(nk_model_setup):
    """Test 3: Adversarial check with varying shock sizes asserting monotonicity of binding duration."""
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    params = nk_model_setup["params"]
    horizon = 40

    # Shock sizes ranging from small to severe
    shock_sizes = [-0.002, -0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.040]
    durations = []

    for s in shock_sizes:
        res = solve_occbin(ref, cons, constraint, shock_sequence=np.array([0.0, s]), horizon=horizon)
        assert res.converged is True, f"Failed to converge for shock size {s}"
        # Rate floor must never be violated under any shock
        assert np.all(res.simulated_path["r"].values >= -params["r_ss"] - 1e-12)
        durations.append(res.binding_periods)

    # Assert monotonicity: larger shocks produce equal or longer binding spells
    for i in range(len(durations) - 1):
        assert durations[i] <= durations[i + 1], (
            f"Monotonicity violated: shock {shock_sizes[i]} -> {durations[i]} periods, "
            f"shock {shock_sizes[i+1]} -> {durations[i+1]} periods"
        )


def test_occbin_callable_constrained_model(nk_model_setup):
    """Verify solve_occbin also works when constrained_model is a callable function."""
    ref = nk_model_setup["ref_model"]
    cons_fn = nk_model_setup["nk_cons_fn"]
    constraint = nk_model_setup["constraint"]
    params = nk_model_setup["params"]

    shock_seq = np.array([0.0, -0.020])
    res = solve_occbin(ref, cons_fn, constraint, shock_sequence=shock_seq, horizon=40)

    assert res.converged is True
    assert res.binding_periods == 4
    np.testing.assert_allclose(res.simulated_path["r"].iloc[:4].values, -params["r_ss"], atol=1e-10)


def test_occbin_reports_and_plotting(nk_model_setup):
    """Test .to_latex(), .to_typst(), .to_markdown(), .summary(), and .plot()."""
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]

    shock_seq = np.array([0.0, -0.020])
    res = solve_occbin(ref, cons, constraint, shock_sequence=shock_seq, horizon=30)

    # 1. to_frame and subscript access
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert "r" in df.columns
    np.testing.assert_array_equal(res["r"].values, df["r"].values)

    # 2. to_markdown
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "|" in md
    assert "r" in md

    # 3. to_latex
    ltx = res.to_latex()
    assert isinstance(ltx, str)
    assert r"\begin{tabular}" in ltx
    assert r"\end{tabular}" in ltx

    # 4. to_typst
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ

    # 5. summary
    summary_str = res.summary()
    assert isinstance(summary_str, str)
    assert "OCCASIONALLY BINDING CONSTRAINTS REPORT" in summary_str
    assert "Binding duration   : 4 period(s)" in summary_str
    assert "Algorithm status   : Converged" in summary_str

    # 6. plot publication style
    fig = res.plot(style="publication")
    assert fig is not None
    assert len(fig.axes) == 4  # 4 variables

    # Plot subset of variables
    fig_sub = res.plot(variables=["y", "r"], style="default")
    assert fig_sub is not None
    assert len(fig_sub.axes) == 2
