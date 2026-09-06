"""Tests for OccBin (Guerrieri & Iacoviello 2015) piecewise-linear solver."""
import warnings

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

    # The constrained regime (a pegged nominal rate) is indeterminate on its
    # own; OccBin only needs its Jacobians, so it is built with strict=False.
    cons_model = build_dynare(
        nk_cons,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
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
    """Test 2: Large negative demand shock hitting ZLB for 5 periods.

    The reference regime's linear rule is y_t = a g_t, pi_t = b g_t with
    a = 1/((1-rho) + phi_y/sigma + (phi_pi-rho) kappa/((1-beta rho) sigma)),
    so the impact response to eps_g is a*eps_g (Dynare ghu). Before 2.3.1 the
    control rows of ghx/ghu were F@G and F@N+L (impact a*(1+rho)*eps_g), which
    fed a wrong reference path into the OccBin recursion and produced a
    4-period spell; the corrected rules give 5 periods.

    Asserts:
    - Constraint holds and binds for exactly 5 periods.
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
    n_bind = 5
    assert res.binding_periods == n_bind
    assert res.regimes[:n_bind] == [1] * n_bind
    assert res.regimes[n_bind:] == [0] * (horizon - n_bind)

    # 0. The reference regime's impact response is the closed-form a*eps_g
    p = params
    a = 1.0 / ((1 - p["rho_g"]) + p["phi_y"] / p["sigma"]
               + (p["phi_pi"] - p["rho_g"]) * p["kappa"] / ((1 - p["beta"] * p["rho_g"]) * p["sigma"]))
    np.testing.assert_allclose(ref.decision_rules().ghu.loc["y", "eps_g"], a, rtol=1e-10)

    # 1. Rate never drops below floor
    r_path = res.simulated_path["r"]
    assert np.all(r_path.values >= -params["r_ss"] - 1e-12), "r dropped below ZLB floor"

    # 2. Rate is clamped at floor for exactly n_bind periods
    np.testing.assert_allclose(r_path.iloc[:n_bind].values, -params["r_ss"], atol=1e-10)

    # 3. Rate smoothly lifts off above floor at the first unconstrained period
    assert r_path.iloc[n_bind] > -params["r_ss"] + 1e-5
    assert r_path.iloc[n_bind + 1] > r_path.iloc[n_bind]  # returning toward steady state

    # 4. Shadow rate verification
    assert res.shadow_path is not None
    shadow_r = res.shadow_path["r_shadow"]
    # Shadow rate is below floor during binding spell
    assert np.all(shadow_r.iloc[:n_bind] < -params["r_ss"])
    # Shadow rate lifts above floor at lift-off
    assert shadow_r.iloc[n_bind] >= -params["r_ss"]


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
    # 5 periods with the corrected reference rule (see test 2 above)
    assert res.binding_periods == 5
    np.testing.assert_allclose(res.simulated_path["r"].iloc[:5].values, -params["r_ss"], atol=1e-10)


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
    assert "Binding duration   : 5 period(s)" in summary_str
    assert "Algorithm status   : Converged" in summary_str

    # 6. plot publication style
    fig = res.plot(style="publication")
    assert fig is not None
    assert len(fig.axes) == 4  # 4 variables

    # Plot subset of variables
    fig_sub = res.plot(variables=["y", "r"], style="default")
    assert fig_sub is not None
    assert len(fig_sub.axes) == 2


def test_occbin_late_shock_unconstrained_matches_linear(nk_model_setup):
    """A shock arriving after t=1 with no binding spell must still propagate.

    Regression: the ``T_star == 0`` branch used to hand back a zero shock loading for
    every period after the first, so any innovation dated t > 1 was silently discarded
    and the solver returned an identically zero path with ``converged=True``.
    """
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    horizon = 20
    n_shocks = len(nk_model_setup["shocks"])

    # Same small demand shock, once at t=1 and once at t=4. Too small to reach the ZLB,
    # so both runs are pure linear model and the second is the first shifted by 3.
    early = np.zeros((horizon, n_shocks))
    early[0, 1] = -0.002
    late = np.zeros((horizon, n_shocks))
    late[3, 1] = -0.002

    res_early = solve_occbin(ref, cons, constraint, shock_sequence=early, horizon=horizon)
    res_late = solve_occbin(ref, cons, constraint, shock_sequence=late, horizon=horizon)

    assert res_early.binding_periods == 0
    assert res_late.binding_periods == 0

    # The late shock must do something at all ...
    assert np.abs(res_late["y"].values).max() > 1e-8, "shock dated t > 1 was discarded"

    # ... and specifically the same thing, three periods later.
    for var in nk_model_setup["variables"]:
        np.testing.assert_allclose(
            res_late[var].values[3:], res_early[var].values[: horizon - 3], atol=1e-12
        )
        np.testing.assert_allclose(res_late[var].values[:3], 0.0, atol=1e-12)


def test_occbin_shock_inside_spell_respects_the_bound(nk_model_setup):
    """A shock landing inside a binding spell must not push the variable off its bound.

    Regression: the shock loading was assigned by date rather than by regime, so for
    1 < t <= T* an innovation was transmitted through the reference-regime matrix. In
    the ZLB model that let a policy shock -- absent from the pegged-rate equation -- move
    the nominal rate straight through its own floor while the solver still reported the
    spell as binding and converged.
    """
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    r_ss = nk_model_setup["params"]["r_ss"]
    horizon = 20
    n_shocks = len(nk_model_setup["shocks"])

    base = np.zeros((horizon, n_shocks))
    base[0, 1] = -0.025
    res_base = solve_occbin(ref, cons, constraint, shock_sequence=base, horizon=horizon)
    assert res_base.binding_periods >= 4, "fixture must produce a multi-period spell"

    # Policy shock strictly inside the spell, where the rate is pegged at -r_ss
    inside = base.copy()
    inside[2, 0] = -0.005
    assert 2 < res_base.binding_periods
    res = solve_occbin(ref, cons, constraint, shock_sequence=inside, horizon=horizon)

    # The bound holds over the whole spell ...
    assert np.all(res["r"].values[: res.binding_periods] >= -r_ss - 1e-10)
    # ... and, since the constrained regime does not contain eps_r, the pegged stretch is
    # bit-for-bit what it was without the shock.
    np.testing.assert_allclose(
        res["r"].values[: res.binding_periods],
        res_base["r"].values[: res_base.binding_periods][: res.binding_periods],
        atol=1e-12,
    )


def test_occbin_bound_holds_for_shocks_arriving_within_the_spell(nk_model_setup):
    """Sweep the arrival date of a second demand shock inside the spell; the floor holds.

    Arrivals 0-5 land while the constraint is still binding, so they only lengthen the
    single spell the solver tracks. The floor must hold in every one of those runs, with
    no warning.
    """
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    r_ss = nk_model_setup["params"]["r_ss"]
    horizon = 24
    n_shocks = len(nk_model_setup["shocks"])

    for arrival in range(0, 6):
        seq = np.zeros((horizon, n_shocks))
        seq[0, 1] = -0.020
        seq[arrival, 1] += -0.010
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # any re-binding warning fails the test
            res = solve_occbin(ref, cons, constraint, shock_sequence=seq, horizon=horizon)
        assert res.converged is True, f"failed to converge for arrival {arrival}"
        assert np.all(res["r"].values >= -r_ss - 1e-10), (
            f"ZLB floor violated when the shock arrives at index {arrival}: "
            f"min r = {res['r'].values.min()}"
        )


def test_occbin_warns_when_the_constraint_binds_again_after_the_spell(nk_model_setup):
    """A second, disjoint spell is outside the single-spell design -- say so, don't hide it.

    The duration loop counts only periods that bind consecutively from t=1. A shock landing
    after the spell has ended can push the economy back against the bound; that second spell
    cannot be represented, and the returned path violates the constraint there. The solver
    must flag it instead of returning the violating path silently.
    """
    ref = nk_model_setup["ref_model"]
    cons = nk_model_setup["cons_model"]
    constraint = nk_model_setup["constraint"]
    r_ss = nk_model_setup["params"]["r_ss"]
    horizon = 24
    n_shocks = len(nk_model_setup["shocks"])

    seq = np.zeros((horizon, n_shocks))
    seq[0, 1] = -0.020
    seq[6, 1] = -0.010          # arrives after the first spell has ended

    with pytest.warns(UserWarning, match="binds again at period"):
        res = solve_occbin(ref, cons, constraint, shock_sequence=seq, horizon=horizon)

    # The warning is a guard, not a fix: the path really does breach the bound there.
    assert np.any(res["r"].values[res.binding_periods :] < -r_ss - 1e-10)
