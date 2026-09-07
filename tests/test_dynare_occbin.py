"""Tests for OccBin (Guerrieri & Iacoviello 2015) piecewise-linear solver."""
import matplotlib
matplotlib.use("Agg")

import warnings

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


# ---------------------------------------------------------------------------
# Regression tests for the honesty of `converged` (2.4.x OccBin audit)
# ---------------------------------------------------------------------------


def _nk_regime_models(order=("is", "pc", "tr", "g"), extra_zlb_fiscal=0.0, accounting=False):
    """Build the reference/constrained NK pair with a configurable equation order.

    ``order`` names the equations in the order the model returns them, which the
    solver must be invariant to. ``extra_zlb_fiscal`` switches on a fiscal
    feedback that is active only in the constrained regime (so two equations
    differ between regimes). ``accounting`` adds a purely block-recursive
    identity ``w`` that feeds back into nothing, so it cannot change the
    solution for (y, pi, r, g) no matter where it sits in the equation list.
    """
    p = {
        "beta": 0.99, "sigma": 1.0, "kappa": 0.1, "phi_pi": 1.5,
        "phi_y": 0.125, "rho_g": 0.8, "r_ss": 0.01, "psi": float(extra_zlb_fiscal),
    }
    variables = ["y", "pi", "r", "g"] + (["w"] if accounting else [])
    shocks = ["eps_r", "eps_g"]
    ss = {v: 0.0 for v in variables}

    def eqs(lead, curr, lag, sh, pp, zlb):
        d = {
            "is": lambda: curr.y - lead.y + (curr.r - lead.pi) / pp.sigma - curr.g,
            "pc": lambda: curr.pi - pp.beta * lead.pi - pp.kappa * curr.y,
            "tr": (lambda: curr.r - (-pp.r_ss)) if zlb
                  else (lambda: curr.r - pp.phi_pi * curr.pi - pp.phi_y * curr.y - sh.eps_r),
            "g": (lambda: curr.g - pp.rho_g * lag.g - sh.eps_g - pp.psi * curr.y) if zlb
                 else (lambda: curr.g - pp.rho_g * lag.g - sh.eps_g),
            # Accounting identity: books the inflation cost only at the ZLB.
            "acc": (lambda: curr.w - curr.y - 0.5 * curr.pi) if zlb
                   else (lambda: curr.w - curr.y),
        }
        return [d[k]() for k in order]

    m_ref = build_dynare(
        lambda lead, curr, lag, sh, pp: eqs(lead, curr, lag, sh, pp, False),
        variables=variables, shocks=shocks, params=p, steady_state=ss,
    )
    m_zlb = build_dynare(
        lambda lead, curr, lag, sh, pp: eqs(lead, curr, lag, sh, pp, True),
        variables=variables, shocks=shocks, params=p, steady_state=ss,
        check_steady_state=False, strict=False,
    )
    c = OccBinConstraint(variable="r", threshold=-p["r_ss"], operator="<")
    return m_ref, m_zlb, c, variables, shocks


def _stacked_occbin_oracle(ref_model, cons_model, constraint, shocks_mat, horizon):
    """Independent oracle: solve the piecewise-linear path as ONE stacked system.

    Shares no code path with ``solve_occbin``'s backward recursion. For a regime
    vector r the whole path X_1..X_H solves the block system

        A_+^{r_t} X_{t+1} + A_0^{r_t} X_t + A_-^{r_t} X_{t-1} + B_u^{r_t} u_t + c^{r_t} = 0

    with X_0 = 0 and the terminal condition X_{H+1} = P_0 X_H, iterated over the
    regime vector until it reproduces itself. Returns (regime, path) or None if
    the regime iteration does not settle.
    """
    from puremacro.dsge.occbin import _extract_model_matrices

    Ap0, A00, Am0, Bu0, c0, _, variables, _ = _extract_model_matrices(ref_model)
    Ap1, A01, Am1, Bu1, c1, _, _, _ = _extract_model_matrices(cons_model, ref_model=ref_model)
    n = len(variables)
    idx = variables.index(constraint.variable)

    dr = ref_model.decision_rules()
    P0 = np.zeros((n, n))
    for s in ref_model.states:
        P0[:, variables.index(s)] = dr.ghx[s].values

    Ap, A0, Am, Bu, cc = (Ap0, Ap1), (A00, A01), (Am0, Am1), (Bu0, Bu1), (c0, c1)

    def path_for(reg):
        M = np.zeros((horizon * n, horizon * n))
        rhs = np.zeros(horizon * n)
        for t in range(1, horizon + 1):
            r = int(reg[t - 1])
            rows = slice((t - 1) * n, t * n)
            M[rows, (t - 1) * n:t * n] += A0[r]
            if t >= 2:
                M[rows, (t - 2) * n:(t - 1) * n] += Am[r]
            if t <= horizon - 1:
                M[rows, t * n:(t + 1) * n] += Ap[r]
            else:
                M[rows, (t - 1) * n:t * n] += Ap[r] @ P0
            rhs[rows] = -(Bu[r] @ shocks_mat[t - 1] + cc[r])
        return np.linalg.solve(M, rhs).reshape(horizon, n)

    # Shadow value from the reference equation the constrained regime replaces.
    diff = np.where(
        (np.linalg.norm(A00 - A01, axis=1) > 1e-8)
        | (np.linalg.norm(Ap0 - Ap1, axis=1) > 1e-8)
        | (np.linalg.norm(Am0 - Am1, axis=1) > 1e-8)
        | (np.linalg.norm(Bu0 - Bu1, axis=1) > 1e-8)
        | (np.abs(c0 - c1) > 1e-8)
    )[0]
    eq_row = next(int(r) for r in diff if abs(A00[r, idx]) > 1e-12)
    a = A00[eq_row, idx]

    def shadow_for(X):
        Xf = np.vstack([np.zeros(n), X])
        out = np.zeros(horizon)
        for t in range(horizon):
            x_next = X[t + 1] if t + 1 < horizon else P0 @ X[t]
            other = sum(A00[eq_row, j] * X[t, j] for j in range(n) if j != idx)
            out[t] = -(other + Ap0[eq_row] @ x_next + Am0[eq_row] @ Xf[t]
                       + Bu0[eq_row] @ shocks_mat[t] + c0[eq_row]) / a
        return out

    reg = np.zeros(horizon, dtype=int)
    seen = {tuple(reg)}
    for _ in range(200):
        X = path_for(reg)
        sh = shadow_for(X)
        new = np.array(
            [1 if constraint.evaluate(sh[t] if reg[t] == 1 else X[t, idx]) else 0
             for t in range(horizon)],
            dtype=int,
        )
        if np.array_equal(new, reg):
            return reg, X
        if tuple(new) in seen:
            return None
        seen.add(tuple(new))
        reg = new
    return None


def test_occbin_anticipated_shock_after_t1_respects_declared_regime():
    """A shock dated t>1 must not break the regime the solver says it solved.

    Before the 2.4.x fix the shock loading inside a spell was the REFERENCE
    regime's Q_0 for every t > 1, so the period-6 demand shock was fed into the
    ZLB spell through the Taylor-rule loading: the solver reported the rate
    pegged at the floor in period 6 while returning r = -0.0356, a 2.6pp
    violation of its own constrained equation, with converged=True.
    """
    ref, zlb, c, variables, shocks = _nk_regime_models()
    horizon = 40
    shock_seq = np.zeros((horizon, len(shocks)))
    shock_seq[0, shocks.index("eps_g")] = -0.020
    shock_seq[5, shocks.index("eps_g")] = -0.020   # anticipated at t=1, lands at t=6

    res = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=horizon)

    assert res.converged is True
    r_path = res.simulated_path["r"].values
    binding = np.flatnonzero(np.array(res.regimes) == 1)
    assert binding.size >= 6, "the anticipated second shock should extend the spell past t=6"
    assert 5 in binding, "period 6 must be inside the spell for this test to bite"

    # The declared constrained regime pegs r at the floor: every period the
    # solver calls constrained must actually satisfy that equation.
    np.testing.assert_allclose(r_path[binding], c.threshold, atol=1e-10)
    assert np.all(r_path >= c.threshold - 1e-12)

    # Cross-check the whole path against an independently stacked solution.
    oracle = _stacked_occbin_oracle(ref, zlb, c, shock_seq, horizon)
    assert oracle is not None
    reg_o, path_o = oracle
    np.testing.assert_array_equal(np.array(res.regimes), reg_o)
    np.testing.assert_allclose(res.simulated_path.values, path_o, atol=1e-12)


def test_occbin_spell_starting_after_t1_is_solved_not_silently_dropped():
    """A shock announced for t=5 must move the path, and the spell may start late.

    The old scalar-T* recursion could only express spells running from t=1, and
    it applied the contemporaneous shock loading at t=1 only: a shock dated t=5
    was dropped entirely and an identically-zero path came back with
    converged=True and binding_periods=0.
    """
    ref, zlb, c, variables, shocks = _nk_regime_models()
    horizon = 40
    shock_seq = np.zeros((horizon, len(shocks)))
    shock_seq[4, shocks.index("eps_g")] = -0.030   # announced at t=1, hits at t=5

    res = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=horizon)

    # The announced shock must actually reach the solver.
    assert np.max(np.abs(res.simulated_path.values)) > 1e-6
    assert res.converged is True
    assert res.binding_periods > 0
    assert np.all(res.simulated_path["r"].values >= c.threshold - 1e-12)

    oracle = _stacked_occbin_oracle(ref, zlb, c, shock_seq, horizon)
    assert oracle is not None
    reg_o, path_o = oracle
    np.testing.assert_array_equal(np.array(res.regimes), reg_o)
    np.testing.assert_allclose(res.simulated_path.values, path_o, atol=1e-12)


def test_occbin_spell_filling_the_horizon_is_not_reported_converged():
    """A spell that runs to the last simulated period leaves the terminal
    condition untested, so it cannot be reported as a converged solution."""
    ref, zlb, c, variables, shocks = _nk_regime_models()
    shock_seq = np.zeros((5, len(shocks)))
    shock_seq[0, shocks.index("eps_g")] = -0.06

    with pytest.warns(UserWarning, match="terminal condition"):
        short = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=5)
    assert short.converged is False
    assert short.regimes[-1] == 1

    # With enough room the same shock resolves inside the window.
    long_seq = np.zeros((60, len(shocks)))
    long_seq[0, shocks.index("eps_g")] = -0.06
    full = solve_occbin(ref, zlb, c, shock_sequence=long_seq, horizon=60)
    assert full.converged is True
    assert full.regimes[-1] == 0
    # The truncated answer is materially different, which is why it must not
    # be advertised as converged.
    assert abs(short.simulated_path["y"].iloc[0] - full.simulated_path["y"].iloc[0]) > 0.1


def test_occbin_regime_cycle_is_not_reported_converged():
    """When the regime guess cycles there is no fixed point to report.

    The old cycle-detection branch picked ``max(T_star, T_new)`` and set
    converged=True on a guess its own verification step had just rejected.
    """
    # Fiscal stabilisation that switches on only at the ZLB makes the regime
    # map non-monotone; the guess flips instead of settling.
    ref, zlb, c, variables, shocks = _nk_regime_models(extra_zlb_fiscal=0.10)
    horizon = 40
    shock_seq = np.zeros((horizon, len(shocks)))
    shock_seq[0, shocks.index("eps_g")] = -0.020

    with pytest.warns(UserWarning, match="cycled"):
        res = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=horizon)
    assert res.converged is False

    # The independent oracle confirms no regime sequence reproduces itself.
    assert _stacked_occbin_oracle(ref, zlb, c, shock_seq, horizon) is None


def test_occbin_answer_does_not_depend_on_equation_order():
    """Re-ordering the model's equations must not change the answer.

    ``eq_row`` used to be the first row differing between regimes, without
    checking that the constrained variable appears in it. Putting a
    block-recursive accounting identity (which differs across regimes but does
    not contain r, and cannot influence y/pi/r/g) first therefore selected the
    wrong row, silently fell back to a unit coefficient, and returned a shadow
    rate that was positive throughout the ZLB spell.
    """
    horizon = 40
    results = {}
    for order in (("is", "pc", "tr", "g", "acc"), ("acc", "is", "pc", "tr", "g")):
        ref, zlb, c, variables, shocks = _nk_regime_models(order=order, accounting=True)
        shock_seq = np.zeros((horizon, len(shocks)))
        shock_seq[0, shocks.index("eps_g")] = -0.020
        results[order[0]] = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=horizon)

    a, b = results["is"], results["acc"]
    assert a.converged is True and b.converged is True
    assert a.regimes == b.regimes
    np.testing.assert_allclose(a.simulated_path.values, b.simulated_path.values, atol=1e-12)
    np.testing.assert_allclose(
        a.shadow_path["r_shadow"].values, b.shadow_path["r_shadow"].values, atol=1e-12
    )

    # And the shadow rate must be what it claims to be: the notional rate, below
    # the floor for exactly the periods the constraint binds.
    n_bind = a.binding_periods
    assert n_bind > 0
    assert np.all(a.shadow_path["r_shadow"].values[:n_bind] < -0.01)


def test_occbin_validates_horizon_and_max_iter():
    """Invalid sizes raise a named ValueError, not IndexError/UnboundLocalError."""
    ref, zlb, c, variables, shocks = _nk_regime_models()
    shock_seq = np.array([0.0, -0.02])

    for bad in (0, -5):
        with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
            solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=bad)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            solve_occbin(ref, zlb, c, shock_sequence=shock_seq, max_iter=bad)


def test_occbin_relax_threshold_is_actually_used():
    """``relax_threshold``/``relax_operator`` are documented, so they must bite.

    The solver used to call ``constraint.evaluate`` on the shadow value inside
    a spell, which ignores both fields; setting them changed nothing.
    """
    ref, zlb, c, variables, shocks = _nk_regime_models()
    horizon = 40
    shock_seq = np.zeros((horizon, len(shocks)))
    shock_seq[0, shocks.index("eps_g")] = -0.020

    base = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=horizon)
    assert base.converged is True
    assert base.binding_periods == 5

    # Relax as soon as the notional rate climbs back above -0.02, i.e. earlier
    # than the default rule (which relaxes at the -0.01 floor).
    early = OccBinConstraint(
        variable="r", threshold=-0.01, operator="<",
        relax_threshold=-0.02, relax_operator=">=",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_occbin(ref, zlb, early, shock_sequence=shock_seq, horizon=horizon)
    assert res.binding_periods != base.binding_periods


def test_occbin_simulated_path_is_indexed_from_period_one():
    """`simulated_path` shares the 1-based period index used by plot()/summary()."""
    ref, zlb, c, variables, shocks = _nk_regime_models()
    shock_seq = np.zeros((12, len(shocks)))
    shock_seq[0, shocks.index("eps_g")] = -0.020
    res = solve_occbin(ref, zlb, c, shock_sequence=shock_seq, horizon=12)

    assert res.simulated_path.index[0] == 1
    assert res.simulated_path.index[-1] == 12
    assert res.simulated_path.index.name == "t"
    assert res.shadow_path.index[0] == 1
    # summary() labels row 0 "Impact (t=1)"; the frame now agrees.
    np.testing.assert_allclose(
        res.simulated_path.loc[1].values, res.simulated_path.iloc[0].values
    )
