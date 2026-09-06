"""Tests for 2nd-order DSGE perturbation with pruning (SGU 2004, Kim et al. 2008)."""
import numpy as np
import pytest

from puremacro.dsge import (
    LinearModel,
    ModelError,
    PrunedDSGESolution,
    build_dynare,
    load_mod,
    solve_dynare_2nd_order,
)


def rbc_model(lead, curr, lag, shocks, p):
    """Canonical RBC model equilibrium conditions in Dynare lead-lag form."""
    eq1 = curr.c ** (-p.gamma) - p.beta * lead.c ** (-p.gamma) * (
        p.alpha * np.exp(lead.a) * curr.k ** (p.alpha - 1.0) + 1.0 - p.delta
    )
    eq2 = curr.k - (
        np.exp(curr.a) * lag.k**p.alpha - curr.c + (1.0 - p.delta) * lag.k
    )
    eq3 = curr.a - (p.rho * lag.a + shocks.eps)
    return [eq1, eq2, eq3]


@pytest.fixture
def rbc_setup():
    alpha = 0.3
    beta = 0.99
    delta = 0.025
    gamma = 1.0
    rho = 0.8
    r_ss = 1.0 / beta - (1.0 - delta)
    k_ss = (alpha / r_ss) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss
    a_ss = 0.0
    params = {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma, "rho": rho}
    variables = ["k", "a", "c"]
    shocks = ["eps"]
    steady_state = {"k": k_ss, "a": a_ss, "c": c_ss}
    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
    }


def test_solve_dynare_2nd_order_rbc(rbc_setup):
    """Verify SGU (2004) second-order Sylvester solve on canonical RBC."""
    sol = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=2,
    )
    assert isinstance(sol, PrunedDSGESolution)
    assert sol.state_names == ("k", "a")
    assert sol.control_names == ("c",)
    assert sol.shock_names == ("eps",)

    # State transition G must be stable
    assert sol.is_stable
    assert np.all(np.abs(sol.eigenvalues) < 1.0)

    # Linear state technology 'a' must have identically zero Hessian
    np.testing.assert_allclose(sol.H_xx[1, :], 0.0, atol=1e-12)

    # Cross-derivatives d2/d(k)d(a) must equal d2/d(a)d(k)
    assert abs(sol.H_xx[0, 1] - sol.H_xx[0, 2]) < 1e-10
    assert abs(sol.G_xx[0, 1] - sol.G_xx[0, 2]) < 1e-10

    # Technology 'a' has no volatility drift
    assert abs(sol.H_sigmasigma[1]) < 1e-10

    # The resource constraint k_t + c_t = f(k_{t-1}, a_t) has no sigma
    # dependence, so the risk corrections of k and c cancel exactly.
    np.testing.assert_allclose(
        sol.H_sigmasigma[0] + sol.G_sigmasigma[0], 0.0, atol=1e-8
    )
    assert abs(sol.H_sigmasigma[0]) > 1e-3  # a genuine, nonzero risk correction

    # Before 2.3.1 the sigma-sigma equation used g_xx (h_u ⊗ h_u) in place of
    # next period's own shock curvature g_uu and reported ghs2[k] > 0 for this
    # calibration; the corrected solver (pinned to closed forms in
    # tests/test_dsge_engine_closed_form.py) gives ghs2[k] < 0 here and the
    # same value whether the model is written in levels or in logs.
    def rbc_logs(lead, curr, lag, shocks, p):
        return [
            np.exp(-p.gamma * curr.c)
            - p.beta * np.exp(-p.gamma * lead.c)
            * (p.alpha * np.exp(lead.a) * np.exp((p.alpha - 1.0) * curr.k) + 1.0 - p.delta),
            np.exp(curr.k)
            - (np.exp(curr.a) * np.exp(p.alpha * lag.k) - np.exp(curr.c) + (1.0 - p.delta) * np.exp(lag.k)),
            curr.a - (p.rho * lag.a + shocks.eps),
        ]

    ss = rbc_setup["steady_state"]
    sol_logs = build_dynare(
        rbc_logs,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state={"k": np.log(ss["k"]), "a": 0.0, "c": np.log(ss["c"])},
        order=2,
    )
    # d k / d sigma^2 in levels equals k_ss * (d log k / d sigma^2) to second order
    np.testing.assert_allclose(
        sol_logs.H_sigmasigma[0] * ss["k"], sol.H_sigmasigma[0], rtol=1e-6
    )
    assert sol.H_sigmasigma[0] < 0.0


def test_linear_model_solve_second_order(rbc_setup):
    """Verify LinearModel.solve_second_order() reproduces build_dynare(order=2)."""
    m1 = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=1,
    )
    assert isinstance(m1, LinearModel)

    m2 = m1.solve_second_order()
    assert isinstance(m2, PrunedDSGESolution)

    direct2 = solve_dynare_2nd_order(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
    )
    np.testing.assert_allclose(m2.H_xx, direct2.H_xx, atol=1e-12)
    np.testing.assert_allclose(m2.G_xx, direct2.G_xx, atol=1e-12)
    np.testing.assert_allclose(m2.H_sigmasigma, direct2.H_sigmasigma, atol=1e-12)
    np.testing.assert_allclose(m2.G_sigmasigma, direct2.G_sigmasigma, atol=1e-12)


def test_load_mod_order_2():
    """Verify load_mod with order=2 parses and simulates with pruning."""
    mod_text = """
    var c k a;
    varexo eps;
    parameters alpha beta delta gamma rho;

    alpha = 0.30;
    beta = 0.99;
    delta = 0.025;
    gamma = 1.0;
    rho = 0.80;

    model;
      c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
      k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
      a = rho * a(-1) + eps;
    end;

    initval;
      k = 38.0;
      a = 0.0;
      c = 2.0;
    end;
    """
    sol2 = load_mod(mod_text, order=2)
    assert isinstance(sol2, PrunedDSGESolution)
    assert sol2.state_names == ("k", "a")
    assert sol2.control_names == ("c",)

    # Simulate with pruning
    sim = sol2.simulate(periods=30, sigma=0.01, seed=42)
    assert len(sim.states) == 30
    assert len(sim.controls) == 30
    assert not sim.states.isna().any().any()
    assert not sim.controls.isna().any().any()

    # GIRF analysis
    girf = sol2.girf("eps", size=0.01, horizon=10, sigma=0.01)
    assert len(girf) == 11
    # Check technology persistence decay
    rho_est = girf["a"].iloc[2] / girf["a"].iloc[1]
    np.testing.assert_allclose(rho_est, 0.80, atol=1e-6)

    # Stochastic steady state
    sss = sol2.stochastic_steady_state(sigma=0.01)
    assert "states" in sss and "controls" in sss
    assert sss["states"]["k"] > 0.0
    np.testing.assert_allclose(sss["states"]["a"], 0.0, atol=1e-10)


def test_solve_second_order_unsupported_model():
    """Calling solve_second_order on model without lead-lag equations raises ModelError."""
    from puremacro.dsge.build import build

    def simple_eqs(xp, x, e, p):
        return [xp.x - p.rho * x.x - e.eps]

    m = build(
        simple_eqs,
        variables=["x"],
        states=["x"],
        shocks=["eps"],
        params={"rho": 0.5},
        steady_state={"x": 0.0},
    )
    with pytest.raises(ModelError, match="solve_second_order requires the model to be built with build_dynare"):
        m.solve_second_order()


def test_pruned_dsge_solution_dynare_parity(rbc_setup):
    """Verify PrunedDSGESolution matches Dynare's oo_.dr structure and methods."""
    sol = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=2,
    )
    dr = sol.decision_rules()
    assert dr.ghx.equals(sol.oo_dr.ghx)
    assert dr.ghx.equals(sol.dynare_dr.ghx)

    # Check shapes
    assert dr.ghx.shape == (3, 2)
    assert dr.ghu.shape == (3, 1)
    assert dr.ghxx.shape == (3, 4)
    assert dr.ghxu.shape == (3, 2)
    assert dr.ghuu.shape == (3, 1)
    assert len(dr.ghs2) == 3
    assert len(dr.ys) == 3

    # Dict-like access matching MATLAB oo_.dr struct
    np.testing.assert_allclose(dr["ghx"].to_numpy(), dr.ghx.to_numpy())
    np.testing.assert_allclose(dr["ghxx"].to_numpy(), dr.ghxx.to_numpy())
    np.testing.assert_allclose(dr["ghxu"].to_numpy(), dr.ghxu.to_numpy())
    np.testing.assert_allclose(dr["ghuu"].to_numpy(), dr.ghuu.to_numpy())
    np.testing.assert_allclose(dr["ghs2"].to_numpy(), dr.ghs2.to_numpy())

    with pytest.raises(KeyError, match="Dynare2ndDR has no field 'invalid_key'"):
        _ = dr["invalid_key"]

    # Table layout and export formats
    df = dr.to_frame()
    assert "Constant" in df.index
    assert "0.5 * ghs2" in df.index
    assert "k(-1)" in df.index
    assert "eps" in df.index
    assert list(df.columns) == list(rbc_setup["variables"])

    summary_str = dr.summary()
    assert "SECOND-ORDER POLICY AND TRANSITION FUNCTIONS" in summary_str
    assert "ghxx" in summary_str

    md_str = dr.to_markdown()
    assert "|" in md_str

    latex_str = dr.to_latex()
    assert "\\begin{tabular}" in latex_str or "\\begin{table}" in latex_str

    typst_str = dr.to_typst()
    assert "#table" in typst_str


def test_cross_terms_xu_uu(rbc_setup):
    """Verify second-order cross-terms (H_xu, H_uu, G_xu, G_uu) are non-trivial."""
    sol = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=2,
    )
    # H_xu: (n_x, n_x * n_e) = (2, 2)
    assert sol.H_xu.shape == (2, 2)
    # H_uu: (n_x, n_e * n_e) = (2, 1)
    assert sol.H_uu.shape == (2, 1)
    # G_xu: (n_y, n_x * n_e) = (1, 2)
    assert sol.G_xu.shape == (1, 2)
    # G_uu: (n_y, n_e * n_e) = (1, 1)
    assert sol.G_uu.shape == (1, 1)

    # Technology shock innovation creates positive curvature on capital accumulation
    assert sol.H_uu[0, 0] > 0.0
    # Shock enters technology purely linearly: H_uu for 'a' is 0
    np.testing.assert_allclose(sol.H_uu[1, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol.H_xu[1, :], 0.0, atol=1e-12)


def test_pruned_theoretical_moments(rbc_setup):
    """Verify theoretical_moments() works on PrunedDSGESolution."""
    sol = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=2,
    )
    tm = sol.theoretical_moments(sigma=0.01, lags=3)
    assert list(tm.moments.columns) == ["Mean", "Std.Dev.", "Variance"]
    assert len(tm.moments) == 3
    assert np.all(tm.moments["Std.Dev."] > 0.0)
    assert tm.covariance.shape == (3, 3)
    assert tm.correlation.shape == (3, 3)
    np.testing.assert_allclose(np.diag(tm.correlation), 1.0)
    assert tm.autocorr.shape == (3, 3)
    assert "Lag 1" in tm.autocorr.columns


def test_linear_model_oo_dr_dict_access(rbc_setup):
    """Verify LinearModel.oo_dr property and __getitem__ support."""
    m1 = build_dynare(
        rbc_model,
        variables=rbc_setup["variables"],
        shocks=rbc_setup["shocks"],
        params=rbc_setup["params"],
        steady_state=rbc_setup["steady_state"],
        order=1,
    )
    dr1 = m1.oo_dr
    assert dr1.ghx.equals(m1.decision_rules().ghx)
    np.testing.assert_allclose(dr1["ghx"].to_numpy(), dr1.ghx.to_numpy())
    np.testing.assert_allclose(dr1["ghu"].to_numpy(), dr1.ghu.to_numpy())
    with pytest.raises(KeyError, match="DynareDR has no field 'bad_key'"):
        _ = dr1["bad_key"]


def test_load_mod_shocks_and_stoch_simul(tmp_path):
    """Verify load_mod parses shocks block and stoch_simul options."""
    mod_text = """
    var c, k, a;
    varexo eps;
    parameters alpha, beta, delta, gamma, rho;

    alpha = 0.30;
    beta = 0.99;
    delta = 0.025;
    gamma = 1.0;
    rho = 0.80;

    model;
      c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
      k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
      a = rho * a(-1) + eps;
    end;

    initval;
      k = 21.436;
      a = 0.0;
      c = 1.972;
    end;

    shocks;
      var eps; stderr 0.015;
    end;

    stoch_simul(order=2, pruning, irf=15);
    """
    p = tmp_path / "rbc_simul.mod"
    p.write_text(mod_text, encoding="utf-8")

    # Call load_mod with default order=None -> should pick up order=2 from stoch_simul
    sol = load_mod(p)
    assert isinstance(sol, PrunedDSGESolution)
    assert sol.state_names == ("k", "a")
    assert sol.control_names == ("c",)

    # Check that shock_cov was parsed from stderr 0.015
    sss = sol.stochastic_steady_state()
    assert sss["states"]["k"] > 0.0
