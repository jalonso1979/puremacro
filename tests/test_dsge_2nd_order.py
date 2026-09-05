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

    # Precautionary behavior: risk expands capital and contracts consumption
    assert sol.H_sigmasigma[0] > 0.0  # precautionary capital accumulation
    assert sol.G_sigmasigma[0] < 0.0  # precautionary savings / consumption cut
    np.testing.assert_allclose(
        sol.H_sigmasigma[0] + sol.G_sigmasigma[0], 0.0, atol=1e-8
    )


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
