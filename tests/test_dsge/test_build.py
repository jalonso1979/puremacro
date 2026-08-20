"""Tests for puremacro.dsge.build — the model DSL and complex-step Jacobians.

Anchored on the one model with a closed form: neoclassical growth with
full depreciation and log utility, where

    k_{t+1} = alpha*beta*z_t*k_t^alpha,   c_t = (1 - alpha*beta)*z_t*k_t^alpha

so the log-linear solution is exactly ``k^' = alpha*k^ + z^`` and
``c^ = alpha*k^ + z^`` — every entry of G, F, N and L is known in advance.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro import dsge
from puremacro.dsge.klein import BlanchardKahnError

ALPHA, BETA, RHO = 0.33, 0.98, 0.9
PARAMS = dict(alpha=ALPHA, beta=BETA, rho=RHO)
GUESS = dict(c=0.5, k=0.1, z=1.0)


def growth_equations(xp, x, e, p):
    return [
        1 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]


def build_growth(**kwargs):
    options = dict(variables=["c", "k", "z"], states=["k", "z"], shocks=["eps"],
                   params=PARAMS, guess=GUESS)
    options.update(kwargs)
    return dsge.build(growth_equations, **options)


@pytest.fixture(scope="module")
def model():
    return build_growth()


# --- steady state -------------------------------------------------------

@pytest.mark.pyodide_smoke
def test_steady_state_matches_closed_form(model):
    k = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    assert model.steady_state["k"] == pytest.approx(k, rel=1e-10)
    assert model.steady_state["c"] == pytest.approx((1 - ALPHA * BETA) * k ** ALPHA, rel=1e-10)
    assert model.steady_state["z"] == pytest.approx(1.0, rel=1e-10)
    assert model.residual_norm < 1e-12


def test_supplied_steady_state_is_verified_not_trusted():
    k = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    good = dict(c=(1 - ALPHA * BETA) * k ** ALPHA, k=k, z=1.0)
    built = build_growth(steady_state=good, guess=None)
    assert built.residual_norm < 1e-12

    with pytest.raises(dsge.SteadyStateError, match="does not solve"):
        build_growth(steady_state=dict(c=1.0, k=1.0, z=1.0), guess=None)


def test_missing_guess_or_steady_state_is_an_error():
    with pytest.raises(dsge.ModelError, match="steady_state=|guess="):
        build_growth(guess=None)
    with pytest.raises(dsge.ModelError, match="missing values"):
        build_growth(guess=dict(c=0.5, k=0.1))


def test_hopeless_guess_reports_failure_to_converge():
    with pytest.raises(dsge.SteadyStateError, match="did not converge"):
        build_growth(guess=dict(c=-50.0, k=-50.0, z=-50.0))


# --- solution -----------------------------------------------------------

@pytest.mark.pyodide_smoke
def test_solution_matches_closed_form(model):
    np.testing.assert_allclose(model.solution.G, [[ALPHA, 1.0], [0.0, RHO]], atol=1e-9)
    np.testing.assert_allclose(model.solution.F, [[ALPHA, 1.0]], atol=1e-9)
    np.testing.assert_allclose(model.solution.N.ravel(), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(model.solution.L.ravel(), [0.0], atol=1e-9)


def test_complex_and_central_differentiation_agree(model):
    central = build_growth(method="central")
    assert central.method == "central"
    np.testing.assert_allclose(central.solution.G, model.solution.G, atol=1e-7)
    np.testing.assert_allclose(central.solution.F, model.solution.F, atol=1e-7)


def test_non_analytic_residual_is_diagnosed():
    def kinked(xp, x, e, p):
        # abs() destroys the imaginary part complex-step relies on.
        return [abs(x.c) - 1.0, abs(x.k) - 1.0, xp.z - x.z]

    with pytest.raises(dsge.ModelError, match="not analytic|all-zero Jacobian"):
        dsge.build(kinked, variables=["c", "k", "z"], states=["k", "z"],
                   shocks=["eps"], params={},
                   steady_state=dict(c=1.0, k=1.0, z=0.0))


def test_indeterminate_model_raises_under_strict():
    # Two predetermined states and no forward-looking root: BK violated.
    def bad(xp, x, e, p):
        return [xp.a - 2.0 * x.a - e.u, xp.b - 3.0 * x.b, x.c - x.a]

    with pytest.raises(BlanchardKahnError):
        dsge.build(bad, variables=["a", "b", "c"], states=["a", "b"],
                   shocks=["u"], params={},
                   steady_state=dict(a=0.0, b=0.0, c=0.0))


# --- reporting ----------------------------------------------------------

def test_irf_matches_the_analytic_path(model):
    irf = model.irf("eps", horizon=3)
    assert list(irf.columns) == ["c", "k", "z"]
    assert irf.index.name == "h"
    # h=0: the states are still at zero (the innovation moves them into
    # t+1), and here consumption is too, because with full depreciation
    # and log utility c_t depends on current z and k alone — there is no
    # anticipation channel. A forward-looking control generally would move.
    np.testing.assert_allclose(irf.loc[0].to_numpy(), [0.0, 0.0, 0.0], atol=1e-9)
    # h=1: z jumps by 1, k is still at its old value, c = alpha*k + z = 1.
    np.testing.assert_allclose(irf.loc[1].to_numpy(), [1.0, 0.0, 1.0], atol=1e-9)
    # h=2: k = 1, z = rho, c = alpha*1 + rho.
    np.testing.assert_allclose(
        irf.loc[2].to_numpy(), [ALPHA + RHO, 1.0, RHO], atol=1e-9)


def test_irf_scales_linearly(model):
    np.testing.assert_allclose(
        model.irf("eps", horizon=5, size=2.5).to_numpy(),
        2.5 * model.irf("eps", horizon=5).to_numpy(), atol=1e-12)


def test_irf_rejects_an_unknown_shock(model):
    with pytest.raises(dsge.ModelError, match="no shock named"):
        model.irf("monetary")


def test_units_are_reported(model):
    assert model.units == {"c": "log", "k": "log", "z": "log"}


def test_non_positive_steady_state_falls_back_to_levels():
    def with_zero_ss(xp, x, e, p):
        return [xp.s - p.rho * x.s - e.u, x.g - x.s]

    built = dsge.build(with_zero_ss, variables=["s", "g"], states=["s"],
                       shocks=["u"], params=dict(rho=0.5),
                       steady_state=dict(s=0.0, g=0.0))
    assert built.units == {"s": "level", "g": "level"}


def test_policy_table_is_labelled(model):
    policy = model.policy()
    assert list(policy.index) == ["c", "k", "z"]
    assert list(policy.columns) == ["k", "z"]
    assert policy.loc["c", "z"] == pytest.approx(1.0, abs=1e-9)


def test_simulate_shape_and_reproducibility(model):
    a = model.simulate(periods=100, sigma={"eps": 0.01}, seed=7)
    b = model.simulate(periods=100, sigma={"eps": 0.01}, seed=7)
    assert a.shape == (100, 3)
    assert list(a.columns) == ["c", "k", "z"]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(),
                           model.simulate(periods=100, sigma=0.01, seed=8).to_numpy())


def test_summary_reports_the_bk_verdict(model):
    text = model.summary()
    assert "unique stable solution" in text
    assert "complex-step" in text


# --- declaration errors -------------------------------------------------

def test_unknown_state_name():
    with pytest.raises(dsge.ModelError, match="not in variables"):
        build_growth(states=["k", "wealth"])


def test_states_are_required():
    with pytest.raises(dsge.ModelError, match="at least one predetermined"):
        build_growth(states=[])


def test_duplicate_names():
    with pytest.raises(dsge.ModelError, match="duplicate variable"):
        build_growth(variables=["c", "c", "z"])


def test_wrong_equation_count():
    def two_equations(xp, x, e, p):
        return [x.c - 1.0, xp.k - x.k]

    with pytest.raises(dsge.ModelError, match="one equation per variable"):
        dsge.build(two_equations, variables=["c", "k", "z"], states=["k", "z"],
                   shocks=["eps"], params={},
                   steady_state=dict(c=1.0, k=1.0, z=1.0))


def test_named_access_styles_are_equivalent(model):
    def by_index(xp, x, e, p):
        c, k, z = x
        c1, k1, z1 = xp
        return [
            1 / c - p["beta"] * (p["alpha"] * z1 * k1 ** (p["alpha"] - 1)) / c1,
            c + k1 - z * k ** p["alpha"],
            z1 - z ** p["rho"] * np.exp(e[0]),
        ]

    other = dsge.build(by_index, variables=["c", "k", "z"], states=["k", "z"],
                       shocks=["eps"], params=PARAMS, guess=GUESS)
    np.testing.assert_allclose(other.solution.G, model.solution.G, atol=1e-12)


def test_helpful_message_for_a_misspelled_variable():
    def typo(xp, x, e, p):
        return [x.consumption - 1.0, xp.k - x.k, xp.z - x.z]

    with pytest.raises(AttributeError, match="no variable named 'consumption'"):
        dsge.build(typo, variables=["c", "k", "z"], states=["k", "z"],
                   shocks=["eps"], params={},
                   steady_state=dict(c=1.0, k=1.0, z=1.0))


# --- a forward-looking model, checked against its own equations ---------
# The three-equation New Keynesian block: unlike the growth model above it
# has genuinely forward-looking controls, so it exercises the shock
# loading L on both a contemporaneous shock and an anticipated one.

NK = dict(beta=0.99, sigma=1.0, kappa=0.1275, phi_pi=1.5, phi_x=0.125, rho_r=0.9)


def nk_equations(xp, x, e, p):
    return [
        xp.rn - p.rho_r * x.rn - e.eps_demand,
        xp.x - x.x - (x.i - xp.pi - x.rn) / p.sigma,
        p.beta * xp.pi + p.kappa * x.x - x.pi,
        p.phi_pi * x.pi + p.phi_x * x.x + e.eps_policy - x.i,
    ]


def build_nk(**overrides):
    return dsge.build(
        nk_equations, variables=["rn", "x", "pi", "i"], states=["rn"],
        shocks=["eps_demand", "eps_policy"], params={**NK, **overrides},
        steady_state={k: 0.0 for k in ("rn", "x", "pi", "i")},
        linearize="level",
    )


@pytest.mark.parametrize("shock", ["eps_demand", "eps_policy"])
def test_nk_irf_satisfies_the_structural_equations(shock):
    """The strongest available check: put the IRF path back into the model.

    After impact the economy is deterministic, so realised future values
    are the expected ones and every equation must hold exactly along the
    path — including the Euler and Phillips equations that carry the
    expectations.
    """
    model = build_nk()
    horizon = 8
    irf = model.irf(shock, horizon=horizon)
    rn, x, pi, i = (irf[c].to_numpy() for c in ["rn", "x", "pi", "i"])
    demand = 1.0 if shock == "eps_demand" else 0.0
    policy = 1.0 if shock == "eps_policy" else 0.0

    for h in range(horizon):
        now_d = demand if h == 0 else 0.0
        now_p = policy if h == 0 else 0.0
        assert abs(rn[h + 1] - NK["rho_r"] * rn[h] - now_d) < 1e-10
        assert abs(x[h + 1] - x[h] - (i[h] - pi[h + 1] - rn[h]) / NK["sigma"]) < 1e-10
        assert abs(NK["beta"] * pi[h + 1] + NK["kappa"] * x[h] - pi[h]) < 1e-10
        assert abs(NK["phi_pi"] * pi[h] + NK["phi_x"] * x[h] + now_p - i[h]) < 1e-10


def test_nk_states_are_zero_on_impact_but_controls_jump():
    """The timing convention, stated as a test.

    A shock routed through a state leaves that state at zero in the h=0
    row, while forward-looking controls move immediately because the
    innovation is already in their information set.
    """
    impact = build_nk().irf("eps_demand", horizon=1).loc[0]
    assert impact["rn"] == pytest.approx(0.0, abs=1e-12)
    assert abs(impact["x"]) > 1e-3
    assert abs(impact["pi"]) > 1e-3


def test_contemporaneous_shock_is_gone_next_period():
    """An i.i.d. shock with no state behind it must not persist."""
    irf = build_nk().irf("eps_policy", horizon=3)
    assert abs(irf.loc[0, "i"]) > 1e-3
    np.testing.assert_allclose(irf.loc[1].to_numpy(), 0.0, atol=1e-12)


def test_violating_the_taylor_principle_fails_blanchard_kahn():
    with pytest.raises(BlanchardKahnError):
        build_nk(phi_pi=0.9)
