"""identification(): state set at the evaluation point, and edge cases (IDENT, round 2).

``build_dynare`` detects the predetermined variables numerically at the
parameter vector it is solved at. ``model.states`` therefore comes from the
calibration, and a lag whose coefficient is calibrated to 0 is not a state.
Every re-solve inside identification() used to reuse that set, so at a point
where the coefficient is not 0 the lag was silently dropped, and the
derivative with respect to the coefficient itself was lost even at the
calibration (SW07's crhoms, crhopinf, crhow, cmap, cmaw: sensitivity exactly
0.0, ranks 29/31/29/29 of 36 on the default call).

Oracle: a brute-force central difference that rebuilds the model with
``build_dynare`` at every perturbed point, **with states auto-detected there**,
and computes the moments with independent code in this file.

Edge cases from the adversarial review:
- names-only ``ME_<obs>`` takes its ``measurement_error`` entry;
- the one-sided fallback never steps across a declared bound;
- only solver failures trigger fallbacks / skipped prior draws;
- ``prior_mc`` always returns a dict, and warns when no draw can be solved;
- the ``estimate()`` pre-flight names the bad ``fixed_params`` argument.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import puremacro.dsge as D
from puremacro.dsge import identification
from puremacro.dsge._estimated_params import EstimatedParams, EstimatedParamSpec
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge.priors import BetaPrior, UniformPrior

idm = sys.modules["puremacro.dsge.identification"]

SW07 = Path(D.__file__).parent / "_references" / "sw07_pfeifer.mod"


# ---------------------------------------------------------------------------
# y = rho*y(-1) + a*x + ey ;  x = c*x(-1) + ex   (rho calibrated at 0)
# ---------------------------------------------------------------------------

def _eqs(lead, curr, lag, sh, p):
    return [curr.y - p.rho * lag.y - p.a * curr.x - sh.ey,
            curr.x - p.c * lag.x - sh.ex]


def _mk(rho=0.0, c=0.7, a=0.5, states=None):
    return build_dynare(_eqs, variables=["y", "x"], shocks=["ey", "ex"],
                        params={"rho": rho, "a": a, "c": c},
                        guess={"y": 0.0, "x": 0.0}, states=states)


def _moments(theta, lags=1):
    """[d; vec Gamma_0; vec Gamma_1] of y, model rebuilt at theta, states auto-detected."""
    m = _mk(rho=theta["rho"], c=theta["c"], a=theta.get("a", 0.5))
    s = make_state_space_from_varobs(m, ["y"], shock_cov=np.eye(2))
    sig = scipy.linalg.solve_discrete_lyapunov(s.T, s.R @ s.Q @ s.R.T)
    out = [s.d, (s.Z @ sig @ s.Z.T).ravel()]
    tk = s.T.copy()
    for _ in range(lags):
        out.append((s.Z @ tk @ sig @ s.Z.T).ravel())
        tk = tk @ s.T
    return np.concatenate(out)


def _bruteforce_j2(theta0, names, step=1e-6):
    cols = []
    for n in names:
        up, dn = dict(theta0), dict(theta0)
        up[n] += step
        dn[n] -= step
        cols.append((_moments(up) - _moments(dn)) / (2 * step))
    return np.column_stack(cols)


def _jacobians(model, params, obs, lags=1, n_freq=16, **kw):
    resolved = idm._resolve_params(model, params, list(obs), kw.get("measurement_error"))
    omega = idm._build_frequency_grid(n_freq=n_freq)
    return idm._identification_jacobians(model, resolved, list(obs), lags=lags, omega=omega, **kw)


def test_zero_calibrated_lag_is_a_state_at_the_requested_point():
    """Calibrated at rho=0 (states ('x',)), asked at rho=0.5, c=0.7.

    Before: rho column [0, 0, 0] and c column [0, 1.3456, 1.4321] (y(-1)
    dropped from every re-solve). Truth: [0, 5.7535, 6.1361] and
    [0, 5.2733, 5.2460].
    """
    m = _mk(rho=0.0)
    assert m.states == ("x",)
    point = {"rho": 0.5, "c": 0.7}
    _, J2, _, _ = _jacobians(m, point, ["y"])
    np.testing.assert_allclose(J2, _bruteforce_j2(point, ["rho", "c"]), rtol=1e-5, atol=1e-8)

    # Same answer as the model built at the point itself.
    _, J2_at, _, _ = _jacobians(_mk(rho=0.5), point, ["y"])
    np.testing.assert_allclose(J2, J2_at, rtol=1e-8, atol=1e-10)


def test_zero_calibrated_lag_names_only_at_the_calibration():
    """Names only, at the calibration rho=0: d/d rho is not zero.

    Before: the rho column was identically zero and J2 had rank 1 of 2.
    """
    m = _mk(rho=0.0)
    _, J2, _, _ = _jacobians(m, ["rho", "c"], ["y"])
    np.testing.assert_allclose(J2, _bruteforce_j2({"rho": 0.0, "c": 0.7}, ["rho", "c"]),
                               rtol=1e-5, atol=1e-8)
    assert np.linalg.norm(J2[:, 0]) > 0.1

    res = identification(m, params=["rho", "c"], varobs=["y"])
    assert (res.j1_rank, res.j2_rank, res.jh_rank, res.js_rank) == (2, 2, 2, 2)
    assert res.strength.loc["rho", "sensitivity"] > 0.1


def test_state_set_helper_adds_only_what_the_derivatives_need():
    m = _mk(rho=0.0)
    ss = m.steady_state.to_numpy(dtype=float)
    cal = dict(m._params)
    rho = idm._ResolvedParam(name="rho", kind="param", target=("rho",), base_val=0.0)
    c = idm._ResolvedParam(name="c", kind="param", target=("c",), base_val=0.7)
    # rho not analysed and 0 at theta_0: nothing to add, model.states unchanged.
    assert idm._state_set(m, ss, cal, [c]) == ("x",)
    # rho analysed: its step switches y(-1) on.
    assert idm._state_set(m, ss, cal, [rho, c]) == ("y", "x")
    # rho non-zero at theta_0 (e.g. through fixed_params): y is a state there.
    assert idm._state_set(m, ss, {**cal, "rho": 0.3}, [c]) == ("y", "x")


def test_fixed_params_switching_a_lag_on_use_the_right_states():
    m = _mk(rho=0.0)
    _, J2, _, _ = _jacobians(m, ["c"], ["y"], fixed_params={"rho": 0.5})
    np.testing.assert_allclose(J2, _bruteforce_j2({"rho": 0.5, "c": 0.7}, ["c"]),
                               rtol=1e-5, atol=1e-8)


def test_prior_mc_recomputes_the_state_set_per_draw():
    """rho calibrated at 0, drawn around 0.5: every draw has a live y(-1)."""
    m = _mk(rho=0.0)
    spec = EstimatedParams((
        EstimatedParamSpec(kind="param", target=("rho",), name="rho",
                           prior=BetaPrior(0.5, 0.05), init=None, lb=0.0, ub=0.99),
        EstimatedParamSpec(kind="param", target=("c",), name="c",
                           prior=BetaPrior(0.7, 0.05), init=None, lb=0.0, ub=0.99),
    ))
    res = identification(m, params=spec, varobs=["y"], prior_mc=8, seed=2)
    assert res.prior_mc_results["n_draws"] == 8
    assert res.prior_mc_results["j2_identified_rate"] == 1.0


def test_build_models_keep_their_declared_states(monkeypatch):
    """The state-set search is for build_dynare models only."""
    from puremacro.dsge.build import build

    def eqs(xp, x, e, p):
        return [xp.k - p.rho * x.k - e.u]

    m = build(eqs, variables=["k"], states=["k"], shocks=["u"],
              params={"rho": 0.5}, guess={"k": 0.0}, linearize="level")
    called = []
    monkeypatch.setattr(idm, "_state_set", lambda *a, **k: called.append(1) or ("k",))
    identification(m, params={"rho": 0.6}, varobs=["k"])
    assert called == []


# ---------------------------------------------------------------------------
# SW07: AR/MA coefficients calibrated at 0
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sw07():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return D.load_mod(SW07)


ZERO_CAL = ["crhoms", "crhopinf", "crhow", "cmap", "cmaw"]


def test_sw07_zero_calibrated_coefficients_have_nonzero_columns_at_calibration(sw07):
    """Names only at the calibration. Before: ranks 1/1/1/1 of 6 and five zero columns."""
    assert all(sw07._params[n] == 0.0 for n in ZERO_CAL)
    res = identification(sw07, params=ZERO_CAL + ["crhoa"], lags=1)
    assert (res.j1_rank, res.j2_rank, res.jh_rank, res.js_rank) == (6, 6, 6, 6)
    assert (res.strength.loc[ZERO_CAL, "sensitivity"] > 1.0).all()


def test_sw07_default_call_differentiates_the_zero_calibrated_coefficients(sw07):
    """Default call (spec starts). Before: sensitivity exactly 0.0 for all five."""
    res = identification(sw07, lags=1)
    assert (res.strength.loc[ZERO_CAL, "sensitivity"] > 1.0).all()


# ---------------------------------------------------------------------------
# Names-only ME_ takes its measurement_error entry
# ---------------------------------------------------------------------------

def _ar1(rho=0.5):
    return build_dynare(
        lambda lead, curr, lag, sh, p: [curr.x - p.rho * lag.x - sh.e],
        variables=["x"], shocks=["e"], params={"rho": rho}, guess={"x": 0.0}, states=["x"],
    )


def test_me_name_only_defaults_to_its_measurement_error_entry():
    """d Gamma_0 / d sigma_me = 2 sigma_me = 0.2, not the 2h placeholder at 0."""
    m = _ar1()
    res = identification(m, params=["ME_x"], varobs=["x"], measurement_error={"x": 0.1})
    assert res.strength.loc["ME_x", "sensitivity"] == pytest.approx(0.2, rel=1e-12)
    _, J2, _, _ = _jacobians(m, ["ME_x"], ["x"], measurement_error={"x": 0.1})
    np.testing.assert_allclose(J2[:, 0], [0.0, 0.2, 0.0], atol=1e-15)
    # An explicit value still wins.
    res_v = identification(m, params={"ME_x": 0.3}, varobs=["x"], measurement_error={"x": 0.1})
    assert res_v.strength.loc["ME_x", "sensitivity"] == pytest.approx(0.6, rel=1e-12)


# ---------------------------------------------------------------------------
# One-sided fallback never crosses a declared bound
# ---------------------------------------------------------------------------

def _f_failing_above(theta0, asked):
    def f(t):
        asked.append(t)
        if t > theta0:
            raise ValueError("no solution above theta0")
        return (np.array([t ** 2]),)
    return f


def test_fallback_at_lower_bound_does_not_step_below_it_when_the_upper_side_fails():
    asked = []
    with pytest.raises(ValueError, match="away from the bound 0.5"):
        idm._structural_derivative(_f_failing_above(0.5, asked), 0.5, 1e-3, 0.5, 1.0,
                                   (np.array([0.25]),), name="rho")
    assert asked and min(asked) >= 0.5


def test_fallback_at_upper_bound_does_not_step_above_it():
    asked = []

    def f(t):
        asked.append(t)
        if t < 0.5:
            raise ValueError("no solution below")
        return (np.array([t ** 2]),)

    with pytest.raises(ValueError, match="away from the bound 0.5"):
        idm._structural_derivative(f, 0.5, 1e-3, 0.0, 0.5, (np.array([0.25]),))
    assert asked and max(asked) <= 0.5


def test_fallback_inside_bounds_uses_the_other_side_second_order():
    asked = []
    (d,) = idm._structural_derivative(_f_failing_above(0.5, asked), 0.5, 1e-3, None, None,
                                      (np.array([0.25]),))
    assert d[0] == pytest.approx(1.0, abs=1e-12)  # second order is exact on t**2


def test_bounds_narrower_than_the_step_raise_clearly():
    with pytest.raises(ValueError, match="leave no room"):
        idm._structural_derivative(lambda t: (np.array([t]),), 0.5, 1e-3, 0.4999, 0.5001,
                                   (np.array([0.5]),), name="rho")


def test_programming_errors_are_not_swallowed_by_the_fallback():
    def f(t):
        raise TypeError("bug")

    with pytest.raises(TypeError, match="bug"):
        idm._structural_derivative(f, 0.5, 1e-3, None, None, (np.array([0.0]),))


# ---------------------------------------------------------------------------
# prior_mc bookkeeping
# ---------------------------------------------------------------------------

def test_prior_mc_returns_a_dict_and_warns_when_no_draw_can_be_solved():
    """Start 0.5 is solvable, every prior draw (rho in [1.1, 1.5]) is explosive."""
    m = _ar1(0.5)
    spec = [EstimatedParamSpec(kind="param", target=("rho",), name="rho",
                               prior=UniformPrior(1.1, 1.5), init=0.5, lb=None, ub=None)]
    with pytest.warns(RuntimeWarning, match="could not be solved at any of the 5 prior draws"):
        res = identification(m, params=spec, varobs=["x"], prior_mc=5, seed=0)
    mc = res.prior_mc_results
    assert mc["n_draws"] == 0 and mc["n_failed"] == 5
    assert math.isnan(mc["j2_identified_rate"]) and math.isnan(mc["mean_j2_rank"])
    assert res.j2_rank == 1  # the analysis at the start itself is unaffected


def test_prior_mc_does_not_count_programming_errors_as_failed_draws(monkeypatch):
    m = _ar1(0.5)
    spec = [EstimatedParamSpec(kind="param", target=("rho",), name="rho",
                               prior=BetaPrior(0.5, 0.05), init=None, lb=0.0, ub=0.99)]
    real = idm._identification_jacobians
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise TypeError("bug in a draw")
        return real(*a, **k)

    monkeypatch.setattr(idm, "_identification_jacobians", flaky)
    with pytest.raises(TypeError, match="bug in a draw"):
        identification(m, params=spec, varobs=["x"], prior_mc=3, seed=0)


# ---------------------------------------------------------------------------
# estimate() pre-flight: a bad fixed_params name is reported as such
# ---------------------------------------------------------------------------

MOD = """
var y pi r u;
varexo eps_u eps_r;
parameters beta sigma kappa phi_pi rho_u;
beta = 0.99; sigma = 1; kappa = 0.3; phi_pi = 1.5; rho_u = 0.7;
model;
y = y(+1) - (1/sigma)*(r - pi(+1));
pi = beta*pi(+1) + kappa*y + u;
r = phi_pi*pi + eps_r;
u = rho_u*u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1; var eps_r; stderr 1; end;
estimated_params;
kappa, gamma_pdf, 0.2, 0.05;
rho_u, beta_pdf, 0.7, 0.1;
end;
varobs y pi;
"""


def test_estimate_preflight_names_the_bad_fixed_params_argument():
    m = build_dynare(MOD)
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)) * 0.5,
                        columns=["y", "pi"])
    with pytest.raises(ValueError, match=r"estimate\(\): fixed_params names \['phi_pii'\]"):
        m.estimate(data, fixed_params={"phi_pii": 2.0}, check_identification=True,
                   n_draws=10, n_chains=1, burn_in=2, mode_compute="none")
