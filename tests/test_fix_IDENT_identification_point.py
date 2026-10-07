"""identification() evaluates every Jacobian column at the requested parameter point.

Regression tests for the IDENT defect (reviews/2026-09-30-notebook-review):
before the fix the base state space, the shock covariance, the analytic
SE_/CORR_/ME_ columns and the one-sided fallback baseline were all taken at the
model's calibration, and each structural column at (user theta_j, calibrated
everything else) -- so a user Mapping, EstimatedParams starts, the default
path of a .mod file with estimated_params, the estimate() pre-flight and
prior_mc all reported ranks of a calibration/user hybrid.

Oracles:
- AR(1) ``x_t = rho x_{t-1} + a e_t``: ``Gamma_0 = a^2 s^2/(1-rho^2)``,
  ``Gamma_1 = rho Gamma_0``, differentiated by hand.
- A 3-equation NK model with correlated shocks and measurement error: a
  brute-force central-difference Jacobian that **re-builds the model at each
  perturbed point** with build_dynare and computes moments and spectra with
  independent code in this file.
- ``k1 * k2`` entering only as a product: rank 1 at every point.
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.linalg

import puremacro.dsge as dsge
from puremacro.dsge import identification
from puremacro.dsge._estimated_params import EstimatedParams, EstimatedParamSpec
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge.priors import (
    BetaPrior,
    GammaPrior,
    InvGammaPrior,
    NormalPrior,
    UniformPrior,
    WeibullPrior,
)

idm = sys.modules["puremacro.dsge.identification"]


# ---------------------------------------------------------------------------
# AR(1) with closed-form moments
# ---------------------------------------------------------------------------

def _ar1(rho=0.9, a=1.0, shock_cov=None):
    return build_dynare(
        lambda lead, curr, lag, sh, p: [curr.x - p.rho * lag.x - p.a * sh.e],
        variables=["x"], shocks=["e"], params={"rho": rho, "a": a},
        guess={"x": 0.0}, states=["x"], shock_cov=shock_cov,
    )


def _ar1_j2(rho, a, sig=1.0):
    """Closed-form d[ys; Gamma_0; Gamma_1]/d(rho, a, sigma) for x = rho x(-1) + a sigma e."""
    v = 1.0 - rho ** 2
    s2 = sig ** 2
    d_rho = [0.0, 2 * rho * a * a * s2 / v ** 2, a * a * s2 * (1 + rho ** 2) / v ** 2]
    d_a = [0.0, 2 * a * s2 / v, 2 * rho * a * s2 / v]
    d_sig = [0.0, 2 * a * a * sig / v, 2 * rho * a * a * sig / v]
    return np.array(d_rho), np.array(d_a), np.array(d_sig)


def _jacobians(model, params, obs, lags=1, n_freq=16, **kw):
    resolved = idm._resolve_params(model, params, list(obs))
    omega = idm._build_frequency_grid(n_freq=n_freq)
    return idm._identification_jacobians(model, resolved, list(obs), lags=lags, omega=omega, **kw)


def test_ar1_moment_jacobian_at_user_mapping_matches_closed_form():
    """Calibrated at (0.9, 1), asked at (0.5, 2): J2 is the closed form at (0.5, 2).

    Before the fix: rho column [1.7778, 2.2222] (a at 1) and a column
    [21.0526, 18.9474] (rho at 0.9); the truth is [7.1111, 8.8889] and
    [5.3333, 2.6667].
    """
    m = _ar1(0.9, 1.0)
    _, J2, _, _ = _jacobians(m, {"rho": 0.5, "a": 2.0}, ["x"])
    d_rho, d_a, _ = _ar1_j2(0.5, 2.0)
    np.testing.assert_allclose(J2[:, 0], d_rho, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(J2[:, 1], d_a, rtol=1e-6, atol=1e-9)

    res = identification(m, params={"rho": 0.5, "a": 2.0}, varobs=["x"])
    J2_true = np.column_stack([d_rho, d_a])
    np.testing.assert_allclose(
        res.j2_singular_values, np.linalg.svd(J2_true, compute_uv=False), rtol=1e-6
    )
    np.testing.assert_allclose(
        res.strength["sensitivity"].to_numpy(), np.linalg.norm(J2_true, axis=0), rtol=1e-6
    )

    # p_dict is the same request.
    res_p = identification(m, p_dict={"rho": 0.5, "a": 2.0}, varobs=["x"])
    np.testing.assert_allclose(res_p.j2_singular_values, res.j2_singular_values, rtol=1e-12)


def test_rank_at_user_point_not_polluted_by_placeholder_calibration():
    """A placeholder a=1e-4 in the calibration no longer makes rho look unidentified."""
    m = _ar1(0.9, 1e-4)
    res = identification(m, params={"rho": 0.5, "a": 1.0}, varobs=["x"])
    assert res.j2_rank == 2 and res.is_identified is True
    assert not any("rank deficient" in w for w in res.warnings)


def test_shock_std_column_uses_requested_point():
    """{rho: 0.5, SE_e: 0.5}: T at rho=0.5 and Q = 0.25 in both columns."""
    m = _ar1(0.9, 1.0)
    _, J2, _, _ = _jacobians(m, {"rho": 0.5, "SE_e": 0.5}, ["x"])
    d_rho, _, d_sig = _ar1_j2(0.5, 1.0, sig=0.5)
    np.testing.assert_allclose(J2[:, 0], d_rho, rtol=1e-6, atol=1e-9)  # [0, 0.4444, 0.5556]
    np.testing.assert_allclose(J2[:, 1], d_sig, rtol=1e-12, atol=1e-12)  # [0, 1.3333, 0.6667]


def test_se_name_defaults_to_declared_shock_std():
    """A bare 'SE_e' is evaluated at the declared sigma (2), not at 1."""
    m = _ar1(0.5, 1.0, shock_cov=np.array([[4.0]]))
    _, J2, _, _ = _jacobians(m, ["rho", "SE_e"], ["x"])
    d_rho, _, d_sig = _ar1_j2(0.5, 1.0, sig=2.0)
    np.testing.assert_allclose(J2[:, 0], d_rho, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(J2[:, 1], d_sig, rtol=1e-12)


@pytest.mark.parametrize("bound", ["lb", "ub"])
def test_one_sided_fallback_at_a_bound_is_taken_at_the_requested_point(bound):
    """init = 0.5 on a bound forces a one-sided difference; it must be around 0.5.

    Before the fix the forward difference subtracted the calibrated (rho=0.9)
    baseline and returned [-78594.7, -81401.3] for the rho column.
    """
    lb, ub = (0.5, 0.99) if bound == "lb" else (0.0, 0.5)
    spec = EstimatedParams((
        EstimatedParamSpec(kind="param", target=("rho",), name="rho", prior=None,
                           init=0.5, lb=lb, ub=ub),
        EstimatedParamSpec(kind="param", target=("a",), name="a", prior=None,
                           init=1.0, lb=0.0, ub=10.0),
    ))
    m = _ar1(0.9, 1.0)
    _, J2, _, _ = _jacobians(m, spec, ["x"])
    d_rho, d_a, _ = _ar1_j2(0.5, 1.0)
    np.testing.assert_allclose(J2[:, 0], d_rho, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(J2[:, 1], d_a, rtol=1e-6, atol=1e-9)


def test_one_sided_fallback_steps_away_from_upper_bound(monkeypatch):
    """At theta = ub the solver is never asked for a point above ub."""
    m = _ar1(0.9, 1.0)
    spec = [EstimatedParamSpec(kind="param", target=("rho",), name="rho", prior=None,
                               init=0.5, lb=0.0, ub=0.5)]
    asked = []
    real = idm._solve_at

    def spy(model, params, guess, *args, **kwargs):
        asked.append(params["rho"])
        return real(model, params, guess, *args, **kwargs)

    monkeypatch.setattr(idm, "_solve_at", spy)
    _jacobians(m, spec, ["x"])
    assert asked and max(asked) <= 0.5


# ---------------------------------------------------------------------------
# Brute force on a small NK model with correlated shocks and measurement error
# ---------------------------------------------------------------------------

NK_VARS = ["y", "pi", "r", "u", "v"]
NK_SHOCKS = ["eps_u", "eps_v", "eps_r"]
NK_OBS = ["y", "pi", "r"]
NK_CAL = dict(beta=0.99, sigma=1.0, kappa=0.3, phi_pi=1.5, phi_y=0.5, rho_u=0.6, rho_v=0.5)


def _nk_eqs(lead, curr, lag, sh, p):
    return [
        curr.y - lead.y + (1.0 / p.sigma) * (curr.r - lead.pi) - curr.v,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y - curr.u,
        curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - sh.eps_r,
        curr.u - p.rho_u * lag.u - sh.eps_u,
        curr.v - p.rho_v * lag.v - sh.eps_v,
    ]


def _nk(params=None, shock_cov=None):
    return build_dynare(
        _nk_eqs, variables=NK_VARS, shocks=NK_SHOCKS, params=dict(params or NK_CAL),
        guess={v: 0.0 for v in NK_VARS}, states=["u", "v"], shock_cov=shock_cov,
    )


def _q_of(theta, base_cov):
    """Independent Sigma_u(theta): SE_ on the diagonal, CORR_ as rho*sigma_i*sigma_j."""
    Q = np.array(base_cov, dtype=float, copy=True)
    for i, s in enumerate(NK_SHOCKS):
        if f"SE_{s}" in theta:
            Q[i, i] = theta[f"SE_{s}"] ** 2
    for i, s1 in enumerate(NK_SHOCKS):
        for j, s2 in enumerate(NK_SHOCKS):
            key = f"CORR_{s1}_{s2}"
            if key in theta:
                Q[i, j] = Q[j, i] = theta[key] * math.sqrt(Q[i, i] * Q[j, j])
    return Q


def _psi_bruteforce(theta, lags, omega, base_cov):
    """psi1..psiS at theta, re-building the model there; moments and spectra in-file."""
    struct = {k: theta.get(k, v) for k, v in NK_CAL.items()}
    m = _nk(struct)
    Q = _q_of(theta, base_cov)
    me = {o: theta[f"ME_{o}"] for o in NK_OBS if f"ME_{o}" in theta}
    ssm = make_state_space_from_varobs(m, NK_OBS, shock_cov=Q, measurement_error=me or None)
    T, R, Z, H, d = ssm.T, ssm.R, ssm.Z, ssm.H, ssm.d
    psi1 = np.concatenate([T.ravel(), R.ravel(), Q.ravel(), Z.ravel(), H.ravel()])
    Sig = scipy.linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
    mom = [d, (Z @ Sig @ Z.T + H).ravel()]
    for k in range(1, lags + 1):
        mom.append((Z @ np.linalg.matrix_power(T, k) @ Sig @ Z.T).ravel())
    psi2 = np.concatenate(mom)
    n = len(NK_OBS)
    tri, tri_s = np.tril_indices(n), np.tril_indices(n, -1)
    s_parts = []
    for w in omega:
        G = Z @ np.linalg.inv(np.exp(1j * w) * np.eye(T.shape[0]) - T) @ R
        S = (G @ Q @ G.conj().T + H) / (2 * np.pi)
        s_parts += [S.real[tri], S.imag[tri_s]]
    psiS = np.concatenate(s_parts)
    # The transfer function depends on the Cholesky convention; use the module's.
    psiH, _, _, _ = idm._compute_frequency_representations(T, R, Q, Z, H, omega)
    return psi1, psi2, psiH, psiS


def _bruteforce_jacobians(theta0, names, lags, omega, base_cov):
    cols = [[], [], [], []]
    for name in names:
        step = 1e-6 * max(1.0, abs(theta0[name]))
        up, dn = dict(theta0), dict(theta0)
        up[name] += step
        dn[name] -= step
        pu = _psi_bruteforce(up, lags, omega, base_cov)
        pd_ = _psi_bruteforce(dn, lags, omega, base_cov)
        for c, a, b in zip(cols, pu, pd_):
            c.append((a - b) / (2 * step))
    return [np.column_stack(c) for c in cols]


USER_POINT = {
    "kappa": 0.12, "phi_pi": 2.1, "rho_u": 0.85, "rho_v": 0.3, "sigma": 1.7,
    "SE_eps_u": 0.4, "SE_eps_v": 1.3, "SE_eps_r": 0.25,
    "CORR_eps_u_eps_v": 0.35, "ME_pi": 0.2,
}


def test_all_four_jacobians_equal_bruteforce_resolve_at_non_calibrated_point():
    m = _nk()
    base_cov = np.eye(3)
    lags = 2
    omega = idm._build_frequency_grid(n_freq=8)
    resolved = idm._resolve_params(m, USER_POINT, NK_OBS)
    got = idm._identification_jacobians(m, resolved, NK_OBS, lags=lags, omega=omega)
    want = _bruteforce_jacobians(dict(USER_POINT), list(USER_POINT), lags, omega, base_cov)
    for label, G, W in zip(("J1", "J2", "JH", "JS"), got, want):
        scale = np.maximum(np.abs(W).max(axis=0), 1e-12)
        err = np.abs(G - W).max(axis=0) / scale
        assert np.all(err < 1e-5), (label, dict(zip(USER_POINT, err)))


def test_result_equals_model_built_at_the_requested_point():
    """identification(m_cal, point) == identification(m_built_at_point, point)."""
    struct = {k: USER_POINT.get(k, v) for k, v in NK_CAL.items()}
    m_cal = _nk()
    m_pt = _nk(struct)
    r_cal = identification(m_cal, params=USER_POINT, varobs=NK_OBS, lags=2, n_freq=8)
    r_pt = identification(m_pt, params=USER_POINT, varobs=NK_OBS, lags=2, n_freq=8)
    for attr in ("j1_singular_values", "j2_singular_values",
                 "jh_singular_values", "js_singular_values"):
        np.testing.assert_allclose(getattr(r_cal, attr), getattr(r_pt, attr), rtol=1e-6,
                                   atol=1e-10, err_msg=attr)
    assert (r_cal.j1_rank, r_cal.j2_rank, r_cal.jh_rank, r_cal.js_rank) == (
        r_pt.j1_rank, r_pt.j2_rank, r_pt.jh_rank, r_pt.js_rank)


def test_declared_correlated_shock_cov_is_the_default_point():
    """Names only on a model that declares a correlated Sigma_u: evaluated there."""
    sd = np.array([0.4, 1.3, 0.25])
    corr = np.array([[1.0, 0.35, 0.0], [0.35, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = corr * np.outer(sd, sd)
    m = _nk(shock_cov=cov)
    names = ["kappa", "rho_u", "SE_eps_u", "SE_eps_v", "CORR_eps_u_eps_v"]
    omega = idm._build_frequency_grid(n_freq=6)
    resolved = idm._resolve_params(m, names, NK_OBS)
    vals = {p.name: p.base_val for p in resolved}
    assert vals["SE_eps_u"] == pytest.approx(0.4) and vals["SE_eps_v"] == pytest.approx(1.3)
    assert vals["CORR_eps_u_eps_v"] == pytest.approx(0.35)
    got = idm._identification_jacobians(m, resolved, NK_OBS, lags=1, omega=omega)
    theta0 = {"kappa": NK_CAL["kappa"], "rho_u": NK_CAL["rho_u"], "SE_eps_u": 0.4,
              "SE_eps_v": 1.3, "CORR_eps_u_eps_v": 0.35}
    want = _bruteforce_jacobians(theta0, names, 1, omega, cov)
    for label, G, W in zip(("J1", "J2", "JH", "JS"), got, want):
        scale = np.maximum(np.abs(W).max(axis=0), 1e-12)
        assert np.all(np.abs(G - W).max(axis=0) / scale < 1e-5), label


# ---------------------------------------------------------------------------
# build() models keep their linearisation when re-solved
# ---------------------------------------------------------------------------

def test_build_level_model_with_positive_steady_state():
    """A linearize='level' model with ss = mu > 0 is re-solved in levels.

    y' = (1 - rho) mu + rho y + e': ys = mu, Gamma_0 = 1/(1 - rho^2),
    Gamma_1 = rho Gamma_0. Before the fix the perturbed solves used the
    default linearize='log', so d(ys)/d(mu) came out 1/mu instead of 1.
    """
    def eqs(xp, x, e, p):
        return [xp.y - (1 - p.rho) * p.mu - p.rho * x.y - e.eps]

    m = dsge.build(eqs, variables=["y"], states=["y"], shocks=["eps"],
                   params={"rho": 0.6, "mu": 2.0}, guess={"y": 2.0}, linearize="level")
    assert m.units["y"] == "level"
    _, J2, _, _ = _jacobians(m, {"rho": 0.5, "mu": 3.0}, ["y"])
    d_rho, _, _ = _ar1_j2(0.5, 1.0)
    np.testing.assert_allclose(J2[:, 0], d_rho, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(J2[:, 1], [1.0, 0.0, 0.0], atol=1e-8)


# ---------------------------------------------------------------------------
# k1 * k2: unidentified at every point
# ---------------------------------------------------------------------------

PRODUCT_MOD = """
var y pi r u;
varexo eps_u eps_r;
parameters beta sigma k1 k2 phi_pi rho_u;
beta = 0.99; sigma = 1; k1 = 0.5; k2 = 0.2; phi_pi = 1.5; rho_u = 0.7;
model;
y = y(+1) - (1/sigma)*(r - pi(+1));
pi = beta*pi(+1) + k1*k2*y + u;
r = phi_pi*pi + eps_r;
u = rho_u*u(-1) + eps_u;
end;
shocks; var eps_u; stderr 1; var eps_r; stderr 1; end;
estimated_params;
k1, beta_pdf, 0.8, 0.05;
k2, beta_pdf, 0.3, 0.05;
end;
varobs y pi;
"""


def test_product_pair_is_rank_one_at_a_user_point():
    m = build_dynare(PRODUCT_MOD)
    res = identification(m, params={"k1": 1.0, "k2": 0.1}, varobs=["y", "pi", "r"])
    assert (res.j1_rank, res.j2_rank, res.jh_rank, res.js_rank) == (1, 1, 1, 1)
    assert res.is_identified is False


def test_default_mod_path_evaluates_at_estimated_params_start():
    """No params on a .mod with estimated_params: evaluated at the prior means."""
    m = build_dynare(PRODUCT_MOD)
    res = identification(m)
    assert res.param_names == ("k1", "k2")
    assert res.is_identified is False and res.j2_rank == 1

    explicit = identification(m, params={"k1": 0.8, "k2": 0.3}, varobs=["y", "pi"])
    np.testing.assert_allclose(res.j2_singular_values, explicit.j2_singular_values,
                               rtol=1e-10, atol=1e-12)


def test_default_mod_path_uses_initval_when_declared():
    """With an INITVAL the spec's start is the INITVAL (differs from Dynare's prior_mean)."""
    mod = PRODUCT_MOD.replace("k1, beta_pdf, 0.8, 0.05;", "k1, 0.6, 0, 1, beta_pdf, 0.8, 0.05;")
    m = build_dynare(mod)
    assert m._estimated_params.by_name("k1").start == pytest.approx(0.6)
    res = identification(m)
    at_init = identification(m, params={"k1": 0.6, "k2": 0.3}, varobs=["y", "pi"])
    np.testing.assert_allclose(res.j1_singular_values, at_init.j1_singular_values,
                               rtol=1e-10, atol=1e-12)


def test_estimate_preflight_raises_when_prior_means_differ_from_calibration():
    """Before the fix the pre-flight ran at a hybrid point and let this through."""
    m = build_dynare(PRODUCT_MOD)
    data = pd.DataFrame({"y": np.zeros(40), "pi": np.zeros(40)})
    with pytest.raises(ValueError, match="pre-flight check failed"):
        m.estimate(data, check_identification="raise")


def test_prior_mc_on_product_priors_is_never_identified():
    m = build_dynare(PRODUCT_MOD)
    res = identification(m, prior_mc=10, seed=0)
    mc = res.prior_mc_results
    assert mc["n_draws"] + mc["n_failed"] == 10 and mc["n_draws"] > 0
    for key in ("j1", "j2", "jh", "js"):
        assert mc[f"{key}_identified_rate"] == 0.0


# ---------------------------------------------------------------------------
# fixed_params and the estimate() pre-flight
# ---------------------------------------------------------------------------

def _two_shock(a=0.5):
    # y = rho y(-1) + a x + eps_y ; x = c x(-1) + eps_x ; only y observed.
    return build_dynare(
        lambda lead, curr, lag, sh, p: [
            curr.y - p.rho * lag.y - p.a * curr.x - sh.eps_y,
            curr.x - p.c * lag.x - sh.eps_x,
        ],
        variables=["y", "x"], shocks=["eps_y", "eps_x"],
        params={"rho": 0.5, "a": a, "c": 0.7}, guess={"y": 0.0, "x": 0.0}, states=["y", "x"],
    )


def test_fixed_params_move_the_evaluation_point():
    m = _two_shock(a=0.5)
    at_a0 = identification(_two_shock(a=0.0), params=["rho", "c"], varobs=["y"], lags=3)
    fixed = identification(m, params=["rho", "c"], varobs=["y"], lags=3,
                           fixed_params={"a": 0.0})
    np.testing.assert_allclose(fixed.j2_singular_values, at_a0.j2_singular_values,
                               rtol=1e-6, atol=1e-12)
    assert fixed.j2_rank == 1  # with a = 0, x never reaches y: c is not identified
    assert identification(m, params=["rho", "c"], varobs=["y"], lags=3).j2_rank == 2

    with pytest.raises(ValueError, match="fixed_params"):
        identification(m, params=["rho"], varobs=["y"], fixed_params={"nope": 1.0})


def test_estimate_preflight_uses_fixed_params():
    m = _two_shock(a=0.5)
    data = pd.DataFrame({"y": np.zeros(30)})
    priors = {"rho": BetaPrior(0.5, 0.1), "c": BetaPrior(0.7, 0.1)}
    with pytest.raises(ValueError, match="pre-flight check failed"):
        m.estimate(data, priors=priors, varobs=["y"], fixed_params={"a": 0.0},
                   check_identification="raise")


# ---------------------------------------------------------------------------
# prior_mc draws are prior draws and each is analysed at its own point
# ---------------------------------------------------------------------------

def test_prior_mc_analyses_each_draw_at_its_own_point():
    """Calibration a=1e-4, priors around (0.5, 1): every draw is identified.

    Before the fix the draws were analysed around the calibration and the J2
    identified rate was 0.0.
    """
    m = _ar1(0.9, 1e-4)
    priors = EstimatedParams((
        EstimatedParamSpec(kind="param", target=("rho",), name="rho",
                           prior=BetaPrior(0.5, 0.05), init=None, lb=0.0, ub=0.99),
        EstimatedParamSpec(kind="param", target=("a",), name="a",
                           prior=NormalPrior(1.0, 0.05), init=None, lb=0.0, ub=10.0),
    ))
    res = identification(m, params=priors, varobs=["x"], prior_mc=10, seed=1)
    assert res.prior_mc_results["n_draws"] == 10
    assert res.prior_mc_results["j2_identified_rate"] == 1.0


def test_prior_mc_counts_unsolvable_draws():
    m = _ar1(0.5, 1.0)
    spec = [EstimatedParamSpec(kind="param", target=("rho",), name="rho",
                               prior=NormalPrior(1.0, 0.1), init=0.9, lb=-5.0, ub=5.0)]
    res = identification(m, params=spec, varobs=["x"], prior_mc=20, seed=3)
    mc = res.prior_mc_results
    assert mc["n_failed"] > 0 and mc["n_draws"] + mc["n_failed"] == 20


@pytest.mark.parametrize("prior", [
    BetaPrior(0.7, 0.1),
    BetaPrior(1.2, 0.3, shift=0.5, scale=2.0),
    GammaPrior(1.5, 0.4),
    GammaPrior(2.0, 0.3, shift=1.0),
    InvGammaPrior(0.1, 0.05),
    InvGammaPrior(0.5, 0.2, kind="type2"),
    NormalPrior(0.3, 0.2),
    UniformPrior(-1.0, 2.0),
    WeibullPrior(1.0, 0.4),
], ids=lambda p: f"{p.dist}-{p.mean:g}")
def test_prior_draws_follow_the_prior_density(prior):
    """Empirical CDF of 20000 draws vs the CDF integrated from prior.pdf."""
    p = idm._ResolvedParam(name="t", kind="param", target=("t",), base_val=prior.mean,
                           prior=prior, lb=prior.lb, ub=prior.ub)
    rng = np.random.default_rng(12345)
    draws = np.array([idm._draw_parameter(p, rng) for _ in range(20000)])
    assert np.all((draws >= prior.lb) & (draws <= prior.ub))
    lo = prior.lb if math.isfinite(prior.lb) else prior.mean - 12 * prior.std
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        x = float(np.quantile(draws, q))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F, _ = scipy.integrate.quad(lambda t: float(prior.pdf(t)), lo, x, limit=200)
        assert abs(F - q) < 0.015, (q, x, F)


def test_prior_draws_respect_declared_bounds():
    prior = NormalPrior(0.0, 1.0)
    p = idm._ResolvedParam(name="t", kind="param", target=("t",), base_val=0.0,
                           prior=prior, lb=0.2, ub=0.4)
    rng = np.random.default_rng(0)
    draws = [idm._draw_parameter(p, rng) for _ in range(500)]
    assert min(draws) >= 0.2 and max(draws) <= 0.4


# ---------------------------------------------------------------------------
# Points the model cannot be solved at, and the calibration fast path
# ---------------------------------------------------------------------------

def test_unsolvable_point_raises_a_clear_error():
    m = _ar1(0.5, 1.0)
    with pytest.raises(ValueError, match="cannot be solved at the requested evaluation point"):
        identification(m, params={"rho": 1.2}, varobs=["x"])


def test_calibration_point_does_not_resolve(monkeypatch):
    """Values equal to the calibration reuse the solved model (no extra build)."""
    m = _ar1(0.5, 1.0)
    calls = []
    real = idm._solve_at
    monkeypatch.setattr(idm, "_solve_at",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    identification(m, params={"rho": 0.5, "SE_e": 1.0}, varobs=["x"])
    assert len(calls) == 2  # the two central-difference solves for rho only
