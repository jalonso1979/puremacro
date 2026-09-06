"""Closed-form and consistency regressions for the Dynare-compatible engine.

Every test here failed on the 2.3.0 engine, whose ``build_dynare`` dropped
the lead-of-state Jacobian block ``A_+[:, states]`` (so any variable at both
``t-1`` and ``t+1`` lost its lead), reported control rows of ``ghx`` / ``ghu``
one period off (``F@G`` and ``F@N+L`` instead of ``F`` and ``L``), computed
theoretical moments with a spurious state-shock cross term, used
``g_xx (h_u ⊗ h_u)`` instead of ``g_uu`` in the second-order risk correction,
ignored the ``shocks`` block outside ``compute_fevd``, and returned all-zero
decision rules when Blanchard-Kahn failed.

The anchor is Brock-Mirman (full depreciation, log utility), whose policy
``k_t = alpha*beta*z_t*k_{t-1}^alpha`` is exact under uncertainty, written
three ways (levels with ``z(+1)`` in the Euler equation, levels with
``E_t z(+1)`` substituted analytically, and logs).
"""
from __future__ import annotations

import io
import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.dsge
from puremacro import dsge
from puremacro.dsge import (
    LinearModel,
    ModelError,
    PrunedDSGESolution,
    build_dynare,
    compute_fevd,
    load_mod,
    parse_mod,
)
from puremacro.dsge.klein import BlanchardKahnError
from puremacro.dsge._moments import ZeroVarianceWarning

ALPHA, BETA, RHO = 0.33, 0.98, 0.9
K_SS = (ALPHA * BETA) ** (1.0 / (1.0 - ALPHA))
C_SS = (1.0 - ALPHA * BETA) * K_SS**ALPHA

BM_LEVELS = """
var c k z;
varexo eps;
parameters alpha beta rho;
alpha = 0.33; beta = 0.98; rho = 0.9;
model;
  1/c = beta * alpha * z(+1) * k^(alpha-1) / c(+1);
  c + k = z * k(-1)^alpha;
  z = (1 - rho) + rho * z(-1) + eps;
end;
initval; k = 0.2; c = 0.4; z = 1.0; end;
"""

BM_LEVELS_NOLEAD = """
var c k z;
varexo eps;
parameters alpha beta rho;
alpha = 0.33; beta = 0.98; rho = 0.9;
model;
  1/c = beta * alpha * ((1-rho) + rho*z) * k^(alpha-1) / c(+1);
  c + k = z * k(-1)^alpha;
  z = (1 - rho) + rho * z(-1) + eps;
end;
initval; k = 0.2; c = 0.4; z = 1.0; end;
"""

BM_LOGS = """
var lc lk z;
varexo eps;
parameters alpha beta rho;
alpha = 0.33; beta = 0.98; rho = 0.9;
model;
  exp(-lc) = beta * alpha * exp(z(+1)) * exp((alpha-1)*lk) * exp(-lc(+1));
  exp(lc) + exp(lk) = exp(z) * exp(alpha*lk(-1));
  z = rho * z(-1) + eps;
end;
initval; lk = -1.7; lc = -0.9; z = 0; end;
"""

RBC_DOCS = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;
alpha = 0.30; beta = 0.99; delta = 0.025; gamma = 1.0; rho = 0.80;
model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;
initval; k = 38.0; a = 0.0; c = 2.0; end;
shocks; var eps; stderr 0.01; end;
stoch_simul(order=1, irf=20);
"""

RBC_NOLEADSTATE = RBC_DOCS.replace("exp(a(+1))", "exp(rho*a)")

NK_TEMPLATE = """
var y pi r g;
varexo eps_r eps_g;
parameters beta sigma kappa phi_pi phi_y rho_g;
beta = 0.99; sigma = 1.0; kappa = 0.1; phi_pi = {phi_pi}; phi_y = 0.125; rho_g = 0.8;
model;
  y = y(+1) - (r - pi(+1))/sigma + g;
  pi = beta*pi(+1) + kappa*y;
  r = phi_pi*pi + phi_y*y + eps_r;
  g = rho_g*g(-1) + eps_g;
end;
initval; y = 0; pi = 0; r = 0; g = 0; end;
"""

RBC2 = """
var c k a g y;
varexo eps_a eps_g;
parameters alpha beta delta rho_a rho_g gbar;
alpha = 0.30; beta = 0.99; delta = 0.025; rho_a = 0.9; rho_g = 0.7; gbar = 0.2;
model;
  1/c = beta/c(+1) * (alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
  y = exp(a)*k(-1)^alpha;
  c + k - (1-delta)*k(-1) + g = y;
  a = rho_a*a(-1) + eps_a;
  g = (1-rho_g)*gbar + rho_g*g(-1) + eps_g;
end;
initval; k = 20; c = 1.8; a = 0; g = 0.2; y = 2.5; end;
shocks;
  var eps_a; stderr 0.01;
  var eps_g; stderr 0.05;
end;
stoch_simul(order=1, irf=20);
"""

EXPECTED_GHX_LEVELS = pd.DataFrame(
    {"k": [ALPHA * C_SS / K_SS, ALPHA, 0.0], "z": [RHO * C_SS, RHO * K_SS, RHO]},
    index=["c", "k", "z"],
)
EXPECTED_GHU_LEVELS = pd.Series([C_SS, K_SS, 1.0], index=["c", "k", "z"])


# ---------------------------------------------------------------------------
# 1. First-order closed forms (finding C18 / C19)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_text", [BM_LEVELS, BM_LEVELS_NOLEAD], ids=["z(+1) in Euler", "E_t z(+1) substituted"])
def test_brock_mirman_levels_first_order_matches_closed_form(mod_text):
    """ghx/ghu equal the closed form whether or not the state z appears with a lead.

    2.3.0 gave ghx[k, z(-1)] = 0.0235 (expected rho*k_ss = 0.1669) for the
    version with z(+1) and control rows F@G / F@N+L for both versions.
    """
    m = load_mod(mod_text)
    assert m.timing == "dynare"
    dr = m.decision_rules()
    pd.testing.assert_frame_equal(dr.ghx[["k", "z"]], EXPECTED_GHX_LEVELS, atol=1e-10, check_names=False)
    pd.testing.assert_series_equal(dr.ghu["eps"], EXPECTED_GHU_LEVELS, atol=1e-10, check_names=False)
    assert m.units == {"c": "level", "k": "level", "z": "level"}


def test_brock_mirman_logs_first_order_matches_closed_form():
    """In logs the exact policy is lk = log(alpha*beta) + z + alpha*lk(-1)."""
    m = load_mod(BM_LOGS)
    dr = m.decision_rules()
    exp_ghx = pd.DataFrame({"lk": [ALPHA, ALPHA, 0.0], "z": [RHO, RHO, RHO]}, index=["lc", "lk", "z"])
    pd.testing.assert_frame_equal(dr.ghx[["lk", "z"]], exp_ghx, atol=1e-10, check_names=False)
    np.testing.assert_allclose(dr.ghu["eps"].to_numpy(), [1.0, 1.0, 1.0], atol=1e-10)


def test_docs_rbc_with_lead_of_state_equals_analytic_substitution():
    """rbc.mod written with exp(a(+1)) and with exp(rho*a) give identical first-order rules."""
    ma = load_mod(RBC_DOCS)
    mb = load_mod(RBC_NOLEADSTATE)
    pd.testing.assert_frame_equal(ma.decision_rules().ghx, mb.decision_rules().ghx, atol=1e-10)
    pd.testing.assert_frame_equal(ma.decision_rules().ghu, mb.decision_rules().ghu, atol=1e-10)


def test_solution_satisfies_the_lead_lag_equilibrium_conditions():
    """A_+ F G + A_0 F + A_- = 0 and A_+ F N + A_0 L + B_u = 0 for the docs RBC."""
    m = load_mod(RBC_DOCS)
    dr = m.decision_rules()
    F_full, L_full = dr.ghx.to_numpy(), dr.ghu.to_numpy()
    s_idx = [list(m.variables).index(s) for s in m.states]
    G, N = m.solution.G, m.solution.N
    assert np.abs(m._A_plus @ F_full @ G + m._A_0 @ F_full + m._A_minus[:, s_idx]).max() < 1e-10
    assert np.abs(m._A_plus @ F_full @ N + m._A_0 @ L_full + m._B_u).max() < 1e-10
    np.testing.assert_allclose(G, F_full[s_idx], atol=1e-12)
    np.testing.assert_allclose(N, L_full[s_idx], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. IRF timing, simulation and theoretical moments (finding C19)
# ---------------------------------------------------------------------------

def test_irf_rows_are_aligned_in_dynare_time():
    """States and controls share a row: k_0 = k_ss and c_0 = c_ss for a unit eps.

    2.3.0 showed c at h=0 but k only at h=1 (states one row behind controls).
    """
    m = load_mod(BM_LEVELS_NOLEAD)
    irf = m.irf("eps", horizon=3, size=1.0)
    assert irf.loc[0, "k"] == pytest.approx(K_SS, abs=1e-10)
    assert irf.loc[0, "c"] == pytest.approx(C_SS, abs=1e-10)
    assert irf.loc[0, "z"] == pytest.approx(1.0, abs=1e-12)
    # k_1 = alpha*k_0 + rho*k_ss*z_0 ... in closed form k_1 = alpha*k_0 + rho*k_ss
    assert irf.loc[1, "k"] == pytest.approx(ALPHA * K_SS + RHO * K_SS, abs=1e-10)
    assert irf.loc[1, "c"] == pytest.approx(C_SS / K_SS * irf.loc[1, "k"], abs=1e-10)


def test_build_klein_timing_is_unchanged():
    """build() keeps its documented convention: states are zero at h=0 and move at h=1."""
    def eqs(xp, x, e, p):
        return [1 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1)) / xp.c,
                x.c + xp.k - x.z * x.k ** p.alpha,
                xp.z - x.z ** p.rho * np.exp(e.eps)]
    m = dsge.build(eqs, variables=["c", "k", "z"], states=["k", "z"], shocks=["eps"],
                   params=dict(alpha=ALPHA, beta=BETA, rho=RHO), guess=dict(c=0.5, k=0.1, z=1.0))
    assert m.timing == "klein"
    irf = m.irf("eps", horizon=2)
    np.testing.assert_allclose(irf.loc[0].to_numpy(), [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(irf.loc[1].to_numpy(), [1.0, 0.0, 1.0], atol=1e-9)
    # decision rules in Dynare form are (G, N) / (F, L)
    dr = m.decision_rules()
    np.testing.assert_allclose(dr.ghx.loc["c"].to_numpy(), m.solution.F[0], atol=1e-12)
    np.testing.assert_allclose(dr.ghu.loc["c"].to_numpy(), m.solution.L[0], atol=1e-12)


def test_theoretical_moments_match_lyapunov_and_long_simulation():
    """Var(c) = F Sx F' + L S L' (no state-shock cross term) and agrees with simulate().

    2.3.0 reported Std(c) = 0.02253 against 0.02105 from its own simulate().
    """
    m = load_mod(RBC_DOCS)
    mom = m.theoretical_moments(sigma=0.01)
    G, F, N, L = m.solution.G, m.solution.F, m.solution.N, m.solution.L
    import scipy.linalg
    Sx = scipy.linalg.solve_discrete_lyapunov(G, N @ N.T * 1e-4)
    var_c = (F @ Sx @ F.T + L @ L.T * 1e-4)[0, 0]
    assert mom.moments.loc["c", "Std.Dev."] == pytest.approx(np.sqrt(var_c), rel=1e-10)
    # reported states are end-of-period: Var(k_t) = Var(x_{t+1}) = Sx
    assert mom.moments.loc["k", "Variance"] == pytest.approx(Sx[0, 0], rel=1e-10)

    sim = m.simulate(periods=200_000, sigma=0.01, seed=1, burn=500)
    for v in m.variables:
        assert sim[v].std() == pytest.approx(mom.moments.loc[v, "Std.Dev."], rel=0.02), v
    # cross-correlation in the same (Dynare) timing
    assert sim[["c", "k"]].corr().iloc[0, 1] == pytest.approx(mom.correlation.loc["c", "k"], abs=0.01)
    # lag-1 autocorrelation
    assert sim["c"].autocorr(1) == pytest.approx(mom.autocorr.loc["c", "Lag 1"], abs=0.01)


def test_theoretical_moments_closed_form_brock_mirman():
    """c_t = (c_ss/k_ss) k_t exactly, so corr(c, k) = 1 and Std(c)/Std(k) = c_ss/k_ss."""
    m = load_mod(BM_LEVELS)
    mom = m.theoretical_moments(sigma=0.01)
    assert mom.correlation.loc["c", "k"] == pytest.approx(1.0, abs=1e-10)
    ratio = mom.moments.loc["c", "Std.Dev."] / mom.moments.loc["k", "Std.Dev."]
    assert ratio == pytest.approx(C_SS / K_SS, rel=1e-10)
    # z is an AR(1): Var = sigma^2 / (1 - rho^2)
    assert mom.moments.loc["z", "Variance"] == pytest.approx(1e-4 / (1 - RHO**2), rel=1e-10)


def test_build_theoretical_moments_have_no_state_shock_cross_term():
    """For build() models too, Var(y) = F Sx F' + L S L' matches a long simulation."""
    NK = dict(beta=0.99, sigma=1.0, kappa=0.1275, phi_pi=1.5, phi_x=0.125, rho_r=0.9)

    def nk(xp, x, e, p):
        return [xp.rn - p.rho_r * x.rn - e.eps_demand,
                xp.x - x.x - (x.i - xp.pi - x.rn) / p.sigma,
                p.beta * xp.pi + p.kappa * x.x - x.pi,
                p.phi_pi * x.pi + p.phi_x * x.x + e.eps_policy - x.i]

    m = dsge.build(nk, variables=["rn", "x", "pi", "i"], states=["rn"], shocks=["eps_demand", "eps_policy"],
                   params=NK, steady_state={k: 0.0 for k in ("rn", "x", "pi", "i")}, linearize="level")
    mom = m.theoretical_moments()
    sim = m.simulate(periods=200_000, seed=3, burn=200)
    for v in m.variables:
        assert sim[v].std() == pytest.approx(mom.moments.loc[v, "Std.Dev."], rel=0.02), v


# ---------------------------------------------------------------------------
# 3. Second order (finding C20)
# ---------------------------------------------------------------------------

def test_second_order_brock_mirman_levels_closed_form():
    """ghxx, ghxu, ghuu, ghs2 of the exactly-solvable model in levels."""
    sol = load_mod(BM_LEVELS, order=2)
    assert isinstance(sol, PrunedDSGESolution)
    d = sol.oo_dr
    assert d.ghxx.loc["k", "k_k"] == pytest.approx(ALPHA * (ALPHA - 1) / K_SS, abs=1e-7)
    assert d.ghxx.loc["k", "k_z"] == pytest.approx(RHO * ALPHA, abs=1e-7)
    assert d.ghxx.loc["k", "z_k"] == pytest.approx(RHO * ALPHA, abs=1e-7)
    assert d.ghxx.loc["k", "z_z"] == pytest.approx(0.0, abs=1e-7)
    assert d.ghxu.loc["k", "k_eps"] == pytest.approx(ALPHA, abs=1e-7)
    assert d.ghxu.loc["k", "z_eps"] == pytest.approx(0.0, abs=1e-7)
    assert d.ghuu.loc["k", "eps_eps"] == pytest.approx(0.0, abs=1e-7)
    assert d.ghs2["k"] == pytest.approx(0.0, abs=1e-7)
    assert d.ghs2["c"] == pytest.approx(0.0, abs=1e-7)
    assert d.ghxx.loc["c", "k_k"] == pytest.approx(ALPHA * (ALPHA - 1) * C_SS / K_SS**2, abs=1e-7)
    assert d.ghxu.loc["c", "k_eps"] == pytest.approx(ALPHA * C_SS / K_SS, abs=1e-7)
    # first-order block unchanged by the second-order solve
    pd.testing.assert_frame_equal(d.ghx[["k", "z"]], EXPECTED_GHX_LEVELS, atol=1e-10, check_names=False)


def test_second_order_brock_mirman_logs_has_no_curvature():
    """The policy is exactly log-linear: every second-order term vanishes."""
    sol = load_mod(BM_LOGS, order=2)
    d = sol.oo_dr
    assert np.abs(d.ghxx.to_numpy()).max() < 1e-8
    assert np.abs(d.ghxu.to_numpy()).max() < 1e-8
    assert np.abs(d.ghuu.to_numpy()).max() < 1e-8
    assert np.abs(d.ghs2.to_numpy()).max() < 1e-8


def test_second_order_risk_correction_closed_form():
    """p_t = beta E_t exp(z_{t+1}) with z AR(1): ghs2[p] = beta * Var(u) exactly."""
    beta, rho, s = 0.96, 0.7, 0.3

    def eqs(lead, curr, lag, sh, p):
        return [curr.p - p.beta * np.exp(lead.z), curr.z - p.rho * lag.z - sh.u]

    sol = build_dynare(eqs, variables=["p", "z"], shocks=["u"], params=dict(beta=beta, rho=rho),
                       steady_state=dict(p=beta, z=0.0), order=2, shock_cov=np.array([[s**2]]))
    d = sol.oo_dr
    assert d.ghx.loc["p", "z"] == pytest.approx(beta * rho**2, rel=1e-9)
    assert d.ghu.loc["p", "u"] == pytest.approx(beta * rho, rel=1e-9)
    assert d.ghxx.loc["p", "z_z"] == pytest.approx(beta * rho**4, rel=1e-6)
    assert d.ghxu.loc["p", "z_u"] == pytest.approx(beta * rho**3, rel=1e-6)
    assert d.ghuu.loc["p", "u_u"] == pytest.approx(beta * rho**2, rel=1e-6)
    assert d.ghs2["p"] == pytest.approx(beta * s**2, rel=1e-6)
    assert d.ghs2["z"] == pytest.approx(0.0, abs=1e-10)


def _rbc_levels(lead, curr, lag, sh, p):
    return [curr.c**-1 - p.beta * lead.c**-1 * (p.alpha * np.exp(lead.a) * curr.k**(p.alpha - 1) + 1 - p.delta),
            curr.k - (np.exp(curr.a) * lag.k**p.alpha - curr.c + (1 - p.delta) * lag.k),
            curr.a - (p.rho * lag.a + sh.eps)]


def test_second_order_rule_satisfies_euler_equation_in_expectation():
    """At the steady state the second-order rule leaves an O(sigma^4) expected residual.

    With the wrong risk correction the expected Euler residual is O(sigma^2).
    """
    alpha, beta, delta, rho = 0.3, 0.99, 0.025, 0.8
    r_ss = 1 / beta - (1 - delta)
    k_ss = (alpha / r_ss) ** (1 / (1 - alpha))
    c_ss = k_ss**alpha - delta * k_ss
    params = dict(alpha=alpha, beta=beta, delta=delta, rho=rho)
    from puremacro.dsge.build import _Vec
    nodes, weights = np.polynomial.hermite_e.hermegauss(15)
    weights = weights / weights.sum()
    par = _Vec(list(params), list(params.values()), what="parameter")
    V = ["k", "a", "c"]

    def expected_residual(s: float) -> float:
        sol = build_dynare(_rbc_levels, variables=V, shocks=["eps"], params=params,
                           steady_state=dict(k=k_ss, a=0.0, c=c_ss), order=2, shock_cov=np.array([[s**2]]))
        ss = sol.steady_state.loc[V].to_numpy()
        y_t = ss + 0.5 * sol.ghs2
        x_next = (y_t - ss)[[0, 1]]  # states k, a
        kron_xx = np.kron(x_next, x_next)
        res = np.zeros(3)
        for node, w in zip(nodes, weights):
            u = np.array([s * node])
            y_next = (ss + 0.5 * sol.ghs2 + sol.ghx @ x_next + sol.ghu @ u
                      + 0.5 * sol.ghxx @ kron_xx + sol.ghxu @ np.kron(x_next, u) + 0.5 * sol.ghuu @ np.kron(u, u))
            res += w * np.asarray(_rbc_levels(_Vec(V, y_next), _Vec(V, y_t), _Vec(V, ss), _Vec(["eps"], np.zeros(1)), par))
        return float(np.abs(res).max())

    r1, r2 = expected_residual(0.05), expected_residual(0.025)
    assert r1 < 1e-5
    assert 8.0 < r1 / r2 < 32.0  # fourth-order scaling, not second-order (ratio 4)


def test_second_order_levels_and_logs_agree_on_the_risk_correction():
    alpha, beta, delta, rho = 0.3, 0.99, 0.025, 0.8
    r_ss = 1 / beta - (1 - delta)
    k_ss = (alpha / r_ss) ** (1 / (1 - alpha))
    c_ss = k_ss**alpha - delta * k_ss
    params = dict(alpha=alpha, beta=beta, delta=delta, rho=rho)

    def rbc_logs(lead, curr, lag, sh, p):
        return [np.exp(-curr.c) - p.beta * np.exp(-lead.c) * (p.alpha * np.exp(lead.a) * np.exp((p.alpha - 1) * curr.k) + 1 - p.delta),
                np.exp(curr.k) - (np.exp(curr.a) * np.exp(p.alpha * lag.k) - np.exp(curr.c) + (1 - p.delta) * np.exp(lag.k)),
                curr.a - (p.rho * lag.a + sh.eps)]

    lev = build_dynare(_rbc_levels, variables=["k", "a", "c"], shocks=["eps"], params=params,
                       steady_state=dict(k=k_ss, a=0.0, c=c_ss), order=2)
    log = build_dynare(rbc_logs, variables=["k", "a", "c"], shocks=["eps"], params=params,
                       steady_state=dict(k=np.log(k_ss), a=0.0, c=np.log(c_ss)), order=2)
    assert lev.ghs2[0] == pytest.approx(log.ghs2[0] * k_ss, rel=1e-6)
    assert lev.ghs2[2] == pytest.approx(log.ghs2[2] * c_ss, rel=1e-6)
    assert lev.ghs2[0] + lev.ghs2[2] == pytest.approx(0.0, abs=1e-8)


def test_method_central_is_honoured_at_order_2():
    """method='central' is used for the Hessians too and lands close to complex step."""
    a = load_mod(RBC_DOCS, order=2)
    b = load_mod(RBC_DOCS, order=2, method="central")
    assert b.first_order.method == "central"
    np.testing.assert_allclose(b.ghxx, a.ghxx, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(b.ghs2, a.ghs2, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Pruned second-order objects are consistent with the first-order model
# ---------------------------------------------------------------------------

def test_pruned_first_order_component_reproduces_linear_model():
    """simulate() and girf() of the pruned solution are in Dynare timing like the LinearModel."""
    m = load_mod(RBC_DOCS)
    sol = m.solve(order=2)
    rng = np.random.default_rng(5)
    eps = rng.standard_normal((60, 1)) * 0.01
    sim2 = sol.simulate(periods=60, shocks=eps, burn=0)
    first = pd.concat([sim2.states_1st, sim2.controls_1st], axis=1)[list(m.variables)]
    # first-order component == LinearModel recursion with the same innovations
    M_x, M_u = m._reported_loadings()
    x = np.zeros(m.n_states)
    rows = []
    for t in range(60):
        rows.append(M_x @ x + M_u @ eps[t])
        x = m.solution.G @ x + m.solution.N @ eps[t]
    lin = pd.DataFrame(rows, columns=list(m.states) + list(m.controls))[list(m.variables)]
    np.testing.assert_allclose(first.to_numpy(), lin.to_numpy(), atol=1e-12)
    # and the linear model's own simulate() with those innovations agrees
    # impact of a GIRF at h=0 coincides with the first-order IRF up to O(size^2)
    # the GIRF carries genuine O(size^2) curvature terms (the levels policy is
    # not linear), so agreement is to first order in ``size`` only
    g = sol.girf("eps", size=1e-4, horizon=5, sigma=0.0)
    irf1 = m.irf("eps", horizon=5, size=1e-4)
    np.testing.assert_allclose(g[list(m.variables)].to_numpy(), irf1.to_numpy(), rtol=1e-3, atol=1e-12)


def test_pruned_theoretical_moments_agree_with_first_order_and_have_real_fevd():
    """Order-2 covariance / autocorrelation equal order-1 and the FEVD is per shock (finding M46)."""
    m = load_mod(RBC2)
    sol = m.solve(order=2)
    mom1 = m.theoretical_moments(lags=4)
    mom2 = sol.theoretical_moments(lags=4)
    pd.testing.assert_frame_equal(mom1.covariance, mom2.covariance, atol=1e-12)
    pd.testing.assert_frame_equal(mom1.autocorr, mom2.autocorr, atol=1e-12)
    assert list(mom2.fevd.columns) == ["eps_a", "eps_g"]
    pd.testing.assert_frame_equal(mom1.fevd, mom2.fevd, atol=1e-10)
    np.testing.assert_allclose(mom2.fevd.sum(axis=1).to_numpy(), 100.0, atol=1e-8)
    assert mom2.fevd.loc[("g", 1), "eps_g"] == pytest.approx(100.0)
    assert 0.0 < mom2.fevd.loc[("c", 1), "eps_g"] < 100.0
    # simulated moments (long) agree with the theoretical ones
    res = sol.stoch_simul(irf=0, periods=100_000, seed=2)
    for v in m.variables:
        assert res.simulated_moments.loc[v, "Std.Dev."] == pytest.approx(mom2.moments.loc[v, "Std.Dev."], rel=0.05), v


def test_brock_mirman_order2_irfs_equal_order1_irfs():
    """In logs the Brock-Mirman policy is exactly linear, so the pruned
    second-order stoch_simul IRFs coincide with the first-order ones (in
    levels the policy k' = a*b*z*k^a is curved and the orders legitimately differ)."""
    m = load_mod(BM_LOGS)
    r1 = m.stoch_simul(irf=6, sigma=0.01)
    r2 = m.stoch_simul(order=2, irf=6, sigma=0.01)
    for key in r1.irfs:
        np.testing.assert_allclose(r2.irfs[key].to_numpy(), r1.irfs[key].to_numpy(), atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Blanchard-Kahn handling (findings M44 / C42)
# ---------------------------------------------------------------------------

def test_indeterminate_model_raises_blanchard_kahn_error():
    with pytest.raises(BlanchardKahnError, match="indeterminacy"):
        load_mod(NK_TEMPLATE.format(phi_pi=0.9))
    m = load_mod(NK_TEMPLATE.format(phi_pi=1.5))
    assert m.is_determinate


def test_non_strict_model_refuses_to_pretend():
    """strict=False returns the flagged model, whose reporting methods raise ModelError."""
    m = load_mod(NK_TEMPLATE.format(phi_pi=0.9), strict=False)
    assert isinstance(m, LinearModel)
    assert m.solution.eu != (1, 1)
    assert "Blanchard-Kahn" in m.summary()
    for call in (m.decision_rules, m.irf, m.simulate, m.theoretical_moments, m.stoch_simul):
        with pytest.raises(ModelError, match="unique stable solution"):
            call("eps_g") if call.__name__ == "irf" else call()


def test_cli_reports_true_bk_status(tmp_path, capsys):
    from puremacro.dsge.cli import run_cli

    bad = tmp_path / "nk_indet.mod"
    bad.write_text(NK_TEMPLATE.format(phi_pi=0.9), encoding="utf-8")
    code = run_cli([str(bad), "--irf", "5", "--outdir", str(tmp_path / "out_bad")])
    captured = capsys.readouterr()
    assert code == 3
    assert "Blanchard-Kahn" in captured.err
    assert "verified" not in captured.out
    assert not (tmp_path / "out_bad" / "irfs.csv").exists()

    good = tmp_path / "rbc2.mod"
    good.write_text(RBC2, encoding="utf-8")
    code = run_cli([str(good), "--irf", "5", "--fevd", "--outdir", str(tmp_path / "out_good")])
    captured = capsys.readouterr()
    assert code == 0
    assert "Blanchard-Kahn condition verified" in captured.out
    fevd = pd.read_csv(tmp_path / "out_good" / "fevd.csv")
    assert fevd["eps_a"].between(0, 1).all()
    # order-2 header is populated for the pruned solution
    code = run_cli([str(good), "--order", "2", "--irf", "3", "--outdir", str(tmp_path / "out2")])
    captured = capsys.readouterr()
    assert code == 0
    assert "Endogenous  : 5" in captured.out
    assert "Exogenous   : 2" in captured.out
    assert "Parameters  : 6" in captured.out


def test_compute_fevd_never_pads_zero_variance_rows():
    """A variable with zero forecast-error variance gets NaN shares and a warning, not 1/n_u."""
    mod = RBC2.replace("var c k a g y;", "var c k a g y klag;").replace(
        "  g = (1-rho_g)*gbar + rho_g*g(-1) + eps_g;", "  g = (1-rho_g)*gbar + rho_g*g(-1) + eps_g;\n  klag = k(-1);"
    ).replace("initval; k = 20;", "initval; klag = 20; k = 20;")
    m = load_mod(mod)
    with pytest.warns(ZeroVarianceWarning, match="klag@h=1"):
        res = compute_fevd(m, horizons=[1, 2, None])
    assert res.table.loc[("klag", 1)].isna().all()
    assert res.table.loc[("klag", 2)].sum() == pytest.approx(1.0)
    defined = res.table.dropna()
    np.testing.assert_allclose(defined.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
    assert not np.allclose(defined.loc[("c", 1)].to_numpy(), 0.5)


# ---------------------------------------------------------------------------
# 6. Smets-Wouters reference model (findings C42 / C43)
# ---------------------------------------------------------------------------

def _sw07_path() -> Path:
    return Path(puremacro.dsge.__file__).parent / "_references" / "sw07_pfeifer.mod"


def test_sw07_pfeifer_is_determinate_with_genuine_irfs():
    """Pfeifer's SW07 has a unique stable solution; a monetary tightening lowers inflation."""
    p = _sw07_path()
    assert p.is_file()
    m = load_mod(p, order=1)
    assert m.solution.eu == (1, 1)
    sd = dict(zip(m.shocks, m._shock_sd(None)))
    assert sd["em"] == pytest.approx(0.2397)
    irf = m.irf("em", horizon=8, size=sd["em"])
    assert irf.loc[0, "robs"] > 0.05
    assert irf.loc[0, "pinfobs"] < 0.0
    assert irf.loc[0, "dy"] < 0.0
    res = m.stoch_simul(irf=24)
    assert np.abs(res.irfs["robs_em"].to_numpy()).max() > 0.05
    fevd = compute_fevd(m, horizons=[1, 4, None]).table
    assert not np.allclose(fevd.to_numpy(), 1.0 / 7.0)
    np.testing.assert_allclose(fevd.sum(axis=1).to_numpy(), 1.0, atol=1e-10)


def test_sw07_mod_resolves_from_the_package_and_importlib_resources():
    from importlib import resources

    ref = resources.files("puremacro.dsge") / "_references" / "sw07_pfeifer.mod"
    assert ref.is_file()
    assert _sw07_path().read_text(encoding="utf-8") == ref.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. The shocks block flows everywhere (findings M43 / M47 / M66)
# ---------------------------------------------------------------------------

def test_shocks_block_is_the_default_covariance_everywhere():
    m = load_mod(RBC_DOCS)
    np.testing.assert_allclose(m._shock_cov, [[1e-4]])
    mom = m.theoretical_moments()
    assert mom.moments.loc["a", "Std.Dev."] == pytest.approx(0.01 / np.sqrt(1 - 0.8**2), rel=1e-10)
    res = m.stoch_simul(irf=5, periods=2000, seed=0)
    assert res.irfs["a_eps"].iloc[0] == pytest.approx(0.01)
    assert res.irfs["a_eps"].iloc[1] == pytest.approx(0.008)
    assert res.simulated_moments.loc["a", "Std.Dev."] == pytest.approx(0.01 / 0.6, rel=0.1)
    assert res.theoretical_moments.moments.loc["a", "Std.Dev."] == pytest.approx(0.01 / 0.6, rel=1e-10)
    sim = m.simulate(periods=20000, seed=1)
    assert sim["a"].std() == pytest.approx(0.01 / 0.6, rel=0.05)
    # explicit sigma still overrides
    assert m.theoretical_moments(sigma=1.0).moments.loc["a", "Std.Dev."] == pytest.approx(1 / 0.6, rel=1e-10)
    # and the second-order solve inherits the covariance
    sol = m.solve(order=2)
    np.testing.assert_allclose(sol.shock_cov, [[1e-4]])
    assert sol.stoch_simul(irf=2).irfs["a_eps"].iloc[0] == pytest.approx(0.01)


def test_order2_stoch_simul_honours_mapping_sigma():
    m = load_mod(RBC_DOCS)
    r_map = m.stoch_simul(order=2, irf=5, sigma={"eps": 0.02})
    r_flt = m.stoch_simul(order=2, irf=5, sigma=0.02)
    r_one = m.stoch_simul(order=2, irf=5, sigma=1.0)
    assert r_map.irfs["k_eps"].iloc[0] == pytest.approx(r_flt.irfs["k_eps"].iloc[0])
    assert r_map.irfs["k_eps"].iloc[0] != pytest.approx(r_one.irfs["k_eps"].iloc[0])
    assert r_map.irfs["a_eps"].iloc[0] == pytest.approx(0.02)
    with pytest.raises(TypeError, match="scalar perturbation scale"):
        m.solve(order=2).stoch_simul(sigma={"eps": 0.02})


def test_linear_model_solve_api():
    m = load_mod(RBC_DOCS)
    assert m.solve() is m
    assert m.solve(order=1) is m
    sol = m.solve(order=2)
    assert isinstance(sol, PrunedDSGESolution)
    assert sol.first_order is not None and sol.first_order.timing == "dynare"
    assert sol.variables == m.variables and sol.shocks == m.shocks
    with pytest.raises(ValueError, match="order"):
        m.solve(order=3)


# ---------------------------------------------------------------------------
# 8. Derivative verification, parser idioms, path handling (M45, M48, M94)
# ---------------------------------------------------------------------------

def test_verify_derivatives_catches_non_analytic_equations():
    def rbc_abs(lead, curr, lag, sh, p):
        return [curr.c**-1 - p.beta * lead.c**-1 * (p.alpha * np.exp(lead.a) * curr.k**(p.alpha - 1) + 1 - p.delta),
                curr.k - (np.exp(curr.a) * np.abs(lag.k)**p.alpha - curr.c + (1 - p.delta) * lag.k),
                curr.a - (p.rho * lag.a + sh.eps)]

    P = dict(alpha=0.3, beta=0.99, delta=0.025, rho=0.8)
    with pytest.raises(ModelError, match="not analytic"):
        build_dynare(rbc_abs, variables=["c", "k", "a"], shocks=["eps"], params=P, guess=dict(c=2, k=38, a=0))
    m = build_dynare(rbc_abs, variables=["c", "k", "a"], shocks=["eps"], params=P,
                     guess=dict(c=2, k=38, a=0), method="central")
    ref = load_mod(RBC_DOCS)
    np.testing.assert_allclose(m.decision_rules().ghx.to_numpy(), ref.decision_rules().ghx.to_numpy(), rtol=1e-5)


def test_parse_mod_keeps_stoch_simul_options_with_a_variable_list():
    parsed = parse_mod(RBC_DOCS.replace("stoch_simul(order=1, irf=20);", "stoch_simul(order=2, irf=15) c k;"))
    assert parsed["options"] == {"order": 2, "irf": 15}
    sol = load_mod(RBC_DOCS.replace("stoch_simul(order=1, irf=20);", "stoch_simul(order=2, irf=15) c k;"))
    assert isinstance(sol, PrunedDSGESolution)


def test_parse_mod_evaluates_steady_state_model_temporaries():
    text = RBC_DOCS.replace(
        "initval; k = 38.0; a = 0.0; c = 2.0; end;",
        "steady_state_model;\n  rk = 1/beta - 1 + delta;\n  k = (alpha/rk)^(1/(1-alpha));\n  a = 0;\n  c = k^alpha - delta*k;\nend;",
    )
    parsed = parse_mod(text)
    k_ss = (0.3 / (1 / 0.99 - 1 + 0.025)) ** (1 / 0.7)
    assert parsed["steady_state"]["k"] == pytest.approx(k_ss)
    assert "rk" not in parsed["steady_state"]
    m = load_mod(text)
    assert m.steady_state["k"] == pytest.approx(k_ss)
    with pytest.raises(ValueError, match="steady_state_model"):
        parse_mod(text.replace("rk = 1/beta - 1 + delta;", "rk = undefined_name + 1;"))


def test_load_mod_raises_file_not_found_for_missing_paths():
    with pytest.raises(FileNotFoundError, match=r"exist\.mod"):  # separator-agnostic: Windows renders the path with backslashes
        load_mod(Path("does/not/exist.mod"))
    with pytest.raises(FileNotFoundError):
        load_mod("puremacro/dsge/_references/no_such_model.mod")
