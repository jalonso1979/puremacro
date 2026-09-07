"""``make_state_space_from_varobs`` — the varobs -> Kalman connector.

The observable loads the *contemporaneous* innovation (`v_t = ys + C x_t +
D u_t`) while ``StateSpaceModel`` assumes measurement noise independent of the
state shock. The connector resolves that by carrying the innovation in the
state, ``alpha_t = [x_t; u_t]``. These tests pin that algebra three ways: the
recursion it produces, the likelihood it implies, and the covariance it implies.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from puremacro.dsge import load_mod
from puremacro.dsge.decomposition import _extract_companion_matrices
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import kalman_filter


_RBC = """
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
"""


@pytest.fixture(scope="module")
def rbc():
    return load_mod(_RBC)


def test_augmented_state_space_reproduces_the_model_recursion(rbc):
    """The (T, R, Z, d) construction must trace the same path as the model's
    own companion recursion — the loop LinearModel.simulate runs."""
    A, B, C, D, ys, variables, shocks, states, _ = _extract_companion_matrices(rbc)
    obs = ["c", "k"]
    rows = [variables.index(v) for v in obs]
    ssm = make_state_space_from_varobs(rbc, obs)

    rng = np.random.default_rng(0)
    n_t = 25
    u = rng.standard_normal((n_t, len(shocks))) * 0.01

    ref = np.zeros((n_t, len(obs)))
    x = np.zeros(len(states))
    for t in range(n_t):
        ref[t] = (ys + C @ x + D @ u[t])[rows]
        x = A @ x + B @ u[t]

    got = np.zeros((n_t, len(obs)))
    alpha = np.zeros(ssm.T.shape[0])
    for t in range(n_t):
        alpha = ssm.T @ alpha + ssm.R @ u[t]
        got[t] = ssm.d + ssm.Z @ alpha

    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_shapes_and_blocks(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c"])
    n_s, n_e = rbc.n_states, len(rbc.shocks)
    assert ssm.T.shape == (n_s + n_e, n_s + n_e)
    assert ssm.R.shape == (n_s + n_e, n_e)
    assert ssm.Q.shape == (n_e, n_e)
    assert ssm.Z.shape == (1, n_s + n_e)
    np.testing.assert_allclose(ssm.T[n_s:], 0.0)            # innovations are iid
    np.testing.assert_allclose(ssm.R[:n_s], 0.0)
    np.testing.assert_allclose(ssm.R[n_s:], np.eye(n_e))


def test_kalman_loglik_matches_a_stacked_multivariate_normal(rbc):
    """Independent check of T, R, Z, Q, H and d together: the filter's
    log-likelihood must equal the density of the stacked observations."""
    ssm = make_state_space_from_varobs(rbc, ["c"], ridge=1e-4)
    Tm, Rm, Qm, Zm, Hm, dv = ssm.T, ssm.R, ssm.Q, ssm.Z, ssm.H, ssm.d
    P = solve_discrete_lyapunov(Tm, Rm @ Qm @ Rm.T)

    n, n_t = Zm.shape[0], 8
    Sigma = np.zeros((n * n_t, n * n_t))
    for i in range(n_t):
        for j in range(n_t):
            h = i - j
            cross = (np.linalg.matrix_power(Tm, h) @ P) if h >= 0 \
                else P @ np.linalg.matrix_power(Tm.T, -h)
            block = Zm @ cross @ Zm.T + (Hm if h == 0 else 0.0)
            Sigma[i * n:(i + 1) * n, j * n:(j + 1) * n] = block
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = np.tile(dv, n_t)

    rng = np.random.default_rng(1)
    y = rng.multivariate_normal(mu, Sigma).reshape(n_t, n)
    ref = float(multivariate_normal(mean=mu, cov=Sigma).logpdf(y.ravel()))
    got = kalman_filter(y, ssm, a0=np.zeros(Tm.shape[0]), P0=P)["loglik"]
    assert got == pytest.approx(ref, rel=1e-8)


def test_observable_covariance_matches_theoretical_moments(rbc):
    """Z P Z' + H must equal the model's own analytical covariance block.

    This holds because the Lyapunov solution has Cov(x_t, u_t) = 0 — the [x, u]
    block of both T P T' and R Q R' is zero — so Z P Z' is exactly
    Var(C x_t + D u_t), the observables in the timing the model reports.
    """
    obs = ["c", "k"]
    ssm = make_state_space_from_varobs(rbc, obs)
    P = solve_discrete_lyapunov(ssm.T, ssm.R @ ssm.Q @ ssm.R.T)
    n_s = rbc.n_states
    np.testing.assert_allclose(P[:n_s, n_s:], 0.0, atol=1e-12)   # the premise
    got = ssm.Z @ P @ ssm.Z.T + ssm.H
    ref = rbc.theoretical_moments().covariance.loc[obs, obs].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_intercept_is_the_steady_state_for_level_units(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c", "k"])
    assert all(rbc.units[v] == "level" for v in ("c", "k"))
    np.testing.assert_allclose(
        ssm.d, rbc.steady_state[["c", "k"]].to_numpy(), rtol=1e-12)


def test_intercept_is_log_of_the_steady_state_for_log_units():
    """build() log-linearises by default, so its deviations are log
    deviations and the measurement is log(y_t) = log(ss) + Z alpha_t."""
    from puremacro.dsge import build

    def eqs(xp, x, e, p):
        # Productivity is written so its steady state is 1, not 0: the plain
        # `z - rho*z` of the build() docstring gives z_ss = 0, which makes
        # output identically zero and has no log deviation at all.
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma
            * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1 - p.rho) - p.rho * x.z - e.eps,
        ]

    m = build(eqs, variables=["c", "k", "z"], states=["k", "z"], shocks=["eps"],
              params=dict(alpha=.33, beta=.99, delta=.025, sigma=1.0, rho=.95),
              guess=dict(c=2.3, k=28.0, z=1.0))
    assert m.steady_state["z"] == pytest.approx(1.0)
    assert m.units["c"] == "log"
    ssm = make_state_space_from_varobs(m, ["c"])
    np.testing.assert_allclose(ssm.d, [np.log(m.steady_state["c"])], rtol=1e-12)


def test_prefilter_zeroes_the_intercept(rbc):
    assert np.any(make_state_space_from_varobs(rbc, ["c"]).d != 0.0)
    np.testing.assert_allclose(
        make_state_space_from_varobs(rbc, ["c"], prefilter=True).d, 0.0)


def test_measurement_error_lands_on_the_diagonal_of_H(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c", "k"], measurement_error={"k": 0.02})
    np.testing.assert_allclose(np.diag(ssm.H), [0.0, 0.02 ** 2])
    np.testing.assert_allclose(ssm.H - np.diag(np.diag(ssm.H)), 0.0)


def test_H_is_zero_by_default_not_a_ridge(rbc):
    """Dynare's default is H = 0; sw07_observation.py's 1e-8 ridge is a
    property of that hand-built file, not of the connector."""
    np.testing.assert_array_equal(make_state_space_from_varobs(rbc, ["c"]).H,
                                  np.zeros((1, 1)))


def test_ridge_is_opt_in(rbc):
    np.testing.assert_allclose(
        make_state_space_from_varobs(rbc, ["c"], ridge=1e-6).H, [[1e-6]])


def test_shock_cov_override_is_used(rbc):
    ssm = make_state_space_from_varobs(rbc, ["c"], shock_cov=np.array([[0.25]]))
    np.testing.assert_allclose(ssm.Q, [[0.25]])


def test_declared_shocks_block_supplies_q_by_default(rbc):
    """The .mod file says `stderr 0.01`, so Q is 1e-4 without being asked."""
    np.testing.assert_allclose(make_state_space_from_varobs(rbc, ["c"]).Q,
                               [[0.01 ** 2]])


def test_unknown_observable_raises_naming_it(rbc):
    with pytest.raises(ValueError, match="output"):
        make_state_space_from_varobs(rbc, ["output"])


def test_measurement_error_for_a_non_observable_raises(rbc):
    with pytest.raises(ValueError, match="k"):
        make_state_space_from_varobs(rbc, ["c"], measurement_error={"k": 0.01})


def test_empty_varobs_raises(rbc):
    with pytest.raises(ValueError, match="at least one"):
        make_state_space_from_varobs(rbc, [])


def test_second_order_solution_is_refused():
    """PrunedDSGESolution has .decision_rules(), so the companion extractor
    would happily take its FIRST-order block and drop every quadratic term.
    A linear measurement equation cannot represent that model, so it is
    refused rather than silently linearised."""
    sol = load_mod(_RBC, order=2)
    with pytest.raises(TypeError, match="second-order"):
        make_state_space_from_varobs(sol, ["c"])
