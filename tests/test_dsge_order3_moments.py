"""Exact unconditional moments of the pruned third-order solution.

``Order3PrunedSolution.theoretical_moments`` evaluates the closed forms of the
pruned state space of Andreasen, Fernandez-Villaverde and Rubio-Ramirez (2018).
Four independent checks:

1. The centred augmented state space reproduces puremacro's pruned simulator
   period by period, to machine precision.
2. For a claim to a cubic payoff, y = A x^3 + B x with x a Gaussian AR(1) is the
   exact solution, and Isserlis' theorem gives every moment of y in closed form.
3. Means and covariances equal Dynare 8's "theoretical moments based on pruned
   state space" on the five live Dynare models and the claim to a cube, frozen
   in ``tests/fixtures/dynare_order3_pruned_moments.json``.
4. Autocovariances agree with Monte Carlo: many independent puremacro chains,
   and Dynare's own 4,000,000-period pruned simulations. Dynare's theoretical
   order-3 autocorrelations drop the correlation between its uncentred
   innovation xf (x) u (x) u and past shocks, and miss both the closed form and
   Dynare's own simulations.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import load_mod
from puremacro.dsge import pruning
from puremacro.dsge._moments import first_order_moments
from puremacro.dsge._pruned_moments import (
    gaussian_moment_tensor,
    pruned_order3_moments,
    pruned_state_space_order3,
)
from puremacro.dsge.pruning import Order3PrunedSolution

# Variables with no first-order variance (y in correlated_cubic) have undefined
# first-order variance shares; the warning is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:forecast-error variance is zero")

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "tests" / "fixtures" / "dynare_live"
DYNARE = json.loads((ROOT / "tests" / "fixtures" / "dynare_order3_pruned_moments.json").read_text(encoding="utf-8"))
MODELS = sorted(DYNARE["cases"])


def _model_text(name: str) -> str:
    case = DYNARE["cases"][name]
    return case["model_text"] if "model_text" in case else (LIVE / f"{name}.mod").read_text(encoding="utf-8")


def _solve(name: str) -> Order3PrunedSolution:
    return load_mod(_model_text(name), order=3)


def _claim_to_cube(beta: float, rho: float, s: float) -> Order3PrunedSolution:
    return load_mod(f"var y x; varexo e; parameters beta rho; beta={beta!r}; rho={rho!r};\n"
                    "model; x = rho*x(-1) + e; y = beta*y(+1) + x^3; end;\n"
                    f"initval; x=0; y=0; end;\nshocks; var e; stderr {s!r}; end;\n", order=3)


def _claim_to_cube_moments(beta: float, rho: float, s: float, lags: int):
    """Variance and autocorrelations of y = A x^3 + B x, x ~ AR(1) with innovation s.d. s."""
    a = 1.0 / (1.0 - beta * rho**3)
    b = 3.0 * beta * a * rho * s**2 / (1.0 - beta * rho)
    v = s**2 / (1.0 - rho**2)
    # E[x^6] = 15 v^3, E[x^4] = 3 v^2; for corr(x_t, x_{t-k}) = r, E[x_t^3 x_{t-k}^3] = v^3 (9 r + 6 r^3).
    var = 15.0 * a**2 * v**3 + 6.0 * a * b * v**2 + b**2 * v
    autocorr = []
    for k in range(1, lags + 1):
        r = rho**k
        autocorr.append((a**2 * v**3 * (9.0 * r + 6.0 * r**3) + 6.0 * a * b * v**2 * r + b**2 * v * r) / var)
    return var, np.array(autocorr)


@pytest.fixture(scope="module")
def solutions():
    return {name: _solve(name) for name in MODELS}


def _rowkron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)


def _simulate_chains(sol, chains: int, periods: int, seed: int) -> list[np.ndarray]:
    """Independent pruned chains from the steady state, straight from the decision rule."""
    n, e = sol.n_states, sol.n_shocks
    g = sol._pruned_coefficients(1.0)
    h = {k: v[:n] for k, v in g.items()}
    chol = np.linalg.cholesky(sol._sigma_u(1.0, None))
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (np.zeros((chains, n)) for _ in range(3))
    path = []
    for _ in range(periods):
        u = rng.standard_normal((chains, e)) @ chol.T
        x11, uu = _rowkron(x1, x1), _rowkron(u, u)

        def rule(c):
            first = x1 @ c["gx"].T + u @ c["gu"].T
            second = (x2 @ c["gx"].T + 0.5 * x11 @ c["gxx"].T + _rowkron(x1, u) @ c["gxu"].T
                      + 0.5 * uu @ c["guu"].T + 0.5 * c["gss"])
            third = (x3 @ c["gx"].T + _rowkron(x1, x2) @ c["gxx"].T + _rowkron(x2, u) @ c["gxu"].T
                     + _rowkron(x11, x1) @ c["gxxx"].T / 6.0 + 0.5 * _rowkron(x11, u) @ c["gxxu"].T
                     + 0.5 * _rowkron(x1, uu) @ c["gxuu"].T + _rowkron(uu, u) @ c["guuu"].T / 6.0
                     + 0.5 * x1 @ c["gxss"].T + 0.5 * u @ c["guss"].T)
            return first, second, third

        path.append(sum(rule(g)))
        x1, x2, x3 = rule(h)
    return path


# ---------------------------------------------------------------------------
# 1. The augmented system reproduces the pruned simulator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,sigma", [("correlated_cubic", 1.0), ("rbc", 1.0), ("nonlinear_multishock", 1.7)])
def test_centred_state_space_reproduces_pruned_simulation(solutions, name, sigma):
    sol = solutions[name]
    n, e = sol.n_states, sol.n_shocks
    sigma_u = sol._sigma_u(sigma, None)
    ss = pruned_state_space_order3(sol._pruned_coefficients(sigma**2), n, sigma_u)
    a, b, c, cm, dm, d = (ss[k] for k in ("A", "B", "c", "C", "D", "d"))
    shocks = np.random.default_rng(7).multivariate_normal(np.zeros(e), sigma_u, size=150)
    x1, x2, x3, y1, y2, y3 = sol._pruned_path(shocks, sigma)
    levels = np.hstack([x1 + x2 + x3, y1 + y2 + y3])
    vs = sigma_u.reshape(-1)

    def z(t):
        if t < 0:
            return np.zeros(a.shape[0])
        return np.concatenate([x1[t], x2[t], np.kron(x1[t], x1[t]), x3[t], np.kron(x1[t], x2[t]),
                               np.kron(np.kron(x1[t], x1[t]), x1[t])])

    for t, u in enumerate(shocks):
        xf = x1[t - 1] if t else np.zeros(n)
        xs = x2[t - 1] if t else np.zeros(n)
        uu = np.kron(u, u)
        eps = np.concatenate([u, uu - vs, np.kron(xf, u), np.kron(xs, u), np.kron(np.kron(xf, xf), u),
                              np.kron(xf, uu - vs), np.kron(uu, u)])
        np.testing.assert_allclose(c + a @ z(t - 1) + b @ eps, z(t), rtol=0, atol=1e-13)
        np.testing.assert_allclose(d + cm @ z(t - 1) + dm @ eps, levels[t], rtol=0, atol=1e-13)


def test_second_order_blocks_and_isserlis_agree(solutions):
    sol = solutions["nonlinear_multishock"]
    n = sol.n_states
    ss = pruned_state_space_order3(sol._pruned_coefficients(1.0), n, sol.shock_cov)
    zb = ss["z_blocks"]
    three = slice(0, zb[2].stop)
    np.testing.assert_allclose(ss["Var_z"][three, three], ss["Var_z2"], rtol=1e-10, atol=1e-16)
    vec_s1 = ss["E_xfxf"].reshape(-1)
    fourth = gaussian_moment_tensor(ss["E_xfxf"], 4).reshape(n * n, n * n)
    np.testing.assert_allclose(ss["Var_z2"][zb[2], zb[2]] + np.outer(vec_s1, vec_s1), fourth,
                               rtol=1e-10, atol=1e-18)


# ---------------------------------------------------------------------------
# 2. Closed form: the claim to a cubic payoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta,rho,s", [(0.95, 0.8, 0.1), (0.75, 0.86, 0.08), (0.97, -0.32, 0.22),
                                        (0.51, 0.49, 0.27), (0.72, -0.65, 0.21)])
def test_claim_to_cube_moments_match_closed_form(beta, rho, s):
    th = _claim_to_cube(beta, rho, s).theoretical_moments(lags=5)
    var, autocorr = _claim_to_cube_moments(beta, rho, s, 5)
    assert th.moments.loc["y", "Mean"] == pytest.approx(0.0, abs=1e-15)
    assert th.covariance.loc["y", "y"] == pytest.approx(var, rel=1e-12)
    np.testing.assert_allclose(th.autocorr.loc["y"].to_numpy(), autocorr, rtol=1e-12, atol=1e-15)


def test_dynare_autocorrelation_misses_the_closed_form():
    ref = DYNARE["cases"]["claim_to_cube"]
    iy = ref["variables"].index("y")
    var, autocorr = _claim_to_cube_moments(0.95, 0.8, 0.1, 5)
    dynare = np.array([ref["autocorr"][k][iy][iy] for k in range(5)])
    assert ref["var"][iy][iy] == pytest.approx(var, rel=1e-10)
    assert autocorr[0] == pytest.approx(0.7634, abs=5e-5) and dynare[0] == pytest.approx(0.7085, abs=5e-5)
    assert np.all(autocorr - dynare > 0.05)


# ---------------------------------------------------------------------------
# 3. Dynare 8 parity for the mean and the covariance
# ---------------------------------------------------------------------------

def test_fixture_models_are_the_ones_dynare_solved():
    import hashlib

    for name, case in DYNARE["cases"].items():
        assert hashlib.sha256(_model_text(name).encode()).hexdigest() == case["model_sha256"], name

@pytest.mark.parametrize("name", MODELS)
def test_mean_and_covariance_match_dynare(solutions, name):
    ref = DYNARE["cases"][name]
    names = ref["variables"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th = solutions[name].theoretical_moments(lags=5)
    np.testing.assert_allclose(th.moments.loc[names, "Mean"].to_numpy(), ref["mean"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(th.covariance.loc[names, names].to_numpy(), np.array(ref["var"]),
                               rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Autocovariances: Monte Carlo, and where Dynare's differ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DYNARE["simulated"]["cases"]))
def test_dynare_own_simulations_side_with_the_exact_autocorrelations(solutions, name):
    sim = DYNARE["simulated"]["cases"][name]
    theory = DYNARE["cases"][name]
    names = sim["variables"]
    th = solutions[name].theoretical_moments(lags=5)
    exact = th.autocorr.loc[names].to_numpy()
    simulated = np.array([np.diag(np.array(m)) for m in sim["autocorr"]]).T
    dynare = np.array([np.diag(np.array(m)) for m in theory["autocorr"]]).T
    # 4,000,000 periods: the sampling error of an autocorrelation is about 1e-3 or less.
    assert np.max(np.abs(simulated - exact)) < 3e-3
    if name in ("correlated_cubic", "quadratic_feedback"):
        # Here Dynare's error dwarfs the sampling error; on nonlinear_state it is of its order.
        iy = names.index("y")
        assert abs(simulated[iy, 0] - dynare[iy, 0]) > 10.0 * abs(simulated[iy, 0] - exact[iy, 0])

def test_autocovariances_match_monte_carlo_not_dynare(solutions):
    sol = solutions["correlated_cubic"]
    th = sol.theoretical_moments(lags=2)
    ref = DYNARE["cases"]["correlated_cubic"]
    path = _simulate_chains(sol, chains=200_000, periods=80, seed=11)
    order = list(sol.state_names) + list(sol.control_names)
    iy = order.index("y")
    y_now, y_lag = path[-1][:, iy], path[-2][:, iy]
    prod = (y_now - y_now.mean()) * (y_lag - y_lag.mean())
    cov_mc, se = prod.mean(), prod.std() / np.sqrt(prod.size)

    variance = th.covariance.loc["y", "y"]
    exact = th.autocorr.loc["y", "Lag 1"] * variance
    dynare = ref["autocorr"][0][ref["variables"].index("y")][ref["variables"].index("y")] * variance
    assert abs(cov_mc - exact) < 4.0 * se
    assert abs(cov_mc - dynare) > 20.0 * se
    # The mean and the variance, by contrast, are the same in both.
    assert th.autocorr.loc["y", "Lag 1"] == pytest.approx(0.4750546, abs=1e-6)


@pytest.mark.parametrize("name", ["nonlinear_state", "rbc"])
def test_all_moments_agree_with_monte_carlo(solutions, name):
    sol = solutions[name]
    lags = 2
    order = list(sol.state_names) + list(sol.control_names)
    mean_dev, gamma_0, gammas = pruned_order3_moments(sol._pruned_coefficients(1.0), sol.n_states,
                                                      sol.shock_cov, lags)
    path = _simulate_chains(sol, chains=60_000, periods=400, seed=3)
    now = path[-1]
    dev = now - now.mean(axis=0)
    z_mean = (now.mean(axis=0) - mean_dev) / (now.std(axis=0) / np.sqrt(now.shape[0]))
    sq = dev**2
    z_var = (sq.mean(axis=0) - np.diag(gamma_0)) / (sq.std(axis=0) / np.sqrt(now.shape[0]))
    assert np.all(np.abs(z_mean) < 4.5) and np.all(np.abs(z_var) < 4.5), (z_mean, z_var)
    for k in range(1, lags + 1):
        prod = dev * (path[-1 - k] - path[-1 - k].mean(axis=0))
        z_cov = (prod.mean(axis=0) - np.diag(gammas[k - 1])) / (prod.std(axis=0) / np.sqrt(now.shape[0]))
        assert np.all(np.abs(z_cov) < 4.5), (k, z_cov)
    assert order  # names align with the stacked [states; controls] rows


# ---------------------------------------------------------------------------
# 4. Structure of the result and limiting cases
# ---------------------------------------------------------------------------

def test_linear_solution_reduces_to_first_order_moments():
    g = np.array([[0.9, 0.1], [0.0, 0.5]])
    n_mat = np.array([[1.0, 0.0], [0.3, 1.0]])
    f = np.array([[0.4, -0.2]])
    l_mat = np.array([[0.1, 0.7]])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    sol = Order3PrunedSolution(G=g, N=n_mat, F=f, L=l_mat, state_names=("a", "b"), control_names=("c",),
                               shock_names=("e1", "e2"), shock_cov=cov)
    th = sol.theoretical_moments(lags=3)
    m_x, m_u = np.vstack([g, f]), np.vstack([n_mat, l_mat])
    _, gamma_0, gammas = first_order_moments(g, n_mat, m_x, m_u, cov, 3)
    np.testing.assert_allclose(th.covariance.to_numpy(), gamma_0, rtol=1e-12, atol=1e-15)
    for k, gamma_k in enumerate(gammas, start=1):
        np.testing.assert_allclose(th.autocorr[f"Lag {k}"].to_numpy(), np.diag(gamma_k) / np.diag(gamma_0),
                                   rtol=1e-12)
        np.testing.assert_allclose(th.autocorr_matrix(k).to_numpy(),
                                   gamma_k / np.sqrt(np.outer(np.diag(gamma_0), np.diag(gamma_0))), rtol=1e-12)
    np.testing.assert_allclose(th.moments["Mean"].to_numpy(), 0.0, atol=1e-15)


def test_third_order_terms_are_of_order_sigma_squared_relative(solutions):
    sol = solutions["rbc"]
    m_x, m_u = sol._pruned_coefficients(1.0)["gx"], sol._pruned_coefficients(1.0)["gu"]
    gaps = []
    for sigma in (1.0, 0.5):
        th = sol.theoretical_moments(sigma=sigma, lags=1)
        _, gamma_0, _ = first_order_moments(sol.G, sol.N, m_x, m_u, sol._sigma_u(sigma, None), 1)
        order = sol._var_order()
        gaps.append(np.diag(th.covariance.to_numpy()) / np.diag(gamma_0[np.ix_(order, order)]) - 1.0)
    # Relative variance corrections scale with sigma^2: halving sigma quarters them. The
    # technology process a is linear, so its correction is zero.
    corrected = np.abs(gaps[0]) > 1e-8
    assert corrected.sum() == 3
    np.testing.assert_allclose(gaps[1][corrected] / gaps[0][corrected], 0.25, rtol=0.02)


def test_shape_moments_are_not_reported_as_gaussian(solutions):
    th = solutions["correlated_cubic"].theoretical_moments()
    assert th.moments["Skewness"].isna().all() and th.moments["Kurtosis"].isna().all()
    assert th.skewness.isna().all() and th.kurtosis.isna().all()
    assert len(th.autocorr_matrices) == 4


def test_ergodic_moments_exact_mean_variance_and_simulated_shape(solutions):
    sol = solutions["correlated_cubic"]
    th = sol.theoretical_moments(lags=1)
    erg = sol.ergodic_moments(periods=40_000, seed=5)
    np.testing.assert_allclose(erg["Mean"].to_numpy(), th.moments["Mean"].to_numpy(), rtol=1e-14)
    np.testing.assert_allclose(erg["Variance"].to_numpy(), th.moments["Variance"].to_numpy(), rtol=1e-14)
    np.testing.assert_allclose(erg["StdDev"].to_numpy(), th.moments["Std.Dev."].to_numpy(), rtol=1e-14)
    # x and z are Gaussian AR(1) processes; y loads on (x + 2 z)^3, so its tails are fat.
    assert abs(erg.loc["x", "Skewness"]) < 0.1 and abs(erg.loc["x", "Kurtosis"] - 3.0) < 0.15
    assert erg.loc["y", "Kurtosis"] > 10.0


def test_size_guard_and_stoch_simul_fallback(solutions, monkeypatch):
    sol = solutions["rbc"]
    with pytest.raises(ValueError, match="max_state_dim"):
        sol.theoretical_moments(max_state_dim=10)
    monkeypatch.setattr(pruning, "MAX_PRUNED_STATE_DIM", 10)
    with pytest.warns(RuntimeWarning, match="exact theoretical moments skipped"):
        res = sol.stoch_simul(irf=0, periods=50)
    assert res.theoretical_moments is None and res.simulated_moments is not None


def test_solve_accepts_pruning_keyword():
    model = load_mod((LIVE / "rbc.mod").read_text(encoding="utf-8"), order=1)
    assert isinstance(model.solve(order=3, pruning=True), Order3PrunedSolution)
    assert model.solve(order=1, pruning=False) is model
    for order in (2, 3):
        with pytest.raises(NotImplementedError, match="pruning"):
            model.solve(order=order, pruning=False)
    stoch = model.stoch_simul(order=3, irf=0)
    assert isinstance(stoch.theoretical_moments.moments, pd.DataFrame)
