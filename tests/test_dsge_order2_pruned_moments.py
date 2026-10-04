"""Exact moments of the pruned second-order solution (``pruning=True``).

``PrunedDSGESolution.theoretical_moments`` reports Dynare's order-2 convention
by default (first-order second moments, second-order mean). With
``pruning=True`` it reports the exact moments of the pruned solution, the
convention of Dynare's ``stoch_simul(order=2, pruning)``. Checks:

1. The augmented state space reproduces the pruned simulator period by period.
2. For a claim to x^2, y = A x^2 + C with x a Gaussian AR(1) is the exact
   solution, so Var(y) = 2 A^2 v^2 and corr(y_t, y_{t-k}) = rho^(2k).
3. Means, covariances and autocorrelations equal Dynare 8's pruned moments on
   the five live Dynare models and the claim (frozen in
   ``tests/fixtures/dynare_order2_pruned_moments.json``). At order 2 Dynare's
   innovations are uncorrelated over time, so its autocorrelations are exact.
4. The default convention is unchanged, and ``stoch_simul``, ``LinearModel``
   and ``verify_dynare_parity`` forward ``pruning``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge import load_mod
from puremacro.dsge._moments import first_order_moments
from puremacro.dsge._pruned_moments import pruned_state_space_order2
from puremacro.dsge.parity import verify_dynare_parity

pytestmark = pytest.mark.filterwarnings("ignore:forecast-error variance is zero")

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "tests" / "fixtures" / "dynare_live"
DYNARE = json.loads((ROOT / "tests" / "fixtures" / "dynare_order2_pruned_moments.json").read_text(encoding="utf-8"))
MODELS = sorted(DYNARE["cases"])


def _model_text(name: str) -> str:
    case = DYNARE["cases"][name]
    return case["model_text"] if "model_text" in case else (LIVE / f"{name}.mod").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def solutions():
    return {name: load_mod(_model_text(name), order=2) for name in MODELS}


def _claim_to_square(beta: float, rho: float, s: float):
    return load_mod(f"var y x; varexo e; parameters beta rho; beta={beta!r}; rho={rho!r};\n"
                    "model; x = rho*x(-1) + e; y = beta*y(+1) + x^2; end;\n"
                    f"initval; x=0; y=0; end;\nshocks; var e; stderr {s!r}; end;\n", order=2)


def test_fixture_models_are_the_ones_dynare_solved():
    for name, case in DYNARE["cases"].items():
        assert hashlib.sha256(_model_text(name).encode()).hexdigest() == case["model_sha256"], name


@pytest.mark.parametrize("name,sigma", [("correlated_cubic", 1.0), ("rbc", 1.0), ("nonlinear_multishock", 1.6)])
def test_state_space_reproduces_pruned_simulation(solutions, name, sigma):
    sol = solutions[name]
    n, e = sol.n_states, sol.n_shocks
    sigma_u = sol._sigma_u(sigma, None)
    ss = pruned_state_space_order2(sol._pruned_coefficients(sigma**2), n, sigma_u)
    a, b, c, cm, dm, d = (ss[k] for k in ("A", "B", "c", "C", "D", "d"))
    shocks = np.random.default_rng(5).multivariate_normal(np.zeros(e), sigma_u, size=150)
    x1, x2, y1, y2 = sol._pruned_path(shocks, sigma)
    levels = np.hstack([x1 + x2, y1 + y2])
    vs = sigma_u.reshape(-1)

    def z(t):
        return np.zeros(a.shape[0]) if t < 0 else np.concatenate([x1[t], x2[t], np.kron(x1[t], x1[t])])

    for t, u in enumerate(shocks):
        xf = x1[t - 1] if t else np.zeros(n)
        eps = np.concatenate([u, np.kron(u, u) - vs, np.kron(xf, u)])
        np.testing.assert_allclose(c + a @ z(t - 1) + b @ eps, z(t), rtol=0, atol=1e-13)
        np.testing.assert_allclose(d + cm @ z(t - 1) + dm @ eps, levels[t], rtol=0, atol=1e-13)


@pytest.mark.parametrize("beta,rho,s", [(0.95, 0.8, 0.1), (0.7, -0.5, 0.3), (0.99, 0.95, 0.05)])
def test_claim_to_square_matches_closed_form(beta, rho, s):
    sol = _claim_to_square(beta, rho, s)
    th = sol.theoretical_moments(lags=4, pruning=True)
    a = 1.0 / (1.0 - beta * rho**2)
    v = s**2 / (1.0 - rho**2)
    const = beta * a * s**2 / (1.0 - beta)
    assert th.covariance.loc["y", "y"] == pytest.approx(2.0 * a**2 * v**2, rel=1e-12)
    assert th.moments.loc["y", "Mean"] - sol.steady_state["y"] == pytest.approx(a * v + const, rel=1e-12)
    np.testing.assert_allclose(th.autocorr.loc["y"].to_numpy(), rho ** (2.0 * np.arange(1, 5)), rtol=1e-11)
    # The default convention has no second-order variance: y has no first-order response.
    assert sol.theoretical_moments(lags=1).covariance.loc["y", "y"] == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("name", MODELS)
def test_matches_dynare_pruned_moments(solutions, name):
    ref = DYNARE["cases"][name]
    names = ref["variables"]
    th = solutions[name].theoretical_moments(lags=5, pruning=True)
    np.testing.assert_allclose(th.moments.loc[names, "Mean"].to_numpy(), ref["mean"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(th.covariance.loc[names, names].to_numpy(), np.array(ref["var"]),
                               rtol=1e-9, atol=1e-12)
    for k, expected in enumerate(ref["autocorr"], start=1):
        np.testing.assert_allclose(th.autocorr_matrix(k).loc[names, names].to_numpy(), np.array(expected),
                                   rtol=1e-9, atol=1e-12)


def test_default_convention_is_first_order_second_moments(solutions):
    sol = solutions["rbc"]
    default = sol.theoretical_moments(lags=2)
    pruned = sol.theoretical_moments(lags=2, pruning=True)
    order = sol._var_order()
    m_x, m_u = np.vstack([sol.G, sol.F]), np.vstack([sol.N, sol.L])
    _, gamma_0, _ = first_order_moments(sol.G, sol.N, m_x, m_u, sol.shock_cov, 2)
    np.testing.assert_allclose(default.covariance.to_numpy(), gamma_0[np.ix_(order, order)], rtol=1e-12)
    # Both conventions share the second-order mean; the pruned variances are larger at order sigma^4.
    np.testing.assert_allclose(pruned.moments["Mean"].to_numpy(), default.moments["Mean"].to_numpy(), rtol=1e-12)
    gap = np.diag(pruned.covariance.to_numpy()) / np.diag(default.covariance.to_numpy()) - 1.0
    assert gap.max() > 1e-4 and np.all(gap > -1e-12)
    assert len(default.autocorr_matrices) == 2 and len(pruned.autocorr_matrices) == 2


def test_stoch_simul_and_linear_model_forward_pruning(solutions):
    sol = solutions["rbc"]
    direct = sol.theoretical_moments(lags=3, pruning=True)
    via_solution = sol.stoch_simul(irf=0, lags=3, pruning=True).theoretical_moments
    np.testing.assert_allclose(via_solution.covariance.to_numpy(), direct.covariance.to_numpy(), rtol=1e-14)
    model = load_mod((LIVE / "rbc.mod").read_text(encoding="utf-8"), order=1)
    via_model = model.stoch_simul(order=2, irf=0, lags=3, pruning=True).theoretical_moments
    np.testing.assert_allclose(via_model.covariance.loc[direct.covariance.index, direct.covariance.columns]
                               .to_numpy(), direct.covariance.to_numpy(), rtol=1e-10)
    default = model.stoch_simul(order=2, irf=0, lags=3).theoretical_moments
    assert not np.allclose(default.covariance.to_numpy(), via_model.covariance.to_numpy(), rtol=1e-6)
    with pytest.raises(NotImplementedError, match="pruning"):
        model.stoch_simul(order=3, irf=0, pruning=False)


def test_verify_dynare_parity_compares_pruned_references(solutions):
    ref = DYNARE["cases"]["rbc"]
    reference = np.load(LIVE / "rbc_order2.npz", allow_pickle=True)
    vn = [str(v).strip() for v in reference["variable_names"]]
    sn = [str(v).strip() for v in reference["state_names"]]
    en = [str(v).strip() for v in reference["shock_names"]]
    perm = [ref["variables"].index(v) for v in vn]
    oo = {"dr": {"ghx": reference["ghx"], "ghu": reference["ghu"], "ys": reference["ys"],
                 "state_var": np.array([vn.index(s) + 1 for s in sn]),
                 "ghxx": reference["ghxx"], "ghxu": reference["ghxu"], "ghuu": reference["ghuu"],
                 "ghs2": reference["ghs2"]},
          "mean": np.array(ref["mean"])[perm],
          "var": np.array(ref["var"])[np.ix_(perm, perm)],
          "autocorr": np.stack([np.diag(np.array(m))[perm] for m in ref["autocorr"]], axis=1)}
    raw = {"oo_": oo, "M_": {"endo_names": np.array(vn), "exo_names": np.array(en),
                             "Sigma_e": reference["shock_cov"]}}
    pruned = verify_dynare_parity(solutions["rbc"], raw, order=2, tol=1e-10, pruning=True)
    assert pruned.details["moments_status"] == "COMPARED" and pruned.passed
    rows = pruned.moments_diff.set_index("moment")
    assert (rows["status"] == "PASS").all() and rows["deviation"].max() < 1e-12
    # The default convention compares first-order second moments and misses this reference.
    default = verify_dynare_parity(solutions["rbc"], raw, order=2, tol=1e-10)
    assert not default.passed


def test_size_guard(solutions):
    with pytest.raises(ValueError, match="max_state_dim"):
        solutions["rbc"].theoretical_moments(pruning=True, max_state_dim=5)
