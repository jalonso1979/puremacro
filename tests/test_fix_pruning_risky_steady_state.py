"""Ergodic mean versus risky steady state of the pruned perturbation solutions.

Before the fix ``stochastic_steady_state()`` was the only entry point and its
name and docstring ("risk-adjusted steady state") suggested the zero-shock
risky steady state, while it returns the ergodic mean. In the RBC model of
``tests/fixtures/dynare_live/rbc.mod`` the two put capital on opposite sides of
the deterministic steady state (+0.0785% against -0.0045%), and notebook 68
read the positive ergodic-mean shift as "precautionary capital".

The references are closed forms derived in the docstrings below and numbers
produced by Dynare itself (the frozen fixtures in ``tests/fixtures``), never by
puremacro.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import PrunedDSGESolution, load_mod
from puremacro.dsge.pruning import Order3PrunedSolution

FIXTURES = Path(__file__).parent / "fixtures"
RBC_TEXT = (FIXTURES / "dynare_live" / "rbc.mod").read_text(encoding="utf-8")
PARTS = ["risk", "state_curvature", "shock_curvature"]


def _as_series(d: dict[str, pd.Series]) -> pd.Series:
    return pd.concat([d["states"], d["controls"]])


def _claim_text(beta: float, rho: float, s: float) -> str:
    return (f"var y x; varexo e; parameters beta rho; beta={beta!r}; rho={rho!r};\n"
            "model; x = rho*x(-1) + e; y = beta*y(+1) + x^2; end;\n"
            f"initval; x=0; y=0; end;\nshocks; var e; stderr {s!r}; end;\n")


def _claim_closed_form(beta: float, rho: float, s: float) -> dict[str, float]:
    """y_t = beta E_t y_{t+1} + x_t^2 with x_t = rho x_{t-1} + e_t, Var(e) = s^2.

    The exact solution is y = A x^2 + B with A = 1/(1 - beta rho^2) and
    B = beta A s^2 / (1 - beta). With v = Var(x) = s^2/(1 - rho^2):
    E[y] = A v + B, and with no realized shocks x stays at 0, so the risky
    steady state of y is B. In Dynare timing y_t = A(rho x_{t-1} + e_t)^2 + B,
    so 0.5 g_xx vec(Omega) = A rho^2 v, 0.5 g_uu s^2 = A s^2, 0.5 g_ss = B.
    """
    A = 1.0 / (1.0 - beta * rho**2)
    B = beta * A * s**2 / (1.0 - beta)
    v = s**2 / (1.0 - rho**2)
    return {"A": A, "B": B, "v": v, "mean": A * v + B, "risk": B,
            "state_curvature": A * rho**2 * v, "shock_curvature": A * s**2}


CLAIM_CALIBRATIONS = [(0.95, 0.8, 0.1), (0.9, -0.5, 0.3), (0.99, 0.95, 0.02), (0.6, 0.3, 1.0)]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("beta,rho,s", CLAIM_CALIBRATIONS)
def test_claim_to_square_mean_risky_and_decomposition(order, beta, rho, s):
    sol = load_mod(_claim_text(beta, rho, s), order=order)
    cf = _claim_closed_form(beta, rho, s)

    mean = _as_series(sol.ergodic_mean())
    risky = _as_series(sol.risky_steady_state())
    np.testing.assert_allclose(mean["y"], cf["mean"], rtol=1e-10)
    np.testing.assert_allclose(risky["y"], cf["risk"], rtol=1e-10)
    assert abs(mean["x"]) < 1e-14 and abs(risky["x"]) < 1e-14

    dec = sol.risk_decomposition()
    assert list(dec.columns) == PARTS + ["ergodic_mean"]
    for part in PARTS:
        np.testing.assert_allclose(dec.loc["y", part], cf[part], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(dec.loc["y", "ergodic_mean"], cf["mean"], rtol=1e-10)

    # x is linear, so the unpruned zero-shock fixed point is the pruned one.
    unpruned = _as_series(sol.risky_steady_state(pruned=False))
    np.testing.assert_allclose(unpruned["y"], cf["risk"], rtol=1e-10)


def test_claim_to_square_matches_dynare8_pruned_mean():
    """Dynare 8's oo_.mean for stoch_simul(order=2, pruning) on the same text."""
    case = json.loads((FIXTURES / "dynare_order2_pruned_moments.json").read_text(encoding="utf-8"))["cases"]["claim_to_square"]
    sol = load_mod(case["model_text"], order=2)
    level = _as_series(sol.ergodic_mean()) + sol.steady_state
    dynare = pd.Series(case["mean"], index=case["variables"])
    np.testing.assert_allclose(level[dynare.index].to_numpy(), dynare.to_numpy(), rtol=1e-12, atol=1e-14)
    cf = _claim_closed_form(0.95, 0.8, 0.1)
    np.testing.assert_allclose(dynare["y"], cf["mean"], rtol=1e-12)


def _brock_mirman(order: int, alpha=0.33, beta=0.96, rho=0.9, s=0.05):
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    c_ss = (1.0 - alpha * beta) * k_ss**alpha
    text = (f"var c k a; varexo e; parameters alpha beta rho; alpha={alpha!r}; beta={beta!r}; rho={rho!r};\n"
            "model; 1/c = beta/c(+1)*alpha*exp(a(+1))*k^(alpha-1); c + k = exp(a)*k(-1)^alpha; "
            "a = rho*a(-1) + e; end;\n"
            f"initval; k={k_ss!r}; c={c_ss!r}; a=0; end;\nshocks; var e; stderr {s!r}; end;\n")
    # log k_t - log k_ss = alpha (log k_{t-1} - log k_ss) + a_t is an AR(2) with
    # roots alpha and rho: Var = (1 + alpha rho) s^2 / ((1 - alpha rho)(1 - alpha^2)(1 - rho^2)).
    var_log = (1 + alpha * rho) * s**2 / ((1 - alpha * rho) * (1 - alpha**2) * (1 - rho**2))
    return load_mod(text, order=order), k_ss, c_ss, var_log


@pytest.mark.parametrize("order", [2, 3])
def test_brock_mirman_has_no_risk_term_but_a_positive_mean(order):
    """Certainty equivalence holds exactly: g_sigma_sigma = 0 and the risky steady state is the deterministic one.

    k_t = k_ss exp(l_t) with l_t linear, so to second order (pruned) k - k_ss =
    k_ss (l + l^2/2) and E[k] - k_ss = k_ss Var(l)/2 > 0: a mean shift with no
    precaution at all. c_t is proportional to k_t, so E[c] - c_ss = c_ss Var(l)/2.
    """
    sol, k_ss, c_ss, var_log = _brock_mirman(order)
    dec = sol.risk_decomposition()
    risky = _as_series(sol.risky_steady_state())
    assert np.max(np.abs(dec["risk"])) < 1e-14
    assert np.max(np.abs(risky)) < 1e-14
    assert np.max(np.abs(_as_series(sol.risky_steady_state(pruned=False)))) < 1e-14

    mean = _as_series(sol.ergodic_mean())
    np.testing.assert_allclose(mean["k"], k_ss * var_log / 2, rtol=1e-9)
    np.testing.assert_allclose(mean["c"], c_ss * var_log / 2, rtol=1e-9)
    assert abs(mean["a"]) < 1e-15
    np.testing.assert_allclose(dec["ergodic_mean"], mean[dec.index], rtol=0, atol=0)


def _dynare7(order: int) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "dynare_live" / f"rbc_order{order}.npz") as ref:
        return {k: ref[k] for k in ref.files}


def _parts_from_dynare_tensors(ref: dict[str, np.ndarray]) -> pd.DataFrame:
    """The three mean components computed from Dynare 7's own ghx, ghu, ghxx, ghuu, ghs2."""
    names = [str(v) for v in ref["variable_names"]]
    states = [str(v) for v in ref["state_names"]]
    rows = [names.index(v) for v in states]
    gx, gu = ref["ghx"], ref["ghu"]
    G, N = gx[rows], gu[rows]
    cov = ref["shock_cov"]
    omega = np.zeros_like(G)
    for _ in range(20_000):  # Omega = G Omega G' + N cov N', by fixed-point iteration
        omega = G @ omega @ G.T + N @ cov @ N.T
    forcing = {"risk": 0.5 * ref["ghs2"].ravel(), "state_curvature": 0.5 * ref["ghxx"] @ omega.ravel(),
               "shock_curvature": 0.5 * ref["ghuu"] @ cov.ravel()}
    out = {}
    for part, f in forcing.items():
        x = np.linalg.solve(np.eye(len(states)) - G, f[rows])
        out[part] = gx @ x + f
        out[part][rows] = x
    return pd.DataFrame(out, index=names)


@pytest.mark.parametrize("order", [2, 3])
def test_rbc_decomposition_sums_to_ergodic_mean_and_matches_dynare(order):
    sol = load_mod(RBC_TEXT, order=order)
    assert isinstance(sol, PrunedDSGESolution if order == 2 else Order3PrunedSolution)
    dec = sol.risk_decomposition()
    mean = _as_series(sol.ergodic_mean())

    # The three parts sum to the ergodic mean.
    np.testing.assert_allclose(dec[PARTS].sum(axis=1), dec["ergodic_mean"], rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(dec["ergodic_mean"], mean[dec.index], rtol=0, atol=0)
    assert list(dec.index) == list(sol.variable_names)

    # Each part against the same closed form evaluated on Dynare 7's own tensors.
    ref = _parts_from_dynare_tensors(_dynare7(order))
    for part in PARTS:
        np.testing.assert_allclose(dec.loc[ref.index, part], ref[part], rtol=1e-8, atol=1e-12)

    # The total against the independent augmented-state moment code ...
    kwargs = {"pruning": True} if order == 2 else {}
    theo_mean = sol.theoretical_moments(lags=1, **kwargs).moments["Mean"] - sol.steady_state
    np.testing.assert_allclose(theo_mean[dec.index], dec["ergodic_mean"], rtol=1e-10, atol=1e-14)

    # ... and against Dynare 8's exact pruned oo_.mean.
    fixture = f"dynare_order{order}_pruned_moments.json"
    case = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))["cases"]["rbc"]
    dynare_mean = pd.Series(case["mean"], index=case["variables"])
    level = mean + sol.steady_state
    np.testing.assert_allclose(level[dynare_mean.index], dynare_mean, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("order", [2, 3])
def test_rbc_ergodic_mean_and_risky_steady_state_have_opposite_signs(order):
    """The defect's teaching consequence: the ergodic mean of k is not precautionary capital."""
    sol = load_mod(RBC_TEXT, order=order)
    k_ss = sol.steady_state["k"]
    mean_k = sol.ergodic_mean()["states"]["k"]
    risky_k = sol.risky_steady_state()["states"]["k"]
    dec = sol.risk_decomposition()

    assert round(100 * mean_k / k_ss, 4) == 0.0785
    assert round(100 * risky_k / k_ss, 4) == -0.0045
    assert mean_k > 0 > risky_k
    np.testing.assert_allclose(dec.loc["k", "risk"], risky_k, rtol=1e-14)
    assert round(100 * dec.loc["k", "state_curvature"] / k_ss, 4) == 0.0705
    assert round(100 * dec.loc["k", "shock_curvature"] / k_ss, 4) == 0.0125

    # Dynare 7's own risk term for k is negative: anticipated risk lowers capital.
    ref = _dynare7(order)
    names = [str(v) for v in ref["variable_names"]]
    assert ref["ghs2"].ravel()[names.index("k")] < 0

    # The pruned risky steady state is the limit of a zero-shock pruned path.
    T = 6000
    path = sol.simulate(periods=T, shocks=np.zeros((T, sol.n_shocks)), burn=0)
    last = pd.concat([path.states.iloc[-1], path.controls.iloc[-1]])
    risky = _as_series(sol.risky_steady_state())
    np.testing.assert_allclose(last[risky.index], risky, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("order", [2, 3])
def test_unpruned_fixed_point_differs_by_order_sigma4(order):
    sol = load_mod(RBC_TEXT, order=order)
    gap = {m: sol.risky_steady_state(sigma=m, pruned=False)["states"]["k"]
           - sol.risky_steady_state(sigma=m)["states"]["k"] for m in (1.0, 2.0, 4.0)}
    assert gap[1.0] != 0.0
    assert abs(gap[2.0] / gap[1.0] - 16.0) < 0.05
    assert abs(gap[4.0] / gap[2.0] - 16.0) < 0.05
    assert abs(gap[1.0]) < 1e-6 * sol.steady_state["k"]


def test_unpruned_fixed_point_is_the_limit_of_simulate_raw():
    sol = load_mod(RBC_TEXT, order=2)
    T = 6000
    x_raw, y_raw = sol.simulate_raw(periods=T, shocks=np.zeros((T + 100, 1)), burn=100)
    fp = sol.risky_steady_state(pruned=False)
    np.testing.assert_allclose(x_raw[-1], fp["states"].to_numpy(), rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(y_raw[-1], fp["controls"].to_numpy(), rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("order", [2, 3])
def test_stochastic_steady_state_is_a_documented_alias_of_ergodic_mean(order):
    sol = load_mod(RBC_TEXT, order=order)
    for sigma in (1.0, 2.5):
        new, old = sol.ergodic_mean(sigma=sigma), sol.stochastic_steady_state(sigma=sigma)
        for key in ("states", "controls"):
            pd.testing.assert_series_equal(new[key], old[key], check_exact=True)
        assert new["states"].name == "ergodic_mean_states"
    doc = inspect.getdoc(type(sol).stochastic_steady_state)
    assert "ergodic mean" in doc and "not the risky steady state" in doc
    assert "risky_steady_state" in doc
    assert "risk-adjusted steady state" not in doc
    assert "risk term" in inspect.getdoc(type(sol).ergodic_mean)


def test_risk_and_curvature_scale_with_the_variance():
    sol = load_mod(RBC_TEXT, order=3)
    one, three = sol.risk_decomposition(sigma=1.0), sol.risk_decomposition(sigma=3.0)
    np.testing.assert_allclose(three.loc["k"], 9.0 * one.loc["k"], rtol=1e-10)


def test_risky_steady_state_errors():
    sol = load_mod(RBC_TEXT, order=2)
    with pytest.raises(RuntimeError, match="Newton did not reach"):
        sol.risky_steady_state(pruned=False, maxiter=0)
    with pytest.raises(TypeError):
        sol.risky_steady_state(sigma={"e": 1.0})

    unit_root = PrunedDSGESolution(
        G=np.array([[1.0]]), N=np.array([[1.0]]), F=np.array([[0.5]]), L=np.array([[0.0]]),
        H_xx=np.zeros((1, 1)), H_sigmasigma=np.array([0.1]), G_xx=np.zeros((1, 1)),
        G_sigmasigma=np.array([0.0]), state_names=("k",), control_names=("c",), shock_names=("e",),
    )
    with pytest.raises(ValueError, match="unit circle"):
        unit_root.risky_steady_state()
