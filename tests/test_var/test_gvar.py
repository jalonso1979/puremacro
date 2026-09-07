"""Behaviour pins for :mod:`puremacro.var.gvar`.

The tests are identities, reductions to estimators already in the package,
invariances, planted-parameter recovery and the degenerate cases from the
design -- not restatements of the implementation. Where the design and the
adversarial critique disagreed on a tolerance or a claim, the critique wins:
the ``>= 1`` bound on the unnormalised GFEVD row sums is pinned only where it
is actually provable (``G = I``) and replaced elsewhere by the Rayleigh-quotient
bound that does hold, the lag-selection test pins *within-grid* comparability
rather than a cross-grid invariance the algorithm does not have, and the
weak-exogeneity F test's measured size distortion is pinned by a Monte Carlo
rather than waved away.

Every assertion here is meant to be *falsifiable*: no bound that is true by
construction, no threshold placed forty standard errors from the measured value.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.var.estimate import estimate_var
from puremacro.var.gvar import (
    CountryVARX,
    GVARGIRFResult,
    GVARResult,
    WeakExogeneityResult,
    gvar,
    solve_gvar,
    star_variables,
)
from puremacro.var.irf import gfevd as var_gfevd

COUNTRIES = ("A", "B", "C")


def ma_coefficients(A_list, horizon):
    """Independent MA(inf) recursion, written here and not imported.

    ``Psi_0 = I``, ``Psi_h = sum_{l=1..min(h, s)} A_l Psi_{h-l}``. Deliberately
    a second implementation: a reference built from the module under test
    checks that module against itself.
    """
    k = A_list[0].shape[0]
    s = len(A_list)
    Psi = np.zeros((horizon + 1, k, k))
    Psi[0] = np.eye(k)
    for h in range(1, horizon + 1):
        acc = np.zeros((k, k))
        for l in range(1, min(h, s) + 1):
            acc += A_list[l - 1] @ Psi[h - l]
        Psi[h] = acc
    return Psi


def R_matrices(res, horizon):
    """``R_h = Psi_h G^-1`` from the result's own solved matrices."""
    return ma_coefficients([res.F[l] for l in range(res.s)], horizon) @ res.G_inv


# ---------------------------------------------------------------------------
# Fixtures / DGPs
# ---------------------------------------------------------------------------
def _panel_from_array(X, countries=COUNTRIES, varnames=("y", "r")):
    """Wrap a (T, N*k_i) array as a dict of per-country frames."""
    T = X.shape[0]
    kv = len(varnames)
    dates = pd.RangeIndex(T, name="date")
    return {
        c: pd.DataFrame(X[:, i * kv:(i + 1) * kv], columns=list(varnames), index=dates)
        for i, c in enumerate(countries)
    }


def _trade_weights(countries=COUNTRIES, values=None):
    if values is None:
        values = [[0.0, 0.6, 0.4], [0.5, 0.0, 0.5], [0.3, 0.7, 0.0]]
    return pd.DataFrame(np.asarray(values, float),
                        index=list(countries), columns=list(countries))


def stable_global_dgp(T=200, seed=3, k=6, radius=0.85):
    """A stable global VAR(1) on k variables; 3 countries x 2 variables."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-0.25, 0.25, (k, k))
    np.fill_diagonal(A, rng.uniform(0.3, 0.6, k))
    A *= radius / np.max(np.abs(np.linalg.eigvals(A)))
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + rng.standard_normal(k)
    return X


def independent_country_dgp(T=180, seed=11, n_countries=3, kv=2, rho=0.5):
    """Countries that never interact: the weak-exogeneity null."""
    rng = np.random.default_rng(seed)
    k = n_countries * kv
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = rho * X[t - 1] + rng.standard_normal(k)
    return X


def leaking_country_dgp(T=250, seed=5, b=2.0, rho_foreign=0.9):
    """Country A's innovation feeds B and C one period later, persistently.

    Country A's own foreign block is then Granger-caused by A's own model
    innovations -- exactly the alternative the weak-exogeneity test targets.
    The foreign countries are highly persistent so the innovation's imprint on
    ``D x*`` is not spanned by the differenced regressors the auxiliary
    regression already conditions on.
    """
    rng = np.random.default_rng(seed)
    k = 6
    u = rng.standard_normal((T, k))
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + u[t, 0]
        X[t, 1] = 0.5 * X[t - 1, 1] + u[t, 1]
        for j in (2, 3, 4, 5):
            X[t, j] = rho_foreign * X[t - 1, j] + b * u[t - 1, 0] + u[t, j]
    return X


@pytest.fixture(scope="module")
def fitted():
    """A three-country VARX*(2, 1) on a stable global DGP."""
    frames = _panel_from_array(stable_global_dgp())
    return gvar(frames, _trade_weights(), p=2, q=1)


# ---------------------------------------------------------------------------
# The load-bearing identities
# ---------------------------------------------------------------------------
def test_link_identity_exact(fitted):
    """G x_t = a_0 + a_1 t + sum_l H_l x_{t-l} + eps_t at every estimation date."""
    res = fitted
    X = res.panel.to_numpy(dtype=float)
    E = res.resid.to_numpy(dtype=float)
    a0 = res.G @ res.intercept
    a1 = res.G @ res.trend_coef
    t0 = res.spec.t0
    err = 0.0
    for i, t in enumerate(range(t0, len(X))):
        rhs = a0 + a1 * (t + 1.0) + E[i]
        for l in range(res.s):
            rhs = rhs + res.H[l] @ X[t - l - 1]
        err = max(err, float(np.max(np.abs(res.G @ X[t] - rhs))))
    assert err < 1e-12, f"link identity broken at {err:.3e}"


def test_link_identity_exact_with_trend():
    """The identity must survive trend='ct', where a_1 t is non-zero."""
    frames = _panel_from_array(stable_global_dgp(seed=19))
    res = gvar(frames, _trade_weights(), p=2, q=1, trend="ct")
    assert np.any(res.trend_coef != 0.0)
    X = res.panel.to_numpy(dtype=float)
    E = res.resid.to_numpy(dtype=float)
    a0, a1 = res.G @ res.intercept, res.G @ res.trend_coef
    err = 0.0
    for i, t in enumerate(range(res.spec.t0, len(X))):
        rhs = a0 + a1 * (t + 1.0) + E[i]
        for l in range(res.s):
            rhs = rhs + res.H[l] @ X[t - l - 1]
        err = max(err, float(np.max(np.abs(res.G @ X[t] - rhs))))
    assert err < 1e-12


def test_solved_form_identity(fitted):
    """x_t = intercept + sum_l F_l x_{t-l} + G^-1 eps_t, pinning F = G^-1 H."""
    res = fitted
    X = res.panel.to_numpy(dtype=float)
    E = res.resid.to_numpy(dtype=float)
    err = 0.0
    for i, t in enumerate(range(res.spec.t0, len(X))):
        rhs = res.intercept + res.trend_coef * (t + 1.0) + res.G_inv @ E[i]
        for l in range(res.s):
            rhs = rhs + res.F[l] @ X[t - l - 1]
        err = max(err, float(np.max(np.abs(X[t] - rhs))))
    assert err < 1e-12
    for l in range(res.s):
        assert np.allclose(res.G @ res.F[l], res.H[l], atol=1e-10)
    assert np.allclose(res.G @ res.G_inv, np.eye(res.n_variables), atol=1e-10)


def test_no_star_block_equals_independent_country_vars():
    """star_vars={} makes G the identity and each block a plain estimate_var."""
    frames = _panel_from_array(stable_global_dgp(seed=23))
    # q > 0 is now an error for a country with no star block (G3): there is
    # nothing to lag, so the keyword cannot bite and must not be a no-op.
    with pytest.raises(ValueError, match=r"q=1 was requested for country 'A'"):
        gvar(frames, _trade_weights(), p=2, q=1,
             star_vars={c: () for c in COUNTRIES})
    res = gvar(frames, _trade_weights(), p=2, q=0,
               star_vars={c: () for c in COUNTRIES})
    k = res.n_variables
    assert np.array_equal(res.G, np.eye(k)), "G must be exactly I with no star block"
    # H is exactly block diagonal: every off-block entry is a hard zero.
    for l in range(res.s):
        M = res.H[l]
        for a in COUNTRIES:
            ra = res.spec.own_idx[a]
            for b in COUNTRIES:
                if a == b:
                    continue
                rb = res.spec.own_idx[b]
                assert np.count_nonzero(M[np.ix_(ra, rb)]) == 0

    for c in COUNTRIES:
        m = res.country_models[c]
        assert m.q == 0
        ref = estimate_var(frames[c].to_numpy(dtype=float), 2)
        for l in range(2):
            assert np.allclose(m.Phi[l], ref.A_list[l], atol=1e-12)
        assert np.allclose(m.a0, ref.c, atol=1e-12)
        assert np.allclose(m.resid.to_numpy(dtype=float), ref.resid, atol=1e-12)
        # T_eff - m_i == T - p - 1 - k_i*p, so the divisors coincide exactly.
        assert np.allclose(m.Sigma, ref.Sigma, atol=1e-12)


def test_solve_gvar_matches_a_hand_built_two_country_system():
    """The link algebra on hand-built blocks, checked against pen and paper."""
    S = {"A": np.array([[1.0, 0.0]]), "B": np.array([[0.0, 1.0]])}
    Wst = {"A": np.array([[0.0, 1.0]]), "B": np.array([[1.0, 0.0]])}
    link = {c: np.vstack([S[c], Wst[c]]) for c in ("A", "B")}
    A0 = {"A": np.array([[1.0, -0.3]]), "B": np.array([[1.0, -0.2]])}
    Al = {"A": [np.array([[0.5, 0.1]])], "B": [np.array([[0.4, 0.05]])]}
    sol = solve_gvar(link, A0, Al, country_order=["A", "B"],
                     block_sizes={"A": 1, "B": 1},
                     a0={"A": np.array([1.0]), "B": np.array([-2.0])})
    assert np.allclose(sol.G, np.array([[1.0, -0.3], [-0.2, 1.0]]))
    assert np.allclose(sol.H[0], np.array([[0.5, 0.1], [0.05, 0.4]]))
    assert np.allclose(sol.F[0], np.linalg.solve(sol.G, sol.H[0]))
    assert np.allclose(sol.a0, np.linalg.solve(sol.G, np.array([1.0, -2.0])))


# ---------------------------------------------------------------------------
# Star weights and the star panel
# ---------------------------------------------------------------------------
def test_star_weights_renormalise_over_donor_countries():
    """A country missing a variable is dropped from that variable's star row."""
    X = stable_global_dgp(seed=31)
    frames = _panel_from_array(X)
    frames["C"] = frames["C"].assign(r=np.nan)  # C does not carry 'r'
    with pytest.warns(RuntimeWarning, match=r"allow_missing_variables=True dropped"):
        res = gvar(frames, _trade_weights(), p=1, q=1,
                   allow_missing_variables=True)
    assert res.n_variables == 5
    assert res.names == ("A:y", "A:r", "B:y", "B:r", "C:y")
    for c in ("A", "B"):
        row = res.star_weights[c].loc["r"]
        assert abs(float(row.sum()) - 1.0) < 1e-12
        assert float(row["C"]) == 0.0
        # 'y' is carried by everyone, so its row is the raw trade row.
        assert np.allclose(res.star_weights[c].loc["y"].to_numpy(),
                           res.weights.loc[c].to_numpy(), atol=1e-12)
    with pytest.warns(RuntimeWarning):
        sd = star_variables(frames, _trade_weights(),
                            allow_missing_variables=True)
    assert np.allclose(sd.to_numpy(dtype=float),
                       res.star_data.to_numpy(dtype=float),
                       atol=0.0, equal_nan=True)


def test_star_variables_matches_a_hand_rolled_weighted_average():
    frames = _panel_from_array(stable_global_dgp(T=40, seed=2))
    W = _trade_weights()
    xs = star_variables(frames, W)
    manual = (W.loc["A", "B"] * frames["B"]["y"] + W.loc["A", "C"] * frames["C"]["y"])
    assert np.allclose(xs.loc["A"]["y_star"].to_numpy(dtype=float),
                       manual.to_numpy(dtype=float), atol=1e-12)


def test_star_variable_may_be_borrowed_from_a_donor_country():
    """A country can carry a foreign variable it does not hold itself."""
    frames = _panel_from_array(stable_global_dgp(seed=41))
    frames["C"] = frames["C"].assign(r=np.nan)
    with pytest.warns(RuntimeWarning):
        res = gvar(frames, _trade_weights(), p=1, q=1,
                   star_vars={"C": ("y", "r"), "A": ("y", "r"), "B": ("y", "r")},
                   allow_missing_variables=True)
    assert res.country_models["C"].star_variables == ("y", "r")
    assert res.country_models["C"].variables == ("y",)
    row = res.star_weights["C"].loc["r"]
    assert abs(float(row.sum()) - 1.0) < 1e-12


def test_star_variable_without_donor_raises():
    frames = _panel_from_array(stable_global_dgp(seed=43))
    with pytest.raises(ValueError, match=r"star variable 'z'.*country 'A'"):
        gvar(frames, _trade_weights(), p=1, q=1, star_vars={"A": ("z",)})


def test_star_variables_suffix_collision_raises():
    frames = _panel_from_array(stable_global_dgp(T=40, seed=7))
    frames = {c: df.assign(y_star=df["y"] * 0.5) for c, df in frames.items()}
    with pytest.raises(ValueError, match="collide"):
        star_variables(frames, _trade_weights())


# ---------------------------------------------------------------------------
# GIRF / GFEVD / persistence profiles
# ---------------------------------------------------------------------------
def test_single_country_girf_equals_pesaran_shin_generalised_irf():
    """N = 1 reduces to the textbook generalised IRF of a VAR(p)."""
    X = stable_global_dgp(T=160, seed=53, k=3)
    dates = pd.RangeIndex(len(X), name="date")
    frames = {"US": pd.DataFrame(X, columns=["y", "r", "pi"], index=dates)}
    W = pd.DataFrame([[0.0]], index=["US"], columns=["US"])
    m_i = 1 + 3 * 2
    res = gvar(frames, W, p=2, q=0, sigma_ddof=m_i, weak_exogeneity=False)
    assert np.array_equal(res.G, np.eye(3))
    ref = estimate_var(X, 2)
    # `ma_coefficients` is written in this file, not imported from the module
    # under test: a reference built from the implementation would only check
    # the implementation against itself.
    Psi = ma_coefficients(ref.A_list, 12)
    assert np.allclose(res.Sigma_eps, ref.Sigma, atol=1e-12)
    for j, v in enumerate(["y", "r", "pi"]):
        want = (Psi @ ref.Sigma[:, j]) / np.sqrt(ref.Sigma[j, j])
        got = res.girf("US", v, horizon=12).irf
        assert np.allclose(got, want, atol=1e-12)


def test_girf_is_homogeneous_of_degree_one_half_in_sigma():
    """sigma_ddof scales every response by sqrt(T_eff / (T_eff - ddof))."""
    frames = _panel_from_array(stable_global_dgp(seed=61))
    base = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    d = 17
    scaled = gvar(frames, _trade_weights(), p=1, q=1, sigma_ddof=d,
                  weak_exogeneity=False)
    factor = np.sqrt(base.n_obs / (base.n_obs - d))
    a = base.girf("A", "y", horizon=6).irf
    b = scaled.girf("A", "y", horizon=6).irf
    assert np.allclose(b, factor * a, atol=1e-12)
    # ... and nothing else moved.
    assert np.allclose(base.F, scaled.F, atol=1e-14)


def test_girf_combination_reduces_to_a_single_shock_and_ignores_scale(fitted):
    single = fitted.girf("B", "r", horizon=10)
    combo = fitted.girf_combination({("B", "r"): 1.0}, horizon=10)
    assert np.array_equal(single.irf, combo.irf)
    bigger = fitted.girf_combination({("B", "r"): 3.0}, horizon=10)
    assert np.allclose(bigger.irf, combo.irf, atol=1e-14)
    mixed = fitted.girf_combination({("A", "y"): 1.0, ("B", "y"): 1.0}, horizon=4)
    assert not np.allclose(mixed.irf, fitted.girf("A", "y", horizon=4).irf)


def test_girf_cumulative_is_the_cumulative_sum(fitted):
    lvl = fitted.girf("A", "y", horizon=8).irf
    cum = fitted.girf("A", "y", horizon=8, cumulative=True).irf
    assert np.allclose(cum, np.cumsum(lvl, axis=0), atol=1e-14)


def test_gfevd_matches_var_gfevd_when_G_is_identity():
    frames = _panel_from_array(stable_global_dgp(seed=67))
    res = gvar(frames, _trade_weights(), p=2, q=0,
               star_vars={c: () for c in COUNTRIES}, weak_exogeneity=False)
    assert np.array_equal(res.G, np.eye(res.n_variables))
    want = var_gfevd([res.F[l] for l in range(res.s)], res.Sigma_eps, 8)
    assert np.allclose(res.gfevd(8), want, atol=1e-12)
    raw = res.gfevd(8, normalize=False)
    assert np.allclose(raw,
                       var_gfevd([res.F[l] for l in range(res.s)],
                                 res.Sigma_eps, 8, normalize=False),
                       atol=1e-12)
    # THIS is where the Pesaran-Shin bound lives: R_0 = G^-1 = I, so the impact
    # row sum is sum_j r_ij^2 >= r_ii^2 = 1 for the residual correlation r.
    d = np.sqrt(np.diag(res.Sigma_eps))
    corr = res.Sigma_eps / np.outer(d, d)
    assert np.allclose(raw[0].sum(axis=1), (corr ** 2).sum(axis=1), atol=1e-12)
    assert np.all(raw[0].sum(axis=1) >= 1.0 - 1e-12)


def test_gfevd_impact_row_sums_obey_the_correlation_eigenvalue_bound(fitted):
    """The Pesaran-Shin ``>= 1`` bound is FALSE for a GVAR, even at h = 0.

    Their proof needs ``Psi_0 = I``; a GVAR has ``R_0 = G^-1``. With
    ``a = R_0' e_i``, ``b = Sigma^(1/2) a`` and ``C`` the column-normalised
    ``Sigma^(1/2)``, the impact row sum is ``|C'b|^2/|b|^2``, a Rayleigh
    quotient of ``C C'`` whose spectrum is that of ``corr(Sigma)``. That is the
    bound this test pins -- and it pins that the naive ``>= 1`` claim really
    does fail here, so nobody reinstates it.
    """
    raw = fitted.gfevd(12, normalize=False)
    assert np.all(raw >= -1e-15), "generalised shares are non-negative"

    S = fitted.Sigma_eps
    d = np.sqrt(np.diag(S))
    corr = S / np.outer(d, d)
    lo, hi = np.linalg.eigvalsh(corr)[[0, -1]]
    assert lo < 1.0 < hi, "a non-degenerate correlation matrix straddles 1"

    rows = raw[0].sum(axis=1)
    assert np.all(rows >= lo - 1e-10), (rows, lo)
    assert np.all(rows <= hi + 1e-10), (rows, hi)
    # The bound is tight enough to be informative, i.e. not vacuous.
    assert hi - lo < 1.0

    norm = fitted.gfevd(12, normalize=True)
    assert np.allclose(norm.sum(axis=2), 1.0, atol=1e-12)
    assert np.all(norm >= -1e-15)


def test_gfevd_impact_row_sums_fall_below_one_on_a_normal_gvar():
    """Not a knife-edge: on this DGP family most seeds break the >= 1 claim.

    The shipped module used to assert ``>= 1`` at impact and passed only
    because the fixture's seed is one of the few that satisfy it. Sweeping the
    seed shows the claim is false for the majority of fits, which is why the
    docstring states the eigenvalue bound instead.
    """
    below = 0
    for seed in range(1, 25):
        frames = _panel_from_array(stable_global_dgp(seed=seed))
        res = gvar(frames, _trade_weights(), p=2, q=1, weak_exogeneity=False)
        rows = res.gfevd(0, normalize=False)[0].sum(axis=1)
        if rows.min() < 1.0 - 1e-12:
            below += 1
    assert below >= 15, f"only {below}/24 seeds had an impact row sum below 1"


def test_persistence_profile_matches_an_independent_recomputation(fitted):
    """``PP(b, h) = b' R_h Sigma R_h' b / (b' R_0 Sigma R_0' b)`` term by term.

    ``pp[0] == 1`` is true by construction (the method divides by its own first
    entry) and so cannot fail for any input; it is asserted here only as a
    by-product. The content is the whole profile, rebuilt from ``F``, ``G_inv``
    and ``Sigma_eps`` with the MA recursion written in this file, plus the
    scale invariance in ``b`` that the ratio implies.
    """
    H = 10
    R = R_matrices(fitted, H)
    S = fitted.Sigma_eps
    for vec in (
        {("A", "y"): 1.0},
        {("A", "y"): 1.0, ("B", "y"): -1.0},
        np.arange(1.0, fitted.n_variables + 1.0),
    ):
        if isinstance(vec, dict):
            b = np.zeros(fitted.n_variables)
            for key, val in vec.items():
                b[fitted.name_pairs.index(key)] = val
        else:
            b = np.asarray(vec, float)
        num = np.array([b @ R[h] @ S @ R[h].T @ b for h in range(H + 1)])
        want = num / num[0]
        got = fitted.pp(vec, horizon=H)
        assert np.allclose(got, want, atol=1e-12), (got, want)
        assert got[0] == pytest.approx(1.0, abs=1e-14)
        # The profile is a ratio of quadratic forms in b, hence scale free.
        assert np.allclose(fitted.pp(2.5 * b, horizon=H), got, atol=1e-12)
        # ... and it is genuinely non-trivial, not a flat line at one.
        assert got[1:].min() < 0.9

    # A stationary system's profiles die out.
    assert fitted.pp({("A", "y"): 1.0}, horizon=48)[-1] < 0.05


def test_persistence_profile_converges_on_a_fixed_unit_root_system():
    """Pinned on hand-built coefficients, never on an estimated levels system.

    An estimated levels GVAR has a largest root of 1.00x rather than exactly 1,
    so ``R_h`` drifts and a tail-convergence assertion on it is fragile. Here
    ``F`` is exact: ``x1`` is a random walk fed by a stationary ``x2``, so the
    profile of ``e_1`` converges to ``1 + (0.3/0.5)^2 = 1.36`` and that of
    ``e_2`` decays to zero.
    """
    frames = _panel_from_array(stable_global_dgp(T=120, seed=71, k=2),
                               countries=("A", "B"), varnames=("y",))
    W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    res = gvar(frames, W, p=1, q=1, weak_exogeneity=False)
    F = np.array([[[1.0, 0.3], [0.0, 0.5]]])
    fixed = dataclasses.replace(
        res, F=F, H=F, G=np.eye(2), G_inv=np.eye(2), Sigma_eps=np.eye(2), s=1,
    )
    prof = fixed.pp(np.array([1.0, 0.0]), horizon=48)
    assert prof[0] == pytest.approx(1.0, abs=1e-14)
    assert prof[-1] == pytest.approx(1.0 + (0.3 / 0.5) ** 2, abs=1e-9)
    assert abs(prof[-1] - prof[-9]) < 1e-3
    decaying = fixed.pp(np.array([0.0, 1.0]), horizon=48)
    assert decaying[-1] < 1e-12


def test_pp_and_girf_reject_a_degenerate_shock(fitted):
    """A zero residual variance mirrors var.irf.gfevd's sigma_jj guard."""
    Sigma = fitted.Sigma_eps.copy()
    Sigma[0, :] = 0.0
    Sigma[:, 0] = 0.0
    broken = dataclasses.replace(fitted, Sigma_eps=Sigma)
    with pytest.raises(np.linalg.LinAlgError, match="not positive"):
        broken.girf("A", "y", horizon=4)
    with pytest.raises(np.linalg.LinAlgError, match="not positive"):
        fitted.girf_combination({}, horizon=4)
    with pytest.raises(np.linalg.LinAlgError, match=r"non-positive diagonal"):
        broken.gfevd(4)


def test_girf_unknown_name_raises(fitted):
    with pytest.raises(KeyError, match="unknown"):
        fitted.girf("A", "nope")
    with pytest.raises(KeyError, match="unknown"):
        fitted.girf("ZZ", "y")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_is_deterministic_in_the_seed(fitted):
    a = fitted.girf("A", "y", horizon=5, n_boot=50, seed=7)
    b = fitted.girf("A", "y", horizon=5, n_boot=50, seed=7)
    assert np.array_equal(a.lower, b.lower)
    assert np.array_equal(a.upper, b.upper)
    c = fitted.girf("A", "y", horizon=5, n_boot=50, seed=8)
    assert not np.allclose(a.lower, c.lower)
    plain = fitted.girf("A", "y", horizon=5)
    assert plain.lower is None and plain.upper is None and plain.median is None


def test_bootstrap_band_is_the_percentile_of_the_stored_draws(fitted):
    """The band must BE the percentiles it claims, and nest as ``ci`` widens.

    "the band contains the point estimate" is close to tautological -- the
    draws are simulated from the point estimate, so a percentile band around
    them is centred on it. These are the checks that can actually fail:
    the reported ``lower``/``upper`` are exactly the requested percentiles of
    ``boot_irf``, a wider ``ci`` gives a strictly nested wider band, and the
    band narrows when the sample gets longer.
    """
    out = fitted.girf("A", "y", horizon=8, n_boot=120, seed=3, ci=0.90,
                      store_draws=True)
    assert out.n_boot_ok + out.n_boot_failed == out.n_boot
    assert out.boot_irf.shape == (out.n_boot_ok, 9, fitted.n_variables)
    assert np.allclose(out.lower, np.percentile(out.boot_irf, 5.0, axis=0),
                       atol=1e-12)
    assert np.allclose(out.upper, np.percentile(out.boot_irf, 95.0, axis=0),
                       atol=1e-12)
    assert np.allclose(out.median, np.percentile(out.boot_irf, 50.0, axis=0),
                       atol=1e-12)
    assert np.all(out.lower <= out.upper)

    wide = fitted.girf("A", "y", horizon=8, n_boot=120, seed=3, ci=0.99)
    assert np.all(wide.lower <= out.lower + 1e-12)
    assert np.all(wide.upper >= out.upper - 1e-12)
    assert np.any(wide.upper > out.upper + 1e-12), "ci= must actually bite"


def test_bootstrap_band_narrows_as_the_sample_grows():
    """Sampling uncertainty is real: quadrupling T must shrink the band."""
    widths = {}
    for T in (120, 480):
        frames = _panel_from_array(stable_global_dgp(T=T, seed=311))
        res = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
        out = res.girf("A", "y", horizon=6, n_boot=120, seed=4)
        widths[T] = float(np.mean(out.upper - out.lower))
    assert widths[480] < 0.65 * widths[120], widths


def test_bootstrap_wild_runs_and_store_draws(fitted):
    out = fitted.girf("B", "y", horizon=4, n_boot=60, bootstrap="wild",
                      seed=11, store_draws=True)
    assert out.boot_irf.shape == (out.n_boot_ok, 5, fitted.n_variables)
    assert out.bootstrap == "wild"
    with pytest.raises(ValueError, match="store_draws"):
        fitted.girf("B", "y", horizon=4, store_draws=True)
    with pytest.raises(ValueError, match="bootstrap"):
        fitted.girf("B", "y", horizon=4, n_boot=50, bootstrap="block")
    with pytest.raises(ValueError, match="ci"):
        fitted.girf("B", "y", horizon=4, n_boot=50, ci=1.5)


def test_few_bootstrap_draws_warn(fitted):
    with pytest.warns(RuntimeWarning, match="unreliable"):
        fitted.girf("A", "y", horizon=2, n_boot=5, seed=1)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def test_forecast_matches_the_hand_rolled_recursion(fitted):
    res = fitted
    steps = 5
    got = res.forecast(steps).to_numpy(dtype=float)
    X = res.panel.to_numpy(dtype=float)
    T = len(X)
    hist = list(X)
    want = []
    for h in range(1, steps + 1):
        val = res.intercept + res.trend_coef * (T + h)
        for l in range(res.s):
            val = val + res.F[l] @ hist[T + h - l - 2]
        hist.append(val)
        want.append(val)
    assert np.allclose(got, np.array(want), atol=1e-12)
    with pytest.raises(ValueError, match="steps"):
        res.forecast(0)


# ---------------------------------------------------------------------------
# Lag selection
# ---------------------------------------------------------------------------
def test_lag_selection_recovers_a_planted_order_within_one_grid():
    """BIC picks (1, 0) on a p=1 DGP whose countries do not interact."""
    frames = _panel_from_array(independent_country_dgp(T=300, seed=101))
    res = gvar(frames, _trade_weights(), ic="bic", max_p=2, max_q=1,
               weak_exogeneity=False)
    assert res.ic == "bic"
    for c in COUNTRIES:
        assert res.lag_orders[c] == (1, 0)
    # Within one grid every candidate shares the selection sample, so the
    # criteria are comparable. Across grids they are NOT: raising max_p moves
    # the grid sample start, which is why no cross-max_p invariance is claimed.
    # What IS asserted is recovery on a WIDER grid -- `1 <= p <= max_p` would
    # be the grid bound restated and could not fail.
    res4 = gvar(frames, _trade_weights(), ic="bic", max_p=4, max_q=1,
                weak_exogeneity=False)
    for c in COUNTRIES:
        assert res4.lag_orders[c] == (1, 0), res4.lag_orders
    # AIC on the same DGP is looser but must not run away either.
    res_aic = gvar(frames, _trade_weights(), ic="aic", max_p=4, max_q=2,
                   weak_exogeneity=False)
    for c in COUNTRIES:
        assert res_aic.lag_orders[c] == (1, 0), res_aic.lag_orders


def test_lag_selection_skips_candidates_with_a_singular_sigma():
    """T_g - m_i >= k_i + 1, so a rank-deficient Sigma cannot win the grid.

    Without the guard a candidate whose ``U'U/T_g`` is singular returns a
    garbage ``slogdet`` and a spuriously huge log-likelihood, so the *largest*
    order always wins.
    """
    X = independent_country_dgp(T=20, seed=5, n_countries=2, kv=3)
    dates = pd.RangeIndex(20, name="date")
    frames = {
        c: pd.DataFrame(X[:, 3 * i:3 * i + 3], columns=["y", "r", "pi"], index=dates)
        for i, c in enumerate(("A", "B"))
    }
    W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = gvar(frames, W, ic="aic", max_p=4, max_q=0, weak_exogeneity=False)
    # T_g = 16; m_i = 1 + 3 (contemporaneous star) + 3p, and k_i = 3, so only
    # p <= 2 leaves the required k_i + 1 = 4 residual degrees of freedom.
    for c in ("A", "B"):
        assert res.lag_orders[c][0] <= 2


def test_infeasible_lag_grid_raises():
    X = independent_country_dgp(T=12, seed=5, n_countries=2, kv=3)
    dates = pd.RangeIndex(12, name="date")
    frames = {
        c: pd.DataFrame(X[:, 3 * i:3 * i + 3], columns=["y", "r", "pi"], index=dates)
        for i, c in enumerate(("A", "B"))
    }
    W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="infeasible") as exc:
        gvar(frames, W, ic="bic", max_p=3, max_q=1)
    text = str(exc.value)
    # The remedial advice must survive whichever branch of the message fires;
    # it used to be dropped in the branch that names an order.
    assert "Lower max_p / max_q" in text, text
    assert "'A'" in text and "T = 12" in text and "k_i = 3" in text
    # Here no candidate clears the degrees-of-freedom guard at all, so no
    # order is named; the module must not invent one.
    assert "largest order" not in text, text


def test_ic_and_explicit_orders_conflict():
    frames = _panel_from_array(stable_global_dgp(T=120, seed=7))
    with pytest.raises(ValueError, match="not both"):
        gvar(frames, _trade_weights(), p=1, ic="bic")


# ---------------------------------------------------------------------------
# Weak exogeneity
# ---------------------------------------------------------------------------
def test_weak_exogeneity_degrees_of_freedom_and_p_value_identity(fitted):
    from scipy import stats

    we = fitted.weak_exogeneity
    assert we is not None
    tab = we.table
    assert list(tab.columns) == [
        "country", "star_variable", "F", "df_num", "df_den", "p_value",
        "p_holm", "reject_nominal", "n_obs", "lags",
    ]
    for _, row in tab.iterrows():
        c = row["country"]
        k_i = len(fitted.country_models[c].variables)
        k_star = len(fitted.country_models[c].star_variables)
        r = we.lags[c]
        T_e = fitted.n_obs - 1 - r          # first differences cost one row
        m = 1 + r * k_star + r * k_i + k_i
        assert int(row["df_num"]) == k_i
        assert int(row["df_den"]) == T_e - m
        assert int(row["n_obs"]) == T_e
        assert row["p_value"] == pytest.approx(
            float(stats.f.sf(row["F"], row["df_num"], row["df_den"])), abs=1e-12
        )
    assert we.n_tests == len(tab)
    assert we.n_reject_nominal == int(tab["reject_nominal"].sum())
    # The calibrated-count headline is gone on purpose: the test's measured
    # size is roughly twice nominal on I(1) levels, so `alpha * n_tests` was a
    # number nobody should have been comparing against.
    assert not hasattr(we, "expected_rejections")
    # Holm is monotone in the raw p-value and never smaller than it.
    assert np.all(tab["p_holm"].to_numpy() >= tab["p_value"].to_numpy() - 1e-15)


def test_weak_exogeneity_has_power_against_a_leaking_country():
    """No size claim is made -- only that the test is not blind."""
    frames = _panel_from_array(leaking_country_dgp(T=250, seed=5))
    res = gvar(frames, _trade_weights(), p=1, q=1)
    tab = res.weak_exogeneity.table
    a_rows = tab[tab["country"] == "A"]
    assert len(a_rows) == 2
    assert bool(a_rows["reject_nominal"].all()), a_rows.to_dict("records")
    assert float(a_rows["p_value"].max()) < 0.01


def _we_null_rejection_rate(rho, T, n_rep, base_seed):
    """Rejection rate of the weak-exogeneity F test when the null is TRUE."""
    W = _trade_weights()
    p_values = []
    with warnings.catch_warnings():
        # rho = 1 puts a unit root in the solved companion, which is exactly
        # the canonical GVAR case and warns by design.
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_rep):
            frames = _panel_from_array(
                independent_country_dgp(T=T, seed=base_seed + i, rho=rho)
            )
            res = gvar(frames, W, p=1, q=1)
            p_values.append(res.weak_exogeneity.table["p_value"].to_numpy())
    return np.concatenate(p_values)


def test_weak_exogeneity_is_over_sized_on_i1_levels_as_documented():
    """The size distortion the module discloses must be the one it has.

    The countries never interact, so weak exogeneity holds by construction and
    every rejection is a false one. The old version of this test asserted
    ``n_reject <= 18`` out of 60 -- roughly 45 standard errors above the truth,
    a bound that could not fail -- and never exercised the I(1) case where the
    test actually misbehaves. This one pins BOTH failure directions, at the
    magnitudes named in the module docstring, so the disclosure cannot silently
    drift away from the code.
    """
    # (a) I(1) levels, the canonical GVAR input: roughly twice nominal.
    p_lvl = _we_null_rejection_rate(rho=1.0, T=200, n_rep=250, base_seed=500)
    assert p_lvl.size == 1500
    rate_lvl = float(np.mean(p_lvl < 0.05))
    assert 0.07 <= rate_lvl <= 0.20, (
        f"nominal 5% test rejected {rate_lvl:.3f} of the time on random walks; "
        "the module docstring documents ~0.11 -- update BOTH or neither"
    )
    assert float(np.mean(p_lvl < 0.10)) >= 0.13

    # (b) stationary data: badly conservative in the other direction.
    p_stat = _we_null_rejection_rate(rho=0.5, T=200, n_rep=250, base_seed=900)
    rate_stat = float(np.mean(p_stat < 0.05))
    assert rate_stat <= 0.02, (
        f"nominal 5% test rejected {rate_stat:.3f} of the time on stationary "
        "data; the module docstring documents ~0.002"
    )
    assert rate_stat < rate_lvl


def test_weak_exogeneity_reports_no_calibrated_rejection_count():
    """G6/G3: the summary must not read as a calibrated decision rule."""
    frames = _panel_from_array(independent_country_dgp(T=200, seed=205))
    we = gvar(frames, _trade_weights(), p=1, q=1).weak_exogeneity
    assert "expected_rejections" not in {f.name for f in dataclasses.fields(we)}
    assert "reject" not in we.table.columns
    assert "reject_nominal" in we.table.columns
    text = we.summary()
    assert "MEASURED SIZE" in text
    assert "expected under the null" not in text
    assert "NOMINAL" in text


def test_weak_exogeneity_skips_countries_without_a_star_block():
    frames = _panel_from_array(stable_global_dgp(T=150, seed=77))
    res = gvar(frames, _trade_weights(), p=1, q={"A": 1, "B": 1, "C": 0},
               star_vars={"C": ()})
    assert "C" in res.weak_exogeneity.skipped
    assert set(res.weak_exogeneity.table["country"]) == {"A", "B"}


def test_weak_exogeneity_levels_transform_runs():
    frames = _panel_from_array(stable_global_dgp(T=150, seed=79))
    res = gvar(frames, _trade_weights(), p=1, q=1, we_transform="levels",
               we_lags=2)
    assert res.weak_exogeneity.transform == "levels"
    assert set(res.weak_exogeneity.lags.values()) == {2}
    with pytest.raises(ValueError, match="we_transform"):
        gvar(frames, _trade_weights(), p=1, q=1, we_transform="log")


# ---------------------------------------------------------------------------
# Conditioning, stability, units
# ---------------------------------------------------------------------------
def test_singular_G_raises_naming_the_country_blocks():
    """G = [[1, -l1], [-l2, 1]] is singular exactly when l1*l2 = 1."""
    S = {"A": np.array([[1.0, 0.0]]), "B": np.array([[0.0, 1.0]])}
    Wst = {"A": np.array([[0.0, 1.0]]), "B": np.array([[1.0, 0.0]])}
    link = {c: np.vstack([S[c], Wst[c]]) for c in ("A", "B")}
    A0 = {"A": np.array([[1.0, -2.0]]), "B": np.array([[1.0, -0.5]])}
    Al = {"A": [np.array([[0.5, 0.0]])], "B": [np.array([[0.5, 0.0]])]}
    with pytest.raises(np.linalg.LinAlgError) as exc:
        solve_gvar(link, A0, Al, country_order=["A", "B"],
                   block_sizes={"A": 1, "B": 1})
    msg = str(exc.value)
    assert "A" in msg and "B" in msg
    assert "singular" in msg or "ill-conditioned" in msg


def test_condition_numbers_are_the_quantities_they_claim_to_be(fitted):
    """`isfinite and >= 1` holds for every condition number ever computed.

    These assertions instead pin what the two fields *are*: the raw one is
    exactly ``cond(G)``, the equilibrated one is strictly smaller here (that is
    the whole point of equilibrating), and both collapse to exactly 1 when
    ``G`` is the identity.
    """
    assert fitted.condition_number_raw == pytest.approx(
        float(np.linalg.cond(fitted.G)), rel=1e-10
    )
    assert 1.0 <= fitted.condition_number <= fitted.condition_number_raw + 1e-9

    frames = _panel_from_array(stable_global_dgp(seed=317))
    plain = gvar(frames, _trade_weights(), p=1, q=0,
                 star_vars={c: () for c in COUNTRIES}, weak_exogeneity=False)
    assert np.array_equal(plain.G, np.eye(plain.n_variables))
    assert plain.condition_number == pytest.approx(1.0, abs=1e-12)
    assert plain.condition_number_raw == pytest.approx(1.0, abs=1e-12)


def test_cond_of_G_is_invariant_to_the_units_of_the_data():
    """G -> D G D^-1 under a rescaling, so only the equilibrated cond is meaningful."""
    X = stable_global_dgp(seed=83)
    frames = _panel_from_array(X)
    base = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    scaled_frames = {c: df.assign(r=df["r"] * 1e-6) for c, df in frames.items()}
    scaled = gvar(scaled_frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    # The raw number is meaningless: it moves by eleven orders of magnitude.
    assert scaled.condition_number_raw > 1e8 * base.condition_number_raw
    # The equilibrated one is stable to within a small factor, and the model
    # is accepted rather than rejected by the conditioning gate.
    assert scaled.condition_number < 20.0 * base.condition_number
    assert base.condition_number < 20.0 * scaled.condition_number
    # The eigenvalues of the solved system are a similarity invariant too.
    assert scaled.max_eigenvalue == pytest.approx(base.max_eigenvalue, rel=1e-8)


def test_stability_is_a_flag_not_an_error():
    """Unit roots are expected in levels; the result must still be complete."""
    rng = np.random.default_rng(97)
    T, k = 200, 6
    X = np.cumsum(rng.standard_normal((T, k)), axis=0)  # random walks
    frames = _panel_from_array(X)
    res = gvar(frames, _trade_weights(), p=2, q=1, weak_exogeneity=False)
    # An I(1) system has a root at (or just under) unity; the flag reports it
    # rather than raising, and the result is fully populated either way.
    assert res.max_eigenvalue > 0.95
    assert res.stable is bool(res.max_eigenvalue < 1.0)
    assert res.G.shape == (6, 6)
    assert np.all(np.isfinite(res.F))
    assert np.all(np.isfinite(res.girf("A", "y", horizon=4).irf))
    assert np.all(np.isfinite(res.pp({("A", "y"): 1.0}, horizon=12)))


def test_explosive_system_warns_but_still_returns():
    rng = np.random.default_rng(101)
    T, k = 200, 6
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = 1.03 * X[t - 1] + rng.standard_normal(k)
    frames = _panel_from_array(X)
    with pytest.warns(RuntimeWarning, match="max .eigenvalue"):
        res = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    assert res.max_eigenvalue > 1.0 + 1e-6
    assert res.stable is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        quiet = gvar(frames, _trade_weights(), p=1, q=1, stability_warn=False,
                     weak_exogeneity=False)
    assert quiet.stable is False


# ---------------------------------------------------------------------------
# Exogenous block
# ---------------------------------------------------------------------------
def test_exog_shifts_the_effective_sample_and_enters_the_identity():
    X = stable_global_dgp(seed=103)
    frames = _panel_from_array(X)
    rng = np.random.default_rng(3)
    dates = pd.RangeIndex(len(X), name="date")
    d = pd.DataFrame({"oil": rng.standard_normal(len(X))}, index=dates)
    res = gvar(frames, _trade_weights(), p=2, q=1, exog=d, exog_lags=4,
               weak_exogeneity=False)
    assert res.n_obs == len(X) - 4          # t0 = max(s, q_d) = 4
    assert res.exog_names == ("oil",)
    assert res.Upsilon.shape == (5, 6, 1)
    Xg = res.panel.to_numpy(dtype=float)
    E = res.resid.to_numpy(dtype=float)
    D = d.to_numpy(dtype=float)
    err = 0.0
    for i, t in enumerate(range(res.spec.t0, len(Xg))):
        rhs = res.intercept + res.G_inv @ E[i]
        for l in range(res.s):
            rhs = rhs + res.F[l] @ Xg[t - l - 1]
        for j in range(res.Upsilon.shape[0]):
            rhs = rhs + res.Upsilon[j] @ D[t - j]
        err = max(err, float(np.max(np.abs(Xg[t] - rhs))))
    assert err < 1e-10


def test_default_exog_lags_is_max_q_under_ic_not_the_selected_q():
    """The documented default differs between the two lag-order modes.

    With explicit ``p``/``q`` the default ``q_d`` is ``max_i q_i``. With
    ``ic=`` it is the grid BOUND ``max_q``, because ``q_d`` fixes the common
    selection sample before any ``q_i`` is known. The docstring says so; this
    pins that it is true, by choosing a DGP whose selected ``q_i`` are all 0
    while ``max_q`` is 3 -- if ``q_d`` followed the selected orders the
    effective sample would be three observations longer.
    """
    T = 200
    X = independent_country_dgp(T=T, seed=211)
    frames = _panel_from_array(X)
    dates = pd.RangeIndex(T, name="date")
    d = pd.DataFrame({"oil": np.random.default_rng(9).standard_normal(T)},
                     index=dates)

    res = gvar(frames, _trade_weights(), ic="bic", max_p=2, max_q=3, exog=d,
               weak_exogeneity=False)
    assert all(res.lag_orders[c][1] == 0 for c in COUNTRIES), res.lag_orders
    s = res.s
    assert res.n_obs == T - max(s, 3), (res.n_obs, s)
    assert res.Upsilon.shape[0] == 4          # q_d + 1 with q_d = max_q = 3

    # The explicit-order path really does use max_i q_i instead.
    fixed = gvar(frames, _trade_weights(), p=2, q=1, exog=d,
                 weak_exogeneity=False)
    assert fixed.Upsilon.shape[0] == 2        # q_d + 1 with q_d = max_i q_i = 1


def test_forecast_requires_exog_future_when_exog_was_used():
    frames = _panel_from_array(stable_global_dgp(T=150, seed=107))
    dates = pd.RangeIndex(150, name="date")
    rng = np.random.default_rng(4)
    d = pd.DataFrame({"oil": rng.standard_normal(150)}, index=dates)
    res = gvar(frames, _trade_weights(), p=1, q=1, exog=d, weak_exogeneity=False)
    with pytest.raises(ValueError, match="exog_future"):
        res.forecast(3)
    future = pd.DataFrame({"oil": [0.1, -0.2, 0.3]}, index=[150, 151, 152])
    out = res.forecast(3, exog_future=future)
    assert out.shape == (3, 6)
    # Hand-rolled recursion including the exogenous term.
    Xg = res.panel.to_numpy(dtype=float)
    D = d.to_numpy(dtype=float)
    hist = list(Xg)
    dpath = list(D.ravel()) + [0.1, -0.2, 0.3]
    T = len(Xg)
    want = []
    for h in range(1, 4):
        val = res.intercept + res.trend_coef * (T + h)
        for l in range(res.s):
            val = val + res.F[l] @ hist[T + h - l - 2]
        for j in range(res.Upsilon.shape[0]):
            val = val + res.Upsilon[j] @ np.array([dpath[T + h - j - 1]])
        hist.append(val)
        want.append(val)
    assert np.allclose(out.to_numpy(dtype=float), np.array(want), atol=1e-10)
    plain = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    with pytest.raises(ValueError, match="without an exogenous block"):
        plain.forecast(2, exog_future=future)


def test_exog_name_colliding_with_an_endogenous_variable_raises():
    frames = _panel_from_array(stable_global_dgp(T=120, seed=109))
    dates = pd.RangeIndex(120, name="date")
    d = pd.DataFrame({"y": np.zeros(120)}, index=dates)
    with pytest.raises(ValueError, match="also appear in"):
        gvar(frames, _trade_weights(), p=1, q=1, exog=d)


def test_exog_index_must_match_the_data_dates():
    frames = _panel_from_array(stable_global_dgp(T=120, seed=111))
    d = pd.DataFrame({"oil": np.zeros(100)}, index=pd.RangeIndex(100, name="date"))
    with pytest.raises(ValueError, match="indexed by the data's dates"):
        gvar(frames, _trade_weights(), p=1, q=1, exog=d)


# ---------------------------------------------------------------------------
# Data contract and degenerate cases
# ---------------------------------------------------------------------------
def test_dict_alias_matches_the_multiindex_form():
    X = stable_global_dgp(T=150, seed=113)
    frames = _panel_from_array(X)
    long = pd.concat(frames.values(), keys=list(frames),
                     names=["country", "date"])
    a = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    b = gvar(long, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    for attr in ("G", "H", "F", "G_inv", "Sigma_eps", "intercept", "trend_coef"):
        assert np.allclose(getattr(a, attr), getattr(b, attr), atol=1e-15)
    assert a.names == b.names


def test_data_must_be_a_frame_or_mapping():
    with pytest.raises(TypeError, match="gvar"):
        gvar(np.zeros((10, 3)), _trade_weights())


def test_index_contract_errors():
    X = stable_global_dgp(T=60, seed=117)
    frames = _panel_from_array(X)
    long = pd.concat(frames.values(), keys=list(frames), names=["country", "date"])
    flat = long.reset_index(drop=True)
    with pytest.raises(ValueError, match="2-level MultiIndex"):
        gvar(flat, _trade_weights())
    renamed = long.rename_axis(index=["ctry", "date"])
    with pytest.raises(ValueError, match="level names"):
        gvar(renamed, _trade_weights())
    dup = pd.concat([long, long.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated"):
        gvar(dup.sort_index(), _trade_weights())


def test_unbalanced_dates_raise():
    frames = _panel_from_array(stable_global_dgp(T=60, seed=119))
    frames["B"] = frames["B"].drop(index=17)
    with pytest.raises(ValueError, match=r"differs across countries"):
        gvar(frames, _trade_weights(), p=1, q=1)


def test_a_missing_variable_raises_by_default_and_is_opt_in_only():
    """BUILD_RULES section 2: the default is a COMMON variable set.

    An entirely NaN ``(country, variable)`` is far more often a misnamed or
    accidentally dropped series than a deliberate asymmetry, and silently
    estimating a smaller global model is the worst possible response. The
    permissive reading is available, but only when asked for, and it must say
    out loud what it dropped.
    """
    X = stable_global_dgp(T=120, seed=127)
    frames = _panel_from_array(X)
    frames["C"] = frames["C"].assign(r=np.nan)

    with pytest.raises(ValueError, match=r"country 'C' does not carry variable 'r'"):
        gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False)
    with pytest.raises(ValueError, match=r"country 'C' does not carry variable 'r'"):
        star_variables(frames, _trade_weights())

    with pytest.warns(RuntimeWarning, match=r"\('C', 'r'\)"):
        res = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False,
                   allow_missing_variables=True)
    assert res.n_variables == 5
    assert ("C", "r") not in res.name_pairs
    # The full common set is what the default would have used.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = gvar(_panel_from_array(X), _trade_weights(), p=1, q=1,
                    weak_exogeneity=False)
    assert full.n_variables == 6

    bad = _panel_from_array(X)
    bad["B"] = bad["B"].copy()
    bad["B"].loc[40, "y"] = np.nan
    with pytest.raises(ValueError, match=r"interior missing value.*'B'.*'y'"):
        gvar(bad, _trade_weights(), p=1, q=1)
    # An interior NaN stays an error even under the opt-in.
    with pytest.raises(ValueError, match=r"interior missing value"):
        gvar(bad, _trade_weights(), p=1, q=1, allow_missing_variables=True)


def test_single_period_panel_raises():
    dates = pd.RangeIndex(1, name="date")
    frames = {c: pd.DataFrame([[0.1, 0.2]], columns=["y", "r"], index=dates)
              for c in COUNTRIES}
    with pytest.raises(ValueError, match="date"):
        gvar(frames, _trade_weights(), p=1, q=1)


def test_non_numeric_column_raises():
    frames = _panel_from_array(stable_global_dgp(T=60, seed=131))
    frames = {c: df.assign(label="x") for c, df in frames.items()}
    with pytest.raises(ValueError, match="non-numeric"):
        gvar(frames, _trade_weights(), p=1, q=1)


def test_weights_contract_errors():
    frames = _panel_from_array(stable_global_dgp(T=120, seed=137))
    with pytest.raises(ValueError, match="non-zero diagonal"):
        gvar(frames, _trade_weights(values=[[0.1, 0.5, 0.4],
                                            [0.5, 0.0, 0.5],
                                            [0.3, 0.7, 0.0]]), p=1, q=1)
    with pytest.raises(ValueError, match="sum to 1"):
        gvar(frames, _trade_weights(values=[[0.0, 0.5, 0.4],
                                            [0.5, 0.0, 0.5],
                                            [0.3, 0.7, 0.0]]), p=1, q=1)
    with pytest.raises(ValueError, match="non-negative"):
        gvar(frames, _trade_weights(values=[[0.0, 1.2, -0.2],
                                            [0.5, 0.0, 0.5],
                                            [0.3, 0.7, 0.0]]), p=1, q=1)
    with pytest.raises(KeyError, match="Missing rows"):
        gvar(frames, _trade_weights(countries=("A", "B", "D")), p=1, q=1)
    with pytest.raises(TypeError, match="time-varying"):
        gvar(frames, np.zeros((2, 3, 3)), p=1, q=1)


def test_island_country_raises_and_star_vars_is_the_fix():
    """An all-zero trade row is an island: every star regressor would be zero."""
    frames = _panel_from_array(stable_global_dgp(T=120, seed=139))
    W = _trade_weights(values=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.3, 0.7, 0.0]])
    with pytest.raises(ValueError, match=r"country 'A'.*all-zero row"):
        gvar(frames, W, p=1, q=1)
    # An empty star block makes q meaningless for that country, so an explicit
    # q > 0 is an error rather than a silent rewrite to 0 (G3).
    with pytest.raises(ValueError, match=r"q=1 was requested for country 'A'.*"
                                         r"empty star block"):
        gvar(frames, W, p=1, q=1, star_vars={"A": ()}, weak_exogeneity=False)
    res = gvar(frames, W, p=1, q={"A": 0, "B": 1, "C": 1},
               star_vars={"A": ()}, weak_exogeneity=False)
    assert res.country_models["A"].star_variables == ()
    assert res.country_models["A"].q == 0
    # ... and with q left to its default the empty block still resolves to 0.
    res_default = gvar(frames, W, p=1, star_vars={"A": ()},
                       weak_exogeneity=False)
    assert res_default.lag_orders["A"] == (1, 0)


def test_disconnected_trade_graph_gives_a_block_diagonal_link():
    """Two trade blocs that never touch: G and H stay block diagonal."""
    X = stable_global_dgp(T=200, seed=149, k=8)
    frames = _panel_from_array(X, countries=("A", "B", "C", "D"))
    W = pd.DataFrame(
        [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0]],
        index=list("ABCD"), columns=list("ABCD"),
    )
    res = gvar(frames, W, p=1, q=1, weak_exogeneity=False)
    idx = res.spec.own_idx
    for a, b in (("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")):
        assert np.count_nonzero(res.G[np.ix_(idx[a], idx[b])]) == 0
        assert np.count_nonzero(res.H[0][np.ix_(idx[a], idx[b])]) == 0


def test_a_single_partner_taking_all_the_weight_is_admissible():
    frames = _panel_from_array(stable_global_dgp(T=200, seed=151))
    W = _trade_weights(values=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    res = gvar(frames, W, p=1, q=1, weak_exogeneity=False)
    assert np.allclose(res.star_weights["A"].loc["y"].to_numpy(), [0.0, 1.0, 0.0])
    assert np.allclose(res.star_data.loc["A"]["y_star"].to_numpy(dtype=float),
                       frames["B"]["y"].to_numpy(dtype=float), atol=1e-12)


def test_single_variable_country_alongside_two_variable_countries():
    X = stable_global_dgp(T=200, seed=157)
    frames = _panel_from_array(X)
    frames["C"] = frames["C"].assign(r=np.nan)
    with pytest.warns(RuntimeWarning):
        res = gvar(frames, _trade_weights(), p=1, q=1, weak_exogeneity=False,
                   allow_missing_variables=True)
    assert res.country_models["C"].variables == ("y",)
    assert res.n_variables == 5
    assert res.girf("C", "y", horizon=4).irf.shape == (5, 5)


def test_constant_star_column_raises_before_ols():
    """A constant regressor is caught by name, not by a downstream LinAlgError."""
    X = stable_global_dgp(T=150, seed=163)
    frames = _panel_from_array(X)
    frames["B"] = frames["B"].assign(y=1.0)
    frames["C"] = frames["C"].assign(y=1.0)
    with pytest.raises(ValueError, match=r"constant star regressor"):
        gvar(frames, _trade_weights(), p=1, q=1, star_vars={"A": ("y",),
                                                            "B": ("r",),
                                                            "C": ("r",)})


def test_insufficient_observations_raises_and_the_boundary_case_works():
    X = independent_country_dgp(T=14, seed=5, n_countries=2, kv=2)
    dates = pd.RangeIndex(14, name="date")
    frames = {c: pd.DataFrame(X[:, 2 * i:2 * i + 2], columns=["y", "r"], index=dates)
              for i, c in enumerate(("A", "B"))}
    W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match=r"residual degrees of freedom"):
        gvar(frames, W, p=4, q=1, weak_exogeneity=False)
    # k_i = 2, so k_i + 1 = 3 residual dof are the documented minimum.
    with pytest.warns(RuntimeWarning, match="residual degrees of freedom"):
        res = gvar(frames, W, p=1, q=1, weak_exogeneity=False)
    for c in ("A", "B"):
        m = res.country_models[c]
        assert m.dof >= len(m.variables) + 1
    assert np.all(np.isfinite(res.F))


def test_p_and_q_validation():
    frames = _panel_from_array(stable_global_dgp(T=120, seed=167))
    with pytest.raises(ValueError, match="p must be >= 1"):
        gvar(frames, _trade_weights(), p=0)
    with pytest.raises(ValueError, match="q must be >= 0"):
        gvar(frames, _trade_weights(), p=1, q=-1)
    with pytest.raises(ValueError, match="trend"):
        gvar(frames, _trade_weights(), p=1, trend="quadratic")
    with pytest.raises(KeyError, match="unknown country"):
        gvar(frames, _trade_weights(), p={"A": 1, "B": 1, "C": 1, "Z": 1})
    with pytest.raises(KeyError, match="no entry"):
        gvar(frames, _trade_weights(), p={"A": 1})
    with pytest.raises(KeyError, match="star block"):
        gvar(frames, _trade_weights(), p=1, q=1,
             contemporaneous_star={"A": ("nope",)})


def test_contemporaneous_star_restriction_zeroes_the_right_block():
    """The canonical US asymmetry: no contemporaneous foreign y."""
    frames = _panel_from_array(stable_global_dgp(T=200, seed=173))
    res = gvar(frames, _trade_weights(), p=1, q=1,
               contemporaneous_star={"A": ("r",)}, weak_exogeneity=False)
    m = res.country_models["A"]
    assert m.contemporaneous_star == ("r",)
    j = m.star_variables.index("y")
    assert np.all(m.Lambda[0][:, j] == 0.0)
    assert "y_star" not in m.regressor_names
    assert "r_star" in m.regressor_names


# ---------------------------------------------------------------------------
# Contract: no kwargs sinks, frozen results, presentation
# ---------------------------------------------------------------------------
def test_no_kwargs_sink(fitted):
    frames = _panel_from_array(stable_global_dgp(T=60, seed=179))
    with pytest.raises(TypeError):
        gvar(frames, _trade_weights(), lagz=2)
    with pytest.raises(TypeError):
        fitted.girf("A", "y", horizonz=4)
    with pytest.raises(TypeError):
        fitted.gfevd(4, normalise=True)
    with pytest.raises(TypeError):
        solve_gvar({}, {}, {}, country_order=[], block_sizes={}, foo=1)
    with pytest.raises(TypeError):
        star_variables(frames, _trade_weights(), starvars={})


def test_arguments_meaningless_for_the_chosen_mode_raise(fitted):
    """A keyword that cannot bite is an error, not a silent no-op."""
    frames = _panel_from_array(stable_global_dgp(T=120, seed=181))
    W = _trade_weights()
    with pytest.raises(ValueError, match="max_p / max_q"):
        gvar(frames, W, p=1, q=1, max_p=4)
    with pytest.raises(ValueError, match="exog_lags"):
        gvar(frames, W, p=1, q=1, exog_lags=2)
    for kwargs in ({"we_lags": 3}, {"we_transform": "levels"}, {"alpha": 0.20}):
        with pytest.raises(ValueError,
                           match="we_lags / we_transform / alpha"):
            gvar(frames, W, p=1, q=1, weak_exogeneity=False, **kwargs)
    # ... and each of those is accepted when the test IS switched on.
    assert gvar(frames, W, p=1, q=1, alpha=0.20).weak_exogeneity.alpha == 0.20

    for kwargs in ({"ci": 0.68}, {"bootstrap": "wild"}, {"store_draws": True},
                   {"seed": 99}):
        with pytest.raises(ValueError, match="n_boot=0"):
            fitted.girf("A", "y", horizon=2, **kwargs)
    # `seed` is not exempt just because it happens to have a scalar default:
    # all four bootstrap-only keywords behave identically.
    assert fitted.girf("A", "y", horizon=2, n_boot=60, seed=99).seed == 99


def test_result_objects_are_frozen(fitted):
    girf = fitted.girf("A", "y", horizon=2)
    for obj in (fitted, fitted.country_models["A"], girf, fitted.weak_exogeneity):
        assert dataclasses.is_dataclass(obj)
        assert obj.__dataclass_params__.frozen is True
        assert obj.__dataclass_params__.eq is False
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, dataclasses.fields(obj)[0].name, None)


def test_presentation_contract(fitted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    girf = fitted.girf("A", "y", horizon=6, n_boot=50, seed=2)
    for obj in (fitted, fitted.country_models["A"], girf, fitted.weak_exogeneity):
        frame = obj.to_frame()
        assert isinstance(frame, pd.DataFrame) and len(frame) > 0
        text = obj.summary()
        assert isinstance(text, str) and text.strip()
        for meth in ("to_markdown", "to_latex", "to_typst"):
            rendered = getattr(obj, meth)()
            assert isinstance(rendered, str) and rendered.strip()
            # Every renderer must actually carry the frame's columns, not just
            # return "some non-empty string".
            for col in frame.columns:
                # LaTeX and Typst both escape '_', so compare unescaped.
                assert str(col) in rendered.replace("\\", ""), (meth, col)
        fig = obj.plot()
        assert fig.__class__.__name__ == "Figure"
        plt.close(fig)
    # Underscores in regressor names such as 'y_star' must be escaped.
    assert r"\_" in fitted.to_frame().pipe(lambda d: fitted.to_latex())

    # The summaries must name the things a reader needs, not merely be strings.
    top = fitted.summary()
    for token in ("GVAR", "weak exogeneity", "NOMINAL"):
        assert token in top, token
    for c in COUNTRIES:
        assert c in top
    assert f"{fitted.n_obs}" in top
    assert "y_star" in fitted.country_models["A"].summary()
    assert "A:y" in girf.summary()
    assert list(fitted.to_frame().columns) == [
        "country", "equation", "regressor", "block", "lag", "coef", "se", "t",
        "p_value", "dof",
    ]
    assert list(girf.to_frame().columns) == [
        "h", "country", "variable", "name", "response", "lower", "upper", "median",
    ]
    fig = girf.plot(by_variable=True)
    plt.close(fig)
    fig = girf.plot(select=[("A", "y"), "B:r"])
    plt.close(fig)
    with pytest.raises(KeyError):
        girf.plot(select=["nope"])
    sub = fitted.coef_frame("A")
    assert set(sub["country"]) == {"A"}
    with pytest.raises(KeyError, match="unknown country"):
        fitted.coef_frame("Z")


def test_disclaimer_text_is_carried_as_data(fitted):
    we = fitted.weak_exogeneity
    assert "NOT" in we.test_name and "Dees" in we.test_name
    assert "NOT" in we.summary() and "Dees" in we.summary()
    assert isinstance(we, WeakExogeneityResult)
    girf = fitted.girf("A", "y", horizon=2)
    assert "generalised, not orthogonalised" in girf.summary()
    assert isinstance(girf, GVARGIRFResult)
    assert isinstance(fitted, GVARResult)
    assert isinstance(fitted.country_models["A"], CountryVARX)
    assert "Sims" in fitted.summary()


def test_public_surface_advertises_only_what_it_delivers(fitted):
    """G2: everything in ``__all__`` carries the six presentation methods.

    ``GVARSolution`` is a bare NamedTuple of solved matrices with none of them,
    so it must not be exported; it stays reachable as ``solve_gvar``'s return
    value. And ``GVARResult.spec`` -- an internal design bundle explicitly
    documented as private -- must not be a dataclass field, or the package's
    public-API snapshot would advertise it as part of the result contract.
    """
    import importlib

    # `from .gvar import gvar` in puremacro/var/__init__.py rebinds the
    # package attribute to the FUNCTION (the same house style as
    # did.callaway_santanna), so reach the module through importlib.
    g = importlib.import_module("puremacro.var.gvar")

    assert "GVARSolution" not in g.__all__
    assert isinstance(g.GVARSolution, type)          # still importable
    for name in g.__all__:
        obj = getattr(g, name)
        if not (dataclasses.is_dataclass(obj) and isinstance(obj, type)):
            continue
        for meth in ("summary", "to_frame", "to_markdown", "to_latex",
                     "to_typst", "plot"):
            assert callable(getattr(obj, meth, None)), (name, meth)

    field_names = {f.name for f in dataclasses.fields(GVARResult)}
    assert "spec" not in field_names
    assert not any(n.startswith("_") for n in field_names)
    # ... but it is still reachable for debugging, and still the live object.
    assert fitted.spec is not None
    assert fitted.spec.t0 == fitted.s
    with pytest.raises(AttributeError):
        fitted.spec = None


def test_pyodide_purity():
    """Importing and running gvar must not pull matplotlib into sys.modules."""
    code = (
        "import sys, numpy as np, pandas as pd\n"
        "import importlib\n"
        "g = importlib.import_module('puremacro.var.gvar')\n"
        "rng = np.random.default_rng(0)\n"
        "d = pd.RangeIndex(80, name='date')\n"
        "f = {c: pd.DataFrame(rng.standard_normal((80, 2)), columns=['y', 'r'],"
        " index=d) for c in ('A', 'B')}\n"
        "W = pd.DataFrame([[0., 1.], [1., 0.]], index=['A', 'B'],"
        " columns=['A', 'B'])\n"
        "r = g.gvar(f, W, p=1, q=1)\n"
        "r.girf('A', 'y', horizon=4); r.gfevd(4); r.pp({('A','y'): 1.0}, horizon=4)\n"
        "r.summary(); r.to_markdown()\n"
        "assert 'matplotlib' not in sys.modules, sorted(sys.modules)[:5]\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
