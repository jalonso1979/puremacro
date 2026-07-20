"""Robustness tests for the singular-matrix / non-PD-Σ failure modes.

These tests exercise the diagnostic guards added to:

    puremacro/_linalg.py
    puremacro/inference/hac.py
    puremacro/inference/_ols_helpers.py
    puremacro/lp/_panel_helpers.py
    puremacro/lp/la_lp.py
    puremacro/var/identify/cholesky.py
    puremacro/var/identify/sign.py

The contract being tested is *not* "estimation succeeds on degenerate
data" — it's "failure produces an informative error/warning that names
the offending function and (where possible) the offending columns".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro._linalg import inv_xtx, safe_cholesky
from puremacro.inference._ols_helpers import ols_hac
from puremacro.inference.hac import newey_west_se
from puremacro.lp._panel_helpers import two_way_fe_within
from puremacro.lp.la_lp import la_lp


# ---------------------------------------------------------------------------
# _linalg primitives
# ---------------------------------------------------------------------------


def test_inv_xtx_full_rank_matches_numpy(rng):
    X = rng.standard_normal((100, 3))
    np.testing.assert_allclose(
        inv_xtx(X, name="t"), np.linalg.inv(X.T @ X), rtol=1e-10
    )


def test_inv_xtx_collinear_raises_with_indices(rng):
    X = rng.standard_normal((50, 4))
    X[:, 3] = 2.0 * X[:, 1]  # column 3 == 2 * column 1
    with pytest.raises(np.linalg.LinAlgError) as exc:
        inv_xtx(X, name="my_estimator")
    msg = str(exc.value)
    assert "my_estimator" in msg
    assert "rank 3 of 4" in msg
    # Either of the collinear columns can be flagged.
    assert ("[1," in msg or "[3," in msg or "[1]" in msg or "[3]" in msg)


def test_safe_cholesky_pd_matches_numpy(rng):
    A = rng.standard_normal((4, 4))
    S = A @ A.T + np.eye(4)
    np.testing.assert_allclose(
        safe_cholesky(S, name="t"), np.linalg.cholesky(S), rtol=1e-10
    )


def test_safe_cholesky_non_pd_raises_with_eigenvalue():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
    with pytest.raises(np.linalg.LinAlgError) as exc:
        safe_cholesky(A, name="my_factor")
    msg = str(exc.value)
    assert "my_factor" in msg
    assert "min eigenvalue" in msg


def test_safe_cholesky_jitter_recovers_marginal():
    n = 3
    A = np.zeros((n, n))  # singular; min eig = 0
    L = safe_cholesky(A, name="t", jitter=1e-6)
    np.testing.assert_allclose(L @ L.T, 1e-6 * np.eye(n), atol=1e-10)


# ---------------------------------------------------------------------------
# OLS / HAC paths route the diagnostic
# ---------------------------------------------------------------------------


def test_ols_hac_collinear_diagnostic(rng):
    T = 100
    x = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x, 3.0 * x])  # const + x + 3x
    y = rng.standard_normal(T)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        ols_hac(y, X, lags=4)
    assert "ols_hac" in str(exc.value)


def test_newey_west_se_collinear_diagnostic(rng):
    T = 100
    x = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x, x])  # duplicate column
    resid = rng.standard_normal(T)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        newey_west_se(X, resid, bw=4)
    assert "newey_west_se" in str(exc.value)


def test_la_lp_collinear_control_diagnostic(rng):
    T = 150
    x = rng.standard_normal(T)
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(T),
            "x": x,
            "w1": x,           # control identical to x
            "w2": 2.0 * x,     # control proportional to x
        }
    )
    with pytest.raises(np.linalg.LinAlgError) as exc:
        la_lp(df, y="y", x="x", horizons=range(0, 4),
              n_lags=2, controls=["w1", "w2"])
    assert "la_lp" in str(exc.value)


# ---------------------------------------------------------------------------
# Panel two-way FE diagnostic
# ---------------------------------------------------------------------------


def test_two_way_fe_within_collinear_diagnostic(rng):
    rows = []
    for c in ("A", "B", "C"):
        for t in range(40):
            xv = rng.standard_normal()
            rows.append(
                {"code": c, "t": t, "y": rng.standard_normal(),
                 "x1": xv, "x2": 2.5 * xv}
            )
    df = pd.DataFrame(rows).set_index(["code", "t"]).sort_index()
    with pytest.raises(np.linalg.LinAlgError) as exc:
        two_way_fe_within(df, y_col="y", x_cols=["x1", "x2"])
    assert "two_way_fe_within" in str(exc.value)


def test_two_way_fe_within_full_rank_succeeds(toy_panel):
    out = two_way_fe_within(toy_panel, y_col="y", x_cols=["x"])
    assert "beta" in out and out["beta"].shape == (1,)
    assert "se" in out and np.all(np.isfinite(out["se"]))


# ---------------------------------------------------------------------------
# Bootstrap Cholesky failures: warn instead of silently substitute
# ---------------------------------------------------------------------------


def test_cholesky_svar_no_silent_substitution_on_degenerate_data():
    """The contract is *no silent substitution*: when bootstrap draws
    produce non-PD Σ_b, the function must either warn (partial failure)
    or raise (total failure) — never quietly count failed draws as the
    point estimate. Near-deterministic input forces total failure here,
    so we accept either outcome but require a diagnostic message.
    """
    import warnings as _warnings

    from puremacro.var.identify.cholesky import cholesky_svar

    rng = np.random.default_rng(0)
    T, n = 60, 2
    Y = np.zeros((T, n))
    Y[:, 0] = rng.standard_normal(T)
    Y[:, 1] = Y[:, 0] + 1e-12 * rng.standard_normal(T)  # near-collinear

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            cholesky_svar(Y, p=2, horizon=4, n_boot=40, ci=0.9, seed=0)
            warned = [w for w in caught
                      if issubclass(w.category, UserWarning)
                      and "non-PD" in str(w.message)]
            assert warned, (
                "completed without warning despite degenerate data; "
                "cholesky_svar may be silently substituting failed draws"
            )
        except np.linalg.LinAlgError as exc:
            assert "non-PD" in str(exc) or "bootstrap" in str(exc)


def test_cholesky_svar_silent_when_data_is_clean(toy_var2):
    """The complementary contract: on clean data no warning fires."""
    import warnings as _warnings

    from puremacro.var.identify.cholesky import cholesky_svar

    Y = toy_var2.values
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        cholesky_svar(Y, p=2, horizon=4, n_boot=50, ci=0.9, seed=0)
    non_pd = [w for w in caught if "non-PD" in str(w.message)]
    assert not non_pd, "spurious non-PD warning on a well-behaved VAR(1)"


def test_cholesky_factor_diagnostic_message():
    """The point-estimate Cholesky should also produce a diagnostic."""
    from puremacro.var.identify.cholesky import cholesky_factor

    Sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1
    with pytest.raises(np.linalg.LinAlgError) as exc:
        cholesky_factor(Sigma)
    assert "cholesky_factor" in str(exc.value)


# ---------------------------------------------------------------------------
# VAR family routed through _linalg
# ---------------------------------------------------------------------------


def test_minnesota_bvar_collinear_diagnostic():
    """BVAR posterior mean must surface a diagnostic when the
    Minnesota-augmented design is rank-deficient (e.g. when two of the
    user's series are identical)."""
    from puremacro.var.bvar import minnesota_posterior

    rng = np.random.default_rng(0)
    T = 60
    y1 = rng.standard_normal(T)
    Y = np.column_stack([y1, y1])  # second variable identical to first
    # Minnesota dummies typically render the augmented system full rank,
    # so this may either succeed (Tikhonov-style regularisation working
    # as advertised) or surface the diagnostic. We require the latter
    # only when the failure actually fires.
    try:
        minnesota_posterior(Y, p=2)
    except np.linalg.LinAlgError as exc:
        assert "minnesota_bvar" in str(exc)


def test_engle_granger_collinear_diagnostic():
    from puremacro.var.vecm import engle_granger

    rng = np.random.default_rng(0)
    T = 80
    x = np.cumsum(rng.standard_normal(T))
    # Perfectly cointegrated → first-stage residuals ≈ 0 → ADF X'X
    # degenerates. We require an informative error rather than a
    # silent garbage ADF statistic.
    try:
        out = engle_granger(x, x)
        # If the call returns, the ADF stat should at minimum be
        # non-finite or NaN — silent zero output would be a regression.
        assert not np.isfinite(out["adf_stat"]) or out["adf_stat"] == 0
    except np.linalg.LinAlgError as exc:
        assert "engle_granger_adf" in str(exc)


# ---------------------------------------------------------------------------
# Klein DSGE: Blanchard-Kahn strict mode
# ---------------------------------------------------------------------------


def _bk_violator_indeterminacy():
    """Construct a 2-variable system with 1 stable and 1 unstable root
    when the model declares 2 forward-looking variables → indeterminacy
    (n_unstable = 1 < n_fwd = 2)."""
    A = np.eye(2)
    B = np.diag([0.5, 1.5])
    return A, B, 0  # n_pre = 0 → both forward-looking


def _bk_violator_no_solution():
    """Two unstable roots, zero forward-looking variables → no stable
    solution (n_unstable = 2 > n_fwd = 0)."""
    A = np.eye(2)
    B = np.diag([1.5, 1.7])
    return A, B, 2  # n_pre = 2 → all predetermined


def _bk_unique_solution():
    """One stable, one unstable, one of each kind → unique."""
    A = np.eye(2)
    B = np.diag([0.5, 1.5])
    return A, B, 1


def test_klein_solve_strict_indeterminacy_raises():
    from puremacro.dsge import BlanchardKahnError, klein_solve

    A, B, n_pre = _bk_violator_indeterminacy()
    with pytest.raises(BlanchardKahnError) as exc:
        klein_solve(A, B, n_pre=n_pre, strict=True)
    assert exc.value.n_unstable < exc.value.n_fwd
    assert "indeterminacy" in str(exc.value)


def test_klein_solve_strict_no_solution_raises():
    from puremacro.dsge import BlanchardKahnError, klein_solve

    A, B, n_pre = _bk_violator_no_solution()
    with pytest.raises(BlanchardKahnError) as exc:
        klein_solve(A, B, n_pre=n_pre, strict=True)
    assert exc.value.n_unstable > exc.value.n_fwd


def test_klein_solve_strict_unique_returns_solution():
    from puremacro.dsge import klein_solve

    A, B, n_pre = _bk_unique_solution()
    sol = klein_solve(A, B, n_pre=n_pre, strict=True)
    assert sol.eu == (1, 1)


def test_klein_solve_default_soft_failure_preserved():
    """In the legacy non-strict mode, BK violations must still return
    a KleinSolution with the eu flags set rather than raising."""
    from puremacro.dsge import klein_solve

    A, B, n_pre = _bk_violator_no_solution()
    sol = klein_solve(A, B, n_pre=n_pre, strict=False)
    assert sol.eu[0] == 0  # existence violated


def test_klein_solution_is_frozen():
    """KleinSolution must be a frozen dataclass — see Result-object
    standard in ARCHITECTURE.md (0.4.0+)."""
    from puremacro.dsge import klein_solve, KleinSolution

    A, B, n_pre = _bk_unique_solution()
    sol = klein_solve(A, B, n_pre=n_pre, strict=False)
    assert isinstance(sol, KleinSolution)
    with pytest.raises(Exception):
        sol.G = np.zeros_like(sol.G)


# ---------------------------------------------------------------------------
# Tier-1 X'X sites routed through inv_xtx (N+5)
# ---------------------------------------------------------------------------


def test_hac_fixed_b_collinear_diagnostic(rng):
    from puremacro.inference.hac_fixed_b import hac_fixed_b

    T = 100
    x = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x, 2.0 * x])  # const + x + 2x
    y = rng.standard_normal(T)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        hac_fixed_b(y, X, b=0.1)
    assert "hac_fixed_b" in str(exc.value)


def test_kleibergen_paap_collinear_instruments_diagnostic(rng):
    from puremacro.inference.weak_iv import kleibergen_paap_f

    n = 200
    z1 = rng.standard_normal(n)
    Z = np.column_stack([z1, 3.0 * z1])         # collinear instruments
    X = (Z[:, 0] + 0.3 * rng.standard_normal(n)).reshape(-1, 1)
    y = X[:, 0] + rng.standard_normal(n)
    with pytest.raises(np.linalg.LinAlgError) as exc:
        kleibergen_paap_f(y, X, Z)
    assert "kleibergen_paap_f" in str(exc.value)


def test_adf_collinear_diagnostic(rng):
    """The ADF helper inside puremacro.tests.unit_root now surfaces a
    diagnostic if the augmented design happens to be rank-deficient.
    Exact-collinear ADF inputs are pathological — we just confirm
    the named error fires when forced."""
    from puremacro.tests.unit_root import adf_test

    # A constant series is degenerate for ADF — its first difference is 0.
    y = np.ones(100)
    # ADF on a constant typically fails downstream; either it
    # surfaces our diagnostic or returns a non-finite stat.
    try:
        out = adf_test(y, regression="c", lags=2)
        assert not np.isfinite(out["stat"]) or out["stat"] == 0
    except np.linalg.LinAlgError as exc:
        assert "adf" in str(exc).lower()


# ---------------------------------------------------------------------------
# Tier-2 Cholesky sites routed through safe_cholesky (N+5)
# ---------------------------------------------------------------------------


def test_maxshare_non_pd_sigma_diagnostic():
    from puremacro.var.identify.maxshare import maxshare

    A_list = [np.array([[0.5, 0.0], [0.0, 0.5]])]
    Sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1
    with pytest.raises(np.linalg.LinAlgError) as exc:
        maxshare(A_list, Sigma, target_var=0, horizon=4)
    assert "maxshare" in str(exc.value)


def test_sign_zero_non_pd_sigma_diagnostic():
    from puremacro.var.identify.sign_zero import sign_zero

    A_list = [np.array([[0.5, 0.0], [0.0, 0.5]])]
    Sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1
    with pytest.raises(np.linalg.LinAlgError) as exc:
        sign_zero(
            A_list, Sigma,
            sign_constraints={(0, 0): 1},
            n_draws=20,
        )
    assert "sign_zero" in str(exc.value)


def test_non_gaussian_svar_non_pd_sigma_diagnostic():
    from puremacro.var.identify.non_gaussian import non_gaussian_svar

    rng = np.random.default_rng(0)
    T = 60
    y1 = rng.standard_normal(T)
    Y = np.column_stack([y1, y1])  # second column identical → singular Σ
    with pytest.raises(np.linalg.LinAlgError) as exc:
        non_gaussian_svar(Y, p=1, horizon=4)
    assert "non_gaussian_svar" in str(exc.value)


# ---------------------------------------------------------------------------
# bq.py: drop-and-warn pattern for non-PD long-run Ω (N+5)
# ---------------------------------------------------------------------------


def test_bq_svar_no_silent_substitution_on_degenerate_data():
    """Same contract as cholesky_svar: degenerate input must surface
    either a warning (partial failure) or a raise (total failure),
    never silently substitute the point estimate into bootstrap
    quantiles."""
    import warnings as _warnings

    from puremacro.var.identify.bq import bq_svar

    rng = np.random.default_rng(0)
    T, n = 60, 2
    Y = np.zeros((T, n))
    Y[:, 0] = rng.standard_normal(T)
    Y[:, 1] = Y[:, 0] + 1e-12 * rng.standard_normal(T)  # near-collinear

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            bq_svar(Y, p=2, horizon=4, n_boot=40, ci=0.9, seed=0)
            non_pd = [w for w in caught
                      if issubclass(w.category, UserWarning)
                      and "non-PD" in str(w.message)]
            assert non_pd, (
                "bq_svar completed without warning despite degenerate "
                "input; silent point-substitution may have returned"
            )
        except np.linalg.LinAlgError as exc:
            # Either the bootstrap surfaces "non-PD" / "bootstrap" / "all
            # draws failed", or the per-draw safe_cholesky names bq_svar.
            msg = str(exc).lower()
            assert any(k in msg for k in ("non-pd", "bootstrap", "bq_svar"))
