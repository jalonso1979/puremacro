"""Tests for Magnusson-Mavroeidis SVAR (puremacro.var.identify.magmav)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.identify import magmav as mm


def _synthetic_var2_with_breaks(T: int, breaks: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """VAR(1) on 2 vars with regime-specific structural variances. Returns (Y, true_B)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.1], [0.0, 0.5]])
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    n = 2
    # Build per-period variances: regimes between break boundaries.
    boundaries = (0,) + tuple(breaks) + (T,)
    sigma_per_t = np.zeros((T, n))
    for g in range(len(boundaries) - 1):
        s_lo, s_hi = boundaries[g], boundaries[g + 1]
        # Each regime has different shock variances
        v0 = 0.5 + 1.5 * g  # var of shock 0 in regime g
        v1 = 2.0 - 0.4 * g  # var of shock 1 in regime g
        sigma_per_t[s_lo:s_hi] = [np.sqrt(v0), np.sqrt(v1)]
    Y = np.zeros((T, n))
    for t in range(1, T):
        eps = rng.standard_normal(n) * sigma_per_t[t]
        Y[t] = A @ Y[t - 1] + B @ eps
    return Y, B


def test_sup_wald_scan_finds_known_break():
    Y, _ = _synthetic_var2_with_breaks(T=400, breaks=(200,), seed=0)
    from puremacro.var.estimate import estimate_var
    A_list, _, _, resid, _ = estimate_var(Y, 1)
    # Single-break scan
    tau, stat = mm._sup_wald_one_break(resid, lo_frac=0.15, hi_frac=0.85)
    # True break at t=200 (residual index = 199 because of one lag)
    assert abs(tau - 199) < 30, f"detected break {tau} far from true 199"
    assert stat > 0


def test_sup_wald_returns_neg_inf_on_degenerate_segment():
    # A residual matrix too short for any trimmed break: lo > hi.
    # T=6, n=2: lo=max(int(0.15*6), 2+2)=max(0,4)=4, hi=min(int(0.85*6), 6-2-2)=min(5,2)=2 -> hi<lo
    rng = np.random.default_rng(0)
    resid_short = rng.standard_normal((6, 2))
    tau, stat = mm._sup_wald_one_break(resid_short, lo_frac=0.15, hi_frac=0.85)
    assert stat == -np.inf, f"expected -inf on degenerate segment, got {stat}"


def test_estimate_B_recovers_known_matrix():
    rng = np.random.default_rng(7)
    n, T = 2, 1000
    B_true = np.array([[1.0, 0.3], [0.2, 0.8]])
    # Two regimes with distinct structural-shock variances
    D0 = np.diag([0.5, 2.0])
    D1 = np.diag([2.5, 0.6])
    # Generate residuals: u = B eps, eps ~ N(0, D_g)
    u0 = (rng.standard_normal((T // 2, n)) * np.sqrt(np.diag(D0))) @ B_true.T
    u1 = (rng.standard_normal((T // 2, n)) * np.sqrt(np.diag(D1))) @ B_true.T
    Sigmas = [u0.T @ u0 / (T // 2), u1.T @ u1 / (T // 2)]
    B_hat, D_hat, success = mm._estimate_B_from_regime_covariances(
        Sigmas, n_starts=3, seed=0,
    )
    assert success
    # B is identified up to column permutation + sign; compare via B B^T
    err = np.linalg.norm(B_hat @ B_hat.T - B_true @ B_true.T, ord="fro")
    assert err < 0.4, f"frobenius error {err:.3f} too large"


def test_magmav_svar_returns_result_dataclass():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=2)
    from puremacro.var.identify._results import MagMavSVARResult
    res = mm.magmav_svar(Y, p=1, horizon=8, k_breaks=1, n_boot=50, seed=2)
    assert isinstance(res, MagMavSVARResult)
    assert res.irf_point.shape == (9, 2, 2)
    assert res.irf_lower.shape == res.irf_point.shape
    assert res.irf_upper.shape == res.irf_point.shape
    assert res.B.shape == (2, 2)
    assert res.k_breaks == 1
    assert res.n_boot == 50
    assert res.ci == 0.9
    assert isinstance(res.variance_change_dates, tuple)


def test_magmav_svar_bic_selects_no_breaks_on_homoskedastic_data():
    rng = np.random.default_rng(3)
    n, T = 2, 250
    B_true = np.array([[1.0, 0.2], [0.0, 0.9]])
    Y = np.zeros((T, n))
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B_true @ rng.standard_normal(n)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=None, n_boot=20, seed=0)
    # k_breaks=0 means BIC fell back; eu signals failure.
    assert res.k_breaks == 0, f"BIC should pick k=0 on homoskedastic data, got {res.k_breaks}"
    assert res.eu == (0, 0)


def test_magmav_svar_seed_reproducibility():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=4)
    r1 = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=30, seed=11)
    r2 = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=30, seed=11)
    np.testing.assert_allclose(r1.irf_lower, r2.irf_lower)
    np.testing.assert_allclose(r1.irf_upper, r2.irf_upper)


def test_magmav_svar_irf_shapes_match_horizon():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=5)
    res = mm.magmav_svar(Y, p=1, horizon=12, k_breaks=1, n_boot=30, seed=5)
    H = 12
    n = 2
    for arr in (res.irf_point, res.irf_lower, res.irf_upper):
        assert arr.shape == (H + 1, n, n)


def test_magmav_svar_detects_two_breaks_when_present():
    Y, _ = _synthetic_var2_with_breaks(T=600, breaks=(200, 400), seed=6)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=2, n_boot=20, seed=6)
    assert res.k_breaks == 2
    assert len(res.variance_change_dates) == 2
    detected = np.array(res.variance_change_dates)
    # Allow generous tolerance — breaks are noisy with T=600 and n=2.
    assert np.min(np.abs(detected - 199)) < 80
    assert np.min(np.abs(detected - 399)) < 80


def test_magmav_svar_handles_explicit_k_breaks_arg():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=7)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=2, n_boot=20, seed=7)
    # Explicit k=2 means BIC is skipped.
    assert res.k_breaks <= 2


def test_magmav_svar_bootstrap_band_covers_irf_majority():
    Y, _ = _synthetic_var2_with_breaks(T=400, breaks=(200,), seed=8)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=100, seed=8)
    # Point IRF should sit within bands at most horizons (not all — bands are 90%).
    inside = (res.irf_point >= res.irf_lower) & (res.irf_point <= res.irf_upper)
    assert inside.mean() > 0.5


def test_magmav_svar_exported_from_identify_package():
    from puremacro.var.identify import magmav_svar as exported
    assert exported is mm.magmav_svar
    from puremacro.var.identify import MagMavSVARResult
    from puremacro.var.identify._results import MagMavSVARResult as direct
    assert MagMavSVARResult is direct


def test_magmav_svar_variance_change_dates_is_int_tuple():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=9)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=10, seed=9)
    assert isinstance(res.variance_change_dates, tuple)
    for x in res.variance_change_dates:
        assert isinstance(x, int)
