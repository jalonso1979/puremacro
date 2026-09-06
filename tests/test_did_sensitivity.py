"""Tests for puremacro.did.honest_did (Rambachan & Roth 2023 sensitivity analysis).

Reference numbers are derived by hand (closed forms) or by an independent
scipy.optimize.linprog formulation of the restriction sets, never copied from
the implementation.  Several tests are regression tests for defects found by
the 2.3.0 audit; their docstrings describe the behaviour of the old code.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.optimize
from scipy.stats import norm

from puremacro.did import (
    callaway_santanna,
    honest_did,
    honest_did_sensitivity,
    HonestDiDResult,
)
from puremacro.did.sensitivity import (
    _SDFrontier,
    _build_window,
    _folded_normal_quantile,
    _imbens_manski_critical_value,
    _imbens_manski_interval,
    _rm_bounds,
    _second_difference_matrix,
)

Z975 = float(norm.ppf(0.975))


# =============================================================================
# Independent reference implementations
# =============================================================================


def im_critical(width: float, se: float, alpha: float = 0.05) -> float:
    """Imbens-Manski critical value solved independently with brentq."""
    if width / se < 1e-8:
        return float(norm.ppf(1 - alpha / 2))
    f = lambda c: norm.cdf(c + width / se) - norm.cdf(-c) - (1 - alpha)  # noqa: E731
    return float(scipy.optimize.brentq(f, norm.ppf(1 - alpha) - 1e-9, norm.ppf(1 - alpha / 2) + 1e-9))


def brute_sd(b_pre, b_post, M, l_vec):
    """Paper / HonestDiD-R identified set for Delta^SD(M) via linprog over the full window.

    delta = (b_pre fixed, 0 at the reference period, delta_post free); every
    consecutive triple of the full window obeys |Delta^2 delta| <= M.
    Returns (id_lo, id_hi) or (nan, nan) when the LP is infeasible.
    """
    K, L = len(b_pre), len(b_post)
    T = K + 1 + L
    rows = []
    for t in range(T - 2):
        r = np.zeros(T)
        r[t], r[t + 1], r[t + 2] = 1.0, -2.0, 1.0
        rows.append(r)
    A = np.array(rows)
    A_post = A[:, K + 1 :]
    const = A[:, :K] @ b_pre
    A_ub = np.vstack([A_post, -A_post])
    b_ub = np.concatenate([M - const, M + const])
    rmax = scipy.optimize.linprog(-l_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * L, method="highs")
    rmin = scipy.optimize.linprog(l_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * L, method="highs")
    if not (rmax.success and rmin.success):
        return np.nan, np.nan
    th = l_vec @ b_post
    return th - (-rmax.fun), th - rmin.fun


def brute_rm_fd(b_pre, b_post, Mbar, l_vec):
    """Paper Delta^RM(Mbar): |delta_{t+1}-delta_t| <= Mbar * max_{s<0}|delta_{s+1}-delta_s|."""
    L = len(b_post)
    pre_full = np.concatenate([b_pre, [0.0]])
    dmax = np.max(np.abs(np.diff(pre_full)))
    D = np.eye(L) - np.eye(L, k=-1)
    A_ub = np.vstack([D, -D])
    b_ub = np.full(2 * L, Mbar * dmax)
    rmax = scipy.optimize.linprog(-l_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * L, method="highs")
    rmin = scipy.optimize.linprog(l_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * L, method="highs")
    th = l_vec @ b_post
    return th - (-rmax.fun), th - rmin.fun


# =============================================================================
# Benzarti & Carloni (2019) inputs as used by Rambachan & Roth (2023, Section 6.1)
# =============================================================================
BC_BETAHAT = np.array(
    [
        0.0066963518,
        0.0293450337,
        -0.0064729722,
        0.0730149895,
        0.1959611177,
        0.3120639026,
        0.2395415455,
        0.1260425001,
    ]
)
BC_YEARS = [2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012]
BC_REF_YEAR = 2008
BC_SIGMA = np.array(
    [
        [0.0008428358, 0.0004768687, 0.0002618051, 0.0002354220, 0.0001676371, 0.0001128708, 0.0000199282, -0.0001368265],
        [0.0004768687, 0.0006425420, 0.0003987425, 0.0002435515, 0.0002201960, 0.0001804591, 0.0000384377, -0.0000296042],
        [0.0002618051, 0.0003987425, 0.0005229950, 0.0002117686, 0.0001840722, 0.0001458528, 0.0000700520, 0.0000595299],
        [0.0002354220, 0.0002435515, 0.0002117686, 0.0003089595, 0.0001197866, 0.0001334081, 0.0001016335, 0.0001079052],
        [0.0001676371, 0.0002201960, 0.0001840722, 0.0001197866, 0.0003599704, 0.0002478819, 0.0001749579, 0.0001654257],
        [0.0001128708, 0.0001804591, 0.0001458528, 0.0001334081, 0.0002478819, 0.0004263950, 0.0002171438, 0.0002892748],
        [0.0000199282, 0.0000384377, 0.0000700520, 0.0001016335, 0.0001749579, 0.0002171438, 0.0004886698, 0.0003805322],
        [-0.0001368265, -0.0000296042, 0.0000595299, 0.0001079052, 0.0001654257, 0.0002892748, 0.0003805322, 0.0007617394],
    ]
)


# =============================================================================
# Relative magnitudes: RR's first-difference Delta^RM with delta-method IM CI
# =============================================================================


def test_honest_did_basic_relative_magnitude():
    """RR Delta^RM(Mbar) in first differences with hand-derived bounds, SEs and M*.

    Old code (audit M108/C46): the default bound was |delta_l| <= Mbar*max|beta_pre|
    (levels, benchmark 0.20 here) and the CI used se(theta_hat) for both endpoints,
    ignoring the sampling variance of the benchmark.
    """
    event_time = [-3, -2, -1, 0, 1, 2]
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]

    res = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=0,
        method="relative_magnitude",
        m_grid=[0.0, 0.5, 1.0, 2.0],
        ci=0.95,
    )

    assert isinstance(res, HonestDiDResult)
    assert res.method == "relative_magnitude"
    assert res.bound == "first_difference"
    assert res.ci_method == "imbens_manski"
    assert res.ci == 0.95
    # pre first differences: (-0.20 - 0.15) = -0.35, (0 - (-0.20)) = 0.20 -> benchmark 0.35
    assert res.pre_trend_max == pytest.approx(0.35)

    tbl = res.to_frame()
    assert list(tbl["M"]) == [0.0, 0.5, 1.0, 2.0]

    m0 = tbl[tbl["M"] == 0.0].iloc[0]
    assert m0["id_lo"] == pytest.approx(1.50)
    assert m0["id_hi"] == pytest.approx(1.50)
    assert m0["ci_lo"] == pytest.approx(1.50 - Z975 * 0.15, rel=1e-6)
    assert m0["ci_hi"] == pytest.approx(1.50 + Z975 * 0.15, rel=1e-6)
    assert bool(m0["significant"]) is True

    # Mbar = 1: |delta_0| <= 0.35 -> [1.15, 1.85];  theta_lo = beta_0 - Mbar*|beta_{-2} - beta_{-3}|
    # var(theta_lo) = 0.15^2 + Mbar^2 (0.10^2 + 0.10^2)
    def hand_ci(mbar):
        lo, hi = 1.5 - 0.35 * mbar, 1.5 + 0.35 * mbar
        s = np.sqrt(0.15**2 + 0.02 * mbar**2)
        c = im_critical(hi - lo, s)
        return lo - c * s, hi + c * s

    for mbar in [0.5, 1.0, 2.0]:
        row = tbl[tbl["M"] == mbar].iloc[0]
        assert row["id_lo"] == pytest.approx(1.5 - 0.35 * mbar)
        assert row["id_hi"] == pytest.approx(1.5 + 0.35 * mbar)
        lo, hi = hand_ci(mbar)
        assert row["ci_lo"] == pytest.approx(lo, abs=1e-8)
        assert row["ci_hi"] == pytest.approx(hi, abs=1e-8)

    m_star_hand = scipy.optimize.brentq(lambda m: hand_ci(m)[0], 0.01, 10.0)
    assert isinstance(res.breakdown_value, float)
    assert res.breakdown_value == pytest.approx(m_star_hand, abs=1e-6)
    assert 2.4 < res.breakdown_value < 2.6


def test_relative_magnitude_matches_independent_lp():
    """Closed-form Delta^RM bounds equal an independent linprog formulation (h=0, h=2, average)."""
    et = [-5, -4, -3, -2, 0, 1, 2]
    b_pre = np.array([-0.30, -0.05, -0.25, -0.20])
    b_post = np.array([1.0, 1.2, 1.4])
    b = np.concatenate([b_pre, b_post])
    se = np.full(7, 0.1)
    for lv in [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([1 / 3, 1 / 3, 1 / 3])]:
        for mb in [0.0, 0.5, 1.0, 2.0]:
            lo, hi = brute_rm_fd(b_pre, b_post, mb, lv)
            row = honest_did(b, se=se, event_time=et, method="relative_magnitude", m_vec=[mb], l_vec=lv).table.iloc[0]
            assert row["id_lo"] == pytest.approx(lo, abs=1e-9)
            assert row["id_hi"] == pytest.approx(hi, abs=1e-9)


def test_relative_magnitude_levels_option():
    """bound='levels' is the explicitly named level bound |delta_t| <= Mbar*max|beta_pre| (not Delta^RM)."""
    event_time = [-3, -2, -1, 0, 1]
    beta = [0.10, -0.10, 0.0, 1.0, 1.2]
    se = [0.05, 0.05, 0.0, 0.10, 0.15]
    res = honest_did(
        b_hat=beta, se=se, event_time=event_time, method="relative_magnitude",
        bound="levels", m_vec=[0.0, 0.5, 1.0], target_horizon=1,
    )
    assert res.bound == "levels"
    assert res.pre_trend_max == pytest.approx(0.10)
    row = res.table[res.table["M"] == 1.0].iloc[0]
    assert row["id_lo"] == pytest.approx(1.2 - 0.10)
    assert row["id_hi"] == pytest.approx(1.2 + 0.10)
    # se of the endpoints: var = 0.15^2 + Mbar^2 * 0.05^2 (benchmark |beta_{-3}| or |beta_{-2}|, tie -> first)
    s = np.sqrt(0.15**2 + 0.05**2)
    c = im_critical(0.20, s)
    assert row["ci_lo"] == pytest.approx(1.1 - c * s, abs=1e-8)
    # RR's first-difference version for the same inputs: benchmark 0.20, h=1 -> [0.8, 1.6]
    res_fd = honest_did(b_hat=beta, se=se, event_time=event_time, method="relative_magnitude", m_vec=[1.0], target_horizon=1)
    assert res_fd.table.iloc[0]["id_lo"] == pytest.approx(0.8)
    assert res_fd.table.iloc[0]["id_hi"] == pytest.approx(1.6)
    # the old spelling of the option is still accepted
    res_alias = honest_did(
        b_hat=beta, se=se, event_time=event_time, method="relative_magnitude",
        bound="deviation from parallel trends", m_vec=[1.0], target_horizon=1,
    )
    assert res_alias.bound == "levels"


def test_breakdown_value_brentq_relative_magnitude():
    """M* solves ci_lo(M*) = 0 for the delta-method IM interval (hand root 3.0467)."""
    event_time = [-3, -2, -1, 0]
    beta = [0.10, -0.10, 0.0, 1.0]
    se = [0.05, 0.05, 0.0, 0.10]
    res = honest_did(b_hat=beta, se=se, event_time=event_time, method="relative_magnitude", target_horizon=0)
    m_star = res.breakdown_value
    assert isinstance(m_star, float)

    # benchmark max(|-0.10-0.10|, |0+0.10|) = 0.20 with var 0.05^2 + 0.05^2 = 0.005
    def ci_lo(m):
        lo, hi = 1 - 0.2 * m, 1 + 0.2 * m
        s = np.sqrt(0.01 + 0.005 * m * m)
        return lo - im_critical(hi - lo, s) * s

    assert m_star == pytest.approx(scipy.optimize.brentq(ci_lo, 0.01, 10.0), abs=1e-6)
    assert 3.0 < m_star < 3.1

    res_at = honest_did(b_hat=beta, se=se, event_time=event_time, method="relative_magnitude", m_vec=[m_star], target_horizon=0)
    assert abs(res_at.table.iloc[0]["ci_lo"]) < 1e-6


# =============================================================================
# Smoothness: RR's Delta^SD and the FLCI
# =============================================================================


def test_smoothness_plugin_set_matches_paper_lp_and_is_empty_when_infeasible():
    """Plug-in Delta^SD(M) bounds equal the paper's LP over the full window (audit C45 / M7).

    Old code anchored the first post-period slope on an OLS fit of the pre
    coefficients (np.polyfit) and never constrained pre-period second
    differences: with a non-linear pre-trend it returned finite bounds where
    the RR set is empty and bounds offset by +0.19 (h=0) where it is not.
    """
    et = [-5, -4, -3, -2, 0, 1, 2]
    se = np.full(7, 0.1)
    # Case A: exactly linear pre-trend through the reference period (slope 0.2)
    b_pre = np.array([0.2 * (t + 1) for t in [-5, -4, -3, -2]])
    b_post = np.array([1.0, 1.2, 1.4])
    b = np.concatenate([b_pre, b_post])
    specs = [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([1 / 3, 1 / 3, 1 / 3])]
    for lv in specs:
        for M in [0.0, 0.1, 0.3, 1.0]:
            lo, hi = brute_sd(b_pre, b_post, M, lv)
            row = honest_did(b, se=se, event_time=et, method="smoothness", m_vec=[M], l_vec=lv).table.iloc[0]
            assert row["id_lo"] == pytest.approx(lo, abs=1e-8)
            assert row["id_hi"] == pytest.approx(hi, abs=1e-8)
    # M = 0 continues the pre-trend linearly: theta = beta_0 - 0.2 = 0.8 (point)
    r0 = honest_did(b, se=se, event_time=et, method="smoothness", m_vec=[0.0], target_horizon=0)
    assert r0.table.iloc[0]["id_lo"] == pytest.approx(0.8)
    assert r0.table.iloc[0]["id_hi"] == pytest.approx(0.8)
    assert r0.pre_trend_slope == pytest.approx(0.2)
    assert r0.pre_trend_max_second_diff == pytest.approx(0.0, abs=1e-12)

    # Case B: non-linear pre-trend, max |Delta^2 beta_pre| = 0.45 -> empty for M < 0.45
    b_pre = np.array([-0.30, -0.05, -0.25, -0.20])
    b = np.concatenate([b_pre, b_post])
    with pytest.warns(UserWarning, match="identified set is empty"):
        res = honest_did(b, se=se, event_time=et, method="smoothness", m_vec=[0.0, 0.3, 0.45, 0.6, 1.0], target_horizon=0)
    assert res.pre_trend_max_second_diff == pytest.approx(0.45)
    tbl = res.table
    assert np.isnan(tbl[tbl["M"] == 0.0].iloc[0]["id_lo"])
    assert np.isnan(tbl[tbl["M"] == 0.3].iloc[0]["id_hi"])
    for M in [0.45, 0.6, 1.0]:
        lo, hi = brute_sd(b_pre, b_post, M, np.array([1.0, 0, 0]))
        row = tbl[tbl["M"] == M].iloc[0]
        assert row["id_lo"] == pytest.approx(lo, abs=1e-8)
        assert row["id_hi"] == pytest.approx(hi, abs=1e-8)
    # the FLCI is finite and valid everywhere, including where the plug-in set is empty
    assert np.all(np.isfinite(tbl["ci_lo"])) and np.all(np.isfinite(tbl["ci_hi"]))
    assert (np.diff(tbl["ci_hi"] - tbl["ci_lo"]) >= -1e-10).all()


def test_smoothness_reference_period_anchor():
    """The triple spanning the reference period links delta_0 to beta_{-2} (audit C45).

    Old code: id set at M=0 was 1.35 (polyfit slope -0.15 over the pre coefficients
    only); RR's Delta^SD(0) with delta_{-1}=0 gives delta_0 = -beta_{-2}, theta = 1.15.
    With two pre periods the plug-in set at M=0 is empty unless the pre-trend is
    linear, so the check uses a linear pre-trend and, separately, the boundary
    constraint at M = max|Delta^2 beta_pre|.
    """
    et = [-3, -2, -1, 0, 1]
    b = [0.10, 0.05, 0.0, 1.2, 1.5]  # linear through the base: slope -0.05
    res = honest_did(b, se=[0.1, 0.1, 0.0, 0.2, 0.2], event_time=et, method="smoothness", m_vec=[0.0], target_horizon=0)
    assert res.table.iloc[0]["id_lo"] == pytest.approx(1.2 + 0.05)
    assert res.table.iloc[0]["id_hi"] == pytest.approx(1.2 + 0.05)
    # audit example: pre = [0.1, -0.05]; second difference at the base = 0 - 2(-0.05) + 0.1 = 0.2
    b2 = [0.1, -0.05, 0.0, 1.2, 1.5]
    with pytest.warns(UserWarning, match="identified set is empty"):
        res2 = honest_did(b2, se=[0.1, 0.1, 0.0, 0.2, 0.2], event_time=et, method="smoothness", m_vec=[0.0, 0.2], target_horizon=0)
    assert np.isnan(res2.table.iloc[0]["id_lo"])
    # at M = 0.2: delta_0 in [-beta_{-2} - M, -beta_{-2} + M] = [-0.15, 0.25] -> theta in [0.95, 1.35]
    assert res2.table.iloc[1]["id_lo"] == pytest.approx(0.95)
    assert res2.table.iloc[1]["id_hi"] == pytest.approx(1.35)


def test_flci_worst_case_bias_equals_lp():
    """M * ||lambda||_1 with A' lambda = a equals sup_{|A delta| <= M} a'delta (dual of the bias LP)."""
    rng = np.random.default_rng(3)
    T, base = 8, 3
    A = _second_difference_matrix(T)
    nb = [p for p in range(T) if p != base]
    A_nb = A[:, nb]
    tvec = np.array([p - base for p in nb], dtype=float)
    for _ in range(5):
        w = rng.standard_normal(T - 1)
        w -= tvec * (w @ tvec) / (tvec @ tvec)  # annihilate linear trends
        lam = np.linalg.lstsq(A_nb.T, w, rcond=None)[0]
        res = scipy.optimize.linprog(
            -w, A_ub=np.vstack([A_nb, -A_nb]), b_ub=np.ones(2 * (T - 2)),
            bounds=[(None, None)] * (T - 1), method="highs",
        )
        assert res.success
        assert -res.fun == pytest.approx(np.abs(lam).sum(), rel=1e-8)


def test_flci_single_pre_period_equals_imbens_manski_closed_form():
    """With one pre period the FLCI equals the IM interval around beta_0 + beta_{-2} with the full-Sigma SE."""
    sigma = np.array([[0.01, 0.002], [0.002, 0.04]])
    res = honest_did([0.1, 1.2], sigma=sigma, event_time=[-2, 0], method="smoothness", m_vec=[0.3], target_horizon=0)
    row = res.table.iloc[0]
    theta = 1.3
    s = np.sqrt(0.01 + 0.04 + 2 * 0.002)
    c = im_critical(0.6, s)
    assert row["ci_lo"] == pytest.approx(theta - 0.3 - c * s, abs=1e-8)
    assert row["ci_hi"] == pytest.approx(theta + 0.3 + c * s, abs=1e-8)


def test_flci_family_is_near_optimal():
    """The frontier family's half-length is within 0.1% of a direct Nelder-Mead minimisation."""
    win = _build_window(np.array(BC_YEARS, dtype=float), BC_BETAHAT, BC_SIGMA, float(BC_REF_YEAR))
    fr = _SDFrontier(win.sigma_nb, np.array([1.0, 0, 0, 0]), win.pre_idx, win.post_idx, win.tvec)
    M = 0.02
    lo, hi = fr.flci(win.beta_nb, M, 0.05)
    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(6):
        v0 = rng.standard_normal(fr.nfree) * 3

        def half(v):
            w = fr.w0 + fr.N @ v
            lam = fr.lam0 + fr.Lam @ v
            s = np.sqrt(w @ win.sigma_nb @ w)
            return float(s * _folded_normal_quantile(M * np.abs(lam).sum() / s, 0.05)[0])

        r = scipy.optimize.minimize(half, v0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
        best = min(best, r.fun)
    assert (hi - lo) / 2 <= best * 1.001
    assert (hi - lo) / 2 >= best * 0.999  # it cannot beat the true optimum


def test_sigma_pre_rows_and_covariances_enter_the_confidence_sets():
    """Pre-period variances and pre-post covariances change the CI and M* (audit C46 / M8).

    Old code used se(l'beta_post) for both CI ends, so zeroing all pre-post
    covariances and inflating the pre variances 1000x left every number unchanged.
    """
    S2 = BC_SIGMA.copy()
    S2[:4, 4:] = 0.0
    S2[4:, :4] = 0.0
    S2[:4, :4] = np.eye(4) * 1.0
    for meth in ["smoothness", "relative_magnitude"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = honest_did(BC_BETAHAT, sigma=BC_SIGMA, event_time=BC_YEARS, base_period=2008, method=meth, m_vec=[0.1, 1.0], target_horizon=2009)
            b = honest_did(BC_BETAHAT, sigma=S2, event_time=BC_YEARS, base_period=2008, method=meth, m_vec=[0.1, 1.0], target_horizon=2009)
        assert not np.allclose(a.table[["ci_lo", "ci_hi"]].to_numpy(), b.table[["ci_lo", "ci_hi"]].to_numpy())
        assert b.breakdown_value < a.breakdown_value
        # wider CIs with noisier pre-periods
        assert (b.table["ci_hi"] - b.table["ci_lo"]).to_numpy()[0] > (a.table["ci_hi"] - a.table["ci_lo"]).to_numpy()[0]


# =============================================================================
# Monte Carlo coverage (the audit's DGPs)
# =============================================================================


def test_coverage_monte_carlo_smoothness():
    """Coverage of the 95% FLCI on the audit's Delta^SD DGPs is at least 0.94.

    40,000-draw results of the shipped procedure (tests use 20,000 draws):
      DGP1  M=0, delta=0 (parallel trends exact), Sigma = BC:               0.9494
      DGP2  M=0, delta=0, diag(se_pre=0.15, se_post=0.05):                  0.9496
      DGP3  M=0, delta=0, diag(se=0.05):                                    0.9502
      DGP4  M=0.1, linear pre-trend 0.2, delta_post0 = 0.3 (boundary):     0.9683
    Old code (audit coverage_mc.py / g_coverage_bc.py): 0.912, 0.758, 0.925 and 0.933.
    The FLCI weights depend only on Sigma, so the family is built once and
    applied to every draw exactly as honest_did does.
    """
    rng = np.random.default_rng(0)
    n = 20000
    years = np.array(BC_YEARS, dtype=float)
    tau = np.array([0, 0, 0, 0, 0.196, 0.31, 0.24, 0.13])
    dgps = [
        ("BC Sigma", BC_SIGMA),
        ("diag pre .15 post .05", np.diag(np.r_[np.full(4, 0.15**2), np.full(4, 0.05**2)])),
        ("diag .05", np.diag(np.full(8, 0.05**2))),
    ]
    for label, sig in dgps:
        win = _build_window(years, tau, sig, 2008.0)
        fr = _SDFrontier(win.sigma_nb, np.array([1.0, 0, 0, 0]), win.pre_idx, win.post_idx, win.tvec)
        draws = rng.multivariate_normal(tau, sig, size=n)
        half = fr.S * _folded_normal_quantile(0.0 * fr.B / fr.S, 0.05)
        k = int(np.argmin(half))
        centers = draws @ fr.W[k]
        cover = float(np.mean(np.abs(centers - 0.196) <= half[k]))
        assert cover >= 0.94, f"{label}: coverage {cover:.4f}"
        # the public function must use the very same interval
        bh = draws[0]
        r = honest_did(bh, sigma=sig, event_time=years, base_period=2008, method="smoothness", m_vec=[0.0], target_horizon=2009)
        assert r.table.iloc[0]["ci_lo"] == pytest.approx(centers[0] - half[k], abs=1e-12)
        assert r.table.iloc[0]["ci_hi"] == pytest.approx(centers[0] + half[k], abs=1e-12)

    # DGP4: truth on the boundary of Delta^SD(0.1)
    years5 = np.array([-5, -4, -3, -2, 0], dtype=float)
    theta = 0.5
    delta5 = np.array([0.2 * (t + 1) for t in [-5, -4, -3, -2]] + [0.2 + 0.1])
    tau5 = np.array([0, 0, 0, 0, theta])
    sig5 = np.diag(np.full(5, 0.01))
    win = _build_window(years5, tau5, sig5, -1.0)
    fr = _SDFrontier(win.sigma_nb, np.array([1.0]), win.pre_idx, win.post_idx, win.tvec)
    draws = rng.multivariate_normal(tau5 + delta5, sig5, size=n)
    half = fr.S * _folded_normal_quantile(0.1 * fr.B / fr.S, 0.05)
    k = int(np.argmin(half))
    cover = float(np.mean(np.abs(draws @ fr.W[k] - theta) <= half[k]))
    assert cover >= 0.94, f"boundary DGP: coverage {cover:.4f}"


def test_coverage_monte_carlo_relative_magnitude():
    """Coverage of the 95% delta-method IM interval under Delta^RM(1) with the truth on the boundary.

    40,000-draw results (tests use 10,000): first differences 0.9792, level bound
    0.9589, tied benchmark with interior truth 1.0000.  Old code (audit
    g_coverage_bc.py, level bound, se of the point estimate for both ends): 0.888.
    """
    rng = np.random.default_rng(1)
    n = 10000
    years5 = np.array([-5, -4, -3, -2, 0], dtype=float)
    theta = 0.5
    sig5 = np.diag(np.full(5, 0.01))
    dpre = np.array([0.3, 0, 0, 0])
    truth = np.r_[dpre, theta + 0.3]  # benchmark 0.3, delta_post = 0.3 on the boundary
    win = _build_window(years5, truth, sig5, -1.0)
    draws = rng.multivariate_normal(truth, sig5, size=n)
    for bound in ["first_difference", "levels"]:
        cover = 0
        for bh in draws:
            dk = np.r_[bh[:4], 0.0]
            lo, hi, slo, shi = _rm_bounds(np.array([1.0]), bh, sig5, dk, win.known_map, win.post_idx, 1.0, bound)
            clo, chi = _imbens_manski_interval(lo, hi, slo, shi, 0.05)
            cover += clo <= theta <= chi
        assert cover / n >= 0.94, f"{bound}: coverage {cover / n:.4f}"
    # the public function must use the very same interval
    bh = draws[0]
    r = honest_did(bh, sigma=sig5, event_time=years5, method="relative_magnitude", m_vec=[1.0], target_horizon=0)
    dk = np.r_[bh[:4], 0.0]
    lo, hi, slo, shi = _rm_bounds(np.array([1.0]), bh, sig5, dk, win.known_map, win.post_idx, 1.0, "first_difference")
    clo, chi = _imbens_manski_interval(lo, hi, slo, shi, 0.05)
    assert r.table.iloc[0]["ci_lo"] == pytest.approx(clo, abs=1e-12)
    assert r.table.iloc[0]["ci_hi"] == pytest.approx(chi, abs=1e-12)


# =============================================================================
# Breakdown value
# =============================================================================


def test_breakdown_sign_taken_from_m0_confidence_set():
    """M* is found when the M=0 set has the opposite sign of the raw estimate (audit C5 / M109).

    Old code took the sign from orig_theta (+0.2) and tracked only the lower CI
    bound, returned inf and printed 'remains significant across all tested M'
    although the table flipped between M=0.1 and M=0.2.
    """
    et = [-5, -4, -3, -2, 0, 1, 2]
    se = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    b = np.array([-2.0, -1.5, -1.0, -0.5, 0.2, 0.3, 0.4])  # beta_t = 0.5 (t+1): extrapolated effect -0.3
    res = honest_did(b, se=se, event_time=et, method="smoothness", m_vec=[0.0, 0.1, 0.2, 0.3], target_horizon=0)
    tbl = res.table
    assert tbl.iloc[0]["id_lo"] == pytest.approx(-0.3)
    assert bool(tbl.iloc[0]["significant"]) is True
    assert bool(tbl.iloc[2]["significant"]) is False
    assert np.isfinite(res.breakdown_value)
    assert 0.1 < res.breakdown_value < 0.2
    at = honest_did(b, se=se, event_time=et, method="smoothness", m_vec=[res.breakdown_value], target_horizon=0)
    assert abs(at.table.iloc[0]["ci_hi"]) < 1e-6
    assert "remains statistically significant for all" not in res.summary()
    # mirror case with a negative raw estimate
    res_neg = honest_did(-b, se=se, event_time=et, method="smoothness", m_vec=[0.0, 0.3], target_horizon=0)
    assert res_neg.breakdown_value == pytest.approx(res.breakdown_value, abs=1e-8)


def test_honest_did_already_insignificant():
    event_time = [-2, -1, 0]
    beta = [0.1, 0.0, 0.05]
    se = [0.1, 0.0, 0.5]
    sens = honest_did_sensitivity(event_time=event_time, beta=beta, se=se, target_horizon=0)
    assert sens.breakdown_value == 0.0
    assert "already not statistically distinguishable" in sens.summary()


def test_breakdown_inf_when_no_pre_trend_under_rm():
    """A zero pre-trend makes Delta^RM(Mbar) = {delta_post = 0} for every Mbar: M* = inf, benchmark 0.0.

    Old code floored the benchmark at 1e-6 'to prevent division by zero'.
    """
    res = honest_did([0.0, 0.0, 0.0, 1.8, 2.2, 2.5], event_time=[-3, -2, -1, 0, 1, 2], se=[0.1] * 6, method="relative_magnitude", m_vec=[0.0, 1.0, 2.0])
    assert res.pre_trend_max == 0.0
    assert np.all(res.table["id_lo"] == 1.8) and np.all(res.table["id_hi"] == 1.8)
    assert np.isinf(res.breakdown_value)
    assert res.m_search_max > 100.0
    assert "significant for all" in res.summary()
    assert "inf" in res.summary()


# =============================================================================
# Input handling and validation
# =============================================================================


def test_pre_post_periods_integer_syntax_keeps_last_pre_coefficient():
    """pre_periods=3 must map the coefficients to [-4,-3,-2] with -1 omitted (audit C44).

    Old code used arange(-3, 0) = [-3, -2, -1] and then dropped -1 as the base,
    silently discarding b_hat[2] = 0.40 (benchmark 0.05 instead of 0.42).
    """
    b_hat = [0.05, -0.02, 0.40, 1.2, 1.5]
    se = [0.05] * 5
    res = honest_did(b_hat=b_hat, se=se, pre_periods=3, post_periods=2, method="relative_magnitude", m_vec=[0.0, 1.0])
    ref = honest_did(b_hat=b_hat, se=se, event_time=[-4, -3, -2, 0, 1], method="relative_magnitude", m_vec=[0.0, 1.0])
    # first differences: -0.07, 0.42, -0.40 -> benchmark 0.42; h=0 at Mbar=1 -> [0.78, 1.62]
    assert res.pre_trend_max == pytest.approx(0.42)
    assert res.table.iloc[1]["id_lo"] == pytest.approx(0.78)
    assert res.table.iloc[1]["id_hi"] == pytest.approx(1.62)
    pd.testing.assert_frame_equal(res.table, ref.table)
    # base included (len = pre + 1 + post)
    res_b = honest_did(b_hat=[0.05, -0.02, 0.40, 0.0, 1.2, 1.5], se=[0.05] * 6, pre_periods=3, post_periods=2, method="relative_magnitude", m_vec=[0.0, 1.0])
    pd.testing.assert_frame_equal(res_b.table, ref.table)
    # 5 coefficients match neither 2 + 1 (pre + post) nor 2 + 1 + 1 (base included)
    with pytest.raises(ValueError, match="matches neither"):
        honest_did(b_hat=b_hat, se=se, pre_periods=2, post_periods=1, method="relative_magnitude")
    with pytest.raises(ValueError, match="base_period"):
        honest_did(b_hat=b_hat, se=se, pre_periods=3, post_periods=2, base_period=-2)


def test_no_event_time_raises_instead_of_guessing():
    """honest_did(b_hat, se=se) without event times must not split the vector in half (audit M10)."""
    b = [-0.8, -0.6, -0.4, -0.2, 1.0, 1.2, 1.4]
    with pytest.raises(ValueError, match="event times are required"):
        honest_did(b, se=[0.1] * 7)


def test_unknown_keyword_arguments_raise():
    """Misspelled parameters raise TypeError instead of being swallowed (audit M11 / M111)."""
    b = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    et = [-3, -2, -1, 0, 1, 2]
    with pytest.raises(TypeError):
        honest_did(b, se=se, event_time=et, mvec=[0.0, 9.0])
    with pytest.raises(TypeError):
        honest_did(b, se=se, event_time=et, Mbar=3.0)
    with pytest.raises(TypeError):
        honest_did_sensitivity(event_time=et, beta=b, se=se, alpah=0.5)


def test_honest_did_sensitivity_honours_alpha():
    """honest_did_sensitivity(alpha=0.10) must give a 90% interval (audit M110)."""
    et = [-3, -2, -1, 0, 1, 2]
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    ra = honest_did_sensitivity(event_time=et, beta=beta, se=se, target_horizon=0, m_grid=[0.0], alpha=0.10)
    rb = honest_did(b_hat=beta, se=se, event_time=et, method="relative_magnitude", m_vec=[0.0], target_horizon=0, alpha=0.10)
    rc = honest_did_sensitivity(event_time=et, beta=beta, se=se, target_horizon=0, m_grid=[0.0])
    assert ra.ci == pytest.approx(0.90)
    assert ra.table.iloc[0]["ci_lo"] == pytest.approx(rb.table.iloc[0]["ci_lo"])
    assert ra.table.iloc[0]["ci_lo"] > rc.table.iloc[0]["ci_lo"]
    assert rc.ci == pytest.approx(0.95)
    with pytest.raises(ValueError, match="inconsistent"):
        honest_did_sensitivity(event_time=et, beta=beta, se=se, target_horizon=0, alpha=0.10, ci=0.95)
    with pytest.raises(ValueError, match="inconsistent"):
        honest_did(b_hat=beta, se=se, event_time=et, alpha=0.01, ci=0.90)
    assert honest_did(b_hat=beta, se=se, event_time=et, alpha=0.10, ci=0.90, m_vec=[0.0]).ci == pytest.approx(0.90)


def test_bound_and_grid_validation():
    """Unknown bound strings and negative M raise (audit M112 / finding 9)."""
    b = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    et = [-3, -2, -1, 0, 1, 2]
    with pytest.raises(ValueError, match="bound must be"):
        honest_did(b, se=se, event_time=et, method="relative_magnitude", bound="deviaton from pre-trend slope")
    with pytest.raises(ValueError, match="bound must be"):
        honest_did(b, se=se, event_time=et, method="relative_magnitude", bound="bogus")
    with pytest.raises(ValueError, match="only meaningful"):
        honest_did(b, se=se, event_time=et, method="smoothness", bound="levels")
    with pytest.raises(ValueError, match="non-negative"):
        honest_did(b, se=se, event_time=et, method="relative_magnitude", m_vec=[-1.0, 0.0])
    with pytest.raises(ValueError, match="at least one value"):
        honest_did(b, se=se, event_time=et, m_vec=[])


def test_sigma_shape_validation():
    """sigma of a shape matching neither (n, n) nor the post block raises (audit M112)."""
    b = np.array([-0.8, -0.6, -0.4, -0.2, 1.0, 1.2, 1.4])
    se = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.3])
    et = [-5, -4, -3, -2, 0, 1, 2]
    with pytest.raises(ValueError, match="sigma has shape"):
        honest_did(b, sigma=np.eye(5), se=se, event_time=et, method="smoothness", m_vec=[0.0])
    with pytest.raises(ValueError, match="sigma has shape"):
        honest_did(b, sigma=np.eye(5), event_time=et, method="smoothness", m_vec=[0.0])
    with pytest.raises(ValueError, match="together with se"):
        honest_did(b, sigma=np.diag([0.04, 0.09, 0.16]), event_time=et, method="smoothness", m_vec=[0.0])
    # (L, L) post block together with se is accepted and used for the post block
    r = honest_did(b, sigma=np.diag([0.04, 0.09, 0.16]), se=se, event_time=et, method="smoothness", m_vec=[0.0], target_horizon=1)
    assert r.table.iloc[0]["orig_se"] == pytest.approx(0.3)


def test_l_vec_is_labelled_honestly():
    """A custom contrast is labelled 'l_vec', and l_vec cannot be combined with target_horizon."""
    et = [-3, -2, -1, 0, 1]
    beta = [0.1, -0.05, 0.0, 1.2, 1.5]
    se = [0.1, 0.1, 0.0, 0.2, 0.2]
    r = honest_did(b_hat=beta, se=se, event_time=et, method="relative_magnitude", l_vec=[0.0, 1.0], m_vec=[0.0])
    assert r.target_horizons == ["l_vec"]
    assert list(r.table["horizon"]) == ["l_vec"]
    assert "h = l_vec" in r.summary()
    assert r.table.iloc[0]["orig_estimate"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="not both"):
        honest_did(b_hat=beta, se=se, event_time=et, method="relative_magnitude", l_vec=[0.0, 1.0], target_horizon=1)
    with pytest.raises(ValueError, match="l_vec must have length"):
        honest_did(b_hat=beta, se=se, event_time=et, l_vec=[1.0, 0.0, 0.0])
    ax = r.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_base_period_minus_two_handled_chronologically():
    """base_period=-2 with -1 present: -1 is a known period between the reference and the post block.

    Old code paired the first differences (-3,-1),(-1,-2) non-chronologically and
    skipped delta_{-1} in the Delta^SD chain (M=0.2: [0.2, 1.0] vs paper [0.4, 0.8]).
    """
    b_full = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 1.0, 1.2, 1.4])
    et = [-5, -4, -3, -2, -1, 0, 1, 2]
    se = np.array([0.1, 0.1, 0.1, 0.0, 0.1, 0.2, 0.25, 0.3])
    for M, (lo_ref, hi_ref) in {0.0: (0.6, 0.6), 0.2: (0.4, 0.8), 0.5: (0.1, 1.1)}.items():
        row = honest_did(b_full, se=se, event_time=et, method="smoothness", m_vec=[M], target_horizon=0, base_period=-2).table.iloc[0]
        assert row["id_lo"] == pytest.approx(lo_ref)
        assert row["id_hi"] == pytest.approx(hi_ref)
    # Delta^RM: benchmark over the chronological known block (-3, -2=base, -1): |0-0.30| = 0.30, |0.10-0| = 0.10
    # first post difference is delta_0 - delta_{-1} -> theta in [1.0 - 0.10 - 0.30, 1.0 - 0.10 + 0.30] = [0.6, 1.2]
    r = honest_did([0.30, 0.0, 0.10, 1.0], se=[0.05, 0.0, 0.05, 0.10], event_time=[-3, -2, -1, 0], base_period=-2, method="relative_magnitude", m_vec=[1.0], target_horizon=0)
    assert r.pre_trend_max == pytest.approx(0.30)
    assert r.table.iloc[0]["id_lo"] == pytest.approx(0.6)
    assert r.table.iloc[0]["id_hi"] == pytest.approx(1.2)


def test_unsorted_input_and_permuted_sigma_give_identical_results():
    """Event times may arrive in any order; sigma is permuted consistently."""
    et = np.array([-5, -4, -3, -2, 0, 1, 2])
    b0 = np.array([-0.8, -0.6, -0.4, -0.2, 1.0, 1.2, 1.4])
    S = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.3]) ** 2)
    S[4, 5] = S[5, 4] = 0.02
    perm = np.array([6, 0, 5, 1, 4, 2, 3])
    for meth in ["smoothness", "relative_magnitude"]:
        a = honest_did(b0, sigma=S, event_time=et, method=meth, m_vec=[0.0, 0.2], target_horizon=1)
        b = honest_did(b0[perm], sigma=S[np.ix_(perm, perm)], event_time=et[perm], method=meth, m_vec=[0.0, 0.2], target_horizon=1)
        pd.testing.assert_frame_equal(a.table, b.table)
        assert a.breakdown_value == pytest.approx(b.breakdown_value)
    with pytest.raises(ValueError, match="duplicate"):
        honest_did(b0, sigma=S, event_time=[-5, -4, -3, -2, 0, 1, 1], method="smoothness")


def test_reference_coefficient_is_normalised_with_a_warning():
    """A non-zero coefficient at the reference period is normalised to 0 and warned about."""
    b = [0.1, 0.3, 1.0, 1.2]
    with pytest.warns(UserWarning, match="reference period"):
        r = honest_did(b, se=[0.1, 0.1, 0.1, 0.1], event_time=[-2, -1, 0, 1], method="relative_magnitude", m_vec=[1.0])
    r0 = honest_did([0.1, 0.0, 1.0, 1.2], se=[0.1, 0.0, 0.1, 0.1], event_time=[-2, -1, 0, 1], method="relative_magnitude", m_vec=[1.0])
    pd.testing.assert_frame_equal(r.table, r0.table)


def test_honest_did_validation_errors():
    with pytest.raises(ValueError, match="method must be"):
        honest_did_sensitivity(event_time=[-1, 0], beta=[0, 1], se=[0, 0.1], method="unknown")
    with pytest.raises(ValueError, match="must provide either 'result'"):
        honest_did_sensitivity()
    with pytest.raises(ValueError, match="at least one pre-treatment period"):
        honest_did_sensitivity(event_time=[-1, 0, 1], beta=[0.0, 1.0, 1.2], se=[0.0, 0.1, 0.1], base_period=-1)
    with pytest.raises(ValueError, match="equal lengths"):
        honest_did_sensitivity(event_time=[-2, -1, 0], beta=[0.1, 0.0], se=[0.1, 0.1])
    with pytest.raises(ValueError, match="must provide either 'se' or 'sigma'"):
        honest_did([0.1, 0.0, 1.0], event_time=[-2, -1, 0])
    with pytest.raises(ValueError, match="not found in post-treatment"):
        honest_did([0.1, 0.0, 1.0], se=[0.1, 0.0, 0.1], event_time=[-2, -1, 0], target_horizon=3)


# =============================================================================
# Result objects, presentation and integrations
# =============================================================================


def test_honest_did_with_callaway_santanna_result():
    rng = np.random.default_rng(42)
    units = []
    for u in range(16):
        treat_yr = 2012 if u < 6 else (2014 if u < 12 else np.nan)
        for yr in range(2009, 2016):
            d = 1.0 if not np.isnan(treat_yr) and yr >= treat_yr else 0.0
            y = 3.0 * d + 0.2 * (yr - 2009) + rng.standard_normal() * 0.5
            units.append({"unit": f"U{u}", "year": yr, "treat_time": treat_yr, "outcome": y})

    df_panel = pd.DataFrame(units)
    cs_res = callaway_santanna(df_panel, unit="unit", time="year", outcome="outcome", treat_time="treat_time")

    sens = honest_did_sensitivity(cs_res, target_horizon=0, ci=0.90)
    assert isinstance(sens, HonestDiDResult)
    assert sens.target_horizons == [0]
    assert sens.ci == pytest.approx(0.90)
    assert len(sens.table) > 0

    summ = sens.summary()
    assert "Honest DiD Sensitivity Analysis" in summ
    assert "Breakdown Value" in summ
    assert "|" in sens.to_markdown()
    assert "\\begin{tabular}" in sens.to_latex()
    assert "#table" in sens.to_typst()
    ascii_art = sens.plot_ascii()
    assert "Honest DiD Confidence Intervals vs M" in ascii_art
    assert "|" in ascii_art
    # smoothness on the same object: FLCI finite even if the plug-in set is empty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = honest_did(cs_res, method="smoothness", target_horizon=[0, 1])
    assert np.all(np.isfinite(sd.table["ci_lo"]))
    assert isinstance(sd.breakdown_value, dict)


def test_honest_did_smoothness_method():
    event_time = [-3, -2, -1, 0, 1]
    beta = [0.1, -0.05, 0.0, 1.2, 1.5]
    se = [0.1, 0.1, 0.0, 0.2, 0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sens = honest_did_sensitivity(event_time=event_time, beta=beta, se=se, target_horizon=[0, 1], method="smoothness", m_grid=[0.0, 0.1, 0.2])
    assert sens.method == "smoothness"
    assert sens.bound == "second_difference"
    assert sens.ci_method == "flci"
    assert sens.pre_trend_slope == pytest.approx(0.05)  # delta_{-1} - delta_{-2} = 0 - (-0.05)
    assert sens.pre_trend_max_second_diff == pytest.approx(0.2)
    assert isinstance(sens.breakdown_value, dict)
    assert set(sens.breakdown_value) == {0, 1}
    assert all(np.isfinite(v) and v > 0 for v in sens.breakdown_value.values())
    # FLCI half-length is non-decreasing in M for each horizon
    for h in [0, 1]:
        sub = sens.table[sens.table["horizon"] == h]
        assert (np.diff(sub["ci_hi"] - sub["ci_lo"]) >= -1e-10).all()


def test_honest_did_primary_api():
    b_hat = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    et = [-3, -2, -1, 0, 1, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_sd = honest_did(b_hat=b_hat, se=se, event_time=et, method="smoothness", m_vec=[0.0, 0.1, 0.2], base_period=-1, alpha=0.05)
    assert isinstance(res_sd, HonestDiDResult)
    assert res_sd.method == "smoothness"
    assert res_sd.ci == 0.95
    assert len(res_sd.table) == 3
    assert np.isfinite(res_sd.breakdown_value)
    assert res_sd.breakdown_value > 0.0
    res_rm = honest_did(b_hat=b_hat, se=se, event_time=et, method="relative_magnitude", m_vec=[0.0, 0.5, 1.0, 2.0], base_period=-1, alpha=0.05)
    assert res_rm.method == "relative_magnitude"
    assert 2.4 < res_rm.breakdown_value < 2.6
    assert "Honest DiD Sensitivity Analysis" in res_sd.summary()
    assert "FLCI" in res_sd.summary()
    assert "Imbens" in res_rm.summary()
    for m in ["SD", "DeltaRM", "rm", "sd"]:
        assert honest_did(b_hat=b_hat, se=se, event_time=et, method=m, m_vec=[0.5]).method in ("smoothness", "relative_magnitude")


def test_honest_did_covariance_matrix_sigma():
    """orig_se uses the post block of Sigma, including covariances through l_vec."""
    b_hat = np.array([0.05, 0.0, 1.2, 1.6])
    event_time = [-2, -1, 0, 1]
    cov = np.array(
        [
            [0.04, 0.01, 0.00, 0.00],
            [0.01, 0.01, 0.00, 0.00],
            [0.00, 0.00, 0.04, 0.02],
            [0.00, 0.00, 0.02, 0.09],
        ]
    )
    res_h0 = honest_did(b_hat=b_hat, sigma=cov, event_time=event_time, method="smoothness", target_horizon=0, m_vec=[0.0, 0.1])
    assert res_h0.table["orig_se"].iloc[0] == pytest.approx(np.sqrt(0.04), rel=1e-5)
    res_avg = honest_did(b_hat=b_hat, sigma=cov, event_time=event_time, method="smoothness", l_vec=[0.5, 0.5], m_vec=[0.0, 0.1])
    assert res_avg.table["orig_estimate"].iloc[0] == pytest.approx(0.5 * 1.2 + 0.5 * 1.6)
    assert res_avg.table["orig_se"].iloc[0] == pytest.approx(np.sqrt(0.0425), rel=1e-4)
    # single pre period: FLCI center is beta_0 + beta_{-2} with se from the full Sigma
    row = res_h0.table.iloc[0]
    assert 0.5 * (row["ci_lo"] + row["ci_hi"]) == pytest.approx(1.25)
    assert 0.5 * (row["ci_hi"] - row["ci_lo"]) == pytest.approx(Z975 * np.sqrt(0.04 + 0.04), rel=1e-6)


def test_honest_did_plot_method():
    event_time = [-3, -2, -1, 0, 1, 2]
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]
    res = honest_did(b_hat=beta, se=se, event_time=event_time, method="relative_magnitude", m_vec=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0], target_horizon=0)

    ax = res.plot()
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    plt.close("all")

    fig, ax2 = res.plot(return_fig=True, title="Custom Test Title")
    assert isinstance(fig, plt.Figure)
    assert ax2.get_title() == "Custom Test Title"
    assert len(ax2.collections) >= 2
    assert any(line.get_linestyle() == ":" for line in ax2.lines)
    plt.close("all")

    fig_custom, ax_custom = plt.subplots(figsize=(10, 6))
    assert res.plot(ax=ax_custom) is ax_custom
    plt.close("all")

    # unknown styling kwargs are rejected instead of being silently ignored (audit M9)
    with pytest.raises(TypeError):
        res.plot(linewidth=9.0)
    with pytest.raises(ValueError, match="No data found"):
        res.plot(horizon=99)

    # smoothness with an empty plug-in set at some M still plots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = honest_did(b_hat=beta, se=se, event_time=event_time, method="smoothness", m_vec=[0.0, 0.2, 0.4, 0.6], target_horizon=0)
    assert isinstance(sd.plot(), plt.Axes)
    assert "(empty)" in sd.plot_ascii()
    plt.close("all")


def test_honest_did_multi_horizon_evaluation():
    event_time = [-2, -1, 0, 1, 2]
    beta = [0.1, 0.0, 1.2, 1.4, 1.6]
    se = [0.1, 0.0, 0.2, 0.25, 0.3]
    res = honest_did(b_hat=beta, se=se, event_time=event_time, method="smoothness", target_horizon=[0, 1, 2], m_vec=[0.0, 0.1])
    assert isinstance(res.breakdown_value, dict)
    assert set(res.breakdown_value.keys()) == {0, 1, 2}
    for h in [0, 1, 2]:
        assert res.breakdown_value[h] > 0.0
    # later horizons accumulate more bias per unit of M -> smaller breakdown values
    assert res.breakdown_value[0] > res.breakdown_value[1] > res.breakdown_value[2]
    tbl = res.to_frame()
    assert len(tbl) == 6
    assert set(tbl["horizon"].unique()) == {0, 1, 2}
    assert isinstance(res.plot(horizon=1), plt.Axes)
    plt.close("all")


def test_imbens_manski_critical_value_and_folded_normal_quantile():
    for w in [0.0, 0.5, 1.0, 2.0, 5.0]:
        c = _imbens_manski_critical_value(w, 0.05, is_half_width=False)
        assert norm.cdf(c + w) - norm.cdf(-c) == pytest.approx(0.95, abs=1e-8)
    assert _imbens_manski_critical_value(0.0, 0.05, is_half_width=False) == pytest.approx(Z975)
    t = np.array([0.0, 0.3, 1.0, 4.0])
    q = _folded_normal_quantile(t, 0.05)
    assert np.allclose(norm.cdf(q - t) - norm.cdf(-q - t), 0.95, atol=1e-10)
    assert q[0] == pytest.approx(Z975)
    # q(t) = t + c_IM(2t): the FLCI and IM critical values are the same equation
    assert q[2] == pytest.approx(1.0 + _imbens_manski_critical_value(2.0, 0.05, is_half_width=False), abs=1e-8)


# =============================================================================
# Benzarti & Carloni (2019) application of Rambachan & Roth (2023, Section 6.1)
# =============================================================================


def test_honest_did_benzarti_carloni_2019():
    """Benzarti & Carloni (2019) VAT-cut event study (RR 2023, Section 6.1), 2009 effect.

    Reference period 2008, pre 2004-2007, post 2009-2012.  All reference numbers
    below are hand-derived from RR's definitions and the delta method; RR's own
    figure uses their conditional/hybrid confidence sets, which this package
    does not implement, so the breakdown values are not RR's exact numbers.
    """
    se_vec = np.sqrt(np.diag(BC_SIGMA))
    b_2009, se_2009 = BC_BETAHAT[4], se_vec[4]
    assert b_2009 == pytest.approx(0.1960, abs=1e-3)
    assert se_2009 == pytest.approx(0.0190, abs=1e-3)

    # ---- Delta^RM(Mbar), first differences: benchmark = |beta_2007 - beta_2006| = 0.07949
    pre_full = np.r_[BC_BETAHAT[:4], 0.0]
    dmax = float(np.max(np.abs(np.diff(pre_full))))
    assert dmax == pytest.approx(0.079488, abs=1e-6)
    res_rm = honest_did(
        b_hat=BC_BETAHAT, sigma=BC_SIGMA, event_time=BC_YEARS, base_period=BC_REF_YEAR,
        method="relative_magnitude", m_vec=[0.0, 0.5, 1.0, 1.5, 2.0], target_horizon=2009, alpha=0.05,
    )
    assert res_rm.pre_trend_max == pytest.approx(dmax)
    tbl = res_rm.to_frame()
    row0 = tbl[tbl["M"] == 0.0].iloc[0]
    assert row0["id_lo"] == pytest.approx(b_2009) and row0["id_hi"] == pytest.approx(b_2009)
    assert row0["ci_lo"] == pytest.approx(b_2009 - Z975 * se_2009, abs=1e-8)

    # delta method: theta_lo = beta_2009 - Mbar*(beta_2007 - beta_2006)
    def hand_ci(mbar):
        g_lo = np.zeros(8)
        g_lo[4] = 1.0
        g_lo[3] -= mbar
        g_lo[2] += mbar
        g_hi = np.zeros(8)
        g_hi[4] = 1.0
        g_hi[3] += mbar
        g_hi[2] -= mbar
        se_lo = np.sqrt(g_lo @ BC_SIGMA @ g_lo)
        se_hi = np.sqrt(g_hi @ BC_SIGMA @ g_hi)
        lo, hi = b_2009 - mbar * dmax, b_2009 + mbar * dmax
        c = im_critical(hi - lo, max(se_lo, se_hi))
        return lo - c * se_lo, hi + c * se_hi

    for mbar in [0.5, 1.0, 1.5, 2.0]:
        row = tbl[tbl["M"] == mbar].iloc[0]
        assert row["id_lo"] == pytest.approx(b_2009 - mbar * dmax, abs=1e-9)
        assert row["id_hi"] == pytest.approx(b_2009 + mbar * dmax, abs=1e-9)
        lo, hi = hand_ci(mbar)
        assert row["ci_lo"] == pytest.approx(lo, abs=1e-8)
        assert row["ci_hi"] == pytest.approx(hi, abs=1e-8)
    assert bool(tbl[tbl["M"] == 1.5].iloc[0]["significant"]) is True
    assert bool(tbl[tbl["M"] == 2.0].iloc[0]["significant"]) is False
    m_star_hand = scipy.optimize.brentq(lambda m: hand_ci(m)[0], 0.5, 5.0)
    assert res_rm.breakdown_value == pytest.approx(m_star_hand, abs=1e-6)
    assert 1.6 < res_rm.breakdown_value < 1.65

    # ---- level bound (explicit option): benchmark max|beta_pre| = 0.0730, M* ~ 1.95
    res_lv = honest_did(
        b_hat=BC_BETAHAT, sigma=BC_SIGMA, event_time=BC_YEARS, base_period=BC_REF_YEAR,
        method="relative_magnitude", bound="levels", m_vec=[1.0], target_horizon=2009,
    )
    assert res_lv.pre_trend_max == pytest.approx(0.073015, abs=1e-6)
    assert res_lv.table.iloc[0]["id_lo"] == pytest.approx(b_2009 - 0.073015, abs=1e-6)
    assert 1.9 < res_lv.breakdown_value < 2.0

    # ---- Delta^SD(M) with the FLCI
    # pre-period second differences: max |.| = 0.1525 -> plug-in set empty for M < 0.1525
    with pytest.warns(UserWarning, match="identified set is empty"):
        res_sd = honest_did(
            b_hat=BC_BETAHAT, sigma=BC_SIGMA, event_time=BC_YEARS, base_period=BC_REF_YEAR,
            method="smoothness", m_vec=[0.0, 0.05, 0.10, 0.15, 0.20], target_horizon=2009, alpha=0.05,
        )
    assert res_sd.pre_trend_max_second_diff == pytest.approx(0.152503, abs=1e-6)
    assert res_sd.pre_trend_slope == pytest.approx(-BC_BETAHAT[3])
    tsd = res_sd.to_frame()
    assert np.isnan(tsd[tsd["M"] == 0.10].iloc[0]["id_lo"])
    r20 = tsd[tsd["M"] == 0.20].iloc[0]
    assert r20["id_lo"] == pytest.approx(b_2009 + BC_BETAHAT[3] - 0.20)  # delta_2009 in [-beta_2007 - M, -beta_2007 + M]
    assert r20["id_hi"] == pytest.approx(b_2009 + BC_BETAHAT[3] + 0.20)

    # M = 0: minimum-variance estimator unbiased under any linear trend (GLS detrending, all 4 pre periods)
    tpre = np.array([-4.0, -3.0, -2.0, -1.0])
    S_pp, S_p9 = BC_SIGMA[:4, :4], BC_SIGMA[:4, 4]
    kkt = np.block([[2 * S_pp, tpre[:, None]], [tpre[None, :], np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.r_[-2 * S_p9, -1.0])
    w = np.r_[sol[:4], 1.0, 0.0, 0.0, 0.0]
    r0 = tsd[tsd["M"] == 0.0].iloc[0]
    assert 0.5 * (r0["ci_lo"] + r0["ci_hi"]) == pytest.approx(w @ BC_BETAHAT, abs=1e-6)
    assert 0.5 * (r0["ci_hi"] - r0["ci_lo"]) == pytest.approx(Z975 * np.sqrt(w @ BC_SIGMA @ w), rel=1e-4)

    # M >= 0.05: the minimum-bias estimator beta_2009 + beta_2007 (worst-case bias M) is shortest
    theta_lp = b_2009 + BC_BETAHAT[3]
    s_lp = np.sqrt(BC_SIGMA[4, 4] + BC_SIGMA[3, 3] + 2 * BC_SIGMA[4, 3])
    for M in [0.10, 0.15, 0.20]:
        row = tsd[tsd["M"] == M].iloc[0]
        half = s_lp * _folded_normal_quantile(np.array([M / s_lp]), 0.05)[0]
        assert row["ci_lo"] == pytest.approx(theta_lp - half, abs=1e-8)
        assert row["ci_hi"] == pytest.approx(theta_lp + half, abs=1e-8)
        assert bool(row["significant"]) is True
    m_star_sd = scipy.optimize.brentq(lambda m: theta_lp - s_lp * _folded_normal_quantile(np.array([m / s_lp]), 0.05)[0], 0.05, 1.0)
    assert res_sd.breakdown_value == pytest.approx(m_star_sd, abs=1e-6)
    assert 0.21 < res_sd.breakdown_value < 0.23
