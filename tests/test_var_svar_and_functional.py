"""Comprehensive tests for Track 3 (puremacro.var.svar) and Track 4b (puremacro.var.functional)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.var.functional import FunctionalVARResult, estimate_functional_var
from puremacro.var.svar import (
    SVARResult,
    svar_heteroskedastic,
    svar_non_gaussian,
    svar_proxy,
    svar_sign_narrative,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _simulate_svar_dgp(
    T: int,
    A1: np.ndarray,
    B0: np.ndarray,
    shocks: np.ndarray,
) -> np.ndarray:
    """Simulate a VAR(1) process y_t = A1 y_{t-1} + B0 eps_t."""
    n = B0.shape[0]
    Y = np.zeros((T, n), dtype=float)
    for t in range(1, T):
        Y[t] = A1 @ Y[t - 1] + B0 @ shocks[t]
    return Y


def _assert_presentation_standard(res: object) -> None:
    """Verify the 7-method Presentation Standard 2.0+ contract on a result object."""
    sm = res.summary()  # type: ignore[attr-defined]
    assert isinstance(sm, pd.DataFrame) and not sm.empty

    df1 = res.to_dataframe()  # type: ignore[attr-defined]
    df2 = res.to_frame()  # type: ignore[attr-defined]
    assert isinstance(df1, pd.DataFrame) and not df1.empty
    pd.testing.assert_frame_equal(df1, df2)

    md = res.to_markdown()  # type: ignore[attr-defined]
    assert isinstance(md, str) and "|" in md

    tex = res.to_latex()  # type: ignore[attr-defined]
    assert isinstance(tex, str) and "\\begin{tabular}" in tex

    typ = res.to_typst()  # type: ignore[attr-defined]
    assert isinstance(typ, str) and "#table(" in typ

    fig = res.plot(show=False)  # type: ignore[attr-defined]
    assert fig is not None


# ---------------------------------------------------------------------------
# Track 3: svar_heteroskedastic
# ---------------------------------------------------------------------------


def test_svar_heteroskedastic_recovers_b0_and_wald():
    """svar_heteroskedastic recovers true structural B0 on a 2-regime volatility DGP."""
    rng = np.random.default_rng(123)
    T = 1200
    t_break = 600
    A1 = np.array([[0.5, 0.1], [-0.1, 0.4]])
    B0_true = np.array([[1.0, 0.3], [-0.4, 1.1]])

    eps = rng.standard_normal((T, 2))
    # Regime 2 scales shock 0 std by 2.5 (variance x 6.25) and shock 1 std by 0.6 (variance x 0.36)
    eps[t_break:, 0] *= 2.5
    eps[t_break:, 1] *= 0.6

    Y = _simulate_svar_dgp(T, A1, B0_true, eps)
    res = svar_heteroskedastic(
        Y,
        p=1,
        regime_indicator=t_break,
        horizon=12,
        var_names=("output", "inflation"),
    )

    assert isinstance(res, SVARResult)
    assert res.method == "heteroskedastic"
    assert res.var_names == ("output", "inflation")
    assert res.irf.shape == (13, 2, 2)
    assert res.fevd.shape == (13, 2, 2)
    np.testing.assert_allclose(res.B0, B0_true, atol=0.12)
    assert res.diagnostics["identified"] is True
    assert res.diagnostics["min_wald_stat"] > 10.0
    assert res.diagnostics["max_wald_pvalue"] < 1e-3
    _assert_presentation_standard(res)


def test_svar_heteroskedastic_array_indicator_and_dataframe_input():
    """svar_heteroskedastic accepts boolean/integer regime arrays and DataFrames."""
    rng = np.random.default_rng(77)
    T = 600
    A1 = np.array([[0.4, 0.0], [0.0, 0.3]])
    B0_true = np.array([[1.2, -0.2], [0.25, 0.9]])
    eps = rng.standard_normal((T, 2))
    regime = np.zeros(T, dtype=int)
    regime[300:] = 1
    eps[regime == 1, 0] *= 3.0

    df = pd.DataFrame(_simulate_svar_dgp(T, A1, B0_true, eps), columns=["gdp", "rate"])
    res = svar_heteroskedastic(df, p=2, regime_indicator=regime == 1, horizon=8)
    assert res.var_names == ("gdp", "rate")
    np.testing.assert_allclose(res.B0, B0_true, atol=0.18)
    assert res.diagnostics["min_lambda_gap"] > 2.0


def test_svar_heteroskedastic_validation_errors():
    """svar_heteroskedastic raises informative ValueErrors on invalid regime indicators."""
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((100, 2))
    with pytest.raises(ValueError, match="regime break index"):
        svar_heteroskedastic(Y, p=2, regime_indicator=2)
    with pytest.raises(ValueError, match="2 distinct"):
        svar_heteroskedastic(Y, p=2, regime_indicator=np.zeros(100, dtype=int))
    with pytest.raises(ValueError, match="length"):
        svar_heteroskedastic(Y, p=2, regime_indicator=np.zeros(50, dtype=int))


# ---------------------------------------------------------------------------
# Track 3: svar_non_gaussian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["gmm", "fastica"])
def test_svar_non_gaussian_recovers_non_cholesky_b0(estimator: str):
    """svar_non_gaussian recovers a non-triangular B0 on Laplace structural shocks."""
    rng = np.random.default_rng(2024)
    T = 1500
    A1 = np.array([[0.45, 0.05], [-0.08, 0.35]])
    B0_true = np.array([[1.0, 0.35], [-0.40, 1.0]])

    # Unit-variance Laplace shocks (scale = 1 / sqrt(2))
    eps = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(T, 2))
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_non_gaussian(Y, p=1, method=estimator, horizon=10, seed=42)
    assert isinstance(res, SVARResult)
    assert res.method == "non_gaussian"
    np.testing.assert_allclose(res.B0, B0_true, atol=0.15)
    assert np.all(res.diagnostics["excess_kurtosis"] > 1.0)
    assert res.diagnostics["joint_jb_pvalue"] < 1e-6
    _assert_presentation_standard(res)


def test_svar_non_gaussian_invalid_method():
    """svar_non_gaussian rejects unknown method names."""
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((120, 2))
    with pytest.raises(ValueError, match="method"):
        svar_non_gaussian(Y, p=1, method="invalid_method")


# ---------------------------------------------------------------------------
# Track 3: svar_proxy
# ---------------------------------------------------------------------------


def test_svar_proxy_recovers_impact_and_diagnostics():
    """svar_proxy recovers structural impact column, MOP F-stat, bootstrap bands, and AR intervals."""
    rng = np.random.default_rng(999)
    T = 800
    A1 = np.array([[0.5, -0.1], [0.15, 0.4]])
    B0_true = np.array([[1.0, 0.2], [0.6, 0.9]])

    eps = rng.standard_normal((T, 2))
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)
    # Strong external instrument for shock 0 with some missing observations
    z = 0.85 * eps[:, 0] + 0.25 * rng.standard_normal(T)
    z[:15] = np.nan

    res = svar_proxy(
        Y,
        z,
        p=1,
        shocked_var=0,
        scale=1.0,
        horizon=10,
        n_boot=100,
        alpha=0.10,
        seed=42,
        var_names=("policy", "output"),
    )

    assert isinstance(res, SVARResult)
    assert res.method == "proxy"
    np.testing.assert_allclose(res.B0[:, 0], B0_true[:, 0], atol=0.12)
    assert res.diagnostics["mop_f"] > 50.0
    assert res.diagnostics["weak_iv"] is False
    assert res.diagnostics["ar_bounded"] is True

    # True impact of variable 1 lies inside the Anderson-Rubin confidence interval
    ar_lo = res.diagnostics["ar_ci_lower"][1]
    ar_hi = res.diagnostics["ar_ci_upper"][1]
    assert ar_lo < B0_true[1, 0] < ar_hi

    # Wild bootstrap confidence bands bracket point estimate
    assert res.ci_lower is not None and res.ci_upper is not None
    assert np.all(res.ci_lower <= res.ci_upper)
    _assert_presentation_standard(res)


def test_svar_proxy_weak_instrument_detection():
    """svar_proxy flags weak instruments when z is pure noise."""
    rng = np.random.default_rng(321)
    T = 250
    Y = rng.standard_normal((T, 2))
    z = rng.standard_normal(T)
    res = svar_proxy(Y, z, p=1, n_boot=20, seed=11)
    assert "mop_f" in res.diagnostics
    assert isinstance(res.diagnostics["weak_iv"], bool)


def test_svar_proxy_validation_errors():
    """svar_proxy validates instrument length, shocked_var, and alpha."""
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((100, 2))
    with pytest.raises(ValueError, match="length"):
        svar_proxy(Y, np.ones(10), p=2)
    with pytest.raises(ValueError, match="shocked_var"):
        svar_proxy(Y, np.ones(100), p=2, shocked_var=5)
    with pytest.raises(ValueError, match="alpha"):
        svar_proxy(Y, np.ones(100), p=2, alpha=1.5)


# ---------------------------------------------------------------------------
# Track 3: svar_sign_narrative
# ---------------------------------------------------------------------------


def test_svar_sign_narrative_enforces_restrictions_and_gk_bounds():
    """svar_sign_narrative enforces sign + narrative restrictions and narrows GK bounds."""
    rng = np.random.default_rng(42)
    T = 300
    A1 = np.array([[0.5, -0.15], [0.2, 0.45]])
    B0_true = np.array([[1.0, -0.3], [0.5, 0.8]])
    eps = rng.standard_normal((T, 2))
    # Inject a large positive structural shock 0 at residual index t_idx = 50 (t = 51)
    eps[51, 0] = 3.8
    eps[51, 1] = 0.1
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    sign_restr = {
        (0, 0, 0): 1,
        (1, 0, 0): 1,
    }
    res_sign = svar_sign_narrative(
        Y,
        p=1,
        sign_restrictions=sign_restr,
        horizon=10,
        n_draws=200,
        max_trials=5000,
        seed=42,
    )

    assert isinstance(res_sign, SVARResult)
    assert res_sign.gk_lower is not None and res_sign.gk_upper is not None
    assert res_sign.ci_lower is not None and res_sign.ci_upper is not None

    # Every accepted draw must satisfy the sign restrictions at h=0
    irf_draws = res_sign.diagnostics["accepted_irf_draws"]
    assert np.all(irf_draws[:, 0, 0, 0] > 0.0)
    assert np.all(irf_draws[:, 0, 1, 0] > 0.0)

    # Giacomini-Kitagawa bounds enclose the confidence interval and point IRF
    assert np.all(res_sign.gk_lower <= res_sign.ci_lower + 1e-12)
    assert np.all(res_sign.ci_lower <= res_sign.irf + 1e-12)
    assert np.all(res_sign.irf <= res_sign.ci_upper + 1e-12)
    assert np.all(res_sign.ci_upper <= res_sign.gk_upper + 1e-12)

    # Adding narrative restrictions narrows or preserves the GK identified set
    narr_restr = [
        ("shock_sign", 0, 50, 1),
        ("hd_dominance", 0, 50, 0),
    ]
    res_narr = svar_sign_narrative(
        Y,
        p=1,
        sign_restrictions=[(0, 0, 0, 1), (1, 0, 0, 1)],
        narrative_restrictions=narr_restr,
        horizon=10,
        n_draws=150,
        max_trials=8000,
        seed=42,
    )
    assert res_narr.diagnostics["identified_set_mean_width"] <= (
        res_sign.diagnostics["identified_set_mean_width"] + 1e-6
    )
    assert res_narr.structural_shocks[50, 0] > 0.0
    _assert_presentation_standard(res_narr)


def test_svar_sign_narrative_impossible_restriction_raises():
    """svar_sign_narrative raises RuntimeError when restrictions are mutually contradictory."""
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((120, 2))
    with pytest.raises(RuntimeError, match="0 draws accepted"):
        svar_sign_narrative(
            Y,
            p=1,
            sign_restrictions=[(0, 0, 0, 1)],
            narrative_restrictions=[("shock_sign", 0, 10, 1), ("shock_sign", 0, 10, -1)],
            n_draws=10,
            max_trials=200,
        )


# ---------------------------------------------------------------------------
# Track 4b: estimate_functional_var & FunctionalVARResult
# ---------------------------------------------------------------------------


def _simulate_functional_var_data(T: int = 240, B: int = 5, seed: int = 101):
    """Simulate coupled aggregate macro variables and B cross-sectional bin shares."""
    rng = np.random.default_rng(seed)
    macro = np.zeros((T, 2), dtype=float)
    logits = np.zeros((T, B), dtype=float)
    base_logits = np.linspace(0.5, -0.5, B)
    slope = np.linspace(-0.35, 0.35, B)

    for t in range(1, T):
        macro[t, 0] = 0.55 * macro[t - 1, 0] + rng.normal(0.0, 0.8)
        macro[t, 1] = 0.25 * macro[t - 1, 0] + 0.45 * macro[t - 1, 1] + rng.normal(0.0, 0.6)
        logits[t] = (
            base_logits
            + 0.5 * (logits[t - 1] - base_logits)
            + slope * macro[t, 0]
            + rng.normal(0.0, 0.12, size=B)
        )
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    shares = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return macro, shares


def test_functional_var_exact_adding_up_and_inequality_irfs():
    """estimate_functional_var satisfies exact cross-sectional adding up (< 1e-14)."""
    macro, shares = _simulate_functional_var_data(T=260, B=6, seed=42)
    grid = np.array([10.0, 25.0, 50.0, 90.0, 160.0, 350.0])

    res = estimate_functional_var(
        macro,
        shares,
        bin_grid=grid,
        p=2,
        horizon=16,
        macro_names=("gdp_growth", "unemployment"),
    )

    assert isinstance(res, FunctionalVARResult)
    assert res.adding_up_max_error < 1e-14
    # Check sum across bins b at every horizon h and every shock j
    bin_sums = np.sum(res.dist_irf, axis=1)
    np.testing.assert_allclose(bin_sums, 0.0, atol=1e-14)

    n_total = 2 + 6 - 1
    assert res.macro_irf.shape == (17, 2, n_total)
    assert res.dist_irf.shape == (17, 6, n_total)
    assert res.gini_irf.shape == (17, n_total)
    assert res.top_share_irf.shape == (17, n_total)
    assert res.p90_p10_irf.shape == (17, n_total)

    # Top share IRF equals top bin share response identically
    np.testing.assert_allclose(res.top_share_irf, res.dist_irf[:, -1, :], atol=1e-15)
    # Positive macro shock 0 shifts weight toward upper bins, raising Gini and top share
    assert np.max(np.abs(res.gini_irf[:, 0])) > 1e-4
    assert np.max(np.abs(res.p90_p10_irf[:, 0])) > 1e-4
    _assert_presentation_standard(res)


def test_functional_var_dataframe_and_1d_macro_inputs():
    """estimate_functional_var accepts DataFrames and 1D macro arrays."""
    macro, shares = _simulate_functional_var_data(T=180, B=4, seed=19)
    df_macro = pd.DataFrame(macro[:, :1], columns=["output"])
    df_shares = pd.DataFrame(shares, columns=["q1", "q2", "q3", "q4"])

    res = estimate_functional_var(df_macro, df_shares, p=1, horizon=8)
    assert res.macro_names == ("output",)
    assert res.adding_up_max_error < 1e-14
    assert res.dist_irf.shape == (9, 4, 4)

    # Also test 1D numpy macro input and pre-allocated axes in plot()
    res_1d = estimate_functional_var(macro[:, 0], shares, p=1, horizon=6)
    fig, axes = plt.subplots(1, 3)
    res_1d.plot(ax=axes, show=False)


def test_functional_var_validation_errors():
    """estimate_functional_var validates bin count, row sums, and sample lengths."""
    macro, shares = _simulate_functional_var_data(T=100, B=4, seed=7)
    with pytest.raises(ValueError, match="B >= 3"):
        estimate_functional_var(macro, shares[:, :2], p=1)
    with pytest.raises(ValueError, match="sum to 1.0"):
        estimate_functional_var(macro, shares * 2.0, p=1)
    with pytest.raises(ValueError, match="shape"):
        estimate_functional_var(macro[:80], shares, p=1)
    with pytest.raises(ValueError, match="bin_grid"):
        estimate_functional_var(macro, shares, bin_grid=[1.0, 2.0], p=1)


def test_svar_heteroskedastic_three_variable_dgp():
    """svar_heteroskedastic recovers a 3x3 structural impact matrix."""
    rng = np.random.default_rng(555)
    T = 1600
    A1 = np.diag([0.4, 0.35, 0.3])
    B0_true = np.array([
        [1.0, 0.2, -0.15],
        [-0.25, 1.1, 0.2],
        [0.1, -0.3, 0.95],
    ])
    eps = rng.standard_normal((T, 3))
    eps[800:, 0] *= 3.0
    eps[800:, 1] *= 1.6
    eps[800:, 2] *= 0.5
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_heteroskedastic(Y, p=1, regime_indicator=800, horizon=8)
    np.testing.assert_allclose(res.B0, B0_true, atol=0.18)
    assert res.diagnostics["identified"] is True
    assert len(res.diagnostics["wald_tests"]) == 3


def test_svar_heteroskedastic_unidentified_homoskedastic_dgp():
    """svar_heteroskedastic flags identified=False when regime variances are equal."""
    rng = np.random.default_rng(808)
    T = 600
    A1 = np.diag([0.3, 0.3])
    B0_true = np.eye(2)
    eps = rng.standard_normal((T, 2))
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_heteroskedastic(Y, p=1, regime_indicator=300, horizon=6)
    assert res.diagnostics["identified"] is False
    assert res.diagnostics["max_wald_pvalue"] > 0.05


def test_svar_non_gaussian_skewed_shocks_gmm():
    """svar_non_gaussian identifies B0 under asymmetric/skewed shocks via co-skewness GMM."""
    rng = np.random.default_rng(2025)
    T = 1600
    A1 = np.array([[0.3, 0.05], [-0.05, 0.25]])
    B0_true = np.array([[1.0, -0.35], [0.30, 1.0]])
    # Standardized exponential shocks have skewness = 2, excess kurtosis = 6
    eps = rng.exponential(scale=1.0, size=(T, 2)) - 1.0
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_non_gaussian(Y, p=1, method="gmm", horizon=8, seed=17)
    np.testing.assert_allclose(res.B0, B0_true, atol=0.15)
    assert np.all(np.abs(res.diagnostics["skewness"]) > 0.8)


def test_svar_non_gaussian_student_t_three_vars():
    """svar_non_gaussian identifies a 3-variable system with Student-t(5) shocks."""
    rng = np.random.default_rng(404)
    T = 2000
    A1 = np.diag([0.3, 0.25, 0.2])
    B0_true = np.array([
        [1.0, 0.2, -0.1],
        [-0.2, 1.0, 0.15],
        [0.15, -0.2, 1.0],
    ])
    raw_t = rng.standard_t(df=5.0, size=(T, 3))
    eps = raw_t / np.sqrt(5.0 / 3.0)
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_non_gaussian(Y, p=1, method="gmm", horizon=6, seed=99)
    np.testing.assert_allclose(res.B0, B0_true, atol=0.18)


def test_svar_proxy_non_zero_shocked_var_and_custom_scale():
    """svar_proxy works when shocked_var=1 and scale != 1.0."""
    rng = np.random.default_rng(717)
    T = 700
    A1 = np.array([[0.4, 0.1], [-0.1, 0.45]])
    B0_true = np.array([[0.9, -0.35], [0.25, 1.0]])
    eps = rng.standard_normal((T, 2))
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)
    z = 0.9 * eps[1:, 1] + 0.2 * rng.standard_normal(T - 1)

    res = svar_proxy(Y, z, p=1, shocked_var=1, scale=2.0, horizon=8, n_boot=50, seed=12)
    np.testing.assert_allclose(res.B0[:, 1], 2.0 * B0_true[:, 1], atol=0.25)
    assert res.diagnostics["unit_shock_impact"][1] == pytest.approx(2.0)


def test_svar_proxy_no_bootstrap():
    """svar_proxy with n_boot=0 skips bootstrap while still computing AR intervals."""
    rng = np.random.default_rng(61)
    T = 300
    Y = rng.standard_normal((T, 2))
    z = Y[:, 0] + 0.3 * rng.standard_normal(T)
    res = svar_proxy(Y, z, p=1, n_boot=0)
    assert res.ci_lower is None and res.ci_upper is None
    assert res.diagnostics["ar_ci_lower"].shape == (2,)


def test_svar_sign_narrative_multi_horizon_and_custom_ax():
    """svar_sign_narrative supports multi-horizon restrictions and plotting on custom axes."""
    rng = np.random.default_rng(88)
    T = 250
    A1 = np.array([[0.7, 0.0], [0.3, 0.6]])
    B0_true = np.array([[1.0, 0.0], [0.4, 0.8]])
    eps = rng.standard_normal((T, 2))
    Y = _simulate_svar_dgp(T, A1, B0_true, eps)

    res = svar_sign_narrative(
        Y,
        p=1,
        sign_restrictions={(0, 0, 0): 1, (0, 0, 1): 1, (1, 0, 0): 1},
        horizon=6,
        n_draws=80,
        max_trials=3000,
        seed=5,
    )
    irf_draws = res.diagnostics["accepted_irf_draws"]
    assert np.all(irf_draws[:, 0, 0, 0] > 0.0)
    assert np.all(irf_draws[:, 1, 0, 0] > 0.0)
    assert np.all(irf_draws[:, 0, 1, 0] > 0.0)

    fig, axes = plt.subplots(2, 2)
    out_fig = res.plot(ax=axes, show=False)
    assert out_fig is fig


@pytest.mark.parametrize("B", [3, 5, 8, 12])
def test_functional_var_helmert_basis_orthonormality_and_zero_sum(B: int):
    """Helmert contrast basis H is orthonormal and orthogonal to 1_B for all bin counts B."""
    macro, shares = _simulate_functional_var_data(T=150, B=B, seed=B * 11)
    res = estimate_functional_var(macro, shares, p=1, horizon=10)
    H = res.metadata["helmert_basis"]
    assert H.shape == (B, B - 1)
    np.testing.assert_allclose(H.T @ np.ones(B), np.zeros(B - 1), atol=1e-15)
    np.testing.assert_allclose(H.T @ H, np.eye(B - 1), atol=1e-14)
    assert res.adding_up_max_error < 1e-14


def test_result_classes_are_frozen_and_interactive_html_compatible():
    """SVARResult and FunctionalVARResult are frozen dataclasses and work with interactive HTML."""
    from dataclasses import FrozenInstanceError
    from puremacro.reports import result_to_interactive_html

    macro, shares = _simulate_functional_var_data(T=140, B=4, seed=33)
    f_res = estimate_functional_var(macro, shares, p=1, horizon=6)
    s_res = svar_heteroskedastic(macro, p=1, regime_indicator=70, horizon=6)

    with pytest.raises(FrozenInstanceError):
        f_res.horizon = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        s_res.horizon = 99  # type: ignore[misc]

    html_f = result_to_interactive_html(f_res)
    html_s = result_to_interactive_html(s_res)
    assert "<svg" in html_f and "<table" in html_f
    assert "<svg" in html_s and "<table" in html_s

