"""Tests for puremacro.inference.supt — sup-t simultaneous confidence bands.

Montiel Olea & Plagborg-Møller (2019), "Simultaneous confidence bands:
Theory, implementation, and an application to SVARs", JAE 34(1), 1-17.

Coverage strategy
-----------------
- Ordering: on random PSD Sigma the sup-t critical value (hence band) is
  never below the pointwise normal one and never above Bonferroni.
- Known value: iid case (identity correlation) has the closed form
  c* = Phi^{-1}((1 + (1-alpha)^{1/H})/2); the MC plugin value matches to
  2 decimals.
- Joint coverage Monte Carlo: H=6 AR-correlated t-stats, 2000 reps —
  empirical joint coverage of the 90% sup-t band in [0.87, 0.93] while the
  pointwise band's joint coverage falls well below 0.90.
- Bootstrap / bayes constructions: containment-fraction guarantee
  (>= 1-alpha of draws entirely inside, and minimal up to one order
  statistic), centering conventions, result contract.
- Error handling: NaN/degenerate/shape/alpha/method errors are diagnostic.
- Wiring smoke tests: var.bootstrap.bootstrap_bands(band='sup-t') and
  inference.lp_block_bootstrap.cum_irf_block_bootstrap(band='sup-t').
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro.inference.supt import SupTBandResult, supt_band


def _closed_form_crit(H: int, alpha: float) -> float:
    """Exact sup-t critical value under an identity correlation matrix."""
    return float(norm.ppf(0.5 * (1.0 + (1.0 - alpha) ** (1.0 / H))))


def _random_psd_sigma(H: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H, H))
    return A @ A.T + 0.5 * np.eye(H)


# ===========================================================================
# Result contract
# ===========================================================================

def test_plugin_result_contract():
    H = 5
    theta = np.arange(H, dtype=float)
    Sigma = _random_psd_sigma(H, 0)
    res = supt_band(theta, Sigma, alpha=0.10, n_mc=20_000, rng=0)
    assert isinstance(res, SupTBandResult)
    assert res.method == "plugin"
    assert res.alpha == 0.10
    assert res.n_draws == 20_000
    assert res.lower.shape == res.upper.shape == (H,)
    sd = np.sqrt(np.diag(Sigma))
    assert np.allclose(res.lower, theta - res.crit_value * sd)
    assert np.allclose(res.upper, theta + res.crit_value * sd)
    assert np.all(res.lower <= theta) and np.all(theta <= res.upper)
    assert "Sup-t" in res.summary()


def test_result_is_frozen():
    res = supt_band(np.zeros(3), np.eye(3), n_mc=5_000, rng=0)
    with pytest.raises((AttributeError, TypeError)):
        res.crit_value = 0.0  # type: ignore[misc]


def test_plugin_reproducible_with_seed():
    theta, Sigma = np.zeros(4), _random_psd_sigma(4, 1)
    r1 = supt_band(theta, Sigma, n_mc=20_000, rng=123)
    r2 = supt_band(theta, Sigma, n_mc=20_000, rng=123)
    assert r1.crit_value == r2.crit_value
    r3 = supt_band(theta, Sigma, n_mc=20_000, rng=np.random.default_rng(123))
    assert r3.crit_value == r1.crit_value


# ===========================================================================
# Ordering: pointwise <= sup-t <= Bonferroni (random PSD Sigma)
# ===========================================================================

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("H", [4, 8])
def test_supt_between_pointwise_and_bonferroni(H, seed):
    alpha = 0.10
    Sigma = _random_psd_sigma(H, seed)
    theta = np.zeros(H)
    res = supt_band(theta, Sigma, alpha=alpha, n_mc=50_000, rng=seed + 10)
    z_pw = norm.ppf(1 - alpha / 2)
    z_bonf = norm.ppf(1 - alpha / (2 * H))
    mc_tol = 0.02  # MC error on a 50k-draw quantile is ~0.005; 4x margin
    assert res.crit_value >= z_pw - mc_tol, (
        f"sup-t crit {res.crit_value:.4f} below pointwise {z_pw:.4f}"
    )
    assert res.crit_value <= z_bonf + mc_tol, (
        f"sup-t crit {res.crit_value:.4f} above Bonferroni {z_bonf:.4f}"
    )
    # Same ordering for the band widths, elementwise.
    sd = np.sqrt(np.diag(Sigma))
    width = res.upper - res.lower
    assert np.all(width >= 2 * (z_pw - mc_tol) * sd)
    assert np.all(width <= 2 * (z_bonf + mc_tol) * sd)


def test_perfect_correlation_collapses_to_pointwise():
    """With perfectly correlated coordinates the max is one |N(0,1)|."""
    H, alpha = 5, 0.10
    Sigma = np.ones((H, H))  # rank-1: all coordinates identical
    res = supt_band(np.zeros(H), Sigma, alpha=alpha, n_mc=100_000, rng=7)
    assert abs(res.crit_value - norm.ppf(1 - alpha / 2)) < 0.02


# ===========================================================================
# Known value: iid closed form (2-decimal MC check)
# ===========================================================================

@pytest.mark.parametrize("H", [1, 4, 6])
def test_plugin_iid_matches_closed_form(H):
    alpha = 0.10
    res = supt_band(np.zeros(H), np.eye(H), alpha=alpha, n_mc=200_000, rng=42)
    c_exact = _closed_form_crit(H, alpha)
    assert abs(res.crit_value - c_exact) < 0.01, (
        f"H={H}: MC crit {res.crit_value:.4f} vs exact {c_exact:.4f}"
    )


def test_plugin_iid_closed_form_alpha_05():
    H, alpha = 6, 0.05
    res = supt_band(np.zeros(H), np.eye(H), alpha=alpha, n_mc=200_000, rng=43)
    assert abs(res.crit_value - _closed_form_crit(H, alpha)) < 0.01


# ===========================================================================
# Joint-coverage Monte Carlo (H=6, AR-correlated, 2000 reps)
# ===========================================================================

def test_joint_coverage_ar_correlated():
    H, alpha, rho, reps = 6, 0.10, 0.7, 2000
    idx = np.arange(H)
    R = rho ** np.abs(np.subtract.outer(idx, idx))  # AR(1) correlation
    # Critical values (Sigma known and fixed across reps).
    c_supt = supt_band(np.zeros(H), R, alpha=alpha, n_mc=100_000,
                       rng=99).crit_value
    c_pw = norm.ppf(1 - alpha / 2)
    # Simulate theta_hat ~ N(0, R) with truth 0 and unit variances: the band
    # covers the truth iff max_h |z_h| <= c.
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(20260718)
    Z = rng.standard_normal((reps, H)) @ L.T
    m = np.abs(Z).max(axis=1)
    cov_supt = float(np.mean(m <= c_supt))
    cov_pw = float(np.mean(m <= c_pw))
    assert 0.87 <= cov_supt <= 0.93, (
        f"sup-t joint coverage {cov_supt:.4f} outside [0.87, 0.93]"
    )
    assert cov_pw < 0.85, (
        f"pointwise joint coverage {cov_pw:.4f} not well below 0.90"
    )
    assert cov_pw < cov_supt


# ===========================================================================
# Bootstrap construction (Algorithm 2)
# ===========================================================================

def _correlated_draws(n: int, H: int, seed: int, mu: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H, H))
    S = A @ A.T + np.eye(H)
    L = np.linalg.cholesky(S)
    return mu + rng.standard_normal((n, H)) @ L.T


def test_bootstrap_containment_guarantee():
    """>= 1-alpha of the draws lie entirely inside the band, minimally so."""
    n, H, alpha = 5000, 8, 0.10
    draws = _correlated_draws(n, H, seed=3, mu=1.5)
    theta = draws.mean(axis=0)  # stand-in point estimate
    res = supt_band(theta=theta, draws=draws, method="bootstrap", alpha=alpha)
    inside = np.all((draws >= res.lower[None, :]) &
                    (draws <= res.upper[None, :]), axis=1).mean()
    # One-draw allowance: the calibrating draw sits exactly ON the band edge,
    # and recomputing center ± c*scale can flip it outside at the last ulp
    # (observed across BLAS backends: Accelerate vs OpenBLAS).
    assert inside >= 1 - alpha - 1.0 / n - 1e-12
    # Minimality: the calibrated c is an order statistic, so the fraction
    # inside exceeds 1-alpha by at most ~1/n (ties aside).
    assert inside <= 1 - alpha + 2.0 / n + 1e-12


def test_bootstrap_center_defaults_to_draw_mean():
    draws = _correlated_draws(500, 4, seed=5)
    res = supt_band(draws=draws, method="bootstrap", alpha=0.10)
    assert np.allclose(res.center, draws.mean(axis=0))
    assert res.method == "bootstrap"
    assert res.n_draws == 500


def test_bootstrap_explicit_theta_centers_band():
    draws = _correlated_draws(500, 4, seed=6)
    theta = np.array([10.0, -3.0, 0.5, 2.0])
    res = supt_band(theta=theta, draws=draws, method="bootstrap", alpha=0.10)
    assert np.allclose(res.center, theta)
    assert np.allclose(0.5 * (res.lower + res.upper), theta)


def test_bootstrap_scale_is_draw_std():
    draws = _correlated_draws(800, 5, seed=7)
    res = supt_band(draws=draws, method="bootstrap", alpha=0.10)
    assert np.allclose(res.scale, draws.std(axis=0, ddof=1))


# ===========================================================================
# Bayes construction (Algorithm 3, simple calibrated version)
# ===========================================================================

def test_bayes_containment_guarantee():
    n, H, alpha = 4000, 6, 0.20
    draws = _correlated_draws(n, H, seed=11, mu=-2.0)
    res = supt_band(draws=draws, method="bayes", alpha=alpha)
    assert np.allclose(res.center, draws.mean(axis=0))
    inside = np.all((draws >= res.lower[None, :]) &
                    (draws <= res.upper[None, :]), axis=1).mean()
    assert inside >= 1 - alpha - 1e-12
    assert inside <= 1 - alpha + 2.0 / n + 1e-12


def test_bayes_smallest_c_property():
    """Any strictly smaller c drops the containment below 1-alpha."""
    n, alpha = 1000, 0.10
    draws = _correlated_draws(n, 5, seed=12)
    res = supt_band(draws=draws, method="bayes", alpha=alpha)
    t = np.max(np.abs(draws - res.center[None, :]) / res.scale[None, :],
               axis=1)
    c_smaller = np.max(t[t < res.crit_value])  # next order statistic down
    assert np.mean(t <= c_smaller) < 1 - alpha


def test_bayes_rejects_theta():
    draws = _correlated_draws(100, 3, seed=13)
    with pytest.raises(ValueError, match="posterior mean"):
        supt_band(theta=np.zeros(3), draws=draws, method="bayes")


# ===========================================================================
# Error handling — diagnostic errors over silent garbage
# ===========================================================================

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        supt_band(np.zeros(3), np.eye(3), method="jackknife")


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_bad_alpha_raises(bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        supt_band(np.zeros(3), np.eye(3), alpha=bad_alpha, n_mc=1_000)


def test_plugin_requires_theta_and_sigma():
    with pytest.raises(ValueError, match="plugin"):
        supt_band(np.zeros(3))
    with pytest.raises(ValueError, match="plugin"):
        supt_band(Sigma=np.eye(3))


def test_plugin_rejects_draws():
    with pytest.raises(ValueError, match="draws"):
        supt_band(np.zeros(3), np.eye(3), draws=np.zeros((50, 3)))


def test_plugin_nan_theta_raises():
    theta = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="theta"):
        supt_band(theta, np.eye(3))


def test_plugin_nan_sigma_raises():
    Sigma = np.eye(3)
    Sigma[0, 1] = Sigma[1, 0] = np.nan
    with pytest.raises(ValueError, match="Sigma"):
        supt_band(np.zeros(3), Sigma)


def test_plugin_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Sigma must be"):
        supt_band(np.zeros(4), np.eye(3))


def test_plugin_asymmetric_sigma_raises():
    Sigma = np.eye(3)
    Sigma[0, 1] = 0.9  # not mirrored
    with pytest.raises(ValueError, match="symmetric"):
        supt_band(np.zeros(3), Sigma)


def test_plugin_zero_variance_raises():
    Sigma = np.diag([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="non-positive variance"):
        supt_band(np.zeros(3), Sigma)


def test_plugin_non_psd_raises():
    Sigma = np.array([[1.0, 0.99, 0.0],
                      [0.99, 1.0, 0.99],
                      [0.0, 0.99, 1.0]])  # indefinite
    with pytest.raises(ValueError, match="positive semi-definite"):
        supt_band(np.zeros(3), Sigma, n_mc=1_000)


def test_plugin_tiny_n_mc_raises():
    with pytest.raises(ValueError, match="n_mc"):
        supt_band(np.zeros(3), np.eye(3), n_mc=10)


def test_draws_must_be_2d():
    with pytest.raises(ValueError, match="2-D"):
        supt_band(draws=np.zeros(50), method="bootstrap")


def test_too_few_draws_raises():
    with pytest.raises(ValueError, match="at least 20"):
        supt_band(draws=np.zeros((5, 3)), method="bootstrap")


def test_nan_draws_raise_with_row_count():
    draws = _correlated_draws(100, 3, seed=14)
    draws[3, 1] = np.nan
    draws[7, 0] = np.inf
    with pytest.raises(ValueError, match="2 row"):
        supt_band(draws=draws, method="bootstrap")


def test_degenerate_draw_column_raises():
    draws = _correlated_draws(100, 3, seed=15)
    draws[:, 2] = 4.2  # constant coordinate
    with pytest.raises(ValueError, match=r"degenerate.*\[2\]"):
        supt_band(draws=draws, method="bootstrap")


def test_bootstrap_rejects_sigma():
    draws = _correlated_draws(100, 3, seed=16)
    with pytest.raises(ValueError, match="not Sigma"):
        supt_band(Sigma=np.eye(3), draws=draws, method="bootstrap")


# ===========================================================================
# Wiring: var.bootstrap.bootstrap_bands(band='sup-t')
# ===========================================================================

def _stable_var1_data(T: int = 100, n: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(n)
    L = np.linalg.cholesky(np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n)))
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def _cholesky_identify(A_list, Sigma):
    from puremacro.var.identify.cholesky import cholesky_factor
    return cholesky_factor(Sigma)


class TestVarBootstrapWiring:
    def test_default_band_contract_unchanged(self):
        from puremacro.var.bootstrap import bootstrap_bands
        Y = _stable_var1_data()
        r = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=4, n_boot=40,
                            rng=np.random.default_rng(0))
        assert set(r.keys()) == {"point", "lower", "upper", "draws"}

    def test_supt_band_keys_and_shapes(self):
        from puremacro.var.bootstrap import bootstrap_bands
        Y = _stable_var1_data(seed=1)
        r = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=4, n_boot=60, band="sup-t",
                            rng=np.random.default_rng(1))
        assert {"point", "lower", "upper", "draws", "band",
                "crit_value"} <= set(r.keys())
        assert r["band"] == "sup-t"
        assert r["point"].shape == (5, 2, 2)
        assert r["lower"].shape == (5, 2, 2)
        assert r["crit_value"].shape == (2, 2)
        assert np.all(np.isfinite(r["crit_value"]))
        assert np.all(r["lower"] <= r["point"] + 1e-12)
        assert np.all(r["point"] <= r["upper"] + 1e-12)

    def test_supt_joint_containment_of_draws(self):
        """Per (response, shock) pair, >= 1-alpha of the finite bootstrap
        paths lie entirely inside the sup-t band — the defining property."""
        from puremacro.var.bootstrap import bootstrap_bands
        alpha = 0.10
        Y = _stable_var1_data(seed=2)
        r = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=4, n_boot=80, alpha=alpha, band="sup-t",
                            rng=np.random.default_rng(2))
        draws = r["draws"]
        finite = ~np.isnan(draws).any(axis=(1, 2, 3))
        good = draws[finite]
        for i in range(2):
            for j in range(2):
                inside = np.all(
                    (good[:, :, i, j] >= r["lower"][None, :, i, j] - 1e-12) &
                    (good[:, :, i, j] <= r["upper"][None, :, i, j] + 1e-12),
                    axis=1,
                ).mean()
                assert inside >= 1 - alpha - 1e-9, (
                    f"pair ({i},{j}): only {inside:.3f} of paths inside"
                )

    def test_supt_degenerate_impact_zeros_collapse(self):
        """Cholesky impact of shock 2 on variable 1 is 0 in every draw; the
        sup-t band must collapse to the point there rather than blow up."""
        from puremacro.var.bootstrap import bootstrap_bands
        Y = _stable_var1_data(seed=3)
        r = bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=3, n_boot=40, band="sup-t",
                            rng=np.random.default_rng(3))
        assert r["lower"][0, 0, 1] == r["upper"][0, 0, 1] == r["point"][0, 0, 1] == 0.0

    def test_supt_same_draws_as_pointwise(self):
        """band only changes post-processing; same rng gives same draws."""
        from puremacro.var.bootstrap import bootstrap_bands
        Y = _stable_var1_data(seed=4)
        kw = dict(p=1, identify_fn=_cholesky_identify, horizon=3, n_boot=30)
        r_pw = bootstrap_bands(Y, **kw, rng=np.random.default_rng(5))
        r_st = bootstrap_bands(Y, **kw, band="sup-t",
                               rng=np.random.default_rng(5))
        assert np.allclose(r_pw["draws"], r_st["draws"], equal_nan=True)
        assert np.allclose(r_pw["point"], r_st["point"])

    def test_unknown_band_raises(self):
        from puremacro.var.bootstrap import bootstrap_bands
        Y = _stable_var1_data()
        with pytest.raises(ValueError, match="unknown band"):
            bootstrap_bands(Y, p=1, identify_fn=_cholesky_identify,
                            horizon=2, n_boot=25, band="simultaneous",
                            rng=np.random.default_rng(0))


# ===========================================================================
# Wiring: inference.lp_block_bootstrap.cum_irf_block_bootstrap(band='sup-t')
# ===========================================================================

def _make_panel(n_entities: int = 6, n_periods: int = 40,
                seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"E{i:02d}" for i in range(n_entities)]
    dates = pd.period_range("2000Q1", periods=n_periods, freq="Q").to_timestamp()
    index = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    return pd.DataFrame(
        {"y": rng.standard_normal(n_entities * n_periods),
         "x": rng.standard_normal(n_entities * n_periods)},
        index=index,
    )


class TestLpBlockBootstrapWiring:
    def test_supt_columns_and_ordering(self):
        from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
        df = _make_panel()
        r = cum_irf_block_bootstrap(df, y="y", x="x", horizons=[0, 1, 2],
                                    B=40, seed=0, band="sup-t")
        assert list(r.columns) == ["h", "beta", "cum_beta", "cum_lo", "cum_hi"]
        assert (r["cum_lo"] <= r["cum_beta"]).all()
        assert (r["cum_beta"] <= r["cum_hi"]).all()
        assert r.attrs["band"] == "sup-t"
        assert r.attrs["supt_crit"] > 0

    def test_supt_band_symmetric_around_cum_beta(self):
        """Sup-t band is cum_beta ± c*sd — symmetric around the point."""
        from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
        df = _make_panel(seed=1)
        r = cum_irf_block_bootstrap(df, y="y", x="x", horizons=[0, 1, 2],
                                    B=40, seed=1, band="sup-t")
        mid = 0.5 * (r["cum_lo"] + r["cum_hi"])
        assert np.allclose(mid.to_numpy(), r["cum_beta"].to_numpy(), atol=1e-10)

    def test_default_band_unchanged_and_attrs_additive(self):
        from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
        df = _make_panel(seed=2)
        kw = dict(y="y", x="x", horizons=[0, 1], B=30, seed=2)
        r_default = cum_irf_block_bootstrap(df, **kw)
        r_pw = cum_irf_block_bootstrap(df, **kw, band="pointwise")
        assert np.allclose(r_default.to_numpy(), r_pw.to_numpy())
        assert r_default.attrs["band"] == "pointwise"
        assert "supt_crit" not in r_default.attrs

    def test_supt_differs_from_pointwise(self):
        from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
        df = _make_panel(seed=3)
        kw = dict(y="y", x="x", horizons=[0, 1, 2], B=40, seed=3)
        r_pw = cum_irf_block_bootstrap(df, **kw)
        r_st = cum_irf_block_bootstrap(df, **kw, band="sup-t")
        # Same point estimates, different (joint) bands.
        assert np.allclose(r_pw["cum_beta"], r_st["cum_beta"])
        assert not np.allclose(r_pw[["cum_lo", "cum_hi"]].to_numpy(),
                               r_st[["cum_lo", "cum_hi"]].to_numpy())

    def test_unknown_band_raises(self):
        from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
        df = _make_panel()
        with pytest.raises(ValueError, match="unknown band"):
            cum_irf_block_bootstrap(df, y="y", x="x", horizons=[0, 1],
                                    B=10, seed=0, band="joint")
