"""Coverage tests for puremacro.cross_nest.

Covers:
  - CrossNestFit dataclass (frozen, field values)
  - _within_demean (known-value demeaning)
  - fit_cross_nest_pooled: OLS known-value, 2SLS known-value, edge cases
  - fit_all_cross_nest: all four pairs, partial IV fallback
  - sato_wald_test: analytical chi2 values, NaN/degenerate guard, both variants
  - build_cross_nest_columns: known-value log-diffs, country boundaries, NaN on first obs
  - crossnest_summary: shape, column names, description strings
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from puremacro.cross_nest import (
    CrossNestFit,
    _within_demean,
    build_cross_nest_columns,
    crossnest_summary,
    fit_all_cross_nest,
    fit_cross_nest_pooled,
    sato_wald_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ols_panel(
    sigma_true: float,
    *,
    n_country: int = 3,
    n_per: int = 40,
    noise: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic balanced panel where qty = sigma * price + noise."""
    rng = np.random.default_rng(seed)
    codes = [chr(65 + i) for i in range(n_country)]  # 'A', 'B', ...
    rows = []
    for c in codes:
        price = rng.standard_normal(n_per)
        qty = sigma_true * price + rng.standard_normal(n_per) * noise
        rows.extend(
            {"code": c, "qty": float(qty[i]), "price": float(price[i])}
            for i in range(n_per)
        )
    return pd.DataFrame(rows)


def _make_2sls_panel(
    sigma_true: float,
    *,
    n_country: int = 4,
    n_per: int = 50,
    pi: float = 0.8,
    noise_first: float = 0.2,
    noise_second: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate panel with a valid IV: z -> x -> y."""
    rng = np.random.default_rng(seed)
    codes = [chr(65 + i) for i in range(n_country)]
    rows = []
    for c in codes:
        z = rng.standard_normal(n_per)
        x = pi * z + rng.standard_normal(n_per) * noise_first
        y = sigma_true * x + rng.standard_normal(n_per) * noise_second
        rows.extend(
            {"code": c, "qty": float(y[i]), "price": float(x[i]), "iv": float(z[i])}
            for i in range(n_per)
        )
    return pd.DataFrame(rows)


def _identical_fits(sigma: float = 1.5, se: float = 0.1) -> dict[str, CrossNestFit]:
    """Four identical CrossNestFit objects — Sato restriction holds exactly."""
    return {
        p: CrossNestFit(
            sigma=sigma, se=se, first_stage_F=None,
            n_obs=100, n_country=3, estimator="OLS", pair=p,
        )
        for p in ("es", "eu", "is", "iu")
    }


# ---------------------------------------------------------------------------
# CrossNestFit dataclass
# ---------------------------------------------------------------------------

class TestCrossNestFit:
    def test_fields_accessible(self):
        f = CrossNestFit(
            sigma=1.2, se=0.05, first_stage_F=15.0,
            n_obs=200, n_country=5, estimator="2SLS", pair="es",
        )
        assert f.sigma == 1.2
        assert f.se == 0.05
        assert f.first_stage_F == 15.0
        assert f.n_obs == 200
        assert f.n_country == 5
        assert f.estimator == "2SLS"
        assert f.pair == "es"

    def test_frozen(self):
        """CrossNestFit is frozen — assignment must raise."""
        f = CrossNestFit(
            sigma=1.0, se=0.1, first_stage_F=None,
            n_obs=10, n_country=2, estimator="OLS", pair="eu",
        )
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            f.sigma = 99.0  # type: ignore[misc]

    def test_first_stage_f_none_for_ols(self):
        f = CrossNestFit(
            sigma=1.0, se=0.1, first_stage_F=None,
            n_obs=50, n_country=2, estimator="OLS", pair="is",
        )
        assert f.first_stage_F is None


# ---------------------------------------------------------------------------
# _within_demean
# ---------------------------------------------------------------------------

class TestWithinDemean:
    def test_known_values(self):
        """Group means are removed exactly."""
        df = pd.DataFrame({"code": ["A", "A", "B", "B"], "x": [1.0, 3.0, 2.0, 6.0]})
        out = _within_demean(df, ["x"])
        # A: mean=2 -> [-1, 1]; B: mean=4 -> [-2, 2]
        np.testing.assert_allclose(out["x_demean"].to_numpy(), [-1.0, 1.0, -2.0, 2.0])

    def test_group_mean_is_zero(self):
        """Demeaned values sum to zero within each group."""
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "code": ["X"] * 10 + ["Y"] * 15,
            "v": rng.standard_normal(25),
        })
        out = _within_demean(df, ["v"])
        for grp in ("X", "Y"):
            s = out.loc[out["code"] == grp, "v_demean"].sum()
            assert abs(s) < 1e-10, f"group {grp} residuals don't sum to zero: {s}"

    def test_original_columns_preserved(self):
        """Original column is not modified; new column is appended."""
        df = pd.DataFrame({"code": ["A", "A"], "x": [2.0, 4.0]})
        out = _within_demean(df, ["x"])
        assert "x" in out.columns
        assert "x_demean" in out.columns
        # original column unchanged
        np.testing.assert_array_equal(out["x"].to_numpy(), [2.0, 4.0])

    def test_multiple_columns(self):
        """Multiple columns are demeaned independently."""
        df = pd.DataFrame({
            "code": ["A", "A"],
            "x": [0.0, 4.0],  # mean 2
            "y": [1.0, 5.0],  # mean 3
        })
        out = _within_demean(df, ["x", "y"])
        np.testing.assert_allclose(out["x_demean"].to_numpy(), [-2.0, 2.0])
        np.testing.assert_allclose(out["y_demean"].to_numpy(), [-2.0, 2.0])


# ---------------------------------------------------------------------------
# fit_cross_nest_pooled — OLS
# ---------------------------------------------------------------------------

class TestFitCrossNestOLS:
    def test_known_value_recovery(self):
        """OLS recovers the true sigma within tolerance for a large sample."""
        sigma_true = 1.5
        df = _make_ols_panel(sigma_true, n_country=5, n_per=100, noise=0.1, seed=1)
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price", pair="es")
        assert fit.estimator == "OLS"
        assert fit.first_stage_F is None
        assert abs(fit.sigma - sigma_true) < 0.05

    def test_negative_sigma_recovered(self):
        """OLS recovers a negative elasticity correctly."""
        sigma_true = -0.8
        df = _make_ols_panel(sigma_true, n_country=3, n_per=60, noise=0.05, seed=2)
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        assert abs(fit.sigma - sigma_true) < 0.05

    def test_se_positive(self):
        """SE is strictly positive for valid panel data."""
        df = _make_ols_panel(1.5, seed=3)
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        assert fit.se > 0.0

    def test_n_obs_and_n_country(self):
        """n_obs and n_country are counted correctly."""
        df = _make_ols_panel(1.0, n_country=4, n_per=25, seed=4)
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        assert fit.n_obs == 4 * 25
        assert fit.n_country == 4

    def test_nan_rows_dropped(self):
        """Rows with NaN in lhs_col or rhs_col are silently dropped."""
        df = _make_ols_panel(1.5, n_country=2, n_per=20, seed=5)
        df.loc[0, "qty"] = np.nan  # inject one NaN
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        assert fit.n_obs == 2 * 20 - 1

    def test_pair_label_preserved(self):
        """The pair= label is returned unchanged in the result."""
        df = _make_ols_panel(1.5, seed=6)
        for label in ("es", "eu", "is", "iu"):
            fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price", pair=label)
            assert fit.pair == label

    def test_empty_panel_returns_nan_ols(self):
        """An all-NaN / empty effective sample returns nan sigma."""
        df = pd.DataFrame({"code": pd.Series([], dtype=str), "qty": [], "price": []})
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price", pair="es")
        assert fit.n_obs == 0
        assert fit.n_country == 0
        assert math.isnan(fit.sigma)
        assert fit.estimator == "OLS"

    def test_constant_price_returns_nan(self):
        """When rhs has zero variance after demeaning, sigma is nan."""
        df = pd.DataFrame({
            "code": ["X"] * 5,
            "qty": [1.0, 2.0, 3.0, 4.0, 5.0],
            "price": [2.0] * 5,  # constant -> zero variance after demean
        })
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        assert math.isnan(fit.sigma)

    def test_more_data_tighter_se(self):
        """Larger sample size should produce a smaller SE (stochastic property)."""
        sigma_true = 1.5
        # Use different seeds but same DGP
        df_small = _make_ols_panel(sigma_true, n_country=3, n_per=20, noise=0.5, seed=10)
        df_large = _make_ols_panel(sigma_true, n_country=3, n_per=200, noise=0.5, seed=11)
        se_small = fit_cross_nest_pooled(df_small, lhs_col="qty", rhs_col="price").se
        se_large = fit_cross_nest_pooled(df_large, lhs_col="qty", rhs_col="price").se
        assert se_large < se_small

    def test_ci_contains_point_estimate(self):
        """A 95% CI (sigma ± 1.96 * se) must bracket the point estimate by construction."""
        df = _make_ols_panel(1.5, seed=12)
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price")
        ci_lo = fit.sigma - 1.96 * fit.se
        ci_hi = fit.sigma + 1.96 * fit.se
        assert ci_lo < fit.sigma < ci_hi


# ---------------------------------------------------------------------------
# fit_cross_nest_pooled — 2SLS
# ---------------------------------------------------------------------------

class TestFitCrossNest2SLS:
    def test_known_value_recovery(self):
        """2SLS recovers the true sigma within tolerance for a large sample."""
        sigma_true = 2.0
        df = _make_2sls_panel(sigma_true, n_country=5, n_per=100, seed=20)
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv", pair="eu"
        )
        assert fit.estimator == "2SLS"
        assert abs(fit.sigma - sigma_true) < 0.15

    def test_first_stage_f_positive(self):
        """With a strong IV the first-stage F is positive and finite."""
        df = _make_2sls_panel(2.0, n_country=4, n_per=80, pi=0.9, seed=21)
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv"
        )
        assert fit.first_stage_F is not None
        assert math.isfinite(fit.first_stage_F)
        assert fit.first_stage_F > 0

    def test_se_positive_2sls(self):
        """2SLS SE is strictly positive for valid data."""
        df = _make_2sls_panel(1.5, seed=22)
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv"
        )
        assert fit.se > 0.0

    def test_empty_panel_returns_nan_2sls(self):
        """Empty effective sample returns nan sigma and '2SLS' estimator label."""
        df = pd.DataFrame({"code": pd.Series([], dtype=str), "qty": [], "price": [], "iv": []})
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv", pair="eu"
        )
        assert math.isnan(fit.sigma)
        assert fit.estimator == "2SLS"
        assert fit.n_obs == 0

    def test_constant_instrument_returns_nan(self):
        """When the instrument has zero variance (after demean), sigma is nan."""
        df = pd.DataFrame({
            "code": ["A"] * 5,
            "qty": [1.0, 2.0, 3.0, 4.0, 5.0],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "iv": [3.0] * 5,  # constant -> zero variance in demeaned space
        })
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv"
        )
        assert math.isnan(fit.sigma)

    def test_n_obs_and_n_country_2sls(self):
        """n_obs and n_country are reported correctly for 2SLS."""
        df = _make_2sls_panel(1.5, n_country=3, n_per=30, seed=23)
        fit = fit_cross_nest_pooled(
            df, lhs_col="qty", rhs_col="price", iv_col="iv"
        )
        assert fit.n_obs == 3 * 30
        assert fit.n_country == 3

    def test_instrument_orthogonal_to_regressor_returns_nan(self):
        """When z is non-constant but orthogonal to x after demeaning (sx ~ 0),
        the 2SLS first stage has near-zero projection and sigma is nan.
        (Covers the abs(sx) < 1e-12 degenerate branch in 2SLS.)
        """
        # Construct z_d and x_d orthogonal in demeaned space within a single country.
        z_d = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        v = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        x_d = v - (np.dot(v, z_d) / np.dot(z_d, z_d)) * z_d
        # dot(z_d, x_d) is ~0 (floating-point machine epsilon)
        assert abs(np.dot(z_d, x_d)) < 1e-12
        z = z_d + 3.5  # restore arbitrary mean
        x = x_d + 5.0
        y = 2.0 * x
        df = pd.DataFrame({"code": ["AUS"] * 6, "qty": y, "price": x, "iv": z})
        fit = fit_cross_nest_pooled(df, lhs_col="qty", rhs_col="price", iv_col="iv", pair="es")
        assert math.isnan(fit.sigma)
        assert fit.estimator == "2SLS"
        # first_stage_F is set (not None) in this branch
        assert fit.first_stage_F is not None


# ---------------------------------------------------------------------------
# fit_all_cross_nest
# ---------------------------------------------------------------------------

class TestFitAllCrossNest:
    def _build_four_col_panel(self, sigma: float = 1.5, seed: int = 30) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for c in ("AUS", "CAN", "DEU"):
            p = rng.standard_normal(40)
            for suffix in ("es", "eu", "is", "iu"):
                noise = rng.standard_normal(40) * 0.2
                qty = sigma * p + noise
                if not rows or rows[-1]["code"] != c:
                    pass
            # Build one row per obs with four LHS and four RHS columns
            q_es = sigma * p + rng.standard_normal(40) * 0.2
            q_eu = sigma * p + rng.standard_normal(40) * 0.2
            q_is = sigma * p + rng.standard_normal(40) * 0.2
            q_iu = sigma * p + rng.standard_normal(40) * 0.2
            p_es = p.copy()
            p_eu = p.copy()
            p_is = p.copy()
            p_iu = p.copy()
            rows.extend(
                {
                    "code": c,
                    "q_es": q_es[i], "q_eu": q_eu[i],
                    "q_is": q_is[i], "q_iu": q_iu[i],
                    "p_es": p_es[i], "p_eu": p_eu[i],
                    "p_is": p_is[i], "p_iu": p_iu[i],
                }
                for i in range(40)
            )
        return pd.DataFrame(rows)

    def test_returns_all_four_pairs(self):
        df = self._build_four_col_panel()
        results = fit_all_cross_nest(
            df,
            quantity_cols={"es": "q_es", "eu": "q_eu", "is": "q_is", "iu": "q_iu"},
            price_cols={"es": "p_es", "eu": "p_eu", "is": "p_is", "iu": "p_iu"},
        )
        assert set(results.keys()) == {"es", "eu", "is", "iu"}

    def test_ols_without_iv_cols(self):
        """When iv_cols is None all four estimators are OLS."""
        df = self._build_four_col_panel(seed=31)
        results = fit_all_cross_nest(
            df,
            quantity_cols={"es": "q_es", "eu": "q_eu", "is": "q_is", "iu": "q_iu"},
            price_cols={"es": "p_es", "eu": "p_eu", "is": "p_is", "iu": "p_iu"},
        )
        for p in ("es", "eu", "is", "iu"):
            assert results[p].estimator == "OLS"
            assert results[p].first_stage_F is None

    def test_partial_iv_fallback(self):
        """Keys missing from iv_cols fall back to OLS for that pair."""
        rng = np.random.default_rng(32)
        n = 30
        rows = []
        for c in ("X", "Y", "Z"):
            p = rng.standard_normal(n)
            iv = rng.standard_normal(n)
            q = 1.5 * p + rng.standard_normal(n) * 0.2
            rows.extend(
                {"code": c, "qty": q[i], "price": p[i], "iv": iv[i]}
                for i in range(n)
            )
        df = pd.DataFrame(rows)
        results = fit_all_cross_nest(
            df,
            quantity_cols={"es": "qty", "eu": "qty", "is": "qty", "iu": "qty"},
            price_cols={"es": "price", "eu": "price", "is": "price", "iu": "price"},
            iv_cols={"es": "iv", "eu": "iv"},  # 'is' and 'iu' missing -> OLS
        )
        assert results["es"].estimator == "2SLS"
        assert results["eu"].estimator == "2SLS"
        assert results["is"].estimator == "OLS"
        assert results["iu"].estimator == "OLS"

    def test_pair_labels_match_keys(self):
        """The pair field of each result matches its dict key."""
        df = self._build_four_col_panel(seed=33)
        results = fit_all_cross_nest(
            df,
            quantity_cols={"es": "q_es", "eu": "q_eu", "is": "q_is", "iu": "q_iu"},
            price_cols={"es": "p_es", "eu": "p_eu", "is": "p_is", "iu": "p_iu"},
        )
        for key, fit in results.items():
            assert fit.pair == key


# ---------------------------------------------------------------------------
# sato_wald_test
# ---------------------------------------------------------------------------

class TestSatoWaldTest:
    def test_identical_estimates_chi2_is_zero(self):
        """When all four sigmas are equal, chi2 = 0 and p_value = 1."""
        result = sato_wald_test(_identical_fits(sigma=1.5))
        assert result["chi2"] == pytest.approx(0.0, abs=1e-12)
        assert result["p_value"] == pytest.approx(1.0, abs=1e-10)
        assert result["df"] == 3

    def test_analytical_chi2_no_sigma_kl(self):
        """chi2 = sum( (sigma_j - sigma_es)^2 / (se_j^2 + se_es^2) ) for j != es."""
        # sigma_es=1.5, sigma_eu=2.0, rest=1.5; se=0.1 for all
        # diff = [0.5, 0, 0]; var_diff = [0.02, 0.02, 0.02]
        # chi2 = 0.5^2/0.02 = 12.5
        ests = _identical_fits(sigma=1.5, se=0.1)
        ests = dict(ests)
        ests["eu"] = CrossNestFit(
            sigma=2.0, se=0.1, first_stage_F=None,
            n_obs=100, n_country=3, estimator="OLS", pair="eu",
        )
        result = sato_wald_test(ests)
        assert result["chi2"] == pytest.approx(12.5, rel=1e-10)
        assert result["df"] == 3

    def test_analytical_chi2_with_sigma_kl(self):
        """chi2 = sum( (sigma_j - sigma_kl)^2 / se_j^2 ) for all j when sigma_kl given."""
        # All sigmas=1.5, sigma_kl=2.0, se=0.2
        # chi2 = 4 * (0.5^2 / 0.04) = 4 * 6.25 = 25.0
        ests = _identical_fits(sigma=1.5, se=0.2)
        result = sato_wald_test(ests, sigma_kl=2.0)
        assert result["chi2"] == pytest.approx(25.0, rel=1e-10)
        assert result["df"] == 4

    def test_sigma_kl_matches_estimates_chi2_zero(self):
        """When sigma_kl equals all four sigmas, chi2 = 0."""
        ests = _identical_fits(sigma=1.8)
        result = sato_wald_test(ests, sigma_kl=1.8)
        assert result["chi2"] == pytest.approx(0.0, abs=1e-12)
        assert result["p_value"] == pytest.approx(1.0, abs=1e-10)

    def test_large_chi2_small_p_value(self):
        """Very different sigmas produce a large chi2 and near-zero p-value."""
        ests = {
            "es": CrossNestFit(sigma=0.0, se=0.01, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="es"),
            "eu": CrossNestFit(sigma=5.0, se=0.01, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="eu"),
            "is": CrossNestFit(sigma=0.0, se=0.01, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="is"),
            "iu": CrossNestFit(sigma=5.0, se=0.01, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="iu"),
        }
        result = sato_wald_test(ests)
        assert result["chi2"] > 100
        assert result["p_value"] < 0.001

    def test_nan_sigma_propagates(self):
        """Any nan sigma in an estimate yields nan chi2/p_value."""
        ests = _identical_fits()
        ests = dict(ests)
        ests["es"] = CrossNestFit(
            sigma=np.nan, se=0.1, first_stage_F=None,
            n_obs=100, n_country=3, estimator="OLS", pair="es",
        )
        result = sato_wald_test(ests)
        assert math.isnan(result["chi2"])
        assert math.isnan(result["p_value"])

    def test_zero_se_propagates_nan(self):
        """A zero SE (invalid) yields nan chi2/p_value."""
        ests = _identical_fits()
        ests = dict(ests)
        ests["eu"] = CrossNestFit(
            sigma=1.5, se=0.0, first_stage_F=None,
            n_obs=100, n_country=3, estimator="OLS", pair="eu",
        )
        result = sato_wald_test(ests)
        assert math.isnan(result["chi2"])

    def test_p_value_in_unit_interval(self):
        """p_value must lie in [0, 1] for any valid finite estimates."""
        rng = np.random.default_rng(40)
        for _ in range(10):
            sigmas = rng.uniform(0.5, 3.0, size=4)
            ses = rng.uniform(0.05, 0.5, size=4)
            ests = {
                p: CrossNestFit(
                    sigma=float(sigmas[i]), se=float(ses[i]),
                    first_stage_F=None, n_obs=100, n_country=3,
                    estimator="OLS", pair=p,
                )
                for i, p in enumerate(("es", "eu", "is", "iu"))
            }
            result = sato_wald_test(ests)
            assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# build_cross_nest_columns
# ---------------------------------------------------------------------------

class TestBuildCrossNestColumns:
    def _base_panel(self) -> pd.DataFrame:
        """Two countries, two years each — minimal panel."""
        return pd.DataFrame({
            "code": ["USA", "USA", "DEU", "DEU"],
            "year": [2000, 2001, 2000, 2001],
            "L_s": [2.0, 4.0, 3.0, 6.0],
            "L_u": [3.0, 3.0, 2.0, 4.0],
            "K_e": [1.0, 2.0, 1.0, 1.0],
            "K_i": [1.0, 1.0, 1.0, 2.0],
            "w_s": [1.0, 2.0, 1.0, 1.0],
            "w_u": [1.0, 1.0, 1.0, 2.0],
            "r_e": [2.0, 4.0, 1.0, 2.0],
            "r_i": [1.0, 2.0, 1.0, 1.0],
        })

    def test_output_has_all_new_columns(self):
        df = build_cross_nest_columns(self._base_panel())
        expected_new = {
            "dlog_ls_ke", "dlog_lu_ke", "dlog_ls_ki", "dlog_lu_ki",
            "dlog_re_ws", "dlog_re_wu", "dlog_ri_ws", "dlog_ri_wu",
        }
        assert expected_new.issubset(set(df.columns))

    def test_first_obs_per_country_is_nan(self):
        """The first period for each country must be NaN (no lag exists)."""
        df = build_cross_nest_columns(self._base_panel())
        for code in ("USA", "DEU"):
            sub = df[df["code"] == code].sort_values("year")
            assert math.isnan(sub["dlog_ls_ke"].iloc[0])

    def test_known_value_dlog_ls_ke(self):
        """dlog_ls_ke = log(L_s(t)/K_e(t)) - log(L_s(t-1)/K_e(t-1)).
        USA 2001: log(4/2) - log(2/1) = log(2) - log(2) = 0.
        """
        df = build_cross_nest_columns(self._base_panel())
        sub = df[df["code"] == "USA"].sort_values("year")
        assert sub["dlog_ls_ke"].iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_known_value_dlog_lu_ke(self):
        """dlog_lu_ke for USA 2001: log(3/2) - log(3/1) = log(1.5) - log(3) = -log(2)."""
        df = build_cross_nest_columns(self._base_panel())
        sub = df[df["code"] == "USA"].sort_values("year")
        expected = math.log(3 / 2) - math.log(3 / 1)  # = -log(2)
        assert sub["dlog_lu_ke"].iloc[1] == pytest.approx(expected, abs=1e-10)

    def test_known_value_dlog_re_ws(self):
        """dlog_re_ws for USA 2001: log(r_e/w_s) diff.
        log(4/2) - log(2/1) = log(2) - log(2) = 0.
        """
        df = build_cross_nest_columns(self._base_panel())
        sub = df[df["code"] == "USA"].sort_values("year")
        assert sub["dlog_re_ws"].iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_country_boundary_not_crossed(self):
        """Differences are computed within country, not across countries.
        Both USA and DEU should have NaN on their first obs and a valid diff on the second.
        """
        df = build_cross_nest_columns(self._base_panel())
        for code in ("USA", "DEU"):
            sub = df[df["code"] == code].sort_values("year")
            assert math.isnan(sub["dlog_ls_ke"].iloc[0])
            assert math.isfinite(sub["dlog_ls_ke"].iloc[1])

    def test_output_shape(self):
        """Row count equals input row count; all new columns are added."""
        panel = self._base_panel()
        out = build_cross_nest_columns(panel)
        assert out.shape[0] == panel.shape[0]
        assert out.shape[1] == panel.shape[1] + 8  # eight new columns

    def test_sorted_by_code_year(self):
        """Output is sorted by (code, year) even if input is shuffled."""
        panel = self._base_panel().sample(frac=1, random_state=0)
        out = build_cross_nest_columns(panel)
        sorted_out = out.sort_values(["code", "year"])
        pd.testing.assert_frame_equal(out.reset_index(drop=True), sorted_out.reset_index(drop=True))

    def test_symmetric_dlog_ri_ws(self):
        """dlog_ri_ws value for DEU 2001: log(r_i(01)/w_s(01)) - log(r_i(00)/w_s(00)).
        DEU: r_i=[1,1], w_s=[1,1] -> diff = log(1/1) - log(1/1) = 0.
        """
        df = build_cross_nest_columns(self._base_panel())
        sub = df[df["code"] == "DEU"].sort_values("year")
        assert sub["dlog_ri_ws"].iloc[1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# crossnest_summary
# ---------------------------------------------------------------------------

class TestCrossNestSummary:
    def _ests(self) -> dict[str, CrossNestFit]:
        return {
            "es": CrossNestFit(sigma=1.2, se=0.1, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="es"),
            "eu": CrossNestFit(sigma=1.3, se=0.15, first_stage_F=None, n_obs=95, n_country=3, estimator="OLS", pair="eu"),
            "is": CrossNestFit(sigma=1.1, se=0.12, first_stage_F=None, n_obs=100, n_country=3, estimator="OLS", pair="is"),
            "iu": CrossNestFit(sigma=1.4, se=0.08, first_stage_F=20.0, n_obs=90, n_country=3, estimator="2SLS", pair="iu"),
        }

    def test_shape(self):
        """Summary has exactly 4 rows and the expected columns."""
        summ = crossnest_summary(self._ests())
        assert summ.shape[0] == 4
        expected_cols = {"pair", "description", "estimator", "sigma", "se", "first_stage_F", "n_obs", "n_country"}
        assert expected_cols.issubset(set(summ.columns))

    def test_pairs_in_order(self):
        """Rows are in canonical order es, eu, is, iu."""
        summ = crossnest_summary(self._ests())
        assert list(summ["pair"]) == ["es", "eu", "is", "iu"]

    def test_descriptions(self):
        """Description strings match the paper nomenclature."""
        summ = crossnest_summary(self._ests())
        desc = list(summ["description"])
        assert desc == [
            "equipment-skilled",
            "equipment-unskilled",
            "intangibles-skilled",
            "intangibles-unskilled",
        ]

    def test_sigma_values_preserved(self):
        """Sigma values from the input CrossNestFit appear in the table."""
        ests = self._ests()
        summ = crossnest_summary(ests)
        for _, row in summ.iterrows():
            assert row["sigma"] == ests[row["pair"]].sigma

    def test_first_stage_f_iu_is_present(self):
        """A non-None first_stage_F is preserved in the summary table."""
        summ = crossnest_summary(self._ests())
        iu_row = summ[summ["pair"] == "iu"].iloc[0]
        assert iu_row["first_stage_F"] == pytest.approx(20.0)
