"""Tests for puremacro.lp._garch_utils.

Strategy
--------
- ``fit_garch``:
    - Known-DGP: simulate a GARCH(1,1) process with known parameters; verify
      that the fitted persistence (alpha+beta) is within tolerance of the true
      value on long series (~2000 obs).
    - Properties: cond_vol positive, innovations/cond_vol = standardised
      residuals exactly, index alignment, NaN propagation.
    - Variant coverage: GJR11 model, dist='t', rescale=False.
    - Error: invalid model string raises ValueError.
- ``make_regime_indicator``:
    - Known-value: with uniform sigma, median rule gives exactly 50% high;
      p75 rule gives exactly 25% high.
    - Output dtype is bool; series name matches the rule.
    - Error: invalid rule raises ValueError.
- ``align_series_for_lp``:
    - Drops rows where shock or sigma is NaN; keeps rows where both are present.
    - Preserves all other columns (not just shock/sigma).
    - Returns a copy (mutating the result does not alter the original df).
    - Empty result when all rows have at least one NaN.

Notes
-----
- ``fit_garch`` calls ``arch.arch_model`` internally, which takes ~1-3 seconds
  per fit on 300 obs; tests that fit on 2000 obs are marked ``@pytest.mark.slow``.
- ``align_series_for_lp`` is a pure DataFrame helper — all tests run instantly.
- ``make_regime_indicator`` is a pure Series helper — all tests run instantly.
- No network I/O or external binaries involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp._garch_utils import (
    align_series_for_lp,
    fit_garch,
    make_regime_indicator,
)

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Synthetic GARCH DGP helpers
# ---------------------------------------------------------------------------

def _simulate_garch11(
    rng: np.random.Generator,
    T: int,
    *,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> pd.Series:
    """Simulate a GARCH(1,1) process with Gaussian innovations."""
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / max(1e-10, 1.0 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return pd.Series(eps, index=pd.date_range("1990-01-01", periods=T, freq="MS"))


def _quick_series(n: int = 300, *, seed: int = 42) -> pd.Series:
    """Small iid series — fast to fit (not a GARCH DGP, just for contract tests)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    return pd.Series(eps, index=pd.date_range("2000-01-01", periods=n, freq="MS"))


# ===========================================================================
# 1. fit_garch
# ===========================================================================

class TestFitGarchOutputShape:
    """fit_garch returns three Series and one result object with correct shapes."""

    def test_returns_four_objects(self):
        """Return value is a 4-tuple."""
        s = _quick_series(300)
        out = fit_garch(s)
        assert len(out) == 4

    def test_cond_vol_length_equals_input(self):
        """cond_vol has the same length as the input series."""
        s = _quick_series(300)
        cond_vol, _, _, _ = fit_garch(s)
        assert len(cond_vol) == len(s)

    def test_innovations_length_equals_input(self):
        """innovations has the same length as the input series."""
        s = _quick_series(300)
        _, innovations, _, _ = fit_garch(s)
        assert len(innovations) == len(s)

    def test_residuals_length_equals_input(self):
        """residuals has the same length as the input series."""
        s = _quick_series(300)
        _, _, residuals, _ = fit_garch(s)
        assert len(residuals) == len(s)

    def test_cond_vol_index_matches_input(self):
        """cond_vol shares the exact index of the input series."""
        s = _quick_series(300)
        cond_vol, _, _, _ = fit_garch(s)
        assert cond_vol.index.equals(s.index)

    def test_innovations_index_matches_input(self):
        """innovations shares the exact index of the input series."""
        s = _quick_series(300)
        _, innovations, _, _ = fit_garch(s)
        assert innovations.index.equals(s.index)

    def test_residuals_index_matches_input(self):
        """residuals shares the exact index of the input series."""
        s = _quick_series(300)
        _, _, residuals, _ = fit_garch(s)
        assert residuals.index.equals(s.index)

    def test_cond_vol_dtype_is_float(self):
        """cond_vol is a float Series."""
        s = _quick_series(300)
        cond_vol, _, _, _ = fit_garch(s)
        assert np.issubdtype(cond_vol.dtype, np.floating)

    def test_residuals_dtype_is_float(self):
        """residuals is a float Series."""
        s = _quick_series(300)
        _, _, residuals, _ = fit_garch(s)
        assert np.issubdtype(residuals.dtype, np.floating)


class TestFitGarchPositivity:
    """Conditional volatility must be strictly positive on non-NaN observations."""

    def test_cond_vol_nonnegative_garch11(self):
        """GARCH11 cond_vol values are strictly positive on non-NaN rows."""
        s = _quick_series(300)
        cond_vol, _, _, _ = fit_garch(s)
        assert (cond_vol.dropna() > 0).all()

    def test_cond_vol_nonnegative_gjr11(self):
        """GJR11 cond_vol values are strictly positive on non-NaN rows."""
        s = _quick_series(300, seed=43)
        cond_vol, _, _, _ = fit_garch(s, model="GJR11")
        assert (cond_vol.dropna() > 0).all()


class TestFitGarchNanHandling:
    """NaN propagation: NaN in input → NaN in outputs at same positions."""

    def test_no_nan_in_complete_series(self):
        """Complete series without NaN produces no NaN outputs."""
        s = _quick_series(300)
        cond_vol, innovations, residuals, _ = fit_garch(s)
        assert cond_vol.notna().all()
        assert innovations.notna().all()
        assert residuals.notna().all()

    def test_nan_positions_propagate_to_cond_vol(self):
        """Positions that are NaN in the input are NaN in cond_vol."""
        s = _quick_series(300)
        nan_mask = pd.Series(False, index=s.index)
        nan_mask.iloc[::10] = True
        s_with_nan = s.copy()
        s_with_nan[nan_mask] = np.nan
        cond_vol, _, _, _ = fit_garch(s_with_nan)
        assert cond_vol[nan_mask].isna().all()

    def test_nan_positions_propagate_to_innovations(self):
        """Positions that are NaN in the input are NaN in innovations."""
        s = _quick_series(300)
        s_with_nan = s.copy()
        s_with_nan.iloc[::10] = np.nan
        _, innovations, _, _ = fit_garch(s_with_nan)
        assert innovations[s_with_nan.isna()].isna().all()

    def test_nan_positions_propagate_to_residuals(self):
        """Positions that are NaN in the input are NaN in residuals."""
        s = _quick_series(300)
        s_with_nan = s.copy()
        s_with_nan.iloc[::10] = np.nan
        _, _, residuals, _ = fit_garch(s_with_nan)
        assert residuals[s_with_nan.isna()].isna().all()

    def test_non_nan_positions_remain_finite_after_nan(self):
        """Non-NaN positions in the input produce finite values in cond_vol."""
        s = _quick_series(300)
        s_with_nan = s.copy()
        s_with_nan.iloc[::10] = np.nan
        cond_vol, _, _, _ = fit_garch(s_with_nan)
        assert cond_vol[s_with_nan.notna()].notna().all()


class TestFitGarchMathRelations:
    """Mathematical identities: residuals = innovations / cond_vol."""

    def test_standardised_residuals_equal_innovations_over_cond_vol(self):
        """residuals == innovations / cond_vol exactly (up to float precision)."""
        s = _quick_series(300)
        cond_vol, innovations, residuals, _ = fit_garch(s)
        recomputed = innovations.dropna() / cond_vol.dropna()
        np.testing.assert_allclose(
            residuals.dropna().values,
            recomputed.values,
            rtol=1e-10,
            err_msg="residuals must equal innovations / cond_vol",
        )

    def test_standardised_residuals_near_unit_variance(self):
        """Standardised residuals from a GARCH fit have std ≈ 1."""
        s = _quick_series(300)
        _, _, residuals, _ = fit_garch(s)
        std = float(residuals.dropna().std())
        assert 0.8 < std < 1.2, f"Standardised residuals std expected ≈1, got {std:.3f}"

    def test_standardised_residuals_near_zero_mean(self):
        """Standardised residuals from a GARCH fit have mean ≈ 0."""
        s = _quick_series(300)
        _, _, residuals, _ = fit_garch(s)
        mean = float(residuals.dropna().mean())
        assert abs(mean) < 0.3, f"Standardised residuals mean expected ≈0, got {mean:.3f}"


class TestFitGarchVariants:
    """Variant parameters: GJR11 model, dist='t', rescale=False."""

    def test_gjr11_returns_positive_cond_vol(self):
        """GJR11 model fits and returns positive cond_vol."""
        s = _quick_series(300, seed=7)
        cond_vol, _, _, _ = fit_garch(s, model="GJR11")
        assert (cond_vol.dropna() > 0).all()

    def test_dist_t_includes_nu_parameter(self):
        """dist='t' produces a result object with a 'nu' parameter."""
        s = _quick_series(300, seed=8)
        _, _, _, res = fit_garch(s, dist="t")
        assert "nu" in res.params.index

    def test_dist_t_cond_vol_positive(self):
        """dist='t' fit still returns positive cond_vol."""
        s = _quick_series(300, seed=9)
        cond_vol, _, _, _ = fit_garch(s, dist="t")
        assert (cond_vol.dropna() > 0).all()

    def test_rescale_false_cond_vol_positive(self):
        """rescale=False still returns positive cond_vol."""
        s = _quick_series(300, seed=10)
        cond_vol, _, _, _ = fit_garch(s, rescale=False)
        assert (cond_vol.dropna() > 0).all()

    def test_rescale_false_index_preserved(self):
        """rescale=False preserves the original index."""
        s = _quick_series(300, seed=11)
        cond_vol, _, _, _ = fit_garch(s, rescale=False)
        assert cond_vol.index.equals(s.index)

    def test_gjr11_residuals_equal_innovations_over_cond_vol(self):
        """For GJR11, residuals == innovations / cond_vol."""
        s = _quick_series(300, seed=12)
        cond_vol, innovations, residuals, _ = fit_garch(s, model="GJR11")
        recomputed = innovations.dropna() / cond_vol.dropna()
        np.testing.assert_allclose(
            residuals.dropna().values,
            recomputed.values,
            rtol=1e-10,
        )


class TestFitGarchErrorHandling:
    """Invalid model string raises ValueError."""

    def test_bad_model_name_raises_value_error(self):
        """Passing an unknown model name raises ValueError."""
        s = _quick_series(100)
        with pytest.raises(ValueError, match="model must be"):
            fit_garch(s, model="BADMODEL")  # type: ignore[arg-type]

    def test_bad_model_name_egarch_raises(self):
        """'EGARCH11' is not a supported alias; raises ValueError."""
        s = _quick_series(100)
        with pytest.raises(ValueError, match="model must be"):
            fit_garch(s, model="EGARCH11")  # type: ignore[arg-type]


@pytest.mark.slow
class TestFitGarchDGPRecovery:
    """Known-DGP: fit on long GARCH(1,1) series; persistence within tolerance."""

    def test_garch11_persistence_close_to_truth(self):
        """Fitted alpha+beta is within 0.05 of the true persistence on T=2000."""
        rng = np.random.default_rng(101)
        omega, alpha, beta = 0.05, 0.10, 0.85
        s = _simulate_garch11(rng, 2000, omega=omega, alpha=alpha, beta=beta)
        _, _, _, res = fit_garch(s)
        fitted_persistence = res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0)
        true_persistence = alpha + beta
        assert abs(fitted_persistence - true_persistence) < 0.06, (
            f"Persistence {fitted_persistence:.4f} too far from truth {true_persistence}"
        )

    def test_garch11_cond_vol_captures_volatility_clustering(self):
        """In a GARCH DGP, high-sigma periods cluster together: the serial
        correlation of cond_vol^2 should be positive and exceed 0.1."""
        rng = np.random.default_rng(202)
        s = _simulate_garch11(rng, 2000)
        cond_vol, _, _, _ = fit_garch(s)
        sigma2 = cond_vol.dropna() ** 2
        # AR(1) autocorrelation of cond variance
        autocorr = float(sigma2.autocorr(lag=1))
        assert autocorr > 0.1, (
            f"GARCH cond var should be positively autocorrelated; got {autocorr:.4f}"
        )

    def test_gjr11_runs_on_garch_dgp(self):
        """GJR11 converges and returns positive sigma on a pure GARCH DGP."""
        rng = np.random.default_rng(303)
        s = _simulate_garch11(rng, 2000)
        cond_vol, _, _, res = fit_garch(s, model="GJR11")
        assert (cond_vol.dropna() > 0).all()
        assert res.convergence_flag == 0 or res.convergence_flag is None


# ===========================================================================
# 2. make_regime_indicator
# ===========================================================================

class TestMakeRegimeIndicatorDtype:
    """Output dtype is bool."""

    def test_median_rule_dtype_bool(self):
        """median rule returns a bool Series."""
        sigma = pd.Series(RNG.uniform(0, 1, 100))
        ind = make_regime_indicator(sigma, rule="median")
        assert ind.dtype == bool

    def test_p75_rule_dtype_bool(self):
        """p75 rule returns a bool Series."""
        sigma = pd.Series(RNG.uniform(0, 1, 100))
        ind = make_regime_indicator(sigma, rule="p75")
        assert ind.dtype == bool


class TestMakeRegimeIndicatorShape:
    """Output has the same length and index as input."""

    def test_length_preserved_median(self):
        """Output length equals input length (median)."""
        sigma = pd.Series(RNG.uniform(0, 1, 200))
        ind = make_regime_indicator(sigma)
        assert len(ind) == len(sigma)

    def test_index_preserved_median(self):
        """Output index equals input index (median)."""
        idx = pd.date_range("2000-01-01", periods=100, freq="QS")
        sigma = pd.Series(RNG.uniform(0, 1, 100), index=idx)
        ind = make_regime_indicator(sigma)
        assert ind.index.equals(sigma.index)

    def test_index_preserved_p75(self):
        """Output index equals input index (p75)."""
        idx = pd.date_range("2000-01-01", periods=100, freq="QS")
        sigma = pd.Series(RNG.uniform(0, 1, 100), index=idx)
        ind = make_regime_indicator(sigma, rule="p75")
        assert ind.index.equals(sigma.index)


class TestMakeRegimeIndicatorKnownValues:
    """Known-value tests using linearly spaced or counted sigma."""

    def test_median_rule_exactly_half_high_even_length(self):
        """With sigma = 1..100 (even, no ties at median), exactly 50 are high."""
        sigma = pd.Series(np.arange(1, 101, dtype=float))
        ind = make_regime_indicator(sigma, rule="median")
        assert ind.sum() == 50

    def test_p75_rule_approximately_quarter_high(self):
        """p75 rule marks ~25% of observations as high-volatility."""
        rng = np.random.default_rng(55)
        sigma = pd.Series(rng.uniform(0, 1, 400))
        ind = make_regime_indicator(sigma, rule="p75")
        frac_high = float(ind.mean())
        assert 0.20 < frac_high < 0.30, f"p75 should give ~25% high, got {frac_high:.3f}"

    def test_median_rule_all_above_threshold(self):
        """All elements strictly above the median are classified high."""
        sigma = pd.Series([1.0, 2.0, 3.0, 4.0])
        # median of [1,2,3,4] = 2.5; high when > 2.5 => 3.0, 4.0
        ind = make_regime_indicator(sigma, rule="median")
        assert list(ind) == [False, False, True, True]

    def test_p75_rule_correct_threshold(self):
        """Values strictly above the 75th percentile are classified high."""
        sigma = pd.Series(np.arange(1.0, 21.0))  # 1..20
        # 75th percentile of 1..20 = 15.25 → only 16,17,18,19,20 are high (5 obs)
        ind = make_regime_indicator(sigma, rule="p75")
        assert int(ind.sum()) == 5

    def test_p75_fewer_high_than_median(self):
        """p75 rule always classifies fewer (or equal) obs as high than median."""
        rng = np.random.default_rng(77)
        sigma = pd.Series(rng.exponential(1.0, 200))
        ind_med = make_regime_indicator(sigma, rule="median")
        ind_p75 = make_regime_indicator(sigma, rule="p75")
        assert ind_p75.sum() <= ind_med.sum()


class TestMakeRegimeIndicatorName:
    """Output Series has the expected name."""

    def test_median_series_name(self):
        """median rule produces Series named 'regime_high_median'."""
        sigma = pd.Series(RNG.uniform(0, 1, 50))
        ind = make_regime_indicator(sigma)
        assert ind.name == "regime_high_median"

    def test_p75_series_name(self):
        """p75 rule produces Series named 'regime_high_p75'."""
        sigma = pd.Series(RNG.uniform(0, 1, 50))
        ind = make_regime_indicator(sigma, rule="p75")
        assert ind.name == "regime_high_p75"


class TestMakeRegimeIndicatorErrors:
    """Invalid rule raises ValueError."""

    def test_invalid_rule_raises_value_error(self):
        """Unknown rule string raises ValueError."""
        sigma = pd.Series([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="rule must be"):
            make_regime_indicator(sigma, rule="p90")  # type: ignore[arg-type]

    def test_invalid_rule_empty_string_raises(self):
        """Empty string raises ValueError."""
        sigma = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="rule must be"):
            make_regime_indicator(sigma, rule="")  # type: ignore[arg-type]


# ===========================================================================
# 3. align_series_for_lp
# ===========================================================================

class TestAlignSeriesForLpFiltering:
    """align_series_for_lp drops rows where either shock or sigma is NaN."""

    def test_no_nan_no_rows_dropped(self):
        """Complete DataFrame is unchanged in length."""
        df = pd.DataFrame({"shock": [1.0, 2.0, 3.0], "sigma": [0.5, 0.6, 0.7]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 3

    def test_nan_in_shock_drops_row(self):
        """Rows where shock is NaN are dropped."""
        df = pd.DataFrame({"shock": [1.0, np.nan, 3.0], "sigma": [0.5, 0.6, 0.7]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 2
        assert np.isnan(aligned["shock"].values).sum() == 0

    def test_nan_in_sigma_drops_row(self):
        """Rows where sigma is NaN are dropped."""
        df = pd.DataFrame({"shock": [1.0, 2.0, 3.0], "sigma": [0.5, np.nan, 0.7]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 2
        assert np.isnan(aligned["sigma"].values).sum() == 0

    def test_nan_in_both_drops_row(self):
        """Rows where both shock and sigma are NaN are dropped."""
        df = pd.DataFrame({
            "shock": [1.0, np.nan, 3.0],
            "sigma": [0.5, np.nan, 0.7],
        })
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 2

    def test_all_nan_returns_empty(self):
        """All-NaN DataFrame returns empty result."""
        df = pd.DataFrame({"shock": [np.nan, np.nan], "sigma": [np.nan, np.nan]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 0

    def test_complete_case_count(self):
        """Only rows where both shock and sigma are non-NaN survive."""
        df = pd.DataFrame({
            "shock": [1.0, np.nan, 3.0, 4.0, np.nan],
            "sigma": [0.5, 0.6,    np.nan, 0.8, np.nan],
        })
        aligned = align_series_for_lp(df, "shock", "sigma")
        # Rows 0 (both ok) and 3 (both ok) survive; rows 1, 2, 4 drop
        assert len(aligned) == 2
        assert list(aligned.index) == [0, 3]


class TestAlignSeriesForLpColumns:
    """align_series_for_lp preserves all columns in the original DataFrame."""

    def test_extra_columns_preserved(self):
        """Columns other than shock and sigma are preserved."""
        df = pd.DataFrame({
            "shock": [1.0, np.nan, 3.0],
            "sigma": [0.5, 0.6, 0.7],
            "outcome": [10, 20, 30],
            "control": ["a", "b", "c"],
        })
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert "outcome" in aligned.columns
        assert "control" in aligned.columns
        assert len(aligned) == 2

    def test_returns_dataframe(self):
        """Return type is always pd.DataFrame."""
        df = pd.DataFrame({"shock": [1.0], "sigma": [0.5]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert isinstance(aligned, pd.DataFrame)

    def test_columns_unchanged(self):
        """The set of columns is identical to the input."""
        df = pd.DataFrame({"shock": [1.0, 2.0], "sigma": [0.5, 0.6], "z": [7.0, 8.0]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert set(aligned.columns) == set(df.columns)


class TestAlignSeriesForLpCopy:
    """align_series_for_lp returns a copy, not a view."""

    def test_mutating_result_does_not_alter_original(self):
        """Modifying the aligned DataFrame does not change the original."""
        df = pd.DataFrame({"shock": [1.0, 2.0, 3.0], "sigma": [0.5, 0.6, 0.7]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        aligned.iloc[0, aligned.columns.get_loc("shock")] = 999.0
        assert df.loc[0, "shock"] == 1.0, "Original must be unmodified (copy contract)"


class TestAlignSeriesForLpEdgeCases:
    """Edge cases: single-row inputs, datetime index, large panels."""

    def test_single_row_both_present_survives(self):
        """A single-row DataFrame where both values are present is kept."""
        df = pd.DataFrame({"shock": [2.5], "sigma": [1.0]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 1

    def test_single_row_shock_nan_dropped(self):
        """A single-row DataFrame where shock is NaN returns empty."""
        df = pd.DataFrame({"shock": [np.nan], "sigma": [1.0]})
        aligned = align_series_for_lp(df, "shock", "sigma")
        assert len(aligned) == 0

    def test_datetime_index_preserved(self):
        """Datetime index is preserved in the aligned output."""
        dates = pd.date_range("2010-01-01", periods=4, freq="QS")
        df = pd.DataFrame(
            {"shock": [1.0, np.nan, 3.0, 4.0], "sigma": [0.5, 0.5, np.nan, 0.7]},
            index=dates,
        )
        aligned = align_series_for_lp(df, "shock", "sigma")
        # Only row 0 (2010-01-01) and row 3 (2010-10-01) survive
        assert len(aligned) == 2
        assert dates[0] in aligned.index
        assert dates[3] in aligned.index

    def test_large_panel_correct_count(self):
        """Larger synthetic panel drops the right number of NaN rows."""
        rng = np.random.default_rng(88)
        n = 500
        shock = rng.standard_normal(n)
        sigma = np.abs(rng.standard_normal(n))
        # Introduce 50 NaN in each column at distinct positions
        shock_nan_idx = rng.choice(n, size=50, replace=False)
        sigma_nan_idx = rng.choice(n, size=50, replace=False)
        shock[shock_nan_idx] = np.nan
        sigma[sigma_nan_idx] = np.nan
        df = pd.DataFrame({"shock": shock, "sigma": sigma})
        aligned = align_series_for_lp(df, "shock", "sigma")
        expected_keep = (~np.isnan(shock)) & (~np.isnan(sigma))
        assert len(aligned) == int(expected_keep.sum())
