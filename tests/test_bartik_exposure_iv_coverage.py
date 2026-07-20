"""Coverage tests for puremacro.bartik.exposure_iv.

Tests cover:
- build_exposure_iv: known-value, shape/dtype, error handling, edge cases
- rotemberg_weights: sum-to-one property, sign, error handling, edge cases
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from puremacro.bartik.exposure_iv import build_exposure_iv, rotemberg_weights


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_shares(
    counties: list[str],
    industries: list[str],
    share_values: list[float],
) -> pd.DataFrame:
    """Build a shares DataFrame with explicit (county, naics, share) triples."""
    rows = []
    for county, naics, share in zip(counties, industries, share_values):
        rows.append({"county_fips": county, "naics": naics, "share": share})
    return pd.DataFrame(rows)


def _make_sensitivities(industries: list[str], betas: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"naics": industries, "shrunk_beta": betas})


def _make_shock_variances(
    industries: list[str], variances: list[float], col: str = "sigma_sq_nat"
) -> pd.DataFrame:
    return pd.DataFrame({"naics": industries, col: variances})


# ===========================================================================
# build_exposure_iv — known-value tests
# ===========================================================================

class TestBuildExposureIVKnownValues:
    def test_single_county_single_industry(self):
        """E_c = share * beta when one county and one industry."""
        shares = _make_shares(["06001"], ["31"], [0.6])
        sens = _make_sensitivities(["31"], [2.0])
        result = build_exposure_iv(shares, sens)
        assert len(result) == 1
        assert result.iloc[0]["county_fips"] == "06001"
        # E = 0.6 * 2.0 = 1.2
        assert abs(result.iloc[0]["exposure_c"] - 1.2) < 1e-12

    def test_single_county_two_industries(self):
        """E_c = sum of w_i * beta_i across industries."""
        shares = _make_shares(
            ["06001", "06001"],
            ["31", "32"],
            [0.4, 0.6],
        )
        sens = _make_sensitivities(["31", "32"], [1.0, 3.0])
        result = build_exposure_iv(shares, sens)
        assert len(result) == 1
        # E = 0.4*1.0 + 0.6*3.0 = 0.4 + 1.8 = 2.2
        assert abs(result.iloc[0]["exposure_c"] - 2.2) < 1e-12

    def test_two_counties_independent(self):
        """Two counties, non-overlapping industries; each county computed correctly."""
        shares = _make_shares(
            ["06001", "06003"],
            ["31",    "32"],
            [1.0,     1.0],
        )
        sens = _make_sensitivities(["31", "32"], [2.0, 5.0])
        result = build_exposure_iv(shares, sens).set_index("county_fips")
        assert abs(result.loc["06001", "exposure_c"] - 2.0) < 1e-12
        assert abs(result.loc["06003", "exposure_c"] - 5.0) < 1e-12

    def test_zero_share_gives_zero_contribution(self):
        """An industry with share=0 contributes nothing even if beta is large."""
        shares = _make_shares(["06001", "06001"], ["31", "32"], [1.0, 0.0])
        sens = _make_sensitivities(["31", "32"], [1.0, 999.0])
        result = build_exposure_iv(shares, sens)
        assert abs(result.iloc[0]["exposure_c"] - 1.0) < 1e-12

    def test_zero_beta_gives_zero_contribution(self):
        """An industry with beta=0 contributes nothing regardless of share."""
        shares = _make_shares(["06001", "06001"], ["31", "32"], [0.5, 0.5])
        sens = _make_sensitivities(["31", "32"], [0.0, 0.0])
        result = build_exposure_iv(shares, sens)
        assert abs(result.iloc[0]["exposure_c"] - 0.0) < 1e-12

    def test_negative_beta_gives_negative_exposure(self):
        """Negative sensitivity produces negative exposure."""
        shares = _make_shares(["06001"], ["31"], [1.0])
        sens = _make_sensitivities(["31"], [-3.5])
        result = build_exposure_iv(shares, sens)
        assert result.iloc[0]["exposure_c"] < 0

    def test_output_columns(self):
        """Output must have exactly county_fips and exposure_c."""
        shares = _make_shares(["06001"], ["31"], [0.5])
        sens = _make_sensitivities(["31"], [1.0])
        result = build_exposure_iv(shares, sens)
        assert set(result.columns) == {"county_fips", "exposure_c"}

    def test_output_one_row_per_county(self):
        """Output has one row per unique county."""
        counties = ["A", "A", "B", "B", "C"]
        industries = ["X", "Y", "X", "Z", "Y"]
        shares_df = _make_shares(counties, industries, [0.5, 0.5, 0.3, 0.7, 1.0])
        sens = _make_sensitivities(["X", "Y", "Z"], [1.0, 2.0, 3.0])
        result = build_exposure_iv(shares_df, sens)
        assert len(result) == 3

    def test_county_fips_dtype_coerced_to_str(self):
        """county_fips is coerced to str even if provided as int."""
        shares = pd.DataFrame(
            {"county_fips": [6001], "naics": ["31"], "share": [0.5]}
        )
        sens = _make_sensitivities(["31"], [2.0])
        result = build_exposure_iv(shares, sens)
        # string dtype: object under pandas 2, the string dtype under pandas 3
        assert pd.api.types.is_string_dtype(result["county_fips"])

    def test_naics_dtype_coerced_to_str(self):
        """naics is coerced to str even if provided as int in both DataFrames."""
        shares = pd.DataFrame(
            {"county_fips": ["A"], "naics": [31], "share": [0.5]}
        )
        sens = pd.DataFrame({"naics": [31], "shrunk_beta": [4.0]})
        result = build_exposure_iv(shares, sens)
        # Should match and produce E = 0.5 * 4.0 = 2.0
        assert abs(result.iloc[0]["exposure_c"] - 2.0) < 1e-12


# ===========================================================================
# build_exposure_iv — unmatched industry handling
# ===========================================================================

class TestBuildExposureIVUnmatched:
    def test_unmatched_below_1pct_treated_as_zero(self):
        """Industry with no matching beta contributes 0 (fillna path)."""
        # 1 out of 1000 rows unmatched → 0.1% < 1% threshold
        n = 1000
        counties = [str(i) for i in range(n)]
        # First n-1 matched, last one unmatched
        industries = ["31"] * (n - 1) + ["UNMATCHED"]
        shares_val = [0.001] * n
        shares_df = _make_shares(counties, industries, shares_val)
        sens = _make_sensitivities(["31"], [1.0])
        # Should not raise or warn
        result = build_exposure_iv(shares_df, sens)
        assert len(result) == n

    def test_unmatched_between_1pct_and_5pct_logs_warning(self, caplog):
        """Between 1% and 5% unmatched: a warning is logged."""
        # 30 out of 1000 rows unmatched → 3% > 1% but < 5%
        n = 1000
        industries = ["31"] * 970 + ["MISS"] * 30
        counties = [str(i) for i in range(n)]
        shares_val = [0.001] * n
        shares_df = _make_shares(counties, industries, shares_val)
        sens = _make_sensitivities(["31"], [1.0])
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.exposure_iv"):
            result = build_exposure_iv(shares_df, sens)
        assert any("no matching" in r.message.lower() for r in caplog.records)
        assert len(result) == n

    def test_unmatched_above_5pct_raises_valueerror(self):
        """More than 5% unmatched raises ValueError."""
        shares = _make_shares(
            ["A", "A", "A", "A", "A", "A"],
            ["31", "MISS1", "MISS2", "MISS3", "MISS4", "MISS5"],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        sens = _make_sensitivities(["31"], [1.0])
        with pytest.raises(ValueError, match="no matching"):
            build_exposure_iv(shares, sens)


# ===========================================================================
# build_exposure_iv — error handling
# ===========================================================================

class TestBuildExposureIVErrors:
    def test_duplicate_county_naics_raises(self):
        """Duplicate (county_fips, naics) rows must raise ValueError."""
        shares = pd.DataFrame([
            {"county_fips": "A", "naics": "31", "share": 0.4},
            {"county_fips": "A", "naics": "31", "share": 0.6},  # duplicate
        ])
        sens = _make_sensitivities(["31"], [1.0])
        with pytest.raises(ValueError, match="duplicate"):
            build_exposure_iv(shares, sens)

    def test_empty_shares_returns_empty_dataframe(self):
        """Empty shares input produces empty output."""
        shares = pd.DataFrame(columns=["county_fips", "naics", "share"])
        sens = _make_sensitivities(["31"], [1.0])
        result = build_exposure_iv(shares, sens)
        assert len(result) == 0
        assert "exposure_c" in result.columns


# ===========================================================================
# build_exposure_iv — does not mutate inputs
# ===========================================================================

class TestBuildExposureIVImmutability:
    def test_shares_not_mutated(self):
        """build_exposure_iv should not mutate the input shares DataFrame."""
        shares = _make_shares(["A"], ["31"], [0.5])
        original_cols = list(shares.columns)
        original_values = shares.copy()
        build_exposure_iv(shares, _make_sensitivities(["31"], [1.0]))
        pd.testing.assert_frame_equal(shares, original_values)
        assert list(shares.columns) == original_cols

    def test_sensitivities_not_mutated(self):
        """build_exposure_iv should not mutate the input sensitivities DataFrame."""
        shares = _make_shares(["A"], ["31"], [0.5])
        sens = _make_sensitivities(["31"], [2.0])
        sens_copy = sens.copy()
        build_exposure_iv(shares, sens)
        pd.testing.assert_frame_equal(sens, sens_copy)


# ===========================================================================
# rotemberg_weights — sum-to-one property (key mathematical contract)
# ===========================================================================

class TestRotembergWeightsSumToOne:
    def test_alpha_sums_to_one_simple(self):
        """By construction, sum of alpha_k must equal 1.0."""
        shares = _make_shares(
            ["A", "A", "B", "B"],
            ["31", "32", "31", "32"],
            [0.6, 0.4, 0.3, 0.7],
        )
        sens = _make_sensitivities(["31", "32"], [1.5, 2.0])
        sv = _make_shock_variances(["31", "32"], [0.1, 0.2])
        result = rotemberg_weights(shares, sens, sv)
        total = result["alpha"].sum()
        assert abs(total - 1.0) < 1e-12

    def test_alpha_sums_to_one_three_industries(self):
        """Three-industry case still sums to 1."""
        shares = _make_shares(
            ["A", "A", "A", "B", "B", "B"],
            ["X", "Y", "Z", "X", "Y", "Z"],
            [0.5, 0.3, 0.2, 0.4, 0.4, 0.2],
        )
        sens = _make_sensitivities(["X", "Y", "Z"], [1.0, 2.0, 0.5])
        sv = _make_shock_variances(["X", "Y", "Z"], [0.1, 0.2, 0.3])
        result = rotemberg_weights(shares, sens, sv)
        assert abs(result["alpha"].sum() - 1.0) < 1e-12

    def test_alpha_sums_to_one_single_industry(self):
        """Single industry: its alpha must equal 1."""
        shares = _make_shares(["A", "B"], ["31", "31"], [0.5, 0.5])
        sens = _make_sensitivities(["31"], [3.0])
        sv = _make_shock_variances(["31"], [0.5])
        result = rotemberg_weights(shares, sens, sv)
        assert abs(result.iloc[0]["alpha"] - 1.0) < 1e-12


# ===========================================================================
# rotemberg_weights — known-value tests
# ===========================================================================

class TestRotembergWeightsKnownValues:
    def test_two_industries_equal_weight_same_beta_and_var(self):
        """With symmetric setup, each industry gets alpha=0.5."""
        shares = _make_shares(
            ["A", "A", "B", "B"],
            ["X", "Y", "X", "Y"],
            [0.5, 0.5, 0.5, 0.5],
        )
        sens = _make_sensitivities(["X", "Y"], [1.0, 1.0])
        sv = _make_shock_variances(["X", "Y"], [1.0, 1.0])
        result = rotemberg_weights(shares, sens, sv).set_index("naics")
        assert abs(result.loc["X", "alpha"] - 0.5) < 1e-12
        assert abs(result.loc["Y", "alpha"] - 0.5) < 1e-12

    def test_higher_beta_gets_larger_alpha(self):
        """Industry with larger beta (and same sum_w_sq and sigma_sq) gets larger alpha."""
        shares = _make_shares(
            ["A", "A"],
            ["X", "Y"],
            [0.5, 0.5],
        )
        sens = _make_sensitivities(["X", "Y"], [2.0, 1.0])  # X has bigger beta
        sv = _make_shock_variances(["X", "Y"], [1.0, 1.0])
        result = rotemberg_weights(shares, sens, sv).set_index("naics")
        assert result.loc["X", "alpha"] > result.loc["Y", "alpha"]

    def test_higher_variance_gets_larger_alpha(self):
        """Industry with larger sigma_sq (and same beta, sum_w_sq) gets larger alpha."""
        shares = _make_shares(
            ["A", "A"],
            ["X", "Y"],
            [0.5, 0.5],
        )
        sens = _make_sensitivities(["X", "Y"], [1.0, 1.0])
        sv = _make_shock_variances(["X", "Y"], [2.0, 1.0])  # X has bigger variance
        result = rotemberg_weights(shares, sens, sv).set_index("naics")
        assert result.loc["X", "alpha"] > result.loc["Y", "alpha"]

    def test_output_columns(self):
        """Output must have naics, alpha, beta, sum_w_sq columns."""
        shares = _make_shares(["A"], ["X"], [1.0])
        sens = _make_sensitivities(["X"], [1.0])
        sv = _make_shock_variances(["X"], [1.0])
        result = rotemberg_weights(shares, sens, sv)
        for col in ("naics", "alpha", "beta", "sum_w_sq"):
            assert col in result.columns

    def test_sum_w_sq_computed_correctly(self):
        """sum_w_sq = sum over counties of share^2."""
        # County A: share_X = 0.5, County B: share_X = 0.3 → sum_w_sq = 0.25+0.09=0.34
        shares = _make_shares(["A", "B"], ["X", "X"], [0.5, 0.3])
        sens = _make_sensitivities(["X"], [1.0])
        sv = _make_shock_variances(["X"], [1.0])
        result = rotemberg_weights(shares, sens, sv)
        expected_sum_w_sq = 0.5**2 + 0.3**2  # = 0.34
        assert abs(result.iloc[0]["sum_w_sq"] - expected_sum_w_sq) < 1e-12

    def test_beta_column_echoes_sensitivity(self):
        """beta column in output must match the input shrunk_beta."""
        shares = _make_shares(["A"], ["X"], [1.0])
        sens = _make_sensitivities(["X"], [3.7])
        sv = _make_shock_variances(["X"], [1.0])
        result = rotemberg_weights(shares, sens, sv)
        assert abs(result.iloc[0]["beta"] - 3.7) < 1e-12

    def test_one_row_per_industry(self):
        """Output has one row per unique industry in shares."""
        shares = _make_shares(
            ["A", "A", "B", "B"],
            ["X", "Y", "X", "Y"],
            [0.6, 0.4, 0.7, 0.3],
        )
        sens = _make_sensitivities(["X", "Y"], [1.0, 2.0])
        sv = _make_shock_variances(["X", "Y"], [0.5, 0.5])
        result = rotemberg_weights(shares, sens, sv)
        assert len(result) == 2


# ===========================================================================
# rotemberg_weights — negative alpha (industry pulls against headline)
# ===========================================================================

class TestRotembergWeightsNegativeAlpha:
    def test_negative_beta_gives_negative_alpha(self):
        """Industry with negative beta has negative alpha, but total still sums to 1."""
        shares = _make_shares(
            ["A", "A"],
            ["X", "Y"],
            [0.5, 0.5],
        )
        # Y has negative beta; total numerator = 1*0.25*1 + (-0.5)*0.25*1 = 0.125
        sens = _make_sensitivities(["X", "Y"], [1.0, -0.5])
        sv = _make_shock_variances(["X", "Y"], [1.0, 1.0])
        result = rotemberg_weights(shares, sens, sv).set_index("naics")
        assert result.loc["Y", "alpha"] < 0
        assert abs(result["alpha"].sum() - 1.0) < 1e-12


# ===========================================================================
# rotemberg_weights — error handling
# ===========================================================================

class TestRotembergWeightsErrors:
    def test_duplicate_county_naics_raises(self):
        """Duplicate (county_fips, naics) rows raise ValueError."""
        shares = pd.DataFrame([
            {"county_fips": "A", "naics": "31", "share": 0.5},
            {"county_fips": "A", "naics": "31", "share": 0.5},  # duplicate
        ])
        sens = _make_sensitivities(["31"], [1.0])
        sv = _make_shock_variances(["31"], [1.0])
        with pytest.raises(ValueError, match="duplicate"):
            rotemberg_weights(shares, sens, sv)

    def test_missing_variance_col_raises_keyerror(self):
        """Asking for a non-existent variance column raises KeyError."""
        shares = _make_shares(["A"], ["X"], [0.5])
        sens = _make_sensitivities(["X"], [1.0])
        sv = _make_shock_variances(["X"], [1.0])  # has 'sigma_sq_nat'
        with pytest.raises(KeyError, match="nonexistent_col"):
            rotemberg_weights(shares, sens, sv, variance_col="nonexistent_col")

    def test_custom_variance_col_works(self):
        """A custom variance column name is used when specified."""
        shares = _make_shares(["A"], ["X"], [0.5])
        sens = _make_sensitivities(["X"], [1.0])
        sv = pd.DataFrame({"naics": ["X"], "my_var": [0.5]})
        result = rotemberg_weights(shares, sens, sv, variance_col="my_var")
        assert abs(result.iloc[0]["alpha"] - 1.0) < 1e-12


# ===========================================================================
# rotemberg_weights — zero denominator edge case
# ===========================================================================

class TestRotembergWeightsZeroDenominator:
    def test_zero_denominator_returns_nan_alphas(self, caplog):
        """When all betas are zero, denominator is 0 and alphas are NaN."""
        shares = _make_shares(["A", "A"], ["X", "Y"], [0.5, 0.5])
        sens = _make_sensitivities(["X", "Y"], [0.0, 0.0])  # all zero
        sv = _make_shock_variances(["X", "Y"], [1.0, 1.0])
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.exposure_iv"):
            result = rotemberg_weights(shares, sens, sv)
        assert result["alpha"].isna().all()
        assert any("zero" in r.message.lower() for r in caplog.records)

    def test_zero_variances_gives_zero_numerators_and_nan_alphas(self, caplog):
        """When all sigma_sq are zero, numerators are 0 → denominator 0 → NaN alphas."""
        shares = _make_shares(["A"], ["X"], [0.5])
        sens = _make_sensitivities(["X"], [2.0])
        sv = _make_shock_variances(["X"], [0.0])  # sigma_sq=0
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.exposure_iv"):
            result = rotemberg_weights(shares, sens, sv)
        assert result["alpha"].isna().all()


# ===========================================================================
# rotemberg_weights — immutability
# ===========================================================================

class TestRotembergWeightsImmutability:
    def test_inputs_not_mutated(self):
        """rotemberg_weights must not mutate any of its three inputs."""
        shares = _make_shares(["A", "A"], ["X", "Y"], [0.5, 0.5])
        sens = _make_sensitivities(["X", "Y"], [1.0, 2.0])
        sv = _make_shock_variances(["X", "Y"], [0.5, 0.5])
        shares_orig = shares.copy()
        sens_orig = sens.copy()
        sv_orig = sv.copy()
        rotemberg_weights(shares, sens, sv)
        pd.testing.assert_frame_equal(shares, shares_orig)
        pd.testing.assert_frame_equal(sens, sens_orig)
        pd.testing.assert_frame_equal(sv, sv_orig)


# ===========================================================================
# Integration: build_exposure_iv then rotemberg_weights consistency
# ===========================================================================

class TestIntegration:
    def test_exposure_and_rotemberg_use_same_shares(self):
        """build_exposure_iv and rotemberg_weights operate on the same shares;
        neither should error out on a consistent input set."""
        shares = _make_shares(
            ["A", "A", "B", "B", "C"],
            ["X", "Y", "X", "Z", "Y"],
            [0.6, 0.4, 0.7, 0.3, 1.0],
        )
        sens = _make_sensitivities(["X", "Y", "Z"], [1.0, 2.0, -0.5])
        sv = _make_shock_variances(["X", "Y", "Z"], [0.2, 0.1, 0.3])
        exposure = build_exposure_iv(shares, sens)
        weights = rotemberg_weights(shares, sens, sv)
        # Exposure: one row per county
        assert len(exposure) == 3
        # Weights: one row per industry
        assert len(weights) == 3
        assert abs(weights["alpha"].sum() - 1.0) < 1e-12
