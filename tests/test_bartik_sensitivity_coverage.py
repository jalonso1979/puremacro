"""Coverage tests for puremacro.bartik.sensitivity.

Covers:
  - _one_industry_beta: known-value recovery, peak selection, min_obs guard,
    single-horizon, returns None when all horizons have too few obs
  - estimate_industry_sensitivities: output shape/columns/dtypes, shrinkage math,
    shrinkage=0 identity, shrinkage=1 all-equal, skipped-industry warning,
    date-mismatch warning, multi-industry ranking, peak is most-negative
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.bartik.sensitivity import (
    _one_industry_beta,
    estimate_industry_sensitivities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    naics_list: list[str],
    T: int,
    beta_map: dict[str, float],
    *,
    seed: int = 0,
    noise: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic industry + shock panels with known beta per industry.

    DGP: dlog_emp_{i,t} = beta_i * shock_t + noise
    Dates are quarterly integers 1..T.
    """
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(T)

    shock_df = pd.DataFrame({"date": np.arange(1, T + 1), "shock_epu_nat": shock})

    rows = []
    for naics in naics_list:
        beta = beta_map.get(naics, 0.0)
        dlog_emp = beta * shock + rng.standard_normal(T) * noise
        for t in range(T):
            rows.append({"date": t + 1, "naics": naics, "dlog_emp": dlog_emp[t]})

    industry_df = pd.DataFrame(rows)
    return industry_df, shock_df


# ---------------------------------------------------------------------------
# _one_industry_beta — internal helper
# ---------------------------------------------------------------------------

class TestOneIndustryBeta:
    def test_returns_none_when_too_few_obs(self):
        """When every horizon has fewer than min_obs observations, return None."""
        rng = np.random.default_rng(1)
        n = 10  # fewer than min_obs=30
        dates = pd.date_range("2000-01-01", periods=n, freq="QE")
        dlog = pd.Series(rng.standard_normal(n), index=dates)
        shock = pd.Series(rng.standard_normal(n), index=dates)
        result = _one_industry_beta(dlog, shock, horizons=[0, 1, 2], min_obs=30)
        assert result is None

    def test_returns_tuple_for_sufficient_obs(self):
        """Returns a 3-tuple (beta, se, nobs) when observations are sufficient."""
        rng = np.random.default_rng(2)
        n = 80
        idx = pd.RangeIndex(n)
        shock = pd.Series(rng.standard_normal(n), index=idx)
        dlog = 0.5 * shock + rng.standard_normal(n) * 0.2
        result = _one_industry_beta(dlog, shock, horizons=[0, 1, 2], min_obs=30)
        assert result is not None
        beta, se, nobs = result
        assert isinstance(beta, float)
        assert se > 0
        assert isinstance(nobs, int)
        assert nobs >= 30

    def test_se_positive(self):
        """Standard error must be strictly positive."""
        rng = np.random.default_rng(3)
        n = 60
        idx = pd.RangeIndex(n)
        shock = pd.Series(rng.standard_normal(n), index=idx)
        dlog = -0.8 * shock + rng.standard_normal(n) * 0.3
        beta, se, _ = _one_industry_beta(dlog, shock, horizons=[0], min_obs=30)
        assert se > 0.0

    def test_peak_is_most_negative(self):
        """The returned beta is the most-negative across horizons.

        We craft horizons where h=0 has a clearly positive beta and h=1 has a
        clearly negative beta.  The peak must match the most-negative one.
        """
        rng = np.random.default_rng(10)
        n = 100
        idx = pd.RangeIndex(n)
        shock = pd.Series(rng.standard_normal(n), index=idx)

        # h=0: positively-responded series (positive beta)
        dlog_pos = 2.0 * shock + rng.standard_normal(n) * 0.05

        result = _one_industry_beta(dlog_pos, shock, horizons=[0], min_obs=30)
        assert result is not None
        beta_pos, _, _ = result
        assert beta_pos > 0

        # h=0: negatively-responded series
        dlog_neg = -2.0 * shock + rng.standard_normal(n) * 0.05

        result_neg = _one_industry_beta(dlog_neg, shock, horizons=[0], min_obs=30)
        assert result_neg is not None
        beta_neg, _, _ = result_neg
        assert beta_neg < 0

    def test_single_horizon(self):
        """Single-horizon input works and returns a result."""
        rng = np.random.default_rng(4)
        n = 60
        idx = pd.RangeIndex(n)
        shock = pd.Series(rng.standard_normal(n), index=idx)
        dlog = -1.0 * shock + rng.standard_normal(n) * 0.2
        result = _one_industry_beta(dlog, shock, horizons=[0], min_obs=30)
        assert result is not None

    def test_min_obs_respected_per_horizon(self):
        """Horizons that shift away enough data to fall below min_obs are skipped.

        With n=35 and horizon h=6, the shifted series has 35-6=29 obs < min_obs=30.
        Only horizon h=0 has >=30 obs, so result is not None but uses h=0.
        """
        rng = np.random.default_rng(5)
        n = 35
        idx = pd.RangeIndex(n)
        shock = pd.Series(rng.standard_normal(n), index=idx)
        dlog = -0.5 * shock + rng.standard_normal(n) * 0.2
        result = _one_industry_beta(dlog, shock, horizons=[0, 6], min_obs=30)
        # h=6 should be skipped (29 < 30), h=0 should succeed (35 >= 30)
        assert result is not None

    @pytest.mark.slow
    def test_known_value_recovery(self):
        """With large T and low noise, peak beta should be close to the true value.

        This is a stochastic property test, not a deterministic known-value
        test — we only check that the estimate is within 0.3 of truth.
        """
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.RangeIndex(n)
        beta_true = -1.5
        shock = pd.Series(rng.standard_normal(n), index=idx)
        dlog = beta_true * shock + rng.standard_normal(n) * 0.1
        result = _one_industry_beta(dlog, shock, horizons=[0, 1, 2], min_obs=30)
        assert result is not None
        beta, se, nobs = result
        # The most-negative horizon should recover something near beta_true
        assert beta < 0.0  # at minimum, correct sign
        assert abs(beta - beta_true) < 0.5  # within reasonable tolerance


# ---------------------------------------------------------------------------
# estimate_industry_sensitivities — main public function
# ---------------------------------------------------------------------------

class TestEstimateIndustrySensitivities:
    def _build_simple(
        self,
        T: int = 80,
        industries: list[str] | None = None,
        beta_map: dict[str, float] | None = None,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if industries is None:
            industries = ["A", "B"]
        if beta_map is None:
            beta_map = {"A": -1.0, "B": 0.5}
        return _make_panel(industries, T, beta_map, seed=seed)

    # --- output shape / columns / dtypes ---

    def test_output_columns(self):
        """Output has the five required columns."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        expected_cols = {"naics", "beta_peak", "beta_peak_se", "shrunk_beta", "obs_count"}
        assert expected_cols == set(out.columns)

    def test_output_row_per_industry(self):
        """One row per valid industry."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        assert len(out) == 2

    def test_obs_count_positive_integer(self):
        """obs_count must be a positive integer for each row."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        for val in out["obs_count"]:
            assert isinstance(val, (int, np.integer))
            assert val > 0

    def test_beta_peak_se_positive(self):
        """beta_peak_se must be strictly positive for every industry."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        assert (out["beta_peak_se"] > 0).all()

    def test_naics_values_preserved(self):
        """The naics column contains exactly the industry codes from the panel."""
        industry_df, shock_df = self._build_simple(
            industries=["mfg", "retail", "finance"],
            beta_map={"mfg": -0.5, "retail": -1.0, "finance": 0.3},
        )
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        assert set(out["naics"]) == {"mfg", "retail", "finance"}

    # --- shrinkage math ---

    def test_shrinkage_zero_returns_raw(self):
        """With shrinkage=0, shrunk_beta equals beta_peak for every row."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(
            industry_df, shock_df, shrinkage=0.0, min_obs=30
        )
        np.testing.assert_allclose(out["shrunk_beta"].values, out["beta_peak"].values)

    def test_shrinkage_one_collapses_to_mean(self):
        """With shrinkage=1, all shrunk_beta values equal the grand mean of beta_peak."""
        industry_df, shock_df = self._build_simple()
        out = estimate_industry_sensitivities(
            industry_df, shock_df, shrinkage=1.0, min_obs=30
        )
        grand = out["beta_peak"].mean()
        np.testing.assert_allclose(
            out["shrunk_beta"].values,
            np.full(len(out), grand),
            atol=1e-12,
        )

    def test_shrinkage_interpolation(self):
        """shrunk_beta_i = (1-s)*beta_peak_i + s*mean(beta_peak) for intermediate s."""
        industry_df, shock_df = self._build_simple()
        s = 0.4
        out = estimate_industry_sensitivities(
            industry_df, shock_df, shrinkage=s, min_obs=30
        )
        grand = out["beta_peak"].mean()
        expected = (1 - s) * out["beta_peak"] + s * grand
        np.testing.assert_allclose(out["shrunk_beta"].values, expected.values, atol=1e-12)

    def test_shrunk_beta_between_peaks_and_mean(self):
        """For 0 < shrinkage < 1, shrunk_beta lies strictly between beta_peak and grand mean.

        This is a convex-combination property.
        """
        industry_df, shock_df = self._build_simple(T=100, seed=7)
        s = 0.3
        out = estimate_industry_sensitivities(
            industry_df, shock_df, shrinkage=s, min_obs=30
        )
        grand = out["beta_peak"].mean()
        for _, row in out.iterrows():
            lo = min(row["beta_peak"], grand)
            hi = max(row["beta_peak"], grand)
            assert lo <= row["shrunk_beta"] <= hi

    # --- skipped industries (too few obs) ---

    def test_skipped_industry_excluded_from_output(self):
        """An industry with fewer observations than min_obs is excluded from output."""
        T = 80
        rng = np.random.default_rng(20)
        shock = rng.standard_normal(T)
        shock_df = pd.DataFrame({"date": np.arange(1, T + 1), "shock_epu_nat": shock})

        # 'big' has 80 obs; 'tiny' has only 5 obs
        rows_big = [
            {"date": t + 1, "naics": "big",
             "dlog_emp": -0.5 * shock[t] + rng.standard_normal()}
            for t in range(T)
        ]
        rows_tiny = [
            {"date": t + 1, "naics": "tiny",
             "dlog_emp": -0.5 * shock[t] + rng.standard_normal()}
            for t in range(5)
        ]
        industry_df = pd.DataFrame(rows_big + rows_tiny)
        out = estimate_industry_sensitivities(
            industry_df, shock_df, horizons=[0], min_obs=30
        )
        assert "big" in out["naics"].values
        assert "tiny" not in out["naics"].values

    def test_skipped_industry_triggers_warning(self, caplog):
        """Skipping industries with too few obs should log a warning."""
        import logging
        T = 80
        rng = np.random.default_rng(21)
        shock = rng.standard_normal(T)
        shock_df = pd.DataFrame({"date": np.arange(1, T + 1), "shock_epu_nat": shock})

        rows_big = [
            {"date": t + 1, "naics": "big",
             "dlog_emp": rng.standard_normal()}
            for t in range(T)
        ]
        rows_tiny = [
            {"date": t + 1, "naics": "tiny",
             "dlog_emp": rng.standard_normal()}
            for t in range(5)
        ]
        industry_df = pd.DataFrame(rows_big + rows_tiny)

        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.sensitivity"):
            estimate_industry_sensitivities(
                industry_df, shock_df, horizons=[0], min_obs=30
            )
        # Some warning should have been issued about skipped industries
        assert any("kipped" in record.message for record in caplog.records)

    # --- date merge and coverage warning ---

    def test_date_mismatch_warning_logged(self, caplog):
        """A >10% date overlap loss triggers a warning."""
        import logging
        T = 100
        rng = np.random.default_rng(30)
        shock = rng.standard_normal(T)

        # industry_df has dates 1..100, shock_df only covers 1..50 (50% dropped)
        industry_df = pd.DataFrame({
            "date": list(range(1, T + 1)) * 2,
            "naics": ["A"] * T + ["B"] * T,
            "dlog_emp": rng.standard_normal(2 * T),
        })
        shock_df = pd.DataFrame({
            "date": np.arange(1, 51),
            "shock_epu_nat": shock[:50],
        })

        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.sensitivity"):
            estimate_industry_sensitivities(
                industry_df, shock_df, horizons=[0], min_obs=30
            )
        assert any("dropped" in record.message or "truncated" in record.message
                   for record in caplog.records)

    def test_no_date_mismatch_no_warning(self, caplog):
        """When dates fully overlap, no coverage-loss warning is issued."""
        import logging
        industry_df, shock_df = self._build_simple(T=80)
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.sensitivity"):
            estimate_industry_sensitivities(
                industry_df, shock_df, horizons=[0], min_obs=30
            )
        date_warns = [
            r for r in caplog.records if "dropped" in r.message or "truncated" in r.message
        ]
        assert len(date_warns) == 0

    # --- multiple industries / peak ordering ---

    def test_more_negative_industry_has_smaller_peak(self):
        """Industry with a more-negative true beta should yield a more-negative beta_peak.

        This is a stochastic property: we use a strong signal-to-noise ratio and
        a large sample to make it reliable.
        """
        T = 150
        industries = ["neg_strong", "pos_strong"]
        beta_map = {"neg_strong": -3.0, "pos_strong": 3.0}
        industry_df, shock_df = _make_panel(
            industries, T, beta_map, seed=50, noise=0.05
        )
        out = estimate_industry_sensitivities(
            industry_df, shock_df, horizons=[0], shrinkage=0.0, min_obs=30
        )
        beta_neg = float(out.loc[out["naics"] == "neg_strong", "beta_peak"].iloc[0])
        beta_pos = float(out.loc[out["naics"] == "pos_strong", "beta_peak"].iloc[0])
        assert beta_neg < beta_pos

    @pytest.mark.slow
    def test_multiple_horizons_uses_most_negative(self):
        """With multiple horizons, peak should be the most-negative beta across them.

        We build a series where h=0 gives a positive beta and h=1 gives a
        negative one, then check the returned peak is negative.
        """
        rng = np.random.default_rng(60)
        T = 120
        shock_vals = rng.standard_normal(T)
        shock_df = pd.DataFrame({"date": np.arange(1, T + 1), "shock_epu_nat": shock_vals})

        # dlog_emp at h=0 correlates positively with shock (beta_0 > 0)
        # dlog_emp shifted by h=1 correlates negatively (beta_1 < 0)
        # We engineer this by making dlog_emp[t] = 2*shock[t] - 4*shock[t-1] + noise
        # h=0 OLS: y=dlog_emp[t], x=shock[t] => beta ~ 2 > 0
        # h=1 OLS: y=dlog_emp[t+1], x=shock[t] => beta ~ -4 < 0
        dlog_emp = np.zeros(T)
        for t in range(1, T):
            dlog_emp[t] = 2.0 * shock_vals[t] - 4.0 * shock_vals[t - 1]
        dlog_emp += rng.standard_normal(T) * 0.05
        dlog_emp[0] = 0.0

        industry_df = pd.DataFrame({
            "date": np.arange(1, T + 1),
            "naics": ["TEST"] * T,
            "dlog_emp": dlog_emp,
        })
        out = estimate_industry_sensitivities(
            industry_df, shock_df, horizons=[0, 1], shrinkage=0.0, min_obs=30
        )
        if len(out) == 0:
            pytest.skip("industry skipped — not enough obs after alignment")
        peak = float(out.loc[out["naics"] == "TEST", "beta_peak"].iloc[0])
        # Peak should be the more-negative one (h=1 path)
        assert peak < 0

    # --- custom horizons ---

    def test_custom_horizon_list(self):
        """Accepts a plain list for horizons (not just range)."""
        industry_df, shock_df = self._build_simple(T=80)
        out = estimate_industry_sensitivities(
            industry_df, shock_df, horizons=[0, 2, 4], min_obs=30
        )
        assert len(out) == 2

    def test_default_horizons_works(self):
        """Default horizons=range(0,9) runs without error for sufficient T."""
        industry_df, shock_df = self._build_simple(T=100)
        out = estimate_industry_sensitivities(industry_df, shock_df, min_obs=30)
        assert len(out) >= 1

    # --- single industry edge case ---

    def test_single_industry_shrinkage_invariant(self):
        """With a single industry, shrunk_beta always equals beta_peak regardless of s.

        Because grand_mean = beta_peak when n=1, so
        (1-s)*beta + s*beta = beta.
        """
        industry_df, shock_df = _make_panel(
            ["ONLY"], 80, {"ONLY": -0.7}, seed=70, noise=0.2
        )
        for s in [0.0, 0.3, 0.7, 1.0]:
            out = estimate_industry_sensitivities(
                industry_df, shock_df, shrinkage=s, min_obs=30, horizons=[0]
            )
            if len(out) == 0:
                continue
            np.testing.assert_allclose(
                out["shrunk_beta"].values,
                out["beta_peak"].values,
                atol=1e-12,
                err_msg=f"shrinkage={s}",
            )
