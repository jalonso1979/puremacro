"""Coverage tests for puremacro.bartik.county_epu.

Module contract:
    build_county_epu(shares, state_crosswalk, state_epu_monthly, *, freq='Q')
    -> DataFrame[county_fips, state_fips, date, epu_county_bartik_{m|q}]

The Bartik EPU for a county c at time t is:
    epu_county_{c,t} = (sum_i w_{c,i,base}) * epu_state_{s(c),t}

where the sum of shares (share_sum) acts as a coverage-completeness weight.
When shares sum to 1 the county EPU equals the state EPU exactly.
Quarterly mode averages monthly EPU values within each quarter.

Tests cover:
 - Known-value: monthly and quarterly EPU with analytically computed answers
 - Partial-shares (share_sum < 1) produce scaled EPU
 - Multi-naics shares are summed per county
 - Integer FIPS codes are coerced to strings
 - Counties in different states receive different EPU
 - Counties in same state with same shares receive same EPU
 - Missing county in shares -> NaN epu (warning path)
 - County linked to unknown state -> NaN epu (high-NaN warning path)
 - Invalid freq raises ValueError
 - Output schema: exactly 4 columns, correct dtypes
 - Output is reset_index (integer index)
 - Quarterly date column is the quarter-start timestamp
 - Monthly mode preserves original monthly dates
 - epu values are non-negative when state EPU is non-negative (sign contract)
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np
import pytest

from puremacro.bartik.county_epu import build_county_epu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal(
    *,
    n_months: int = 3,
    share: float = 1.0,
    epu_val: float = 100.0,
    county_fips: str = "01001",
    state_fips: str = "01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (shares, crosswalk, state_epu_monthly) for a single county."""
    shares = pd.DataFrame({
        "county_fips": [county_fips],
        "naics": [11],
        "share": [share],
    })
    crosswalk = pd.DataFrame({
        "county_fips": [county_fips],
        "state_fips": [state_fips],
    })
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    state_epu = pd.DataFrame({
        "state_fips": [state_fips] * n_months,
        "date": dates,
        "epu_state_m": [epu_val] * n_months,
    })
    return shares, crosswalk, state_epu


# ===========================================================================
# Error handling
# ===========================================================================

class TestInvalidFreq:
    def test_invalid_freq_raises_value_error(self):
        shares, crosswalk, epu = _make_minimal()
        with pytest.raises(ValueError, match="freq must be"):
            build_county_epu(shares, crosswalk, epu, freq="Y")

    def test_invalid_freq_message_includes_value(self):
        shares, crosswalk, epu = _make_minimal()
        with pytest.raises(ValueError, match="'A'"):
            build_county_epu(shares, crosswalk, epu, freq="A")

    def test_valid_freq_m_does_not_raise(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert result is not None

    def test_valid_freq_q_does_not_raise(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="Q")
        assert result is not None


# ===========================================================================
# Output schema
# ===========================================================================

class TestOutputSchema:
    def test_monthly_columns(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert list(result.columns) == [
            "county_fips", "state_fips", "date", "epu_county_bartik_m"
        ]

    def test_quarterly_columns(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="Q")
        assert list(result.columns) == [
            "county_fips", "state_fips", "date", "epu_county_bartik_q"
        ]

    def test_county_fips_dtype_is_object(self):
        """county_fips must be string, not numeric (object under pandas 2,
        the string dtype under pandas 3 — astype(str) differs)."""
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert pd.api.types.is_string_dtype(result["county_fips"])

    def test_state_fips_dtype_is_object(self):
        """state_fips must be string, not numeric (object under pandas 2,
        the string dtype under pandas 3 — astype(str) differs)."""
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert pd.api.types.is_string_dtype(result["state_fips"])

    def test_date_dtype_is_datetime(self):
        """date column is datetime64."""
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_epu_m_dtype_is_float(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert result["epu_county_bartik_m"].dtype.kind == "f"

    def test_epu_q_dtype_is_float(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="Q")
        assert result["epu_county_bartik_q"].dtype.kind == "f"

    def test_index_is_reset(self):
        """Index should be a default RangeIndex (reset)."""
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert isinstance(result.index, pd.RangeIndex)
        assert result.index[0] == 0

    def test_returns_dataframe(self):
        shares, crosswalk, epu = _make_minimal()
        result = build_county_epu(shares, crosswalk, epu, freq="M")
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# Known-value: monthly mode
# ===========================================================================

class TestKnownValueMonthly:
    def test_full_share_equals_state_epu(self):
        """When share_sum == 1, county EPU equals state EPU exactly."""
        epu_val = 137.5
        shares, crosswalk, state_epu = _make_minimal(
            n_months=3, share=1.0, epu_val=epu_val
        )
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        np.testing.assert_allclose(
            result["epu_county_bartik_m"].values,
            epu_val,
            rtol=1e-12,
        )

    def test_partial_share_scales_epu(self):
        """When share_sum == 0.7, county EPU == 0.7 * state_epu."""
        epu_val = 200.0
        share_val = 0.7
        shares, crosswalk, state_epu = _make_minimal(
            n_months=2, share=share_val, epu_val=epu_val
        )
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        expected = share_val * epu_val
        np.testing.assert_allclose(
            result["epu_county_bartik_m"].values,
            expected,
            rtol=1e-12,
        )

    def test_monthly_dates_preserved(self):
        """Monthly dates in output match the input monthly dates."""
        shares, crosswalk, state_epu = _make_minimal(n_months=4)
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        expected_dates = pd.date_range("2020-01-01", periods=4, freq="MS")
        for d_out, d_exp in zip(result["date"].values, expected_dates):
            assert pd.Timestamp(d_out) == d_exp

    def test_row_count_monthly_one_county(self):
        """One county, n months -> n rows."""
        n = 6
        shares, crosswalk, state_epu = _make_minimal(n_months=n)
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert len(result) == n

    def test_varying_monthly_epu_values(self):
        """Each month's EPU is independently reflected in the county EPU."""
        epu_vals = [100.0, 150.0, 80.0]
        shares = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"] * 3,
            "date": dates,
            "epu_state_m": epu_vals,
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        np.testing.assert_allclose(
            result["epu_county_bartik_m"].values, epu_vals, rtol=1e-12
        )


# ===========================================================================
# Known-value: quarterly mode
# ===========================================================================

class TestKnownValueQuarterly:
    def test_single_quarter_mean(self):
        """Q1 2020: months 100, 120, 110 -> quarterly mean = 110."""
        shares = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"] * 3,
            "date": dates,
            "epu_state_m": [100.0, 120.0, 110.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="Q")
        assert len(result) == 1
        np.testing.assert_allclose(
            result["epu_county_bartik_q"].iloc[0],
            (100.0 + 120.0 + 110.0) / 3,
            rtol=1e-12,
        )

    def test_two_quarters_correct_means(self):
        """Q1 mean=150, Q2 mean=60 — both verified analytically."""
        shares = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        # Q1: Jan=100, Feb=200, Mar=150; Q2: Apr=90, May=60, Jun=30
        epu_vals = [100.0, 200.0, 150.0, 90.0, 60.0, 30.0]
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"] * 6,
            "date": dates,
            "epu_state_m": epu_vals,
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="Q")
        assert len(result) == 2
        np.testing.assert_allclose(result.iloc[0]["epu_county_bartik_q"], 150.0, rtol=1e-12)
        np.testing.assert_allclose(result.iloc[1]["epu_county_bartik_q"], 60.0, rtol=1e-12)

    def test_quarterly_dates_are_quarter_starts(self):
        """Quarterly output dates should be the first day of each quarter."""
        shares, crosswalk, state_epu = _make_minimal(n_months=6)
        result = build_county_epu(shares, crosswalk, state_epu, freq="Q")
        # All dates should fall on the first of a quarter-start month (Jan, Apr, Jul, Oct)
        for d in result["date"]:
            ts = pd.Timestamp(d)
            assert ts.month in (1, 4, 7, 10), f"Unexpected month {ts.month} for quarter start"
            assert ts.day == 1

    def test_quarterly_fewer_rows_than_monthly(self):
        """Quarterly output has fewer rows than monthly (aggregation)."""
        shares, crosswalk, state_epu = _make_minimal(n_months=6)
        r_m = build_county_epu(shares, crosswalk, state_epu, freq="M")
        r_q = build_county_epu(shares, crosswalk, state_epu, freq="Q")
        assert len(r_q) < len(r_m)

    def test_quarterly_partial_share_scales(self):
        """Quarterly EPU also scales by share_sum."""
        share_val = 0.6
        # Q1: 3 months avg = 100; county EPU = 0.6 * 100 = 60
        shares = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [share_val]})
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"] * 3,
            "date": dates,
            "epu_state_m": [100.0, 100.0, 100.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="Q")
        np.testing.assert_allclose(
            result["epu_county_bartik_q"].iloc[0],
            share_val * 100.0,
            rtol=1e-12,
        )


# ===========================================================================
# Multi-industry / multi-county
# ===========================================================================

class TestMultiIndustryAndCounty:
    def test_multi_naics_shares_summed_per_county(self):
        """
        County 01001 has two industries: share=0.3 + share=0.4 = 0.7.
        With state EPU=200, county EPU = 0.7 * 200 = 140.
        """
        shares = pd.DataFrame({
            "county_fips": ["01001", "01001"],
            "naics": [11, 21],
            "share": [0.3, 0.4],
        })
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({"state_fips": ["01"], "date": dates, "epu_state_m": [200.0]})
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        np.testing.assert_allclose(
            result["epu_county_bartik_m"].iloc[0], 140.0, rtol=1e-12
        )

    def test_two_counties_same_state_same_share_get_same_epu(self):
        """Counties in the same state with the same share_sum receive identical EPU."""
        shares = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "naics": [11, 11],
            "share": [1.0, 1.0],
        })
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
        })
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01", "01"],
            "date": dates,
            "epu_state_m": [100.0, 120.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        c1 = result[result["county_fips"] == "01001"].sort_values("date")["epu_county_bartik_m"].values
        c3 = result[result["county_fips"] == "01003"].sort_values("date")["epu_county_bartik_m"].values
        np.testing.assert_allclose(c1, c3, rtol=1e-12)

    def test_two_counties_different_states_different_epu(self):
        """Counties in different states receive their respective state EPU."""
        shares = pd.DataFrame({
            "county_fips": ["01001", "06001"],
            "naics": [11, 11],
            "share": [1.0, 1.0],
        })
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "06001"],
            "state_fips": ["01", "06"],
        })
        date = pd.Timestamp("2020-01-01")
        state_epu = pd.DataFrame({
            "state_fips": ["01", "06"],
            "date": [date, date],
            "epu_state_m": [100.0, 200.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        epu_al = result[result["county_fips"] == "01001"]["epu_county_bartik_m"].iloc[0]
        epu_ca = result[result["county_fips"] == "06001"]["epu_county_bartik_m"].iloc[0]
        assert abs(epu_al - 100.0) < 1e-10
        assert abs(epu_ca - 200.0) < 1e-10
        assert epu_al != epu_ca

    def test_row_count_two_counties_monthly(self):
        """Two counties × n months -> 2n rows."""
        n = 3
        shares = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "naics": [11, 11],
            "share": [1.0, 1.0],
        })
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
        })
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"] * n,
            "date": dates,
            "epu_state_m": [100.0] * n,
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert len(result) == 2 * n


# ===========================================================================
# Missing / NaN handling
# ===========================================================================

class TestMissingData:
    def test_county_missing_from_shares_gets_nan_epu(self):
        """County 01003 in crosswalk but absent from shares -> NaN EPU."""
        shares = pd.DataFrame({
            "county_fips": ["01001"],
            "naics": [11],
            "share": [1.0],
        })
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
        })
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"],
            "date": dates,
            "epu_state_m": [100.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        epu_missing = result[result["county_fips"] == "01003"]["epu_county_bartik_m"]
        assert epu_missing.isna().all()

    def test_county_missing_from_shares_emits_warning(self, caplog):
        """Missing share data logs a warning."""
        shares = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
        })
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01"],
            "date": dates,
            "epu_state_m": [100.0],
        })
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.county_epu"):
            build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert any("no share data" in r.message for r in caplog.records)

    def test_county_with_unknown_state_gets_nan_epu(self):
        """County linked to a state not present in state_epu -> NaN EPU."""
        shares = pd.DataFrame({"county_fips": ["99999"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": ["99999"], "state_fips": ["99"]})
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01", "01"],
            "date": dates,
            "epu_state_m": [100.0, 120.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert result["epu_county_bartik_m"].isna().all()

    def test_high_nan_fraction_emits_warning(self, caplog):
        """When more than 10% of county-date rows have NaN EPU, a warning is logged."""
        # All counties linked to unknown state -> 100% NaN
        shares = pd.DataFrame({"county_fips": ["99999"], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": ["99999"], "state_fips": ["99"]})
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": ["01", "01"],
            "date": dates,
            "epu_state_m": [100.0, 120.0],
        })
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.county_epu"):
            build_county_epu(shares, crosswalk, state_epu, freq="M")
        # Either the "no share data" or the "no matching state EPU" warning should appear
        messages = [r.message for r in caplog.records]
        assert any(
            "no match" in m.lower() or "nan" in m.lower() or "state epu" in m.lower()
            for m in messages
        )

    def test_known_county_still_gets_epu_when_other_missing(self):
        """A valid county still gets correct EPU even when another county has NaN."""
        shares = pd.DataFrame({
            "county_fips": ["01001"],
            "naics": [11],
            "share": [1.0],
        })
        crosswalk = pd.DataFrame({
            "county_fips": ["01001", "01003"],
            "state_fips": ["01", "01"],
        })
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({"state_fips": ["01"], "date": dates, "epu_state_m": [123.0]})
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        epu_valid = result[result["county_fips"] == "01001"]["epu_county_bartik_m"].iloc[0]
        np.testing.assert_allclose(epu_valid, 123.0, rtol=1e-12)


# ===========================================================================
# FIPS code type coercion
# ===========================================================================

class TestFIPSCoercion:
    def test_integer_county_fips_coerced_to_string(self):
        """Integer county_fips inputs are coerced to string and the merge still works."""
        shares = pd.DataFrame({"county_fips": [1001], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": [1001], "state_fips": [1]})
        dates = pd.date_range("2020-01-01", periods=2, freq="MS")
        state_epu = pd.DataFrame({
            "state_fips": [1, 1],
            "date": dates,
            "epu_state_m": [100.0, 120.0],
        })
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        # No NaN: merge succeeded despite integer inputs
        assert not result["epu_county_bartik_m"].isna().any()
        np.testing.assert_allclose(
            result["epu_county_bartik_m"].values, [100.0, 120.0], rtol=1e-12
        )

    def test_integer_fips_output_is_string(self):
        """Even with integer FIPS inputs, output dtype is string (object under
        pandas 2, the string dtype under pandas 3), never numeric."""
        shares = pd.DataFrame({"county_fips": [1001], "naics": [11], "share": [1.0]})
        crosswalk = pd.DataFrame({"county_fips": [1001], "state_fips": [1]})
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({"state_fips": [1], "date": dates, "epu_state_m": [50.0]})
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert pd.api.types.is_string_dtype(result["county_fips"])
        assert pd.api.types.is_string_dtype(result["state_fips"])


# ===========================================================================
# Sign / magnitude properties
# ===========================================================================

class TestSignProperties:
    def test_positive_epu_produces_positive_county_epu(self):
        """If state EPU > 0 and share_sum > 0, county EPU > 0."""
        shares, crosswalk, state_epu = _make_minimal(share=0.5, epu_val=80.0)
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert (result["epu_county_bartik_m"] > 0).all()

    def test_zero_epu_produces_zero_county_epu(self):
        """State EPU = 0 -> county EPU = 0 (regardless of share)."""
        shares, crosswalk, state_epu = _make_minimal(share=1.0, epu_val=0.0)
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        np.testing.assert_allclose(result["epu_county_bartik_m"].values, 0.0, atol=1e-12)

    def test_county_epu_bounded_by_state_epu_when_share_le_one(self):
        """When share_sum <= 1, county EPU <= state EPU (scaling down, not up)."""
        epu_val = 250.0
        share_val = 0.8
        shares, crosswalk, state_epu = _make_minimal(share=share_val, epu_val=epu_val)
        result = build_county_epu(shares, crosswalk, state_epu, freq="M")
        assert (result["epu_county_bartik_m"] <= epu_val + 1e-10).all()

    def test_higher_share_gives_higher_county_epu(self):
        """Ceteris paribus, a larger share_sum yields a higher county EPU."""
        shares_lo = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [0.3]})
        shares_hi = pd.DataFrame({"county_fips": ["01001"], "naics": [11], "share": [0.9]})
        crosswalk = pd.DataFrame({"county_fips": ["01001"], "state_fips": ["01"]})
        dates = pd.date_range("2020-01-01", periods=1, freq="MS")
        state_epu = pd.DataFrame({"state_fips": ["01"], "date": dates, "epu_state_m": [100.0]})
        r_lo = build_county_epu(shares_lo, crosswalk, state_epu, freq="M")
        r_hi = build_county_epu(shares_hi, crosswalk, state_epu, freq="M")
        assert (
            r_hi["epu_county_bartik_m"].values > r_lo["epu_county_bartik_m"].values
        ).all()
