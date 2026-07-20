"""Coverage tests for puremacro.bartik.shares.

Covers:
  - _assert_suppressed_dtype: bool, object (pass), int64 (fail)
  - build_base_year_shares:
      known-value shares (single and multi-county)
      suppressed cells dropped before normalization
      share conservation (sum to 1 per county)
      zero-suppressed county dropped with warning
      no-match base year returns empty DataFrame
      multi-quarter averaging in base year
  - build_rolling_shares:
      known-value share from analytic window
      single-industry county share = 1.0
      shares sum to 1 per (county_fips, period_end)
      counties below min_county_obs threshold are dropped
      all-suppressed panel returns empty DataFrame
      insufficient history returns empty DataFrame
      output column names and period_end type
      lag_quarters offsets window correctly
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_panel(
    *,
    years=(2000,),
    qtrs=(1,),
    counties=("001", "002"),
    industries=("11", "22"),
    emp_values: dict | None = None,
    suppressed: dict | None = None,
) -> pd.DataFrame:
    """Return a small, fully-unsuppressed industry panel."""
    rows = []
    for yr in years:
        for q in qtrs:
            for county in counties:
                for ind in industries:
                    key = (yr, q, county, ind)
                    emp = (emp_values or {}).get(key, 100.0)
                    sup = (suppressed or {}).get(key, False)
                    rows.append({
                        "year": yr, "qtr": q,
                        "county_fips": county, "naics": ind,
                        "emp_avg": emp, "suppressed": sup,
                    })
    return pd.DataFrame(rows)


def _long_panel(
    n_years: int = 6,
    counties: tuple[str, ...] = ("001", "002"),
    industries: tuple[str, ...] = ("11", "22"),
    emp_base: float = 100.0,
    start_year: int = 2000,
) -> pd.DataFrame:
    """Return a balanced panel spanning n_years × 4 quarters with constant employment."""
    rows = []
    for yr in range(start_year, start_year + n_years):
        for q in range(1, 5):
            for county in counties:
                for ind in industries:
                    rows.append({
                        "year": yr, "qtr": q,
                        "county_fips": county, "naics": ind,
                        "emp_avg": emp_base, "suppressed": False,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _assert_suppressed_dtype
# ---------------------------------------------------------------------------


class TestAssertSuppressedDtype:
    def test_bool_dtype_passes(self):
        from puremacro.bartik.shares import _assert_suppressed_dtype
        panel = pd.DataFrame({"suppressed": pd.array([True, False, True], dtype=bool)})
        _assert_suppressed_dtype(panel)  # must not raise

    def test_object_dtype_passes(self):
        from puremacro.bartik.shares import _assert_suppressed_dtype
        panel = pd.DataFrame({"suppressed": pd.array([True, False], dtype=object)})
        _assert_suppressed_dtype(panel)  # must not raise

    def test_int64_dtype_fails(self):
        """Integer dtype must raise AssertionError with a descriptive message."""
        from puremacro.bartik.shares import _assert_suppressed_dtype
        panel = pd.DataFrame({"suppressed": pd.array([1, 0], dtype="int64")})
        with pytest.raises(AssertionError, match="int64"):
            _assert_suppressed_dtype(panel)

    def test_float_dtype_fails(self):
        """Float dtype must also raise AssertionError."""
        from puremacro.bartik.shares import _assert_suppressed_dtype
        panel = pd.DataFrame({"suppressed": pd.array([1.0, 0.0], dtype="float64")})
        with pytest.raises(AssertionError):
            _assert_suppressed_dtype(panel)


# ---------------------------------------------------------------------------
# build_base_year_shares — known-value tests
# ---------------------------------------------------------------------------


class TestBuildBaseYearSharesKnownValues:
    def test_two_county_two_industry_shares(self):
        """Known employment split: county 001 has 100 in industry 11, 300 in 22.
        So share_11 = 0.25, share_22 = 0.75.
        County 002 has 200 in each -> share = 0.50 each.
        """
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000, 2000, 2000],
            "qtr": [1, 1, 1, 1],
            "county_fips": ["001", "001", "002", "002"],
            "naics": ["11", "22", "11", "22"],
            "emp_avg": [100.0, 300.0, 200.0, 200.0],
            "suppressed": [False, False, False, False],
        })
        result = build_base_year_shares(panel, base_year=2000)

        r001 = result[result["county_fips"] == "001"].set_index("naics")["share"]
        r002 = result[result["county_fips"] == "002"].set_index("naics")["share"]

        assert r001["11"] == pytest.approx(100 / 400)
        assert r001["22"] == pytest.approx(300 / 400)
        assert r002["11"] == pytest.approx(0.5)
        assert r002["22"] == pytest.approx(0.5)

    def test_single_industry_county_share_is_one(self):
        """A county with only one non-suppressed industry must have share = 1.0."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000],
            "qtr": [1],
            "county_fips": ["001"],
            "naics": ["11"],
            "emp_avg": [500.0],
            "suppressed": [False],
        })
        result = build_base_year_shares(panel, base_year=2000)
        assert len(result) == 1
        assert result.iloc[0]["share"] == pytest.approx(1.0)

    def test_multi_quarter_averaging(self):
        """Base year spans multiple quarters; emp_avg is averaged before sharing.

        County 001 / industry 11: emp = 100 (Q1), 300 (Q2) -> mean = 200
        County 001 / industry 22: emp = 200 (Q1), 200 (Q2) -> mean = 200
        Expected share_11 = 200/(200+200) = 0.5
        """
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000, 2000, 2000],
            "qtr": [1, 2, 1, 2],
            "county_fips": ["001", "001", "001", "001"],
            "naics": ["11", "11", "22", "22"],
            "emp_avg": [100.0, 300.0, 200.0, 200.0],
            "suppressed": [False, False, False, False],
        })
        result = build_base_year_shares(panel, base_year=2000)
        by_naics = result.set_index("naics")["share"]
        assert by_naics["11"] == pytest.approx(0.5)
        assert by_naics["22"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_base_year_shares — properties
# ---------------------------------------------------------------------------


class TestBuildBaseYearSharesProperties:
    def test_shares_sum_to_one_per_county(self):
        """Shares must partition employment exactly within each county."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel(
            emp_values={
                (2000, 1, "001", "11"): 120.0,
                (2000, 1, "001", "22"): 80.0,
                (2000, 1, "002", "11"): 300.0,
                (2000, 1, "002", "22"): 100.0,
            }
        )
        result = build_base_year_shares(panel, base_year=2000)
        totals = result.groupby("county_fips")["share"].sum()
        for county, total in totals.items():
            assert total == pytest.approx(1.0, abs=1e-12), f"county {county} sums to {total}"

    def test_shares_in_unit_interval(self):
        """All shares must be in [0, 1]."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel()
        result = build_base_year_shares(panel, base_year=2000)
        assert (result["share"] >= 0).all()
        assert (result["share"] <= 1.0 + 1e-12).all()

    def test_output_columns(self):
        """Output DataFrame must have exactly county_fips, naics, share columns."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel()
        result = build_base_year_shares(panel, base_year=2000)
        assert set(result.columns) == {"county_fips", "naics", "share"}

    def test_output_row_count(self):
        """Two counties × two industries -> exactly 4 rows."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel()
        result = build_base_year_shares(panel, base_year=2000)
        assert len(result) == 4

    def test_no_nan_in_shares(self):
        """All returned share values must be finite (no NaN)."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel()
        result = build_base_year_shares(panel, base_year=2000)
        assert result["share"].notna().all()

    def test_uses_only_base_year_data(self):
        """Data from non-base years is ignored entirely."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [1999, 2000, 1999, 2000],
            "qtr": [1, 1, 1, 1],
            "county_fips": ["001", "001", "001", "001"],
            "naics": ["11", "11", "22", "22"],
            "emp_avg": [9999.0, 100.0, 9999.0, 300.0],
            "suppressed": [False, False, False, False],
        })
        result = build_base_year_shares(panel, base_year=2000)
        by_naics = result.set_index("naics")["share"]
        assert by_naics["11"] == pytest.approx(100.0 / 400.0)
        assert by_naics["22"] == pytest.approx(300.0 / 400.0)


# ---------------------------------------------------------------------------
# build_base_year_shares — suppression handling
# ---------------------------------------------------------------------------


class TestBuildBaseYearSharesSuppressionHandling:
    def test_suppressed_cell_dropped_before_normalization(self):
        """Suppressed cell (county 001 / naics 22) is dropped.
        County 001 only has naics 11 -> share = 1.0.
        """
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000, 2000],
            "qtr": [1, 1, 1],
            "county_fips": ["001", "001", "002"],
            "naics": ["11", "22", "11"],
            "emp_avg": [100.0, 0.0, 50.0],
            "suppressed": [False, True, False],
        })
        result = build_base_year_shares(panel, base_year=2000)
        r001 = result[result["county_fips"] == "001"]
        assert len(r001) == 1
        assert r001.iloc[0]["naics"] == "11"
        assert r001.iloc[0]["share"] == pytest.approx(1.0)

    def test_fully_suppressed_county_dropped(self):
        """A county where all base-year observations are suppressed is dropped.
        It would produce NaN shares, so it is silently excluded.
        """
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000, 2000, 2000],
            "qtr": [1, 1, 1, 1],
            "county_fips": ["001", "001", "002", "002"],
            "naics": ["11", "22", "11", "22"],
            "emp_avg": [100.0, 200.0, 50.0, 150.0],
            "suppressed": [True, True, False, False],
        })
        result = build_base_year_shares(panel, base_year=2000)
        assert "001" not in result["county_fips"].values
        assert set(result["county_fips"].unique()) == {"002"}

    def test_zero_emp_county_warns_and_drops(self, caplog):
        """County with zero (non-suppressed) base-year employment triggers a WARNING
        and is dropped — rather than producing NaN shares.

        Note: a fully-suppressed county disappears silently (suppressed rows are
        filtered before groupby, so the county never appears in 'annual'). The
        WARNING path only fires when a county appears in 'annual' but sums to zero.
        """
        import logging
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000, 2000],
            "qtr": [1, 1, 1],
            "county_fips": ["001", "001", "002"],
            "naics": ["11", "22", "11"],
            "emp_avg": [0.0, 0.0, 50.0],  # zero emp, NOT suppressed -> triggers warning
            "suppressed": [False, False, False],
        })
        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.shares"):
            result = build_base_year_shares(panel, base_year=2000)

        # At least one WARNING message about the dropped county
        assert any("001" in msg for msg in caplog.messages)
        assert "001" not in result["county_fips"].values

    def test_object_dtype_suppressed_works(self):
        """Object-dtype suppressed column is handled correctly via astype(bool)."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000, 2000],
            "qtr": [1, 1],
            "county_fips": ["001", "001"],
            "naics": ["11", "22"],
            "emp_avg": [100.0, 200.0],
            "suppressed": pd.array([False, False], dtype=object),
        })
        result = build_base_year_shares(panel, base_year=2000)
        assert len(result) == 2

    def test_bad_suppressed_dtype_raises(self):
        """Integer suppressed column raises AssertionError immediately."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = pd.DataFrame({
            "year": [2000],
            "qtr": [1],
            "county_fips": ["001"],
            "naics": ["11"],
            "emp_avg": [100.0],
            "suppressed": pd.array([0], dtype="int64"),
        })
        with pytest.raises(AssertionError):
            build_base_year_shares(panel, base_year=2000)


# ---------------------------------------------------------------------------
# build_base_year_shares — edge cases
# ---------------------------------------------------------------------------


class TestBuildBaseYearSharesEdgeCases:
    def test_wrong_base_year_returns_empty_dataframe(self):
        """When no rows match base_year, return an empty DataFrame with correct columns."""
        from puremacro.bartik.shares import build_base_year_shares

        panel = _simple_panel(years=(2000,))
        result = build_base_year_shares(panel, base_year=1999)
        assert result.empty
        assert set(result.columns) == {"county_fips", "naics", "share"}

    def test_large_employment_values_share_invariance(self):
        """Shares should be invariant to the absolute scale of employment.
        Doubling all emp values should leave shares unchanged.
        """
        from puremacro.bartik.shares import build_base_year_shares

        panel1 = pd.DataFrame({
            "year": [2000, 2000],
            "qtr": [1, 1],
            "county_fips": ["001", "001"],
            "naics": ["11", "22"],
            "emp_avg": [100.0, 300.0],
            "suppressed": [False, False],
        })
        panel2 = panel1.copy()
        panel2["emp_avg"] = panel2["emp_avg"] * 1_000_000

        r1 = build_base_year_shares(panel1, base_year=2000).set_index("naics")["share"]
        r2 = build_base_year_shares(panel2, base_year=2000).set_index("naics")["share"]

        assert r1["11"] == pytest.approx(r2["11"])
        assert r1["22"] == pytest.approx(r2["22"])


# ---------------------------------------------------------------------------
# build_rolling_shares — known-value tests
# ---------------------------------------------------------------------------


class TestBuildRollingSharesKnownValues:
    def test_analytic_window_known_shares(self):
        """Verify the exact share for a specific period_end from analytic window.

        Panel: periods 2000Q1 to 2004Q4 (20 quarters).
        window_quarters=4, lag_quarters=2.
        For period_end=2001Q3: window_start=2001Q3-5=2000Q2, window_end=2001Q3-2=2001Q1.
        Window contains: 2000Q2, 2000Q3, 2000Q4, 2001Q1 (indices 1,2,3,4 from 0).
        naics 11 emp_avg (index+1): 2,3,4,5 -> mean=3.5
        naics 22 emp_avg (10*(index+1)): 20,30,40,50 -> mean=35
        share_11 = 3.5/38.5, share_22 = 35/38.5
        """
        from puremacro.bartik.shares import build_rolling_shares

        rows = []
        period_idx = 0
        for yr in range(2000, 2005):
            for q in range(1, 5):
                rows.append({
                    "year": yr, "qtr": q, "county_fips": "001",
                    "naics": "11", "emp_avg": float(period_idx + 1),
                    "suppressed": False,
                })
                rows.append({
                    "year": yr, "qtr": q, "county_fips": "001",
                    "naics": "22", "emp_avg": float(10 * (period_idx + 1)),
                    "suppressed": False,
                })
                period_idx += 1

        panel = pd.DataFrame(rows)
        result = build_rolling_shares(panel, window_quarters=4, lag_quarters=2, min_county_obs=4)

        target_end = pd.Period("2001Q3", freq="Q")
        target = result[result["period_end"] == target_end].set_index("naics")["share"]

        expected_11 = 3.5 / 38.5
        expected_22 = 35.0 / 38.5
        assert target["11"] == pytest.approx(expected_11, abs=1e-10)
        assert target["22"] == pytest.approx(expected_22, abs=1e-10)

    def test_single_industry_county_always_one(self):
        """A county with one industry in the window must have share = 1.0 for every period_end."""
        import numpy as np
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6, counties=("001",), industries=("11",))
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert not result.empty
        assert (result["share"] - 1.0).abs().max() < 1e-12

    def test_equal_employment_shares_half(self):
        """Two industries with identical constant employment -> each share = 0.5."""
        import numpy as np
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6, counties=("001",), industries=("11", "22"), emp_base=100.0)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert not result.empty
        assert (result["share"] - 0.5).abs().max() < 1e-12


# ---------------------------------------------------------------------------
# build_rolling_shares — properties
# ---------------------------------------------------------------------------


class TestBuildRollingSharesProperties:
    def test_shares_sum_to_one_per_county_period(self):
        """For every (county_fips, period_end) the shares must sum to exactly 1."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert not result.empty
        group_sums = result.groupby(["county_fips", "period_end"])["share"].sum()
        for idx, total in group_sums.items():
            assert total == pytest.approx(1.0, abs=1e-12), f"group {idx} sums to {total}"

    def test_shares_in_unit_interval(self):
        """All shares must be in [0, 1]."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert (result["share"] >= 0).all()
        assert (result["share"] <= 1.0 + 1e-12).all()

    def test_output_columns(self):
        """Output must have exactly county_fips, naics, period_end, share."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert set(result.columns) == {"county_fips", "naics", "period_end", "share"}

    def test_period_end_is_period_type(self):
        """period_end column must have PeriodDtype (quarterly)."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert hasattr(result["period_end"].dtype, "freq"), (
            f"period_end dtype {result['period_end'].dtype} is not PeriodDtype"
        )

    def test_no_nan_shares(self):
        """No NaN share values in non-empty result."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=4)
        assert result["share"].notna().all()


# ---------------------------------------------------------------------------
# build_rolling_shares — filtering / suppression
# ---------------------------------------------------------------------------


class TestBuildRollingSharesFiltering:
    def test_min_county_obs_filters_sparse_county(self):
        """County with too few observations is excluded for that period_end."""
        from puremacro.bartik.shares import build_rolling_shares

        # county 001: full coverage (24 quarter × 1 industry obs)
        # county 002: only 1 obs total -> fails min_county_obs=12
        rows = []
        for yr in range(2000, 2006):
            for q in range(1, 5):
                rows.append({
                    "year": yr, "qtr": q, "county_fips": "001",
                    "naics": "11", "emp_avg": 100.0, "suppressed": False,
                })
        # single observation for county 002
        rows.append({"year": 2002, "qtr": 1, "county_fips": "002",
                     "naics": "11", "emp_avg": 50.0, "suppressed": False})

        panel = pd.DataFrame(rows)
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=12)
        assert "002" not in result["county_fips"].values

    def test_suppressed_rows_excluded_before_window(self):
        """Suppressed observations must not contribute to window statistics."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6, counties=("001",), industries=("11",))
        panel_supp = panel.copy()
        # Suppress the naics=11 observations; replace with a non-suppressed version
        # that has only naics=22 (an industry not in the base).
        # Easier: suppress ALL -> empty result.
        panel_supp["suppressed"] = True
        result = build_rolling_shares(panel_supp, window_quarters=8, lag_quarters=2, min_county_obs=1)
        assert result.empty

    def test_all_suppressed_returns_empty_df(self):
        """Completely suppressed panel returns empty DataFrame with correct columns."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6)
        panel["suppressed"] = True
        result = build_rolling_shares(panel, window_quarters=8, lag_quarters=2, min_county_obs=1)
        assert result.empty
        assert set(result.columns) == {"county_fips", "naics", "period_end", "share"}

    def test_insufficient_history_returns_empty_df(self):
        """When the panel has too few periods to satisfy lag + window, return empty."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = pd.DataFrame({
            "year": [2000],
            "qtr": [1],
            "county_fips": ["001"],
            "naics": ["11"],
            "emp_avg": [100.0],
            "suppressed": [False],
        })
        result = build_rolling_shares(
            panel, window_quarters=8, lag_quarters=4, min_county_obs=1
        )
        assert result.empty
        assert set(result.columns) == {"county_fips", "naics", "period_end", "share"}

    def test_lag_shifts_window_endpoint(self):
        """Period_end values must reflect the lag offset: no period_end from the
        first lag_quarters periods of the panel.
        """
        from puremacro.bartik.shares import build_rolling_shares

        panel = _long_panel(n_years=6, counties=("001",), industries=("11",))
        lag_quarters = 4
        result = build_rolling_shares(panel, window_quarters=4, lag_quarters=lag_quarters, min_county_obs=4)
        all_periods = sorted(result["period_end"].unique())
        earliest = all_periods[0] if all_periods else None
        # The earliest possible period_end = panel_start + (window + lag - 1)
        # panel_start = 2000Q1; earliest_end >= 2000Q1 + 4 + 4 - 1 = 2001Q4 (index 7 from 0)
        # In absolute terms: end >= 2000Q1 + 7 quarters = 2001Q4
        if earliest is not None:
            panel_start = pd.Period("2000Q1", freq="Q")
            # earliest period_end must be at least lag_quarters + window_quarters ahead of panel_start
            assert earliest >= panel_start + (lag_quarters + 4 - 1), (
                f"earliest period_end {earliest} is unexpectedly early"
            )

    def test_bad_suppressed_dtype_raises_in_rolling(self):
        """Rolling builder also enforces the suppressed dtype contract."""
        from puremacro.bartik.shares import build_rolling_shares

        panel = pd.DataFrame({
            "year": [2000],
            "qtr": [1],
            "county_fips": ["001"],
            "naics": ["11"],
            "emp_avg": [100.0],
            "suppressed": pd.array([0], dtype="int64"),
        })
        with pytest.raises(AssertionError):
            build_rolling_shares(panel, window_quarters=4, lag_quarters=2, min_county_obs=1)

    def test_zero_employment_county_dropped_in_rolling(self, caplog):
        """A window with zero total employment triggers a WARNING and drops the county."""
        import logging
        from puremacro.bartik.shares import build_rolling_shares

        # All emp_avg = 0 -> window sum = 0 -> county dropped
        rows = []
        for yr in range(2000, 2006):
            for q in range(1, 5):
                rows.append({
                    "year": yr, "qtr": q, "county_fips": "001",
                    "naics": "11", "emp_avg": 0.0, "suppressed": False,
                })
        panel = pd.DataFrame(rows)

        with caplog.at_level(logging.WARNING, logger="puremacro.bartik.shares"):
            result = build_rolling_shares(
                panel, window_quarters=4, lag_quarters=2, min_county_obs=4
            )

        assert result.empty or "001" not in result["county_fips"].values
        assert any("zero" in msg.lower() or "dropping" in msg.lower()
                   for msg in caplog.messages)
