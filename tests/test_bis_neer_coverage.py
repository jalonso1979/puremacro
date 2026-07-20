"""Tests for puremacro.bis_neer — BIS NEER fetcher.

Strategy:
- No network I/O: monkeypatch ``puremacro.bis_neer.safe_get_bytes`` (the
  name bound in the bis_neer module at import time) with synthetic CSV bytes
  that mimic the real BIS API response format.
- Known-value: build CSV with exact monthly values and assert the
  quarterly aggregation (mean-of-months) matches the hand-calculated
  known answer.
- Properties: output schema [code, qdate, neer_q], dtypes, sort order,
  non-empty result on valid input.
- Edge cases + error handling: unknown BIS code, HTTP exception,
  unparseable CSV, missing TIME_PERIOD/OBS_VALUE columns, all-NaN values,
  invalid date strings, empty CSV body.
- Panel: concat of multiple countries, default ``codes=None`` path.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from puremacro.bis_neer import (
    _BIS_TO_ISO3,
    _EMPTY,
    fetch_bis_neer,
    fetch_bis_neer_panel,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic BIS CSV bytes
# ---------------------------------------------------------------------------

def _make_bis_csv(rows: list[tuple[str, float | str]]) -> bytes:
    """Build a minimal BIS-format CSV with TIME_PERIOD and OBS_VALUE columns.

    ``rows`` is a list of (time_period, obs_value) tuples.
    """
    lines = ["TIME_PERIOD,OBS_VALUE"]
    for tp, val in rows:
        lines.append(f"{tp},{val}")
    return "\n".join(lines).encode()


def _make_bis_csv_with_headers(
    time_col: str,
    val_col: str,
    rows: list[tuple[str, float | str]],
) -> bytes:
    """Build a BIS CSV with arbitrary column names."""
    lines = [f"{time_col},{val_col}"]
    for tp, val in rows:
        lines.append(f"{tp},{val}")
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# 1. Known-value: quarterly aggregation math
# ---------------------------------------------------------------------------

class TestFetchBisNeerKnownValue:
    """Known-value tests for quarterly mean aggregation."""

    def test_single_quarter_one_month(self, monkeypatch):
        """One month in Q1-2020: neer_q must equal that month's value."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert len(df) == 1
        assert float(df["neer_q"].iloc[0]) == pytest.approx(100.0)

    def test_single_quarter_two_months_mean(self, monkeypatch):
        """Two months in Q1-2020 (Jan, Feb): neer_q = mean(100, 110) = 105."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-02", 110.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert len(df) == 1
        assert float(df["neer_q"].iloc[0]) == pytest.approx(105.0)

    def test_single_quarter_three_months_mean(self, monkeypatch):
        """Three months in Q1-2020: neer_q = mean(90, 100, 110) = 100."""
        csv_bytes = _make_bis_csv([
            ("2020-01", 90.0),
            ("2020-02", 100.0),
            ("2020-03", 110.0),
        ])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert len(df) == 1
        assert float(df["neer_q"].iloc[0]) == pytest.approx(100.0)

    def test_two_quarters_correct_means(self, monkeypatch):
        """Two full quarters: check neer_q for each matches the known mean."""
        csv_bytes = _make_bis_csv([
            ("2019-10", 95.0),
            ("2019-11", 97.0),
            ("2019-12", 99.0),  # Q4-2019, mean=97
            ("2020-01", 101.0),
            ("2020-02", 103.0),
            ("2020-03", 105.0),  # Q1-2020, mean=103
        ])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("DE")
        assert len(df) == 2
        df_sorted = df.sort_values("qdate").reset_index(drop=True)
        assert float(df_sorted["neer_q"].iloc[0]) == pytest.approx(97.0)
        assert float(df_sorted["neer_q"].iloc[1]) == pytest.approx(103.0)

    def test_iso3_mapping_correct_for_us(self, monkeypatch):
        """Fetching 'US' yields code='USA' in the output."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df["code"].iloc[0] == "USA"

    def test_iso3_mapping_correct_for_de(self, monkeypatch):
        """Fetching 'DE' yields code='DEU' in the output."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("DE")
        assert df["code"].iloc[0] == "DEU"

    def test_qdate_is_quarter_start_timestamp(self, monkeypatch):
        """qdate values should be Timestamps at quarter-start (month 1, 4, 7, 10)."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-04", 99.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert len(df) == 2
        for ts in df["qdate"]:
            assert isinstance(ts, pd.Timestamp)
            assert ts.month in (1, 4, 7, 10), f"Unexpected quarter-start month: {ts.month}"

    def test_result_sorted_by_qdate(self, monkeypatch):
        """Output rows are in ascending qdate order."""
        # Deliberately supply in reverse chronological order
        csv_bytes = _make_bis_csv([
            ("2021-01", 98.0),
            ("2020-10", 96.0),
            ("2020-07", 94.0),
            ("2020-04", 92.0),
        ])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        qdates = list(df["qdate"])
        assert qdates == sorted(qdates)


# ---------------------------------------------------------------------------
# 2. Output schema and dtype properties
# ---------------------------------------------------------------------------

class TestFetchBisNeerSchema:
    """Output schema, dtypes, and structural properties."""

    def test_columns_are_correct(self, monkeypatch):
        """Output columns are exactly [code, qdate, neer_q]."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_code_column_is_object_dtype(self, monkeypatch):
        """'code' column has string dtype (object under pandas 2, the string
        dtype under pandas 3), never numeric."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert pd.api.types.is_string_dtype(df["code"])

    def test_qdate_column_is_datetime(self, monkeypatch):
        """'qdate' column contains pd.Timestamp objects."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert pd.api.types.is_datetime64_any_dtype(df["qdate"])

    def test_neer_q_column_is_float(self, monkeypatch):
        """'neer_q' column is numeric (float)."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert pd.api.types.is_float_dtype(df["neer_q"])

    def test_all_code_values_equal_iso3(self, monkeypatch):
        """All 'code' values in the output are the correct ISO-3 string."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-04", 99.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("JP")
        assert (df["code"] == "JPN").all()

    def test_neer_q_values_are_finite_on_valid_input(self, monkeypatch):
        """neer_q values are finite (no NaN, no inf) for clean input."""
        csv_bytes = _make_bis_csv([
            ("2020-01", 100.0), ("2020-02", 101.0), ("2020-03", 99.0),
        ])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df["neer_q"].notna().all()
        assert np.isfinite(df["neer_q"].values).all()

    def test_index_is_reset_integer(self, monkeypatch):
        """Output index is a clean integer RangeIndex."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-04", 99.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert isinstance(df.index, pd.RangeIndex)
        assert list(df.index) == list(range(len(df)))

    def test_case_insensitive_bis_code(self, monkeypatch):
        """BIS code lookup is case-insensitive ('us' == 'US')."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df_lower = fetch_bis_neer("us")
        df_upper = fetch_bis_neer("US")
        # Both should return non-empty with code='USA'
        assert not df_lower.empty
        assert not df_upper.empty
        assert df_lower["code"].iloc[0] == "USA"


# ---------------------------------------------------------------------------
# 3. Case-insensitive column matching (TIME vs TIME_PERIOD, VALUE vs OBS_VALUE)
# ---------------------------------------------------------------------------

class TestFetchBisNeerColumnNames:
    """Tests that TIME_PERIOD/OBS_VALUE fallbacks work."""

    def test_time_period_exact_match(self, monkeypatch):
        """Standard TIME_PERIOD column name is accepted."""
        csv_bytes = _make_bis_csv_with_headers(
            "TIME_PERIOD", "OBS_VALUE", [("2020-01", 100.0)]
        )
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert not df.empty

    def test_time_col_fallback_lower(self, monkeypatch):
        """TIME_PERIOD column lookup is case-insensitive."""
        csv_bytes = _make_bis_csv_with_headers(
            "time_period", "obs_value", [("2020-01", 100.0)]
        )
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert not df.empty

    def test_time_fallback_name(self, monkeypatch):
        """'TIME' column name is accepted as fallback for time dimension."""
        csv_bytes = _make_bis_csv_with_headers(
            "TIME", "OBS_VALUE", [("2020-01", 100.0)]
        )
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert not df.empty

    def test_value_fallback_name(self, monkeypatch):
        """'VALUE' column name is accepted as fallback for value dimension."""
        csv_bytes = _make_bis_csv_with_headers(
            "TIME_PERIOD", "VALUE", [("2020-01", 100.0)]
        )
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert not df.empty

    def test_missing_time_column_returns_empty(self, monkeypatch):
        """CSV with no TIME_PERIOD or TIME column returns empty DataFrame."""
        csv_bytes = b"WRONG_COL,OBS_VALUE\n2020-01,100.0\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_missing_value_column_returns_empty(self, monkeypatch):
        """CSV with no OBS_VALUE or VALUE column returns empty DataFrame."""
        csv_bytes = b"TIME_PERIOD,BAD_VALUE\n2020-01,100.0\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]


# ---------------------------------------------------------------------------
# 4. Error / edge-case handling
# ---------------------------------------------------------------------------

class TestFetchBisNeerErrors:
    """Error handling and edge cases that should return empty DataFrame."""

    def test_unknown_bis_code_returns_empty(self):
        """Unknown BIS code (not in _BIS_TO_ISO3) returns empty DataFrame."""
        df = fetch_bis_neer("XX")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_unknown_code_zz_returns_empty(self):
        """'ZZ' is not a known BIS code — returns empty."""
        df = fetch_bis_neer("ZZ")
        assert df.empty

    def test_http_exception_returns_empty(self, monkeypatch):
        """Network exception from safe_get_bytes returns empty DataFrame."""
        def _raise(url, timeout=30.0):
            raise ConnectionError("network down")

        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", _raise)
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_timeout_exception_returns_empty(self, monkeypatch):
        """Timeout exception returns empty DataFrame."""
        def _raise(url, timeout=30.0):
            raise TimeoutError("timed out")

        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", _raise)
        df = fetch_bis_neer("US")
        assert df.empty

    def test_unparseable_csv_returns_empty(self, monkeypatch):
        """Non-CSV binary garbage returns empty DataFrame."""
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: b"\x00\x01\x02binary garbage!")
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_all_nan_obs_values_returns_empty(self, monkeypatch):
        """All OBS_VALUE entries are non-numeric (coerce→NaN), so all dropped → empty."""
        csv_bytes = b"TIME_PERIOD,OBS_VALUE\n2020-01,text\n2020-02,also_text\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df.empty

    def test_missing_obs_values_dropped(self, monkeypatch):
        """Rows with unparseable OBS_VALUE are dropped; valid rows survive."""
        csv_bytes = b"TIME_PERIOD,OBS_VALUE\n2020-01,100.0\n2020-02,BAD\n2020-03,110.0\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        # Q1-2020 should have 1 quarter (Jan+Mar valid, Feb dropped → mean=105)
        assert not df.empty
        assert float(df["neer_q"].iloc[0]) == pytest.approx(105.0)

    def test_invalid_date_format_rows_dropped(self, monkeypatch):
        """Rows with malformed date strings are coerced to NaT and dropped."""
        csv_bytes = b"TIME_PERIOD,OBS_VALUE\n2020-01,100.0\nBAD-DATE,99.0\n2020-03,101.0\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        # Bad date row dropped; remaining rows aggregated to Q1-2020
        assert not df.empty
        assert float(df["neer_q"].iloc[0]) == pytest.approx(100.5)  # mean(100, 101)

    def test_empty_csv_body_returns_empty(self, monkeypatch):
        """Empty CSV (header only) returns empty DataFrame."""
        csv_bytes = b"TIME_PERIOD,OBS_VALUE\n"
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_empty_bytes_raises_csv_parse_error_returns_empty(self, monkeypatch):
        """Completely empty bytes (no header) raises EmptyDataError in pd.read_csv → empty."""
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: b"")
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_invalid_utf8_bytes_triggers_csv_parse_error_returns_empty(self, monkeypatch):
        """Invalid UTF-8 bytes cause pd.read_csv UnicodeDecodeError → empty DataFrame."""
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: b"\xff\xfe")
        df = fetch_bis_neer("US")
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]


# ---------------------------------------------------------------------------
# 5. _BIS_TO_ISO3 mapping tests (pure Python, no monkeypatch needed)
# ---------------------------------------------------------------------------

class TestBisToIso3Mapping:
    """Tests for the _BIS_TO_ISO3 lookup dictionary."""

    def test_mapping_nonempty(self):
        """Mapping must contain at least one entry."""
        assert len(_BIS_TO_ISO3) > 0

    def test_all_values_are_length_3(self):
        """All ISO-3 values are exactly 3 characters."""
        for bis, iso3 in _BIS_TO_ISO3.items():
            assert len(iso3) == 3, f"{bis} -> {iso3!r} is not 3 characters"

    def test_all_keys_are_length_2(self):
        """All BIS codes (keys) are exactly 2 characters."""
        for bis in _BIS_TO_ISO3:
            assert len(bis) == 2, f"Key {bis!r} is not 2 characters"

    def test_all_values_are_uppercase(self):
        """All ISO-3 values are uppercase strings."""
        for bis, iso3 in _BIS_TO_ISO3.items():
            assert iso3.isupper(), f"{bis} -> {iso3!r} is not uppercase"

    def test_all_keys_are_uppercase(self):
        """All BIS keys are uppercase strings."""
        for bis in _BIS_TO_ISO3:
            assert bis.isupper(), f"Key {bis!r} is not uppercase"

    def test_specific_known_entries(self):
        """Spot-check a few canonical BIS→ISO3 mappings."""
        assert _BIS_TO_ISO3["US"] == "USA"
        assert _BIS_TO_ISO3["DE"] == "DEU"
        assert _BIS_TO_ISO3["JP"] == "JPN"
        assert _BIS_TO_ISO3["GB"] == "GBR"
        assert _BIS_TO_ISO3["FR"] == "FRA"

    def test_no_duplicate_values(self):
        """Each ISO-3 country appears only once in the mapping."""
        values = list(_BIS_TO_ISO3.values())
        assert len(values) == len(set(values)), "Duplicate ISO-3 codes found"


# ---------------------------------------------------------------------------
# 6. fetch_bis_neer_panel — panel aggregation
# ---------------------------------------------------------------------------

class TestFetchBisNeerPanel:
    """Tests for the multi-country panel fetcher."""

    def test_empty_codes_list_returns_empty(self):
        """Passing an empty codes list returns an empty DataFrame."""
        df = fetch_bis_neer_panel(codes=[])
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_all_unknown_codes_returns_empty(self):
        """Passing all unknown codes returns an empty DataFrame."""
        df = fetch_bis_neer_panel(codes=["XX", "YY", "ZZ"])
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]

    def test_single_known_country_with_mock(self, monkeypatch):
        """Panel with one valid country wraps the same data as single-country fetch."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-04", 99.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df_panel = fetch_bis_neer_panel(codes=["US"])
        df_single = fetch_bis_neer("US")
        assert len(df_panel) == len(df_single)
        pd.testing.assert_frame_equal(df_panel.reset_index(drop=True),
                                      df_single.reset_index(drop=True))

    def test_two_countries_panel_schema(self, monkeypatch):
        """Panel with two countries has the correct columns."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0), ("2020-04", 99.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel(codes=["US", "DE"])
        assert list(df.columns) == ["code", "qdate", "neer_q"]
        # 2 quarters × 2 countries = 4 rows
        assert len(df) == 4

    def test_two_countries_both_codes_present(self, monkeypatch):
        """Panel contains both ISO-3 codes."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel(codes=["US", "DE"])
        codes_in_result = set(df["code"].unique())
        assert "USA" in codes_in_result
        assert "DEU" in codes_in_result

    def test_panel_sorted_by_code_then_qdate(self, monkeypatch):
        """Panel output is sorted by (code, qdate) ascending."""
        csv_bytes = _make_bis_csv([("2020-04", 99.0), ("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel(codes=["US", "DE"])
        assert list(df.columns) == ["code", "qdate", "neer_q"]
        # Within each code, qdate should be ascending
        for code, grp in df.groupby("code"):
            qdates = list(grp["qdate"])
            assert qdates == sorted(qdates), f"qdate not sorted for code={code}"

    def test_panel_index_is_reset(self, monkeypatch):
        """Panel output has a clean integer RangeIndex."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel(codes=["US", "DE"])
        assert isinstance(df.index, pd.RangeIndex)
        assert list(df.index) == list(range(len(df)))

    def test_one_country_fails_other_succeeds(self, monkeypatch):
        """If one country raises and the other succeeds, panel has only the success."""
        def _selective(url: str, timeout: float = 30.0) -> bytes:
            if ".DE." in url or "M.N.B.DE" in url:
                raise ConnectionError("DE failed")
            return _make_bis_csv([("2020-01", 100.0)])

        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", _selective)
        df = fetch_bis_neer_panel(codes=["US", "DE"])
        assert not df.empty
        assert set(df["code"].unique()) == {"USA"}

    def test_default_codes_uses_all_bis_to_iso3_keys(self, monkeypatch):
        """When codes=None, panel iterates over all _BIS_TO_ISO3 keys."""
        csv_bytes = _make_bis_csv([("2020-01", 100.0)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel()  # codes=None → default to all keys
        expected_codes = {v for v in _BIS_TO_ISO3.values()}
        actual_codes = set(df["code"].unique())
        assert expected_codes == actual_codes

    def test_panel_neer_q_values_finite(self, monkeypatch):
        """All neer_q values in the panel are finite."""
        csv_bytes = _make_bis_csv([("2020-01", 98.5), ("2020-04", 101.2)])
        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", lambda url, timeout=30.0: csv_bytes)
        df = fetch_bis_neer_panel(codes=["US", "JP"])
        assert df["neer_q"].notna().all()
        assert np.isfinite(df["neer_q"].values).all()

    def test_all_fail_network_returns_empty_panel(self, monkeypatch):
        """If all fetches fail, panel returns an empty DataFrame."""
        def _raise(url, timeout=30.0):
            raise ConnectionError("all down")

        monkeypatch.setattr("puremacro.bis_neer.safe_get_bytes", _raise)
        df = fetch_bis_neer_panel(codes=["US", "DE", "JP"])
        assert df.empty
        assert list(df.columns) == ["code", "qdate", "neer_q"]


# ---------------------------------------------------------------------------
# 7. _EMPTY sentinel
# ---------------------------------------------------------------------------

class TestEmptySentinel:
    """Tests for the module-level _EMPTY sentinel."""

    def test_empty_sentinel_has_correct_columns(self):
        """_EMPTY has exactly [code, qdate, neer_q] columns."""
        assert list(_EMPTY.columns) == ["code", "qdate", "neer_q"]

    def test_empty_sentinel_has_no_rows(self):
        """_EMPTY has zero rows."""
        assert len(_EMPTY) == 0

    def test_returned_empty_is_copy_not_sentinel(self):
        """fetch_bis_neer returns a copy of _EMPTY, not the sentinel itself.

        Modifying the returned copy must not affect a subsequent call.
        """
        df1 = fetch_bis_neer("XX")  # returns _EMPTY.copy()
        df2 = fetch_bis_neer("YY")  # returns a fresh _EMPTY.copy()
        # Mutating df1's columns should not bleed into df2
        df1["extra"] = "mutated"
        assert list(df2.columns) == ["code", "qdate", "neer_q"]
