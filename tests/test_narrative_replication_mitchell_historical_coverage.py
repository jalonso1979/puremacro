"""Coverage tests for puremacro.narrative.replication.mitchell_historical.

Targets all previously-uncovered lines (19% baseline):
  - mitchell_csv_to_events  (lines 43–86)
  - load                    (lines 102–128)
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest.mock as mock
from pathlib import Path

import pandas as pd
import pytest

from puremacro.narrative.replication.mitchell_historical import (
    load,
    mitchell_csv_to_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df(**extra) -> pd.DataFrame:
    """One-row DataFrame with all required columns; override via kwargs."""
    row = {"country": "USA", "year": 1965, "gdp_share": 0.20}
    row.update(extra)
    return pd.DataFrame([row])


def _multi_df() -> pd.DataFrame:
    """Two countries, two years for integration-style tests."""
    return pd.DataFrame([
        {"country": "USA", "year": 1965, "gdp_share": 0.20},
        {"country": "DEU", "year": 1970, "gdp_share": 25.0},
    ])


def _csv_file(df: pd.DataFrame) -> str:
    """Write df to a temp CSV and return the path (caller must delete)."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f, index=False)
    f.close()
    return f.name


# ===========================================================================
# mitchell_csv_to_events — column detection
# ===========================================================================

class TestCsvToEventsColumnDetection:
    """Column aliases and validation."""

    def test_requires_country_column(self):
        """Missing country column raises ValueError."""
        df = pd.DataFrame([{"year": 1965, "gdp_share": 0.20}])
        with pytest.raises(ValueError, match="country"):
            mitchell_csv_to_events(df)

    def test_requires_year_column(self):
        """Missing year column raises ValueError."""
        df = pd.DataFrame([{"country": "USA", "gdp_share": 0.20}])
        with pytest.raises(ValueError, match="year"):
            mitchell_csv_to_events(df)

    def test_requires_value_column(self):
        """Missing all value column aliases raises ValueError."""
        df = pd.DataFrame([{"country": "USA", "year": 1965}])
        with pytest.raises(ValueError, match="gdp_share"):
            mitchell_csv_to_events(df)

    def test_iso3_alias_for_country(self):
        """'iso3' column is accepted as country alias."""
        df = pd.DataFrame([{"iso3": "GBR", "year": 1960, "gdp_share": 0.22}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4
        assert events[0].country == "GBR"

    def test_govt_pct_gdp_alias(self):
        """'govt_pct_gdp' column is accepted as value alias."""
        df = pd.DataFrame([{"country": "FRA", "year": 1960, "govt_pct_gdp": 22.0}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4

    def test_g_pct_gdp_alias(self):
        """'g_pct_gdp' column is accepted as value alias."""
        df = pd.DataFrame([{"country": "FRA", "year": 1960, "g_pct_gdp": 18.0}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4

    def test_value_alias(self):
        """'value' column is accepted as value alias."""
        df = pd.DataFrame([{"country": "FRA", "year": 1960, "value": 18.0}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4

    def test_case_insensitive_columns(self):
        """Column names are matched case-insensitively."""
        df = pd.DataFrame([{"Country": "USA", "Year": 1965, "GDP_Share": 0.20}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4
        assert events[0].country == "USA"


# ===========================================================================
# mitchell_csv_to_events — magnitude normalisation
# ===========================================================================

class TestCsvToEventsMagnitudeNormalisation:
    """Value <= 1 treated as fraction; > 1 treated as percent directly."""

    def test_fraction_under_one_converted_to_percent(self):
        """gdp_share=0.20 → pct_gdp=20.0 → quarterly magnitude=5.0."""
        events = mitchell_csv_to_events(_simple_df(gdp_share=0.20))
        assert len(events) == 4
        for e in events:
            assert e.magnitude == pytest.approx(5.0)
            assert e.magnitude_unit == "pct_gdp"

    def test_value_greater_than_one_used_directly_as_percent(self):
        """gdp_share=20.0 (already %) → quarterly magnitude=5.0."""
        events = mitchell_csv_to_events(_simple_df(gdp_share=20.0))
        for e in events:
            assert e.magnitude == pytest.approx(5.0)

    def test_fraction_and_percent_yield_same_magnitude(self):
        """0.20 and 20.0 are normalised to the same quarterly magnitude."""
        events_frac = mitchell_csv_to_events(_simple_df(gdp_share=0.20))
        events_pct  = mitchell_csv_to_events(_simple_df(gdp_share=20.0))
        assert events_frac[0].magnitude == pytest.approx(events_pct[0].magnitude)

    def test_known_value_exact_arithmetic(self):
        """gdp_share=0.28 → pct_gdp=28.0 → quarterly=7.0."""
        events = mitchell_csv_to_events(_simple_df(gdp_share=0.28))
        for e in events:
            assert e.magnitude == pytest.approx(7.0)

    def test_large_percent_value(self):
        """gdp_share=48.0 → quarterly=12.0."""
        events = mitchell_csv_to_events(_simple_df(gdp_share=48.0))
        for e in events:
            assert e.magnitude == pytest.approx(12.0)


# ===========================================================================
# mitchell_csv_to_events — skipping invalid rows
# ===========================================================================

class TestCsvToEventsSkipInvalidRows:
    """NaN, zero, and negative values are silently dropped."""

    def test_nan_value_skipped(self):
        """NaN gdp_share → no event emitted."""
        df = _simple_df(gdp_share=float("nan"))
        assert mitchell_csv_to_events(df) == []

    def test_zero_value_skipped(self):
        """gdp_share=0.0 → no event emitted."""
        assert mitchell_csv_to_events(_simple_df(gdp_share=0.0)) == []

    def test_negative_value_skipped(self):
        """gdp_share=-0.05 → no event emitted."""
        assert mitchell_csv_to_events(_simple_df(gdp_share=-0.05)) == []

    def test_empty_dataframe_returns_empty_list(self):
        """Empty DataFrame → empty list, no error."""
        df = pd.DataFrame(columns=["country", "year", "gdp_share"])
        assert mitchell_csv_to_events(df) == []

    def test_mixed_valid_and_invalid_rows(self):
        """Only valid rows produce events; 1968 is the only valid year here."""
        df = pd.DataFrame([
            {"country": "USA", "year": 1965, "gdp_share": float("nan")},
            {"country": "USA", "year": 1966, "gdp_share": 0.0},
            {"country": "USA", "year": 1967, "gdp_share": -1.0},
            {"country": "USA", "year": 1968, "gdp_share": 0.15},
        ])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4
        assert all(e.date.year == 1968 for e in events)


# ===========================================================================
# mitchell_csv_to_events — quarterly date structure
# ===========================================================================

class TestCsvToEventsQuarterlyDates:
    """Each annual row produces exactly 4 quarterly events."""

    def test_exactly_four_events_per_row(self):
        """One data row → 4 quarterly events."""
        events = mitchell_csv_to_events(_simple_df())
        assert len(events) == 4

    def test_quarterly_months_are_correct(self):
        """Q1–Q4 anchors at months 1, 4, 7, 10."""
        events = mitchell_csv_to_events(_simple_df(year=1960))
        months = [e.date.month for e in events]
        assert months == [1, 4, 7, 10]

    def test_quarterly_year_matches_input(self):
        """All 4 events carry the source year."""
        events = mitchell_csv_to_events(_simple_df(year=1972))
        assert all(e.date.year == 1972 for e in events)

    def test_quarterly_day_is_first(self):
        """Each quarterly event falls on day 1."""
        events = mitchell_csv_to_events(_simple_df(year=1965))
        assert all(e.date.day == 1 for e in events)

    def test_float_year_parsed_correctly(self):
        """year=1965.0 (common from pd.read_csv) → date.year == 1965."""
        df = pd.DataFrame([{"country": "USA", "year": 1965.0, "gdp_share": 0.20}])
        events = mitchell_csv_to_events(df)
        assert len(events) == 4
        assert all(e.date.year == 1965 for e in events)

    def test_two_countries_two_rows_eight_events(self):
        """Two distinct country-year rows → 8 events total."""
        events = mitchell_csv_to_events(_multi_df())
        assert len(events) == 8


# ===========================================================================
# mitchell_csv_to_events — event fields / metadata
# ===========================================================================

class TestCsvToEventsFields:
    """Verify hardcoded fields on generated events."""

    def test_country_uppercased(self):
        """Lowercase 'usa' is uppercased to 'USA'."""
        df = pd.DataFrame([{"country": "usa", "year": 1965, "gdp_share": 0.20}])
        events = mitchell_csv_to_events(df)
        assert all(e.country == "USA" for e in events)

    def test_sign_always_positive(self):
        """Mitchell events represent expenditure levels → sign=+1."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.sign == +1 for e in events)

    def test_confidence_is_0_6(self):
        """confidence=0.6 as documented."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.confidence == pytest.approx(0.6) for e in events)

    def test_target_is_both(self):
        """target='both' (expenditure affects investment and consumption)."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.target == "both" for e in events)

    def test_subtarget_is_general(self):
        """subtarget='general' (no category breakdown in Mitchell)."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.subtarget == "general" for e in events)

    def test_scoring_method_is_manual(self):
        """scoring_method='manual' (hand-coded from book)."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.scoring_method == "manual" for e in events)

    def test_replication_metadata_marker(self):
        """metadata['replication'] == 'mitchell_2013' on every event."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.metadata.get("replication") == "mitchell_2013" for e in events)

    def test_source_url_points_to_springer(self):
        """source_url references the Springer book link."""
        events = mitchell_csv_to_events(_simple_df())
        assert all("springer.com" in e.source_url for e in events)

    def test_implementation_profile_is_empty(self):
        """Empty implementation_profile → effective_profile falls back to announcement date."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.implementation_profile == [] for e in events)

    def test_magnitude_unit_is_pct_gdp(self):
        """magnitude_unit == 'pct_gdp'."""
        events = mitchell_csv_to_events(_simple_df())
        assert all(e.magnitude_unit == "pct_gdp" for e in events)


# ===========================================================================
# load() — csv_path branch
# ===========================================================================

class TestLoadCsvPath:
    """load(csv_path=...) without network calls."""

    def _write_csv(self, rows) -> str:
        df = pd.DataFrame(rows)
        return _csv_file(df)

    def test_returns_dict_keyed_by_country(self):
        """Result is a dict with ISO3 country keys."""
        path = self._write_csv([
            {"country": "USA", "year": 1965, "gdp_share": 0.20},
            {"country": "DEU", "year": 1970, "gdp_share": 0.22},
        ])
        try:
            out = load(csv_path=path)
            assert isinstance(out, dict)
            assert "USA" in out
            assert "DEU" in out
        finally:
            os.unlink(path)

    def test_values_are_narrative_instruments(self):
        """Each value is a NarrativeInstrument with a .quarterly Series."""
        from puremacro.narrative.types import NarrativeInstrument
        path = self._write_csv([{"country": "USA", "year": 1965, "gdp_share": 0.20}])
        try:
            out = load(csv_path=path)
            instr = out["USA"]
            assert isinstance(instr, NarrativeInstrument)
            assert isinstance(instr.quarterly, pd.Series)
        finally:
            os.unlink(path)

    def test_country_filter_reduces_output(self):
        """countries=['USA'] drops DEU from the result."""
        path = self._write_csv([
            {"country": "USA", "year": 1965, "gdp_share": 0.20},
            {"country": "DEU", "year": 1970, "gdp_share": 0.22},
        ])
        try:
            out = load(csv_path=path, countries=["USA"])
            assert set(out.keys()) == {"USA"}
        finally:
            os.unlink(path)

    def test_country_filter_case_insensitive(self):
        """countries=['usa'] (lowercase) should still match 'USA' rows."""
        path = self._write_csv([{"country": "USA", "year": 1965, "gdp_share": 0.20}])
        try:
            out = load(csv_path=path, countries=["usa"])
            assert "USA" in out
        finally:
            os.unlink(path)

    def test_year_range_filter_start_year(self):
        """start_year=1970 excludes rows from 1965."""
        path = self._write_csv([
            {"country": "USA", "year": 1965, "gdp_share": 0.20},
            {"country": "USA", "year": 1975, "gdp_share": 0.25},
        ])
        try:
            out = load(csv_path=path, start_year=1970, end_year=1979)
            instr = out["USA"]
            assert all(e.date.year >= 1970 for e in instr.events)
        finally:
            os.unlink(path)

    def test_year_range_filter_end_year(self):
        """end_year=1969 excludes rows from 1975."""
        path = self._write_csv([
            {"country": "USA", "year": 1965, "gdp_share": 0.20},
            {"country": "USA", "year": 1975, "gdp_share": 0.25},
        ])
        try:
            out = load(csv_path=path, start_year=1960, end_year=1969)
            instr = out["USA"]
            assert all(e.date.year <= 1969 for e in instr.events)
        finally:
            os.unlink(path)

    def test_data_outside_default_range_excluded(self):
        """Year 1985 is outside default end_year=1979 → not in output."""
        path = self._write_csv([
            {"country": "GBR", "year": 1985, "gdp_share": 0.30},
            {"country": "GBR", "year": 1970, "gdp_share": 0.22},
        ])
        try:
            out = load(csv_path=path)
            assert "GBR" in out
            instr = out["GBR"]
            assert all(e.date.year != 1985 for e in instr.events)
        finally:
            os.unlink(path)

    def test_float_year_column_handled_in_load(self):
        """pd.read_csv sometimes returns year as float; load() must handle it."""
        # Write CSV with float-looking year
        path = self._write_csv([{"country": "USA", "year": 1965.0, "gdp_share": 0.20}])
        try:
            out = load(csv_path=path, start_year=1960, end_year=1979)
            assert "USA" in out
        finally:
            os.unlink(path)

    def test_events_have_correct_magnitude_sum_per_year(self):
        """4 quarterly events sum to pct_gdp (the annual value normalised)."""
        path = self._write_csv([{"country": "USA", "year": 1965, "gdp_share": 0.20}])
        try:
            out = load(csv_path=path)
            total = sum(e.magnitude for e in out["USA"].events)
            # 0.20 < 1 → pct_gdp = 20.0; 4 quarters × 5.0 = 20.0
            assert total == pytest.approx(20.0)
        finally:
            os.unlink(path)

    def test_instrument_aggregation_is_sum(self):
        """NarrativeInstrument was built with aggregation='sum'."""
        path = self._write_csv([{"country": "USA", "year": 1965, "gdp_share": 0.20}])
        try:
            out = load(csv_path=path)
            assert out["USA"].aggregation == "sum"
        finally:
            os.unlink(path)

    def test_no_countries_filter_returns_all(self):
        """Passing countries=None (default) returns all countries."""
        path = self._write_csv([
            {"country": "USA", "year": 1965, "gdp_share": 0.20},
            {"country": "DEU", "year": 1965, "gdp_share": 0.22},
            {"country": "JPN", "year": 1965, "gdp_share": 0.18},
        ])
        try:
            out = load(csv_path=path)
            assert set(out.keys()) == {"USA", "DEU", "JPN"}
        finally:
            os.unlink(path)


# ===========================================================================
# load() — network branch (mocked)
# ===========================================================================

class TestLoadNetworkBranch:
    """load() without csv_path uses safe_get_bytes (mocked)."""

    _CSV = "country,year,gdp_share\nUSA,1965,0.20\nDEU,1970,22.5\n"

    def _mock_bytes(self):
        return self._CSV.encode("utf-8")

    def test_network_path_returns_dict(self):
        """Mocked network fetch → normal dict result."""
        with mock.patch(
            "puremacro.narrative.replication.mitchell_historical.safe_get_bytes",
            return_value=self._mock_bytes(),
        ):
            out = load(start_year=1960, end_year=1979)
        assert "USA" in out
        assert "DEU" in out

    def test_network_failure_raises_runtime_error(self):
        """Network exception is wrapped in RuntimeError with helpful message."""
        with mock.patch(
            "puremacro.narrative.replication.mitchell_historical.safe_get_bytes",
            side_effect=Exception("network down"),
        ):
            with pytest.raises(RuntimeError, match="Mitchell 2013"):
                load()

    def test_network_error_chained_from_original(self):
        """The RuntimeError should chain from the original exception (__cause__)."""
        orig = OSError("timeout")
        with mock.patch(
            "puremacro.narrative.replication.mitchell_historical.safe_get_bytes",
            side_effect=orig,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                load()
        assert exc_info.value.__cause__ is orig

    def test_network_country_filter_applied(self):
        """countries=['USA'] applied after network fetch."""
        with mock.patch(
            "puremacro.narrative.replication.mitchell_historical.safe_get_bytes",
            return_value=self._mock_bytes(),
        ):
            out = load(countries=["USA"], start_year=1960, end_year=1979)
        assert set(out.keys()) == {"USA"}

    def test_network_year_filter_applied(self):
        """start_year/end_year applied after network fetch."""
        with mock.patch(
            "puremacro.narrative.replication.mitchell_historical.safe_get_bytes",
            return_value=self._mock_bytes(),
        ):
            # Only 1965 row is in 1960-1969; 1970 row is excluded
            out = load(start_year=1960, end_year=1969)
        assert "USA" in out
        assert "DEU" not in out
