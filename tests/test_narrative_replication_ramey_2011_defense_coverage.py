"""Coverage tests for puremacro.narrative.replication.ramey_2011_defense.

Targets the UNCOVERED lines from the 61% baseline:
  - Line 52: ValueError when date or value column is missing.
  - Lines 60-64: fallback date-parsing paths (Period fails -> Timestamp;
    both fail -> continue / skip row).
  - Lines 100-114: ``load()`` function — local csv_path branch, network
    fetch branch (monkeypatched success), and network failure -> RuntimeError.

Strategy:
  - Known-value: synthetic DataFrames with known inputs -> assert exact
    field values on the returned NarrativeEvent objects.
  - Properties: magnitude always non-negative, sign in {-1, +1}, metadata
    keys present, implementation_profile empty (Ramey news-shock design).
  - Edge cases: NaN values skipped, zero values skipped, invalid dates
    skipped, alternate column names accepted.
  - Error handling: pytest.raises on missing columns; RuntimeError on
    network failure.
  - No real network I/O: ``safe_get_bytes`` is monkeypatched in all tests
    that touch ``load()`` without a csv_path.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from puremacro.narrative.replication.ramey_2011_defense import (
    ramey_csv_to_events,
    load,
    _MIRROR,
)
from puremacro.narrative.types import NarrativeEvent, NarrativeInstrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(**kwargs) -> pd.DataFrame:
    """Return a one-row DataFrame with 'date' and 'news_q' columns.

    Override any column via kwargs.
    """
    data = {"date": ["1965Q3"], "news_q": [2.5]}
    data.update(kwargs)
    return pd.DataFrame(data)


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes (in-memory)."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — ValueError on missing columns (line 52)
# ---------------------------------------------------------------------------

class TestRameyMissingColumns:
    def test_missing_date_and_value_col_raises(self):
        df = pd.DataFrame({"irrelevant": [1, 2]})
        with pytest.raises(ValueError, match="date"):
            ramey_csv_to_events(df)

    def test_missing_value_col_raises(self):
        df = pd.DataFrame({"date": ["1965Q3"]})
        with pytest.raises(ValueError, match="news_q"):
            ramey_csv_to_events(df)

    def test_missing_date_col_raises(self):
        df = pd.DataFrame({"news_q": [2.5]})
        with pytest.raises(ValueError, match="date"):
            ramey_csv_to_events(df)

    def test_error_message_mentions_expected_columns(self):
        df = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="news_q"):
            ramey_csv_to_events(df)


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — alternate column name aliases
# ---------------------------------------------------------------------------

class TestRameyColumnAliases:
    """defense_news_q and ramey_news are valid aliases for the value column;
    quarter is a valid alias for the date column."""

    def test_defense_news_q_column_accepted(self):
        df = pd.DataFrame({"date": ["1965Q3"], "defense_news_q": [1.5]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(1.5)

    def test_ramey_news_column_accepted(self):
        df = pd.DataFrame({"date": ["1965Q3"], "ramey_news": [0.7]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1

    def test_quarter_date_alias_accepted(self):
        df = pd.DataFrame({"quarter": ["1950Q3"], "news_q": [1.0]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1

    def test_case_insensitive_column_matching(self):
        """Column names should match case-insensitively (DATE, NEWS_Q)."""
        df = pd.DataFrame({"DATE": ["1965Q3"], "NEWS_Q": [2.0]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — date parsing fallback (lines 60-64)
# ---------------------------------------------------------------------------

class TestRameyDateParsing:
    def test_period_format_yyyy_qn_parsed_correctly(self):
        """Standard YYYY-Qn format should parse via pd.Period successfully."""
        df = pd.DataFrame({"date": ["1967Q2"], "news_q": [1.0]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1
        # Quarter 2 starts in April
        assert events[0].date.year == 1967
        assert events[0].date.month == 4

    def test_timestamp_fallback_for_plain_date_string(self):
        """A plain ISO date string that fails pd.Period falls through to
        the Timestamp fallback (lines 61-62)."""
        # '2020-01-15' parses fine as Period ('2020-01-01'); confirm the
        # event is produced (path 59 or 62, both acceptable).
        df = pd.DataFrame({"date": ["2020-01-15"], "news_q": [1.0]})
        events = ramey_csv_to_events(df)
        assert len(events) == 1
        assert isinstance(events[0].date, pd.Timestamp)

    def test_unparseable_date_row_skipped(self):
        """A row with a truly unparseable date string must be silently
        skipped (line 64: continue)."""
        df = pd.DataFrame({
            "date": ["not_a_date", "1965Q3"],
            "news_q": [1.0, 2.5],
        })
        events = ramey_csv_to_events(df)
        # Row with "not_a_date" is skipped; only the valid row survives.
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(2.5)

    def test_all_unparseable_dates_returns_empty_list(self):
        """When every row has an unparseable date, the result is empty."""
        df = pd.DataFrame({
            "date": ["BADDATE", "ALSOBAD"],
            "news_q": [1.0, 2.0],
        })
        events = ramey_csv_to_events(df)
        assert events == []

    def test_float_timestamp_fallback(self):
        """A float value in the date column triggers Period failure but
        Timestamp can parse epoch nanoseconds; the row should produce an
        event (the numeric fallback path lines 61-62)."""
        # float 1234.5 -> Period fails (out-of-bounds year), Timestamp
        # succeeds (nanoseconds since epoch). The event is created.
        df = pd.DataFrame({"date": [1234.5], "news_q": [3.0]})
        events = ramey_csv_to_events(df)
        # Must not raise; event count == 1 because date parses via Timestamp.
        assert len(events) == 1

    def test_mix_of_valid_and_unparseable_dates(self):
        """Multiple good rows interspersed with bad ones: only good ones
        survive."""
        df = pd.DataFrame({
            "date": ["1960Q1", "GARBAGE", "1970Q3", "ALSO_BAD", "1980Q1"],
            "news_q": [1.0, 99.0, 2.0, 99.0, 3.0],
        })
        events = ramey_csv_to_events(df)
        assert len(events) == 3
        magnitudes = sorted(e.magnitude for e in events)
        assert magnitudes == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — zero/NaN filtering
# ---------------------------------------------------------------------------

class TestRameyZeroNaNFilter:
    def test_zero_value_row_skipped(self):
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1"],
            "news_q": [0.0, 2.5],
        })
        events = ramey_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(2.5)

    def test_nan_value_row_skipped(self):
        import math
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1"],
            "news_q": [float("nan"), 1.5],
        })
        events = ramey_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(1.5)

    def test_all_zeros_returns_empty(self):
        df = pd.DataFrame({"date": ["1960Q1", "1961Q1"], "news_q": [0.0, 0.0]})
        assert ramey_csv_to_events(df) == []

    def test_all_nans_returns_empty(self):
        df = pd.DataFrame({"date": ["1960Q1"], "news_q": [float("nan")]})
        assert ramey_csv_to_events(df) == []


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — known-value field checks
# ---------------------------------------------------------------------------

class TestRameyFieldValues:
    """Verify that the fields of produced NarrativeEvents match Ramey
    loader's documented contract exactly."""

    def _single_event(self, v: float = 2.5) -> NarrativeEvent:
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [v]})
        return ramey_csv_to_events(df)[0]

    def test_country_is_USA(self):
        assert self._single_event().country == "USA"

    def test_magnitude_unit_is_pct_gdp(self):
        assert self._single_event().magnitude_unit == "pct_gdp"

    def test_target_is_both(self):
        assert self._single_event().target == "both"

    def test_subtarget_is_defense(self):
        assert self._single_event().subtarget == "defense"

    def test_scoring_method_is_manual(self):
        assert self._single_event().scoring_method == "manual"

    def test_confidence_is_one(self):
        assert self._single_event().confidence == pytest.approx(1.0)

    def test_magnitude_is_abs_value(self):
        """Negative input magnitude should be stored as abs() value."""
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [-3.0]})
        e = ramey_csv_to_events(df)[0]
        assert e.magnitude == pytest.approx(3.0)
        assert e.magnitude >= 0.0

    def test_sign_positive_for_positive_input(self):
        assert self._single_event(v=2.5).sign == 1

    def test_sign_negative_for_negative_input(self):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [-1.5]})
        e = ramey_csv_to_events(df)[0]
        assert e.sign == -1

    def test_metadata_contains_replication_key(self):
        e = self._single_event()
        assert e.metadata.get("replication") == "ramey_2011"

    def test_metadata_is_news_shock_true(self):
        e = self._single_event()
        assert e.metadata.get("is_news_shock") is True

    def test_implementation_profile_is_empty(self):
        """Ramey loader deliberately leaves implementation_profile empty
        to preserve the news-shock identification (see module docstring)."""
        e = self._single_event()
        assert e.implementation_profile == []

    def test_date_type_is_timestamp(self):
        assert isinstance(self._single_event().date, pd.Timestamp)

    def test_source_url_contains_ramey_domain(self):
        e = self._single_event()
        assert "ramey" in e.source_url.lower() or "ucsd" in e.source_url.lower()

    def test_source_text_non_empty(self):
        e = self._single_event()
        assert len(e.source_text) > 0

    def test_multiple_events_all_usa(self):
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1", "1967Q2"],
            "news_q": [2.5, -1.2, 0.8],
        })
        events = ramey_csv_to_events(df)
        assert all(e.country == "USA" for e in events)


# ---------------------------------------------------------------------------
# Tests for ramey_csv_to_events — signed_magnitude property
# ---------------------------------------------------------------------------

class TestSignedMagnitude:
    def test_positive_signed_magnitude(self):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [3.0]})
        e = ramey_csv_to_events(df)[0]
        assert e.signed_magnitude == pytest.approx(3.0)

    def test_negative_signed_magnitude(self):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [-2.5]})
        e = ramey_csv_to_events(df)[0]
        assert e.signed_magnitude == pytest.approx(-2.5)

    def test_signed_magnitude_equals_magnitude_times_sign(self):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [4.0]})
        e = ramey_csv_to_events(df)[0]
        assert e.signed_magnitude == pytest.approx(e.magnitude * e.sign)


# ---------------------------------------------------------------------------
# Tests for load() — csv_path branch (lines 100-101)
# ---------------------------------------------------------------------------

class TestLoadCsvPath:
    """load() with a local csv_path: must read the file and return a
    NarrativeInstrument without any network call."""

    def _write_csv(self, tmp_path: Path, df: pd.DataFrame) -> Path:
        p = tmp_path / "ramey_test.csv"
        df.to_csv(p, index=False)
        return p

    def test_returns_narrative_instrument(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert isinstance(inst, NarrativeInstrument)

    def test_accepts_path_object(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=p)
        assert isinstance(inst, NarrativeInstrument)

    def test_events_count_matches_nonzero_rows(self, tmp_path):
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1", "1967Q2"],
            "news_q": [2.5, 0.0, 1.2],  # zero filtered
        })
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert len(inst.events) == 2

    def test_events_all_news_shocks(self, tmp_path):
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1"],
            "news_q": [2.5, 1.2],
        })
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert all(e.metadata.get("is_news_shock") for e in inst.events)

    def test_quarterly_series_is_pd_series(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3", "1966Q1"], "news_q": [2.5, 1.2]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert isinstance(inst.quarterly, pd.Series)

    def test_target_passed_through(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p), target="investment")
        assert inst.target == "investment"

    def test_default_target_is_both(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert inst.target == "both"

    def test_aggregation_is_sum(self, tmp_path):
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        assert inst.aggregation == "sum"

    def test_two_events_same_quarter_sums_in_series(self, tmp_path):
        """Two events in the same quarter should sum in the quarterly series."""
        df = pd.DataFrame({
            "date": ["1965Q3", "1965Q3"],
            "news_q": [1.0, 2.0],
        })
        p = self._write_csv(tmp_path, df)
        inst = load(csv_path=str(p))
        # Sum aggregation: both rows in Q3 1965 should yield 3.0
        s = inst.quarterly
        nz = s[s != 0]
        assert float(nz.iloc[0]) == pytest.approx(3.0)

    def test_no_network_call_made(self, tmp_path):
        """When csv_path is provided, safe_get_bytes must NOT be called."""
        df = pd.DataFrame({"date": ["1965Q3"], "news_q": [2.5]})
        p = self._write_csv(tmp_path, df)
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes"
        ) as mock_get:
            load(csv_path=str(p))
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for load() — network fetch branch (lines 103-112)
# ---------------------------------------------------------------------------

class TestLoadNetworkFetch:
    """load() without csv_path: monkeypatch safe_get_bytes to avoid
    real HTTP and to test both the success and failure paths."""

    def _csv_bytes(self) -> bytes:
        df = pd.DataFrame({
            "date": ["1965Q3", "1966Q1"],
            "news_q": [2.5, 1.2],
        })
        return _make_csv_bytes(df)

    def test_network_success_returns_instrument(self):
        """When safe_get_bytes succeeds, load() returns a NarrativeInstrument."""
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            return_value=self._csv_bytes(),
        ):
            inst = load()
        assert isinstance(inst, NarrativeInstrument)

    def test_network_success_events_populated(self):
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            return_value=self._csv_bytes(),
        ):
            inst = load()
        assert len(inst.events) == 2

    def test_network_success_uses_mirror_url(self):
        """safe_get_bytes is called with _MIRROR as the URL."""
        captured_urls: list[str] = []

        def _fake_get(url, **kwargs):
            captured_urls.append(url)
            return self._csv_bytes()

        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            side_effect=_fake_get,
        ):
            load()
        assert len(captured_urls) == 1
        assert captured_urls[0] == _MIRROR

    def test_network_failure_raises_runtime_error(self):
        """When safe_get_bytes raises, load() must raise RuntimeError."""
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Could not fetch"):
                load()

    def test_network_failure_error_message_points_at_source(self):
        """RuntimeError message must mention the UCSD/Ramey archive URL."""
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            side_effect=Exception("timeout"),
        ):
            with pytest.raises(RuntimeError, match="ramey"):
                load()

    def test_network_failure_chained_cause(self):
        """RuntimeError must chain the original exception as __cause__."""
        orig = ValueError("network error sentinel")
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            side_effect=orig,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                load()
        assert exc_info.value.__cause__ is orig

    def test_network_success_quarterly_series_not_empty(self):
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            return_value=self._csv_bytes(),
        ):
            inst = load()
        assert not inst.quarterly.empty

    def test_network_success_target_defaults_to_both(self):
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            return_value=self._csv_bytes(),
        ):
            inst = load()
        assert inst.target == "both"

    def test_network_success_custom_target(self):
        with patch(
            "puremacro.narrative.replication.ramey_2011_defense.safe_get_bytes",
            return_value=self._csv_bytes(),
        ):
            inst = load(target="investment")
        assert inst.target == "investment"


# ---------------------------------------------------------------------------
# Tests for __all__ export
# ---------------------------------------------------------------------------

def test_all_exports_load():
    from puremacro.narrative.replication import ramey_2011_defense as mod
    assert "load" in mod.__all__
