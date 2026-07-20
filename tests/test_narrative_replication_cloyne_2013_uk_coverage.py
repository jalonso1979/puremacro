"""Coverage tests for puremacro.narrative.replication.cloyne_2013_uk.

This module had 0% prior coverage (no prior tests found).  This file
covers every branch and public function:

  - cloyne_csv_to_events: column-name detection (exog, exog_tax, y, tax_shock;
    date, quarter; effective_date, effective_quarter)
  - cloyne_csv_to_events: ValueError when date or value column is missing
  - _to_quarter: None and float-NaN fast-exit
  - _to_quarter: empty-string fast-exit
  - _to_quarter: pd.Period parse (happy path)
  - _to_quarter: fallback pd.Timestamp parse for ISO date strings
  - _to_quarter: both paths fail -> None -> row skipped
  - Row filter: v == 0.0 or pd.isna(v) skipped
  - effective_date profile: eff_q >= date -> profile set
  - effective_date profile: eff_q < date -> profile empty
  - effective_date profile: column None -> profile empty
  - NarrativeEvent contract: country=GBR, magnitude=abs(v), sign, unit, target,
    subtarget, scoring_method, confidence, metadata, source_url
  - load() csv_path branch: Path and str
  - load() network success (monkeypatched)
  - load() network failure -> RuntimeError, chained __cause__
  - load() custom target forwarded
  - NarrativeInstrument.quarterly is pd.Series with nonzero entries
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.replication.cloyne_2013_uk import (
    cloyne_csv_to_events,
    load,
)
from puremacro.narrative.types import NarrativeInstrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows, *, cols=("date", "exog")):
    """Build a minimal DataFrame with the given column names and rows."""
    return pd.DataFrame(rows, columns=list(cols))


def _minimal_csv_bytes(content: str = "date,exog\n2000Q1,0.5\n2001Q2,-0.3\n") -> bytes:
    return content.encode("utf-8")


# ---------------------------------------------------------------------------
# Column detection: missing columns raise ValueError
# ---------------------------------------------------------------------------

class TestMissingColumnsRaises:
    """ValueError when the required date or value column is absent."""

    def test_no_date_no_value_raises(self):
        df = pd.DataFrame({"foo": ["2000Q1"], "bar": [0.5]})
        with pytest.raises(ValueError, match="date"):
            cloyne_csv_to_events(df)

    def test_has_date_no_value_raises(self):
        df = pd.DataFrame({"date": ["2000Q1"], "shock_unknown": [0.5]})
        with pytest.raises(ValueError):
            cloyne_csv_to_events(df)

    def test_has_value_no_date_raises(self):
        df = pd.DataFrame({"not_a_date": ["2000Q1"], "exog": [0.5]})
        with pytest.raises(ValueError):
            cloyne_csv_to_events(df)

    def test_empty_df_wrong_cols_raises(self):
        df = pd.DataFrame(columns=["alpha", "beta"])
        with pytest.raises(ValueError):
            cloyne_csv_to_events(df)


# ---------------------------------------------------------------------------
# Column detection: accepted synonyms
# ---------------------------------------------------------------------------

class TestColumnAliases:
    """All recognised column-name synonyms are accepted."""

    def test_date_and_exog(self):
        df = pd.DataFrame({"date": ["2000Q1"], "exog": [0.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1

    def test_quarter_and_exog_tax(self):
        df = pd.DataFrame({"quarter": ["2000Q1"], "exog_tax": [0.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2000-01-01")

    def test_date_and_y(self):
        df = pd.DataFrame({"date": ["2000Q1"], "y": [1.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(1.5)

    def test_date_and_tax_shock(self):
        df = pd.DataFrame({"date": ["2000Q1"], "tax_shock": [-0.8]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].sign == -1

    def test_effective_quarter_alias(self):
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_quarter": ["2000Q3"],
        })
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].implementation_profile == [(pd.Timestamp("2000-07-01"), 1.0)]

    def test_effective_date_alias(self):
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": ["2000Q3"],
        })
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].implementation_profile == [(pd.Timestamp("2000-07-01"), 1.0)]


# ---------------------------------------------------------------------------
# _to_quarter: None and NaN fast-exit paths
# ---------------------------------------------------------------------------

class TestToQuarterNoneNaN:
    """Rows whose date parses to None are silently skipped."""

    def test_none_date_skipped(self):
        df = pd.DataFrame({"date": [None, "2000Q1"], "exog": [0.5, 0.3]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2000-01-01")

    def test_float_nan_date_skipped(self):
        df = pd.DataFrame({"date": [np.nan, "2000Q1"], "exog": [0.5, 0.3]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1

    def test_all_nan_dates_returns_empty_list(self):
        df = pd.DataFrame({"date": [np.nan], "exog": [0.5]})
        events = cloyne_csv_to_events(df)
        assert events == []

    def test_empty_string_date_skipped(self):
        df = pd.DataFrame({"date": ["", "2000Q1"], "exog": [0.5, 0.3]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# _to_quarter: Period parse (happy path) vs Timestamp fallback
# ---------------------------------------------------------------------------

class TestToQuarterParsePaths:
    """Both parse paths hit correctly depending on input format."""

    def test_quarter_string_period_parse(self):
        # "2003Q2" succeeds via pd.Period directly.
        df = pd.DataFrame({"date": ["2003Q2"], "exog": [0.5]})
        events = cloyne_csv_to_events(df)
        assert events[0].date == pd.Timestamp("2003-04-01")

    def test_iso_date_string_fallback(self):
        # "2000-01-15" fails pd.Period but succeeds via pd.Timestamp.
        df = pd.DataFrame({"date": ["2000-01-15"], "exog": [0.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        # Lands in 2000Q1 -> start of Q1
        assert events[0].date == pd.Timestamp("2000-01-01")

    def test_timestamp_mid_quarter_rounds_to_quarter_start(self):
        # "2003-05-15" is mid-Q2; fallback wraps to 2003Q2 start.
        df = pd.DataFrame({"date": ["2003-05-15"], "exog": [0.4]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2003-04-01")

    def test_completely_unparseable_date_row_skipped(self):
        # Both parse paths fail -> None -> row silently skipped.
        df = pd.DataFrame({"date": ["xyz_bad", "2000Q1"], "exog": [0.6, 0.3]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2000-01-01")


# ---------------------------------------------------------------------------
# Row-level value filters
# ---------------------------------------------------------------------------

class TestValueFilters:
    """Rows with v == 0.0 or v == NaN are dropped."""

    def test_zero_value_row_skipped(self):
        df = pd.DataFrame({"date": ["2000Q1", "2000Q2"], "exog": [0.0, 0.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(0.5)

    def test_nan_value_row_skipped(self):
        df = pd.DataFrame({"date": ["2000Q1", "2000Q2"], "exog": [np.nan, 0.5]})
        events = cloyne_csv_to_events(df)
        assert len(events) == 1

    def test_all_zeros_returns_empty(self):
        df = pd.DataFrame({"date": ["2000Q1", "2000Q2"], "exog": [0.0, 0.0]})
        events = cloyne_csv_to_events(df)
        assert events == []

    def test_all_nan_values_returns_empty(self):
        df = pd.DataFrame({"date": ["2000Q1", "2000Q2"], "exog": [np.nan, np.nan]})
        events = cloyne_csv_to_events(df)
        assert events == []


# ---------------------------------------------------------------------------
# Implementation profile logic
# ---------------------------------------------------------------------------

class TestImplementationProfile:
    """effective_date column controls the implementation_profile."""

    def test_no_eff_col_profile_is_empty(self):
        df = pd.DataFrame({"date": ["2000Q2"], "exog": [0.5]})
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == []

    def test_eff_date_after_announce_profile_set(self):
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": ["2000Q3"],
        })
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == [(pd.Timestamp("2000-07-01"), 1.0)]

    def test_eff_date_equal_announce_profile_set(self):
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": ["2000Q2"],
        })
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == [(pd.Timestamp("2000-04-01"), 1.0)]

    def test_eff_date_before_announce_profile_empty(self):
        # eff_q < date -> guard fails -> profile stays empty
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": ["2000Q1"],
        })
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == []

    def test_eff_date_none_profile_empty(self):
        # None effective_date -> _to_quarter returns None -> no profile
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": [None],
        })
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == []

    def test_eff_date_unparseable_profile_empty(self):
        # Unparseable effective_date -> _to_quarter returns None -> no profile
        df = pd.DataFrame({
            "date": ["2000Q2"],
            "exog": [0.5],
            "effective_date": ["xyz_bad"],
        })
        events = cloyne_csv_to_events(df)
        assert events[0].implementation_profile == []


# ---------------------------------------------------------------------------
# NarrativeEvent contract: known-value / property tests
# ---------------------------------------------------------------------------

class TestEventContract:
    """Known-value tests for event fields emitted by cloyne_csv_to_events."""

    @pytest.fixture
    def events(self):
        df = pd.DataFrame({
            "date": ["2000Q1", "2001Q2", "2003Q3"],
            "exog": [0.6, -0.4, 1.2],
        })
        return cloyne_csv_to_events(df)

    def test_event_count(self, events):
        assert len(events) == 3

    def test_sign_positive_for_positive_value(self, events):
        assert events[0].sign == 1
        assert events[2].sign == 1

    def test_sign_negative_for_negative_value(self, events):
        assert events[1].sign == -1

    def test_magnitude_always_positive(self, events):
        assert all(e.magnitude > 0 for e in events)

    def test_magnitude_equals_abs_of_input(self, events):
        # abs(exog) == magnitude
        assert events[0].magnitude == pytest.approx(0.6)
        assert events[1].magnitude == pytest.approx(0.4)
        assert events[2].magnitude == pytest.approx(1.2)

    def test_signed_magnitude_recovers_original(self, events):
        # sign * magnitude should recover original signed value
        assert events[0].sign * events[0].magnitude == pytest.approx(0.6)
        assert events[1].sign * events[1].magnitude == pytest.approx(-0.4)
        assert events[2].sign * events[2].magnitude == pytest.approx(1.2)

    def test_country_is_gbr(self, events):
        assert all(e.country == "GBR" for e in events)

    def test_magnitude_unit_is_pct_gdp(self, events):
        assert all(e.magnitude_unit == "pct_gdp" for e in events)

    def test_target_is_consumption(self, events):
        assert all(e.target == "consumption" for e in events)

    def test_subtarget_is_tax(self, events):
        assert all(e.subtarget == "tax" for e in events)

    def test_scoring_method_is_manual(self, events):
        assert all(e.scoring_method == "manual" for e in events)

    def test_confidence_is_one(self, events):
        assert all(e.confidence == pytest.approx(1.0) for e in events)

    def test_metadata_replication_key(self, events):
        for e in events:
            assert e.metadata.get("replication") == "cloyne_2013_uk"

    def test_source_url_contains_aeaweb(self, events):
        for e in events:
            assert "aeaweb.org" in e.source_url

    def test_source_text_nonempty(self, events):
        for e in events:
            assert len(e.source_text) > 0

    def test_dates_correct(self, events):
        assert events[0].date == pd.Timestamp("2000-01-01")
        assert events[1].date == pd.Timestamp("2001-04-01")
        assert events[2].date == pd.Timestamp("2003-07-01")


# ---------------------------------------------------------------------------
# cloyne_csv_to_events: multi-row known-value test
# ---------------------------------------------------------------------------

class TestCsvToEventsMultiRow:
    """Larger DataFrame — verify ordering, correct skipping, correct dates."""

    def test_ordering_preserved(self):
        df = pd.DataFrame({
            "date": ["1997Q1", "1998Q2", "1999Q4"],
            "exog": [0.1, -0.2, 0.3],
        })
        events = cloyne_csv_to_events(df)
        assert len(events) == 3
        assert events[0].date < events[1].date < events[2].date

    def test_mixed_valid_and_skipped_rows(self):
        df = pd.DataFrame({
            "date": ["1997Q1", None, "1998Q2", "1999Q4"],
            "exog": [0.1, 0.5, 0.0, 0.3],
        })
        # None date row skipped, zero-value row skipped -> 2 events
        events = cloyne_csv_to_events(df)
        assert len(events) == 2

    def test_all_rows_skipped_returns_empty(self):
        df = pd.DataFrame({
            "date": [np.nan, None, ""],
            "exog": [0.5, 0.5, 0.5],
        })
        events = cloyne_csv_to_events(df)
        assert events == []


# ---------------------------------------------------------------------------
# load() — csv_path branch
# ---------------------------------------------------------------------------

class TestLoadCsvPath:
    """load() with csv_path reads CSV from disk and returns NarrativeInstrument."""

    @pytest.fixture
    def csv_file(self, tmp_path):
        content = (
            "date,exog\n"
            "2000Q1,0.5\n"
            "2001Q2,-0.3\n"
            "2002Q3,0.0\n"   # zero -> skipped
        )
        p = tmp_path / "cloyne_uk.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_returns_narrative_instrument(self, csv_file):
        inst = load(csv_path=csv_file)
        assert isinstance(inst, NarrativeInstrument)

    def test_correct_event_count(self, csv_file):
        inst = load(csv_path=csv_file)
        # Zero row dropped; 2 non-zero events remain
        assert len(inst.events) == 2

    def test_default_target_is_consumption(self, csv_file):
        inst = load(csv_path=csv_file)
        assert inst.target == "consumption"

    def test_custom_target_forwarded(self, csv_file):
        inst = load(csv_path=csv_file, target="investment")
        assert inst.target == "investment"

    def test_quarterly_is_pandas_series(self, csv_file):
        inst = load(csv_path=csv_file)
        assert isinstance(inst.quarterly, pd.Series)

    def test_quarterly_has_nonzero_entries(self, csv_file):
        inst = load(csv_path=csv_file)
        nz = inst.quarterly[inst.quarterly != 0.0]
        assert len(nz) >= 2

    def test_csv_path_as_string_works(self, csv_file):
        inst = load(csv_path=str(csv_file))
        assert isinstance(inst, NarrativeInstrument)

    def test_aggregation_is_sum(self, csv_file):
        inst = load(csv_path=csv_file)
        assert inst.aggregation == "sum"

    def test_events_all_gbr(self, csv_file):
        inst = load(csv_path=csv_file)
        assert all(e.country == "GBR" for e in inst.events)


# ---------------------------------------------------------------------------
# load() — network branch (monkeypatched)
# ---------------------------------------------------------------------------

class TestLoadNetwork:
    """load() without csv_path calls safe_get_bytes; test success + failure."""

    def test_network_success_returns_instrument(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        csv_bytes = b"date,exog\n2000Q1,0.5\n2001Q2,-0.3\n"

        def _fake_fetch(url):
            return csv_bytes

        monkeypatch.setattr(_mod, "safe_get_bytes", _fake_fetch)
        inst = _mod.load()
        assert isinstance(inst, NarrativeInstrument)
        assert len(inst.events) == 2

    def test_network_success_correct_url_called(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        called_urls: list[str] = []

        def _fake_fetch(url):
            called_urls.append(url)
            return b"date,exog\n2000Q1,0.5\n"

        monkeypatch.setattr(_mod, "safe_get_bytes", _fake_fetch)
        _mod.load()
        assert len(called_urls) == 1
        assert "cloyne" in called_urls[0].lower()

    def test_network_failure_raises_runtime_error(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        def _fail(url):
            raise ConnectionError("simulated failure")

        monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
        with pytest.raises(RuntimeError, match="Cloyne 2013"):
            _mod.load()

    def test_runtime_error_chains_original_exception(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        original_exc = OSError("no network")

        def _fail(url):
            raise original_exc

        monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
        with pytest.raises(RuntimeError) as exc_info:
            _mod.load()
        # RuntimeError must chain the original exception via __cause__
        assert exc_info.value.__cause__ is original_exc

    def test_network_success_default_target_consumption(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        monkeypatch.setattr(
            _mod, "safe_get_bytes",
            lambda url: b"date,exog\n2000Q1,0.5\n"
        )
        inst = _mod.load()
        assert inst.target == "consumption"

    def test_network_success_custom_target(self, monkeypatch):
        import puremacro.narrative.replication.cloyne_2013_uk as _mod

        monkeypatch.setattr(
            _mod, "safe_get_bytes",
            lambda url: b"date,exog\n2000Q1,0.5\n"
        )
        inst = _mod.load(target="investment")
        assert inst.target == "investment"


# ---------------------------------------------------------------------------
# NarrativeInstrument structure checks
# ---------------------------------------------------------------------------

class TestNarrativeInstrumentStructure:
    """Structural / property checks on the returned NarrativeInstrument."""

    @pytest.fixture
    def inst(self, tmp_path):
        content = (
            "date,exog\n"
            "2000Q1,0.5\n"
            "2001Q2,-0.3\n"
            "2002Q3,0.8\n"
        )
        p = tmp_path / "cloyne_uk.csv"
        p.write_text(content, encoding="utf-8")
        return load(csv_path=p)

    def test_announcement_series_is_series(self, inst):
        assert isinstance(inst.announcement_series, pd.Series)

    def test_implementation_series_is_series(self, inst):
        assert isinstance(inst.implementation_series, pd.Series)

    def test_quarterly_aliases_implementation_series(self, inst):
        pd.testing.assert_series_equal(inst.quarterly, inst.implementation_series)

    def test_events_list_nonempty(self, inst):
        assert len(inst.events) > 0

    def test_announcement_series_has_date_index(self, inst):
        assert inst.announcement_series.index.name == "date"

    def test_both_series_same_length(self, inst):
        assert len(inst.announcement_series) == len(inst.implementation_series)
