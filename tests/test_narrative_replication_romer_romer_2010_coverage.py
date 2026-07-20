"""Coverage tests for puremacro.narrative.replication.romer_romer_2010.

Targets the lines/branches NOT yet covered by the existing test suite:

* Line 41-44  : ValueError when required columns are absent.
* Line 47-48  : ``_to_quarter`` fast-exit for ``None`` and NaN-float inputs.
* Lines 54-58 : ``_to_quarter`` fallback pd.Timestamp parse path; final
                ``return None`` for completely unparseable strings.
* Line 64     : ``if date is None: continue`` guard.
* Line 67     : ``if v == 0.0 or pd.isna(v): continue`` guard.
* Lines 102-116: ``load()`` — both the csv_path branch and the network
                 branch (monkeypatched both success and failure).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.replication.romer_romer_2010 import (
    romer_romer_2010_csv_to_events,
    load,
)
from puremacro.narrative.types import NarrativeInstrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows, *, cols=None):
    """Build a minimal DataFrame.  ``cols`` defaults to ['date','rr_exog']."""
    if cols is None:
        cols = ["date", "rr_exog"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# romer_romer_2010_csv_to_events: column validation (line 41)
# ---------------------------------------------------------------------------

class TestMissingColumnsRaises:
    """ValueError if the required columns are not present."""

    def test_no_date_col_raises(self):
        df = pd.DataFrame({"quarter_date": ["2003Q1"], "rr_exog": [0.5]})
        with pytest.raises(ValueError, match="date"):
            romer_romer_2010_csv_to_events(df)

    def test_no_value_col_raises(self):
        df = pd.DataFrame({"date": ["2003Q1"], "shock": [0.5]})
        with pytest.raises(ValueError, match="rr_exog"):
            romer_romer_2010_csv_to_events(df)

    def test_empty_df_with_wrong_cols_raises(self):
        df = pd.DataFrame(columns=["foo", "bar"])
        with pytest.raises(ValueError):
            romer_romer_2010_csv_to_events(df)


# ---------------------------------------------------------------------------
# romer_romer_2010_csv_to_events: alternative column names
# ---------------------------------------------------------------------------

class TestAlternativeColumnNames:
    """The parser accepts several synonyms for both the date and value cols."""

    def test_quarter_col_accepted(self):
        df = pd.DataFrame({"quarter": ["2003Q1", "2003Q2"], "exog_tax": [0.3, -0.2]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 2

    def test_y_col_accepted(self):
        df = pd.DataFrame({"date": ["2003Q1"], "y": [1.5]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(1.5)

    def test_tax_shock_col_accepted(self):
        df = pd.DataFrame({"date": ["2003Q1"], "tax_shock": [-0.8]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].sign == -1

    def test_effective_quarter_col_accepted(self):
        df = pd.DataFrame({
            "date": ["2003Q2"],
            "rr_exog": [0.5],
            "effective_quarter": ["2003Q3"],
        })
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].implementation_profile == [(pd.Timestamp("2003-07-01"), 1.0)]


# ---------------------------------------------------------------------------
# _to_quarter: None and NaN-float fast-exit (line 47-48)
# ---------------------------------------------------------------------------

class TestToQuarterNoneNaN:
    """Rows whose date parses to None are skipped silently."""

    def test_none_date_row_skipped(self):
        # pandas stores None in an object column as None.
        df = pd.DataFrame({"date": [None, "2003Q1"], "rr_exog": [0.5, 0.3]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2003-01-01")

    def test_nan_float_date_row_skipped(self):
        # np.nan in a numeric context — pandas 2 stores mixed nan+str as
        # object; pandas 3 infers the string dtype (nan as missing value).
        df = pd.DataFrame({"date": [np.nan, "2003Q1"], "rr_exog": [0.5, 0.3]})
        assert (
            df["date"].dtype == object
            or df["date"].dtype == float
            or pd.api.types.is_string_dtype(df["date"])
        )
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1

    def test_all_nan_dates_returns_empty(self):
        df = pd.DataFrame({"date": [np.nan], "rr_exog": [0.5]})
        events = romer_romer_2010_csv_to_events(df)
        assert events == []

    def test_whitespace_only_date_row_skipped(self):
        # str.strip() yields '' — the empty-string guard fires (line 51).
        df = pd.DataFrame({"date": ["   ", "2003Q1"], "rr_exog": [0.5, 0.3]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2003-01-01")


# ---------------------------------------------------------------------------
# _to_quarter: fallback pd.Timestamp parse path (lines 54-58)
# ---------------------------------------------------------------------------

class TestToQuarterFallbackParse:
    """Date strings that fail pd.Period() are retried via pd.Timestamp."""

    def test_iso_date_string_parsed_via_fallback(self):
        # '1981-07-01' fails the pd.Period("1981-07-01", freq="Q") path but
        # succeeds via pd.Timestamp.
        df = pd.DataFrame({"date": ["1981-07-01"], "rr_exog": [0.6]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        # Quarter-start of 1981Q3.
        assert events[0].date == pd.Timestamp("1981-07-01")

    def test_timestamp_parsed_to_quarter_start(self):
        # '2003-05-15' is mid-Q2 2003; falls back to pd.Timestamp, then
        # wraps to Q2 start.
        df = pd.DataFrame({"date": ["2003-05-15"], "rr_exog": [0.4]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2003-04-01")

    def test_completely_unparseable_date_row_skipped(self):
        # Both parse attempts fail → return None → row skipped.
        df = pd.DataFrame({"date": ["xyz_bogus", "2003Q1"], "rr_exog": [0.6, 0.3]})
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].date == pd.Timestamp("2003-01-01")


# ---------------------------------------------------------------------------
# Row-level filters: zero and NaN values (line 67)
# ---------------------------------------------------------------------------

class TestValueFilters:
    """Rows with v == 0.0 or v == NaN are dropped."""

    def test_zero_value_row_skipped(self):
        df = pd.DataFrame({
            "date": ["2003Q1", "2003Q2"],
            "rr_exog": [0.0, 0.5],
        })
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1
        assert events[0].magnitude == pytest.approx(0.5)

    def test_nan_value_row_skipped(self):
        df = pd.DataFrame({
            "date": ["2003Q1", "2003Q2"],
            "rr_exog": [np.nan, 0.5],
        })
        events = romer_romer_2010_csv_to_events(df)
        assert len(events) == 1

    def test_all_zero_values_returns_empty(self):
        df = pd.DataFrame({"date": ["2003Q1", "2003Q2"], "rr_exog": [0.0, 0.0]})
        events = romer_romer_2010_csv_to_events(df)
        assert events == []


# ---------------------------------------------------------------------------
# Event properties: sign, magnitude contract, metadata
# ---------------------------------------------------------------------------

class TestEventProperties:
    """Known-value tests for the fields produced by romer_romer_2010_csv_to_events."""

    def _make_events(self):
        df = pd.DataFrame({
            "date": ["1981Q3", "1986Q1", "2003Q2"],
            "rr_exog": [0.6, -0.4, 1.2],
        })
        return romer_romer_2010_csv_to_events(df)

    def test_sign_positive_for_tax_cut(self):
        events = self._make_events()
        # Positive rr_exog → expansionary → sign == +1.
        assert events[0].sign == 1
        assert events[2].sign == 1

    def test_sign_negative_for_tax_hike(self):
        events = self._make_events()
        assert events[1].sign == -1

    def test_magnitude_always_positive(self):
        events = self._make_events()
        assert all(e.magnitude > 0 for e in events)

    def test_signed_magnitude_recovers_original_value(self):
        events = self._make_events()
        # signed_magnitude == sign * magnitude == original rr_exog value.
        assert events[0].signed_magnitude == pytest.approx(0.6)
        assert events[1].signed_magnitude == pytest.approx(-0.4)
        assert events[2].signed_magnitude == pytest.approx(1.2)

    def test_country_always_usa(self):
        events = self._make_events()
        assert all(e.country == "USA" for e in events)

    def test_magnitude_unit_pct_gdp(self):
        events = self._make_events()
        assert all(e.magnitude_unit == "pct_gdp" for e in events)

    def test_target_is_consumption(self):
        events = self._make_events()
        assert all(e.target == "consumption" for e in events)

    def test_subtarget_is_tax(self):
        events = self._make_events()
        assert all(e.subtarget == "tax" for e in events)

    def test_metadata_contains_replication_key(self):
        events = self._make_events()
        for e in events:
            assert e.metadata.get("replication") == "romer_romer_2010"

    def test_scoring_method_is_manual(self):
        events = self._make_events()
        assert all(e.scoring_method == "manual" for e in events)

    def test_confidence_is_one(self):
        events = self._make_events()
        assert all(e.confidence == pytest.approx(1.0) for e in events)


# ---------------------------------------------------------------------------
# Implementation profile: effective_date < announcement date is excluded
# ---------------------------------------------------------------------------

class TestEffectiveDateFiltering:
    """Effective dates that predate the announcement quarter are not added to
    the profile (the guard ``eff_q >= date`` on line 72)."""

    def test_effective_date_before_announcement_excluded(self):
        df = pd.DataFrame({
            "date": ["2003Q2"],
            "rr_exog": [0.5],
            "effective_date": ["2003Q1"],   # earlier than 2003Q2
        })
        events = romer_romer_2010_csv_to_events(df)
        # Profile should be empty — the effective date precedes announcement.
        assert events[0].implementation_profile == []

    def test_effective_date_equal_to_announcement_included(self):
        df = pd.DataFrame({
            "date": ["2003Q2"],
            "rr_exog": [0.5],
            "effective_date": ["2003Q2"],
        })
        events = romer_romer_2010_csv_to_events(df)
        assert events[0].implementation_profile == [(pd.Timestamp("2003-04-01"), 1.0)]

    def test_effective_date_after_announcement_included(self):
        df = pd.DataFrame({
            "date": ["2003Q2"],
            "rr_exog": [0.5],
            "effective_date": ["2003Q3"],
        })
        events = romer_romer_2010_csv_to_events(df)
        assert events[0].implementation_profile == [(pd.Timestamp("2003-07-01"), 1.0)]


# ---------------------------------------------------------------------------
# load() — csv_path branch (lines 102-103, 115-116)
# ---------------------------------------------------------------------------

class TestLoadCsvPath:
    """load() with csv_path reads the CSV from disk and returns a
    NarrativeInstrument."""

    @pytest.fixture
    def csv_file(self, tmp_path):
        content = (
            "date,rr_exog\n"
            "1981Q3,0.6\n"
            "1986Q1,-0.4\n"
            "2003Q2,0.0\n"   # zero → skipped
        )
        p = tmp_path / "rr2010.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_returns_narrative_instrument(self, csv_file):
        inst = load(csv_path=csv_file)
        assert isinstance(inst, NarrativeInstrument)

    def test_correct_number_of_events(self, csv_file):
        inst = load(csv_path=csv_file)
        # zero row dropped, 2 non-zero rows.
        assert len(inst.events) == 2

    def test_default_target_is_consumption(self, csv_file):
        inst = load(csv_path=csv_file)
        assert inst.target == "consumption"

    def test_custom_target_forwarded(self, csv_file):
        inst = load(csv_path=csv_file, target="investment")
        assert inst.target == "investment"

    def test_quarterly_series_is_pandas_series(self, csv_file):
        inst = load(csv_path=csv_file)
        assert isinstance(inst.quarterly, pd.Series)

    def test_quarterly_series_has_non_zero_entries(self, csv_file):
        inst = load(csv_path=csv_file)
        # Both events should contribute non-zero values.
        nz = inst.quarterly[inst.quarterly != 0.0]
        assert len(nz) >= 2

    def test_csv_path_as_string_works(self, csv_file):
        inst = load(csv_path=str(csv_file))
        assert isinstance(inst, NarrativeInstrument)


# ---------------------------------------------------------------------------
# load() — url= fetch branch
# ---------------------------------------------------------------------------

_URL = "https://example.invalid/romer_romer_2010.csv"


class TestLoadNetwork:
    """load(url=...) calls safe_get_bytes; monkeypatch both success and
    failure paths."""

    def test_network_failure_raises_runtime_error(self, monkeypatch):
        import puremacro.narrative.replication.romer_romer_2010 as _mod

        def _fail(url):
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
        with pytest.raises(RuntimeError, match="Romer-Romer 2010"):
            _mod.load(url=_URL)

    def test_network_success_returns_instrument(self, monkeypatch):
        import puremacro.narrative.replication.romer_romer_2010 as _mod

        csv_bytes = b"date,rr_exog\n1981Q3,0.6\n2003Q2,0.5\n"

        def _fake_fetch(url):
            assert url == _URL
            return csv_bytes

        monkeypatch.setattr(_mod, "safe_get_bytes", _fake_fetch)
        inst = _mod.load(url=_URL)
        assert isinstance(inst, NarrativeInstrument)
        assert len(inst.events) == 2

    def test_runtime_error_chains_original_exception(self, monkeypatch):
        import puremacro.narrative.replication.romer_romer_2010 as _mod

        original_exc = OSError("disk error")

        def _fail(url):
            raise original_exc

        monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
        with pytest.raises(RuntimeError) as exc_info:
            _mod.load(url=_URL)
        # The RuntimeError should chain the original OSError via __cause__.
        assert exc_info.value.__cause__ is original_exc

    def test_error_message_points_at_offline_fallback(self, monkeypatch):
        """The diagnostic must tell the user load() works with no args."""
        import puremacro.narrative.replication.romer_romer_2010 as _mod

        def _fail(url):
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
        with pytest.raises(RuntimeError, match="shipped snapshot"):
            _mod.load(url=_URL)


# ---------------------------------------------------------------------------
# load() — default = shipped snapshot, fully offline
# ---------------------------------------------------------------------------

class TestLoadDefaultSnapshot:
    """load() with no arguments reads the frozen package-data snapshot
    (puremacro/replication/data/tax14_narrative_tax_shocks.csv) and never
    touches the network."""

    def test_no_args_returns_instrument_offline(self, monkeypatch):
        import puremacro.narrative.replication.romer_romer_2010 as _mod

        def _no_network(url):
            raise AssertionError("default load() must not hit the network")

        monkeypatch.setattr(_mod, "safe_get_bytes", _no_network)
        inst = _mod.load()
        assert isinstance(inst, NarrativeInstrument)

    def test_snapshot_event_count_matches_nonzero_rows(self):
        from puremacro.replication._data import load_csv

        snap = load_csv("tax14_narrative_tax_shocks")
        expected = int((snap["rr_exog"] != 0.0).sum())
        inst = load()
        assert len(inst.events) == expected
        assert expected > 40  # RR's narrative record has dozens of episodes

    def test_snapshot_dates_span_1945_2007(self):
        inst = load()
        years = [e.date.year for e in inst.events]
        assert min(years) >= 1945
        assert max(years) <= 2007

    def test_snapshot_events_are_pct_gdp_usa(self):
        inst = load()
        assert all(e.magnitude_unit == "pct_gdp" for e in inst.events)
        assert all(e.country == "USA" for e in inst.events)

    def test_csv_path_still_overrides_snapshot(self, tmp_path):
        p = tmp_path / "override.csv"
        p.write_text("date,rr_exog\n1999Q1,0.9\n", encoding="utf-8")
        inst = load(csv_path=p)
        assert len(inst.events) == 1
        assert inst.events[0].date == pd.Timestamp("1999-01-01")
