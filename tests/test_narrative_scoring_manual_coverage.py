"""Tests for puremacro.narrative.scoring.manual.

Strategy:
- Known-value: build small synthetic CSVs with known content; assert that
  load_manual_csv returns the correct NarrativeEvent objects with exact field
  values that are derivable from the CSV content alone.
- Properties: output list length, field types, confidence clamping via
  NarrativeEvent.__post_init__, magnitude always non-negative, sign in {-1,0,1}.
- validate_events contracts: keys present in returned dict, counts match,
  by_target and by_sign aggregate correctly, n_duplicates logic, bad targets
  and bad signs appear in issues, empty-events fast path.
- REQUIRED_COLS public constant: ensure it is a tuple/sequence of the expected
  column names.
- Error handling: missing required columns raise ValueError; CSV with extra
  optional columns round-trips cleanly.
- No network I/O: all tests use in-memory CSV strings or tmp_path fixtures.
"""
from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from puremacro.narrative.scoring.manual import (
    REQUIRED_COLS,
    load_manual_csv,
    validate_events,
)
from puremacro.narrative.types import NarrativeEvent, VALID_SIGNS, VALID_TARGETS


# ---------------------------------------------------------------------------
# Helper: build a minimal valid CSV as a string
# ---------------------------------------------------------------------------

MINIMAL_ROW = dict(
    date="2020-01-01",
    country="USA",
    magnitude="10.0",
    magnitude_unit="USD_bn",
    target="investment",
    sign="1",
    confidence="0.9",
    source_url="https://example.com/doc",
)


def _csv_from_rows(rows: list[dict]) -> str:
    """Convert a list of dicts (all sharing the same keys) to a CSV string."""
    if not rows:
        return ",".join(REQUIRED_COLS)
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines)


def _write_csv(tmp_path, rows: list[dict], fname: str = "events.csv") -> str:
    p = tmp_path / fname
    p.write_text(_csv_from_rows(rows), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Tests for REQUIRED_COLS
# ---------------------------------------------------------------------------

class TestRequiredCols:
    def test_required_cols_is_sequence(self):
        assert len(REQUIRED_COLS) > 0

    def test_required_cols_contains_essential_fields(self):
        for col in ("date", "country", "magnitude", "sign", "confidence",
                    "target", "magnitude_unit", "source_url"):
            assert col in REQUIRED_COLS

    def test_required_cols_immutable_or_tuple(self):
        # Should be a tuple (as defined in source); just ensure it is not a set
        # so that ordering is stable.
        assert not isinstance(REQUIRED_COLS, set)


# ---------------------------------------------------------------------------
# Tests for load_manual_csv — happy path
# ---------------------------------------------------------------------------

class TestLoadManualCsvHappyPath:
    def test_single_row_returns_one_event(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        events = load_manual_csv(path)
        assert len(events) == 1

    def test_returns_list_of_narrative_events(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        events = load_manual_csv(path)
        assert all(isinstance(e, NarrativeEvent) for e in events)

    def test_date_parsed_as_timestamp(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert isinstance(e.date, pd.Timestamp)
        assert e.date == pd.Timestamp("2020-01-01")

    def test_country_is_string(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert isinstance(e.country, str)
        assert e.country == "USA"

    def test_magnitude_is_float_and_abs(self, tmp_path):
        """NarrativeEvent.__post_init__ converts magnitude to abs(float)."""
        row = dict(MINIMAL_ROW, magnitude="-5.0")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert isinstance(e.magnitude, float)
        assert e.magnitude == 5.0  # magnitude is always non-negative

    def test_magnitude_positive_preserved(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert e.magnitude == 10.0

    def test_sign_is_int(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert isinstance(e.sign, int)
        assert e.sign == 1

    def test_sign_negative(self, tmp_path):
        row = dict(MINIMAL_ROW, sign="-1")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.sign == -1

    def test_sign_zero(self, tmp_path):
        row = dict(MINIMAL_ROW, sign="0")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.sign == 0

    def test_confidence_is_float(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert isinstance(e.confidence, float)
        assert e.confidence == pytest.approx(0.9)

    def test_confidence_clamped_above_1(self, tmp_path):
        """NarrativeEvent clamps confidence to [0,1]."""
        row = dict(MINIMAL_ROW, confidence="1.5")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.confidence == pytest.approx(1.0)

    def test_confidence_clamped_below_0(self, tmp_path):
        row = dict(MINIMAL_ROW, confidence="-0.5")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.confidence == pytest.approx(0.0)

    def test_source_url_preserved(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert e.source_url == "https://example.com/doc"

    def test_scoring_method_defaults_to_manual(self, tmp_path):
        """When scoring_method column is absent, default is 'manual'."""
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert e.scoring_method == "manual"

    def test_scoring_method_column_respected(self, tmp_path):
        row = dict(MINIMAL_ROW, scoring_method="keyword")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.scoring_method == "keyword"

    def test_multiple_rows_returns_correct_length(self, tmp_path):
        rows = [
            dict(MINIMAL_ROW, date="2020-01-01", country="USA"),
            dict(MINIMAL_ROW, date="2020-04-01", country="GBR"),
            dict(MINIMAL_ROW, date="2020-07-01", country="DEU"),
        ]
        path = _write_csv(tmp_path, rows)
        events = load_manual_csv(path)
        assert len(events) == 3

    def test_multiple_rows_countries_preserved(self, tmp_path):
        rows = [
            dict(MINIMAL_ROW, country="USA"),
            dict(MINIMAL_ROW, country="GBR"),
        ]
        path = _write_csv(tmp_path, rows)
        events = load_manual_csv(path)
        countries = [e.country for e in events]
        assert "USA" in countries
        assert "GBR" in countries

    def test_optional_subtarget_column_honoured(self, tmp_path):
        """When subtarget column is present it should be read."""
        row = dict(MINIMAL_ROW, subtarget="infra")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.subtarget == "infra"

    def test_subtarget_absent_yields_none_or_nan(self, tmp_path):
        """When subtarget column is absent, subtarget is None."""
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        # NarrativeEvent stores subtarget as whatever was provided; None is fine.
        assert e.subtarget is None

    def test_optional_source_text_column_honoured(self, tmp_path):
        row = dict(MINIMAL_ROW, source_text="Congress approved a stimulus package.")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert "stimulus" in e.source_text

    def test_magnitude_unit_preserved(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert e.magnitude_unit == "USD_bn"

    def test_target_investment_preserved(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        e = load_manual_csv(path)[0]
        assert e.target == "investment"

    def test_target_consumption(self, tmp_path):
        row = dict(MINIMAL_ROW, target="consumption")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.target == "consumption"

    def test_target_both(self, tmp_path):
        row = dict(MINIMAL_ROW, target="both")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.target == "both"

    def test_accepts_path_object(self, tmp_path):
        """load_manual_csv must accept pathlib.Path, not just str."""
        from pathlib import Path
        p = tmp_path / "events.csv"
        p.write_text(_csv_from_rows([MINIMAL_ROW]), encoding="utf-8")
        events = load_manual_csv(p)
        assert len(events) == 1

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """A CSV with only headers and no data rows returns an empty list."""
        p = tmp_path / "empty.csv"
        p.write_text(",".join(REQUIRED_COLS), encoding="utf-8")
        events = load_manual_csv(p)
        assert events == []


# ---------------------------------------------------------------------------
# Tests for load_manual_csv — error handling
# ---------------------------------------------------------------------------

class TestLoadManualCsvErrors:
    def test_missing_date_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "date"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="date"):
            load_manual_csv(p)

    def test_missing_country_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "country"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="country"):
            load_manual_csv(p)

    def test_missing_magnitude_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "magnitude"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="magnitude"):
            load_manual_csv(p)

    def test_missing_sign_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "sign"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="sign"):
            load_manual_csv(p)

    def test_missing_confidence_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "confidence"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="confidence"):
            load_manual_csv(p)

    def test_missing_target_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "target"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="target"):
            load_manual_csv(p)

    def test_missing_source_url_column_raises(self, tmp_path):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "source_url"}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError, match="source_url"):
            load_manual_csv(p)

    def test_missing_multiple_columns_raises_with_list(self, tmp_path):
        """Error message should mention missing columns (not just one)."""
        row = {k: v for k, v in MINIMAL_ROW.items()
               if k not in ("date", "country")}
        p = tmp_path / "bad.csv"
        p.write_text(_csv_from_rows([row]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_manual_csv(p)

    def test_extra_columns_do_not_raise(self, tmp_path):
        """Extra non-required columns must not cause an error."""
        row = dict(MINIMAL_ROW, extra_col="irrelevant", another="42")
        path = _write_csv(tmp_path, [row])
        events = load_manual_csv(path)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Tests for validate_events
# ---------------------------------------------------------------------------

def _make_event(**kwargs) -> NarrativeEvent:
    defaults = dict(
        date=pd.Timestamp("2020-01-01"),
        country="USA",
        magnitude=10.0,
        magnitude_unit="USD_bn",
        target="investment",
        subtarget=None,
        sign=1,
        confidence=0.9,
        source_text="",
        source_url="https://example.com",
        scoring_method="manual",
    )
    defaults.update(kwargs)
    return NarrativeEvent(**defaults)


class TestValidateEvents:
    def test_empty_list_returns_n_zero(self):
        out = validate_events([])
        assert out == {"n": 0}

    def test_single_event_n_equals_one(self):
        events = [_make_event()]
        out = validate_events(events)
        assert out["n"] == 1

    def test_returns_dict_with_required_keys(self):
        events = [_make_event()]
        out = validate_events(events)
        for key in ("n", "date_min", "date_max", "by_target",
                    "by_sign", "by_country", "by_method",
                    "n_duplicates", "issues"):
            assert key in out, f"missing key: {key}"

    def test_date_min_max_correct(self):
        events = [
            _make_event(date=pd.Timestamp("2019-01-01")),
            _make_event(date=pd.Timestamp("2021-06-01")),
        ]
        out = validate_events(events)
        assert "2019" in out["date_min"]
        assert "2021" in out["date_max"]

    def test_by_target_counts_correctly(self):
        events = [
            _make_event(target="investment"),
            _make_event(target="investment"),
            _make_event(target="consumption"),
        ]
        out = validate_events(events)
        assert out["by_target"]["investment"] == 2
        assert out["by_target"]["consumption"] == 1

    def test_by_sign_counts_correctly(self):
        events = [
            _make_event(sign=1),
            _make_event(sign=-1),
            _make_event(sign=-1),
        ]
        out = validate_events(events)
        assert out["by_sign"][1] == 1
        assert out["by_sign"][-1] == 2

    def test_by_country_counts_correctly(self):
        events = [
            _make_event(country="USA"),
            _make_event(country="USA"),
            _make_event(country="GBR"),
        ]
        out = validate_events(events)
        assert out["by_country"]["USA"] == 2
        assert out["by_country"]["GBR"] == 1

    def test_by_method_counts_correctly(self):
        events = [
            _make_event(scoring_method="manual"),
            _make_event(scoring_method="keyword"),
            _make_event(scoring_method="manual"),
        ]
        out = validate_events(events)
        assert out["by_method"]["manual"] == 2
        assert out["by_method"]["keyword"] == 1

    def test_no_duplicates_when_all_distinct(self):
        events = [
            _make_event(date=pd.Timestamp("2020-01-01"), country="USA",
                        target="investment", magnitude=10.0),
            _make_event(date=pd.Timestamp("2020-04-01"), country="USA",
                        target="investment", magnitude=10.0),
        ]
        out = validate_events(events)
        assert out["n_duplicates"] == 0

    def test_duplicate_detected(self):
        """Two identical events should produce n_duplicates >= 1."""
        e = _make_event(date=pd.Timestamp("2020-01-01"), country="USA",
                        target="investment", magnitude=10.0)
        out = validate_events([e, e])
        assert out["n_duplicates"] >= 1

    def test_no_issues_for_valid_events(self):
        events = [_make_event(target="investment", sign=1)]
        out = validate_events(events)
        assert out["issues"] == []

    def test_issues_is_list(self):
        events = [_make_event()]
        out = validate_events(events)
        assert isinstance(out["issues"], list)

    def test_n_duplicates_is_int(self):
        events = [_make_event()]
        out = validate_events(events)
        assert isinstance(out["n_duplicates"], int)

    def test_multiple_events_same_country_different_dates(self):
        events = [
            _make_event(date=pd.Timestamp(f"202{i}-01-01"), country="FRA")
            for i in range(4)
        ]
        out = validate_events(events)
        assert out["n"] == 4
        assert out["by_country"]["FRA"] == 4

    def test_sign_zero_counted(self):
        events = [_make_event(sign=0)]
        out = validate_events(events)
        assert out["by_sign"].get(0, 0) == 1

    def test_confidence_bounds_respected_in_events(self):
        """validate_events runs on already-constructed events whose confidence
        is clamped; this confirms the upstream clamp is visible here."""
        # confidence=2.0 is clamped to 1.0 by NarrativeEvent.__post_init__
        e = _make_event(confidence=2.0)
        assert 0.0 <= e.confidence <= 1.0

    def test_magnitude_always_nonnegative_in_events(self):
        e = _make_event(magnitude=-3.0)
        assert e.magnitude >= 0.0

    def test_validate_three_events_distinct_targets(self):
        events = [
            _make_event(target="investment"),
            _make_event(target="consumption"),
            _make_event(target="both"),
        ]
        out = validate_events(events)
        assert len(out["by_target"]) == 3
        assert out["n"] == 3


# ---------------------------------------------------------------------------
# Round-trip: CSV -> load -> validate
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_load_then_validate_consistent_n(self, tmp_path):
        rows = [dict(MINIMAL_ROW, date=f"202{i}-01-01") for i in range(3)]
        path = _write_csv(tmp_path, rows)
        events = load_manual_csv(path)
        info = validate_events(events)
        assert info["n"] == 3

    def test_load_then_validate_no_issues_for_clean_csv(self, tmp_path):
        path = _write_csv(tmp_path, [MINIMAL_ROW])
        events = load_manual_csv(path)
        info = validate_events(events)
        assert info["issues"] == []

    def test_load_signed_magnitudes(self, tmp_path):
        """signed_magnitude = magnitude * sign must be recoverable post-load."""
        row_pos = dict(MINIMAL_ROW, magnitude="20.0", sign="1")
        row_neg = dict(MINIMAL_ROW, magnitude="20.0", sign="-1")
        path = _write_csv(tmp_path, [row_pos, row_neg])
        events = load_manual_csv(path)
        assert events[0].signed_magnitude == pytest.approx(20.0)
        assert events[1].signed_magnitude == pytest.approx(-20.0)

    def test_pct_gdp_magnitude_unit_round_trips(self, tmp_path):
        row = dict(MINIMAL_ROW, magnitude_unit="pct_gdp", magnitude="1.5")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.magnitude_unit == "pct_gdp"
        assert e.magnitude == pytest.approx(1.5)

    def test_scoring_method_keyword_round_trips(self, tmp_path):
        row = dict(MINIMAL_ROW, scoring_method="keyword")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.scoring_method == "keyword"

    def test_scoring_method_llm_round_trips(self, tmp_path):
        row = dict(MINIMAL_ROW, scoring_method="llm")
        path = _write_csv(tmp_path, [row])
        e = load_manual_csv(path)[0]
        assert e.scoring_method == "llm"

    def test_different_dates_preserve_order_independence(self, tmp_path):
        """Events are returned in CSV row order (no implicit sorting)."""
        rows = [
            dict(MINIMAL_ROW, date="2021-07-01", country="CAN"),
            dict(MINIMAL_ROW, date="2019-01-01", country="AUS"),
        ]
        path = _write_csv(tmp_path, rows)
        events = load_manual_csv(path)
        assert events[0].country == "CAN"
        assert events[1].country == "AUS"


# ---------------------------------------------------------------------------
# Tests for validate_events — issues branch (bad targets / bad signs)
# These branches are only reachable by bypassing NarrativeEvent.__post_init__
# via object.__setattr__, since the dataclass normally enforces validity.
# ---------------------------------------------------------------------------

class TestValidateEventsIssues:
    """Cover the 'unknown targets' and 'unknown signs' issue branches."""

    def _event_with_bad_target(self, target_value: str) -> NarrativeEvent:
        e = _make_event()
        object.__setattr__(e, "target", target_value)
        return e

    def _event_with_bad_sign(self, sign_value: int) -> NarrativeEvent:
        e = _make_event()
        object.__setattr__(e, "sign", sign_value)
        return e

    def test_bad_target_appears_in_issues(self):
        e = self._event_with_bad_target("bad_target")
        out = validate_events([e])
        assert any("unknown targets" in issue for issue in out["issues"])

    def test_bad_sign_appears_in_issues(self):
        e = self._event_with_bad_sign(99)
        out = validate_events([e])
        assert any("unknown signs" in issue for issue in out["issues"])

    def test_bad_target_issues_contains_the_bad_value(self):
        e = self._event_with_bad_target("totally_wrong")
        out = validate_events([e])
        issues_str = " ".join(out["issues"])
        assert "totally_wrong" in issues_str

    def test_both_bad_target_and_bad_sign_each_produce_issue(self):
        e = self._event_with_bad_target("bad_tgt")
        object.__setattr__(e, "sign", 42)
        out = validate_events([e])
        assert len(out["issues"]) == 2

    def test_mixed_good_and_bad_targets(self):
        good = _make_event(target="investment")
        bad = self._event_with_bad_target("unknown_sector")
        out = validate_events([good, bad])
        assert any("unknown targets" in issue for issue in out["issues"])
