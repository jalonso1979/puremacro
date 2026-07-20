"""Coverage tests for puremacro.narrative.replication.mertens_ravn_2013.

The existing test (test_narrative_offline.py::test_mertens_ravn_csv_to_events_minimal_csv)
covers only the basic happy-path: 2-row CSV with mtr_u/mtr_a/mtr_total columns for all
three kinds. This file covers every untested branch and edge case:

  - Column-name aliases (mr_u, unanticipated; mr_a, anticipated; mr_total, total; quarter)
  - Zero/NaN value rows are skipped
  - Sign extraction: positive v -> sign=+1, negative v -> sign=-1; magnitude always abs(v)
  - Anticipated + impl_q column: profile set when impl_q >= announce date
  - Anticipated + impl_q column: no profile when impl_q < announce date
  - Anticipated + no impl_col: empty profile (default)
  - implementation_quarter alias for impl_q column
  - _to_quarter fallback: Timestamp string that isn't a Quarter literal
  - _to_quarter returns None for empty string, None, float NaN
  - _to_quarter invalid string -> None -> no profile
  - date parsing: Timestamp fallback branch
  - date parsing: unparseable date row -> skip (continue)
  - ValueError when date or value column missing
  - load() with csv_path: returns NarrativeInstrument
  - load() network download path: success via monkeypatched safe_get_bytes
  - load() network download failure: RuntimeError raised
  - load() kind='anticipated' and kind='all'
  - load() custom target kwarg passed through
  - NarrativeEvent contract: country=USA, magnitude_unit=pct_gdp, subtarget=tax,
    source_url, scoring_method, metadata fields
  - NarrativeInstrument.quarterly is a pd.Series with nonzero entries
  - delay_quarters is correct for anticipated with impl_q profile
  - source_text contains the kind word
"""
from __future__ import annotations

import io
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from puremacro.narrative.replication.mertens_ravn_2013 import (
    mertens_ravn_csv_to_events,
    load,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df(**kwargs) -> pd.DataFrame:
    """Build a one-row MR-style DataFrame. date defaults to '1980Q1'."""
    base = {"date": ["1980Q1"]}
    base.update({k: [v] for k, v in kwargs.items()})
    return pd.DataFrame(base)


def _csv_bytes(rows_csv: str) -> bytes:
    """Return UTF-8 bytes of a CSV string prefixed by header."""
    return rows_csv.encode("utf-8")


# ---------------------------------------------------------------------------
# Basic sign + magnitude extraction
# ---------------------------------------------------------------------------

def test_positive_value_sign_plus_one():
    """Positive mtr_u -> sign=+1, magnitude = abs(v)."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    e = events[0]
    assert e.sign == 1
    assert e.magnitude == pytest.approx(0.4)


def test_negative_value_sign_minus_one():
    """Negative mtr_u -> sign=-1, magnitude = abs(v) (positive number)."""
    df = _simple_df(mtr_u=-0.7)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    e = events[0]
    assert e.sign == -1
    assert e.magnitude == pytest.approx(0.7)


def test_magnitude_is_always_non_negative():
    """magnitude is always non-negative regardless of the value sign."""
    for v in [0.3, -0.3, 1.5, -1.5]:
        df = _simple_df(mtr_u=v)
        events = mertens_ravn_csv_to_events(df, kind="unanticipated")
        assert events[0].magnitude >= 0.0, f"magnitude negative for v={v}"


def test_zero_value_skipped():
    """v == 0.0 is skipped; should produce no events."""
    df = _simple_df(mtr_u=0.0)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events == []


def test_nan_value_skipped():
    """NaN values are skipped."""
    import numpy as np
    df = pd.DataFrame({"date": ["1980Q1", "1981Q1"], "mtr_u": [float("nan"), 0.5]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NarrativeEvent contract fields
# ---------------------------------------------------------------------------

def test_event_contract_country_usa():
    """All events from this loader have country='USA'."""
    df = pd.DataFrame({"date": ["1980Q1", "1985Q2"], "mtr_u": [0.4, -0.2]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert all(e.country == "USA" for e in events)


def test_event_contract_magnitude_unit():
    """magnitude_unit is 'pct_gdp'."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].magnitude_unit == "pct_gdp"


def test_event_contract_subtarget_tax():
    """subtarget is 'tax' for all events."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].subtarget == "tax"


def test_event_contract_scoring_method_manual():
    """scoring_method is 'manual'."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].scoring_method == "manual"


def test_event_contract_source_url():
    """source_url is karelmertens.com."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert "karelmertens.com" in events[0].source_url


def test_event_contract_metadata_replication_key():
    """metadata contains replication='mertens_ravn_2013' and kind."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    md = events[0].metadata
    assert md.get("replication") == "mertens_ravn_2013"
    assert md.get("kind") == "unanticipated"


def test_event_contract_source_text_contains_kind():
    """source_text mentions the kind string."""
    for kind, col in [("unanticipated", "mtr_u"), ("anticipated", "mtr_a"),
                      ("all", "mtr_total")]:
        df = pd.DataFrame({"date": ["1980Q1"], col: [0.4]})
        events = mertens_ravn_csv_to_events(df, kind=kind)
        assert kind in events[0].source_text, f"kind={kind!r} not in source_text"


def test_event_confidence_is_1():
    """confidence is always 1.0."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Date parsing paths
# ---------------------------------------------------------------------------

def test_date_parses_quarter_string():
    """'1980Q1' -> pd.Timestamp('1980-01-01')."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].date == pd.Timestamp("1980-01-01")


def test_date_parses_timestamp_string_fallback():
    """Date given as '1984-04-15' (Timestamp-parseable, not Quarter literal)
    falls back to pd.Timestamp and then snaps to quarter start."""
    df = pd.DataFrame({"date": ["1984-04-15"], "mtr_u": [0.5]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    # 1984-04-15 is in Q2, so quarter-start = 1984-04-01
    assert events[0].date == pd.Timestamp("1984-04-01")


def test_unparseable_date_row_is_skipped():
    """A completely unparseable date string causes that row to be silently skipped."""
    df = pd.DataFrame({"date": ["NOT-A-DATE", "1981Q3"], "mtr_u": [0.4, 0.2]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    # The first row is dropped; only 1981Q3 survives.
    assert len(events) == 1
    assert events[0].date == pd.Timestamp("1981-07-01")


def test_quarter_column_instead_of_date():
    """Column named 'quarter' is accepted as the date column."""
    df = pd.DataFrame({"quarter": ["1982Q2"], "mtr_u": [0.6]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    assert events[0].date == pd.Timestamp("1982-04-01")


# ---------------------------------------------------------------------------
# Column-name aliases
# ---------------------------------------------------------------------------

def test_mr_u_column_alias():
    """'mr_u' column accepted for kind='unanticipated'."""
    df = pd.DataFrame({"date": ["1980Q1"], "mr_u": [0.4]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.4)


def test_unanticipated_column_alias():
    """'unanticipated' column accepted for kind='unanticipated'."""
    df = pd.DataFrame({"date": ["1982Q3"], "unanticipated": [0.7]})
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.7)


def test_mr_a_column_alias():
    """'mr_a' column accepted for kind='anticipated'."""
    df = pd.DataFrame({"date": ["1980Q1"], "mr_a": [0.3]})
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.3)


def test_anticipated_column_alias():
    """'anticipated' column accepted for kind='anticipated'."""
    df = pd.DataFrame({"date": ["1982Q3"], "anticipated": [0.5]})
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.5)


def test_mr_total_column_alias():
    """'mr_total' column accepted for kind='all'."""
    df = pd.DataFrame({"date": ["1980Q1"], "mr_total": [0.5]})
    events = mertens_ravn_csv_to_events(df, kind="all")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(0.5)


def test_total_column_alias():
    """'total' column accepted for kind='all'."""
    df = pd.DataFrame({"date": ["1980Q1"], "total": [1.2]})
    events = mertens_ravn_csv_to_events(df, kind="all")
    assert len(events) == 1
    assert events[0].magnitude == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Anticipated + implementation profile logic
# ---------------------------------------------------------------------------

def test_anticipated_with_impl_q_sets_profile_when_impl_after_announce():
    """impl_q strictly after announce -> profile is set."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": ["1980Q3"],  # 2 quarters later
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events) == 1
    e = events[0]
    assert len(e.implementation_profile) == 1
    impl_date, weight = e.implementation_profile[0]
    assert impl_date == pd.Timestamp("1980-07-01")
    assert weight == pytest.approx(1.0)


def test_anticipated_with_impl_q_same_quarter_sets_profile():
    """impl_q == announce quarter satisfies impl_q >= date -> profile is set."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": ["1980Q1"],  # same quarter
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == [(pd.Timestamp("1980-01-01"), 1.0)]


def test_anticipated_with_impl_q_before_announce_no_profile():
    """impl_q strictly before announce -> profile stays empty."""
    df = pd.DataFrame({
        "date": ["1980Q3"],   # announce Q3
        "mtr_a": [0.4],
        "impl_q": ["1980Q1"],  # impl before announce
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == []


def test_anticipated_no_impl_col_empty_profile():
    """Without an impl_q/implementation_quarter column, profile is empty."""
    df = pd.DataFrame({"date": ["1980Q1"], "mtr_a": [0.5]})
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == []


def test_anticipated_with_impl_q_nan_no_profile():
    """NaN impl_q -> _to_quarter returns None -> profile stays empty."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": [None],
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == []


def test_anticipated_with_impl_q_empty_string_no_profile():
    """Empty string impl_q -> _to_quarter returns None -> no profile."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": [""],
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == []


def test_anticipated_with_impl_q_invalid_string_no_profile():
    """Unparseable impl_q -> _to_quarter returns None -> no profile."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": ["not-a-date"],
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile == []


def test_anticipated_with_impl_q_timestamp_string():
    """impl_q given as a Timestamp-parseable string '1980-07-01' (not Quarter literal)
    falls back to the pd.Timestamp -> pd.Period path."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": ["1980-07-01"],  # Q3 1980
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert len(events[0].implementation_profile) == 1
    assert events[0].implementation_profile[0][0] == pd.Timestamp("1980-07-01")


def test_anticipated_with_implementation_quarter_column_alias():
    """'implementation_quarter' column accepted as impl_col alias."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "implementation_quarter": ["1980Q3"],
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].implementation_profile[0][0] == pd.Timestamp("1980-07-01")


def test_unanticipated_ignores_impl_q_column():
    """impl_q column is ignored when kind='unanticipated' (no profile set)."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_u": [0.4],
        "impl_q": ["1980Q3"],
    })
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].implementation_profile == []


# ---------------------------------------------------------------------------
# delay_quarters property
# ---------------------------------------------------------------------------

def test_delay_quarters_for_anticipated_with_4q_gap():
    """announce Q1 1980, impl Q1 1981 -> delay = 4 quarters."""
    df = pd.DataFrame({
        "date": ["1980Q1"],
        "mtr_a": [0.5],
        "impl_q": ["1981Q1"],
    })
    events = mertens_ravn_csv_to_events(df, kind="anticipated")
    assert events[0].delay_quarters == 4


def test_delay_quarters_zero_for_no_profile():
    """No implementation profile -> delay_quarters == 0."""
    df = _simple_df(mtr_u=0.4)
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert events[0].delay_quarters == 0


# ---------------------------------------------------------------------------
# ValueError: missing date or value column
# ---------------------------------------------------------------------------

def test_missing_date_column_raises_value_error():
    """DataFrame without 'date'/'quarter' column raises ValueError."""
    df = pd.DataFrame({"other": ["1980Q1"], "mtr_u": [0.4]})
    with pytest.raises(ValueError, match="missing date/value columns"):
        mertens_ravn_csv_to_events(df, kind="unanticipated")


def test_missing_value_column_unanticipated_raises():
    """DataFrame without any mtr_u/mr_u/unanticipated column raises ValueError."""
    df = pd.DataFrame({"date": ["1980Q1"], "mtr_a": [0.2]})
    with pytest.raises(ValueError, match="missing date/value columns"):
        mertens_ravn_csv_to_events(df, kind="unanticipated")


def test_missing_value_column_all_kind_raises():
    """kind='all' without total column raises ValueError."""
    df = pd.DataFrame({"date": ["1980Q1"], "mtr_u": [0.4], "mtr_a": [0.1]})
    with pytest.raises(ValueError, match="missing date/value columns"):
        mertens_ravn_csv_to_events(df, kind="all")


def test_unknown_kind_raises_value_error():
    """Passing an unrecognised kind raises ValueError."""
    df = pd.DataFrame({"date": ["1980Q1"], "mtr_u": [0.4]})
    with pytest.raises(ValueError, match="missing date/value columns"):
        mertens_ravn_csv_to_events(df, kind="UNKNOWN")


# ---------------------------------------------------------------------------
# load() with csv_path
# ---------------------------------------------------------------------------

def _make_tmp_csv(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write(content)
    return path


def test_load_csv_path_returns_narrative_instrument():
    """load(csv_path=...) parses local CSV and returns NarrativeInstrument."""
    from puremacro.narrative.types import NarrativeInstrument
    csv = "date,mtr_u,mtr_a,mtr_total\n1980Q1,0.4,0.1,0.5\n1981Q3,-0.2,0.3,0.1\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path, kind="unanticipated")
    finally:
        os.unlink(path)
    assert isinstance(instr, NarrativeInstrument)
    assert len(instr.events) == 2


def test_load_csv_path_default_target_is_consumption():
    """Default target is 'consumption'."""
    csv = "date,mtr_u\n1980Q1,0.4\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path)
    finally:
        os.unlink(path)
    assert instr.target == "consumption"


def test_load_csv_path_custom_target():
    """target kwarg is forwarded to NarrativeInstrument."""
    csv = "date,mtr_u\n1980Q1,0.4\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path, target="investment")
    finally:
        os.unlink(path)
    assert instr.target == "investment"


def test_load_csv_kind_anticipated():
    """kind='anticipated' loads anticipated column."""
    csv = "date,mtr_u,mtr_a,mtr_total,impl_q\n1980Q1,0.4,0.1,0.5,1980Q3\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path, kind="anticipated")
    finally:
        os.unlink(path)
    assert len(instr.events) == 1
    assert instr.events[0].implementation_profile[0][0] == pd.Timestamp("1980-07-01")


def test_load_csv_kind_all():
    """kind='all' reads mtr_total column."""
    csv = "date,mtr_total\n1980Q1,0.5\n1981Q3,0.1\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path, kind="all")
    finally:
        os.unlink(path)
    assert len(instr.events) == 2


def test_load_csv_aggregation_is_sum():
    """NarrativeInstrument.aggregation is 'sum'."""
    csv = "date,mtr_u\n1980Q1,0.4\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path)
    finally:
        os.unlink(path)
    assert instr.aggregation == "sum"


def test_load_csv_quarterly_is_series():
    """quarterly property returns a pd.Series."""
    csv = "date,mtr_u\n1980Q1,0.4\n1981Q3,-0.2\n"
    path = _make_tmp_csv(csv)
    try:
        instr = load(csv_path=path)
    finally:
        os.unlink(path)
    assert isinstance(instr.quarterly, pd.Series)
    # Sum of signed magnitudes: +0.4 + (-0.2) = 0.2
    assert instr.quarterly.sum() == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# load() url= path (monkeypatched)
# ---------------------------------------------------------------------------

_MINIMAL_CSV_BYTES = (
    b"date,mtr_u,mtr_a,mtr_total\n1980Q1,0.4,0.1,0.5\n1981Q3,-0.2,0.3,0.1\n"
)
_URL = "https://example.invalid/mertens_ravn_2013.csv"


def test_load_network_success_via_mock():
    """load(url=...) fetches via safe_get_bytes; mocked to succeed."""
    with patch(
        "puremacro.narrative.replication.mertens_ravn_2013.safe_get_bytes",
        return_value=_MINIMAL_CSV_BYTES,
    ):
        instr = load(kind="unanticipated", url=_URL)
    assert len(instr.events) == 2


def test_load_network_all_kind_via_mock():
    """load(kind='all', url=...) reads the mtr_total column."""
    with patch(
        "puremacro.narrative.replication.mertens_ravn_2013.safe_get_bytes",
        return_value=_MINIMAL_CSV_BYTES,
    ):
        instr = load(kind="all", url=_URL)
    assert len(instr.events) == 2
    # Both mtr_total rows: 0.5 (positive) + 0.1 (positive) = 0.6
    assert instr.quarterly.sum() == pytest.approx(0.6, abs=1e-9)


def test_load_network_failure_raises_runtime_error():
    """When safe_get_bytes raises, load(url=...) wraps it in a RuntimeError."""
    with patch(
        "puremacro.narrative.replication.mertens_ravn_2013.safe_get_bytes",
        side_effect=Exception("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Mertens-Ravn 2013"):
            load(url=_URL)


def test_load_network_failure_chains_original_exception():
    """RuntimeError.__cause__ is the original exception (PEP 3134 chaining)."""
    original_exc = ValueError("bad response")
    with patch(
        "puremacro.narrative.replication.mertens_ravn_2013.safe_get_bytes",
        side_effect=original_exc,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            load(url=_URL)
    assert exc_info.value.__cause__ is original_exc


# ---------------------------------------------------------------------------
# load() default = shipped snapshot, fully offline
# ---------------------------------------------------------------------------

def test_load_default_snapshot_offline_no_network():
    """load() with no source arguments must read the package-data snapshot
    and never call safe_get_bytes."""
    with patch(
        "puremacro.narrative.replication.mertens_ravn_2013.safe_get_bytes",
        side_effect=AssertionError("default load() must not hit the network"),
    ):
        instr = load()
    assert len(instr.events) > 20


@pytest.mark.parametrize("kind,col", [
    ("unanticipated", "mtr_u"),
    ("anticipated", "mtr_a"),
])
def test_load_default_snapshot_counts_match_nonzero_rows(kind, col):
    """Event count per kind == number of nonzero rows in the snapshot."""
    from puremacro.replication._data import load_csv

    snap = load_csv("tax14_narrative_tax_shocks")
    expected = int((snap[col] != 0.0).sum())
    instr = load(kind=kind)
    assert len(instr.events) == expected


def test_load_default_snapshot_kind_all_derives_total():
    """The snapshot ships only mtr_u/mtr_a; kind='all' derives
    mtr_total = mtr_u + mtr_a instead of raising."""
    from puremacro.replication._data import load_csv

    snap = load_csv("tax14_narrative_tax_shocks")
    expected = int(((snap["mtr_u"] + snap["mtr_a"]) != 0.0).sum())
    instr = load(kind="all")
    assert len(instr.events) == expected


def test_load_default_snapshot_dates_span_1945_2007():
    instr = load()
    years = [e.date.year for e in instr.events]
    assert min(years) >= 1945
    assert max(years) <= 2007


# ---------------------------------------------------------------------------
# Multiple rows + ordering
# ---------------------------------------------------------------------------

def test_multiple_rows_preserve_order():
    """Events are emitted in the same row order as the input DataFrame."""
    df = pd.DataFrame({
        "date": ["1980Q1", "1981Q2", "1982Q3"],
        "mtr_u": [0.4, -0.3, 0.6],
    })
    events = mertens_ravn_csv_to_events(df, kind="unanticipated")
    assert [e.date for e in events] == [
        pd.Timestamp("1980-01-01"),
        pd.Timestamp("1981-04-01"),
        pd.Timestamp("1982-07-01"),
    ]


def test_all_zero_rows_gives_empty_event_list():
    """DataFrame with all zeros returns empty list."""
    df = pd.DataFrame({"date": ["1980Q1", "1981Q1"], "mtr_u": [0.0, 0.0]})
    assert mertens_ravn_csv_to_events(df, kind="unanticipated") == []


# ---------------------------------------------------------------------------
# metadata kind field for each kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,col", [
    ("unanticipated", "mtr_u"),
    ("anticipated", "mtr_a"),
    ("all", "mtr_total"),
])
def test_metadata_kind_matches_requested_kind(kind, col):
    """metadata['kind'] equals the kind argument passed."""
    df = pd.DataFrame({"date": ["1980Q1"], col: [0.4]})
    events = mertens_ravn_csv_to_events(df, kind=kind)
    assert events[0].metadata["kind"] == kind
