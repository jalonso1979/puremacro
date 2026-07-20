"""Coverage tests for puremacro/narrative/replication/romer_romer_2017.py.

Targets the branches NOT exercised by the single existing test in
test_narrative_offline.py::test_romer_romer_2017_csv_to_events_minimal_csv.

Missing lines at baseline (54%):
  54        - ValueError on missing required columns
  62-66     - date fallback: pd.Timestamp path + continue on total parse failure
  71        - continue on zero / NaN value
  97-115    - load() function: csv_path branch, countries filter,
              network-error branch (RuntimeError)
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from puremacro.narrative import NarrativeEvent
from puremacro.narrative.replication.romer_romer_2017 import (
    romer_romer_2017_csv_to_events,
    load,
    _ISO2_TO_ISO3,
)
from puremacro.narrative.types import NarrativeInstrument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(**overrides) -> pd.DataFrame:
    """Minimal valid DataFrame with 3 rows (2 nonzero, 1 zero)."""
    base = {
        "date": ["1990Q1", "1995Q3", "2000Q1"],
        "country": ["US", "DE", "FR"],
        "exog": [0.4, -0.3, 0.0],  # 0.0 gets filtered
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# romer_romer_2017_csv_to_events — error handling
# ---------------------------------------------------------------------------


def test_missing_date_column_raises():
    """Missing date/quarter column raises ValueError with useful message."""
    df = pd.DataFrame({"country": ["US"], "exog": [0.5]})
    with pytest.raises(ValueError, match="missing required columns"):
        romer_romer_2017_csv_to_events(df)


def test_missing_country_column_raises():
    """Missing country/iso3/iso/code column raises ValueError."""
    df = pd.DataFrame({"date": ["1990Q1"], "exog": [0.5]})
    with pytest.raises(ValueError, match="missing required columns"):
        romer_romer_2017_csv_to_events(df)


def test_missing_value_column_raises():
    """Missing value column (exog/exog_tax/y/tax_shock/rr_exog) raises ValueError."""
    df = pd.DataFrame({"date": ["1990Q1"], "country": ["US"]})
    with pytest.raises(ValueError, match="missing required columns"):
        romer_romer_2017_csv_to_events(df)


def test_all_columns_missing_raises():
    """Completely wrong columns raise ValueError."""
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="missing required columns"):
        romer_romer_2017_csv_to_events(df)


# ---------------------------------------------------------------------------
# romer_romer_2017_csv_to_events — date parsing branches
# ---------------------------------------------------------------------------


def test_date_as_timestamp_string_fallback():
    """When Period() parsing fails, pd.Timestamp() fallback is used.

    An ISO-format date string like '1990-01-01' is not a valid Quarter
    Period string, so the fallback path (lines 63-64) fires.
    """
    df = pd.DataFrame({
        "date": ["1990-01-01", "1995-07-01"],
        "country": ["US", "DE"],
        "exog": [0.4, -0.3],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 2
    # Timestamps must be valid
    for e in events:
        assert isinstance(e.date, pd.Timestamp)


def test_unparseable_date_is_skipped():
    """Rows with completely unparseable dates (both Period and Timestamp fail)
    are silently skipped (lines 65-66)."""
    df = pd.DataFrame({
        "date": ["NOT_A_DATE", "1990Q1"],
        "country": ["US", "DE"],
        "exog": [0.4, -0.3],
    })
    events = romer_romer_2017_csv_to_events(df)
    # Only the parseable row survives
    assert len(events) == 1
    assert events[0].country == "DEU"


def test_zero_value_is_skipped():
    """Rows with v == 0.0 are skipped (line 71)."""
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3", "2000Q1"],
        "country": ["US", "DE", "FR"],
        "exog": [0.0, 0.0, 0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1
    assert events[0].country == "FRA"


def test_nan_value_is_skipped():
    """Rows with NaN value are skipped (line 71 — pd.isna branch)."""
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3"],
        "country": ["US", "DE"],
        "exog": [float("nan"), -0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1
    assert events[0].country == "DEU"


def test_all_zero_or_nan_returns_empty_list():
    """All filtered rows gives empty list."""
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3"],
        "country": ["US", "DE"],
        "exog": [0.0, float("nan")],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert events == []


# ---------------------------------------------------------------------------
# romer_romer_2017_csv_to_events — column name aliases
# ---------------------------------------------------------------------------


def test_column_alias_quarter():
    """Accepts 'quarter' instead of 'date'."""
    df = pd.DataFrame({
        "quarter": ["1990Q1"],
        "country": ["US"],
        "exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


def test_column_alias_iso3():
    """Accepts 'iso3' as country column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "iso3": ["GBR"],
        "exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1
    assert events[0].country == "GBR"


def test_column_alias_code():
    """Accepts 'code' as country column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "code": ["USA"],
        "exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


def test_column_alias_iso():
    """Accepts 'iso' as country column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "iso": ["ITA"],
        "exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1
    assert events[0].country == "ITA"


def test_column_alias_exog_tax():
    """Accepts 'exog_tax' as value column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog_tax": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


def test_column_alias_tax_shock():
    """Accepts 'tax_shock' as value column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "tax_shock": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


def test_column_alias_rr_exog():
    """Accepts 'rr_exog' as value column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "rr_exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


def test_column_alias_y():
    """Accepts 'y' as value column."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "y": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# romer_romer_2017_csv_to_events — ISO2 → ISO3 conversion
# ---------------------------------------------------------------------------


def test_iso2_to_iso3_all_known_codes():
    """All 15 ISO2 codes in _ISO2_TO_ISO3 are correctly mapped to ISO3."""
    rows = []
    for iso2, iso3 in _ISO2_TO_ISO3.items():
        rows.append({"date": "1990Q1", "country": iso2, "exog": 0.3})
    df = pd.DataFrame(rows)
    events = romer_romer_2017_csv_to_events(df)
    emitted_countries = {e.country for e in events}
    expected = set(_ISO2_TO_ISO3.values())
    assert emitted_countries == expected


def test_unknown_code_passthrough():
    """Country codes not in the ISO2 map are passed through as-is."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["POL"],  # not in the lookup
        "exog": [0.5],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert events[0].country == "POL"


# ---------------------------------------------------------------------------
# romer_romer_2017_csv_to_events — sign / magnitude contract
# ---------------------------------------------------------------------------


def test_positive_value_gives_sign_plus_one():
    """Positive exog value → sign = +1, magnitude = |v|."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog": [0.75],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert events[0].sign == 1
    assert events[0].magnitude == pytest.approx(0.75)


def test_negative_value_gives_sign_minus_one():
    """Negative exog value → sign = -1, magnitude = |v|."""
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog": [-0.75],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert events[0].sign == -1
    assert events[0].magnitude == pytest.approx(0.75)


def test_magnitude_unit_is_pct_gdp():
    """Events carry magnitude_unit = 'pct_gdp'."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.magnitude_unit == "pct_gdp" for e in events)


def test_confidence_is_one():
    """All events have confidence = 1.0 (hand-coded, manual classification)."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.confidence == pytest.approx(1.0) for e in events)


def test_scoring_method_is_manual():
    """All events have scoring_method = 'manual'."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.scoring_method == "manual" for e in events)


def test_metadata_replication_marker():
    """Every event has metadata['replication'] == 'romer_romer_2017'."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.metadata.get("replication") == "romer_romer_2017" for e in events)


def test_target_is_consumption():
    """All events have target='consumption' and subtarget='tax'."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.target == "consumption" for e in events)
    assert all(e.subtarget == "tax" for e in events)


def test_implementation_profile_is_empty():
    """Events have no implementation_profile (RR2017 has no separate timing)."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(e.implementation_profile == [] for e in events)


def test_returns_narrative_event_instances():
    """Return type is a list of NarrativeEvent objects."""
    df = _minimal_df()
    events = romer_romer_2017_csv_to_events(df)
    assert all(isinstance(e, NarrativeEvent) for e in events)


# ---------------------------------------------------------------------------
# load() — csv_path branch
# ---------------------------------------------------------------------------


def test_load_with_csv_path_returns_instrument_dict(tmp_path):
    """load(csv_path=...) reads the file and returns a dict of NarrativeInstrument."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3", "2000Q1"],
        "country": ["US", "DE", "FR"],
        "exog": [0.4, -0.3, 0.5],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"USA", "DEU", "FRA"}
    for inst in result.values():
        assert isinstance(inst, NarrativeInstrument)


def test_load_with_csv_path_returns_correct_events(tmp_path):
    """Events inside the returned instruments match the input data."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["JP"],
        "exog": [0.4],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    assert "JPN" in result
    inst = result["JPN"]
    assert len(inst.events) == 1
    assert inst.events[0].sign == 1
    assert inst.events[0].magnitude == pytest.approx(0.4)


def test_load_csv_path_accepts_string(tmp_path):
    """csv_path can be a string (not just a Path object)."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog": [0.4],
    })
    df.to_csv(csv_file, index=False)
    # Pass as string
    result = load(csv_path=str(csv_file))
    assert "USA" in result


# ---------------------------------------------------------------------------
# load() — countries filter
# ---------------------------------------------------------------------------


def test_load_countries_filter_restricts_output(tmp_path):
    """countries=['USA'] returns only USA, not other countries."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3", "2000Q1"],
        "country": ["US", "DE", "FR"],
        "exog": [0.4, -0.3, 0.5],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file, countries=["USA"])
    assert set(result.keys()) == {"USA"}


def test_load_countries_filter_case_insensitive(tmp_path):
    """countries filter is case-insensitive (lowercase input accepted)."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3"],
        "country": ["US", "DE"],
        "exog": [0.4, -0.3],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file, countries=["usa", "deu"])
    assert set(result.keys()) == {"USA", "DEU"}


def test_load_countries_filter_nonexistent_country_returns_empty(tmp_path):
    """Filtering for a country not in the data yields an empty dict."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog": [0.4],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file, countries=["NZL"])
    assert result == {}


def test_load_no_countries_filter_returns_all(tmp_path):
    """countries=None (default) returns all countries."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3", "2000Q1"],
        "country": ["US", "DE", "FR"],
        "exog": [0.4, -0.3, 0.5],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# load() — NarrativeInstrument properties
# ---------------------------------------------------------------------------


def test_load_instruments_have_quarterly_series(tmp_path):
    """Returned NarrativeInstrument objects have a quarterly series (pd.Series)."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1990Q2"],
        "country": ["US", "US"],
        "exog": [0.4, -0.3],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    inst = result["USA"]
    assert isinstance(inst.quarterly, pd.Series)


def test_load_instrument_events_all_usa(tmp_path):
    """Events in USA instrument all have country == 'USA'."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1", "1990Q2", "1995Q3"],
        "country": ["US", "US", "DE"],
        "exog": [0.4, -0.3, 0.6],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    for ev in result["USA"].events:
        assert ev.country == "USA"


def test_load_target_is_consumption(tmp_path):
    """NarrativeInstrument.target == 'consumption'."""
    csv_file = tmp_path / "rr2017.csv"
    df = pd.DataFrame({
        "date": ["1990Q1"],
        "country": ["US"],
        "exog": [0.4],
    })
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    assert result["USA"].target == "consumption"


# ---------------------------------------------------------------------------
# load() — network error path
# ---------------------------------------------------------------------------


def test_load_network_failure_raises_runtime_error():
    """When csv_path is None and the network fetch fails, RuntimeError is raised."""
    with patch(
        "puremacro.narrative.replication.romer_romer_2017.safe_get_bytes",
        side_effect=OSError("Network unreachable"),
    ):
        with pytest.raises(RuntimeError, match="Could not fetch Romer-Romer 2017"):
            load()


def test_load_network_runtimeerror_wraps_original():
    """RuntimeError from network failure wraps the original exception via __cause__."""
    original = ConnectionError("timeout")
    with patch(
        "puremacro.narrative.replication.romer_romer_2017.safe_get_bytes",
        side_effect=original,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            load()
        assert exc_info.value.__cause__ is original


def test_load_network_success_parses_csv_bytes():
    """When safe_get_bytes succeeds, the CSV bytes are parsed correctly."""
    df_content = pd.DataFrame({
        "date": ["1990Q1", "1995Q3"],
        "country": ["US", "GB"],
        "exog": [0.4, -0.3],
    })
    csv_bytes = df_content.to_csv(index=False).encode("utf-8")

    with patch(
        "puremacro.narrative.replication.romer_romer_2017.safe_get_bytes",
        return_value=csv_bytes,
    ):
        result = load()
    assert "USA" in result
    assert "GBR" in result


# ---------------------------------------------------------------------------
# Edge cases — empty DataFrame
# ---------------------------------------------------------------------------


def test_empty_dataframe_returns_empty_list():
    """Empty DataFrame with correct columns returns []."""
    df = pd.DataFrame(columns=["date", "country", "exog"])
    events = romer_romer_2017_csv_to_events(df)
    assert events == []


def test_load_empty_csv_returns_empty_dict(tmp_path):
    """A CSV with headers but no rows returns an empty dict."""
    csv_file = tmp_path / "rr2017_empty.csv"
    df = pd.DataFrame(columns=["date", "country", "exog"])
    df.to_csv(csv_file, index=False)

    result = load(csv_path=csv_file)
    assert result == {}
