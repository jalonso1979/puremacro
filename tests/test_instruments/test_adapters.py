"""Round-trip tests for as_instrument() adapters on NarrativeInstrument
and JKResult."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, InstrumentLike


# --------------------------------------------------------------------------
# NarrativeInstrument.as_instrument()
# --------------------------------------------------------------------------
def _make_narrative_with_replication_flag():
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=10.0, magnitude_unit="USD_bn",
            target="investment", subtarget="defense", sign=1,
            confidence=1.0, source_text="t1", source_url="u1",
            scoring_method="manual",
            metadata={"replication": "ramey_2011"},
        ),
        NarrativeEvent(
            date=pd.Timestamp("2002-04-01"),
            country="USA", magnitude=12.0, magnitude_unit="USD_bn",
            target="investment", subtarget="defense", sign=1,
            confidence=1.0, source_text="t2", source_url="u2",
            scoring_method="manual",
            metadata={"replication": "ramey_2011"},
        ),
    ]
    return NarrativeInstrument.from_events(events, target="investment")


def test_narrative_instrument_satisfies_protocol():
    narr = _make_narrative_with_replication_flag()
    assert isinstance(narr, InstrumentLike)


def test_narrative_as_instrument_returns_Instrument():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert isinstance(inst, Instrument)
    assert inst.frequency == "Q"


def test_narrative_as_instrument_preserves_quarterly_series_identity():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.series is narr.quarterly


def test_narrative_as_instrument_picks_connector_category_by_default():
    """When no replication flag is in any event metadata, category is
    'narrative_connector'."""
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=5.0, magnitude_unit="USD_bn",
            target="investment", subtarget=None, sign=1,
            confidence=1.0, source_text="t", source_url="u",
            scoring_method="keyword",
            metadata={},
        ),
    ]
    narr = NarrativeInstrument.from_events(events)
    inst = narr.as_instrument()
    assert inst.category == "narrative_connector"


def test_narrative_as_instrument_picks_replication_category_when_flagged():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.category == "narrative_replication"


def test_narrative_as_instrument_metadata_includes_n_events():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.metadata.get("n_events") == 2
    assert inst.metadata.get("target") == "investment"


# --------------------------------------------------------------------------
# JKResult.as_instrument()
# --------------------------------------------------------------------------
def _make_jkresult():
    from puremacro.hfi import JKResult
    rng = np.random.default_rng(0)
    n = 30
    return JKResult(
        mp_shock=rng.standard_normal(n),
        info_shock=rng.standard_normal(n),
        rotation=np.eye(2),
        n_admissible=42,
        method="median_target",
    )


def test_jkresult_satisfies_protocol():
    """JKResult.as_instrument() needs the index= kwarg, but the protocol
    check only requires the method to *exist*. We still expect satisfaction."""
    jk = _make_jkresult()
    assert isinstance(jk, InstrumentLike)


def test_jkresult_as_instrument_mp_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="mp", index=idx)
    assert isinstance(inst, Instrument)
    assert inst.category == "monetary_hfi"
    assert inst.frequency == "M"
    assert inst.name == "jk2020_mp_shock"
    np.testing.assert_array_equal(inst.series.values, jk.mp_shock)


def test_jkresult_as_instrument_info_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="info", index=idx)
    np.testing.assert_array_equal(inst.series.values, jk.info_shock)


def test_jkresult_as_instrument_rejects_bad_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    with pytest.raises(ValueError, match="component"):
        jk.as_instrument(component="bogus", index=idx)


def test_jkresult_as_instrument_rejects_wrong_length_index():
    jk = _make_jkresult()
    bad_idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    with pytest.raises(ValueError, match="length"):
        jk.as_instrument(component="mp", index=bad_idx)


def test_jkresult_as_instrument_method_in_source_string():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="mp", index=idx)
    assert "median_target" in inst.source


def test_narrative_as_instrument_computed_facts_override_metadata():
    """n_events, target, aggregation always reflect the actual object state,
    even when self.metadata claims otherwise."""
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=5.0, magnitude_unit="USD_bn",
            target="investment", subtarget=None, sign=1,
            confidence=1.0, source_text="t", source_url="u",
            scoring_method="keyword",
            metadata={},
        ),
    ]
    narr = NarrativeInstrument.from_events(events, target="investment")
    # Plant misleading values in metadata.
    narr.metadata["n_events"] = 999
    narr.metadata["target"] = "consumption"
    narr.metadata["aggregation"] = "FAKE"
    inst = narr.as_instrument()
    # Computed facts must win.
    assert inst.metadata["n_events"] == 1
    assert inst.metadata["target"] == "investment"
    assert inst.metadata["aggregation"] == "sum"
