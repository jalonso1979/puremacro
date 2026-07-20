"""Tests for the RiskIndex dataclass."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative import RiskIndex


def _example_series():
    return pd.Series(
        [100.0, 110.0, 95.0, 102.0],
        index=pd.date_range("2020-01-01", periods=4, freq="QS"),
        name="risk_index",
    )


def test_construct_basic():
    ri = RiskIndex(
        name="epu_us_news",
        country="USA",
        series=_example_series(),
        method="keyword_count",
        corpus="news",
        language="en",
        normalization="bbd_100",
    )
    assert ri.name == "epu_us_news"
    assert ri.method == "keyword_count"
    assert ri.normalization == "bbd_100"
    assert len(ri.series) == 4


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        RiskIndex(
            name="x", country="USA", series=_example_series(),
            method="not_a_method", corpus="news", language="en",
            normalization="bbd_100",
        )


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        RiskIndex(
            name="x", country="USA", series=_example_series(),
            method="keyword_count", corpus="news", language="en",
            normalization="weird",
        )


def test_diagnostics_returns_expected_keys():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    d = ri.diagnostics()
    assert {"n_quarters", "mean", "std", "first_date", "last_date"} <= set(d)
    assert d["n_quarters"] == 4
    assert d["mean"] == pytest.approx(101.75)


def test_to_frame_is_tidy():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    df = ri.to_frame()
    assert set(df.columns) == {"qdate", "value", "country", "name"}
    assert len(df) == 4
    assert (df["country"] == "USA").all()
    assert (df["name"] == "epu_us_news").all()


def test_as_instrument_round_trip():
    ri = RiskIndex(
        name="epu_us_news", country="USA", series=_example_series(),
        method="keyword_count", corpus="news", language="en",
        normalization="bbd_100",
    )
    inst = ri.as_instrument()
    assert inst.name == "epu_us_news"
    assert inst.category == "text_index"
    assert inst.frequency == "Q"
    assert inst.metadata["corpus"] == "news"
    assert inst.metadata["language"] == "en"
    assert inst.metadata["method"] == "keyword_count"


# ---------------------------------------------------------------------------
# NarrativeInstrument.as_instrument — kind passthrough (Task 4b)
# ---------------------------------------------------------------------------
def test_narrative_instrument_threads_kind_into_metadata():
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument

    events = [
        NarrativeEvent(
            date=pd.Timestamp("2022-03-16"), country="USA",
            magnitude=25.0, magnitude_unit="bps",
            target="policy_rate", subtarget=None, sign=+1,
            confidence=0.9, source_text="t", source_url="u",
            scoring_method="manual", kind="monetary",
        ),
    ]
    inst = NarrativeInstrument.from_events(events).as_instrument()
    assert inst.metadata.get("kinds") == ["monetary"]
    assert inst.category in {"narrative_replication", "narrative_connector"}


def test_narrative_instrument_kinds_with_legacy_fiscal_default():
    """Legacy fiscal events (kind defaulted) still produce kinds=['fiscal']."""
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument

    e = NarrativeEvent(
        date=pd.Timestamp("2020-01-15"), country="USA", magnitude=10.0,
        magnitude_unit="USD_bn", target="investment", subtarget=None,
        sign=+1, confidence=0.9, source_text="t", source_url="u",
        scoring_method="manual",
    )
    inst = NarrativeInstrument.from_events([e]).as_instrument()
    assert inst.metadata.get("kinds") == ["fiscal"]
