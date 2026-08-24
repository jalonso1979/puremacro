"""Detecting archive editions that carry a raw, not adjusted, series.

The OECD STES revisions archive has no seasonal-adjustment dimension.
For twelve reference areas its early editions are the *unadjusted*
series, and the archive switched over area by area between 2000 and
2007 without recording it anywhere — so it cannot be read from the
metadata and has to be read off the data.

It matters because nothing downstream survives it. On the real archive
Sweden's first estimate of 2002Q4 is +16.09% quarterly growth against
+0.07% today; a news-vs-noise test run over that is regressing a
seasonal factor and reports beta = -0.96, i.e. "almost all of the first
release is measurement error". It is not: it is December.

These tests are built on synthetic editions where the answer is known
by construction, so they say something about the screen rather than
about one archive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.realtime._base import VintagePanel, normalize_vintage_frame
from puremacro.fetch.realtime.seasonal import (
    SEASONAL_F_MIN,
    SEASONAL_RANGE_MIN,
    drop_unadjusted_editions,
    seasonal_signature,
)


def _edition(vintage, *, seasonal_pp=0.0, n=80, seed=0, start="1996Q1"):
    """One edition's level series, with a quarterly pattern of a chosen size.

    ``seasonal_pp`` is the peak-to-trough spread of the four quarterly
    mean growth rates, in percentage points — the statistic the screen
    keys on — so a fixture's expected verdict follows from its argument.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n, freq="Q").to_timestamp()
    pattern = np.array([-0.5, -0.1, 0.1, 0.5]) * (seasonal_pp / 1.0)
    g = 0.5 + 0.15 * rng.standard_normal(n) + pattern[np.arange(n) % 4]
    level = 100 * np.exp(np.cumsum(g / 100.0))
    return pd.DataFrame({"date": idx, "vintage": pd.Timestamp(vintage),
                         "value": level})


def _panel(editions, country="XXX"):
    frames = [normalize_vintage_frame(df, country=country,
                                      variable="gdp_real",
                                      provider="oecd_stes",
                                      series_id="X", units="level")
              for df in editions]
    return VintagePanel(df=pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------
def test_an_adjusted_edition_is_not_flagged():
    sig = seasonal_signature(_panel([_edition("2003-01-01", seasonal_pp=0.0)]))
    assert not sig["unadjusted"].any()


def test_a_raw_edition_is_flagged():
    sig = seasonal_signature(_panel([_edition("2003-01-01", seasonal_pp=20.0)]))
    assert sig["unadjusted"].all()
    assert sig["seasonal_range"].iloc[0] > SEASONAL_RANGE_MIN
    assert sig["F"].iloc[0] > SEASONAL_F_MIN


def test_only_the_raw_editions_of_a_mixed_country_are_flagged():
    """The real pattern: early editions raw, later ones adjusted, with
    the switchover undocumented."""
    panel = _panel([
        _edition("2001-01-01", seasonal_pp=20.0, seed=1),
        _edition("2003-01-01", seasonal_pp=20.0, seed=2),
        _edition("2009-01-01", seasonal_pp=0.0, seed=3),
        _edition("2020-01-01", seasonal_pp=0.0, seed=4),
    ])
    sig = seasonal_signature(panel).set_index("vintage")["unadjusted"]
    assert sig[pd.Timestamp("2001-01-01")]
    assert sig[pd.Timestamp("2003-01-01")]
    assert not sig[pd.Timestamp("2009-01-01")]
    assert not sig[pd.Timestamp("2020-01-01")]


def test_both_criteria_are_required():
    """A significance test alone rejects for residual seasonality far too
    small to matter; a range alone can be faked by outliers. Raising
    either threshold past the fixture must clear the flag."""
    panel = _panel([_edition("2003-01-01", seasonal_pp=20.0)])
    assert seasonal_signature(panel)["unadjusted"].all()
    assert not seasonal_signature(panel, range_min=1e6)["unadjusted"].any()
    assert not seasonal_signature(panel, f_min=1e9)["unadjusted"].any()


def test_a_rebasing_does_not_look_like_seasonality():
    """The screen works on each edition's own log growth, so multiplying
    a whole edition by a constant must change nothing at all."""
    base = _edition("2003-01-01", seasonal_pp=0.0, seed=7)
    rebased = base.copy()
    rebased["value"] = rebased["value"] * 137.0
    a = seasonal_signature(_panel([base]))
    b = seasonal_signature(_panel([rebased]))
    assert not a["unadjusted"].any() and not b["unadjusted"].any()
    assert a["seasonal_range"].iloc[0] == pytest.approx(
        b["seasonal_range"].iloc[0])


def test_a_short_edition_is_not_judged():
    sig = seasonal_signature(_panel([_edition("2003-01-01", seasonal_pp=20.0,
                                              n=6)]))
    assert not sig["unadjusted"].any()


def test_signature_reports_the_statistics_it_judged_on():
    sig = seasonal_signature(_panel([_edition("2003-01-01", seasonal_pp=20.0)]))
    assert set(["country", "variable", "vintage", "n", "F", "seasonal_range",
                "unadjusted"]).issubset(sig.columns)


# ---------------------------------------------------------------------------
# dropping
# ---------------------------------------------------------------------------
def test_dropping_removes_only_the_flagged_editions():
    panel = _panel([
        _edition("2001-01-01", seasonal_pp=20.0, seed=1),
        _edition("2009-01-01", seasonal_pp=0.0, seed=3),
    ])
    kept, report = drop_unadjusted_editions(panel)
    assert set(kept.df["vintage"].unique()) == {pd.Timestamp("2009-01-01")}
    assert report.loc["XXX", "editions"] == 1


def test_the_report_says_what_went():
    """A screen that does not say what it removed is indistinguishable
    from a bug."""
    panel = _panel([
        _edition("2001-01-01", seasonal_pp=20.0, seed=1),
        _edition("2003-01-01", seasonal_pp=20.0, seed=2),
        _edition("2009-01-01", seasonal_pp=0.0, seed=3),
    ])
    _, report = drop_unadjusted_editions(panel)
    row = report.loc["XXX"]
    assert row["editions"] == 2
    assert row["first"] == pd.Timestamp("2001-01-01")
    assert row["last"] == pd.Timestamp("2003-01-01")
    assert row["median_range"] > SEASONAL_RANGE_MIN


def test_a_clean_country_loses_nothing_and_reports_nothing():
    panel = _panel([_edition("2009-01-01", seasonal_pp=0.0)])
    kept, report = drop_unadjusted_editions(panel)
    assert len(kept.df) == len(panel.df)
    assert report.empty


def test_dropping_is_a_no_op_on_an_empty_panel():
    kept, report = drop_unadjusted_editions(
        VintagePanel(df=pd.DataFrame()))
    assert kept.is_empty() and report.empty


def test_vintage_panel_drops_by_default(monkeypatch):
    """The default matters: leaving raw editions in is silent, and a
    revision test over them regresses a seasonal factor."""
    import inspect
    from puremacro.fetch.realtime.panel import vintage_panel
    assert inspect.signature(
        vintage_panel).parameters["drop_unadjusted"].default is True
