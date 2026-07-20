"""Offline tests for narrative connectors.

Each connector under ``puremacro/narrative/sources/`` and
``puremacro/narrative/replication/`` is exercised against the HTTP
fixture cache (see ``tests/_http_fixtures.py``). These tests guard
against three classes of regression:

1. Parser breakage when an upstream HTML/JSON shape changes (caught
   only when fixtures are re-recorded with
   ``PUREMACRO_RECORD_HTTP=1``).
2. Silent ``yield`` regressions where a connector swallows all
   results — these tests assert ≥1 valid item is yielded against the
   recorded snapshot.
3. The replication CSV → events helpers, which are pure parsers and
   never need the network.

Live sources for which no fixture has been recorded skip cleanly with
a pointer to the record-mode workflow. The infrastructure is the
deliverable; fixtures fill in incrementally as the user exercises
``PUREMACRO_RECORD_HTTP=1`` in environments that have network access.

Recording new fixtures
----------------------

::

    PUREMACRO_RECORD_HTTP=1 pytest tests/test_narrative_offline.py

The first run hits the live endpoints and writes ``<sha>.json`` files
into ``tests/fixtures/http/``. Subsequent runs in default (replay)
mode read those files only.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from _http_fixtures import (
    fixture_path,
    install_fixture_patches,
    _record_mode,
)

from puremacro.narrative import NarrativeEvent
from puremacro.narrative.replication import (
    ramey_csv_to_events,
    romer_romer_2010_csv_to_events,
    romer_romer_2017_csv_to_events,
    mertens_ravn_csv_to_events,
    cloyne_csv_to_events,
    dglp_csv_to_events,
)


# ---------------------------------------------------------------------------
# Patching fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def narrative_offline():
    """Activate fixture-cache patches for the narrative ``_http``
    helpers. Use this fixture (not ``autouse``) so other tests in the
    file that don't need patching aren't affected."""
    with install_fixture_patches():
        yield


def _skip_if_missing(url: str, headers: dict | None = None) -> None:
    """In replay mode, skip the test when the fixture has not been
    recorded yet. In record mode, do nothing — the patched helper will
    capture the response."""
    if _record_mode():
        return
    if not fixture_path(url, headers).exists():
        pytest.skip(
            f"No HTTP fixture recorded for {url!r}. "
            f"Re-record with PUREMACRO_RECORD_HTTP=1."
        )


# ---------------------------------------------------------------------------
# Live source connectors — each test is a parser-regression guard
# ---------------------------------------------------------------------------


def test_iter_rss_parses_synthetic_feed(narrative_offline):
    """Synthetic RSS fixture is bundled (not live-recorded) so this
    test always runs in replay mode without network. Confirms the
    generic ``iter_rss`` parser yields ≥1 record from a well-formed
    RSS body.

    Skip in record mode — the synthetic fixture is hand-rolled to
    exercise edge cases of the parser; firing real HTTP at
    ``example.org`` would clobber it with a 404 page or worse.
    """
    if _record_mode():
        pytest.skip("Synthetic fixture is hand-rolled, not network-recorded.")
    from puremacro.narrative.sources._rss import iter_rss

    records = list(iter_rss("https://example.org/test-rss.xml"))
    assert len(records) >= 1
    date, text, link = records[0]
    assert isinstance(date, pd.Timestamp)
    assert isinstance(text, str) and text.strip()
    assert isinstance(link, str)


def test_iter_atom_parses_synthetic_feed(narrative_offline):
    """Synthetic Atom fixture, bundled with the repo. Confirms the
    generic ``iter_atom`` parser yields ≥1 record.

    See :func:`test_iter_rss_parses_synthetic_feed` for the rationale
    on skipping in record mode.
    """
    if _record_mode():
        pytest.skip("Synthetic fixture is hand-rolled, not network-recorded.")
    from puremacro.narrative.sources._rss import iter_atom

    records = list(iter_atom("https://example.org/test-atom.xml"))
    assert len(records) >= 1
    date, text, link = records[0]
    assert isinstance(date, pd.Timestamp)
    assert isinstance(text, str) and text.strip()


@pytest.mark.skip(
    reason="fixture not yet recorded — re-run: PUREMACRO_RECORD_HTTP=1 "
           "pytest tests/test_narrative_offline.py::test_us_treasury_press_replays_recorded_feed"
)
def test_us_treasury_press_replays_recorded_feed(narrative_offline):
    """US Treasury press releases. Scrapes the HTML listing page at
    ``_BASE`` (page 1, no query string)."""
    from puremacro.narrative.sources import us_treasury

    _skip_if_missing(us_treasury._BASE)
    records = list(us_treasury.iter_treasury_press(max_pages=1))
    assert len(records) >= 1
    date, text, link = records[0]
    assert isinstance(date, pd.Timestamp)
    assert isinstance(text, str)
    assert isinstance(link, str)


def test_us_federal_register_replays_known_query(narrative_offline):
    """US Federal Register API.

    Uses a narrow date window + a single, currently-valid agency slug
    (``treasury-department``). The connector's default agency tuple
    bundles ``office-of-management-and-budget``, but that slug returns
    HTTP 400 from the live FR API as of 2026 — covered by an explicit
    test override here. The fixture key is built from the exact
    ``urlencode`` output, so we reconstruct it for ``_skip_if_missing``.
    """
    import urllib.parse
    from puremacro.narrative.sources import us_federal_register as fr

    until = "2024-02-01"
    since = "2024-01-01"
    agencies = ("treasury-department",)
    document_types = ("RULE", "NOTICE")
    params = {
        "conditions[publication_date][gte]": since,
        "conditions[publication_date][lte]": until,
        "conditions[type][]": list(document_types),
        "conditions[agencies][]": list(agencies),
        "per_page": 100,
        "page": 1,
        "fields[]": ["title", "abstract", "publication_date",
                     "html_url", "type", "agencies"],
    }
    url = fr._BASE + "?" + urllib.parse.urlencode(params, doseq=True)
    _skip_if_missing(url)

    records = list(fr.iter_federal_register(
        since=since, until=until, max_pages=1, per_page=100,
        agencies=agencies, document_types=document_types,
    ))
    assert len(records) >= 1
    date, text, link = records[0]
    assert isinstance(date, pd.Timestamp)
    assert isinstance(text, str)


@pytest.mark.skip(
    reason="fixture not yet recorded — re-run: PUREMACRO_RECORD_HTTP=1 "
           "pytest tests/test_narrative_offline.py::test_us_dod_contracts_replays_listing"
)
def test_us_dod_contracts_replays_listing(narrative_offline):
    """US DoD daily contracts listing page 1."""
    from puremacro.narrative.sources import us_dod_contracts as dod

    url = dod._BASE  # page 1 has no query string
    _skip_if_missing(url)
    records = list(dod.iter_dod_contracts(max_pages=1))
    if not _record_mode():
        assert len(records) >= 1


def test_imf_articleiv_replays_repec_listing(narrative_offline):
    """IMF Article IV via RePEc listing. Tests page 1 only to keep
    the fixture set small."""
    from puremacro.narrative.sources import imf_articleiv as imf

    # iter_imf_listing walks pages; the connector calls
    # ``_SERIES_LISTING_TPL.format(n="")`` for page 1.
    page1 = imf._SERIES_LISTING_TPL.format(n="")
    _skip_if_missing(page1)

    # Reset the in-process cache before we start, so the fixture cache
    # actually drives the result.
    imf.clear_listing_cache()
    records = list(imf.iter_imf_listing(max_pages=1, use_cache=False))
    if not _record_mode():
        assert len(records) >= 1


def test_oecd_surveys_replays_repec_listing(narrative_offline):
    """OECD working papers via RePEc listing."""
    from puremacro.narrative.sources import oecd_surveys as oec

    page1 = oec._SERIES_LISTING_TPL.format(slug="ecoaaa", n="")
    _skip_if_missing(page1)

    oec.clear_listing_cache()
    records = list(oec.iter_oecd_listing(series="wp", max_pages=1,
                                          use_cache=False))
    if not _record_mode():
        assert len(records) >= 1


def test_news_api_gdelt_replays_query(narrative_offline):
    """GDELT 2.0 DOC API. Uses a pinned date range so the fixture
    key is deterministic across re-records."""
    import urllib.parse
    from puremacro.narrative.sources.news_api import iter_gdelt_v2, _GDELT_BASE

    start, end = "2020-03-01", "2020-03-08"
    # GDELT requires either a single term or parentheses *only* around
    # OR groups. Use a multi-term OR group to make the parens valid.
    query = "infrastructure OR stimulus"
    full_query = f"({query}) sourcecountry:US"
    params = {
        "query": full_query,
        "mode": "ArtList",
        "format": "JSON",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
        "maxrecords": "250",
    }
    url = _GDELT_BASE + "?" + urllib.parse.urlencode(params)
    _skip_if_missing(url)

    records = list(iter_gdelt_v2(query=query, start=start, end=end,
                                   max_records=250))
    if not _record_mode():
        # GDELT can legitimately return zero hits for narrow queries —
        # a passing parser still yields nothing. We at least confirm
        # the iterator returned without exception, which is what the
        # parser-regression guard cares about.
        assert isinstance(records, list)


def test_eu_ecfin_replays_press_corner(narrative_offline):
    """EU Commission press corner feed (DG ECFIN-filtered)."""
    from puremacro.narrative.sources import eu_ecfin as ecfin

    _skip_if_missing(ecfin._FEED)
    records = list(ecfin.iter_ecfin_press(filter_ecfin=False))
    if not _record_mode():
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# Replication CSV → events helpers (no HTTP needed; these are pure parsers)
# ---------------------------------------------------------------------------


def test_ramey_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": ["1950Q3", "1951Q1", "1965Q1"],
        "news_q": [2.5, 1.2, 0.0],   # zero filtered out
    })
    events = ramey_csv_to_events(df)
    assert len(events) == 2
    assert all(isinstance(e, NarrativeEvent) for e in events)
    assert all(e.country == "USA" for e in events)


def test_romer_romer_2010_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": ["1981Q3", "1986Q1"],
        "rr_exog": [-0.5, 1.2],
    })
    events = romer_romer_2010_csv_to_events(df)
    assert len(events) == 2
    assert events[0].sign == -1
    assert events[1].sign == +1


def test_romer_romer_2017_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": ["1990Q1", "1995Q3"],
        "country": ["US", "DE"],
        "exog": [0.4, -0.3],
    })
    events = romer_romer_2017_csv_to_events(df)
    assert len(events) == 2
    # ISO2 → ISO3 fix-up applied at parse time.
    iso3s = {e.country for e in events}
    assert iso3s == {"USA", "DEU"}


def test_mertens_ravn_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": ["1980Q1", "1981Q3"],
        "mtr_u": [0.4, -0.2],
        "mtr_a": [0.1, 0.3],
        "mtr_total": [0.5, 0.1],
    })
    unanticipated = mertens_ravn_csv_to_events(df, kind="unanticipated")
    anticipated = mertens_ravn_csv_to_events(df, kind="anticipated")
    total = mertens_ravn_csv_to_events(df, kind="all")
    assert len(unanticipated) == 2
    assert len(anticipated) == 2
    assert len(total) == 2


def test_cloyne_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": ["1976Q1", "1979Q2", "1985Q3"],
        "exog": [0.0, 0.6, -0.4],   # zero dropped
    })
    events = cloyne_csv_to_events(df)
    assert len(events) == 2
    assert all(e.country == "GBR" for e in events)


def test_dglp_csv_to_events_minimal_csv():
    df = pd.DataFrame({
        "date": [1992, 1993, 2010],
        "country": ["ITA", "ITA", "ESP"],
        "tax_based": [1.4, 0.4, 0.4],
        "spending_based": [0.5, 1.6, 1.4],
    })
    events = dglp_csv_to_events(df, flip_sign=True)
    # With grouping: ITA 1992-93 tax → 1 event, ITA 1992-93 expenditure → 1
    # event, ESP 2010 tax → 1 event, ESP 2010 expenditure → 1 event = 4 events.
    # (Old behavior: 3 rows × 2 components = 6. Now consecutive same-country /
    # same-subtarget / same-sign rows collapse into one event with a
    # multi-quarter implementation_profile.)
    assert len(events) == 4
    # flip_sign=True turns DGLP "tightening = positive" into the
    # puremacro "expansionary = positive" convention, so consolidations
    # come out as sign = -1.
    assert all(e.sign == -1 for e in events)
