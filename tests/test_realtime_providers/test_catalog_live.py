"""Live audit of the real-time catalogue. Opt-in: ``pytest -m network``.

The whole point of this file is that series identifiers rot. puremacro
previously resolved every non-US country to FRED's OECD-MEI codes,
which stopped updating in January 2024 — and nothing in the test suite
noticed, because every offline test used synthetic payloads. A
catalogue that claims 42 countries and delivers 2 looks exactly like a
catalogue that works.

So this walks the real table against the real endpoints and reports
which entries no longer return two or more editions. It is skip-on-
empty by design: a provider being unreachable, or rate-limiting, is not
a test failure — it is a different fact from an identifier being dead,
and conflating them would make the audit untrustworthy in the other
direction.

Run it when a cross-country study comes back thinner than expected.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.network


MIN_VINTAGES = 2


def _alfred_entries():
    from puremacro.fetch.realtime.catalog import vintage_catalog
    cat = vintage_catalog("alfred")
    return cat[cat["variable"] == "gdp_real"].to_dict("records")


@pytest.mark.parametrize(
    "entry", _alfred_entries(), ids=lambda e: f"{e['country']}-{e['series_id']}")
def test_alfred_series_still_has_multiple_vintages(entry):
    """Each catalogued ALFRED series must still archive >= 2 editions."""
    from puremacro.fetch.realtime.alfred import alfred_vintage_dates

    try:
        dates = alfred_vintage_dates(entry["series_id"])
    except Exception as exc:                      # unreachable / throttled
        pytest.skip(f"could not reach ALFRED for {entry['series_id']}: {exc}")
    if not dates:
        pytest.fail(
            f"{entry['country']} -> {entry['series_id']} returned no vintage "
            "dates at all. Either the identifier is dead or the series left "
            "the archive; fix the catalogue rather than the test."
        )
    assert len(dates) >= MIN_VINTAGES, (
        f"{entry['country']} -> {entry['series_id']} has only {len(dates)} "
        f"edition(s); a revision test needs at least {MIN_VINTAGES}."
    )


def test_oecd_stes_country_list_matches_the_live_dataflow():
    """The 42-country list must match what the archive actually serves."""
    import io

    import pandas as pd

    from puremacro.fetch.realtime._base import fetch_with_backoff
    from puremacro.fetch.realtime.oecd_stes import (
        DATAFLOW, OECD_STES_COUNTRIES, OECD_STES_URL,
    )

    url = OECD_STES_URL.format(dataflow=DATAFLOW, key=".Q.B1GQ_Q...202608")
    url += "&startPeriod=2024-Q1&endPeriod=2024-Q1"
    try:
        raw = fetch_with_backoff(url, timeout=180.0)
    except Exception as exc:
        pytest.skip(f"could not reach the OECD STES archive: {exc}")
    live = set(pd.read_csv(io.BytesIO(raw))["REF_AREA"].unique())
    if not live:
        pytest.skip("OECD STES returned no reference areas")
    catalogued = set(OECD_STES_COUNTRIES)
    assert not catalogued - live, (
        f"catalogue claims countries the archive no longer serves: "
        f"{sorted(catalogued - live)}"
    )
    # New countries are a prompt to widen the list, not a failure.
    if live - catalogued:
        pytest.skip(
            f"archive has gained {sorted(live - catalogued)}; widen "
            "OECD_STES_COUNTRIES"
        )


@pytest.mark.parametrize("country,variable", [("MEX", "gdp_real"),
                                              ("ESP", "gdp_real")])
def test_oecd_stes_returns_a_real_triangle(country, variable):
    """End-to-end: a real fetch must yield many editions, not one."""
    from puremacro.fetch.realtime.oecd_stes import fetch_oecd_stes_vintages

    try:
        df = fetch_oecd_stes_vintages(country, variable)
    except Exception as exc:
        pytest.skip(f"could not reach the OECD STES archive: {exc}")
    if df.empty:
        pytest.skip(f"OECD STES returned nothing for {country}")
    assert df["vintage"].nunique() > 50, (
        f"{country} came back with only {df['vintage'].nunique()} editions"
    )


def test_the_dead_dataflow_is_still_dead():
    """Guards the comment explaining why DF_VINTAGES is not used.

    If the OECD ever restores ``DSD_STES@DF_VINTAGES`` this fails, which
    is the prompt to revisit the module docstring that tells users it
    is gone.
    """
    from puremacro.fetch.realtime._base import fetch_with_backoff

    url = ("https://sdmx.oecd.org/public/rest/data/"
           "OECD.SDD.STES,DSD_STES@DF_VINTAGES,4.0/all"
           "?format=csvfile&lastNObservations=1")
    try:
        raw = fetch_with_backoff(url, timeout=60.0, attempts=1)
    except Exception:
        return                                    # dead, as documented
    assert not raw.strip(), (
        "OECD.SDD.STES,DSD_STES@DF_VINTAGES now returns data. The "
        "oecd_stes module docstring says it does not; update it."
    )
