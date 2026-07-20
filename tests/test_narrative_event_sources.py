"""β.1 event-source connectors: schema + network smoke tests."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.sources._schema import validate_records


# ============================================================ WARN CA
def test_iter_us_warn_ca_yields_validated_5tuples():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    try:
        records = list(iter_us_warn(state="CA"))[:50]
    except Exception:
        pytest.skip("WARN CA fetch failed (network)")
    if not records:
        pytest.skip("WARN CA returned empty")
    validated = list(validate_records(records))
    assert all(len(r) == 5 for r in validated)
    assert all(r[3]["state"] == "CA" for r in validated)


@pytest.mark.network
def test_iter_us_warn_ca_min_records():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    records = list(iter_us_warn(state="CA"))
    if not records:
        pytest.skip("WARN CA returned empty")
    # With historical PDFs (FY2014-15 -- FY2024-25) plus the live xlsx,
    # we measured 13,140 records on 2026-05-17. 10,000 is a safe lower
    # bound that still detects regressions if any PDF source breaks.
    assert len(records) >= 10000, f"WARN CA only {len(records)} records"


def test_iter_us_warn_ny_yields_validated_5tuples():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    try:
        records = list(iter_us_warn(state="NY"))[:50]
    except Exception:
        pytest.skip("WARN NY fetch failed (network)")
    if not records:
        pytest.skip("WARN NY returned empty")
    validated = list(validate_records(records))
    assert all(len(r) == 5 for r in validated)
    assert all(r[3]["state"] == "NY" for r in validated)


@pytest.mark.network
def test_iter_us_warn_ny_min_records():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    records = list(iter_us_warn(state="NY"))
    if not records:
        pytest.skip("WARN NY returned empty")
    # With the 2023+2024 HTML-archive backfill plus the live Tableau,
    # we measured 767 records on 2026-05-17. 600 is a safe lower bound.
    assert len(records) >= 600, f"WARN NY only {len(records)} records"


def test_iter_us_warn_rejects_bad_state():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    with pytest.raises(ValueError, match="state must be"):
        list(iter_us_warn(state="XX"))


# ============================================================ FRDB strikes
def test_iter_frdb_strikes_yields_validated_5tuples():
    from puremacro.narrative.sources.frdb_strikes import iter_frdb_strikes
    try:
        records = list(iter_frdb_strikes())[:50]
    except Exception:
        pytest.skip("FRDB strikes fetch failed (network)")
    if not records:
        pytest.skip("FRDB strikes returned empty")
    validated = list(validate_records(records))
    assert all(len(r) == 5 for r in validated)
    assert all(r[3]["doctype"] == "strike" for r in validated)


@pytest.mark.network
def test_iter_frdb_strikes_min_records():
    from puremacro.narrative.sources.frdb_strikes import iter_frdb_strikes
    records = list(iter_frdb_strikes())
    if not records:
        pytest.skip("FRDB strikes empty")
    assert len(records) >= 100, f"FRDB only {len(records)} records"
