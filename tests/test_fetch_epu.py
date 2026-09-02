"""Offline tests for :mod:`puremacro.fetch.epu`, and its coverage guard.

`fetch` builds its country list from the columns the workbook actually has,
never from `_NAME_TO_ISO3`, so a country that disappears upstream disappears
from `epu_m` with no error and no warning — and `build_panel.build_all` merges
whatever arrives. That is how Sweden left the Baker–Bloom–Davis workbook
between the 2025 and 2026 editions without anything saying so.

The guard measures the file against `_EXPECTED_CODES`, a verified snapshot,
rather than against `set(_NAME_TO_ISO3.values())`: the map is a broad
name-to-code translation carrying variant spellings and six codes this file has
never published, so measuring against it would report the same six absences on
every call and teach the reader to ignore the warning.
"""
from __future__ import annotations

import warnings
from io import BytesIO

import pandas as pd
import pytest

from puremacro.fetch import epu as mod


def _workbook(countries=("Spain", "Japan", "US"), months=6,
              extra_cols=None) -> bytes:
    rows = []
    for i in range(months):
        row = {"Year": 2025, "Month": i + 1}
        for c in countries:
            row[c] = 100.0 + i
        for c in (extra_cols or []):
            row[c] = 1.0
        rows.append(row)
    # the aggregates the module excludes by design
    for r in rows:
        r["GEPU_current"], r["GEPU_ppp"] = 200.0, 201.0
    buf = BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def served(monkeypatch):
    def _serve(content):
        monkeypatch.setattr(mod, "cached_get", lambda *a, **k: content)
    return _serve


def _codes(*names):
    return frozenset(mod._NAME_TO_ISO3[n] for n in names)


def test_the_long_schema(served):
    served(_workbook())
    monkey = _codes("Spain", "Japan", "US")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mod.fetch()
    assert list(out.columns) == ["code", "date", "variable", "value",
                                 "sa_source", "source"]
    assert set(out["code"]) == monkey
    assert set(out["variable"]) == {"epu_m"}


def test_the_global_aggregates_are_excluded(served):
    """GEPU_current / GEPU_ppp have no ISO-3 code and must not be countries."""
    served(_workbook())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mod.fetch()
    assert not {"GEPU_current", "GEPU_ppp"} & set(out["code"])


def test_a_country_that_leaves_the_workbook_is_named(monkeypatch, served):
    """The Sweden case: the panel silently lost a country."""
    monkeypatch.setattr(mod, "_EXPECTED_CODES",
                        _codes("Spain", "Japan", "US", "Sweden"))
    served(_workbook(("Spain", "Japan", "US")))
    with pytest.warns(UserWarning, match="no longer carries SWE"):
        out = mod.fetch()
    assert "SWE" not in set(out["code"])     # the data still come back


def test_a_country_that_appears_is_named_and_included(monkeypatch, served):
    monkeypatch.setattr(mod, "_EXPECTED_CODES", _codes("Spain", "Japan"))
    served(_workbook(("Spain", "Japan", "US")))
    with pytest.warns(UserWarning, match="now carries USA"):
        out = mod.fetch()
    assert "USA" in set(out["code"])


def test_an_unmapped_column_is_named_not_dropped_in_silence(monkeypatch, served):
    """A newly published country would otherwise vanish into the map's gap."""
    monkeypatch.setattr(mod, "_EXPECTED_CODES", _codes("Spain", "Japan", "US"))
    served(_workbook(("Spain", "Japan", "US"), extra_cols=["Nigeria"]))
    with pytest.warns(UserWarning, match="no ISO-3 mapping.*Nigeria"):
        mod.fetch()


def test_an_unchanged_workbook_is_silent(monkeypatch, served):
    """Positive control. A guard that always fires is a guard nobody reads."""
    monkeypatch.setattr(mod, "_EXPECTED_CODES", _codes("Spain", "Japan", "US"))
    served(_workbook(("Spain", "Japan", "US")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not mod.fetch().empty


def test_the_snapshot_is_not_the_whole_name_map():
    """`_EXPECTED_CODES` must stay narrower than the translation map.

    Six codes in `_NAME_TO_ISO3` (BEL, COL, HKG, NLD, NZL, ZAF) have never been
    published in this workbook. Measuring coverage against the map would report
    them absent on every call, which is exactly how a warning gets ignored.
    """
    assert mod._EXPECTED_CODES < set(mod._NAME_TO_ISO3.values())
    assert not {"BEL", "COL", "HKG", "NLD", "NZL", "ZAF"} & mod._EXPECTED_CODES
    # Sweden stays in the translation map so a return is reported, not absorbed
    assert mod._NAME_TO_ISO3["Sweden"] == "SWE"
    assert "SWE" not in mod._EXPECTED_CODES


def test_the_snapshot_matches_the_shipped_workbook():
    """Pins the guard to reality: the cached edition must be silent.

    If this fails, either the workbook in `data/raw/` changed or the snapshot
    is stale — both of which are the thing the guard exists to surface.
    """
    path = ("data/raw/www.policyuncertainty.com/"
            "media_All_Country_Data.xlsx")
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    book = root / path
    if not book.is_file():
        pytest.skip("cached EPU workbook not present")
    raw = pd.read_excel(book)
    cols = [c for c in raw.columns if c not in ("Year", "Month")]
    produced = {mod._NAME_TO_ISO3[c] for c in cols if c in mod._NAME_TO_ISO3}
    assert produced == set(mod._EXPECTED_CODES)
