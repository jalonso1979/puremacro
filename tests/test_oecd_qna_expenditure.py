"""The SDMX expenditure gap-filler, and the price base it used to insist on.

Filtering to ``PRICE_BASE == "L"`` returned an empty frame for every reference
area publishing only fixed-base volumes — Mexico, Argentina, Indonesia, India,
South Africa — which is the exact failure `oecd_qna_panel` was written to avoid
and which its module docstring named this module for. Verified against the real
cache before the fix: MEX, ARG and ZAF returned nothing at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import oecd_qna_expenditure as mod
from puremacro.fetch.oecd_qna_expenditure import fetch_qna_expenditure

_QUARTERS = [f"{y}-Q{q}" for y in (2019, 2020) for q in (1, 2, 3, 4)]


def _rows(code: str, price_base: str, *, sector: str, level: float,
          adjustment: str = "Y") -> list[dict]:
    out = []
    for txn, sec in [(t, s) for (t, s) in mod._TRANSACTION_TO_VAR if s == sector]:
        for t, period in enumerate(_QUARTERS):
            out.append({
                "FREQ": "Q", "ADJUSTMENT": adjustment, "REF_AREA": code,
                "SECTOR": sec, "COUNTERPART_SECTOR": "S1", "TRANSACTION": txn,
                "INSTR_ASSET": "_Z", "ACTIVITY": "_Z", "EXPENDITURE": "_Z",
                "UNIT_MEASURE": "XDC", "PRICE_BASE": price_base,
                "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0102",
                "TIME_PERIOD": period, "OBS_VALUE": level * (1.0 + 0.01 * t),
            })
    return out


def _fake(rows):
    def _get(agency_flow, key, start_period, **kw):
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in rows if r["SECTOR"] == sector])
    return _get


@pytest.mark.parametrize("price_base", ["L", "Q"])
def test_both_volume_bases_come_through(monkeypatch, price_base):
    """`L` is chain-linked, `Q` fixed-base. Insisting on `L` silently dropped
    every country that publishes only `Q`."""
    rows = (_rows("AAA", price_base, sector="S1", level=1000.0)
            + _rows("AAA", price_base, sector="S13", level=200.0))
    monkeypatch.setattr(mod, "get_sdmx_csv", _fake(rows))
    got = fetch_qna_expenditure(["AAA"])
    assert not got.empty
    assert set(got["variable"]) == set(mod._TRANSACTION_TO_VAR.values())


def test_a_fixed_base_only_country_is_not_silently_empty(monkeypatch):
    """The regression itself: MEX/ARG/IDN/IND/ZAF publish no `L` at all."""
    rows = (_rows("MEX", "Q", sector="S1", level=1000.0)
            + _rows("MEX", "Q", sector="S13", level=200.0))
    monkeypatch.setattr(mod, "get_sdmx_csv", _fake(rows))
    got = fetch_qna_expenditure(["MEX"])
    assert len(got) == len(_QUARTERS) * len(mod._TRANSACTION_TO_VAR)
    assert "/Q/XDC" in got["source"].iloc[0]


def test_chain_linked_wins_when_a_country_publishes_both(monkeypatch):
    """Preference order, matching qna_panel: `L` where it exists."""
    rows = (_rows("AAA", "L", sector="S1", level=1000.0)
            + _rows("AAA", "Q", sector="S1", level=7777.0)
            + _rows("AAA", "L", sector="S13", level=200.0)
            + _rows("AAA", "Q", sector="S13", level=999.0))
    monkeypatch.setattr(mod, "get_sdmx_csv", _fake(rows))
    got = fetch_qna_expenditure(["AAA"])
    gdp = got[got["variable"] == "log_gdp_real"]["value"]
    assert np.exp(gdp.iloc[0]) == pytest.approx(1000.0)     # the L level
    assert all("/L/XDC" in s for s in got["source"])


def test_the_base_is_chosen_per_country_never_mixed_within_one(monkeypatch):
    """A country served `L` for one variable and `Q` for another would carry a
    level shift inside a series — the defect that took a sibling producer out
    of build_panel."""
    rows = (_rows("AAA", "L", sector="S1", level=1000.0)      # L for S1 only
            + _rows("AAA", "Q", sector="S13", level=999.0)    # Q for S13 only
            + _rows("BBB", "Q", sector="S1", level=500.0)
            + _rows("BBB", "Q", sector="S13", level=100.0))
    monkeypatch.setattr(mod, "get_sdmx_csv", _fake(rows))
    got = fetch_qna_expenditure(["AAA", "BBB"])
    for code, expected in (("AAA", "/L/XDC"), ("BBB", "/Q/XDC")):
        sub = got[got["code"] == code]
        assert sub["source"].map(lambda s: expected in s).all(), code
    # AAA published no L government series, so it must be absent rather than
    # backfilled from the other base
    assert "log_govcon_real" not in set(got[got["code"] == "AAA"]["variable"])


def test_non_volume_price_bases_are_still_excluded(monkeypatch):
    """`V` is current prices and `LR`/`QR` are indices; none is a volume level."""
    rows = (_rows("AAA", "V", sector="S1", level=1000.0)
            + _rows("AAA", "LR", sector="S1", level=100.0))
    monkeypatch.setattr(mod, "get_sdmx_csv", _fake(rows))
    assert fetch_qna_expenditure(["AAA"]).empty


def test_the_fixture_would_have_failed_before_the_fix(monkeypatch):
    """Guards the guard: pin that the Q-only fixture really is Q-only, so the
    regression test above cannot pass for the wrong reason."""
    rows = _rows("MEX", "Q", sector="S1", level=1000.0)
    assert {r["PRICE_BASE"] for r in rows} == {"Q"}
    assert "L" not in {r["PRICE_BASE"] for r in rows}


# The three below were found by `python tools/mutation_check.py
# puremacro/fetch/oecd_qna_expenditure.py`: each guard could be deleted with the
# suite still green, because no fixture above ever produced the condition.

def test_codes_none_asks_for_every_reference_area(monkeypatch):
    """`codes=None` takes a different branch — one unfiltered request per
    sector — and nothing above exercised it."""
    seen: list[str] = []

    def _get(agency_flow, key, start_period, **kw):
        seen.append(key)
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("AAA", "L", sector="S1", level=1000.0)
                             + _rows("AAA", "L", sector="S13", level=200.0)
                             if r["SECTOR"] == sector])

    monkeypatch.setattr(mod, "get_sdmx_csv", _get)
    got = fetch_qna_expenditure(None)
    assert not got.empty
    assert seen and all(k.split(".")[2] == "" for k in seen), \
        f"codes=None must leave REF_AREA open, got {seen}"


def test_an_empty_response_yields_an_empty_frame_not_an_exception(monkeypatch):
    """Every fetcher here degrades to 'this source had nothing'."""
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        lambda *a, **k: pd.DataFrame())
    got = fetch_qna_expenditure(["AAA"])
    assert got.empty and list(got.columns) == list(mod._EMPTY.columns)


def test_a_response_missing_required_columns_is_refused(monkeypatch):
    """A schema change at the source must not produce a half-built frame."""
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        lambda *a, **k: pd.DataFrame({"REF_AREA": ["AAA"],
                                                      "TIME_PERIOD": ["2019-Q1"]}))
    got = fetch_qna_expenditure(["AAA"])
    assert got.empty and list(got.columns) == list(mod._EMPTY.columns)
