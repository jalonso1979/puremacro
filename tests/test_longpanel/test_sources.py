"""Source parsers and the qna_long_panel entry point, all offline.

The payload fixtures reproduce the properties that actually break these
parsers rather than idealised data: INE's millisecond-epoch ``Fecha``
that must be ignored, its null observations, the peseta workbook whose
year appears only on Q1 rows, and ESRI's cp932 encoding with GDP in a
different column depending on SNA vintage.
"""
from __future__ import annotations

import io
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.longpanel import esri_jp as JP
from puremacro.fetch.longpanel import ine_es as ES
from puremacro.fetch.longpanel.panel import (
    KNOWN_GAPS,
    LONG_PANEL_COLUMNS,
    qna_long_panel,
)


# ---------------------------------------------------------------------------
# INE JSON
# ---------------------------------------------------------------------------
def _ine_payload(rows, cod="CTA1527"):
    return {"COD": cod, "Nombre": "Precios corrientes...",
            "Unidad": {"Nombre": "Euros"}, "Escala": {"Nombre": "Millones"},
            "Data": [{"Fecha": 1096581600000,
                      "Periodo": {"Valor": q}, "Anyo": y,
                      "NombrePeriodo": f"{y}T{q}", "Valor": v}
                     for y, q, v in rows]}


def test_ine_series_dates_come_from_anyo_and_periodo():
    """`Fecha` is a local-timezone ms epoch of the quarter START and
    shifts a day across DST; the year/quarter fields are authoritative.
    Every row here carries the SAME bogus Fecha, so a parser reading it
    would collapse them all onto one date."""
    s = ES.parse_ine_series(_ine_payload([(1980, 1, 22933.0),
                                          (1980, 2, 23470.0),
                                          (1981, 3, 25000.0)]))
    assert list(s.index) == [pd.Timestamp("1980-01-01"),
                             pd.Timestamp("1980-04-01"),
                             pd.Timestamp("1981-07-01")]
    assert s.iloc[0] == 22933.0


def test_ine_series_drops_nulls_rather_than_zero_filling():
    s = ES.parse_ine_series(_ine_payload([(1980, 1, None), (1980, 2, 5.0)]))
    assert len(s) == 1 and s.iloc[0] == 5.0


def test_ine_series_accepts_raw_json_text():
    s = ES.parse_ine_series(json.dumps(_ine_payload([(1990, 4, 1.5)])))
    assert s.iloc[0] == 1.5


def test_ine_series_on_empty_payload():
    assert ES.parse_ine_series({"COD": "X", "Data": []}).empty


def test_cons_hh_sums_households_and_npish():
    """OECD's cons_hh is sector S1M = households + NPISH. Mapping only
    the households series would understate Spanish consumption by the
    whole non-profit sector while looking entirely plausible."""
    assert ES.INE_BASE1995_SERIES["cons_hh"] == ("CTA1525", "CTA1523")
    assert len(ES.INE_BASE1995_SERIES["cons_hh"]) == 2


def test_construction_is_not_mapped_onto_the_modern_structures_column():
    """Base-1995 publishes equipment / construction / other, where the
    modern accounts separate dwellings from other structures. Mapping
    construction onto `inv_struct` would silently bury dwellings in it."""
    assert "inv_constr" in ES.INE_BASE1995_SERIES
    assert "inv_struct" not in ES.INE_BASE1995_SERIES
    assert "inv_dwell" not in ES.INE_BASE1995_SERIES


# ---------------------------------------------------------------------------
# INE base-1986 workbook
# ---------------------------------------------------------------------------
def _cntrb86_workbook():
    """A workbook shaped like cntrb86.xls: 8 title/header rows, the year
    only on Q1 rows, values in thousands of millions of pesetas."""
    pytest.importorskip("openpyxl")
    ncols = 19
    rows = [[None] * ncols for _ in range(8)]
    rows[6][2] = "Producto Interior"
    data = [
        [1970, 1, 642.7, 407.9, 433.9, 63.0, 176.4, 76.1, 100.3, 8.99,
         656.29, 79.174, 39.2, 11.074, 28.9, 92.764, 85.2, 4.664, 2.9],
        [None, 2, 649.8, 419.3, 445.9, 61.4, 171.7, 73.7, 98.0, 5.934,
         658.334, 84.025, 43.3, 11.225, 29.5, 92.559, 84.9, 4.759, 2.9],
        [None, 3, 660.4, 431.4, 459.1, 61.2, 168.4, 71.5, 96.9, 3.122,
         664.122, 89.925, 47.7, 11.525, 30.7, 93.647, 85.7, 4.947, 3.0],
        [None, 4, 670.0, 440.0, 465.0, 62.0, 170.0, 72.0, 98.0, 4.0,
         676.0, 92.0, 48.0, 12.0, 31.0, 98.0, 90.0, 5.0, 3.0],
        [1971, 1, 690.0, 452.0, 478.0, 64.0, 175.0, 74.0, 101.0, 5.0,
         696.0, 95.0, 50.0, 12.0, 32.0, 101.0, 93.0, 5.0, 3.0],
    ]
    df = pd.DataFrame(rows + data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=ES.CNTRB86_SHEET, header=False, index=False)
    return buf.getvalue()


def test_cntrb86_carries_the_year_forward():
    """The year appears only on the Q1 row; without a forward fill the
    Q2-Q4 rows are dropped and three quarters in four vanish."""
    out = ES.parse_cntrb86_workbook(_cntrb86_workbook())
    assert list(out.index[:5]) == [
        pd.Timestamp("1970-01-01"), pd.Timestamp("1970-04-01"),
        pd.Timestamp("1970-07-01"), pd.Timestamp("1970-10-01"),
        pd.Timestamp("1971-01-01")]


def test_cntrb86_converts_pesetas_to_millions_of_euro():
    out = ES.parse_cntrb86_workbook(_cntrb86_workbook())
    assert out["gdp"].iloc[0] == pytest.approx(642.7 * 1000 / ES.ESP_PER_EUR)
    assert ES.ESP_PER_EUR == pytest.approx(166.386)


def test_cntrb86_expenditure_identity_closes():
    """Verified against the real file at 1970Q1:
    407.9 + 63 + 176.4 + 8.99 + 79.174 - 92.764 = 642.7. If the column
    map drifts, this is what catches it."""
    out = ES.parse_cntrb86_workbook(_cntrb86_workbook())
    resid = out["gdp"] - (out["cons_hh"] + out["cons_gov"] + out["capform"]
                          + out["exports"] - out["imports"])
    np.testing.assert_allclose(resid.to_numpy(), 0.0, atol=1e-6)


def test_cntrb86_derives_capform_from_gfcf_plus_inventories():
    out = ES.parse_cntrb86_workbook(_cntrb86_workbook())
    np.testing.assert_allclose(
        out["capform"].to_numpy(),
        (out["inv"] + out["inventories"]).to_numpy())


def test_cntrb86_uses_national_not_interior_consumption():
    """Column 3 is 'consumo privado nacional' (residents); column 4 is
    'interior'. Only the national concept closes the identity."""
    assert ES.CNTRB86_COLUMNS["cons_hh"] == 3


# ---------------------------------------------------------------------------
# ESRI CSV
# ---------------------------------------------------------------------------
def _esri_csv(layout: str) -> bytes:
    """cp932-encoded CSV shaped like an ESRI release."""
    if layout == "modern":
        head = ['名目季節調整系列'] + [''] * 13
        cols = ['', 'GDP(Expenditure Approach)', 'PrivateConsumption',
                'Consumption ofHouseholds', 'ExcludingImputed Rent',
                'PrivateResidentialInvestment', 'Private Non-Resi.Investment',
                'Changein PrivateInventories', 'GovernmentConsumption',
                'PublicInvestment', 'Changein PublicInventories',
                'Net Exports', 'Exports', 'Imports']
        # gdp, pc, hh, ex-rent, resid, nonresi, privinv, gov, pubinv,
        # pubinv-chg, netex, exports, imports
        rows = [
            ['1994/ 1- 3.', '100', '60', '55', '50', '5', '15', '1',
             '10', '8', '1', '0', '12', '12'],
            # 62 + 10 + (5+16+8) + (1+1) + 13 - 13 = 103
            ['4- 6.', '103', '62', '57', '52', '5', '16', '1',
             '10', '8', '1', '0', '13', '13'],
        ]
    else:
        head = ['名目季節調整系列'] + [''] * 11
        cols = ['', 'Private Consumption', 'Residential Investment',
                'Non-Resi. Investment', 'Private Inventory',
                'Government Consumption', 'Public Investment',
                'Public Inventory', 'Net Exports', 'Exports', 'Imports',
                'GDE(=GDP)']
        rows = [
            ['1955/ 4- 6.', '60', '5', '15', '1', '10', '8', '1',
             '0', '12', '12', '100'],
            [' 7- 9.', '62', '5', '16', '1', '10', '8', '1',
             '0', '13', '13', '103'],
        ]
    lines = [head, [''] * len(cols), cols, [''] * len(cols)] + rows
    buf = io.StringIO()
    csv_writer = __import__("csv").writer(buf)
    for r in lines:
        csv_writer.writerow(r)
    return buf.getvalue().encode("cp932")


def test_esri_requires_cp932_not_utf8():
    raw = _esri_csv("modern")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert not JP.parse_esri_csv(raw, "modern").empty


@pytest.mark.parametrize("layout", ["modern", "sna68"])
def test_esri_finds_gdp_in_the_right_column(layout):
    """GDP is column 1 in the modern files and column 11 in the 68SNA
    ones. Both fixtures put 100 in GDP and 60 in private consumption, so
    reading the wrong column returns 60 and looks plausible."""
    out = JP.parse_esri_csv(_esri_csv(layout), layout)
    assert out["gdp"].iloc[0] == 100.0
    assert out["cons_hh"].iloc[0] == 60.0


def test_esri_wrong_layout_is_not_silently_tolerated():
    """Parsing a 68SNA file as modern must not quietly return
    consumption as GDP."""
    out = JP.parse_esri_csv(_esri_csv("sna68"), "modern")
    assert out.empty or out["gdp"].iloc[0] != 100.0


def test_esri_unknown_layout_raises():
    with pytest.raises(ValueError, match="unknown ESRI layout"):
        JP.parse_esri_csv(_esri_csv("modern"), "nope")


def test_esri_carries_the_year_forward_across_bare_month_rows():
    out = JP.parse_esri_csv(_esri_csv("modern"), "modern")
    assert list(out.index) == [pd.Timestamp("1994-01-01"),
                               pd.Timestamp("1994-04-01")]


def test_esri_tolerates_inconsistent_leading_whitespace():
    """' 7- 9.' in one vintage vs '7- 9.' in another."""
    out = JP.parse_esri_csv(_esri_csv("sna68"), "sna68")
    assert list(out.index) == [pd.Timestamp("1955-04-01"),
                               pd.Timestamp("1955-07-01")]


def test_esri_stops_at_a_repeated_quarter():
    """The 68SNA CSVs carry more than one table; concatenating the second
    onto the first would double the series."""
    raw = _esri_csv("sna68").decode("cp932")
    doubled = (raw + raw).encode("cp932")
    out = JP.parse_esri_csv(doubled, "sna68")
    assert len(out) == 2


def test_esri_identity_closes_on_both_layouts():
    for layout in ("modern", "sna68"):
        out = JP.parse_esri_csv(_esri_csv(layout), layout)
        resid = out["gdp"] - (out["cons_hh"] + out["cons_gov"]
                              + out["capform"] + out["exports"]
                              - out["imports"])
        np.testing.assert_allclose(resid.to_numpy(), 0.0, atol=1e-9)


def test_the_japanese_chain_is_ordered_newest_first():
    assert JP.JP_DEFAULT_CHAIN == ("jp_sna93", "jp_sna68")


# ---------------------------------------------------------------------------
# qna_long_panel, driven from a synthetic spine (no network)
# ---------------------------------------------------------------------------
def _spine(code="ESP", start="1995Q1", end="2005Q4"):
    idx = pd.period_range(start, end, freq="Q").to_timestamp()
    n = len(idx)
    gdp = pd.Series(100 * np.exp(np.cumsum(np.full(n, 0.01))), index=idx)
    df = pd.DataFrame({
        "gdp": gdp, "cons_hh": gdp * 0.6, "cons_gov": gdp * 0.2,
        "inv": gdp * 0.2, "capform": gdp * 0.22,
        "exports": gdp * 0.3, "imports": gdp * 0.32,
    })
    df.index.name = "date"
    df["code"] = code
    return df.reset_index().set_index(["code", "date"])


def test_a_country_with_no_archived_source_passes_through_unchanged():
    out = qna_long_panel(["ITA"], spine=_spine("ITA"))
    assert len(out) == len(_spine("ITA"))
    assert set(out["src_gdp"].unique()) == {"oecd"}


def test_output_carries_the_qna_panel_schema_plus_provenance():
    out = qna_long_panel(["ITA"], spine=_spine("ITA"))
    for col in LONG_PANEL_COLUMNS:
        assert col in out.columns
        assert f"src_{col}" in out.columns
    assert out.index.names == ["code", "date"]


def test_return_seams_gives_a_tidy_table():
    out, seams = qna_long_panel(["ITA"], spine=_spine("ITA"),
                                return_seams=True)
    assert list(seams.columns)[:5] == ["code", "column", "older", "newer",
                                       "date"]


def test_known_gaps_records_why_a_country_was_not_extended():
    """'We checked and it buys nothing' is a different statement from
    'we did not check', and the difference must survive in the code."""
    for code in ("DEU", "MEX", "BRA", "CHL"):
        assert code in KNOWN_GAPS
        assert len(KNOWN_GAPS[code]) > 40
    assert "West Germany" in KNOWN_GAPS["DEU"]


def test_an_empty_spine_for_a_country_warns_and_skips():
    empty = _spine("ITA").iloc[0:0]
    with pytest.warns(UserWarning, match="nothing usable"):
        out = qna_long_panel(["ITA"], spine=empty)
    assert out.empty
