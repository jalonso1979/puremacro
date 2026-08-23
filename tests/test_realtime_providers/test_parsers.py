"""Pure-parser tests for the five non-ALFRED real-time providers.

Every one of these takes bytes and returns a frame, so they run with
no network. The payload fixtures are trimmed copies of real responses,
kept byte-faithful in the parts that matter: the BOM and ``DD.MM.YYYY``
headers of the Bundesbank matrix, StatCan's ``"May 29, 2026"`` Release
labels and ``YYYY-MM`` quarter starts, the OECD's ``YYYYMM`` editions,
the ECB's ``Replace``/``Delete`` rows.

Each provider gets at least one test aimed at its specific documented
trap, because those are the failures that produce a well-formed wrong
answer rather than an exception.
"""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.fetch.realtime import bundesbank as BBK
from puremacro.fetch.realtime import ecb_rtd as ECB
from puremacro.fetch.realtime import oecd_stes as OECD
from puremacro.fetch.realtime import ons as ONS
from puremacro.fetch.realtime import statcan as SC


# ---------------------------------------------------------------------------
# Statistics Canada
# ---------------------------------------------------------------------------
STATCAN_CSV = (
    '﻿"REF_DATE","GEO","DGUID","Prices","Seasonal adjustment","Estimates",'
    '"Release","UOM","UOM_ID","SCALAR_FACTOR","SCALAR_ID","VECTOR",'
    '"COORDINATE","VALUE","STATUS","SYMBOL","TERMINATED","DECIMALS"\n'
    # target rows: two editions of 2025Q1
    '"2025-01","Canada","","Chained (2017) dollars","Seasonally adjusted at annual rates",'
    '"Gross domestic product at market prices","May 30, 2025","Dollars","81","millions","6","v1","1.1.1.30.51","2456239","","","","0"\n'
    '"2025-01","Canada","","Chained (2017) dollars","Seasonally adjusted at annual rates",'
    '"Gross domestic product at market prices","May 29, 2026","Dollars","81","millions","6","v2","1.1.1.30.55","2502038","","","","0"\n'
    # not yet published in the Feb 2025 edition
    '"2025-01","Canada","","Chained (2017) dollars","Seasonally adjusted at annual rates",'
    '"Gross domestic product at market prices","February 28, 2025","Dollars","81","millions","6","v3","1.1.1.30.50","","..","","","0"\n'
    # wrong Prices member — a percentage series, must not leak in
    '"2025-01","Canada","","Contributions to percent change","Seasonally adjusted at annual rates",'
    '"Gross domestic product at market prices","May 29, 2026","Percent","239","units","0","v4","1.1.3.30.55","0.4","","","","1"\n'
    # wrong Estimates member
    '"2025-01","Canada","","Chained (2017) dollars","Seasonally adjusted at annual rates",'
    '"Household final consumption expenditure","May 29, 2026","Dollars","81","millions","6","v5","1.1.1.2.55","1400000","","","","0"\n'
    # wrong seasonal adjustment member
    '"2025-01","Canada","","Chained (2017) dollars","Unadjusted",'
    '"Gross domestic product at market prices","May 29, 2026","Dollars","81","millions","6","v6","1.1.2.30.55","2400000","","","","0"\n'
)


def _statcan(**kw):
    kw.setdefault("estimate", "Gross domestic product at market prices")
    kw.setdefault("prices", "Chained (2017) dollars")
    return SC.parse_statcan_vintage_csv(STATCAN_CSV, **kw)


def test_statcan_keeps_only_the_requested_slice():
    out = _statcan()
    assert len(out) == 2
    assert set(out["value"]) == {2456239.0, 2502038.0}


def test_statcan_rejects_the_percentage_prices_member():
    """The Prices dimension carries contributions-to-percent-change
    members alongside the level ones; 0.4 must never reach the panel."""
    out = _statcan()
    assert 0.4 not in set(out["value"])


def test_statcan_rejects_the_unadjusted_seasonal_member():
    out = _statcan()
    assert 2400000.0 not in set(out["value"])


def test_statcan_release_label_becomes_the_vintage():
    out = _statcan()
    assert set(out["vintage"]) == {pd.Timestamp("2025-05-30"),
                                   pd.Timestamp("2026-05-29")}


def test_statcan_ref_date_is_a_quarter_not_a_month():
    """'2025-01' is 2025Q1 and '2025-04' is 2025Q2 — consecutive
    quarters, three months apart. A fixture with one REF_DATE cannot
    tell a correct quarter parse from a monthly misreading, so this
    payload carries two."""
    head, _, rest = STATCAN_CSV.partition("\n")
    row = ('"2025-04","Canada","","Chained (2017) dollars",'
           '"Seasonally adjusted at annual rates",'
           '"Gross domestic product at market prices","May 29, 2026",'
           '"Dollars","81","millions","6","v9","1.1.1.30.55","2510000",'
           '"","","","0"\n')
    out = SC.parse_statcan_vintage_csv(
        head + "\n" + rest + row,
        estimate="Gross domestic product at market prices",
        prices="Chained (2017) dollars")
    dates = sorted(set(out["date"]))
    assert dates == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")]
    assert (dates[1] - dates[0]).days == 90


def test_statcan_unpublished_cells_are_dropped_not_zeroed():
    out = _statcan()
    assert pd.Timestamp("2025-02-28") not in set(out["vintage"])
    assert (out["value"] > 0).all()


def test_statcan_missing_columns_raise():
    with pytest.raises(ValueError, match="missing expected columns"):
        SC.parse_statcan_vintage_csv(
            "REF_DATE,VALUE\n2025-01,1\n",
            estimate="x", prices="y")


def test_statcan_empty_payload():
    assert SC.parse_statcan_vintage_csv(b"", estimate="x", prices="y").empty


# ---------------------------------------------------------------------------
# Bundesbank
# ---------------------------------------------------------------------------
BBK_CSV = (
    '﻿"",30.01.2026,25.02.2026,40.02.1997\n'
    '"",,,\n'
    'Baseyear,2020,2020,1991\n'
    'Decimals,2,2,2\n'
    'unit,2020=100,2020=100,DEM\n'
    '1991-Q1,95.10,95.20,80.00\n'
    '2025-Q1,104.86,104.87,\n'
)


def test_bbk_matrix_parses_to_long():
    out = BBK.parse_bbk_rtd_csv(BBK_CSV)
    assert list(out.columns) == ["date", "vintage", "value"]
    assert set(out["date"]) == {pd.Timestamp("1991-01-01"),
                               pd.Timestamp("2025-01-01")}


def test_bbk_attribute_rows_are_not_mistaken_for_data():
    """'Baseyear,2020,2020,1991' must not become an observation.

    Attribute rows are ragged and their label set varies by series, so
    they are classified by the shape of column 0, not by row index.
    """
    out = BBK.parse_bbk_rtd_csv(BBK_CSV)
    assert 2020.0 not in set(out["value"])
    assert len(out) == 5


def test_bbk_unknown_day_sentinels_map_to_month_start():
    ts, exact = BBK.parse_bbk_vintage_date("40.02.1997")
    assert ts == pd.Timestamp("1997-02-01")
    assert exact is False
    ts2, exact2 = BBK.parse_bbk_vintage_date("30.07.2026")
    assert ts2 == pd.Timestamp("2026-07-30")
    assert exact2 is True


def test_bbk_sentinel_vintage_survives_into_the_frame():
    """A naive %d.%m.%Y parse would raise on '40.02.1997' and a lenient
    one would drop the whole edition."""
    out = BBK.parse_bbk_rtd_csv(BBK_CSV)
    assert pd.Timestamp("1997-02-01") in set(out["vintage"])


def test_bbk_blank_cells_are_the_ragged_edge_not_zero():
    out = BBK.parse_bbk_rtd_csv(BBK_CSV)
    row = out[(out["date"] == pd.Timestamp("2025-01-01"))
              & (out["vintage"] == pd.Timestamp("1997-02-01"))]
    assert row.empty


def test_bbk_unparseable_vintage_header_is_skipped():
    csv = '"",junk,30.01.2026\n"",,\n1991-Q1,1.0,2.0\n'
    out = BBK.parse_bbk_rtd_csv(csv)
    assert len(out) == 1
    assert out["value"].iloc[0] == 2.0


@pytest.mark.parametrize("raw", [b"", "   ", b"\n"])
def test_bbk_empty_payload(raw):
    assert BBK.parse_bbk_rtd_csv(raw).empty


# ---------------------------------------------------------------------------
# OECD STES revisions
# ---------------------------------------------------------------------------
OECD_CSV = (
    "DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,EDITION,"
    "TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT,DECIMALS,BASE_PER\n"
    "X,MEX,Q,B1GQ_Q,XDC,_T,202608,2019-Q1,1000.0,A,0,2,\n"
    "X,MEX,Q,B1GQ_Q,XDC,_T,199902,2019-Q1,900.0,A,0,2,\n"
    "X,MEX,Q,B1GQ_Q,XDC,_T,202608,2019-Q2,1010.0,A,0,2,\n"
)


def test_oecd_edition_becomes_a_month_start_vintage():
    out = OECD.parse_stes_revisions_csv(OECD_CSV)
    assert set(out["vintage"]) == {pd.Timestamp("1999-02-01"),
                                   pd.Timestamp("2026-08-01")}


def test_oecd_time_period_becomes_a_quarter_start():
    out = OECD.parse_stes_revisions_csv(OECD_CSV)
    assert set(out["date"]) == {pd.Timestamp("2019-01-01"),
                               pd.Timestamp("2019-04-01")}


def test_oecd_rows_come_back_sorted():
    """SDMX-CSV returns rows in neither edition nor period order.

    Asserting only that `date` is monotonic passes on the unsorted
    input too, because the fixture's dates already happen to be in
    order; the vintage ordering within a date is what actually moves.
    """
    out = OECD.parse_stes_revisions_csv(OECD_CSV)
    assert list(out["value"]) == [900.0, 1000.0, 1010.0]
    assert list(out["vintage"]) == [pd.Timestamp("1999-02-01"),
                                    pd.Timestamp("2026-08-01"),
                                    pd.Timestamp("2026-08-01")]


def test_oecd_key_has_exactly_five_dots():
    """Six dimensions. A miscounted key returns HTTP 422 with no body."""
    key = OECD.build_key("MEX", "B1GQ_Q")
    assert key.count(".") == 5
    assert key == "MEX.Q.B1GQ_Q..."
    assert OECD.build_key("DEU", "B1GQ_Q", edition="202606").count(".") == 5


def test_oecd_missing_columns_raise():
    with pytest.raises(ValueError, match="missing"):
        OECD.parse_stes_revisions_csv("TIME_PERIOD,OBS_VALUE\n2019-Q1,1\n")


def test_oecd_measure_map_covers_the_deflator():
    """The archive carries a genuine GDP deflator, which is what the
    old OECD-MEI catalogue got wrong by pointing at a CPI."""
    assert OECD.OECD_STES_MEASURES["deflator"][0] == "B1GQ_D"
    assert OECD.OECD_STES_MEASURES["gdp_real"][0] == "B1GQ_Q"


def test_oecd_country_list_is_the_verified_42():
    assert len(OECD.OECD_STES_COUNTRIES) == 42
    for c in ("MEX", "IRL", "SVK", "NZL", "ISR", "CHL", "COL", "USA"):
        assert c in OECD.OECD_STES_COUNTRIES


# ---------------------------------------------------------------------------
# ECB RTD
# ---------------------------------------------------------------------------
ECB_CSV = (
    "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,ACTION,VALID_FROM,VALID_TO\n"
    "K,Q,S0,2015-Q1,2434957.11,Replace,2015-06-02T15:30:00.000+02:00,\n"
    "K,Q,S0,2015-Q1,2435111.40,Replace,2015-07-15T15:30:00.000+02:00,\n"
    "K,Q,S0,2015-Q1,,Delete,,2021-01-20T15:30:00.000+01:00\n"
    "K,Q,S0,2015-Q2,2442691.73,Replace,2015-09-02T15:30:00.000+02:00,\n"
)


def test_ecb_history_parses_versions_to_vintages():
    out = ECB.parse_ecb_history_csv(ECB_CSV)
    assert len(out) == 3
    assert set(out["vintage"]) == {
        pd.Timestamp("2015-06-02"), pd.Timestamp("2015-07-15"),
        pd.Timestamp("2015-09-02")}


def test_ecb_delete_rows_are_dropped_without_producing_nat():
    """Delete rows carry an empty VALID_FROM (the stamp is in VALID_TO).
    Parsing them as vintages would yield NaT rows."""
    out = ECB.parse_ecb_history_csv(ECB_CSV)
    assert out["vintage"].notna().all()
    assert out["value"].notna().all()


def test_ecb_withdrawals_are_excluded_even_when_fully_populated():
    """A withdrawal is an event, not an observation.

    In the common payload a Delete row has no OBS_VALUE, so dropna
    would remove it anyway and the ACTION filter never bites. This
    payload gives the Delete row both a value and a VALID_FROM, so the
    filter is the only thing keeping it out.
    """
    csv = (
        "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE,ACTION,VALID_FROM,VALID_TO\n"
        "K,Q,S0,2015-Q1,100.0,Replace,2015-06-02T15:30:00.000+02:00,\n"
        "K,Q,S0,2015-Q1,999.0,Delete,2021-01-20T15:30:00.000+01:00,\n"
    )
    out = ECB.parse_ecb_history_csv(csv)
    assert list(out["value"]) == [100.0]
    assert pd.Timestamp("2021-01-20") not in set(out["vintage"])


def test_ecb_vintage_is_tz_naive_and_date_normalised():
    out = ECB.parse_ecb_history_csv(ECB_CSV)
    assert out["vintage"].dt.tz is None
    assert (out["vintage"] == out["vintage"].dt.normalize()).all()


def test_ecb_without_include_history_raises_rather_than_returning_one_vintage():
    """A plain RTD request silently returns the current vintage only.
    Missing VALID_FROM is the signal that includeHistory was not set."""
    with pytest.raises(ValueError, match="includeHistory"):
        ECB.parse_ecb_history_csv(
            "KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nK,Q,S0,2015-Q1,1.0\n")


def test_ecb_rejects_countries_outside_its_three_areas():
    with pytest.raises(ValueError, match="oecd_stes"):
        ECB.fetch_ecb_rtd_vintages("DEU")


# ---------------------------------------------------------------------------
# ONS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,expected_month,expected_stage", [
    ("Jun-18 [2016 prices]\nQNA", "2018-06-01", "QNA"),
    ("Aug-18\n1st", "2018-08-01", "1st"),
    ("June-26 QNA", "2026-06-01", "QNA"),
    ("Dec-03 [2000 prices]\nQNA", "2003-12-01", "QNA"),
    ("Sep-61 [1954 prices]", "1961-09-01", ""),
    ("Mar-26 QNA", "2026-03-01", "QNA"),
])
def test_ons_vintage_labels(label, expected_month, expected_stage):
    ts, stage = ONS.parse_ons_vintage_label(label)
    assert ts == pd.Timestamp(expected_month)
    assert stage == expected_stage


def test_ons_century_pivot_does_not_come_from_the_sheet_range():
    """'Dec-82' inside the '1983 - 2003' sheet is 1982, not 2082."""
    ts, _ = ONS.parse_ons_vintage_label("Dec-82 [1975 prices]")
    assert ts == pd.Timestamp("1982-12-01")


def test_ons_real_typo_is_rejected_rather_than_guessed():
    assert ONS.parse_ons_vintage_label("Feb-772") == (None, "")
    assert ONS.parse_ons_vintage_label("") == (None, "")
    assert ONS.parse_ons_vintage_label(None) == (None, "")


def test_bbk_two_sentinels_in_one_month_keep_distinct_editions():
    """40 and 41 both mean "day unknown" and both land on the 1st, and a
    real 01.MM release lands there too. Collapsing them would silently
    discard whole editions in the (date, vintage) de-duplication."""
    csv = ('"",41.02.1997,40.02.1997,01.02.1997\n'
           '"",,,\n'
           'Baseyear,1991,1991,1991\n'
           '1991-Q1,10.0,20.0,30.0\n')
    out = BBK.parse_bbk_rtd_csv(csv)
    assert out["vintage"].nunique() == 3
    assert sorted(out["value"]) == [10.0, 20.0, 30.0]
    # The exact date keeps the day the Bundesbank actually published.
    exact = out[out["vintage"] == pd.Timestamp("1997-02-01")]
    assert exact["value"].iloc[0] == 30.0


def test_bbk_duplicate_date_vintage_keeps_the_last_column():
    """Two columns for the same exact edition: the tie-break is
    `keep="last"`, i.e. the rightmost (most recently appended) column."""
    csv = ('"",30.01.2026,30.01.2026\n'
           '"",,\n'
           '1991-Q1,1.0,2.0\n')
    out = BBK.parse_bbk_rtd_csv(csv)
    assert len(out) == 1
    assert out["value"].iloc[0] == 2.0


def test_oecd_duplicate_date_edition_keeps_the_last_row():
    csv = ("DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,EDITION,"
           "TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT,DECIMALS,BASE_PER\n"
           "X,MEX,Q,B1GQ_Q,XDC,_T,202608,2019-Q1,1.0,A,0,2,\n"
           "X,MEX,Q,B1GQ_Q,XDC,_T,202608,2019-Q1,2.0,A,0,2,\n")
    out = OECD.parse_stes_revisions_csv(csv)
    assert len(out) == 1
    assert out["value"].iloc[0] == 2.0


def test_oecd_keep_country_splits_a_batched_response():
    """Batched multi-country requests come back in one payload and have
    to be split by REF_AREA; without keep_country they would be silently
    concatenated into one bogus series."""
    csv = ("DATAFLOW,REF_AREA,FREQ,MEASURE,UNIT_MEASURE,ACTIVITY,EDITION,"
           "TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT_MULT,DECIMALS,BASE_PER\n"
           "X,DEU,Q,B1GQ_Q,XDC,_T,202608,2019-Q1,10.0,A,0,2,\n"
           "X,ESP,Q,B1GQ_Q,XDC,_T,202608,2019-Q1,20.0,A,0,2,\n")
    out = OECD.parse_stes_revisions_csv(csv, keep_country=True)
    assert list(out.columns) == ["country", "date", "vintage", "value"]
    assert set(out["country"]) == {"DEU", "ESP"}
    assert out[out["country"] == "ESP"]["value"].iloc[0] == 20.0
    with pytest.raises(ValueError, match="REF_AREA"):
        OECD.parse_stes_revisions_csv(
            "EDITION,TIME_PERIOD,OBS_VALUE\n202608,2019-Q1,1.0\n",
            keep_country=True)


# ---------------------------------------------------------------------------
# ONS workbook — the actual data path, not just the label regex
# ---------------------------------------------------------------------------
def _ons_workbook(rows, header):
    """Build a minimal ONS-shaped workbook in memory."""
    openpyxl = pytest.importorskip("openpyxl")
    import io
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "2018 - "
    ws.append(["Real-time database for GDP"])         # row 1
    ws.append(["This worksheet contains one table. "])  # row 2
    ws.append(["Source: Office for National Statistics"])  # row 3
    ws.append(["Publication date and time period "] + header)  # row 4
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_ons_workbook_parses_periods_and_vintages():
    raw = _ons_workbook(
        rows=[["Q1 1955", 111480, 111500], ["Q2 1955", 111586, 111600]],
        header=["Aug-18\n1st", "Sep-18\nQNA"],
    )
    out = ONS.parse_ons_realtime_workbook(raw)
    assert set(out["date"]) == {pd.Timestamp("1955-01-01"),
                                pd.Timestamp("1955-04-01")}
    assert sorted(out["stage"].unique()) == ["1st", "QNA"]
    assert out["value"].max() == 111600


def test_ons_same_month_stages_get_distinct_ordered_vintages():
    """Two stages in one month share a publication month and would
    collapse to one edition; the day offset keeps them distinct and in
    workbook (chronological) order."""
    raw = _ons_workbook(
        rows=[["Q1 1955", 100, 200]],
        header=["Aug-18\n1st", "Aug-18\nQNA"],
    )
    out = ONS.parse_ons_realtime_workbook(raw).sort_values("vintage")
    assert len(out) == 2
    assert out["vintage"].nunique() == 2
    # Left-to-right in the sheet is earliest-to-latest.
    assert list(out["value"]) == [100, 200]
    assert list(out["stage"]) == ["1st", "QNA"]


def test_ons_workbook_reports_unparseable_headers_instead_of_guessing():
    raw = _ons_workbook(
        rows=[["Q1 1955", 100, 200]],
        header=["Feb-772", "Sep-18\nQNA"],
    )
    out = ONS.parse_ons_realtime_workbook(raw)
    assert out.attrs["unparsed_headers"] == ["Feb-772"]
    assert len(out) == 1
    assert out["value"].iloc[0] == 200


def test_ons_workbook_skips_metadata_sheets_and_blank_cells():
    raw = _ons_workbook(
        rows=[["Q1 1955", 100, None], ["not a period", 1, 2]],
        header=["Aug-18\n1st", "Sep-18\nQNA"],
    )
    out = ONS.parse_ons_realtime_workbook(raw)
    assert len(out) == 1
    assert out["value"].iloc[0] == 100
