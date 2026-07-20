"""Tests for puremacro.fetch.labor.fetch_oecd_lfs_panel."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.labor import _build_lfs_key  # noqa: PLC2701 — internal API tested


_KEY_RE = re.compile(
    r"^(?P<freq>[MQA]?)\."
    r"(?P<area>[A-Z+]*)\."
    r"(?P<measure>[A-Z+_]*)\."
    r"(?P<unit>[A-Z+_]*)\."
    r"(?P<transform>[A-Z+_]*)\."
    r"(?P<adjust>[A-Z+_]*)\."
    r"(?P<sex>[A-Z+_]*)\."
    r"(?P<age>[A-Z0-9+_T]*)\."
    r"(?P<activity>[A-Z+_0-9]*)$"
)


def test_default_key_skeleton_has_nine_dot_positions():
    key = _build_lfs_key(
        dataflow="DSD_LFS@DF_IALFS_INDIC",
        countries=None,
        sexes=("_T", "M", "F"),
        ages=("Y_GE15", "Y15T24"),
        measures=("WAP", "LF", "EMP", "UNE", "IPOP"),
        activities=None,
        frequency="M",
        adjustment="Y",
    )
    assert key.count(".") == 8  # 9 positions
    m = _KEY_RE.match(key)
    assert m is not None, f"key {key!r} did not match expected schema"
    assert m["freq"] == "M"
    assert m["area"] == ""
    assert m["measure"] == "WAP+LF+EMP+UNE+IPOP"
    assert m["adjust"] == "Y"
    assert m["sex"] == "_T+M+F"
    assert m["age"] == "Y_GE15+Y15T24"
    assert m["activity"] == ""


def test_country_filter_joins_with_plus():
    key = _build_lfs_key(
        dataflow="DSD_LFS@DF_IALFS_INDIC",
        countries=["USA", "MEX"],
        sexes=("M",),
        ages=("Y_GE15",),
        measures=("UNE",),
        activities=None,
        frequency="M",
        adjustment="Y",
    )
    m = _KEY_RE.match(key)
    assert m["area"] == "USA+MEX"
    assert m["sex"] == "M"


def test_unknown_dataflow_raises():
    with pytest.raises(ValueError, match="dataflow"):
        _build_lfs_key(
            dataflow="DSD_LFS@DF_NOT_A_REAL_DATAFLOW",
            countries=None, sexes=("_T",), ages=("Y_GE15",),
            measures=("EMP",), activities=None,
            frequency="M", adjustment="Y",
        )


from puremacro.fetch.labor import _reshape_lfs_csv  # noqa: PLC2701


_FIXTURE = Path(__file__).parent / "fixtures" / "labor" / "synthetic_indic.csv"


def test_reshape_canonical_schema():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_lfs_csv(raw, frequency="M")
    expected_cols = {"code", "date", "sex", "age",
                     "wa", "lf", "emp", "une", "ina",
                     "urate", "epop", "prate"}
    assert set(out.columns) == expected_cols
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out.dtypes["wa"] == np.float64
    assert len(out) == 2  # 2 (code, sex, age) cells × 1 month


def test_reshape_rates_match_definitions():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_lfs_csv(raw, frequency="M").set_index(["code", "sex", "age"])
    usa = out.loc[("USA", "_T", "Y_GE15")]
    np.testing.assert_allclose(usa["urate"], 6000 / 164000, rtol=1e-12)
    np.testing.assert_allclose(usa["epop"], 158000 / 260000, rtol=1e-12)
    np.testing.assert_allclose(usa["prate"], 164000 / 260000, rtol=1e-12)


def test_reshape_synthesizes_ina_when_ipop_missing():
    """MEX/F/Y15T24 has WAP/LF/EMP/UNE but no IPOP — fetcher fills ina = wa - lf."""
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_lfs_csv(raw, frequency="M").set_index(["code", "sex", "age"])
    mex = out.loc[("MEX", "F", "Y15T24")]
    np.testing.assert_allclose(mex["ina"], 11000 - 4000, rtol=1e-12)


def test_reshape_drops_off_frequency_rows():
    """If FREQ != requested, rows are dropped before reshape."""
    raw = pd.read_csv(_FIXTURE).copy()
    raw.loc[0, "FREQ"] = "Q"  # mark first row as quarterly
    out = _reshape_lfs_csv(raw, frequency="M")
    # USA _T Y_GE15 should now lack WAP → row dropped or wa=NaN
    usa = out[(out["code"] == "USA") & (out["sex"] == "_T")]
    if len(usa) > 0:
        assert pd.isna(usa.iloc[0]["wa"])


from puremacro.fetch.labor import _read_local_input  # noqa: PLC2701


def _write_labor_xlsx(path: Path) -> None:
    df = pd.DataFrame({
        "Code":   ["AUS", "AUS"],
        "Period": pd.to_datetime(["1995-01-01", "1995-02-01"]),
        "Sex":    ["Female", "Female"],
        "Age":    ["15-24", "15-24"],
        "emp":    [382.0, 384.0],
        "lf":     [900.0, 902.0],
        "olf":    [415.0, 416.0],
        "une":    [135.0, 134.0],
    })
    df.to_excel(path, sheet_name="OECD-AGE-M", index=False)


def test_excel_path_sheet_syntax(tmp_path: Path):
    xlsx = tmp_path / "fake_labor.xlsx"
    _write_labor_xlsx(xlsx)
    raw = _read_local_input(f"{xlsx}::OECD-AGE-M")
    # Adapter normalizes columns to SDMX convention.
    assert "REF_AREA" in raw.columns
    assert "MEASURE" in raw.columns
    assert "OBS_VALUE" in raw.columns
    assert "FREQ" in raw.columns
    # 2 dates × 4 measures (emp/lf/une + synth ina) = 8 rows
    assert len(raw) >= 6
    # Sex labels normalized.
    assert set(raw["SEX"].unique()) <= {"_T", "M", "F"}


def test_excel_unknown_sheet_raises(tmp_path: Path):
    xlsx = tmp_path / "fake_labor.xlsx"
    _write_labor_xlsx(xlsx)
    with pytest.raises(KeyError, match="OECD-AGE-M"):
        _read_local_input(f"{xlsx}::DOES-NOT-EXIST")


def test_csv_path_returns_raw_sdmx(tmp_path: Path):
    """Plain .csv path should be passed through to read_csv with no munging."""
    raw = _read_local_input(str(_FIXTURE))
    assert "MEASURE" in raw.columns
    assert raw["MEASURE"].iloc[0] in {"WAP", "LF", "EMP", "UNE", "IPOP"}


from puremacro.fetch.labor import fetch_oecd_lfs_panel


def test_fetch_from_local_csv():
    out = fetch_oecd_lfs_panel(csv_path=str(_FIXTURE))
    assert {"code", "date", "sex", "age", "urate", "epop", "prate"} <= set(out.columns)
    assert len(out) > 0


def test_fetch_cache_roundtrip(tmp_path: Path):
    cache = tmp_path / "labor_cache.parquet"
    a = fetch_oecd_lfs_panel(csv_path=str(_FIXTURE), cache_path=str(cache))
    assert cache.exists()
    # Second call: cache hit, csv_path NOT consulted.
    b = fetch_oecd_lfs_panel(csv_path=str(tmp_path / "missing.csv"), cache_path=str(cache))
    pd.testing.assert_frame_equal(a, b)


def test_fetch_refresh_flag_bypasses_cache(tmp_path: Path):
    cache = tmp_path / "labor_cache.parquet"
    fetch_oecd_lfs_panel(csv_path=str(_FIXTURE), cache_path=str(cache))
    # With refresh=True, missing csv_path should now error.
    with pytest.raises(FileNotFoundError):
        fetch_oecd_lfs_panel(
            csv_path=str(tmp_path / "still_missing.csv"),
            cache_path=str(cache),
            refresh=True,
        )


@pytest.mark.network
def test_fetch_live_skip_on_empty():
    """Live SDMX pull for one country / one measure. Skip on empty per project rule."""
    try:
        out = fetch_oecd_lfs_panel(
            countries=["USA"], sexes=("_T",), ages=("Y_GE15",),
            measures=("EMP",), activities=None, frequency="M",
        )
    except (RuntimeError, OSError):
        pytest.skip("OECD SDMX unreachable for USA/_T/Y_GE15/EMP — endpoint may be down")
    if out.empty:
        pytest.skip("OECD SDMX returned empty for USA/_T/Y_GE15/EMP — endpoint may be down")
    assert "emp" in out.columns
    assert (out["code"] == "USA").all()


from puremacro.fetch.labor import _synthesize_age_aggregates  # noqa: PLC2701


def _wide_fixture(rows: list[dict]) -> pd.DataFrame:
    """Build a mini wide-format frame (index columns + level columns)."""
    df = pd.DataFrame(rows)
    for c in ("wa", "lf", "emp", "une", "ina"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def test_synthesize_y25t54_from_5yr_bands():
    """When 25-34, 35-44, 45-54 are all present and Y25T54 is missing, sum levels."""
    wide = _wide_fixture([
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"), "sex": "M", "age": "25-34",
         "wa": 100.0, "lf": 90.0, "emp": 85.0, "une": 5.0, "ina": 10.0},
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"), "sex": "M", "age": "35-44",
         "wa": 120.0, "lf": 110.0, "emp": 105.0, "une": 5.0, "ina": 10.0},
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"), "sex": "M", "age": "45-54",
         "wa": 110.0, "lf": 100.0, "emp": 95.0, "une": 5.0, "ina": 10.0},
    ])
    out = _synthesize_age_aggregates(wide)
    syn = out[out["age"] == "Y25T54"]
    assert len(syn) == 1
    np.testing.assert_allclose(syn["wa"].iloc[0], 330.0)
    np.testing.assert_allclose(syn["emp"].iloc[0], 285.0)
    np.testing.assert_allclose(syn["une"].iloc[0], 15.0)


def test_synthesize_skips_when_target_already_present():
    """If Y25T54 is already in the data for that cell, do NOT overwrite."""
    wide = _wide_fixture([
        {"code": "USA", "date": pd.Timestamp("2020-01-01"), "sex": "_T", "age": "Y25T54",
         "wa": 9999.0, "lf": 0.0, "emp": 0.0, "une": 0.0, "ina": 0.0},
        {"code": "USA", "date": pd.Timestamp("2020-01-01"), "sex": "_T", "age": "25-34",
         "wa": 100.0, "lf": 90.0, "emp": 85.0, "une": 5.0, "ina": 10.0},
        {"code": "USA", "date": pd.Timestamp("2020-01-01"), "sex": "_T", "age": "35-44",
         "wa": 120.0, "lf": 110.0, "emp": 105.0, "une": 5.0, "ina": 10.0},
        {"code": "USA", "date": pd.Timestamp("2020-01-01"), "sex": "_T", "age": "45-54",
         "wa": 110.0, "lf": 100.0, "emp": 95.0, "une": 5.0, "ina": 10.0},
    ])
    out = _synthesize_age_aggregates(wide)
    syn = out[out["age"] == "Y25T54"]
    assert len(syn) == 1
    assert syn["wa"].iloc[0] == 9999.0  # untouched


def test_synthesize_skips_when_any_source_missing():
    """If any source band is absent, no row is synthesized."""
    wide = _wide_fixture([
        {"code": "FRA", "date": pd.Timestamp("2020-01-01"), "sex": "F", "age": "25-34",
         "wa": 100.0, "lf": 90.0, "emp": 85.0, "une": 5.0, "ina": 10.0},
        # missing 35-44
        {"code": "FRA", "date": pd.Timestamp("2020-01-01"), "sex": "F", "age": "45-54",
         "wa": 110.0, "lf": 100.0, "emp": 95.0, "une": 5.0, "ina": 10.0},
    ])
    out = _synthesize_age_aggregates(wide)
    assert (out["age"] == "Y25T54").sum() == 0


def test_synthesize_y15t24_and_y55t64_chain():
    """Y15T24 = 15-64 - 25+ + 65+ ; Y55T64 = 25+ - Y25T54 - 65+ (chained on synth Y25T54)."""
    # Realistic numbers: 15-64 = 1000, 25+ = 800, 65+ = 200, so Y15T24 = 1000-800+200 = 400.
    # 25-34 = 200, 35-44 = 250, 45-54 = 200 → Y25T54 = 650.
    # Y55T64 = 25+ - Y25T54 - 65+ = 800 - 650 - 200 = -50  ← intentionally negative for the test;
    #   real data would never give this, but the formula must apply mechanically.
    cell = {"code": "ITA", "date": pd.Timestamp("2020-01-01"), "sex": "_T"}
    wide = _wide_fixture([
        {**cell, "age": "15-64", "wa": 1000.0, "lf": 800.0, "emp": 750.0, "une": 50.0, "ina": 200.0},
        {**cell, "age": "25+",   "wa":  800.0, "lf": 600.0, "emp": 580.0, "une": 20.0, "ina": 200.0},
        {**cell, "age": "65+",   "wa":  200.0, "lf":  10.0, "emp":  10.0, "une":  0.0, "ina": 190.0},
        {**cell, "age": "25-34", "wa":  200.0, "lf": 180.0, "emp": 170.0, "une": 10.0, "ina":  20.0},
        {**cell, "age": "35-44", "wa":  250.0, "lf": 230.0, "emp": 220.0, "une": 10.0, "ina":  20.0},
        {**cell, "age": "45-54", "wa":  200.0, "lf": 180.0, "emp": 170.0, "une": 10.0, "ina":  20.0},
    ])
    out = _synthesize_age_aggregates(wide)
    y_y = out[out["age"] == "Y15T24"]
    y_p = out[out["age"] == "Y25T54"]
    y_o = out[out["age"] == "Y55T64"]
    assert len(y_y) == 1 and len(y_p) == 1 and len(y_o) == 1
    np.testing.assert_allclose(y_p["wa"].iloc[0], 650.0)
    np.testing.assert_allclose(y_y["wa"].iloc[0], 1000.0 - 800.0 + 200.0)
    np.testing.assert_allclose(y_o["wa"].iloc[0], 800.0 - 650.0 - 200.0)


def test_synthesize_y15t24_from_15plus_minus_25plus():
    """EU-style data: only Y_GE15 and 25+ available. Y15T24 = Y_GE15 - 25+.

    In LABOR.xlsx, "15+" is mapped to "Y_GE15" by _xlsx_to_sdmx_long, so the
    synthesis rule uses Y_GE15 as the source rather than the raw "15+" string.
    """
    cell = {"code": "AUT", "date": pd.Timestamp("2020-01-01"), "sex": "M"}
    wide = _wide_fixture([
        {**cell, "age": "Y_GE15", "wa": 1000.0, "lf": 800.0, "emp": 750.0, "une": 50.0, "ina": 200.0},
        {**cell, "age": "25+",    "wa":  800.0, "lf": 600.0, "emp": 580.0, "une": 20.0, "ina": 200.0},
    ])
    out = _synthesize_age_aggregates(wide)
    syn = out[out["age"] == "Y15T24"]
    assert len(syn) == 1, f"expected 1 Y15T24 row, got {len(syn)}"
    np.testing.assert_allclose(syn["wa"].iloc[0],  200.0)  # 1000 - 800
    np.testing.assert_allclose(syn["lf"].iloc[0],  200.0)  # 800 - 600
    np.testing.assert_allclose(syn["emp"].iloc[0], 170.0)  # 750 - 580
    np.testing.assert_allclose(syn["une"].iloc[0],  30.0)  # 50 - 20


from puremacro.fetch.labor import fetch_labor_panel_union


def _canonical_row(**kw):
    """Build a 12-column canonical row with sensible defaults."""
    base = {"code": "AUT", "date": pd.Timestamp("2024-01-01"),
            "sex": "_T", "age": "Y_GE15",
            "wa": np.nan, "lf": np.nan, "emp": np.nan, "une": np.nan,
            "ina": np.nan, "urate": np.nan, "epop": np.nan, "prate": np.nan}
    base.update(kw)
    return base


def test_union_empty_input_returns_canonical_schema():
    out = fetch_labor_panel_union([])
    assert list(out.columns) == ["code", "date", "sex", "age",
                                  "wa", "lf", "emp", "une", "ina",
                                  "urate", "epop", "prate"]
    assert len(out) == 0


def test_union_concatenates_disjoint_panels():
    a = pd.DataFrame([_canonical_row(code="USA", urate=0.04)])
    b = pd.DataFrame([_canonical_row(code="DEU", urate=0.06)])
    out = fetch_labor_panel_union([a, b])
    assert len(out) == 2
    assert set(out["code"]) == {"USA", "DEU"}


def test_union_first_listed_source_wins_on_collision():
    """Same (code, date, sex, age) — first-listed frame wins per column,
    but NaN cells fall through to the next panel (combine_first semantics)."""
    a = pd.DataFrame([_canonical_row(code="SWE", urate=0.05, lf=100.0)])  # OECD-like (rich)
    b = pd.DataFrame([_canonical_row(code="SWE", urate=0.07, lf=np.nan)])  # Eurostat-like
    out = fetch_labor_panel_union([a, b])
    assert len(out) == 1
    np.testing.assert_allclose(out["urate"].iloc[0], 0.05)  # a wins (non-NaN)
    np.testing.assert_allclose(out["lf"].iloc[0], 100.0)    # a wins (non-NaN)


def test_union_nan_fall_through_to_next_panel():
    """Where the higher-priority panel has NaN, next panel fills in."""
    a = pd.DataFrame([_canonical_row(code="SWE", urate=np.nan, lf=np.nan)])  # OECD with NaN urate
    b = pd.DataFrame([_canonical_row(code="SWE", urate=0.07, lf=np.nan)])     # Eurostat with urate
    out = fetch_labor_panel_union([a, b])
    assert len(out) == 1
    np.testing.assert_allclose(out["urate"].iloc[0], 0.07)  # b fills a's NaN


def test_union_sorted_and_reset_index():
    a = pd.DataFrame([
        _canonical_row(code="USA", date=pd.Timestamp("2024-02-01")),
        _canonical_row(code="USA", date=pd.Timestamp("2024-01-01")),
    ])
    out = fetch_labor_panel_union([a])
    assert list(out["date"]) == sorted(out["date"])
    assert list(out.index) == list(range(len(out)))


def test_fetch_sectoral_panel_union_priority_first_wins():
    """First non-NaN cell wins; later panels fill NaNs."""
    from puremacro.fetch.labor import fetch_sectoral_panel_union
    canonical_cols = ["code", "date", "sex", "eco", "emp", "hours_worked"]
    p1 = pd.DataFrame([
        {"code": "BRA", "date": pd.Timestamp("2020-01-01"), "sex": "_T",
         "eco": "_T", "emp": 100.0, "hours_worked": float("nan")},
    ])
    p2 = pd.DataFrame([
        {"code": "BRA", "date": pd.Timestamp("2020-01-01"), "sex": "_T",
         "eco": "_T", "emp": 999.0, "hours_worked": 38.0},
        {"code": "MEX", "date": pd.Timestamp("2020-01-01"), "sex": "_T",
         "eco": "_T", "emp": 55.0, "hours_worked": 40.0},
    ])
    out = fetch_sectoral_panel_union([p1, p2])
    assert set(out.columns) == set(canonical_cols)
    bra = out[out["code"] == "BRA"].iloc[0]
    # p1's emp wins (100, not 999). p2 fills hours_worked NaN.
    assert bra["emp"] == 100.0
    assert bra["hours_worked"] == 38.0
    # p2-only row preserved.
    mex = out[out["code"] == "MEX"].iloc[0]
    assert mex["emp"] == 55.0


def test_fetch_sectoral_panel_union_empty_input_returns_empty_frame():
    from puremacro.fetch.labor import fetch_sectoral_panel_union
    out = fetch_sectoral_panel_union([])
    assert list(out.columns) == [
        "code", "date", "sex", "eco", "emp", "hours_worked"
    ]
    assert out.empty


def test_fetch_labor_panel_union_three_source_priority_ordering():
    """OECD wins on overlap; Eurostat fills its NaNs; ILOSTAT fills the rest."""
    from puremacro.fetch.labor import fetch_labor_panel_union
    nan = float("nan")

    def _row(code, **vals):
        base = dict(date=pd.Timestamp("2020-01-01"), sex="_T", age="Y_GE15",
                    wa=nan, lf=nan, emp=nan, une=nan, ina=nan,
                    urate=nan, epop=nan, prate=nan)
        base["code"] = code
        base.update(vals)
        return base

    oecd = pd.DataFrame([
        _row("USA", urate=0.04, epop=0.60, prate=0.62),
        _row("DEU", urate=0.03, epop=0.55),  # missing prate
    ])
    eurostat = pd.DataFrame([
        _row("USA", urate=0.99),  # OECD wins
        _row("DEU", urate=0.99, prate=0.59),  # only prate fills
        _row("ESP", urate=0.13),  # ESP-only in Eurostat
    ])
    ilostat = pd.DataFrame([
        _row("BRA", urate=0.12, epop=0.52, prate=0.59),  # ILOSTAT-only
        _row("USA", urate=0.99, epop=0.99, prate=0.99),  # all overridden
        _row("DEU", urate=0.99),
        _row("ESP", prate=0.55, epop=0.50),  # Eurostat wins urate; ILOSTAT fills others
    ])

    out = fetch_labor_panel_union([oecd, eurostat, ilostat])
    out = out.set_index(["code", "date", "sex", "age"])
    # OECD wins for USA.
    assert out.loc[("USA", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "urate"] == 0.04
    assert out.loc[("USA", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "epop"] == 0.60
    # DEU: OECD's urate/epop wins; Eurostat's prate fills.
    assert out.loc[("DEU", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "urate"] == 0.03
    assert out.loc[("DEU", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "prate"] == 0.59
    # ESP: Eurostat's urate wins; ILOSTAT's prate/epop fill.
    assert out.loc[("ESP", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "urate"] == 0.13
    assert out.loc[("ESP", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "prate"] == 0.55
    assert out.loc[("ESP", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "epop"] == 0.50
    # BRA: ILOSTAT-only.
    assert out.loc[("BRA", pd.Timestamp("2020-01-01"), "_T", "Y_GE15"), "urate"] == 0.12
