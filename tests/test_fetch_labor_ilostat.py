"""Tests for puremacro.fetch.labor_ilostat fetchers and helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_FIXTURE_LFS = (
    Path(__file__).parent / "fixtures" / "labor" / "synthetic_ilostat_lfs_q.csv"
)


def test_build_ilostat_lfs_key_default_filters():
    """Default LFS key shape: REF_AREA.FREQ.MEASURE.SEX.AGE.
    Dimension order confirmed from DSD: https://sdmx.ilo.org/rest/datastructure/ILO/EMP_TEMP_SEX_AGE_NB
    """
    from puremacro.fetch.labor_ilostat import _build_ilostat_lfs_key
    key = _build_ilostat_lfs_key(
        countries=None, sexes=("_T", "M", "F"),
        ages=("Y_GE15", "Y15T24"),
        measures=("UNE_DEAP_RT", "EMP_DWAP_RT", "EAP_DWAP_RT"),
        frequency="Q",
    )
    parts = key.split(".")
    assert parts[0] == ""  # REF_AREA: countries=None means all
    assert parts[1] == "Q"  # FREQ
    assert parts[2] == "UNE_DEAP_RT+EMP_DWAP_RT+EAP_DWAP_RT"  # MEASURE
    assert parts[3] == "_T+M+F"  # SEX
    assert parts[4] == "Y_GE15+Y15T24"  # AGE


def test_build_ilostat_lfs_key_with_country_filter():
    """When countries=['BRA','MEX'], REF_AREA slot (pos 0) becomes 'BRA+MEX'."""
    from puremacro.fetch.labor_ilostat import _build_ilostat_lfs_key
    key = _build_ilostat_lfs_key(
        countries=["BRA", "MEX"], sexes=("_T",),
        ages=("Y_GE15",), measures=("UNE_DEAP_RT",),
        frequency="Q",
    )
    parts = key.split(".")
    assert parts[0] == "BRA+MEX"  # REF_AREA is position 0


def test_reshape_ilostat_lfs_drops_aggregates():
    """Aggregate ILO codes (WORLD, LAC_LIC) must be dropped."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_lfs_csv
    raw = pd.read_csv(_FIXTURE_LFS)
    out = _reshape_ilostat_lfs_csv(raw, frequency="Q")
    assert "WORLD" not in out["code"].unique()
    assert "LAC_LIC" not in out["code"].unique()
    assert set(out["code"].unique()) == {"BRA", "MEX"}


def test_reshape_ilostat_lfs_canonical_columns():
    """Output frame has the 12 canonical columns and date is datetime."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_lfs_csv
    raw = pd.read_csv(_FIXTURE_LFS)
    out = _reshape_ilostat_lfs_csv(raw, frequency="Q")
    expected_cols = {
        "code", "date", "sex", "age",
        "wa", "lf", "emp", "une", "ina",
        "urate", "epop", "prate",
    }
    assert set(out.columns) == expected_cols
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_reshape_ilostat_lfs_maps_age_codes():
    """ILOSTAT CLASSIF1 codes (AGE_AGGREGATE_TOTAL, AGE_YTHADULT_Y15-24)
    map to canonical Y_GE15 / Y15T24."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_lfs_csv
    raw = pd.read_csv(_FIXTURE_LFS)
    out = _reshape_ilostat_lfs_csv(raw, frequency="Q")
    ages = set(out["age"].unique())
    # Both canonical age codes should appear; raw ILOSTAT codes should NOT.
    assert "Y_GE15" in ages
    assert "Y15T24" in ages
    assert "AGE_AGGREGATE_TOTAL" not in ages
    assert "AGE_YTHADULT_Y15-24" not in ages


def test_reshape_ilostat_lfs_pivots_measures_to_columns():
    """UNE_RT / EPOP_RT / LFEPOP_RT pivot into the urate / epop / prate columns
    (rates are reported as percent in ILOSTAT; reshape divides by 100 to match
    the OECD/Eurostat schema where rates are proportions in 0..1)."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_lfs_csv
    raw = pd.read_csv(_FIXTURE_LFS)
    out = _reshape_ilostat_lfs_csv(raw, frequency="Q")
    bra_total_q1 = out[
        (out["code"] == "BRA") & (out["sex"] == "_T")
        & (out["age"] == "Y_GE15") & (out["date"] == pd.Timestamp("2020-01-01"))
    ].iloc[0]
    assert bra_total_q1["urate"] == pytest.approx(0.125)   # 12.5% / 100
    assert bra_total_q1["epop"] == pytest.approx(0.524)
    assert bra_total_q1["prate"] == pytest.approx(0.599)


def test_fetch_ilostat_lfs_panel_with_csv_path():
    """Fetcher reads a local SDMX-CSV via csv_path= and returns the
    canonical 12-col schema without hitting the network."""
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    out = fetch_ilostat_lfs_panel(
        frequency="Q",
        adjustment="NSA",  # bypass X13 in this task
        csv_path=str(_FIXTURE_LFS),
    )
    assert not out.empty
    assert set(out.columns) == {
        "code", "date", "sex", "age",
        "wa", "lf", "emp", "une", "ina",
        "urate", "epop", "prate",
    }
    assert "BRA" in set(out["code"].unique())


def test_fetch_ilostat_lfs_panel_cache_roundtrip(tmp_path):
    """First call fetches via csv_path; second call hits the cache."""
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    cache = tmp_path / "ilostat_lfs.parquet"
    a = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="NSA",
        csv_path=str(_FIXTURE_LFS), cache_path=str(cache),
    )
    assert cache.exists()
    # Second call: csv_path missing; cache hit short-circuits.
    b = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="NSA",
        csv_path=str(tmp_path / "missing.csv"), cache_path=str(cache),
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_fetch_ilostat_lfs_panel_refresh_bypasses_cache(tmp_path):
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    cache = tmp_path / "ilostat_lfs.parquet"
    fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="NSA",
        csv_path=str(_FIXTURE_LFS), cache_path=str(cache),
    )
    # Refresh=True must re-read csv_path and overwrite cache; if csv_path is
    # missing on a refresh, expect FileNotFoundError (not silent cache hit).
    with pytest.raises(FileNotFoundError):
        fetch_ilostat_lfs_panel(
            frequency="Q", adjustment="NSA",
            csv_path=str(tmp_path / "missing.csv"),
            cache_path=str(cache), refresh=True,
        )


def test_fetch_ilostat_lfs_panel_country_filter_via_csv(tmp_path):
    """When countries= is supplied to a csv_path call, post-filter applies."""
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    out = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="NSA",
        countries=["BRA"], csv_path=str(_FIXTURE_LFS),
    )
    assert set(out["code"].unique()) == {"BRA"}


import shutil


def _has_x13_binary() -> bool:
    return shutil.which("x13as") is not None or shutil.which("x13as_ascii") is not None


def _make_synthetic_seasonal_quarterly_csv(tmp_path):
    """Build an ILOSTAT-shape CSV where the 'urate' series for one country
    has clear quarterly seasonality, long enough for X13."""
    rng = np.random.default_rng(0)
    quarters = pd.period_range("2010Q1", "2023Q4", freq="Q")
    rows = []
    for i, q in enumerate(quarters):
        base = 8.0 + i * 0.02
        seasonal = (2.0 if q.quarter == 1 else
                    -1.5 if q.quarter == 3 else 0.0)
        val = base + seasonal + rng.normal(0, 0.05)
        rows.append({
            "REF_AREA": "BRA", "FREQ": "Q", "MEASURE": "UNE_RT",
            "SEX": "_T", "CLASSIF1": "AGE_AGGREGATE_TOTAL",
            "TIME_PERIOD": str(q), "OBS_VALUE": val,
        })
    csv = tmp_path / "ilostat_lfs_seasonal.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_fetch_ilostat_lfs_panel_sa_applies_x13_when_quarterly(tmp_path):
    if not _has_x13_binary():
        pytest.skip("x13as binary not installed")
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    csv = _make_synthetic_seasonal_quarterly_csv(tmp_path)
    sa = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="SA",
        countries=["BRA"], csv_path=str(csv),
    )
    nsa = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="NSA",
        countries=["BRA"], csv_path=str(csv),
    )
    # Same shape; values must differ (SA reduces seasonal variation).
    assert sa.shape == nsa.shape
    sa_q1_q3 = sa[sa["date"].dt.quarter == 1]["urate"].mean() - \
               sa[sa["date"].dt.quarter == 3]["urate"].mean()
    nsa_q1_q3 = nsa[nsa["date"].dt.quarter == 1]["urate"].mean() - \
                nsa[nsa["date"].dt.quarter == 3]["urate"].mean()
    assert abs(sa_q1_q3) < abs(nsa_q1_q3) * 0.5


def test_fetch_ilostat_lfs_panel_sa_skipped_for_monthly_frequency(tmp_path):
    """For frequency='M', X13 must NOT be applied even when adjustment='SA'."""
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    # The fixture is quarterly; reuse it but request monthly. The monthly
    # frequency filter drops every row, so the output is empty — but the
    # call must not crash with an X13 error or attempt SA on zero rows.
    out = fetch_ilostat_lfs_panel(
        frequency="M", adjustment="SA",
        csv_path=str(_FIXTURE_LFS),
    )
    assert out.empty
    assert set(out.columns) == {
        "code", "date", "sex", "age",
        "wa", "lf", "emp", "une", "ina",
        "urate", "epop", "prate",
    }


def test_fetch_ilostat_lfs_panel_sa_short_cell_returns_nan(tmp_path):
    """If a (code, sex, age, measure) cell has < 12 non-NaN obs, X13 cannot
    be applied; the cell is left NaN, not crashed on."""
    if not _has_x13_binary():
        pytest.skip("x13as binary not installed")
    from puremacro.fetch.labor_ilostat import fetch_ilostat_lfs_panel
    # The default fixture has only 2 quarters of BRA data — too short for X13.
    out = fetch_ilostat_lfs_panel(
        frequency="Q", adjustment="SA",
        countries=["BRA"], csv_path=str(_FIXTURE_LFS),
    )
    # Output is non-empty but rate cells are NaN (cell had < 12 quarters).
    assert not out.empty
    assert out["urate"].isna().all()


_FIXTURE_SECTORAL = (
    Path(__file__).parent / "fixtures" / "labor" / "synthetic_ilostat_sectoral_q.csv"
)


def test_build_ilostat_sectoral_key_default_filters():
    """Sectoral key: REF_AREA.FREQ.MEASURE.SEX.ECO.
    Dimension order confirmed from DSD: https://sdmx.ilo.org/rest/datastructure/ILO/EMP_TEMP_SEX_ECO_NB
    """
    from puremacro.fetch.labor_ilostat import _build_ilostat_sectoral_key
    key = _build_ilostat_sectoral_key(
        countries=None, sexes=("_T",),
        sectors=("_T", "A", "B-E"),
        measures=("EMP_TEMP_NB", "HOW_TEMP_NB"),
    )
    parts = key.split(".")
    assert parts[0] == ""   # REF_AREA: countries=None means all
    assert parts[1] == "Q"  # FREQ hardcoded to Q for sectoral
    assert parts[2] == "EMP_TEMP_NB+HOW_TEMP_NB"  # MEASURE
    assert parts[3] == "SEX_T"  # SEX (translated to API code)
    # ECO: sectors get ECO_ISIC4_ prefix; _T → ECO_AGGREGATE_TOTAL
    assert "ECO_AGGREGATE_TOTAL" in parts[4]
    assert "ECO_ISIC4_A" in parts[4]
    assert "ECO_ISIC4_B-E" in parts[4]


def test_reshape_ilostat_sectoral_csv_drops_aggregates_and_2digit():
    """Aggregate codes (WORLD) AND 2-digit ISIC divisions (ECO_ISIC4_01) drop."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_sectoral_csv
    raw = pd.read_csv(_FIXTURE_SECTORAL)
    out = _reshape_ilostat_sectoral_csv(raw)
    assert "WORLD" not in out["code"].unique()
    # 2-digit codes (anything starting with ECO_ISIC4_ followed by digits) drop.
    assert all(not str(s).startswith("0") and not str(s).isdigit()
               for s in out["eco"].unique())
    # Section letters and standard groupings are kept (with prefix stripped).
    eco_codes = set(out["eco"].unique())
    assert "_T" in eco_codes
    assert "A" in eco_codes
    assert "B-E" in eco_codes


def test_reshape_ilostat_sectoral_csv_canonical_columns():
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_sectoral_csv
    raw = pd.read_csv(_FIXTURE_SECTORAL)
    out = _reshape_ilostat_sectoral_csv(raw)
    assert set(out.columns) == {"code", "date", "sex", "eco", "emp", "hours_worked"}
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_reshape_ilostat_sectoral_csv_pivots_emp_and_hours():
    """EMP_TEMP and HOW_TEMP land in emp and hours_worked columns."""
    from puremacro.fetch.labor_ilostat import _reshape_ilostat_sectoral_csv
    raw = pd.read_csv(_FIXTURE_SECTORAL)
    out = _reshape_ilostat_sectoral_csv(raw)
    bra_total_q1 = out[
        (out["code"] == "BRA") & (out["eco"] == "_T")
        & (out["date"] == pd.Timestamp("2020-01-01"))
    ].iloc[0]
    assert bra_total_q1["emp"] == pytest.approx(89000.0)
    assert bra_total_q1["hours_worked"] == pytest.approx(38.5)


def test_fetch_ilostat_sectoral_panel_with_csv_path():
    from puremacro.fetch.labor_ilostat import fetch_ilostat_sectoral_panel
    out = fetch_ilostat_sectoral_panel(
        adjustment="NSA",
        csv_path=str(_FIXTURE_SECTORAL),
    )
    assert not out.empty
    assert set(out.columns) == {"code", "date", "sex", "eco", "emp", "hours_worked"}
    assert "BRA" in set(out["code"].unique())
    assert "MEX" in set(out["code"].unique())


def test_fetch_ilostat_sectoral_panel_country_filter(tmp_path):
    from puremacro.fetch.labor_ilostat import fetch_ilostat_sectoral_panel
    out = fetch_ilostat_sectoral_panel(
        adjustment="NSA",
        countries=["BRA"], csv_path=str(_FIXTURE_SECTORAL),
    )
    assert set(out["code"].unique()) == {"BRA"}


def test_fetch_ilostat_sectoral_panel_cache_roundtrip(tmp_path):
    from puremacro.fetch.labor_ilostat import fetch_ilostat_sectoral_panel
    cache = tmp_path / "ilostat_sectoral.parquet"
    a = fetch_ilostat_sectoral_panel(
        adjustment="NSA", csv_path=str(_FIXTURE_SECTORAL), cache_path=str(cache),
    )
    assert cache.exists()
    b = fetch_ilostat_sectoral_panel(
        adjustment="NSA", csv_path=str(tmp_path / "missing.csv"),
        cache_path=str(cache),
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_fetch_ilostat_sectoral_panel_sa_short_cell_returns_nan():
    """Sectoral SA path: cell with < 12 quarters has emp/hours_worked NaN."""
    if not _has_x13_binary():
        pytest.skip("x13as binary not installed")
    from puremacro.fetch.labor_ilostat import fetch_ilostat_sectoral_panel
    out = fetch_ilostat_sectoral_panel(
        adjustment="SA",
        countries=["BRA"], csv_path=str(_FIXTURE_SECTORAL),
    )
    # Fixture has 1 quarter of BRA data → all SA-able cells are NaN.
    assert not out.empty
    assert out["emp"].isna().all()
    assert out["hours_worked"].isna().all()


def test_public_api_exports_ilostat_fetchers():
    """The three new ILOSTAT-related public names should be importable
    directly from puremacro.fetch."""
    import puremacro.fetch as f
    assert hasattr(f, "fetch_ilostat_lfs_panel")
    assert hasattr(f, "fetch_ilostat_sectoral_panel")
    assert hasattr(f, "fetch_sectoral_panel_union")
    assert "fetch_ilostat_lfs_panel" in f.__all__
    assert "fetch_ilostat_sectoral_panel" in f.__all__
    assert "fetch_sectoral_panel_union" in f.__all__
