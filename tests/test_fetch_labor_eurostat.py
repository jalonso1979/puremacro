"""Tests for puremacro.fetch.labor_eurostat.fetch_eurostat_lfs_panel."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.labor_eurostat import (
    _build_eurostat_une_key,
    _EUROSTAT_GEO_TO_ISO3,
    _EUROSTAT_SEX_MAP,
    _EUROSTAT_AGE_MAP,
)


def test_geo_map_covers_eu_and_efta_with_special_cases():
    """EL→GRC (Greece), UK→GBR (United Kingdom) are the two non-trivial mappings."""
    assert _EUROSTAT_GEO_TO_ISO3["EL"] == "GRC"
    assert _EUROSTAT_GEO_TO_ISO3["UK"] == "GBR"
    assert _EUROSTAT_GEO_TO_ISO3["AT"] == "AUT"
    assert _EUROSTAT_GEO_TO_ISO3["DE"] == "DEU"
    # Aggregates must NOT be in the map (they get dropped via missing-key filter).
    for agg in ("EU27_2020", "EA20", "EFTA", "EU28", "EA19", "EU27", "OECD"):
        assert agg not in _EUROSTAT_GEO_TO_ISO3


def test_sex_map_normalizes_to_underscore_total():
    assert _EUROSTAT_SEX_MAP == {"T": "_T", "M": "M", "F": "F"}


def test_age_map_two_entries_only():
    """Only Y_LT25 and TOTAL map; Y25-74 must be dropped (broader than Y25T54)."""
    assert _EUROSTAT_AGE_MAP == {"Y_LT25": "Y15T24", "TOTAL": "Y_GE15"}
    assert "Y25-74" not in _EUROSTAT_AGE_MAP


def test_build_eurostat_une_key_default_shape():
    """Default key: M.SA.Y_LT25+TOTAL.PC_ACT.T+M+F. — 5 dots, empty trailing geo."""
    key = _build_eurostat_une_key(sexes=("_T", "M", "F"), countries=None)
    assert key.count(".") == 5
    parts = key.split(".")
    assert parts[0] == "M"
    assert parts[1] == "SA"
    assert parts[2] == "Y_LT25+TOTAL"
    assert parts[3] == "PC_ACT"
    assert parts[4] == "T+M+F"
    assert parts[5] == ""


def test_build_eurostat_une_key_with_country_filter():
    """ISO-3 inputs translate to ISO-2 in the geo position."""
    key = _build_eurostat_une_key(sexes=("_T",), countries=["AUT", "DEU", "GRC"])
    parts = key.split(".")
    assert parts[5] == "AT+DE+EL"  # GRC → EL via inverted map
    assert parts[4] == "T"


def test_build_eurostat_une_key_unknown_iso3_raises():
    """A non-EU/EFTA ISO-3 code should raise."""
    with pytest.raises(KeyError, match="USA"):
        _build_eurostat_une_key(sexes=("_T",), countries=["USA"])


from puremacro.fetch.labor_eurostat import _reshape_eurostat_une_rt_m


_FIXTURE = Path(__file__).parent / "fixtures" / "labor" / "eurostat_une_rt_m.csv"


def test_reshape_canonical_schema():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    expected_cols = {"code", "date", "sex", "age",
                     "wa", "lf", "emp", "une", "ina",
                     "urate", "epop", "prate"}
    assert set(out.columns) == expected_cols
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    # Rate cells populated; level cells are NaN
    assert out["urate"].notna().any()
    assert out[["wa", "lf", "emp", "une", "ina", "epop", "prate"]].isna().all().all()


def test_reshape_drops_aggregates_and_unknown_age_and_nsa():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    # EU27_2020 row → dropped
    assert "EU27_2020" not in out["code"].values
    # Y25-74 row → dropped
    assert (out["age"] == "Y25-74").sum() == 0
    # NSA row → dropped (we only keep SA)
    # The fixture's NSA row was AT/T/TOTAL with value 5.1; after dropping
    # we keep AT/T/TOTAL/SA with value 4.6.
    aut_t_total = out.loc[(out["code"] == "AUT") & (out["sex"] == "_T") &
                          (out["age"] == "Y_GE15")]
    assert len(aut_t_total) == 1
    np.testing.assert_allclose(aut_t_total["urate"].iloc[0], 0.046, rtol=1e-12)


def test_reshape_translates_special_geo_codes():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    assert "GRC" in out["code"].values  # EL → GRC
    assert "GBR" in out["code"].values  # UK → GBR
    assert "AUT" in out["code"].values  # AT → AUT


def test_reshape_age_normalization():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    # Y_LT25 → Y15T24 ; TOTAL → Y_GE15
    assert set(out["age"].unique()) <= {"Y15T24", "Y_GE15"}
    # AT had both (TOTAL and Y_LT25): should produce two rows for AT/T
    aut_t = out[(out["code"] == "AUT") & (out["sex"] == "_T")]
    assert set(aut_t["age"]) == {"Y_GE15", "Y15T24"}


def test_reshape_sex_normalization():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    assert set(out["sex"].unique()) <= {"_T", "M", "F"}


def test_reshape_urate_in_proportion_units():
    raw = pd.read_csv(_FIXTURE)
    out = _reshape_eurostat_une_rt_m(raw)
    # Eurostat publishes percent (4.6 = 4.6%); we divide by 100 for proportions
    aut_y15t24 = out.loc[(out["code"] == "AUT") & (out["sex"] == "_T") &
                         (out["age"] == "Y15T24")]
    np.testing.assert_allclose(aut_y15t24["urate"].iloc[0], 0.112, rtol=1e-12)


from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_panel


def test_fetch_from_local_csv_via_csv_path(tmp_path: Path):
    """A pre-downloaded SDMX-CSV file at csv_path skips the network."""
    out = fetch_eurostat_lfs_panel(csv_path=str(_FIXTURE))
    assert {"code", "date", "sex", "age", "urate"} <= set(out.columns)
    # AUT, GRC, GBR are all in the fixture; aggregates dropped.
    assert {"AUT", "GRC", "GBR"} <= set(out["code"].unique())
    assert "EU27_2020" not in out["code"].values


def test_fetch_cache_roundtrip(tmp_path: Path):
    cache = tmp_path / "estat_cache.parquet"
    a = fetch_eurostat_lfs_panel(csv_path=str(_FIXTURE), cache_path=str(cache))
    assert cache.exists()
    # Second call: cache hit, csv_path NOT consulted.
    b = fetch_eurostat_lfs_panel(csv_path=str(tmp_path / "missing.csv"),
                                  cache_path=str(cache))
    pd.testing.assert_frame_equal(a, b)


def test_fetch_refresh_flag_bypasses_cache(tmp_path: Path):
    cache = tmp_path / "estat_cache.parquet"
    fetch_eurostat_lfs_panel(csv_path=str(_FIXTURE), cache_path=str(cache))
    with pytest.raises(FileNotFoundError):
        fetch_eurostat_lfs_panel(
            csv_path=str(tmp_path / "still_missing.csv"),
            cache_path=str(cache),
            refresh=True,
        )


@pytest.mark.network
def test_fetch_live_skip_on_empty():
    """Live SDMX pull for one country / both ages. Skip on empty per project rule."""
    try:
        out = fetch_eurostat_lfs_panel(countries=["AUT"], sexes=("_T",))
    except (RuntimeError, OSError) as e:
        pytest.skip(f"Eurostat unreachable: {e}")
    if out.empty:
        pytest.skip("Eurostat returned empty for AUT/_T")
    assert (out["code"] == "AUT").all()
    assert {"Y_GE15", "Y15T24"} <= set(out["age"].unique())
