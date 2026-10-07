"""Offline tests for puremacro.fetch.pwt (Penn World Table 11.0 and Maddison 2023).

The HTTP seam ``puremacro.fetch.pwt._get`` is patched to serve tiny Stata
files written here with ``DataFrame.to_stata``, shaped like the published
ones (a full year grid with empty rows, a categorical flag, the labour
file's float year and persons, the national-accounts file's ``CH2``).
The live tests at the bottom are marked ``network`` and deselected by default.
"""
from __future__ import annotations

import http.client
import io
import urllib.error

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.pwt as pwt
from puremacro.fetch.pwt import fetch_maddison, fetch_pwt, pwt_variables


def _dta(df: pd.DataFrame, labels: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    df.to_stata(buf, write_index=False, version=118, variable_labels=labels or {})
    return buf.getvalue()


def _main_bytes() -> bytes:
    rows = []
    for code, country, first in [("USA", "United States", 1950), ("MEX", "Mexico", 1951),
                                 ("ZWE", "Zimbabwe", 2099)]:
        for year in (1950, 1951, 1952):
            has = year >= first
            rows.append({
                "countrycode": code, "country": country, "currency_unit": "Unit",
                "year": np.int16(year),
                "rgdpna": float(100 + year - 1950) if has else np.nan,
                "pop": 10.0 if has else np.nan,
                "i_xr": "Market-based" if has else None,
            })
    df = pd.DataFrame(rows)
    df["rgdpna"] = df["rgdpna"].astype(np.float32)
    df["i_xr"] = pd.Categorical(df["i_xr"], categories=["Market-based", "Estimated"])
    return _dta(df, {"rgdpna": "Real GDP at constant 2021 national prices (in mil. 2021US$)",
                     "pop": "Population (in millions)", "i_xr": "0/1: market-based or estimated"})


def _na_bytes() -> bytes:
    df = pd.DataFrame({
        "countrycode": ["USA", "USA", "CH2", "SUN"],
        "year": np.array([2020, 2021, 2021, 2021], dtype=np.int16),
        "v_gdp": [2.0e7, 2.3e7, 1.1e8, np.nan],
        "q_gdp": [2.1e7, 2.3e7, 1.1e8, np.nan],
    })
    return _dta(df, {"q_gdp": "GDP at constant national 2017 prices",
                     "v_gdp": "GDP at current national prices"})


def _labor_bytes() -> bytes:
    df = pd.DataFrame({
        "countrycode": ["MEX", "MEX"], "year": [1950.0, 1951.0],
        "emp": np.array([7_771_864.5, 7_868_221.0], dtype=np.float32),
        "labsh": np.array([0.395, 0.395], dtype=np.float32),
        "i_labsh": pd.Categorical(["Adjustment 2, part mixed income"] * 2),
        "countryname": ["", ""], "indicatorname": ["", ""], "seriescode": ["", ""],
    })
    return _dta(df)


def _maddison_bytes() -> bytes:
    df = pd.DataFrame({
        "countrycode": ["USA", "USA", "USA", "SUN", "SUN"],
        "country": ["United States", "United States", "United States",
                    "Former USSR", "Former USSR"],
        "region": ["Western Offshoots"] * 3 + ["Eastern Europe"] * 2,
        "year": np.array([1, 1820, 1950, 1950, 1951], dtype=np.int16),
        "gdppc": [np.nan, 2674.0, 15240.0, 2841.0, np.nan],
        "pop": [680.0, 9981.0, 158804.0, 179571.0, np.nan],
    })
    return _dta(df)


_FILES = {554030: _main_bytes, 554024: _na_bytes, 554028: _labor_bytes,
          421303: _maddison_bytes}


@pytest.fixture
def served(monkeypatch):
    """Serve the synthetic files by datafile id and record each call."""
    calls: list[tuple[str, dict]] = []

    def fake_get(url, **kw):
        calls.append((url, kw))
        return _FILES[int(url.rsplit("/", 1)[1])]()

    monkeypatch.setattr(pwt, "_get", fake_get)
    return calls


def test_main_table_is_indexed_by_code_and_january_first_dates(served):
    out = fetch_pwt("main")
    assert out.index.names == ["code", "date"]
    dates = out.index.get_level_values("date")
    assert (dates.month == 1).all() and (dates.day == 1).all()
    assert set(dates.year) == {1950, 1951, 1952}
    assert out["rgdpna"].dtype == np.float64
    assert out.loc[("USA", pd.Timestamp("1952-01-01")), "rgdpna"] == 102.0
    assert served[0][0] == "https://dataverse.nl/api/access/datafile/554030"


def test_rows_without_any_value_are_dropped_so_counts_are_coverage(served):
    out = fetch_pwt("main")
    assert "ZWE" not in out.index.get_level_values("code")
    assert out.loc["MEX"].index.min() == pd.Timestamp("1951-01-01")
    assert len(out) == 5


def test_flags_are_integer_codes_with_their_value_labels_in_attrs(served):
    out = fetch_pwt("main")
    assert str(out["i_xr"].dtype) == "Int64"
    assert set(out["i_xr"].dropna()) == {0}
    assert out.attrs["value_labels"]["i_xr"] == {0: "Market-based", 1: "Estimated"}


def test_attrs_carry_provenance_licence_citation_units_and_hash(served):
    out = fetch_pwt("main")
    a = out.attrs
    for key in ("source", "doi", "url", "datafile_id", "file", "sha256", "release",
                "license", "citation", "price_base", "units", "labels", "meta",
                "missing", "fetched_at"):
        assert key in a, key
    assert a["license"] == "CC BY 4.0"
    assert a["doi"] == "10.34894/FABVLR"
    assert a["price_base"] == 2021
    assert "Feenstra" in a["citation"]
    assert len(a["sha256"]) == 64
    assert a["units"]["rgdpna"].startswith("millions of 2021 US$")
    assert a["fetched_at"].endswith("+00:00")
    assert a["missing"] == ()
    meta = {m["variable"]: m for m in a["meta"]}
    assert meta["pop"]["unit_mult"] == 6 and meta["pop"]["n_codes"] == 2
    assert meta["rgdpna"]["first"] == 1950 and meta["rgdpna"]["last"] == 1952
    assert all(isinstance(m, dict) for m in a["meta"])


def test_text_columns_are_kept_as_strings(served):
    out = fetch_pwt("main")
    assert out.loc[("MEX", pd.Timestamp("1951-01-01")), "country"] == "Mexico"
    assert "currency_unit" in out.columns


def test_na_table_drops_the_alternative_china_series_unless_asked_for(served):
    out = fetch_pwt("na")
    assert set(out.index.get_level_values("code")) == {"USA"}
    asked = fetch_pwt("na", codes=["CH2"])
    assert set(asked.index.get_level_values("code")) == {"CH2"}


def test_stale_2017_labels_are_annotated_and_units_state_the_2021_base(served):
    out = fetch_pwt("na")
    meta = {m["variable"]: m for m in out.attrs["meta"]}
    assert "2017" in meta["q_gdp"]["label"]
    assert "2021" in meta["q_gdp"]["note"]
    assert "constant 2021 prices" in out.attrs["units"]["q_gdp"]
    assert "note" not in meta["v_gdp"]


def test_labor_table_gets_integer_years_millions_of_persons_and_no_stray_columns(served):
    out = fetch_pwt("labor")
    assert out.loc[("MEX", pd.Timestamp("1950-01-01")), "emp"] == pytest.approx(7.7718645)
    assert not {"countryname", "indicatorname", "seriescode"} & set(out.columns)
    assert out.attrs["units"]["emp"] == "millions of persons"
    assert str(out["i_labsh"].dtype) == "Int64"


def test_codes_and_start_filter_and_absent_codes_are_reported(served):
    with pytest.warns(UserWarning, match="XXX"):
        out = fetch_pwt("main", codes=["usa", "XXX"], start=1951)
    assert set(out.index.get_level_values("code")) == {"USA"}
    assert out.index.get_level_values("date").min() == pd.Timestamp("1951-01-01")
    assert out.attrs["missing"] == ({"code": "XXX", "reason": "no data in PWT 11.0 main"},)


def test_a_provider_failure_returns_an_empty_frame_with_missing_and_does_not_raise(monkeypatch):
    def boom(url, **kw):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(pwt, "_get", boom)
    with pytest.warns(UserWarning, match="HTTP 503"):
        out = fetch_pwt("na")
    assert out.empty
    assert out.index.names == ["code", "date"]
    assert "v_gdp" in out.columns
    assert out.attrs["missing"][0]["reason"] == "HTTP 503"
    assert out.attrs["license"] == "CC BY 4.0" and out.attrs["sha256"] is None


def test_a_network_error_on_maddison_returns_an_empty_frame(monkeypatch):
    def boom(url, **kw):
        raise TimeoutError("timed out")

    monkeypatch.setattr(pwt, "_get", boom)
    with pytest.warns(UserWarning):
        out = fetch_maddison()
    assert out.empty and out.index.names == ["code", "date"]
    assert list(out.columns) == ["country", "region", "gdppc", "pop"]
    assert out.attrs["missing"][0]["reason"] == "timeout"


def test_a_bot_check_page_is_refetched_once_past_the_cache_then_reported(monkeypatch):
    calls = []
    page = b"<!doctype html><html><title>Making sure you're not a bot!</title></html>"

    def challenge(url, **kw):
        calls.append(kw.get("refresh"))
        return page

    monkeypatch.setattr(pwt, "_get", challenge)
    with pytest.warns(UserWarning, match="bot-check"):
        out = fetch_pwt("main")
    assert calls == [False, True]
    assert out.empty and "bot-check" in out.attrs["missing"][0]["reason"]


def test_a_cached_challenge_page_is_replaced_by_the_fresh_stata_file(monkeypatch):
    bodies = iter([b"<html>not a bot</html>", _main_bytes()])
    monkeypatch.setattr(pwt, "_get", lambda url, **kw: next(bodies))
    out = fetch_pwt("main")
    assert len(out) == 5 and out.attrs["missing"] == ()


def test_a_truncated_download_returns_empty_frames_instead_of_raising(monkeypatch):
    def cut(url, **kw):
        raise http.client.IncompleteRead(b"", 10)

    monkeypatch.setattr(pwt, "_get", cut)
    with pytest.warns(UserWarning, match="IncompleteRead"):
        na = fetch_pwt("na")
    with pytest.warns(UserWarning, match="IncompleteRead"):
        mpd = fetch_maddison()
    with pytest.warns(UserWarning, match="IncompleteRead"):
        labels = pwt_variables("na")
    for out in (na, mpd):
        assert out.empty and out.index.names == ["code", "date"]
        assert "IncompleteRead" in out.attrs["missing"][0]["reason"]
    assert labels["label"].isna().all()


def test_a_cached_stata_body_that_does_not_parse_is_refetched_past_the_cache(monkeypatch):
    calls = []
    bodies = iter([b"<stata_dta><header>garbage", _main_bytes()])

    def get(url, **kw):
        calls.append(kw.get("refresh"))
        return next(bodies)

    monkeypatch.setattr(pwt, "_get", get)
    out = fetch_pwt("main")
    assert calls == [False, True]
    assert len(out) == 5 and out.attrs["missing"] == ()


def test_an_unparseable_body_after_the_refetch_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(pwt, "_get", lambda url, **kw: b"<stata_dta><header>garbage")
    with pytest.warns(UserWarning, match="unreadable Stata file"):
        out = fetch_pwt("main")
    assert out.empty and "unreadable" in out.attrs["missing"][0]["reason"]


def test_the_empty_frame_on_failure_has_the_success_attrs_keys_and_dtypes(served, monkeypatch):
    good = fetch_pwt("main")
    good_m = fetch_maddison()

    def boom(url, **kw):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(pwt, "_get", boom)
    with pytest.warns(UserWarning, match="HTTP 429"):
        bad = fetch_pwt("main")
    with pytest.warns(UserWarning, match="HTTP 429"):
        bad_m = fetch_maddison()
    assert set(bad.attrs) == set(good.attrs) and bad.attrs["nbytes"] is None
    assert set(bad_m.attrs) == set(good_m.attrs)
    assert str(bad["i_xr"].dtype) == "Int64" and str(good["i_xr"].dtype) == "Int64"
    assert bad["rgdpna"].dtype == np.float64
    assert bad.attrs["missing"][0]["reason"] == "HTTP 429"


def test_former_economies_and_kosovo_are_kept_and_flagged_in_attrs(monkeypatch):
    df = pd.DataFrame({
        "countrycode": ["USA", "SUN", "ANT", "RKS"],
        "year": np.array([2000, 1980, 2000, 2010], dtype=np.int16),
        "v_gdp": [1.0e7, np.nan, np.nan, 4.0e3],
        "pop": [282.0, 266.0, 0.18, 1.8],
    })
    monkeypatch.setattr(pwt, "_get", lambda url, **kw: _dta(df))
    out = fetch_pwt("na")
    assert set(out.index.get_level_values("code")) == {"USA", "SUN", "ANT", "RKS"}
    assert out.attrs["historical_entities"] == ("ANT", "SUN")
    assert set(out.attrs["nonstandard_codes"]) == {"RKS"}
    assert "XKX" in out.attrs["nonstandard_codes"]["RKS"]
    assert out["v_gdp"].notna().groupby(level="code").any().sum() == 2


def test_caller_errors_raise_value_error_listing_the_choices():
    with pytest.raises(ValueError, match="capital"):
        fetch_pwt("bogus")
    with pytest.raises(ValueError, match="11.0"):
        fetch_pwt("main", version="9.1")
    with pytest.raises(ValueError, match="year"):
        fetch_pwt("main", start="soon")
    with pytest.raises(ValueError, match="year"):
        fetch_maddison(start="soon")


def test_the_http_seam_sends_a_non_browser_agent_through_the_house_cache(monkeypatch):
    seen = {}

    def fake_cached(url, timeout, **kw):
        seen.update(kw, url=url, timeout=timeout)
        return b""

    monkeypatch.setattr(pwt, "safe_get_bytes_cached", fake_cached)
    pwt._get("https://dataverse.nl/api/access/datafile/1", timeout=5.0, refresh=True)
    assert not seen["user_agent"].startswith("Mozilla")
    assert seen["refresh"] is True and seen["timeout"] == 5.0


def test_maddison_defaults_to_1950_with_units_and_citation_policy(served):
    out = fetch_maddison()
    assert out.index.names == ["code", "date"]
    assert out.index.get_level_values("date").min() == pd.Timestamp("1950-01-01")
    assert list(out.columns) == ["country", "region", "gdppc", "pop"]
    assert out.loc[("USA", pd.Timestamp("1950-01-01")), "pop"] == 158804.0
    a = out.attrs
    assert a["price_base"] == 2011 and a["license"] == "CC BY 4.0"
    assert a["units"]["pop"] == "thousands of persons"
    assert "2011 international dollars" in a["units"]["gdppc"]
    assert "12 countries" in a["citation_policy"] and "Bolt" in a["citation"]
    assert a["historical_entities"] == ("SUN",)
    assert ("SUN", pd.Timestamp("1951-01-01")) not in out.index


def test_maddison_without_start_reaches_back_to_year_one(served):
    out = fetch_maddison(["USA"], start=None)
    years = out.index.get_level_values("date").year
    assert list(years) == [1, 1820, 1950]
    assert np.isnan(out["gdppc"].iloc[0]) and out["pop"].iloc[0] == 680.0


def test_pwt_variables_lists_published_labels_true_units_and_notes(served):
    tab = pwt_variables("na")
    assert tab.index.name == "variable"
    assert list(tab.columns) == ["label", "units", "note"]
    assert "2017" in tab.loc["q_gdp", "label"] and "2021" in tab.loc["q_gdp", "note"]
    assert tab.loc["v_gdp", "units"] == "millions of current national currency"


def test_a_2017_equals_one_index_label_is_flagged_as_stale():
    assert pwt._stale("Investment price index for transport equipment (2017=1)")
    assert pwt._stale("Price level of fuels exports, USA GDPo in 2011=1")
    assert not pwt._stale("Real GDP at constant 2021 national prices (in mil. 2021US$)")


def test_pwt_variables_survives_a_provider_failure(monkeypatch):
    def boom(url, **kw):
        raise OSError("unreachable")

    monkeypatch.setattr(pwt, "_get", boom)
    with pytest.warns(UserWarning):
        tab = pwt_variables("capital")
    assert tab["label"].isna().all() and "Ic_Struc" in tab.index


# ---------------------------------------------------------------------------
# Live checks (opt in with ``-m network``)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_live_pwt_main_covers_185_economies_from_1950():
    out = fetch_pwt("main")
    assert out.attrs["missing"] == ()
    assert out.index.get_level_values("code").nunique() == 185
    assert out.index.get_level_values("date").min() == pd.Timestamp("1950-01-01")


@pytest.mark.network
def test_live_capital_detail_investment_adds_up_to_na_gfcf():
    cap = fetch_pwt("capital", codes=["USA", "MEX"])
    na = fetch_pwt("na", codes=["USA", "MEX"])
    ic = cap[[f"Ic_{a}" for a in ("Struc", "Mach", "TraEq", "Other")]].sum(axis=1, min_count=4)
    both = pd.concat([ic.rename("ic"), na["v_gfcf"]], axis=1).dropna()
    assert len(both) > 50
    assert np.allclose(both["ic"], both["v_gfcf"], rtol=1e-5)


@pytest.mark.network
def test_live_maddison_has_169_economies():
    out = fetch_maddison(start=None)
    assert out.index.get_level_values("code").nunique() == 169
