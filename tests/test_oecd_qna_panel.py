"""Offline tests for :func:`puremacro.fetch.oecd_qna_panel.qna_panel`.

The SDMX round-trip is mocked with a synthetic ``csvfilewithlabels`` frame
that reproduces the shape of the real OECD response, including the two cases
the fetcher exists to handle: a chain-linked-volume country (USA, ``L``) and a
fixed-base country (MEX, ``Q``) that the volume-only fetcher silently drops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import oecd_qna_panel as mod
from puremacro.fetch.oecd_qna_panel import (QNA_ASSETS, QNA_COMPONENTS,
                                            qna_meta, qna_panel)

_QUARTERS = [f"{y}-Q{q}" for y in (2019, 2020) for q in (1, 2, 3, 4)]

# (code, volume price base, deflator level the synthetic data encodes)
_COUNTRIES = {"USA": ("L", 125.0), "MEX": ("Q", 140.0)}


def _rows(code: str, vol_base: str, deflator: float) -> list[dict]:
    """Synthetic SDMX rows: one V row and one volume row per component/quarter."""
    out = []
    for i, (name, (txn, sector, _)) in enumerate(QNA_COMPONENTS.items()):
        for t, period in enumerate(_QUARTERS):
            volume = 1000.0 * (i + 1) + 10.0 * t
            for price_base, value in ((vol_base, volume),
                                      ("V", volume * deflator / 100.0)):
                out.append({
                    "FREQ": "Q", "ADJUSTMENT": "Y", "REF_AREA": code,
                    "SECTOR": sector, "COUNTERPART_SECTOR": "S1",
                    "TRANSACTION": txn, "ACTIVITY": "_Z", "EXPENDITURE": "_Z",
                    "UNIT_MEASURE": "XDC", "PRICE_BASE": price_base,
                    "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0102",
                    "TIME_PERIOD": period, "OBS_VALUE": value,
                    "REF_YEAR_PRICE": 2015, "UNIT_MULT": 6,
                    "CURRENCY": "USD" if code == "USA" else "MXN",
                })
    return out


def _asset_rows(code: str, vol_base: str, deflator: float) -> list[dict]:
    """Synthetic GFCF-by-asset rows. Equipment prices fall, structures rise,
    so a test can tell the two apart. MEX publishes them unadjusted."""
    out = []
    for i, (name, (asset, _)) in enumerate(QNA_ASSETS.items()):
        drift = 0.9 if name == "inv_equip" else 1.1
        for t, period in enumerate(_QUARTERS):
            volume = 100.0 * (i + 1) + t
            p = deflator * (drift ** t)
            for price_base, value in ((vol_base, volume), ("V", volume * p / 100.0)):
                out.append({
                    "FREQ": "Q", "ADJUSTMENT": "N" if code == "MEX" else "Y",
                    "REF_AREA": code, "SECTOR": "S1", "COUNTERPART_SECTOR": "S1",
                    "TRANSACTION": "P51G", "INSTR_ASSET": asset, "ACTIVITY": "_Z",
                    "EXPENDITURE": "_Z", "UNIT_MEASURE": "XDC",
                    "PRICE_BASE": price_base, "TRANSFORMATION": "N",
                    "TABLE_IDENTIFIER": "T0102", "TIME_PERIOD": period,
                    "OBS_VALUE": value, "REF_YEAR_PRICE": 2015, "UNIT_MULT": 6,
                    "CURRENCY": "USD" if code == "USA" else "MXN",
                })
    return out


@pytest.fixture
def fake_sdmx(monkeypatch):
    """Patch the SDMX helper so no network is touched; count the calls."""
    calls: list[tuple[str, str]] = []

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        calls.append((key, start_period))
        is_assets = "GFCF_ASSET" in agency_flow
        sector = key.split(".")[3]
        rows: list[dict] = []
        for code, (vol_base, deflator) in _COUNTRIES.items():
            if code not in key and key.split(".")[2] != "":
                continue
            if is_assets:
                rows += _asset_rows(code, vol_base, deflator)
            else:
                rows += [r for r in _rows(code, vol_base, deflator)
                         if r["SECTOR"] == sector]
        return pd.DataFrame(rows)

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    return calls


def test_panel_shape_and_columns(fake_sdmx):
    p = qna_panel(["USA", "MEX"], start="2019")
    assert list(p.index.names) == ["code", "date"]
    assert sorted(p.index.get_level_values("code").unique()) == ["MEX", "USA"]
    assert p.index.get_level_values("date").nunique() == len(_QUARTERS)
    # every component in current prices, and a deflator for each
    for name in QNA_COMPONENTS:
        assert name in p.columns
        assert f"{name}_defl" in p.columns
    # volumes stay hidden unless asked for
    assert not any(c.endswith("_real") for c in p.columns)
    assert "gdp_real" in qna_panel(["USA"], start="2019", real=True).columns


def test_deflator_is_nominal_over_volume(fake_sdmx):
    """P = 100 * nominal / volume, exactly as encoded in the fixture."""
    p = qna_panel(["USA", "MEX"], start="2019", real=True)
    for code, (_, deflator) in _COUNTRIES.items():
        got = p.loc[code, "gdp_defl"]
        assert np.allclose(got, deflator)
        # and real magnitudes round-trip out of nominal + deflator
        implied = 100.0 * p.loc[code, "gdp"] / p.loc[code, "gdp_defl"]
        assert np.allclose(implied, p.loc[code, "gdp_real"])


def test_fixed_base_country_survives(fake_sdmx):
    """MEX publishes fixed-base (Q) volumes only and must not be dropped."""
    p = qna_panel(["USA", "MEX"], start="2019")
    meta = qna_meta(p).set_index("code")
    assert meta.loc["USA", "price_base"] == "L"
    assert meta.loc["MEX", "price_base"] == "Q"
    assert p.loc["MEX"].notna().all().all()


def test_meta_reports_currency_and_adjustment(fake_sdmx):
    meta = qna_meta(qna_panel(["USA", "MEX"], start="2019")).set_index("code")
    assert meta.loc["USA", "currency"] == "USD"
    assert meta.loc["MEX", "currency"] == "MXN"
    assert bool(meta.loc["USA", "sa"]) is True
    assert meta.loc["USA", "n_obs"] == len(_QUARTERS)
    assert meta.loc["USA", "price_ref_year"] == 2015


def test_seasonally_adjusted_rows_win(monkeypatch):
    """When both ADJUSTMENT=Y and N exist, the adjusted series is the one kept."""
    rows = _rows("USA", "L", 125.0)
    nsa = []
    for r in rows:
        r2 = dict(r)
        r2["ADJUSTMENT"] = "N"
        r2["OBS_VALUE"] = r["OBS_VALUE"] * 2.0   # unmistakably different
        nsa.append(r2)
    frame = pd.DataFrame(rows + nsa)

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        return frame[frame["SECTOR"] == key.split(".")[3]]

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    p = qna_panel(["USA"], start="2019")
    # the NSA rows are twice the SA ones; deflators are unchanged either way,
    # so the level is what discriminates.
    assert p.loc[("USA", pd.Timestamp("2019-01-01")), "gdp"] == pytest.approx(1250.0)
    assert bool(qna_meta(p).set_index("code").loc["USA", "sa"]) is True


def test_long_form(fake_sdmx):
    p = qna_panel(["USA"], start="2019", long=True)
    assert list(p.columns) == ["code", "date", "variable", "value"]
    assert set(p["variable"]) >= {"gdp", "gdp_defl"}


def test_empty_download_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        lambda *a, **k: pd.DataFrame())
    p = qna_panel(["USA"], start="2019")
    assert p.empty
    assert qna_meta(p).empty              # never raises, always documented


def test_chunks_requests_by_ten_countries(fake_sdmx):
    codes = [f"C{i:02d}" for i in range(23)]
    qna_panel(codes, start="2019")
    # 3 institutional sectors x ceil(23/10) chunks
    assert len(fake_sdmx) == 3 * 3


def test_assets_off_by_default(fake_sdmx):
    p = qna_panel(["USA"], start="2019")
    assert not any(c.startswith("inv_equip") for c in p.columns)
    # one request per institutional sector, none for the asset dataflow
    assert len(fake_sdmx) == len({s for _, s, _ in QNA_COMPONENTS.values()})


def test_assets_adds_split_and_deflators(fake_sdmx):
    p = qna_panel(["USA", "MEX"], start="2019", assets=True)
    for name in QNA_ASSETS:
        assert name in p.columns
        assert f"{name}_defl" in p.columns
    # equipment prices fall and structures rise, as the fixture encodes
    eq = p.loc["USA", "inv_equip_defl"]
    st = p.loc["USA", "inv_struct_defl"]
    assert eq.iloc[-1] < eq.iloc[0]
    assert st.iloc[-1] > st.iloc[0]


def test_assets_do_not_widen_the_country_index(fake_sdmx):
    """A reference area present only in the asset flow must not appear."""
    base = qna_panel(["USA", "MEX"], start="2019")
    with_assets = qna_panel(["USA", "MEX"], start="2019", assets=True)
    assert (sorted(base.index.get_level_values("code").unique())
            == sorted(with_assets.index.get_level_values("code").unique()))


def test_meta_reports_asset_adjustment_separately(fake_sdmx):
    """MEX publishes the asset split unadjusted while its aggregates are SA."""
    meta = qna_meta(qna_panel(["USA", "MEX"], start="2019", assets=True)).set_index("code")
    assert meta.loc["MEX", "sa"] == "oecd"
    assert meta.loc["MEX", "sa_detail"] == "none"
    assert meta.loc["USA", "sa_detail"] == "oecd"


def _long_nsa_frame(n_years: int = 12) -> pd.DataFrame:
    """One country, unadjusted only, with a visible quarterly seasonal."""
    quarters = [f"{y}-Q{q}" for y in range(2005, 2005 + n_years) for q in (1, 2, 3, 4)]
    season = {1: 0.92, 2: 1.03, 3: 0.99, 4: 1.06}
    rows = []
    for name, (txn, sector, _) in QNA_COMPONENTS.items():
        for t, period in enumerate(quarters):
            q = int(period[-1])
            volume = (1000.0 + 5.0 * t) * season[q]
            for price_base, value in (("L", volume), ("V", volume * 1.25)):
                rows.append({
                    "FREQ": "Q", "ADJUSTMENT": "N", "REF_AREA": "SAU",
                    "SECTOR": sector, "COUNTERPART_SECTOR": "S1",
                    "TRANSACTION": txn, "INSTR_ASSET": "_Z", "ACTIVITY": "_Z",
                    "EXPENDITURE": "_Z", "UNIT_MEASURE": "XDC",
                    "PRICE_BASE": price_base, "TRANSFORMATION": "N",
                    "TABLE_IDENTIFIER": "T0102", "TIME_PERIOD": period,
                    "OBS_VALUE": value, "REF_YEAR_PRICE": 2015, "UNIT_MULT": 6,
                    "CURRENCY": "SAR",
                })
    return pd.DataFrame(rows)


def test_sa_x13_adjusts_what_the_source_left_unadjusted(monkeypatch):
    """A reference area publishing nothing adjusted (CHN, SAU) still comes
    through, seasonally adjusted here, with no X-13 binary installed."""
    frame = _long_nsa_frame()

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        return frame[frame["SECTOR"] == key.split(".")[3]]

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)

    raw = qna_panel(["SAU"], start="2005")
    adj = qna_panel(["SAU"], start="2005", sa="x13")
    assert qna_meta(raw).set_index("code").loc["SAU", "sa"] == "none"
    assert qna_meta(adj).set_index("code").loc["SAU", "sa"] == "puremacro"

    # the engine report names a real adjuster, never the STL last resort
    engines = set(adj.attrs["sa_engines"].values())
    assert engines and "stl" not in engines

    # and the seasonal is actually gone: quarter-of-year means of the
    # detrended series collapse toward each other
    def spread(s):
        d = np.log(s).diff().dropna()
        return d.groupby(d.index.quarter).mean().max() - d.groupby(d.index.quarter).mean().min()

    assert spread(adj.loc["SAU", "gdp"]) < 0.25 * spread(raw.loc["SAU", "gdp"])


def test_sa_min_gain_is_opt_in(monkeypatch):
    """Trading a short official adjustment for a longer home-made one is
    opt-in: sa='x13' alone never discards an adjustment the source published."""
    rows = _rows("USA", "L", 125.0)
    short_y, long_n = [], []
    for r in rows:
        y = dict(r); y["ADJUSTMENT"] = "Y"
        n = dict(r); n["ADJUSTMENT"] = "N"
        # the adjusted vintage exists for the last two quarters only
        if r["TIME_PERIOD"] in _QUARTERS[-2:]:
            short_y.append(y)
        long_n.append(n)
    frame = pd.DataFrame(short_y + long_n)

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        return frame[frame["SECTOR"] == key.split(".")[3]]

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    # sa='prefer' takes the adjusted vintage even though it is 6 quarters shorter
    assert len(qna_panel(["USA"], start="2019")) == 2
    # sa='x13' alone does NOT trade an official adjustment for ours...
    assert len(qna_panel(["USA"], start="2019", sa="x13")) == 2
    # ...that is opt-in, and then the longer raw series wins
    assert len(qna_panel(["USA"], start="2019", sa="x13",
                         sa_min_gain=1)) == len(_QUARTERS)


def test_sa_rejects_unknown_mode(fake_sdmx):
    with pytest.raises(ValueError, match="sa must be"):
        qna_panel(["USA"], start="2019", sa="seats")


def test_slices_survive_pandas_concat(fake_sdmx):
    """attrs must stay comparable: a DataFrame in .attrs makes pd.concat on any
    slice of the panel raise "truth value of a DataFrame is ambiguous"."""
    p = qna_panel(["USA", "MEX"], start="2019")
    parts = [p.loc["USA", "gdp"].rename("a"), p.loc["USA", "cons_hh"].rename("b")]
    joined = pd.concat(parts, axis=1)
    assert list(joined.columns) == ["a", "b"]
    assert not pd.concat([p.loc["USA"], p.loc["MEX"]]).empty
