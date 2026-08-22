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
                                            QNA_LABOR, qna_meta, qna_panel)

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


def _labor_rows(code: str) -> list[dict]:
    """Synthetic labour rows: persons and hours, employees and self-employed.

    Persons come in thousands (UNIT_MULT 3) and hours in millions (6), which
    is how the OECD publishes them, so the test also pins the rescaling. MEX
    publishes the block unadjusted; the fixture keeps AUS heads-only and CAN
    hours-only, the two real gaps in this flow.
    """
    out = []
    for name, (txn, unit, _) in QNA_LABOR.items():
        if code == "AUS" and unit == "H":
            continue
        if code == "CAN" and unit == "PS":
            continue
        base = {"emp": 20_000.0, "emp_employees": 16_000.0,
                "emp_selfemp": 4_000.0, "hours": 8_000.0,
                "hours_employees": 6_200.0, "hours_selfemp": 1_800.0}[name]
        for t, period in enumerate(_QUARTERS):
            out.append({
                "FREQ": "Q", "ADJUSTMENT": "N" if code in ("MEX", "AUS", "CAN") else "Y",
                "REF_AREA": code, "SECTOR": "S1", "COUNTERPART_SECTOR": "S1",
                "TRANSACTION": txn, "INSTR_ASSET": "_Z", "ACTIVITY": "_T",
                "EXPENDITURE": "_Z", "UNIT_MEASURE": unit, "PRICE_BASE": "_Z",
                "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0111",
                "TIME_PERIOD": period, "OBS_VALUE": base * (1.0 + 0.005 * t),
                "REF_YEAR_PRICE": None, "UNIT_MULT": 3 if unit == "PS" else 6,
                "CURRENCY": "_Z",
            })
    return out


@pytest.fixture
def fake_sdmx(monkeypatch):
    """Patch the SDMX helper so no network is touched; count the calls."""
    calls: list[tuple[str, str]] = []

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        calls.append((key, start_period))
        is_assets = "GFCF_ASSET" in agency_flow
        is_labor = "EMPDC" in agency_flow
        sector = key.split(".")[3]
        rows: list[dict] = []
        for code, (vol_base, deflator) in _COUNTRIES.items():
            if code not in key and key.split(".")[2] != "":
                continue
            if is_labor:
                rows += _labor_rows(code)
            elif is_assets:
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


def test_labor_off_by_default(fake_sdmx):
    p = qna_panel(["USA"], start="2019")
    assert not any(c in p.columns for c in QNA_LABOR)
    # one request per institutional sector, none for the labour dataflow
    assert len(fake_sdmx) == len({s for _, s, _ in QNA_COMPONENTS.values()})


def test_labor_adds_heads_and_hours_with_no_deflator(fake_sdmx):
    p = qna_panel(["USA", "MEX"], start="2019", labor=True, real=True)
    for name in QNA_LABOR:
        assert name in p.columns
        # counts of people and hours carry no price, so no deflator, no volume
        assert f"{name}_defl" not in p.columns
        assert f"{name}_real" not in p.columns


def test_labor_puts_persons_and_hours_on_one_scale(fake_sdmx):
    """UNIT_MULT 3 for persons and 6 for hours, both preserved as published."""
    p = qna_panel(["USA"], start="2019", labor=True)
    assert p.loc["USA", "emp"].iloc[0] == pytest.approx(20_000.0)     # thousands
    assert p.loc["USA", "hours"].iloc[0] == pytest.approx(8_000.0)    # millions
    # the split adds up in the fixture, as it does in the accounts
    np.testing.assert_allclose(
        p.loc["USA", "emp_employees"] + p.loc["USA", "emp_selfemp"],
        p.loc["USA", "emp"])


def test_labor_pins_activity_to_the_total_economy(fake_sdmx):
    """The flow publishes every ISIC section; asking for all of them is 12x
    the response for the one aggregate the panel wants."""
    qna_panel(["USA"], start="2019", labor=True)
    labor_keys = [k for k, _ in fake_sdmx if "_T" in k]
    assert labor_keys and all(k.split(".")[7] == "_T" for k in labor_keys)


def test_labor_does_not_widen_the_country_index(monkeypatch):
    """A reference area that publishes labour but not the money block must not
    enter the index. The two flows have different rosters — 49 areas are asked
    for and 39 answer the labour one — so this has to be a real asymmetry in
    the fixture, not two calls over the same country list."""
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            # NOR publishes labour and nothing else: it must be dropped.
            return pd.DataFrame(_labor_rows("USA") + _labor_rows("NOR"))
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("USA", "L", 120.0)
                             if r["SECTOR"] == sector])

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    p = qna_panel(["USA"], start="2019", labor=True)
    assert sorted(p.index.get_level_values("code").unique()) == ["USA"]
    assert "NOR" not in set(qna_meta(p)["code"])


def test_labor_does_not_disturb_the_money_columns(fake_sdmx):
    """The labour rows join on a price base of their own ('_Z'); nothing in
    the expenditure block may move because they are there."""
    base = qna_panel(["USA", "MEX"], start="2019", real=True)
    with_labor = qna_panel(["USA", "MEX"], start="2019", real=True, labor=True)
    shared = [c for c in base.columns if c in with_labor.columns]
    pd.testing.assert_frame_equal(base[shared], with_labor[shared])


def test_meta_reports_labor_adjustment_separately(fake_sdmx):
    """MEX publishes the labour block unadjusted while its aggregates are SA,
    and the currency still comes from the money block, which has one."""
    meta = qna_meta(qna_panel(["USA", "MEX"], start="2019",
                              labor=True)).set_index("code")
    assert meta.loc["MEX", "sa"] == "oecd"
    assert meta.loc["MEX", "sa_labor"] == "none"
    assert meta.loc["USA", "sa_labor"] == "oecd"
    assert meta.loc["MEX", "currency"] == "MXN"


def test_labor_survives_a_country_that_publishes_only_one_unit(monkeypatch):
    """Australia publishes heads and no hours, Canada hours and no heads."""
    calls: list[str] = []

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        calls.append(key)
        sector = key.split(".")[3]
        if "EMPDC" in agency_flow:
            return pd.DataFrame(_labor_rows("AUS") + _labor_rows("CAN"))
        rows = []
        for code in ("AUS", "CAN"):
            rows += [r for r in _rows(code, "L", 120.0) if r["SECTOR"] == sector]
        return pd.DataFrame(rows)

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    p = qna_panel(["AUS", "CAN"], start="2019", labor=True, sa="prefer")
    assert p.loc["AUS", "emp"].notna().all()
    assert p.loc["AUS", "hours"].isna().all()
    assert p.loc["CAN", "hours"].notna().all()
    assert p.loc["CAN", "emp"].isna().all()


def _mixed_adjustment_rows(code: str) -> list[dict]:
    """Korea's real shape: the total is adjusted at source, the parts are not.

    Every ``EMP``/``PS`` quarter exists twice, once adjusted and once raw with
    a seasonal wedge on it, while ``SAL``/``SELF`` exist raw only. Picking the
    adjusted total and the raw parts is what breaks ``emp = SAL + SELF``.
    """
    season = {1: 0.97, 2: 1.02, 3: 1.01, 4: 1.00}
    rows, adjusted = [], []
    for r in _labor_rows(code):
        if r["UNIT_MEASURE"] != "PS":
            continue
        r = dict(r, ADJUSTMENT="N")
        smooth = r["OBS_VALUE"]
        # the same factor on all three, so the RAW triple still adds up
        r["OBS_VALUE"] = smooth * season[int(r["TIME_PERIOD"][-1])]
        rows.append(r)
        if r["TRANSACTION"] == "EMP":
            adjusted.append(dict(r, ADJUSTMENT="Y", OBS_VALUE=smooth))
    return rows + adjusted


@pytest.fixture
def fake_mixed_sdmx(monkeypatch):
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return pd.DataFrame(_mixed_adjustment_rows("USA"))
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("USA", "L", 120.0)
                             if r["SECTOR"] == sector])

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)


def test_labor_keeps_a_decomposition_on_one_adjustment(fake_mixed_sdmx):
    """Heads resolve as one family: a total adjusted at source but with raw
    components falls back to raw for all three, so the parts still sum to the
    total. Mixing the two is a seasonal artefact in ``emp_selfemp / emp``."""
    p = qna_panel(["USA"], start="2019", labor=True)
    np.testing.assert_allclose(
        p.loc["USA", "emp_employees"] + p.loc["USA", "emp_selfemp"],
        p.loc["USA", "emp"])


def test_labor_family_fallback_is_reported_as_unadjusted(fake_mixed_sdmx):
    """The cost of keeping the identity is visible, not silent."""
    assert qna_meta(qna_panel(["USA"], start="2019",
                              labor=True)).set_index("code").loc["USA", "sa_labor"] == "none"


def test_money_blocks_still_resolve_adjustment_per_series(fake_sdmx):
    """`sa_family` is passed only on the labour call. The money blocks keep
    the per-series rule they had before, which several reference areas rely
    on, so turning labour on must not change how they are picked."""
    base = qna_panel(["USA", "MEX"], start="2019", real=True, income=True)
    with_labor = qna_panel(["USA", "MEX"], start="2019", real=True,
                           income=True, labor=True)
    shared = [c for c in base.columns if c in with_labor.columns]
    pd.testing.assert_frame_equal(base[shared], with_labor[shared])


def test_blank_unit_mult_does_not_rescale_persons(monkeypatch):
    """A missing UNIT_MULT must fall back to *this* block's scale. Filling it
    with the money block's 6 would multiply a head count by 10**(6-3)."""
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            rows = _labor_rows("USA")
            for r in rows:
                r["UNIT_MULT"] = float("nan")
            return pd.DataFrame(rows)
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("USA", "L", 120.0)
                             if r["SECTOR"] == sector])

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    p = qna_panel(["USA"], start="2019", labor=True)
    assert p.loc["USA", "emp"].iloc[0] == pytest.approx(20_000.0)
    assert p.loc["USA", "hours"].iloc[0] == pytest.approx(8_000.0)


def _long_nsa_labor_frame(n_years: int = 12) -> pd.DataFrame:
    """The labour block for the same country, unadjusted, with a seasonal."""
    quarters = [f"{y}-Q{q}" for y in range(2005, 2005 + n_years) for q in (1, 2, 3, 4)]
    season = {1: 0.94, 2: 1.04, 3: 1.00, 4: 1.02}
    base = {"emp": 20_000.0, "emp_employees": 16_000.0, "emp_selfemp": 4_000.0,
            "hours": 8_000.0, "hours_employees": 6_200.0, "hours_selfemp": 1_800.0}
    rows = []
    for name, (txn, unit, _) in QNA_LABOR.items():
        for t, period in enumerate(quarters):
            rows.append({
                "FREQ": "Q", "ADJUSTMENT": "N", "REF_AREA": "SAU",
                "SECTOR": "S1", "COUNTERPART_SECTOR": "S1",
                "TRANSACTION": txn, "INSTR_ASSET": "_Z", "ACTIVITY": "_T",
                "EXPENDITURE": "_Z", "UNIT_MEASURE": unit, "PRICE_BASE": "_Z",
                "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0111",
                "TIME_PERIOD": period,
                "OBS_VALUE": base[name] * (1.0 + 0.004 * t) * season[int(period[-1])],
                "REF_YEAR_PRICE": None, "UNIT_MULT": 3 if unit == "PS" else 6,
                "CURRENCY": "_Z",
            })
    return pd.DataFrame(rows)


def test_labor_x13_adjusts_the_reference_areas_that_publish_it_raw(monkeypatch):
    """The release's promise: ten reference areas publish this block with no
    adjusted variant at all, and sa="x13" is what makes them usable. It has to
    reach the labour rows, label them as puremacro's own work in `sa_labor`,
    and actually remove the seasonal."""
    money, labor = _long_nsa_frame(), _long_nsa_labor_frame()

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return labor
        return money[money["SECTOR"] == key.split(".")[3]]

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    raw = qna_panel(["SAU"], start="2005", labor=True)
    adj = qna_panel(["SAU"], start="2005", labor=True, sa="x13")
    assert qna_meta(raw).set_index("code").loc["SAU", "sa_labor"] == "none"
    assert qna_meta(adj).set_index("code").loc["SAU", "sa_labor"] == "puremacro"

    def spread(s):
        d = np.log(s).diff().dropna()
        m = d.groupby(d.index.quarter).mean()
        return m.max() - m.min()

    assert spread(adj.loc["SAU", "emp"]) < 0.25 * spread(raw.loc["SAU", "emp"])
    # and the family still adds up after puremacro adjusts all three
    ratio = ((adj.loc["SAU", "emp_employees"] + adj.loc["SAU", "emp_selfemp"])
             / adj.loc["SAU", "emp"])
    assert ratio.between(0.99, 1.01).all()


def test_labor_units_and_scale_map_agree():
    """QNA_LABOR_UNITS must cover every unit QNA_LABOR names, or _tidy has no
    target for it and the series would pass through unscaled."""
    assert {u for _, u, _ in QNA_LABOR.values()} == set(mod.QNA_LABOR_UNITS)


def test_labor_columns_survive_long_form(fake_sdmx):
    p = qna_panel(["USA"], start="2019", labor=True, long=True)
    assert set(QNA_LABOR).issubset(set(p["variable"]))


def _money_only_fake(labor_rows):
    """A fake SDMX whose labour flow returns `labor_rows` and whose money
    flows return a plain single-country expenditure block."""
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return pd.DataFrame(labor_rows)
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("USA", "L", 120.0)
                             if r["SECTOR"] == sector])
    return _fake


def test_labor_rescales_a_reference_area_on_another_unit_mult(monkeypatch):
    """UNIT_MULT is the power of ten the published figure carries. A head
    count published in units (UNIT_MULT 0) has to arrive in thousands like
    everyone else's, or the same ratio across two countries is off by 1000."""
    rows = _labor_rows("USA")
    for r in rows:
        if r["UNIT_MEASURE"] == "PS":            # same people, different scale
            r["UNIT_MULT"], r["OBS_VALUE"] = 0, r["OBS_VALUE"] * 1000.0
    monkeypatch.setattr(mod, "get_sdmx_csv", _money_only_fake(rows))
    p = qna_panel(["USA"], start="2019", labor=True)
    assert p.loc["USA", "emp"].iloc[0] == pytest.approx(20_000.0)
    assert p.loc["USA", "hours"].iloc[0] == pytest.approx(8_000.0)


def test_meta_span_covers_a_labour_block_that_outruns_the_money_block(monkeypatch):
    """Luxembourg and South Africa publish the labour block one quarter past
    their expenditure block, so the panel carries a quarter the money rows do
    not. `n_obs` / `first` / `last` describe the panel, not one block of it."""
    rows = _labor_rows("USA")
    rows += [dict(r, TIME_PERIOD="2021-Q1") for r in rows
             if r["TIME_PERIOD"] == _QUARTERS[-1]]
    monkeypatch.setattr(mod, "get_sdmx_csv", _money_only_fake(rows))
    p = qna_panel(["USA"], start="2019", labor=True)
    meta = qna_meta(p).set_index("code")
    assert p.loc["USA"].index.max() == pd.Timestamp("2021-01-01")
    assert meta.loc["USA", "n_obs"] == len(p.loc["USA"])
    assert pd.Timestamp(meta.loc["USA", "last"]) == p.loc["USA"].index.max()


def test_sibling_flow_keys_keep_their_wide_open_shape(fake_sdmx):
    """`tail` defaults to a wide-open key. Only the labour flow pins a
    dimension; every other flow must go on asking for all of them, or a
    sibling block silently narrows."""
    qna_panel(["USA"], start="2019", assets=True, durability=True,
              output=True, income=True, labor=True)
    keys = [k for k, _ in fake_sdmx]
    labor = [k for k in keys if "_T" in k.split(".")]
    money = [k for k in keys if k not in labor]
    assert labor and money
    # Past the reference area — and, for the expenditure flow, the
    # institutional sector that sits right after it — every money key is
    # still wide open.
    assert all(set(k.split(".")[4:]) == {""} for k in money)
    # The labour key pins ACTIVITY and nothing else.
    for k in labor:
        assert k.split(".")[7] == "_T"
        assert set(k.split(".")[4:7]) | set(k.split(".")[8:]) == {""}


def _labor_rows_on_basis(code: str, hours_factor: float) -> list[dict]:
    """Labour rows whose hours are published on a mislabelled time base.

    The fixture's honest ratio is 400 hours per worker per quarter. Dividing
    the hours by 13 is Chile (published per week); multiplying by 4 is Costa
    Rica (published at an annual rate). Nothing in the row says so — same
    UNIT_MEASURE, same UNIT_MULT, same everything.
    """
    rows = _labor_rows(code)
    for r in rows:
        if r["UNIT_MEASURE"] == "H":
            r["OBS_VALUE"] = r["OBS_VALUE"] * hours_factor
    return rows


def _labor_fake(rows):
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return pd.DataFrame(rows)
        sector = key.split(".")[3]
        return pd.DataFrame([r for r in _rows("USA", "L", 120.0)
                             if r["SECTOR"] == sector])
    return _fake


@pytest.mark.parametrize("factor,scale", [(1 / 13, 13.0), (4.0, 0.25)])
def test_hours_published_on_another_time_base_are_put_on_a_quarterly_one(
        monkeypatch, factor, scale):
    """Chile publishes hours per week and Costa Rica at an annual rate. Both
    are labelled exactly like a quarterly figure, so `hours / emp` is wrong by
    13x and 4x unless the level itself gives it away."""
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _labor_fake(_labor_rows_on_basis("USA", factor)))
    p = qna_panel(["USA"], start="2019", labor=True)
    ratio = (p.loc["USA", "hours"] * 1e3 / p.loc["USA", "emp"]).median()
    assert ratio == pytest.approx(400.0)
    assert qna_meta(p).set_index("code").loc["USA", "hours_scale"] == scale


def test_hours_rescale_can_be_turned_off(monkeypatch):
    """The correction is a judgement about a published number, so there has to
    be a way to see the number as published."""
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _labor_fake(_labor_rows_on_basis("USA", 1 / 13)))
    p = qna_panel(["USA"], start="2019", labor=True, hours_rescale=False)
    ratio = (p.loc["USA", "hours"] * 1e3 / p.loc["USA", "emp"]).median()
    assert ratio == pytest.approx(400.0 / 13)


def test_a_plausible_country_is_never_rescaled(fake_sdmx):
    """The band has to be far enough out that a country which merely works
    short weeks is left alone. 400 h/quarter is ordinary and must not move."""
    p = qna_panel(["USA", "MEX"], start="2019", labor=True)
    meta = qna_meta(p).set_index("code")
    for code in ("USA", "MEX"):
        assert meta.loc[code, "hours_scale"] == 1.0
        assert (p.loc[code, "hours"] * 1e3 / p.loc[code, "emp"]).median() == pytest.approx(400.0)


def test_hours_are_left_alone_when_the_country_publishes_no_heads(monkeypatch):
    """Canada publishes hours and no employment, so the ratio that detects a
    mislabelled basis cannot be formed. Guessing is worse than leaving it."""
    rows = [r for r in _labor_rows_on_basis("USA", 1 / 13)
            if r["UNIT_MEASURE"] == "H"]
    monkeypatch.setattr(mod, "get_sdmx_csv", _labor_fake(rows))
    p = qna_panel(["USA"], start="2019", labor=True)
    assert p.loc["USA", "hours"].iloc[0] == pytest.approx(8_000.0 / 13)
    assert qna_meta(p).set_index("code").loc["USA", "hours_scale"] == 1.0


def test_hours_basis_is_judged_only_where_heads_and_hours_overlap(monkeypatch):
    """Ten of the 33 reference areas that publish both put them on different
    spans — Estonia's hours start in 1998 against heads from 1995. The ratio
    that detects a mislabelled basis has to be formed on the overlap only; a
    quarter with hours and no heads is not evidence about anything."""
    rows = _labor_rows_on_basis("USA", 1 / 13)
    keep = set(_QUARTERS[-3:])
    rows = [r for r in rows
            if r["UNIT_MEASURE"] == "H" or r["TIME_PERIOD"] in keep]
    monkeypatch.setattr(mod, "get_sdmx_csv", _labor_fake(rows))
    p = qna_panel(["USA"], start="2019", labor=True)
    overlap = (p.loc["USA", "hours"] * 1e3 / p.loc["USA", "emp"]).dropna()
    assert len(overlap) == 3 and overlap.median() == pytest.approx(400.0)
    assert qna_meta(p).set_index("code").loc["USA", "hours_scale"] == 13.0


def test_an_unrecognisable_hours_basis_is_left_as_published(monkeypatch):
    """A correction is applied only if it lands somewhere believable. A level
    no candidate factor explains is a series to leave alone and report, not to
    bend until it looks reasonable."""
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _labor_fake(_labor_rows_on_basis("USA", 1e6)))
    p = qna_panel(["USA"], start="2019", labor=True)
    assert p.loc["USA", "hours"].iloc[0] == pytest.approx(8_000.0 * 1e6)
    assert qna_meta(p).set_index("code").loc["USA", "hours_scale"] == 1.0


def test_hours_scale_distinguishes_blank_from_checked_and_unchanged(monkeypatch):
    """AUS publishes heads and no hours, so there is no scale to report — as
    against USA, which does publish hours and was checked and left alone, and
    reads 1.0. Blank and 1.0 are different answers and the fixture has to hold
    both for the assertion to mean anything."""
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return pd.DataFrame(_labor_rows("USA") + _labor_rows("AUS"))
        sector = key.split(".")[3]
        rows = []
        for code in ("USA", "AUS"):
            rows += [r for r in _rows(code, "L", 120.0) if r["SECTOR"] == sector]
        return pd.DataFrame(rows)

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    meta = qna_meta(qna_panel(["USA", "AUS"], start="2019",
                              labor=True)).set_index("code")
    assert meta.loc["AUS", "hours_scale"] == ""       # publishes no hours
    assert meta.loc["USA", "hours_scale"] == 1.0      # checked, unchanged


def test_hours_scale_is_blank_when_the_labour_block_is_off(fake_sdmx):
    meta = qna_meta(qna_panel(["USA", "MEX"], start="2019")).set_index("code")
    assert (meta["hours_scale"] == "").all()


def test_rescaling_touches_only_the_hours_columns(monkeypatch):
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _labor_fake(_labor_rows_on_basis("USA", 4.0)))
    p = qna_panel(["USA"], start="2019", labor=True)
    q = qna_panel(["USA"], start="2019", labor=True, hours_rescale=False)
    heads = ["emp", "emp_employees", "emp_selfemp"]
    pd.testing.assert_frame_equal(p[heads], q[heads])
    money = [c for c in p.columns if not c.startswith(("emp", "hours"))]
    pd.testing.assert_frame_equal(p[money], q[money])
    # and every hours column moved by the same factor, so the split still sums
    np.testing.assert_allclose(
        p.loc["USA", "hours_employees"] + p.loc["USA", "hours_selfemp"],
        p.loc["USA", "hours"])
