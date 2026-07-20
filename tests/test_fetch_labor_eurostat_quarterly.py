"""Tests for puremacro.fetch.labor_eurostat.fetch_eurostat_lfs_quarterly_panel.

Offline / fixture-driven. Mirrors test_fetch_labor_eurostat.py's style:
the live network is monkeypatched out via a stub of sdmx_get that returns
a synthetic SDMX-CSV DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_lfsi_emp_q_csv() -> pd.DataFrame:
    """Synthetic SDMX-CSV-shaped frame for lfsi_emp_q with indic_em=EMP_LFS.

    3 quarters × 1 country (DE) × 1 sex (T) × 4 valid ages plus 1
    aggregate (Y_GE15 — rejected by lfsi_emp_q's age constraint) we
    expect dropped.
    """
    rows = []
    for age, val_base in [
        ("Y15-24", 25.0),
        ("Y25-54", 80.0),
        ("Y55-64", 60.0),
        ("Y15-64", 70.0),
        ("Y_GE15", 999.0),  # not in dataflow's age constraint — must be dropped
    ]:
        for q_str, q_val_offset in [("2020-Q1", 0.0), ("2020-Q2", -1.0), ("2020-Q3", 1.0)]:
            rows.append({
                "DATAFLOW": "ESTAT:LFSI_EMP_Q(1.0)",
                "freq": "Q",
                "indic_em": "EMP_LFS",
                "s_adj": "SA",
                "sex": "T",
                "age": age,
                "unit": "PC_POP",
                "geo": "DE",
                "TIME_PERIOD": q_str,
                "OBS_VALUE": float(val_base + q_val_offset),
            })
    return pd.DataFrame(rows)


def _synthetic_lfsi_act_q_csv() -> pd.DataFrame:
    """Same shape but indic_em=ACT (activity/LFP rate)."""
    rows = []
    for age, val_base in [
        ("Y15-24", 40.0),
        ("Y25-54", 90.0),
        ("Y55-64", 65.0),
        ("Y15-64", 75.0),
    ]:
        for q_str, q_val_offset in [("2020-Q1", 0.0), ("2020-Q2", -1.0), ("2020-Q3", 1.0)]:
            rows.append({
                "DATAFLOW": "ESTAT:LFSI_EMP_Q(1.0)",
                "freq": "Q",
                "indic_em": "ACT",
                "s_adj": "SA",
                "sex": "T",
                "age": age,
                "unit": "PC_POP",
                "geo": "DE",
                "TIME_PERIOD": q_str,
                "OBS_VALUE": float(val_base + q_val_offset),
            })
    return pd.DataFrame(rows)


def test_lfsi_q_dataflow_constants_match_spec():
    """Sanity check on the constants. lfsi_emp_q is ONE dataflow with two
    indic_em filter values; Y_GE15 is replaced by Y15-64 because the
    dataflow rejects Y_GE15."""
    from puremacro.fetch.labor_eurostat import (
        _LFSI_Q_DATAFLOW, _LFSI_Q_INDICATORS, _EUROSTAT_AGE_MAP_Q,
    )
    assert _LFSI_Q_DATAFLOW == "lfsi_emp_q"
    assert _LFSI_Q_INDICATORS == {"EMP_LFS": "epop", "ACT": "prate"}
    assert _EUROSTAT_AGE_MAP_Q == {
        "Y15-24": "Y15T24",
        "Y25-54": "Y25T54",
        "Y55-64": "Y55T64",
        "Y15-64": "Y15-64",
    }


def test_reshape_lfsi_emp_q_basic_drops_aggregates():
    """Reshape: 4 valid ages survive; Y_GE15 (not in age constraint) is dropped;
    geo DE→DEU; T→_T; values divided by 100."""
    from puremacro.fetch.labor_eurostat import _reshape_eurostat_lfsi_q

    raw = _synthetic_lfsi_emp_q_csv()
    out = _reshape_eurostat_lfsi_q(raw, rate_col="epop")

    # 4 ages × 3 quarters = 12 rows; Y_GE15 dropped.
    assert len(out) == 12
    assert set(out["age"]) == {"Y15T24", "Y25T54", "Y55T64", "Y15-64"}
    assert (out["code"] == "DEU").all()
    assert (out["sex"] == "_T").all()
    # Y15-64 / _T / 2020-Q1 → 70.0/100 = 0.70 (canonical fraction).
    y_q1 = out[(out["age"] == "Y15-64") & (out["date"] == pd.Timestamp("2020-01-01"))]
    assert len(y_q1) == 1
    assert y_q1["epop"].iloc[0] == pytest.approx(0.70)


def test_quarterly_to_monthly_forward_fill_three_months_per_quarter():
    """A single quarterly row expands to 3 monthly rows with the same value."""
    from puremacro.fetch.labor_eurostat import _expand_quarterly_to_monthly

    q_panel = pd.DataFrame([
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"),
         "sex": "_T", "age": "Y_GE15", "epop": 0.70, "prate": 0.75},
        {"code": "DEU", "date": pd.Timestamp("2020-04-01"),
         "sex": "_T", "age": "Y_GE15", "epop": 0.69, "prate": 0.74},
    ])
    out = _expand_quarterly_to_monthly(q_panel)

    # 2 quarters × 3 months = 6 monthly rows.
    assert len(out) == 6
    # Q1 quarter (Jan, Feb, Mar) should all carry epop=0.70.
    q1 = out[out["date"].between(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31"))]
    assert len(q1) == 3
    assert (q1["epop"] == 0.70).all()
    assert (q1["prate"] == 0.75).all()
    # Q2 quarter (Apr, May, Jun) should all carry epop=0.69.
    q2 = out[out["date"].between(pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30"))]
    assert len(q2) == 3
    assert (q2["epop"] == 0.69).all()


def test_fetch_uses_cache_when_present(tmp_path, monkeypatch):
    """If cache_path exists with valid contents, the live SDMX call is skipped."""
    from puremacro.fetch import labor_eurostat
    from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_quarterly_panel

    cache_file = tmp_path / "labor_eurostat_q2m.parquet"
    canonical = ["code", "date", "sex", "age",
                 "wa", "lf", "emp", "une", "ina",
                 "urate", "epop", "prate"]
    cached = pd.DataFrame([
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"), "sex": "_T",
         "age": "Y15-64", "wa": np.nan, "lf": np.nan, "emp": np.nan,
         "une": np.nan, "ina": np.nan, "urate": np.nan, "epop": 0.70,
         "prate": 0.75},
    ])[canonical]
    cached.to_parquet(cache_file)

    def _fail(*args, **kwargs):
        raise RuntimeError("sdmx_get should NOT have been called when cache is present")
    monkeypatch.setattr(labor_eurostat, "sdmx_get", _fail)

    out = fetch_eurostat_lfs_quarterly_panel(cache_path=str(cache_file))
    assert len(out) == 1
    assert (out.columns == canonical).all()
    assert out["epop"].iloc[0] == pytest.approx(0.70)


def test_fetch_end_to_end_offline(tmp_path, monkeypatch):
    """Stub sdmx_get to return synthetic responses; verify full fetcher pipeline."""
    from puremacro.fetch import labor_eurostat
    from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_quarterly_panel

    def _stub_sdmx_get(provider, dataflow, key, csv_path=None):
        # The fetcher now also pulls une_rt_q; this legacy test focuses
        # on lfsi_emp_q so we return empty for the une_rt_q pull (the
        # fetcher tolerates an empty third pull and proceeds).
        if dataflow == "une_rt_q":
            return pd.DataFrame(columns=[
                "DATAFLOW", "freq", "s_adj", "age", "unit", "sex", "geo",
                "TIME_PERIOD", "OBS_VALUE",
            ])
        assert dataflow == "lfsi_emp_q", f"unexpected dataflow {dataflow!r}"
        # Key shape: f"Q.{indic_em}.{adjustment}.{sex_key}.{ages_q_key}.PC_POP.{geo_key}"
        parts = key.split(".")
        indic_em = parts[1]
        if indic_em == "EMP_LFS":
            return _synthetic_lfsi_emp_q_csv()
        if indic_em == "ACT":
            return _synthetic_lfsi_act_q_csv()
        raise ValueError(f"unexpected indic_em {indic_em!r}")

    monkeypatch.setattr(labor_eurostat, "sdmx_get", _stub_sdmx_get)

    cache_file = tmp_path / "labor_eurostat_q2m.parquet"
    out = fetch_eurostat_lfs_quarterly_panel(cache_path=str(cache_file))

    canonical = ["code", "date", "sex", "age",
                 "wa", "lf", "emp", "une", "ina",
                 "urate", "epop", "prate"]
    assert (out.columns == canonical).all()

    # 1 country × 1 sex × 4 ages × 3 quarters × 3 months/q = 36 monthly rows.
    assert len(out) == 36

    assert out["epop"].notna().all()
    assert out["prate"].notna().all()
    for col in ["wa", "lf", "emp", "une", "ina", "urate"]:
        assert out[col].isna().all(), f"{col} should be NaN"

    assert cache_file.exists()

    feb = out[(out["age"] == "Y15-64") & (out["sex"] == "_T")
              & (out["date"] == pd.Timestamp("2020-02-01"))]
    assert len(feb) == 1
    assert feb["epop"].iloc[0] == pytest.approx(0.70)
    assert feb["prate"].iloc[0] == pytest.approx(0.75)


def test_reshape_handles_multi_sex_and_multi_geo():
    """Verify M/F sex codes and multi-country geo translation (DE→DEU, EL→GRC, UK→GBR).
    Also check that aggregate geos (EU27_2020) are dropped."""
    from puremacro.fetch.labor_eurostat import _reshape_eurostat_lfsi_q

    rows = []
    for sex in ("T", "M", "F"):
        for geo in ("DE", "EL", "UK", "EU27_2020"):  # last is aggregate, must drop
            rows.append({
                "DATAFLOW": "ESTAT:LFSI_EMP_Q(1.0)",
                "freq": "Q",
                "indic_em": "EMP_LFS",
                "s_adj": "SA",
                "age": "Y15-64",
                "unit": "PC_POP",
                "sex": sex,
                "geo": geo,
                "TIME_PERIOD": "2020-Q1",
                "OBS_VALUE": 60.0 + ord(sex) * 0.01,  # distinguishable per sex
            })
    raw = pd.DataFrame(rows)

    out = _reshape_eurostat_lfsi_q(raw, rate_col="epop")

    # Aggregate geo dropped → 3 sexes × 3 valid geos = 9 rows.
    assert len(out) == 9
    assert set(out["sex"]) == {"_T", "M", "F"}
    assert set(out["code"]) == {"DEU", "GRC", "GBR"}
    # T-row sanity: epop = 60.0/100 + small offset = 0.60 + tiny.
    deu_t = out[(out["code"] == "DEU") & (out["sex"] == "_T")]
    assert len(deu_t) == 1
    assert deu_t["epop"].iloc[0] == pytest.approx(0.60 + ord("T") * 0.01 / 100, abs=1e-6)


def test_fetch_outer_merge_keeps_partial_rows(tmp_path, monkeypatch):
    """If lfsi_emp_q has a (code,date,sex,age) cell that lfsi_act_q lacks,
    the merged row should keep epop populated and prate=NaN (not drop)."""
    from puremacro.fetch import labor_eurostat
    from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_quarterly_panel

    def _emp_only(provider, dataflow, key, **kwargs):
        # The fetcher now also pulls une_rt_q; this legacy test focuses
        # on the partial-row outer-merge between lfsi_emp_q's two
        # indic_em values, so we return empty for the une_rt_q pull.
        if dataflow == "une_rt_q":
            return pd.DataFrame(columns=[
                "DATAFLOW", "freq", "s_adj", "age", "unit", "sex", "geo",
                "TIME_PERIOD", "OBS_VALUE",
            ])
        assert dataflow == "lfsi_emp_q", f"unexpected dataflow {dataflow!r}"
        indic_em = key.split(".")[1]
        if indic_em == "EMP_LFS":
            # 1 row: DE / T / Y15-64 / 2020-Q1.
            return pd.DataFrame([{
                "DATAFLOW": "ESTAT:LFSI_EMP_Q(1.0)", "freq": "Q",
                "indic_em": "EMP_LFS", "s_adj": "SA",
                "age": "Y15-64", "unit": "PC_POP", "sex": "T", "geo": "DE",
                "TIME_PERIOD": "2020-Q1", "OBS_VALUE": 70.0,
            }])
        if indic_em == "ACT":
            # Empty — simulates Eurostat returning no act rows.
            return pd.DataFrame(columns=[
                "DATAFLOW", "freq", "indic_em", "s_adj", "age", "unit",
                "sex", "geo", "TIME_PERIOD", "OBS_VALUE",
            ])
        raise ValueError(f"unexpected indic_em {indic_em!r}")

    monkeypatch.setattr(labor_eurostat, "sdmx_get", _emp_only)

    out = fetch_eurostat_lfs_quarterly_panel(cache_path=str(tmp_path / "x.parquet"))

    # 1 quarterly row → 3 monthly rows after forward-fill.
    deu_t = out[(out["code"] == "DEU") & (out["sex"] == "_T") & (out["age"] == "Y15-64")]
    assert len(deu_t) == 3
    # epop populated, prate NaN.
    assert deu_t["epop"].notna().all()
    assert deu_t["prate"].isna().all()


def test_cache_schema_drift_raises(tmp_path, monkeypatch):
    """Stale cache without canonical columns must raise ValueError."""
    from puremacro.fetch import labor_eurostat
    from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_quarterly_panel

    cache_file = tmp_path / "stale.parquet"
    # Write a parquet with WRONG schema (missing 'epop' / 'prate').
    pd.DataFrame([
        {"code": "DEU", "date": pd.Timestamp("2020-01-01"), "sex": "_T",
         "age": "Y_GE15", "wa": np.nan},
    ]).to_parquet(cache_file)

    # Sdmx stub raises if called — but we expect schema-mismatch to raise FIRST.
    def _fail(*args, **kwargs):
        raise RuntimeError("should not reach the live call")
    monkeypatch.setattr(labor_eurostat, "sdmx_get", _fail)

    with pytest.raises(ValueError, match="cache schema mismatch"):
        fetch_eurostat_lfs_quarterly_panel(cache_path=str(cache_file))


def _synthetic_une_rt_q_csv() -> pd.DataFrame:
    """Synthetic SDMX-CSV-shaped frame for une_rt_q.

    3 quarters × 1 country (DE) × 1 sex (T) × 1 valid age (Y25-54) plus
    1 rejected age (Y_GE15) we expect dropped.
    """
    rows = []
    for age, val_base in [("Y25-54", 4.5), ("Y_GE15", 999.0)]:
        for q_str, q_val_offset in [("2020-Q1", 0.0), ("2020-Q2", 0.5), ("2020-Q3", -0.2)]:
            rows.append({
                "DATAFLOW": "ESTAT:UNE_RT_Q(1.0)",
                "freq": "Q",
                "s_adj": "SA",
                "age": age,
                "unit": "PC_ACT",
                "sex": "T",
                "geo": "DE",
                "TIME_PERIOD": q_str,
                "OBS_VALUE": float(val_base + q_val_offset),
            })
    return pd.DataFrame(rows)


def test_reshape_une_rt_q_drops_rejected_ages():
    """Reshape: only Y25-54 survives; Y_GE15 dropped; geo DE→DEU; T→_T;
    values divided by 100."""
    from puremacro.fetch.labor_eurostat import _reshape_eurostat_une_rt_q

    raw = _synthetic_une_rt_q_csv()
    out = _reshape_eurostat_une_rt_q(raw)

    # 1 valid age × 3 quarters = 3 rows.
    assert len(out) == 3
    assert set(out["age"]) == {"Y25T54"}
    assert (out["code"] == "DEU").all()
    assert (out["sex"] == "_T").all()
    # Y25-54 / _T / 2020-Q1 → 4.5/100 = 0.045.
    q1 = out[out["date"] == pd.Timestamp("2020-01-01")]
    assert len(q1) == 1
    assert q1["urate"].iloc[0] == pytest.approx(0.045)


def test_fetch_three_dataflow_pulls_offline(tmp_path, monkeypatch):
    """Stub three SDMX dataflows: lfsi_emp_q (×2 indic_em) + une_rt_q.
    Verify the merged panel has all three rate columns populated for
    Y25-54 (where une_rt_q overlaps lfsi_emp_q)."""
    from puremacro.fetch import labor_eurostat
    from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_quarterly_panel

    def _stub_sdmx_get(provider, dataflow, key, csv_path=None):
        if dataflow == "lfsi_emp_q":
            indic_em = key.split(".")[1]
            if indic_em == "EMP_LFS":
                return _synthetic_lfsi_emp_q_csv()
            if indic_em == "ACT":
                return _synthetic_lfsi_act_q_csv()
            raise ValueError(f"unexpected indic_em {indic_em!r}")
        if dataflow == "une_rt_q":
            return _synthetic_une_rt_q_csv()
        raise ValueError(f"unexpected dataflow {dataflow!r}")

    monkeypatch.setattr(labor_eurostat, "sdmx_get", _stub_sdmx_get)

    cache_file = tmp_path / "eurostat_q3_panel.parquet"
    out = fetch_eurostat_lfs_quarterly_panel(cache_path=str(cache_file))

    canonical = ["code", "date", "sex", "age",
                 "wa", "lf", "emp", "une", "ina",
                 "urate", "epop", "prate"]
    assert (out.columns == canonical).all()

    # Y25T54 / _T / 2020-02-01 should have all three rate columns populated.
    feb_y25 = out[(out["age"] == "Y25T54") & (out["sex"] == "_T")
                   & (out["date"] == pd.Timestamp("2020-02-01"))]
    assert len(feb_y25) == 1
    assert feb_y25["epop"].notna().all()
    assert feb_y25["prate"].notna().all()
    assert feb_y25["urate"].notna().all()
    # urate = 4.5/100 = 0.045 from the synthetic.
    assert feb_y25["urate"].iloc[0] == pytest.approx(0.045)

    # Y15-64 / _T / 2020-02-01 should have epop+prate but urate NaN
    # (une_rt_q doesn't ship Y15-64).
    feb_y1564 = out[(out["age"] == "Y15-64") & (out["sex"] == "_T")
                     & (out["date"] == pd.Timestamp("2020-02-01"))]
    assert len(feb_y1564) == 1
    assert feb_y1564["epop"].notna().all()
    assert feb_y1564["prate"].notna().all()
    assert feb_y1564["urate"].isna().all()
