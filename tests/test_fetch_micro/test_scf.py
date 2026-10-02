"""SCF provider against synthetic Stata archives (offline)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.micro import scf
from puremacro.fetch.micro._design import replicate_variance, rubin_combine


def _zip_dta(df: pd.DataFrame, name: str) -> bytes:
    buf = io.BytesIO()
    df.to_stata(buf, write_index=False, version=118)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(name, buf.getvalue())
    return out.getvalue()


def _synthetic(n_hh=40, seed=1, year=2022):
    rng = np.random.default_rng(seed)
    yy1 = np.arange(1, n_hh + 1)
    rows = []
    for h in yy1:
        base_w = rng.uniform(1e4, 5e4)
        nw = rng.lognormal(11, 1.5)
        for i in range(1, 6):
            rows.append({"YY1": h, "Y1": h * 10 + i, "WGT": base_w,
                         "NETWORTH": nw * rng.uniform(0.9, 1.1),
                         "INCOME": rng.lognormal(10.5, 0.7), "AGE": 30 + h % 40})
    summ = pd.DataFrame(rows)
    rw = {"YY1": yy1}
    for k in range(1, 1000):
        rw[f"WT1B{k}"] = rng.uniform(0, 2, n_hh) * 2e4
        rw[f"MM{k}"] = rng.integers(0, 3, n_hh).astype(float)
    rw = pd.DataFrame(rw)
    rw.loc[0, "WT1B7"] = np.nan          # missing replicate weight -> 0
    urls = scf.scf_urls(year)
    files = {urls["summary"]: _zip_dta(summ, f"rscfp{year}.dta"),
             urls["replicates"]: _zip_dta(rw, f"p{year % 100:02d}_rw1.dta")}
    return summ, rw, files


@pytest.fixture
def served(monkeypatch):
    summ, rw, files = _synthetic()
    calls = []

    def fake(url, **k):
        calls.append(url)
        return files[url]

    monkeypatch.setattr(scf._http, "cached_get", fake)
    return summ, rw, calls


def test_urls_follow_the_board_layout():
    assert scf.scf_urls(2022) == {
        "summary": "https://www.federalreserve.gov/econres/files/scfp2022s.zip",
        "replicates": "https://www.federalreserve.gov/econres/files/scf2022rw1s.zip"}
    old = scf.scf_urls(1995)
    assert old["summary"].endswith("/econresdata/scf/files/scfp1995s.zip")
    assert old["replicates"].endswith("/econresdata/scf/files/scf95rw1s.zip")
    with pytest.raises(ValueError, match="no SCF wave"):
        scf.scf_urls(2021)
    assert scf.available_years()[-1] == 2022


def test_fetch_builds_implicates_and_replicate_weights(served):
    summ, rw, calls = served
    mf = scf.fetch_scf(2022, ["NetWorth", "income"])
    assert len(calls) == 2
    assert len(mf.data) == len(summ)
    assert sorted(mf.data["implicate"].unique()) == [1, 2, 3, 4, 5]
    assert {"networth", "income", "wgt", "yy1"} <= set(mf.data.columns)
    assert "age" not in mf.data.columns
    r = mf.data[mf.data["yy1"] == 1].iloc[0]
    assert r["wt1b7"] == 0.0                              # NaN -> 0
    assert r["wt1b3"] == pytest.approx(rw.loc[0, "WT1B3"] * rw.loc[0, "MM3"])
    assert mf.design.scale == pytest.approx(1 / 998)


def test_mean_networth_equals_manual_rubin_combination(served):
    mf = scf.fetch_scf(2022, ["networth"])
    reps = [f"wt1b{k}" for k in range(1, 1000)]
    ests, vars_ = [], []
    for i in range(1, 6):
        d = mf.data[mf.data["implicate"] == i]
        th = np.average(d["networth"], weights=d["wgt"])
        rt = (d[reps].to_numpy().T @ d["networth"].to_numpy()) / d[reps].to_numpy().sum(0)
        ests.append(th)
        vars_.append(replicate_variance(th, rt, scale=1 / 998))
    est, var = rubin_combine(ests, vars_)
    res = mf.mean("networth")
    assert res.loc[0, "estimate"] == pytest.approx(est)
    assert res.loc[0, "se"] == pytest.approx(np.sqrt(var))
    assert res.loc[0, "implicates"] == 5


def test_without_replicates_only_one_download(served):
    _, _, calls = served
    mf = scf.fetch_scf(2022, ["networth"], replicate_weights=False)
    assert len(calls) == 1
    assert not any(c.startswith("wt1b") for c in mf.data.columns)
    assert np.isnan(mf.mean("networth").loc[0, "se"])


def test_unknown_variable_is_reported(served):
    with pytest.raises(KeyError, match="not in the SCF summary extract"):
        scf.fetch_scf(2022, ["nonsense"])


def test_layout_checks():
    bad = pd.DataFrame({"y1": [11, 12], "yy1": [1, 1], "wgt": [1.0, 1.0]})
    with pytest.raises(ValueError, match="5 equal-sized implicates"):
        scf.prepare_summary(bad)
    with pytest.raises(ValueError, match="lacks 'wgt'"):
        scf.prepare_summary(bad.drop(columns="wgt"))
    with pytest.raises(ValueError, match="lacks 'yy1'"):
        scf.prepare_replicates(pd.DataFrame({"x": [1]}))


def test_1989_identifier_names_are_mapped():
    df = pd.DataFrame({"X1": [11], "XX1": [1], "WGT": [1.0]})
    blob = _zip_dta(df, "rscfp1989.dta")
    out = scf._read_dta_zip(blob)
    assert {"y1", "yy1", "wgt"} <= set(out.columns)
