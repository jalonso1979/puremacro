"""Hermetic tests for the JOLTS and Eurostat-vacancy fetchers.

No network: the JOLTS tests monkeypatch the module's ``cached_get``
with canned fredgraph CSVs; the Eurostat tests go through
``sdmx_get(csv_path=...)`` with a synthetic SDMX-CSV file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.jolts as jolts_mod
from puremacro.fetch.jolts import INDUSTRY_CODES, fetch_jolts
from puremacro.fetch.vacancies_eurostat import fetch_eurostat_vacancies


def _fredgraph_csv(sid: str, values: list[float]) -> bytes:
    rows = ["observation_date," + sid]
    for i, v in enumerate(values):
        rows.append(f"200{i // 12}-{i % 12 + 1:02d}-01,{v}")
    return ("\n".join(rows) + "\n").encode()


@pytest.fixture
def fake_fredgraph(monkeypatch):
    served: list[str] = []

    def fake(url, *, refresh=False, **kw):
        sid = url.rsplit("id=", 1)[1]
        served.append(sid)
        base = 100.0 if sid.endswith("L") else 3.0
        return _fredgraph_csv(sid, [base + i * 0.5 for i in range(24)])

    monkeypatch.setattr(jolts_mod, "cached_get", fake)
    return served


def test_fetch_jolts_tidy_shape_and_ids(fake_fredgraph):
    df = fetch_jolts(elements=("openings", "quits"),
                     industries=("total", "manufacturing"))
    assert list(df.columns) == ["date", "industry", "element", "level",
                                "rate"]
    assert set(df["element"]) == {"openings", "quits"}
    assert set(df["industry"]) == {"total", "manufacturing"}
    assert len(df) == 2 * 2 * 24
    # Total nonfarm omits the industry code; manufacturing embeds 3000.
    assert "JTSJOL" in fake_fredgraph and "JTSJOR" in fake_fredgraph
    assert "JTS3000QUL" in fake_fredgraph and "JTS3000QUR" in fake_fredgraph
    assert (df["level"] >= 100).all() and (df["rate"] < 20).all()


def test_fetch_jolts_nsa_prefix_and_raw_code(fake_fredgraph):
    fetch_jolts(elements=("hires",), industries=("4000",), sa=False)
    assert "JTU4000HIL" in fake_fredgraph


def test_fetch_jolts_validates_inputs(fake_fredgraph):
    with pytest.raises(ValueError, match="unknown industry"):
        fetch_jolts(industries=("does_not_exist",))
    with pytest.raises(ValueError, match="unknown element"):
        fetch_jolts(elements=("resignations",))
    assert "total" in INDUSTRY_CODES and INDUSTRY_CODES["total"] == ""


def _jvs_fixture(path):
    rows = []
    for geo, vals in (("DE", (2.5, 2.6)), ("EL", (1.0, 1.1)),
                      ("EA20", (2.2, 2.3))):
        for q, v in zip(("2025-Q1", "2025-Q2"), vals):
            rows.append({"DATAFLOW": "ESTAT:jvs_q_nace2(1.0)",
                         "LAST UPDATE": "x", "freq": "Q", "s_adj": "SA",
                         "nace_r2": "B-S", "sizeclas": "TOTAL",
                         "indic_em": "JVR", "geo": geo,
                         "TIME_PERIOD": q, "OBS_VALUE": v,
                         "OBS_FLAG": "", "CONF_STATUS": ""})
    # A row the filters must drop (wrong indicator).
    rows.append({**rows[0], "indic_em": "JOBVAC", "OBS_VALUE": 999.0})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fetch_eurostat_vacancies_from_csv(tmp_path):
    p = _jvs_fixture(tmp_path / "jvs.csv")
    v = fetch_eurostat_vacancies(csv_path=p)
    assert list(v.columns) == ["code", "date", "jvr"]
    # ISO-3 mapping (EL -> GRC) and aggregate (EA20) dropped.
    assert set(v["code"]) == {"DEU", "GRC"}
    assert v["date"].min() == pd.Timestamp("2025-01-01")
    assert np.isclose(v.loc[(v["code"] == "DEU")
                            & (v["date"] == "2025-04-01"),
                            "jvr"].iloc[0], 2.6)
    assert (v["jvr"] < 100).all()          # the JOBVAC row was filtered

    v_agg = fetch_eurostat_vacancies(csv_path=p, include_aggregates=True)
    assert "EA20" in set(v_agg["code"])


def test_fetch_eurostat_vacancies_validates(tmp_path):
    p = _jvs_fixture(tmp_path / "jvs.csv")
    with pytest.raises(ValueError, match="indicator"):
        fetch_eurostat_vacancies(indicator="VACANCY")
    with pytest.raises(ValueError, match="combination not published"):
        fetch_eurostat_vacancies(csv_path=p, s_adj="NSA")
