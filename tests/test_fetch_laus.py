"""LAUS measure semantics checked independently of the parser's mapping.

BLS defines 03=unemployment rate, 04=unemployment, 05=employment and
06=labor force: https://download.bls.gov/pub/time.series/la/la.measure.
The synthetic observations below obey LF = E + U and rate = 100 U / LF.
They are test data, not observations downloaded from BLS.
"""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import laus
from puremacro.build_subnational_panel import (
    build_county_quarterly_panel,
    build_state_monthly_panel,
)


def _flat_file(geography):
    prefix = "LASST0100000000000" if geography == "state" else "LAUCN0100100000000"
    rows = ["series_id\tyear\tperiod\tvalue\tfootnote_codes"]
    # Explicit BLS-coded values, deliberately not derived from _MEASURE_MAP.
    for month, rate, unemployed, employed in [(1, 5, 100, 1900),
                                               (2, 6, 120, 1880),
                                               (3, 4, 80, 1920)]:
        for code, value in [("03", rate), ("04", unemployed),
                            ("05", employed), ("06", 2000)]:
            rows.append(f"{prefix}{code:<12}\t2024\tM{month:02}\t{value}\t")
    # The annual-average period and other measures must not add monthly rows.
    rows.extend([f"{prefix}06\t2024\tM13\t2000\t",
                 f"{prefix}09\t2024\tM01\t10000\t"])
    return ("\n".join(rows) + "\n").encode()


def _assert_measures(result, geography):
    area = "state_fips" if geography == "state" else "county_fips"
    assert result[area].tolist() == (["01"] if geography == "state" else ["01001"]) * 3
    assert result["date"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert result["lf"].tolist() == [2000, 2000, 2000]
    assert result["emp"].tolist() == [1900, 1880, 1920]
    assert result["une"].tolist() == [100, 120, 80]
    np.testing.assert_allclose(result["lf"], result["emp"] + result["une"])
    np.testing.assert_allclose(result["urate"], 100 * result["une"] / result["lf"])
    assert result["source"].tolist() == ["bls_laus"] * 3


@pytest.mark.parametrize("geography", ["state", "county"])
@pytest.mark.parametrize("input_kind", ["bytes", "path"])
def test_parse_laus_preserves_bls_measures(geography, input_kind, tmp_path):
    content = _flat_file(geography)
    if input_kind == "path":
        source = tmp_path / "laus.txt"
        source.write_bytes(content)
    else:
        source = BytesIO(content)
    _assert_measures(laus.parse_laus_series(source, geography=geography), geography)


@pytest.mark.parametrize("geography,filename", [
    ("state", "la.data.1.CurrentS"), ("county", "la.data.64.County"),
])
def test_fetch_laus_preserves_bls_measures(geography, filename, monkeypatch):
    calls = []

    def download(url, *, refresh):
        calls.append((url, refresh))
        return _flat_file(geography)

    monkeypatch.setattr(laus, "cached_get", download)
    fetch = laus.fetch_state if geography == "state" else laus.fetch_county
    result = fetch(refresh=True)
    assert calls == [(f"https://download.bls.gov/pub/time.series/la/{filename}", True)]
    _assert_measures(result, geography)


def test_state_panel_uses_labor_force_and_unemployment_from_raw_file(tmp_path):
    path = tmp_path / "state.txt"
    path.write_bytes(_flat_file("state"))
    dates = pd.date_range("2024-01-01", periods=3, freq="MS")
    result = build_state_monthly_panel(
        laus_path=path,
        ces_df=pd.DataFrame({"state_fips": ["01"] * 3, "date": dates,
                             "supersector": ["00"] * 3, "emp": [1800] * 3}),
        epu_df=pd.DataFrame({"state_fips": ["01"] * 3, "date": dates,
                             "epu_state_m": [100] * 3}),
    )
    assert result["lf_laus"].tolist() == [2000, 2000, 2000]
    assert result["une_laus"].tolist() == [100, 120, 80]
    np.testing.assert_allclose(result["log_lf_laus"], np.log(2000))
    np.testing.assert_allclose(result["lf_laus"], result["emp_laus"] + result["une_laus"])


def test_county_panel_preserves_labor_accounting_after_quarterly_aggregation(tmp_path):
    path = tmp_path / "county.txt"
    path.write_bytes(_flat_file("county"))
    dates = pd.date_range("2024-01-01", periods=3, freq="MS")
    result, _ = build_county_quarterly_panel(
        laus_path=path,
        qcew_df=pd.DataFrame({"county_fips": ["01001"], "naics": ["10"],
                              "year": [2024], "qtr": [1], "emp_avg": [1800],
                              "aww": [1000], "total_wages": [1_800_000], "estabs": [20]}),
        epu_df=pd.DataFrame({"state_fips": ["01"] * 3, "date": dates,
                             "epu_state_m": [100] * 3}),
        shares_df=pd.DataFrame({"county_fips": ["01001"], "naics": ["3111"], "share": [1.0]}),
        sens_df=pd.DataFrame({"naics": ["3111"], "shrunk_beta": [0.5]}),
    )
    assert result["date"].tolist() == [pd.Timestamp("2024-01-01")]
    assert result["lf_laus_q"].tolist() == [2000]
    assert result["emp_laus_q"].tolist() == [1900]
    assert result["une_laus_q"].tolist() == [100]
    assert result["urate_laus_q"].tolist() == [5]
    np.testing.assert_allclose(result["lf_laus_q"], result["emp_laus_q"] + result["une_laus_q"])
