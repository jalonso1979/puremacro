import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from puremacro.build_subnational_panel import build_county_quarterly_panel

@pytest.fixture
def mock_qcew():
    return pd.DataFrame({
        "county_fips": ["01001", "01001", "01001", "01001", "01001", "01001"],
        "naics": ["10", "10", "11", "11", "21", "21"],
        "year": [2010, 2010, 2010, 2010, 2010, 2010],
        "qtr": [1, 2, 1, 2, 1, 2],
        "emp_avg": [1000, 1100, 200, 250, 300, 350],
        "aww": [500, 510, 400, 420, 600, 610],
        "total_wages": [500000, 561000, 80000, 105000, 180000, 213500],
        "estabs": [50, 52, 10, 12, 5, 5],
    })

@pytest.fixture
def mock_laus():
    return pd.DataFrame({
        "county_fips": ["01001", "01001", "01001"],
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "lf": [5000, 5100, 5200],
        "emp": [4500, 4600, 4700],
        "une": [500, 500, 500],
        "urate": [10.0, 9.8, 9.6]
    })

@pytest.fixture
def mock_epu():
    return pd.DataFrame({
        "state_fips": ["01", "01", "01"],
        "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
        "epu_state_m": [100.0, 110.0, 120.0]
    })

@pytest.fixture
def mock_shares():
    return pd.DataFrame({
        "county_fips": ["01001", "01001"],
        "naics": ["11", "21"],
        "share": [0.4, 0.6]
    })

@pytest.fixture
def mock_sens():
    return pd.DataFrame({
        "naics": ["11", "21"],
        "shrunk_beta": [1.5, 0.5]
    })

def test_build_county_quarterly_panel_basic(mock_qcew, mock_laus, mock_epu, mock_shares, mock_sens):
    totals, ind_panel = build_county_quarterly_panel(
        qcew_df=mock_qcew,
        laus_df=mock_laus,
        epu_df=mock_epu,
        shares_df=mock_shares,
        sens_df=mock_sens
    )

    assert isinstance(totals, pd.DataFrame)
    assert isinstance(ind_panel, pd.DataFrame)

    # Check that ind_panel excluded NAICS '10'
    assert "10" not in ind_panel["naics"].values

    # Check that totals contains only NAICS '10' data
    assert len(totals) == 2  # 2 quarters in mock_qcew for NAICS 10

    # Check if correct columns exist
    expected_cols = [
        "county_fips", "state_fips", "date",
        "emp_qcew_total", "aww_qcew_total", "wages_qcew_total", "estab_qcew_total",
        "epu_state_q", "lf_laus_q", "emp_laus_q", "une_laus_q", "urate_laus_q",
        "epu_county_bartik_q", "exposure_c", "tradable_share_c", "manuf_share_c",
        "right_to_work_state", "log_emp_qcew_total", "log_aww_qcew_total",
        "log_wages_qcew_total", "regime"
    ]
    for col in expected_cols:
        assert col in totals.columns, f"Missing column: {col}"

def test_build_county_quarterly_panel_with_births(mock_qcew, mock_laus, mock_epu, mock_shares, mock_sens):
    births_df = pd.DataFrame({
        "county_fips": ["01001", "01001"],
        "year": [2010, 2011],
        "births": [150, 160]
    })

    state_births_q = pd.DataFrame({
        "state_fips": ["01", "01"],
        "date": pd.to_datetime(["2010-01-01", "2010-04-01"]),
        "births_state_q": [1000, 1100]
    })

    state_births_a = pd.DataFrame({
        "state_fips": ["01", "01"],
        "year": [2010, 2011],
        "births_state_a": [4000, 4200]
    })

    totals, _ = build_county_quarterly_panel(
        qcew_df=mock_qcew,
        laus_df=mock_laus,
        epu_df=mock_epu,
        shares_df=mock_shares,
        sens_df=mock_sens,
        births_df=births_df,
        state_births_q=state_births_q,
        state_births_a=state_births_a
    )

    assert "births_county_a" in totals.columns
    assert "births_state_q" in totals.columns
    assert "births_state_a" in totals.columns

    assert "log_births_county_a" in totals.columns
    assert "log_births_state_q" in totals.columns
    assert "log_births_state_a" in totals.columns

def test_build_county_quarterly_panel_missing_required_args():
    with pytest.raises(ValueError, match="qcew_path is required"):
        build_county_quarterly_panel()

    with pytest.raises(ValueError, match="laus_path is required"):
        build_county_quarterly_panel(qcew_df=pd.DataFrame())

def test_build_county_quarterly_panel_with_paths(tmp_path, monkeypatch):
    """Test providing paths instead of dataframes."""
    qcew_path = tmp_path / "qcew.csv"
    qcew_path.write_text("mock qcew")

    laus_path = tmp_path / "laus.csv"
    laus_path.write_text("mock laus")

    epu_path = tmp_path / "epu.csv"
    epu_path.write_text("mock epu")

    shares_path = tmp_path / "shares.csv"
    shares_path.write_text("mock shares")

    sens_path = tmp_path / "sens.csv"
    sens_path.write_text("mock sens")

    # Mock the parsing functions
    import puremacro.fetch.qcew
    import puremacro.fetch.laus

    def mock_parse_qcew_csv(path):
        assert str(path) == str(qcew_path)
        return pd.DataFrame({
            "county_fips": ["01001", "01001"],
            "naics": ["10", "11"],
            "year": [2010, 2010],
            "qtr": [1, 1],
            "emp_avg": [1000, 200],
            "aww": [500, 400],
            "total_wages": [500000, 80000],
            "estabs": [50, 10],
        })

    def mock_parse_laus_series(path, geography):
        assert str(path) == str(laus_path)
        assert geography == "county"
        return pd.DataFrame({
            "county_fips": ["01001"],
            "date": pd.to_datetime(["2010-01-01"]),
            "lf": [5000],
            "emp": [4500],
            "une": [500],
            "urate": [10.0]
        })

    def mock_read_csv(path, **kwargs):
        if str(path) == str(epu_path):
            return pd.DataFrame({
                "state_fips": ["01"],
                "date": pd.to_datetime(["2010-01-01"]),
                "epu_state_m": [100.0]
            })
        elif str(path) == str(shares_path):
            return pd.DataFrame({
                "county_fips": ["01001"],
                "naics": ["11"],
                "share": [0.4]
            })
        elif str(path) == str(sens_path):
            return pd.DataFrame({
                "naics": ["11"],
                "shrunk_beta": [1.5]
            })
        return pd.read_csv_orig(path, **kwargs)

    monkeypatch.setattr(puremacro.fetch.qcew, "parse_qcew_csv", mock_parse_qcew_csv)
    monkeypatch.setattr(puremacro.fetch.laus, "parse_laus_series", mock_parse_laus_series)

    pd.read_csv_orig = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    totals, ind_panel = build_county_quarterly_panel(
        qcew_path=qcew_path,
        laus_path=laus_path,
        epu_path=epu_path,
        shares_path=shares_path,
        sens_path=sens_path
    )

    assert len(totals) == 1
    assert "emp_qcew_total" in totals.columns
