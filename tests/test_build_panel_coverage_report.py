import pandas as pd
import pytest
from puremacro.build_panel import build_coverage_report

def test_build_coverage_report_empty():
    df = pd.DataFrame(columns=["code", "variable", "date", "value", "sa_source"])
    result = build_coverage_report(df)

    assert result.empty
    expected_cols = ["code", "variable", "first_obs", "last_obs", "n_obs", "pct_missing", "sa_source"]
    assert list(result.columns) == expected_cols

def test_build_coverage_report_happy_path():
    data = [
        # code, variable, date, value, sa_source
        # Monthly variable, missing Feb 2020 (1/3 missing)
        {"code": "US", "variable": "var_m", "date": pd.Timestamp("2020-01-01"), "value": 1.0, "sa_source": "source1"},
        {"code": "US", "variable": "var_m", "date": pd.Timestamp("2020-03-01"), "value": 3.0, "sa_source": "source1"},

        # Quarterly variable, missing 2020 Q3 (1/4 missing)
        {"code": "US", "variable": "var_q", "date": pd.Timestamp("2020-01-01"), "value": 1.0, "sa_source": "source2"},
        {"code": "US", "variable": "var_q", "date": pd.Timestamp("2020-04-01"), "value": 2.0, "sa_source": "source2"},
        {"code": "US", "variable": "var_q", "date": pd.Timestamp("2020-10-01"), "value": 4.0, "sa_source": "source2"},

        # Single observation check (expected=1, n_obs=1 -> pct_missing=0)
        {"code": "GB", "variable": "var_m", "date": pd.Timestamp("2020-01-01"), "value": 1.0, "sa_source": "source3"},

        # Duplicate dates to test clipping (expected=1, n_obs=2 -> pct_missing clipped to 0)
        {"code": "EU", "variable": "var_q", "date": pd.Timestamp("2020-01-01"), "value": 1.0, "sa_source": "source4"},
        {"code": "EU", "variable": "var_q", "date": pd.Timestamp("2020-01-01"), "value": 2.0, "sa_source": "source4"},
    ]
    df = pd.DataFrame(data)
    result = build_coverage_report(df)

    assert not result.empty
    expected_cols = ["code", "variable", "first_obs", "last_obs", "n_obs", "pct_missing", "sa_source"]
    assert list(result.columns) == expected_cols

    # Check US var_m
    us_var_m = result[(result["code"] == "US") & (result["variable"] == "var_m")].iloc[0]
    assert us_var_m["first_obs"] == pd.Timestamp("2020-01-01")
    assert us_var_m["last_obs"] == pd.Timestamp("2020-03-01")
    assert us_var_m["n_obs"] == 2
    assert us_var_m["sa_source"] == "source1"
    assert pytest.approx(us_var_m["pct_missing"]) == 1/3

    # Check US var_q
    us_var_q = result[(result["code"] == "US") & (result["variable"] == "var_q")].iloc[0]
    assert us_var_q["first_obs"] == pd.Timestamp("2020-01-01")
    assert us_var_q["last_obs"] == pd.Timestamp("2020-10-01")
    assert us_var_q["n_obs"] == 3
    assert us_var_q["sa_source"] == "source2"
    assert pytest.approx(us_var_q["pct_missing"]) == 1/4

    # Check GB var_m (single obs)
    gb_var_m = result[(result["code"] == "GB") & (result["variable"] == "var_m")].iloc[0]
    assert gb_var_m["n_obs"] == 1
    assert pytest.approx(gb_var_m["pct_missing"]) == 0.0

    # Check EU var_q (duplicate dates -> clip to 0)
    eu_var_q = result[(result["code"] == "EU") & (result["variable"] == "var_q")].iloc[0]
    assert eu_var_q["n_obs"] == 2
    assert pytest.approx(eu_var_q["pct_missing"]) == 0.0
