import pandas as pd
import pytest
from puremacro.build_panel import build_variable_coverage

def test_build_variable_coverage_empty():
    """Test empty dataframe returns expected columns."""
    df = pd.DataFrame()
    res = build_variable_coverage(df)
    assert res.empty
    assert list(res.columns) == ["variable", "n_countries", "first_obs", "last_obs"]

def test_build_variable_coverage_populated():
    """Test populated dataframe returns correct aggregations."""
    df = pd.DataFrame({
        "variable": ["var1", "var1", "var1", "var2", "var2", "var3"],
        "code": ["USA", "USA", "CAN", "USA", "MEX", "JPN"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-03-01", "2020-04-01", "2019-12-01"])
    })
    res = build_variable_coverage(df)

    assert not res.empty
    assert len(res) == 3
    assert list(res.columns) == ["variable", "n_countries", "first_obs", "last_obs"]

    # Check var1
    var1_res = res[res["variable"] == "var1"].iloc[0]
    assert var1_res["n_countries"] == 2  # USA, CAN
    assert var1_res["first_obs"] == pd.to_datetime("2020-01-01")
    assert var1_res["last_obs"] == pd.to_datetime("2020-02-01")

    # Check var2
    var2_res = res[res["variable"] == "var2"].iloc[0]
    assert var2_res["n_countries"] == 2  # USA, MEX
    assert var2_res["first_obs"] == pd.to_datetime("2020-03-01")
    assert var2_res["last_obs"] == pd.to_datetime("2020-04-01")

    # Check var3
    var3_res = res[res["variable"] == "var3"].iloc[0]
    assert var3_res["n_countries"] == 1  # JPN
    assert var3_res["first_obs"] == pd.to_datetime("2019-12-01")
    assert var3_res["last_obs"] == pd.to_datetime("2019-12-01")

    # Ensure it's sorted by variable
    assert res["variable"].tolist() == ["var1", "var2", "var3"]

def test_build_variable_coverage_missing_columns():
    """Test with dataframe missing required columns, should raise KeyError in pandas groupby/agg."""
    df = pd.DataFrame({
        "wrong_col": [1, 2, 3]
    })
    with pytest.raises(KeyError):
        build_variable_coverage(df)
