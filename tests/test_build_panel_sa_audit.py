import pandas as pd
import numpy as np
import pytest

from puremacro.build_panel import sa_audit

def test_sa_audit_skips_exempt_vars():
    dates = pd.date_range("2000-01-01", periods=40, freq="ME")
    df = pd.DataFrame({
        "code": ["USA"] * 40,
        "variable": ["vix"] * 40,
        "date": dates,
        "value": np.random.randn(40),
        "sa_source": ["none"] * 40
    })
    res = sa_audit(df)
    assert res.empty

def test_sa_audit_skips_non_none_sa_source():
    dates = pd.date_range("2000-01-01", periods=40, freq="ME")
    df = pd.DataFrame({
        "code": ["USA"] * 40,
        "variable": ["some_var"] * 40,
        "date": dates,
        "value": np.random.randn(40),
        "sa_source": ["x13"] * 40
    })
    res = sa_audit(df)
    assert res.empty

def test_sa_audit_skips_short_series():
    dates = pd.date_range("2000-01-01", periods=35, freq="ME")
    df = pd.DataFrame({
        "code": ["USA"] * 35,
        "variable": ["some_var"] * 35,
        "date": dates,
        "value": np.random.randn(35),
        "sa_source": ["none"] * 35
    })
    res = sa_audit(df)
    assert res.empty

def test_sa_audit_monthly_groups():
    # Make a strong seasonal pattern so it fails
    dates = pd.date_range("2000-01-01", periods=60, freq="ME")
    values = []
    for d in dates:
        # High value in January, low elsewhere
        values.append(10.0 if d.month == 1 else 0.0)

    df = pd.DataFrame({
        "code": ["USA"] * 60,
        "variable": ["test_m"] * 60,
        "date": dates,
        "value": values,
        "sa_source": ["none"] * 60
    })
    res = sa_audit(df)
    assert not res.empty
    assert res.iloc[0]["code"] == "USA"
    assert res.iloc[0]["variable"] == "test_m"
    assert res.iloc[0]["sa_flag"] == "fail" # because p < 0.05
    assert res.iloc[0]["kw_p"] < 0.05

def test_sa_audit_quarterly_groups():
    # Make a strong seasonal pattern
    dates = pd.date_range("2000-01-01", periods=60, freq="QE")
    values = []
    for d in dates:
        # High value in Q1
        values.append(10.0 if d.quarter == 1 else 0.0)

    df = pd.DataFrame({
        "code": ["USA"] * 60,
        "variable": ["test_q"] * 60,
        "date": dates,
        "value": values,
        "sa_source": ["none"] * 60
    })
    res = sa_audit(df)
    assert not res.empty
    assert res.iloc[0]["sa_flag"] == "fail"

def test_sa_audit_pass():
    # Random noise should pass
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=100, freq="ME")
    df = pd.DataFrame({
        "code": ["USA"] * 100,
        "variable": ["test_pass"] * 100,
        "date": dates,
        "value": np.random.randn(100),
        "sa_source": ["none"] * 100
    })
    res = sa_audit(df)
    assert not res.empty
    assert res.iloc[0]["sa_flag"] == "pass"

def test_sa_audit_exception_handling():
    # Provide data that causes an error in groupby or kruskal
    # For instance, a series where values can't be converted to float
    dates = pd.date_range("2000-01-01", periods=40, freq="ME")
    df = pd.DataFrame({
        "code": ["USA"] * 40,
        "variable": ["test_ex"] * 40,
        "date": dates,
        "value": ["not_a_float"] * 40,
        "sa_source": ["none"] * 40
    })
    # Should silently continue and return empty dataframe
    res = sa_audit(df)
    assert res.empty

def test_sa_audit_few_groups():
    # Less than 2 valid groups (each with length < 2)
    # All same month but spaced out by years so length > 36 but all in same group
    dates = pd.to_datetime([f"{y}-01-01" for y in range(1980, 2020)])

    df = pd.DataFrame({
        "code": ["USA"] * 40,
        "variable": ["test_group"] * 40,
        "date": dates,
        "value": np.random.randn(40),
        "sa_source": ["none"] * 40
    })
    res = sa_audit(df)
    assert res.empty
