"""Conformance tests for secondary LP estimators specification standards.

Verifies that all 8 secondary LP estimators:
1. Populate res.attrs["specification"] (and for lp_did, res.estimates is an LPResult
   and res.attrs mirrors res.estimates.attrs).
2. Contain all required specification keys:
   'estimator', 'method', 'y_name', 'x_name', 'controls', 'n_lags',
   'horizons', 'cov_type', 'ci_level'.
3. Support standard summary() formatting including header items (lags, SE, N, controls).
4. Verify lp_iv_lewbel uses lowercase 'first_stage_f'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp._results import LPResult
from puremacro.lp.asymmetric import lp_asymmetric
from puremacro.lp.cce import cce_panel_lp
from puremacro.lp.iv_lewbel import lp_iv_lewbel
from puremacro.lp.la_lp import la_lp, la_lp_iv
from puremacro.lp.lp_did import LPDiDResult, lp_did
from puremacro.lp.mean_group import mean_group_panel_lp
from puremacro.lp.quantile import lp_quantile
from puremacro.lp.state_dep import lp_state_dep, lp_state_dep_iv

REQUIRED_SPEC_KEYS = {
    "estimator",
    "method",
    "y_name",
    "x_name",
    "controls",
    "n_lags",
    "horizons",
    "cov_type",
    "ci_level",
}


@pytest.fixture
def ts_data():
    """Synthetic time-series dataset with datetime index."""
    rng = np.random.default_rng(42)
    n = 100
    idx = pd.date_range("2010-01-01", periods=n, freq="QE")
    x = rng.standard_normal(n)
    z = 0.8 * x + 0.2 * rng.standard_normal(n)
    c1 = rng.standard_normal(n)
    state = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 1.2 * x[t] + 0.3 * c1[t] + rng.standard_normal()
    return pd.DataFrame({"y": y, "x": x, "z": z, "c1": c1, "state": state}, index=idx)


@pytest.fixture
def panel_data():
    """Synthetic multi-entity panel dataset."""
    rng = np.random.default_rng(123)
    n_entities = 5
    n_periods = 40
    rows = []
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="ME")
    for i in range(n_entities):
        code = f"C{i}"
        x = rng.standard_normal(n_periods)
        z = rng.standard_normal(n_periods)
        c1 = rng.standard_normal(n_periods)
        y = np.zeros(n_periods)
        for t in range(1, n_periods):
            y[t] = 0.4 * y[t - 1] + 0.8 * x[t] + 0.2 * c1[t] + rng.standard_normal()
        for t in range(n_periods):
            rows.append({
                "code": code,
                "date": dates[t],
                "y": y[t],
                "x": x[t],
                "z": z[t],
                "c1": c1[t],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def did_panel_data():
    """Synthetic absorbing-treatment panel dataset for LP-DiD."""
    rng = np.random.default_rng(456)
    n_units = 12
    n_periods = 20
    rows = []
    for u in range(n_units):
        unit_id = f"U{u}"
        treat_period = 8 if u < 6 else 999  # first 6 treated at t=8, rest never treated
        y_val = 0.0
        for t in range(n_periods):
            d = 1.0 if t >= treat_period else 0.0
            y_val += 0.5 * d + rng.standard_normal() * 0.1
            rows.append({
                "unit": unit_id,
                "time": t,
                "y": y_val,
                "D": d,
                "c1": rng.standard_normal(),
            })
    return pd.DataFrame(rows)


def _assert_spec_standard(res, expected_estimator: str, expected_y: str, expected_x: str):
    """Helper asserting that attrs['specification'] conforms to required standards."""
    assert "specification" in res.attrs, f"Missing 'specification' in res.attrs for {expected_estimator}"
    spec = res.attrs["specification"]
    assert isinstance(spec, dict), f"attrs['specification'] must be dict, got {type(spec)}"

    missing_keys = REQUIRED_SPEC_KEYS - set(spec.keys())
    assert not missing_keys, f"{expected_estimator} missing spec keys: {missing_keys}"

    assert spec["estimator"] == expected_estimator
    assert isinstance(spec["method"], str) and len(spec["method"]) > 0
    assert spec["y_name"] == expected_y
    assert spec["x_name"] == expected_x
    assert isinstance(spec["controls"], list)
    assert isinstance(spec["n_lags"], int)
    assert isinstance(spec["horizons"], list)
    assert isinstance(spec["cov_type"], str) and len(spec["cov_type"]) > 0
    assert isinstance(spec["ci_level"], float)
    assert 0.0 < spec["ci_level"] < 1.0

    # Test summary formatting
    s = res.summary()
    assert isinstance(s, str) and len(s) > 0
    assert f"lags: {spec['n_lags']}" in s
    assert "SE:" in s
    assert "N:" in s
    if spec["controls"]:
        assert "controls:" in s


# -------------------------------------------------------------------------
# Test 1: la_lp & la_lp_iv
# -------------------------------------------------------------------------
def test_la_lp_specification_standards(ts_data):
    res = la_lp(
        ts_data,
        y="y",
        x="x",
        controls=["c1"],
        horizons=range(0, 4),
        n_lags=2,
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "la_lp", "y", "x")
    assert res.attrs["specification"]["controls"] == ["c1"]
    assert res.attrs["specification"]["cov_type"] == "HC0"


def test_la_lp_iv_specification_standards(ts_data):
    res = la_lp_iv(
        ts_data,
        y="y",
        x="x",
        z="z",
        controls=["c1"],
        horizons=range(0, 4),
        n_lags=2,
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "la_lp_iv", "y", "x")
    assert "z_names" in res.attrs["specification"]


# -------------------------------------------------------------------------
# Test 2: state_dep & state_dep_iv
# -------------------------------------------------------------------------
def test_lp_state_dep_specification_standards(ts_data):
    res = lp_state_dep(
        ts_data,
        y="y",
        x="x",
        state="state",
        controls=["c1"],
        horizons=range(0, 4),
        n_lags=2,
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "lp_state_dep", "y", "x")
    assert res.attrs["specification"]["state_name"] == "state"


def test_lp_state_dep_iv_specification_standards(ts_data):
    res = lp_state_dep_iv(
        ts_data,
        y="y",
        x="x",
        z="z",
        state="state",
        controls=["c1"],
        horizons=range(0, 4),
        n_lags=2,
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "lp_state_dep_iv", "y", "x")
    assert res.attrs["specification"]["state_name"] == "state"


# -------------------------------------------------------------------------
# Test 3: lp_did
# -------------------------------------------------------------------------
def test_lp_did_specification_standards(did_panel_data):
    res = lp_did(
        did_panel_data,
        y="y",
        treat="D",
        unit="unit",
        time="time",
        horizons=range(0, 4),
        pre_window=2,
        controls=["c1"],
        alpha=0.10,
        lags=1,
    )
    assert isinstance(res, LPDiDResult)
    assert isinstance(res.estimates, LPResult)
    assert "specification" in res.attrs
    assert "specification" in res.estimates.attrs
    assert res.attrs["specification"] == res.estimates.attrs["specification"]

    _assert_spec_standard(res, "lp_did", "y", "D")
    _assert_spec_standard(res.estimates, "lp_did", "y", "D")


# -------------------------------------------------------------------------
# Test 4: quantile
# -------------------------------------------------------------------------
def test_lp_quantile_specification_standards(ts_data):
    res = lp_quantile(
        ts_data,
        y="y",
        x="x",
        quantiles=[0.25, 0.75],
        controls=["c1"],
        horizons=range(0, 3),
        n_lags=2,
        n_boot=10,
        alpha=0.10,
        seed=1,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "lp_quantile", "y", "x")
    assert res.attrs["specification"]["cov_type"] == "bootstrap"
    assert res.attrs["specification"]["quantiles"] == [0.25, 0.75]


# -------------------------------------------------------------------------
# Test 5: asymmetric
# -------------------------------------------------------------------------
def test_lp_asymmetric_specification_standards(ts_data):
    res = lp_asymmetric(
        ts_data,
        y="y",
        x="x",
        controls=["c1"],
        horizons=range(0, 4),
        n_lags=2,
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "lp_asymmetric", "y", "x")
    assert res.attrs["specification"]["cov_type"] == "HAC"


# -------------------------------------------------------------------------
# Test 6: cce
# -------------------------------------------------------------------------
def test_cce_panel_lp_specification_standards(panel_data):
    res = cce_panel_lp(
        panel_data,
        y="y",
        x="x",
        controls=["c1"],
        horizons=range(0, 3),
        n_lags=1,
        alpha=0.10,
        entity_level="code",
        time_level="date",
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "cce_panel_lp", "y", "x")
    assert res.attrs["specification"]["cov_type"] == "cluster"


# -------------------------------------------------------------------------
# Test 7: mean_group
# -------------------------------------------------------------------------
def test_mean_group_panel_lp_specification_standards(panel_data):
    res = mean_group_panel_lp(
        panel_data,
        y="y",
        x="x",
        controls=["c1"],
        horizons=range(0, 3),
        n_lags=1,
        alpha=0.10,
        entity_level="code",
        time_level="date",
        min_obs=15,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "mean_group_panel_lp", "y", "x")
    assert res.attrs["specification"]["cov_type"] == "mean_group"


# -------------------------------------------------------------------------
# Test 8: iv_lewbel
# -------------------------------------------------------------------------
def test_lp_iv_lewbel_specification_standards(panel_data):
    res = lp_iv_lewbel(
        panel_data,
        y="y",
        x_endog="x",
        heterosk_source="z",
        controls=["c1"],
        horizons=range(0, 3),
        n_lags=1,
        entity_level="code",
        time_level="date",
        alpha=0.10,
    )
    assert isinstance(res, LPResult)
    _assert_spec_standard(res, "lp_iv_lewbel", "y", "x")
    assert "first_stage_f" in res.columns
    assert "first_stage_F" in res.columns
    # Summary should format F_1st column cleanly
    s = res.summary()
    assert "F_1st" in s
