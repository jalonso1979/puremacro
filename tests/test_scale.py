"""Tests for puremacro.scale: to_1sd_pct + peak_irf."""
import numpy as np
import pandas as pd
import pytest

from puremacro.scale import (
    RATE_OUTCOMES,
    is_rate_outcome,
    peak_irf,
    to_1sd_pct,
)


def _fake_series_result(values):
    s = pd.Series(values)
    return {
        "method": "lp",
        "irf_point": s.copy(),
        "irf_lower": s.copy() - 0.01,
        "irf_upper": s.copy() + 0.01,
    }


def test_is_rate_outcome():
    assert is_rate_outcome("urate")
    assert is_rate_outcome("d_log_cpi")
    assert not is_rate_outcome("log_gfcf_real")
    assert "urate" in RATE_OUTCOMES


def test_to_1sd_pct_log_outcome_with_xsd():
    vals = np.array([0.01, 0.02, 0.03, 0.025, 0.018])
    res = _fake_series_result(vals)
    out = to_1sd_pct(res, x_sd=2.0, outcome_name="log_gfcf_real")
    np.testing.assert_allclose(out["irf_point"].values, vals * 200.0)
    np.testing.assert_allclose(out["irf_lower"].values, (vals - 0.01) * 200.0)
    np.testing.assert_allclose(out["irf_upper"].values, (vals + 0.01) * 200.0)
    assert out["y_units"] == "%"
    assert out["shock_sd"] == 2.0


def test_to_1sd_pct_rate_outcome():
    vals = np.array([0.01, 0.02, 0.03, 0.025, 0.018])
    res = _fake_series_result(vals)
    out = to_1sd_pct(res, x_sd=2.0, outcome_name="urate")
    np.testing.assert_allclose(out["irf_point"].values, vals * 2.0)
    np.testing.assert_allclose(out["irf_lower"].values, (vals - 0.01) * 2.0)
    np.testing.assert_allclose(out["irf_upper"].values, (vals + 0.01) * 2.0)
    assert out["y_units"] == "pp"
    assert out["shock_sd"] == 2.0


def test_to_1sd_pct_no_xsd():
    vals = np.array([0.01, 0.02])
    res = _fake_series_result(vals)
    out = to_1sd_pct(res, outcome_name="log_gfcf_real")
    np.testing.assert_allclose(out["irf_point"].values, vals * 100.0)
    assert out["shock_sd"] is None


def test_to_1sd_pct_requires_outcome_info():
    res = _fake_series_result(np.array([0.01]))
    with pytest.raises(ValueError):
        to_1sd_pct(res, x_sd=1.0)


def test_peak_irf_series():
    s = pd.Series([0.01, -0.05, 0.03], index=[0, 1, 2])
    res = {
        "irf_point": s,
        "irf_lower": s - 0.005,
        "irf_upper": s + 0.005,
    }
    out = peak_irf(res)
    assert out["h"] == 1
    assert out["beta"] == pytest.approx(-0.05)
    assert out["lo"] == pytest.approx(-0.055)
    assert out["hi"] == pytest.approx(-0.045)


def test_peak_irf_dataframe():
    idx = pd.Index([0, 1, 2], name="h")
    pt = pd.DataFrame({"x": [0.1, -0.5, 0.3], "y": [0.01, 0.02, 0.7]},
                     index=idx)
    res = {
        "irf_point": pt,
        "irf_lower": pt - 0.01,
        "irf_upper": pt + 0.01,
    }
    out_x = peak_irf(res, response_var="x")
    assert out_x["h"] == 1
    assert out_x["beta"] == pytest.approx(-0.5)

    out_y = peak_irf(res, response_var="y")
    assert out_y["h"] == 2
    assert out_y["beta"] == pytest.approx(0.7)

    with pytest.raises(ValueError):
        peak_irf(res)
