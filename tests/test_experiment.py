"""Tests for puremacro.experiment.run_experiment dispatcher."""
import numpy as np
import pandas as pd
import pytest

from puremacro.experiment import run_experiment
from puremacro.lp.jorda import lp_hac
from puremacro.lp.panel import panel_lp


def _toy_panel(seed=0, T=80, codes=("USA", "MEX")):
    rng = np.random.default_rng(seed)
    rows = []
    for code in codes:
        eps_x = rng.standard_normal(T)
        eps_y = rng.standard_normal(T)
        x = np.zeros(T)
        y = np.zeros(T)
        for t in range(1, T):
            x[t] = 0.6 * x[t - 1] + eps_x[t]
            y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + eps_y[t]
        idx = pd.MultiIndex.from_product(
            [[code], pd.date_range("2000-01-01", periods=T, freq="QE")],
            names=["code", "date"],
        )
        rows.append(pd.DataFrame({"x": x, "y": y}, index=idx))
    return pd.concat(rows).sort_index()


@pytest.fixture
def wide():
    return _toy_panel()


def test_run_experiment_lp_shape(wide):
    out = run_experiment(
        wide, outcome="y", proxy="x", method="lp",
        country="USA", horizons=range(0, 9), n_lags=2,
    )
    assert out["method"] == "lp"
    assert isinstance(out["irf_point"], pd.Series)
    assert len(out["irf_point"]) == 9
    assert len(out["irf_lower"]) == 9
    assert len(out["irf_upper"]) == 9
    assert out["n_obs"] > 0
    assert out["alpha"] == 0.10
    assert out["y_units"] == "%"
    assert np.isfinite(out["shock_sd"])


def test_run_experiment_lp_round_trip(wide):
    horizons = list(range(0, 9))
    out = run_experiment(
        wide, outcome="y", proxy="x", method="lp",
        country="USA", horizons=horizons, n_lags=2,
        normalize="raw",
    )
    sub = wide.xs("USA", level="code").copy()
    sub.index = pd.DatetimeIndex(sub.index)
    sub = sub.sort_index()
    direct = lp_hac(sub, y="y", x="x", horizons=horizons, n_lags=2,
                    alpha=0.10)
    expected = direct.set_index("h")["beta"]
    np.testing.assert_allclose(
        out["irf_point"].values, expected.values, atol=1e-12
    )


def test_run_experiment_panel_lp_shape(wide):
    out = run_experiment(
        wide, outcome="y", proxy="x", method="panel_lp",
        horizons=range(0, 9), n_lags=2,
    )
    assert out["method"] == "panel_lp"
    assert len(out["irf_point"]) == 9
    assert out["n_countries"] == 2
    assert out["alpha"] == 0.10


def test_run_experiment_panel_lp_round_trip(wide):
    horizons = list(range(0, 9))
    out = run_experiment(
        wide, outcome="y", proxy="x", method="panel_lp",
        horizons=horizons, n_lags=2, normalize="raw",
    )
    direct = panel_lp(wide, y="y", x="x", horizons=horizons, n_lags=2,
                     controls=[], alpha=0.10)
    expected = direct.set_index("h")["beta"]
    np.testing.assert_allclose(
        out["irf_point"].values, expected.values, atol=1e-12
    )


def test_run_experiment_var_cholesky_shape(wide):
    out = run_experiment(
        wide, outcome="y", proxy="x", method="var_cholesky",
        country="USA", horizons=range(0, 9), n_lags=2,
        n_boot=50, seed=0,
    )
    assert out["method"] == "var_cholesky"
    assert isinstance(out["irf_point"], pd.DataFrame)
    assert list(out["irf_point"].index) == list(range(9))
    assert list(out["irf_point"].columns) == ["x", "y"]
    assert out["irf_lower"].shape == out["irf_point"].shape
    assert out["irf_upper"].shape == out["irf_point"].shape
    # At h=0, lower <= upper element-wise (within tolerance for ties).
    diff = out["irf_upper"].values - out["irf_lower"].values
    assert (diff >= -1e-9).all()


def test_run_experiment_var_cholesky_normalize_raw_vs_pct(wide):
    raw = run_experiment(
        wide, outcome="y", proxy="x", method="var_cholesky",
        country="USA", horizons=range(0, 9), n_lags=2,
        n_boot=20, seed=0, normalize="raw",
    )
    pct = run_experiment(
        wide, outcome="y", proxy="x", method="var_cholesky",
        country="USA", horizons=range(0, 9), n_lags=2,
        n_boot=20, seed=0, normalize="1sd_pct",
    )
    np.testing.assert_allclose(
        pct["irf_point"].values, raw["irf_point"].values * 100.0,
        atol=1e-9,
    )
    assert pct["y_units"] == "%/pp (per-column)"
    assert pct["shock_sd"] is None


def test_run_experiment_validation_errors(wide):
    with pytest.raises(ValueError):
        run_experiment(wide, outcome="y", proxy="x", method="bogus",
                      country="USA")
    with pytest.raises(ValueError):
        run_experiment(wide, outcome="y", proxy="x", method="lp",
                      country="USA", normalize="bogus")
    with pytest.raises(ValueError):
        run_experiment(wide, outcome="y", proxy="x", method="lp")
    with pytest.raises(ValueError):
        run_experiment(wide, outcome="y", proxy="x", method="var_cholesky")
