import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from puremacro.build_panel import compute_garch_sigma

@pytest.fixture
def mock_arch():
    with patch("arch.arch_model") as mock_model:
        mock_fit = MagicMock()
        mock_result = MagicMock()
        mock_model.return_value = mock_fit
        mock_fit.fit.return_value = mock_result

        def fit_side_effect(**kwargs):
            args, _ = mock_model.call_args
            x = args[0]
            mock_result.conditional_volatility = np.ones(len(x)) * 50.0
            return mock_result

        mock_fit.fit.side_effect = fit_side_effect
        yield mock_model

def test_compute_garch_sigma_happy_path(mock_arch):
    dates = pd.date_range("2000-01-01", periods=45, freq="ME")
    df = pd.DataFrame({
        "code": "USA",
        "date": dates,
        "variable": "unc_proxy",
        "value": np.linspace(1, 10, 45)
    })

    res = compute_garch_sigma(df, "unc_proxy")

    assert len(res) == 44
    assert list(res.columns) == ["code", "date", "variable", "value", "sa_source", "source"]
    assert (res["code"] == "USA").all()
    assert (res["variable"] == "garch_sigma_unc_proxy").all()
    assert (res["value"] == 50.0 / 100.0).all()


def test_compute_garch_sigma_min_obs():
    # Only 39 points, needs 40
    dates = pd.date_range("2000-01-01", periods=39, freq="ME")
    df = pd.DataFrame({
        "code": "USA",
        "date": dates,
        "variable": "unc_proxy",
        "value": np.linspace(1, 10, 39)
    })

    res = compute_garch_sigma(df, "unc_proxy")
    assert res.empty
    assert list(res.columns) == ["code", "date", "variable", "value", "sa_source", "source"]

def test_compute_garch_sigma_arch_exception(mock_arch):
    # Mock arch throwing an exception during fit
    mock_arch.return_value.fit.side_effect = Exception("Convergence failed")

    dates = pd.date_range("2000-01-01", periods=45, freq="ME")
    df = pd.DataFrame({
        "code": "USA",
        "date": dates,
        "variable": "unc_proxy",
        "value": np.linspace(1, 10, 45)
    })

    res = compute_garch_sigma(df, "unc_proxy")
    assert res.empty

def test_compute_garch_sigma_multi_country(mock_arch):
    dates = pd.date_range("2000-01-01", periods=45, freq="ME")
    df_usa = pd.DataFrame({
        "code": "USA",
        "date": dates,
        "variable": "unc_proxy",
        "value": np.linspace(1, 10, 45)
    })
    df_gbr = pd.DataFrame({
        "code": "GBR",
        "date": dates,
        "variable": "unc_proxy",
        "value": np.linspace(2, 20, 45)
    })

    df = pd.concat([df_usa, df_gbr])

    res = compute_garch_sigma(df, "unc_proxy")

    assert len(res) == 88 # 44 per country
    assert set(res["code"]) == {"USA", "GBR"}
    assert (res["variable"] == "garch_sigma_unc_proxy").all()
