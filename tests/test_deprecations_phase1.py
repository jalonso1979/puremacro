"""Tests for Phase 1 deprecation warnings."""
import pytest
import numpy as np
import pandas as pd


def test_regress_lp_emits_future_warning():
    from puremacro.regress.lp import lp_panel

    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "shock": [0.1, -0.1, 0.2, -0.2],
        "unit": ["a", "a", "b", "b"],
        "date": [1, 2, 1, 2],
    })
    with pytest.deprecated_call():
        lp_panel(df, y="y", shock="shock", horizons=[0], unit="unit", date="date", unit_fe=False)


def test_sigma_numpy_emits_future_warning():
    from puremacro.sigma.sigma_numpy import SigmaObject

    with pytest.deprecated_call():
        SigmaObject(
            sigma=np.array([1.0, 2.0]),
            R=np.eye(2),
            labels=["x1", "x2"],
        )


def test_lp_garch_utils_emits_future_warning():
    with pytest.deprecated_call():
        import puremacro.lp.garch_utils  # noqa: F401
