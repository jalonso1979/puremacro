"""Tests for puremacro.hfi result dataclasses."""
import numpy as np
import pytest

from puremacro.hfi._results import JKResult


def test_jk_result_is_frozen():
    res = JKResult(
        mp_shock=np.zeros(10),
        info_shock=np.zeros(10),
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )
    with pytest.raises(Exception):
        res.method = "median_target"


def test_jk_result_summary_poor_man():
    res = JKResult(
        mp_shock=np.array([1.0, 0.0, -2.0]),
        info_shock=np.array([0.0, 3.0, 0.0]),
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )
    s = res.summary()
    assert "poor_man" in s.lower() or "Poor" in s


def test_jk_result_summary_median_target():
    res = JKResult(
        mp_shock=np.zeros(5),
        info_shock=np.zeros(5),
        rotation=np.eye(2),
        n_admissible=1234,
        method="median_target",
    )
    s = res.summary()
    assert "median_target" in s.lower() or "Median" in s
    assert "1234" in s
