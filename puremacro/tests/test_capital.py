import numpy as np
import pytest
from puremacro.capital import _pim

def test_pim_basic():
    # dq = 0.1, k0 = 100.0, mid = False
    inv = np.array([10.0, 20.0, 30.0])
    k = _pim(inv, 0.1, 100.0, False)

    assert len(k) == 3
    # k[0] = 100.0
    # k[1] = 100 * 0.9 + 10 = 100
    # k[2] = 100 * 0.9 + 20 = 110

    np.testing.assert_allclose(k, [100.0, 100.0, 110.0])

def test_pim_mid():
    # dq = 0.19, k0 = 100.0, mid = True
    inv = np.array([10.0, 20.0, 30.0])

    half = (1.0 - 0.19) ** 0.5  # 0.81 ** 0.5 = 0.9
    k = _pim(inv, 0.19, 100.0, True)

    assert len(k) == 3
    # k[0] = 0.9 * 100.0 = 90.0
    # k[1] = 90.0 * 0.81 + 0.9 * 10 = 72.9 + 9.0 = 81.9
    # k[2] = 81.9 * 0.81 + 0.9 * 20 = 66.339 + 18.0 = 84.339

    np.testing.assert_allclose(k, [90.0, 81.9, 84.339])

def test_pim_nan_inf():
    # dq = 0.1, k0 = 100.0, mid = False
    inv = np.array([np.nan, np.inf, -np.inf, 10.0])
    k = _pim(inv, 0.1, 100.0, False)

    # k[0] = 100.0
    # k[1] = 100 * 0.9 + 0 = 90.0
    # k[2] = 90 * 0.9 + 0 = 81.0
    # k[3] = 81 * 0.9 + 0 = 72.9

    np.testing.assert_allclose(k, [100.0, 90.0, 81.0, 72.9])

def test_pim_empty():
    inv = np.array([])
    k = _pim(inv, 0.1, 100.0, False)
    assert len(k) == 0
    np.testing.assert_allclose(k, [])

def test_pim_single():
    inv = np.array([10.0])
    k = _pim(inv, 0.1, 100.0, False)
    assert len(k) == 1
    np.testing.assert_allclose(k, [100.0])
