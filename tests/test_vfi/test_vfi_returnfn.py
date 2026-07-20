from __future__ import annotations

import numpy as np

from puremacro.vfi.returnfn import build_return_tensor


def test_shape_no_decision():
    a = np.linspace(0.1, 2.0, 4)
    z = np.array([-0.1, 0.1])
    R = build_return_tensor(
        lambda ap, a, z, c, xp=np: c + a - ap + 0.0 * z, a, z, {"c": 1.0}
    )
    assert R.shape == (1, 4, 4, 2)


def test_values_no_decision():
    a = np.array([0.0, 1.0])
    z = np.array([0.0])
    # R(ap, a, z) = a - ap ; axes are (d=1, a', a, z)
    R = build_return_tensor(lambda ap, a, z, xp=np: a - ap + 0.0 * z, a, z, {})
    assert R[0, 0, 0, 0] == 0.0   # a'=0, a=0
    assert R[0, 0, 1, 0] == 1.0   # a'=0, a=1
    assert R[0, 1, 0, 0] == -1.0  # a'=1, a=0
    assert R[0, 1, 1, 0] == 0.0   # a'=1, a=1


def test_shape_with_decision():
    a = np.linspace(0.0, 1.0, 3)
    z = np.array([0.0, 1.0])
    d = np.array([0.0, 0.5, 1.0])
    R = build_return_tensor(
        lambda dd, ap, a, z, xp=np: dd + a - ap + 0.0 * z, a, z, {}, d_grid=d
    )
    assert R.shape == (3, 3, 3, 2)


def test_params_positional_in_insertion_order():
    a = np.array([1.0, 2.0])
    z = np.array([0.0])
    R = build_return_tensor(
        lambda ap, a, z, r, w, xp=np: r * 100.0 + w + 0.0 * (a - ap + z),
        a, z, {"r": 0.03, "w": 2.0},
    )
    assert np.allclose(R, 0.03 * 100.0 + 2.0)


def test_partial_return_broadcasts_up():
    # return independent of a' (only depends on a, z) must still fill axis 0
    a = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0])
    R = build_return_tensor(lambda ap, a, z, xp=np: a + 0.0 * z, a, z, {})
    assert R.shape == (1, 3, 3, 1)
    for ap in range(3):
        for ia in range(3):
            assert R[0, ap, ia, 0] == a[ia]
