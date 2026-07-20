from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.returnfn import build_return_tensor


def test_single_shock_unchanged():
    a = np.linspace(0.0, 5.0, 6)
    z = np.array([-0.1, 0.1])

    def rf(ap, a_, z_, r, xp=np):
        return (1.0 + r) * a_ + z_ - ap

    R = build_return_tensor(rf, a, z, {"r": 0.03})
    assert R.shape == (1, 6, 6, 2)
    for api in range(6):
        for ai in range(6):
            for zi in range(2):
                assert R[0, api, ai, zi] == pytest.approx(1.03 * a[ai] + z[zi] - a[api])


def test_two_shocks_shape_and_values():
    a = np.array([0.0, 1.0, 2.0])         # n_a=3
    z1 = np.array([0.0, 1.0])             # 2
    z2 = np.array([0.0, 10.0, 20.0])      # 3  -> n_z = 6 (C-order, z2 fastest)

    def rf(ap, a_, z1_, z2_, xp=np):
        return a_ + z1_ + z2_ - ap        # shocks enter separately

    R = build_return_tensor(rf, a, [z1, z2], {})
    assert R.shape == (1, 3, 3, 6)

    def zc(s):                            # flat z-state s -> (z1, z2)
        return z1[s // 3], z2[s % 3]

    for ai in range(3):
        for s in range(6):
            zz1, zz2 = zc(s)
            assert R[0, 0, ai, s] == pytest.approx(a[ai] + zz1 + zz2 - 0.0)


def test_two_shocks_with_decision_and_multi_asset():
    # multi-asset AND multi-shock together: order [d,] a'_1..a'_K, a_1..a_K, z_1..z_M
    m = np.array([0.0, 1.0])
    k = np.array([0.0, 1.0])              # n_a=4
    z1 = np.array([0.0, 1.0])
    z2 = np.array([0.0, 1.0])             # n_z=4
    dg = np.array([1.0, 2.0])

    def rf(d, m_p, k_p, m_, k_, z1_, z2_, xp=np):
        return d * (m_ + k_) + z1_ - z2_ - m_p - k_p

    R = build_return_tensor(rf, [m, k], [z1, z2], {}, d_grid=dg)
    assert R.shape == (2, 4, 4, 4)


def test_single_element_z_list_equals_bare():
    a = np.linspace(0.0, 3.0, 5)
    z = np.array([0.0, 0.2, 0.4])

    def rf(ap, a_, z_, xp=np):
        return a_ + z_ - ap

    np.testing.assert_allclose(build_return_tensor(rf, a, [z], {}),
                               build_return_tensor(rf, a, z, {}), atol=1e-12)
