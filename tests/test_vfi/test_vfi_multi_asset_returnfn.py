from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.returnfn import build_return_tensor


def test_single_asset_unchanged():
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


def test_two_assets_shape_and_values():
    g1 = np.array([0.0, 1.0, 2.0])      # size 3
    g2 = np.array([0.0, 10.0])          # size 2  -> n_a = 6
    z = np.array([0.0])

    def rf(m_p, k_p, m, k, z_, xp=np):
        return m + 10.0 * k - m_p - 10.0 * k_p

    R = build_return_tensor(rf, [g1, g2], z, {})
    assert R.shape == (1, 6, 6, 1)

    def comp(s):                         # flat s (C-order, k fastest) -> (m, k)
        return g1[s // 2], g2[s % 2]

    for s_ap in range(6):
        for s_a in range(6):
            m, k = comp(s_a)
            mp, kp = comp(s_ap)
            assert R[0, s_ap, s_a, 0] == pytest.approx(m + 10.0 * k - mp - 10.0 * kp)


def test_two_assets_with_decision():
    g1 = np.array([0.0, 1.0])
    g2 = np.array([0.0, 1.0])           # n_a = 4
    z = np.array([0.0])
    dg = np.array([0.5, 1.0])

    def rf(d, m_p, k_p, m, k, z_, xp=np):
        return d * (m + k) - m_p - k_p

    R = build_return_tensor(rf, [g1, g2], z, {}, d_grid=dg)
    assert R.shape == (2, 4, 4, 1)

    def comp(s):
        return g1[s // 2], g2[s % 2]

    for di in range(2):
        for s_ap in range(4):
            for s_a in range(4):
                m, k = comp(s_a)
                mp, kp = comp(s_ap)
                assert R[di, s_ap, s_a, 0] == pytest.approx(dg[di] * (m + k) - mp - kp)


def test_single_element_list_equals_bare_array():
    a = np.linspace(0.0, 3.0, 5)
    z = np.array([0.0, 0.2])

    def rf(ap, a_, z_, xp=np):
        return a_ + z_ - ap

    R_list = build_return_tensor(rf, [a], z, {})
    R_bare = build_return_tensor(rf, a, z, {})
    np.testing.assert_allclose(R_list, R_bare, atol=1e-12)
