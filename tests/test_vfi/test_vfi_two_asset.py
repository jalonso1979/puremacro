from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import two_asset_profile


def test_shapes_and_distribution():
    out = two_asset_profile(n_z=5, n_m=18, n_k=18)
    sol = out["solution"]
    assert sol.endo_shape == (18, 18)
    assert sol.V.shape == (18 * 18, 5)
    mu = out["distribution"]
    assert mu.shape == (18 * 18, 5)
    np.testing.assert_allclose(mu.sum(), 1.0, atol=1e-9)
    for k in ("solution", "distribution", "mean_liquid", "mean_illiquid",
              "illiquid_share", "wealth_gini"):
        assert k in out


def test_illiquid_asset_is_used_and_share_interior():
    out = two_asset_profile(n_z=5, n_m=18, n_k=18)
    assert out["mean_illiquid"] > 0.0                    # the high-return asset is held
    assert out["mean_liquid"] >= 0.0
    assert 0.0 < out["illiquid_share"] < 1.0             # a genuine two-asset portfolio
    assert 0.0 <= out["wealth_gini"] <= 1.0


def test_wider_return_gap_raises_illiquid_share():
    # a bigger illiquid premium tilts the portfolio toward the illiquid asset.
    narrow = two_asset_profile(r_illiquid=0.03, n_z=5, n_m=18, n_k=18)
    wide = two_asset_profile(r_illiquid=0.06, n_z=5, n_m=18, n_k=18)
    assert wide["illiquid_share"] > narrow["illiquid_share"]


def test_two_asset_exported():
    from puremacro.vfi import two_asset_profile as fn

    assert fn is two_asset_profile
