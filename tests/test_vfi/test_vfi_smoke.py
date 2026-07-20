"""Tiny end-to-end solve on the always-available numpy core (Pyodide-safe)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, tauchen


@pytest.mark.pyodide_smoke
def test_vfi_end_to_end_numpy():
    z_grid, P = tauchen(n=3, rho=0.8, sigma=0.1)
    a_grid = np.linspace(0.1, 4.0, 25)

    def rf(ap, a, z, r, xp=np):
        c = (1.0 + r) * a + xp.exp(z) - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

    sol = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                     beta=0.95, params={"r": 0.03}).solve("numpy")
    assert sol.V.shape == (25, 3)
    assert sol.sup_norm < 1e-8
    assert sol.backend == "numpy"
