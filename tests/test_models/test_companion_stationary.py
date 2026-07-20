"""Tests for the sigma_scaled_process MPS gate (sigma_x_loading)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import equilibrium as S


def test_sigma_scaled_process_mps_gated_by_loading():
    # Default loading=0: sigma does NOT scale the idiosyncratic process (sigma -> beliefs only).
    p_off = NestedDMPParameters(n_x=15)
    g0, P0, e0 = S.sigma_scaled_process(p_off, sigma=0.0)
    g1, P1, e1 = S.sigma_scaled_process(p_off, sigma=1.0)
    np.testing.assert_allclose(g1, g0)                 # no MPS at loading=0
    np.testing.assert_allclose(P1, P0)
    # loading=1: sigma=1 doubles the innovation sd -> grid half-width doubles (MPS on).
    p_on = NestedDMPParameters(n_x=15, sigma_x_loading=1.0)
    h0, _, _ = S.sigma_scaled_process(p_on, sigma=0.0)
    h1, _, _ = S.sigma_scaled_process(p_on, sigma=1.0)
    assert h1[-1] == pytest.approx(2.0 * h0[-1], rel=1e-12)
    # Ergodic still a proper symmetric distribution.
    assert e0.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(e0, e0[::-1], atol=1e-10)
    np.testing.assert_allclose(P0.sum(axis=1), np.ones(15))


def test_sigma_scaled_process_rejects_sigma_below_minus_one():
    p = NestedDMPParameters()
    with pytest.raises(ValueError, match="sigma"):
        S.sigma_scaled_process(p, sigma=-1.5)
