"""Regression + new-behaviour tests for the DMP vacancy-cost extension."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.dmp_regime_dependent import (
    DMPParameters, dmp_irf, dmp_steady_state,
)


@pytest.fixture
def baseline_params():
    """Canonical calibration matching R5_03's grid-min point."""
    return DMPParameters(
        beta_bar=0.996, alpha=0.5, eta_b=0.5, s=0.034, z=0.51,
        y_bar=1.0, c=0.5, mu=0.5, eta=0.067, kappa=0.040,
        rho_sigma=0.7, psi=0.5,
    )


def test_phi_v_zero_preserves_baseline_irf(baseline_params):
    """phi_v=0.0 must give identical IRF to a no-phi_v reference run."""
    p_zero = baseline_params  # phi_v defaults to 0.0
    p_explicit = DMPParameters(**{**baseline_params.__dict__, 'phi_v': 0.0})
    irf_default = dmp_irf(p_zero, sigma_shock=1.0, regime='H', horizon=8)
    irf_explicit = dmp_irf(p_explicit, sigma_shock=1.0, regime='H', horizon=8)
    np.testing.assert_allclose(irf_default.log_u, irf_explicit.log_u, atol=1e-12)
    np.testing.assert_allclose(irf_default.log_v, irf_explicit.log_v, atol=1e-12)


def test_phi_v_steady_state_unchanged(baseline_params):
    """Steady state must be invariant to phi_v (adjustment terms vanish at v_lag==v)."""
    ss_zero = dmp_steady_state(baseline_params, sigma=0.0)
    p_high_phi = DMPParameters(**{**baseline_params.__dict__, 'phi_v': 5.0})
    ss_high = dmp_steady_state(p_high_phi, sigma=0.0)
    assert abs(ss_zero.theta - ss_high.theta) < 1e-10
    assert abs(ss_zero.v - ss_high.v) < 1e-10
    assert abs(ss_zero.u - ss_high.u) < 1e-10


def test_phi_v_dampens_u_response(baseline_params):
    """Positive phi_v should DAMPEN the on-impact |u| response.

    Vacancy-adjustment costs spread posting over time, so the on-impact vacancy
    change can be *larger* with phi_v (firms front-load more), but the resulting
    within-period matching and unemployment effect is smaller because the
    adjustment is being phased — the backward-looking cost term raises effective
    posting cost for the lagged-below-target vacancies, slowing the u response.
    """
    p_zero = baseline_params
    p_phi = DMPParameters(**{**baseline_params.__dict__, 'phi_v': 2.0})
    irf_zero = dmp_irf(p_zero, sigma_shock=1.0, regime='H', horizon=8)
    irf_phi = dmp_irf(p_phi, sigma_shock=1.0, regime='H', horizon=8)
    assert abs(irf_phi.log_u[0]) < abs(irf_zero.log_u[0]) + 1e-6, (
        f"phi_v should dampen on-impact |log_u|: "
        f"zero={irf_zero.log_u[0]:.4f}, phi=2.0={irf_phi.log_u[0]:.4f}"
    )


def test_phi_v_generates_hump_in_loose_regime(baseline_params):
    """Positive phi_v + loose regime should produce a peak at h>=1 (hump-shape)."""
    p_phi = DMPParameters(**{**baseline_params.__dict__, 'phi_v': 2.0, 'kappa': 0.5})
    irf = dmp_irf(p_phi, sigma_shock=1.0, regime='L', horizon=8)
    peak_h = int(np.argmax(np.abs(irf.log_u)))
    assert peak_h >= 1, (
        f"loose-regime |log_u| peak should be at h>=1 with phi_v=2.0, kappa=0.5; "
        f"got peak_h={peak_h}, log_u={irf.log_u}"
    )
