"""Perfect-foresight transition + state-dependent IRFs of the nested DMP."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import dynamics as D
from puremacro.models.nested_dmp import estimation as E


def test_belief_path_updates_toward_true_type():
    p = NestedDMPParameters()
    sigma_path = 1.0 * (p.rho_sigma ** np.arange(12))  # decaying shock
    pi_dove = D.belief_path(p, sigma_path=sigma_path, fed_type="dove")
    pi_hawk = D.belief_path(p, sigma_path=sigma_path, fed_type="hawk")
    assert pi_dove.shape == sigma_path.shape
    assert pi_dove[0] == pytest.approx(p.prior_pi0)   # starts at the prior
    # Dove cuts -> rates reveal a dove -> belief rises toward 1; hawk -> falls.
    assert pi_dove[-1] > p.prior_pi0 > pi_hawk[-1]
    assert np.all((pi_dove >= 0.0) & (pi_dove <= 1.0))


def test_belief_path_constant_without_shock():
    # sigma=0 -> dove and hawk rates coincide (r=r*) -> no information -> pi stays.
    p = NestedDMPParameters()
    sigma_path = np.zeros(8)
    pi = D.belief_path(p, sigma_path=sigma_path, fed_type="hawk")
    np.testing.assert_allclose(pi, p.prior_pi0)


def test_theta_sweep_zero_shock_holds_steady_state():
    # With sigma_path == 0 and pi constant, the backward sweep must return the
    # fixed-grid steady-state theta at every period (the zero-shock reduction).
    p = NestedDMPParameters()
    T = 40
    sigma_path = np.zeros(T + 1)
    pi_path = np.full(T + 1, p.prior_pi0)
    grid, erg = D._fixed_grid(p, sigma_ref=1.0)
    ss = D._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
    theta_path, J_path = D._theta_sweep(p, sigma_path=sigma_path, pi_path=pi_path,
                                        grid=grid, erg=erg, ss=ss)
    np.testing.assert_allclose(theta_path, ss["theta"], rtol=1e-6)


def test_forward_sim_zero_shock_holds_steady_state():
    from dataclasses import replace
    p0 = NestedDMPParameters(h_max=3.0)
    p = replace(p0, mu=D.calibrate_mu(p0, sigma_ref=1.0))  # f(theta_ss)=mu<1 => stable LoM
    T = 40
    sigma_path = np.zeros(T + 1)
    pi_path = np.full(T + 1, p.prior_pi0)
    grid, erg = D._fixed_grid(p, sigma_ref=1.0)
    ss = D._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
    assert ss["theta"] == pytest.approx(1.0, rel=1e-6)  # mu-normalization => theta_ss=1
    theta_path, J_path = D._theta_sweep(p, sigma_path=sigma_path, pi_path=pi_path,
                                        grid=grid, erg=erg, ss=ss)
    paths = D._forward_sim(p, theta_path=theta_path, J_path=J_path,
                           sigma_path=sigma_path, pi_path=pi_path, grid=grid, erg=erg, ss=ss)
    u = paths["urate"]
    # Predetermined u sits at the steady state under no shock (f<1 => stable LoM).
    assert np.allclose(u, u[0], rtol=1e-6)
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.allclose(paths["lfpr"], paths["lfpr"][0], rtol=1e-6)


def test_simulate_irf_zero_shock_is_zero():
    from dataclasses import replace
    p0 = NestedDMPParameters(h_max=3.0)
    p = replace(p0, mu=D.calibrate_mu(p0, sigma_ref=1.0))   # f<1: stable dynamics
    irf = D.simulate_irf(p, sigma0=0.0, fed_type="dove", horizon=8)
    for arr in (irf.log_urate, irf.log_v, irf.log_theta, irf.log_lfpr):
        np.testing.assert_allclose(arr, 0.0, atol=1e-8)


def test_simulate_irf_shapes_and_alignment():
    from dataclasses import replace
    p0 = NestedDMPParameters(h_max=3.0)
    p = replace(p0, mu=D.calibrate_mu(p0, sigma_ref=1.0))
    irf = D.simulate_irf(p, sigma0=1.0, fed_type="dove", horizon=8)
    assert irf.log_theta.shape == (9,)         # horizons 0..8
    assert irf.fed_type == "dove"
    assert np.isfinite(irf.log_theta).all()


def test_simulate_irf_guards_unstable_calibration():
    # Default mu leaves f(theta_ss)>1 -> the forward LoM diverges; must raise.
    p = NestedDMPParameters(h_max=3.0)   # un-normalized mu
    with pytest.raises(ValueError, match="f.*theta|calibrate_mu|stable"):
        D.simulate_irf(p, sigma0=1.0, fed_type="dove", horizon=8)


def test_irf_sign_flip_dove_vs_hawk():
    from dataclasses import replace
    p0 = NestedDMPParameters(h_max=3.0)
    p = replace(p0, mu=D.calibrate_mu(p0, sigma_ref=1.0))  # f<1: stable dynamics
    dove = D.simulate_irf(p, sigma0=1.0, fed_type="dove", horizon=8)
    hawk = D.simulate_irf(p, sigma0=1.0, fed_type="hawk", horizon=8)
    h = 4  # a focal horizon past the predetermined-u impact period
    # Tightness: dove expands (theta rises), hawk contracts (theta falls).
    assert dove.log_theta[h] > 0.0 > hawk.log_theta[h]
    # Unemployment moves opposite to tightness: dove urate falls, hawk rises.
    assert dove.log_urate[h] < 0.0 < hawk.log_urate[h]


@pytest.mark.slow
def test_wage_rigidity_amplifies_hawk_contraction():
    from dataclasses import replace
    base = E.baseline_calibration(NestedDMPParameters(psi_down=0.0))   # flexible, recalibrated
    rigid = replace(base, psi_down=0.8)                                # downward-rigid
    hk_flex = D.simulate_irf(base,  sigma0=1.0, fed_type="hawk", horizon=8)
    hk_rig  = D.simulate_irf(rigid, sigma0=1.0, fed_type="hawk", horizon=8)
    h = 4
    # Rigidity deepens the hawk contraction: tightness falls MORE, urate rises MORE.
    assert hk_rig.log_theta[h] < hk_flex.log_theta[h] < 0.0
    assert hk_rig.log_urate[h] > hk_flex.log_urate[h] > 0.0
    # Sanity: psi_down=0 reproduces the flexible IRF exactly (no rigidity).
    hk_zero = D.simulate_irf(replace(base, psi_down=0.0), sigma0=1.0, fed_type="hawk", horizon=8)
    np.testing.assert_allclose(hk_zero.log_theta, hk_flex.log_theta, atol=1e-10)


@pytest.mark.slow
def test_high_rigidity_does_not_crash():
    # Extreme downward rigidity compresses the surplus toward entry collapse; the
    # free-entry solve must degrade gracefully (boundary tightness), never raise.
    base = E.baseline_calibration(NestedDMPParameters(rho_sigma=0.9))
    p = replace(base, psi_down=0.85)
    irf = D.simulate_irf(p, sigma0=1.0, fed_type="hawk", horizon=8)
    assert np.all(np.isfinite(irf.log_theta))
    assert np.all(np.isfinite(irf.log_urate))
