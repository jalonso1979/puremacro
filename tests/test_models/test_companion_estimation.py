"""IRF-matching estimation + magnitude-gap diagnosis (Phase 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from puremacro.models.nested_dmp import estimation as E

# Frozen copies of the empirical LP-IRF target tables (originally
# notebooks/output_tables/36_lp_state_dep_*.csv in the research monorepo),
# vendored so the split repo's tests are self-contained.
_TABLES = Path(__file__).resolve().parents[1] / "fixtures" / "companion"


def test_load_empirical_irf_targets():
    tg = E.load_empirical_irf_targets(_TABLES)
    # urate, V, theta each present for loose and tight, length-9 (h=0..8).
    for outcome in ("urate", "v", "theta"):
        assert tg.loose[outcome].shape == (9,)
        assert tg.tight[outcome].shape == (9,)
        assert tg.loose_se[outcome].shape == (9,)
    # The empirical sign-flip is in the data: loose urate falls, tight rises.
    assert tg.loose["urate"][5] < 0.0 < tg.tight["urate"][5]
    # theta = V - U (log).
    np.testing.assert_allclose(tg.loose["theta"], tg.loose["v"] - tg.loose["urate"], atol=1e-12)


from dataclasses import replace
from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import dynamics as D


def test_calibrate_mu_to_f_hits_target():
    p = NestedDMPParameters()
    mu = E.calibrate_mu_to_f(p, f_target=0.5, sigma_ref=1.0)
    pf = replace(p, mu=mu)
    grid, erg = D._fixed_grid(pf, sigma_ref=1.0)
    ss = D._grid_steady_state(pf, grid=grid, erg=erg, sigma=0.0, pi=pf.prior_pi0)
    f_ss = pf.mu * ss["theta"] ** (1.0 - pf.alpha)
    assert f_ss == pytest.approx(0.5, rel=1e-4)  # realistic + < 1 (stable dynamics)


def test_model_regime_irfs_sign_flip():
    p0 = NestedDMPParameters()
    # Realistic f=0.5 (default sigma_x_loading=0 => sigma drives beliefs only), so the
    # belief-driven sign-flip holds at a credible calibration (the sigma-MPS channel,
    # which previously swamped it, is off by default).
    p = replace(p0, mu=E.calibrate_mu_to_f(p0, f_target=0.5))
    loose, tight = E.model_regime_irfs(p, sigma0=1.0, horizon=8)
    for outcome in ("urate", "v", "theta"):
        assert loose[outcome].shape == (9,)
    # Model sign-flip (matches the empirical direction): loose urate falls, tight rises.
    assert loose["urate"][4] < 0.0 < tight["urate"][4]


def test_fit_report_signs_and_gap():
    p0 = NestedDMPParameters()
    p = replace(p0, mu=E.calibrate_mu_to_f(p0, f_target=0.5))
    tg = E.load_empirical_irf_targets(_TABLES)
    rep = E.fit_report(p, tg, sigma0=1.0)
    # Sign-flip reproduced for urate in both regimes (the mechanism).
    assert rep.sign_match["urate"] is True
    # The model under-responds: peak magnitude ratio (emp/model) >> 1 (the gap).
    assert rep.magnitude_ratio["tight"]["urate"] > 2.0
    # Shape distance is finite and non-negative.
    assert rep.shape_distance >= 0.0 and np.isfinite(rep.shape_distance)


@pytest.mark.slow
def test_estimate_shape_match_improves_fit():
    p0 = NestedDMPParameters()
    p = replace(p0, mu=E.calibrate_mu_to_f(p0, f_target=0.5))
    tg = E.load_empirical_irf_targets(_TABLES)
    d0 = E.fit_report(p, tg, sigma0=1.0).shape_distance
    res = E.estimate_shape_match(p, tg, sigma0=1.0, maxiter=15)  # small maxiter keeps the test fast
    # The fitted params do not worsen the normalized-shape distance.
    assert res.shape_distance <= d0 + 1e-6
    # Estimated response params stay admissible.
    assert res.params.signal_sd > 0.0 and 0.0 <= res.params.rho_sigma < 1.0
    assert res.estimated == ("signal_sd", "rho_sigma")


def test_ablate_beliefs_kills_sign_flip():
    p0 = NestedDMPParameters()
    p = replace(p0, mu=E.calibrate_mu_to_f(p0, f_target=0.5))
    loose, tight = E.ablate_beliefs(p, sigma0=1.0, horizon=8)
    h = 4
    # With one common belief in both regimes, dove and hawk urate move the SAME
    # direction at sigma>0 (no sign-flip): the product is non-negative.
    assert loose["urate"][h] * tight["urate"][h] >= -1e-12


@pytest.mark.slow
def test_baseline_calibration_hits_flow_targets_and_shows_asymmetric_gap():
    p = E.baseline_calibration(NestedDMPParameters(), u_target=0.058, f_target=0.7)
    # Steady-state levels hit the US flow targets.
    grid, erg = D._fixed_grid(p, sigma_ref=1.0)
    ss = D._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
    phi = D._stationary_phi(p, theta=ss["theta"], J=ss["J"], sigma=0.0, grid=grid, erg=erg)
    u = 1.0 - phi.sum()
    f = p.mu * ss["theta"] ** (1.0 - p.alpha)
    assert u == pytest.approx(0.058, abs=3e-3)
    assert f == pytest.approx(0.70, abs=1e-3)
    # At the realistic calibration the gap is ASYMMETRIC: loose nearly closed,
    # tight (the H-side) still large -- exactly the amplification gap to close next.
    tg = E.load_empirical_irf_targets(_TABLES)
    rep = E.fit_report(p, tg, sigma0=1.0)
    assert rep.sign_match["urate"] is True
    assert rep.magnitude_ratio["loose"]["urate"] < 6.0          # loose nearly closed
    assert rep.magnitude_ratio["tight"]["urate"] > 15.0          # tight H-side gap remains


@pytest.mark.slow
def test_default_baseline_locks_calibration_and_documents_gap():
    p = E.default_baseline()
    # The lean belief model at the frozen defaults: flexible wage, rho_sigma=0.7.
    assert p.psi_down == 0.0
    assert p.rho_sigma == 0.7
    grid, erg = D._fixed_grid(p, sigma_ref=1.0)
    ss = D._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
    phi = D._stationary_phi(p, theta=ss["theta"], J=ss["J"], sigma=0.0, grid=grid, erg=erg)
    assert (1.0 - phi.sum()) == pytest.approx(0.058, abs=3e-3)
    # The regime sign-flip is reproduced CLEANLY in both urate and tightness (no
    # transients): dove urate falls / hawk rises; dove theta expands / hawk contracts.
    tg = E.load_empirical_irf_targets(_TABLES)
    rep = E.fit_report(p, tg, sigma0=1.0)
    assert rep.sign_match["urate"] is True
    assert rep.sign_match["theta"] is True
    # Documented gap: loose urate under-responds ~3.5x; tight urate ~37x is the
    # asymmetric, tight-side amplification frontier.
    assert rep.magnitude_ratio["loose"]["urate"] < 6.0
    assert rep.magnitude_ratio["tight"]["urate"] > 15.0
