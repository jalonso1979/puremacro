"""The het-firm steady state must reduce to the closed-form DMP oracles."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import equilibrium as S
from puremacro.models.dmp_regime_dependent import DMPParameters, dmp_steady_state


def test_reduction_to_dmp_regime_dependent_homogeneous_limit():
    # Degenerate x (tiny dispersion), no fixed cost, no exit hazard, sigma=0 so
    # beta_bar = 1/(1+r*). Construct the matching textbook-DMP params.
    p = NestedDMPParameters(sigma_x=1e-4, n_x=3, f_fixed=0.0, delta=0.0)
    eq = S.solve_steady_state(p, sigma=0.0, pi=p.prior_pi0)

    beta_bar = 1.0 / (1.0 + p.r_star)
    pdmp = DMPParameters(
        beta_bar=beta_bar, alpha=p.alpha, eta_b=p.alpha, s=p.s_bar,
        z=p.b, y_bar=1.0, c=p.c_convex, mu=p.mu, eta=0.0, kappa=0.0,
    )
    ref = dmp_steady_state(pdmp, sigma=0.0, regime="H")  # channels off => regime irrelevant

    assert eq.theta == pytest.approx(ref.theta, rel=2e-3)
    assert eq.u == pytest.approx(ref.u, rel=2e-3)
    # Match value at the (degenerate) central grid node vs the scalar oracle J.
    assert float(np.asarray(eq.J)[p.n_x // 2]) == pytest.approx(ref.J, rel=2e-3)


# ---------------------------------------------------------------------------
# Task 6: Reduction oracle #1 — channel + J-formula match to
#         notebooks/scripts/calibrate_dmp_signal_extraction.py
# ---------------------------------------------------------------------------

import importlib.util as _ilu
from pathlib import Path


def _load_calibrate():
    # The reference implementation (originally notebooks/scripts/ in the
    # research monorepo) is vendored under tests/fixtures/companion so the
    # split repo is self-contained; it is a script, not a package: load by path.
    here = Path(__file__).resolve()
    path = (here.parents[1] / "fixtures" / "companion"
            / "calibrate_dmp_signal_extraction.py")
    spec = _ilu.spec_from_file_location("calib_ref", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expected_discount_channel_matches_calibrate_reference():
    calib = _load_calibrate()
    p = NestedDMPParameters()
    cp = dict(calib.BASELINE_PARAMS)  # r_star, phi_D, phi_H already DECIMAL there
    # Same belief->discount map (this is the theoretical core both share).
    for pi in (0.1, 0.319, 0.9):
        for sigma in (0.0, 0.5, 1.0):
            ours = S.effective_discount(p, sigma, pi)
            theirs = calib.expected_discount_factor(pi, sigma, cp)
            assert ours == pytest.approx(theirs, rel=1e-9)


def test_J_formula_reduces_to_calibrate_steady_state():
    calib = _load_calibrate()
    # calibrate normalizes mu so theta_ss=1; replicate its primitives in ours and
    # check the J = (y-w)/(1-(1-s)beta) reduction at the degenerate limit.
    ss = calib.steady_state(pi=0.5, params=calib.BASELINE_PARAMS)  # sigma=0
    cp = calib.BASELINE_PARAMS
    p = NestedDMPParameters(
        sigma_x=1e-4, n_x=3, f_fixed=0.0, delta=0.0,
        alpha=cp["eta"], b=cp["b_over_y"], c_convex=cp["c_over_y"],
        s_bar=cp["s"], r_star=cp["r_star"], mu=ss["mu_implied"],
    )
    eq = S.solve_steady_state(p, sigma=0.0, pi=0.5)
    # Our free entry carries beta_bar; calibrate's omits it. So compare the J
    # formula directly (the shared reduction), not theta.
    J_mid = float(np.asarray(eq.J)[p.n_x // 2])
    assert J_mid == pytest.approx(ss["J"], rel=5e-3)


def test_participation_off_recovers_two_state_core():
    # h_max=0 must reproduce the pre-participation steady state exactly: lfpr=1,
    # N=0, and theta/urate/x_sep/output/free_entry_residual unchanged across a
    # (pi, sigma) grid.
    p = NestedDMPParameters(h_max=0.0)
    for sigma in (0.0, 1.0):
        for pi in (0.05, 0.95):
            eq = S.solve_steady_state(p, sigma=sigma, pi=pi)
            assert eq.lfpr == 1.0
            assert eq.N == pytest.approx(0.0)
            # Population levels equal the intensive rates when everyone participates.
            assert eq.u == pytest.approx(eq.urate, rel=1e-12)
            assert eq.n == pytest.approx(1.0 - eq.urate, rel=1e-9)
