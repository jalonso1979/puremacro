from __future__ import annotations

import numpy as np
import pytest

import puremacro.models.nested_dmp as ndmp
from puremacro.models.nested_dmp import NestedDMPParameters


def test_public_surface_imports():
    p = ndmp.NestedDMPParameters()
    assert isinstance(p.alpha, float)
    # kernels reachable from the package root
    grid, P = ndmp.rouwenhorst(p.n_x, p.rho_x, p.sigma_x)
    assert grid.shape == (p.n_x,)
    flow = np.full(p.n_x, 0.5)
    V = ndmp.bellman_iterate(flow, P, p.beta, tol=1e-12, max_iter=10_000)
    assert V.shape == (p.n_x,)
    assert np.all(V >= 0.0)  # separation option floors V at 0


def test_available_backends_reported():
    assert "numpy" in ndmp.available_backends()


def test_steady_state_public_surface():
    from puremacro.models.nested_dmp import (
        SteadyState, solve_steady_state, comparative_statics, ComparativeStatics,
    )
    p = NestedDMPParameters()
    eq = solve_steady_state(p, sigma=0.0)
    assert isinstance(eq, SteadyState)
    assert 0.0 < eq.urate < 1.0 and eq.output > 0.0
    cs = comparative_statics(p, pi_grid=[0.1, 0.9], sigma_grid=[0.0, 1.0])
    assert isinstance(cs, ComparativeStatics)


def test_participation_public_surface():
    from puremacro.models.nested_dmp import worker_values, participation_rate, solve_steady_state
    p = NestedDMPParameters(h_max=3.0)
    eq = solve_steady_state(p, sigma=1.0, pi=0.9)
    assert 0.0 < eq.lfpr <= 1.0 and eq.N == pytest.approx(1.0 - eq.lfpr)
    assert participation_rate(p, eq.W_U) == pytest.approx(eq.lfpr)


def test_dynamics_public_surface():
    from dataclasses import replace
    from puremacro.models.nested_dmp import (
        simulate_irf, IRFResult, calibrate_mu, tauchen_transition,
    )
    p0 = NestedDMPParameters(h_max=3.0)
    p = replace(p0, mu=calibrate_mu(p0, sigma_ref=1.0))  # f<1 so the IRF is stable
    irf = simulate_irf(p, sigma0=1.0, fed_type="hawk", horizon=6)
    assert isinstance(irf, IRFResult)
    assert irf.log_theta.shape == (7,)


def test_estimation_public_surface():
    from pathlib import Path
    from dataclasses import replace
    from puremacro.models.nested_dmp import (
        load_empirical_irf_targets, fit_report, calibrate_mu_to_f, FitReport,
    )
    # Frozen empirical target tables vendored with the tests (see
    # test_companion_estimation.py for provenance).
    tables = Path(__file__).resolve().parents[1] / "fixtures" / "companion"
    tg = load_empirical_irf_targets(tables)
    p0 = NestedDMPParameters()
    p = replace(p0, mu=calibrate_mu_to_f(p0, f_target=0.5))
    rep = fit_report(p, tg, sigma0=1.0)
    assert isinstance(rep, FitReport)
    assert rep.sign_match["urate"] is True               # mechanism reproduced
    assert rep.magnitude_ratio["tight"]["urate"] > 1.0    # the H-side gap documented
