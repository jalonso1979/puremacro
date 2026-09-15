"""Adversarial Stress Test Suite for Milestone 5 Deliverables (Challenger 1).

Rigorous empirical challenge of:
1. Implicit Upwind HJB Solver & Adjoint KFE (Achdou et al. 2022)
   - Boundary conditions (r == rho, r > rho, r < 0, extreme gamma)
   - Non-uniform / geometric grid stability
   - Multi-state Markov productivity jump generators (Ne >= 3)
   - Adjoint KFE unit mass conservation to 10^-12 and pointwise non-negativity
   - HJBSolution 12-key backward-compatibility mapping protocol
   - Headless presentation rendering (.summary, .to_frame, .to_latex, .to_typst, .to_markdown, .plot)
2. Multi-Constraint OccBin ($M \\ge 2$, Guerrieri & Iacoviello 2015)
   - Degenerate zero-shock paths
   - Input validation guards (horizon, max_iter, conflicting equation rows)
   - Simultaneous multi-constraint binding across 2^K regimes
   - Terminal slack violation detection (converged == False)
3. Double / Debiased Machine Learning (DML PLR, Chernozhukov et al. 2018)
   - High-dimensional controls (p near N)
   - Collinear and constant features
   - Multi-dimensional treatment vectors (D shape (N, d))
   - Neyman orthogonality & asymptotic confidence interval coverage
   - Presentation exports (.to_latex, .to_typst, .to_markdown, .plot)
4. Montiel Olea & Pflueger (2013) Weak IV & Anderson-Rubin Confidence Sets
   - Severe weak instrument identification failure (F_eff < CV_20)
   - AR confidence set inversion into unbounded rays / all-real
   - Strong instrument recovery & bounded AR sets
   - Multi-instrument grid inversion tests
5. Subprocess-isolated Pyodide dependency leak audit for M3/M5 modules.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from puremacro.causal.dml import (
    DMLResult,
    DoubleMLPLR,
    LassoCoordinateDescent,
    RidgeGCV,
    dml_plr,
)
from puremacro.dsge import (
    LinearModel,
    OccBinConstraint,
    OccBinResult,
    build_dynare,
    solve_multiconstraint_occbin,
    solve_occbin,
)
from puremacro.lp.iv import (
    compute_mop_effective_f,
    lp_iv,
    mop_critical_values,
)
from puremacro.lp.la_lp import la_lp_iv
from puremacro.vfi.hjb_achdou import (
    AiyagariContinuousHJBResult,
    HJBSolution,
    _quadrature_weights,
    _stationary_markov_distribution,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)


# ===========================================================================
# 1. Implicit Upwind HJB & Adjoint Continuous KFE Stress Tests
# ===========================================================================

_DEFAULT_A_Z = np.array([[-0.1, 0.1], [0.1, -0.1]])


def _assert_hjb_kfe_invariants(sol, A_z) -> None:
    """Quantitative checks a converged HJB/KFE solution cannot satisfy by construction.

    mass_residual and g >= 0 are enforced by renormalisation and clipping inside
    solve_kfe_achdou, so they hold for any g. These do not: the node mass pi = g * w
    must be a null vector of A^T (the KFE itself), its marginal over e must equal the
    stationary distribution of A_z, consumption must increase in wealth whenever income
    does (r >= 0; with r < 0 the zero-drift branch c = r a + w e itself falls in a), and
    the state constraints s(a_min) >= 0 >= s(a_max) must hold.
    """
    g = sol.g_dist
    assert g is not None and sol.A_generator is not None
    w = _quadrature_weights(sol.a_grid)
    pi = (g * w[:, None]).ravel(order="F")
    assert abs(np.sum(pi) - 1.0) <= 1e-12
    assert np.max(np.abs(sol.A_generator.T @ pi)) < 1e-10
    np.testing.assert_allclose(
        np.sum(g * w[:, None], axis=0), _stationary_markov_distribution(np.asarray(A_z, dtype=float)), atol=1e-10
    )
    if sol.r_rate >= 0.0:
        assert np.all(np.diff(sol.c_policy, axis=0) > 0.0)
    assert np.all(sol.s_drift[0, :] >= -1e-12)
    assert np.all(sol.s_drift[-1, :] <= 1e-12)


def test_hjb_extreme_parameter_boundaries():
    """Verify implicit HJB solver converges under extreme and boundary parameter configurations."""
    # Boundary 1: r == rho (zero steady-state asset drift tendency)
    sol_eq = solve_hjb_achdou(r_rate=0.05, rho_val=0.05, Na=40, max_iter=30)
    assert sol_eq.converged, "HJB failed to converge when r == rho"
    assert sol_eq.n_iter <= 20
    assert sol_eq.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_eq, _DEFAULT_A_Z)

    # Boundary 2: r > rho (impatience dominated by return, strong savings drift)
    sol_high_r = solve_hjb_achdou(r_rate=0.06, rho_val=0.04, Na=40, max_iter=40)
    assert sol_high_r.converged, "HJB failed to converge when r > rho"
    assert sol_high_r.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_high_r, _DEFAULT_A_Z)

    # Boundary 3: r < 0 (negative real interest rate)
    sol_neg_r = solve_hjb_achdou(r_rate=-0.02, rho_val=0.05, Na=40, max_iter=40)
    assert sol_neg_r.converged, "HJB failed to converge when r < 0"
    assert sol_neg_r.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_neg_r, _DEFAULT_A_Z)

    # Boundary 4: Extreme relative risk aversion gamma = 10.0
    sol_crra_10 = solve_hjb_achdou(gamma_r=10.0, Na=40, max_iter=40)
    assert sol_crra_10.converged, "HJB failed to converge with gamma = 10.0"
    assert np.all(np.diff(sol_crra_10.V[:, 0]) > 0)
    assert np.all(np.diff(sol_crra_10.V[:, 1]) > 0)
    _assert_hjb_kfe_invariants(sol_crra_10, _DEFAULT_A_Z)

    # Boundary 5: Low risk aversion gamma = 0.5
    sol_crra_05 = solve_hjb_achdou(gamma_r=0.5, Na=40, max_iter=40)
    assert sol_crra_05.converged, "HJB failed to converge with gamma = 0.5"
    assert sol_crra_05.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_crra_05, _DEFAULT_A_Z)


def test_hjb_coarse_and_nonuniform_grids():
    """Verify HJB and KFE stability on small and geometrically non-uniform grids."""
    # Coarse grid Na = 6
    sol_coarse = solve_hjb_achdou(Na=6, max_iter=30)
    assert sol_coarse.converged
    assert sol_coarse.V.shape == (6, 2)
    assert sol_coarse.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_coarse, _DEFAULT_A_Z)

    # Non-uniform geometrically spaced asset grid (concentrated near borrowing limit)
    a_grid_geom = np.geomspace(0.01, 40.0, 45) - 0.01
    sol_geom = solve_hjb_achdou(a_grid=a_grid_geom, Na=45, max_iter=40)
    assert sol_geom.converged
    assert sol_geom.mass_residual <= 1e-12
    assert sol_geom.g_dist is not None
    assert np.all(sol_geom.g_dist >= 0.0)
    _assert_hjb_kfe_invariants(sol_geom, _DEFAULT_A_Z)


def test_hjb_multistate_markov_generator():
    """Verify continuous HJB and adjoint KFE solver on 4-state Markov income chain."""
    e_grid_4 = np.array([0.1, 0.4, 1.0, 2.5])
    # 4-state transition rate matrix with zero row sums
    A_z_4 = np.array([
        [-0.3,  0.3,  0.0,  0.0],
        [ 0.1, -0.3,  0.2,  0.0],
        [ 0.0,  0.15, -0.3, 0.15],
        [ 0.0,  0.0,  0.3, -0.3],
    ])
    sol_4 = solve_hjb_achdou(e_grid=e_grid_4, A_z=A_z_4, Na=40, max_iter=40)
    assert sol_4.converged
    assert sol_4.V.shape == (40, 4)
    assert sol_4.g_dist.shape == (40, 4)
    assert np.all(sol_4.g_dist >= 0.0)
    assert sol_4.mass_residual <= 1e-12
    _assert_hjb_kfe_invariants(sol_4, A_z_4)

    # Verify monotonicity in productivity: V(a, e_k) < V(a, e_{k+1})
    for k in range(3):
        assert np.all(sol_4.V[:, k + 1] > sol_4.V[:, k])


def test_hjb_solution_strict_12_key_mapping_and_backward_compatibility():
    """Verify HJBSolution satisfies the 12-key dict mapping protocol and backward compatibility."""
    sol = solve_hjb_achdou(Na=25, max_iter=25)

    canonical_12_keys = [
        "V", "c_policy", "s_drift", "a_grid", "e_grid", "n_iter",
        "elapsed", "r_rate", "w_rate", "rho_val", "gamma_r", "converged"
    ]

    # 1. sol.keys() must contain strictly the 12 canonical keys
    assert list(sol.keys()) == canonical_12_keys
    assert len(sol.keys()) == 12
    assert len(sol) == 12

    # 2. Conversion to dict produces exact 12-key mapping
    d = dict(sol)
    assert len(d) == 12
    assert list(d.keys()) == canonical_12_keys

    # 3. Kwargs unpacking (**sol) into legacy function expecting 12 parameters
    def legacy_consumer(V, c_policy, s_drift, a_grid, e_grid, n_iter, elapsed, r_rate, w_rate, rho_val, gamma_r, converged):
        return V.shape[0]

    assert legacy_consumer(**sol) == 25

    # 4. Attribute and mapping access to new continuous KFE properties
    assert sol.g_dist is not None
    assert sol.A_generator is not None
    assert isinstance(sol.mass_residual, float)

    # Subscript indexing protocol
    assert sol["g_dist"] is sol.g_dist
    assert sol["A_generator"] is sol.A_generator
    assert sol["mass_residual"] == sol.mass_residual

    # Membership protocol
    assert "g_dist" in sol
    assert "A_generator" in sol
    assert "mass_residual" in sol
    assert "non_existent_field" not in sol

    # sol.get() protocol
    assert sol.get("g_dist") is sol.g_dist
    assert sol.get("non_existent_field", "default_val") == "default_val"


def test_hjb_presentation_methods_coverage():
    """Verify all presentation export methods execute without errors in headless mode."""
    sol = solve_hjb_achdou(Na=20, max_iter=20)

    # DataFrame conversions
    summ = sol.summary()
    assert isinstance(summ, pd.DataFrame)
    assert not summ.empty

    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20 * 2
    assert "density_g" in df.columns

    # Formatted exports
    md = sol.to_markdown()
    assert isinstance(md, str) and len(md) > 0

    ltx = sol.to_latex()
    assert isinstance(ltx, str) and "\\begin{tabular}" in ltx

    typ = sol.to_typst()
    assert isinstance(typ, str) and "#table(" in typ

    # Headless plot
    fig = sol.plot(show=False)
    assert fig is not None


# ===========================================================================
# 2. Multi-Constraint OccBin Stress Tests
# ===========================================================================


@pytest.fixture
def nk_model_setup():
    """Setup canonical 5-variable New Keynesian model for OccBin multi-constraint tests."""
    params = {
        "beta": 0.99, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "phi_y": 0.25,
        "rho_r": 0.6, "rho_b": 0.5, "rho_g": 0.7, "gamma_y": 0.2, "chi": 0.1,
        "r_ss": 0.015, "b_bar": 0.02,
    }
    variables = ["y", "pi", "r", "b", "g"]
    shocks = ["eps_g", "eps_r", "eps_b"]

    def ref_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    def zlb_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    def borr_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    steady_state = {v: 0.0 for v in variables}
    m_ref = build_dynare(ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
    m_zlb = build_dynare(zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
    m_borr = build_dynare(borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)

    c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

    return m_ref, m_zlb, m_borr, c_zlb, c_borr, params


def test_occbin_zero_shocks_and_degenerate_cases(nk_model_setup):
    """Verify solve_multiconstraint_occbin handles zero shocks seamlessly."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr, _ = nk_model_setup

    res_zero = solve_multiconstraint_occbin(
        m_unconstrained=m_ref,
        m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
        shock_seq=np.zeros(3),
        constraints={"zlb": c_zlb, "borrowing": c_borr},
        horizon=20,
    )
    assert res_zero.converged
    assert res_zero.iterations == 1
    assert np.allclose(res_zero.simulated_path.values, 0.0, atol=1e-12)
    assert all(r == 0 for r in res_zero.regimes)


def test_occbin_input_validation_stress(nk_model_setup):
    """Verify defensive input validation in solve_multiconstraint_occbin."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr, _ = nk_model_setup

    # Negative / non-positive horizon
    with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
        solve_multiconstraint_occbin(m_ref, {"zlb": m_zlb}, np.zeros(3), horizon=0)

    # Negative / non-positive max_iter
    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        solve_multiconstraint_occbin(m_ref, {"zlb": m_zlb}, np.zeros(3), max_iter=0)

    # Empty constrained models dictionary
    with pytest.raises(ValueError, match="requires at least one constrained model"):
        solve_multiconstraint_occbin(m_ref, {}, np.zeros(3))

    # Multiple constraints replacing the exact same equation row (incompatible)
    with pytest.raises(ValueError, match="Incompatible constraint regimes"):
        solve_multiconstraint_occbin(
            m_ref,
            {"zlb1": m_zlb, "zlb2": m_zlb},
            np.zeros(3),
            constraints={"zlb1": c_zlb, "zlb2": c_zlb},
        )


def test_occbin_terminal_violation_and_convergence_guard(nk_model_setup):
    """Verify that a constraint binding at the terminal horizon flags converged=False and warns."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr, _ = nk_model_setup

    # Massive permanent shock with short horizon T=5
    shock_seq = np.zeros((5, 3))
    shock_seq[:, 0] = -0.20  # Extreme persistent deflationary shock

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
            shock_seq=shock_seq,
            constraints={"zlb": c_zlb, "borrowing": c_borr},
            horizon=5,
        )
        assert not res.converged, "Should report converged=False when constraint binds at terminal period"
        assert any("constraint still binds at terminal period" in str(w.message) for w in caught_warnings)


# ===========================================================================
# 3. Double / Debiased Machine Learning (DML PLR) Stress Tests
# ===========================================================================


def test_dml_collinear_and_singular_controls():
    """Verify DML PLR behaves stably when control matrix X has duplicate/collinear columns."""
    rng = np.random.default_rng(2026)
    N = 150
    X_base = rng.normal(size=(N, 10))
    # Add exact duplicate column and constant column
    X_singular = np.column_stack([X_base, X_base[:, 0], np.ones(N)])
    theta_true = 1.75
    D = 0.6 * X_base[:, 0] - 0.4 * X_base[:, 1] + rng.normal(scale=0.5, size=N)
    Y = theta_true * D + 1.1 * X_base[:, 0] + rng.normal(scale=0.5, size=N)

    # Ridge GCV with singular controls
    res_ridge = dml_plr(Y, D, X_singular, learner="ridge", n_folds=3, random_state=42)
    assert np.isfinite(res_ridge.theta)
    assert abs(res_ridge.theta - theta_true) < 3.0 * res_ridge.se

    # Lasso with singular controls
    res_lasso = dml_plr(Y, D, X_singular, learner="lasso", n_folds=3, random_state=42)
    assert np.isfinite(res_lasso.theta)
    assert abs(res_lasso.theta - theta_true) < 3.0 * res_lasso.se


def test_dml_high_dimensional_recovery():
    """Verify DML PLR recovers treatment effect in high dimensions (p = 60, N = 120)."""
    rng = np.random.default_rng(888)
    N, p = 120, 60
    X = rng.normal(size=(N, p))
    theta_true = 3.2
    # Sparse nuisance components
    D = 0.8 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.5, size=N)
    Y = theta_true * D + 1.5 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(scale=0.5, size=N)

    res = dml_plr(Y, D, X, learner="lasso", n_folds=4, random_state=42)
    assert np.isfinite(res.theta)
    assert abs(res.theta - theta_true) < 3.0 * res.se
    assert res.ci_lower < theta_true < res.ci_upper


def test_dml_multidimensional_treatment_stress():
    """Verify DML PLR estimates a multi-dimensional treatment vector D of shape (N, 3)."""
    rng = np.random.default_rng(101)
    N, p = 250, 20
    X = rng.normal(size=(N, p))

    D1 = 0.5 * X[:, 0] + rng.normal(size=N)
    D2 = -0.4 * X[:, 1] + rng.normal(size=N)
    D3 = 0.3 * X[:, 2] + rng.normal(size=N)
    D_mat = np.column_stack([D1, D2, D3])

    thetas_true = np.array([2.0, -1.0, 0.5])
    Y = D_mat @ thetas_true + 1.2 * X[:, 0] - 0.9 * X[:, 1] + rng.normal(size=N)

    res = dml_plr(Y, D_mat, X, learner="ridge", n_folds=3, random_state=42)
    assert len(res.theta) == 3
    assert len(res.se) == 3
    assert np.all(np.abs(res.theta - thetas_true) < 3.0 * res.se)

    # Check presentation methods for multidimensional result
    ltx = res.to_latex()
    assert "\\toprule" in ltx
    typ = res.to_typst()
    assert "#figure(" in typ
    md = res.to_markdown()
    assert "| D_1 |" in md
    assert "| D_2 |" in md
    assert "| D_3 |" in md


# ===========================================================================
# 4. Weak IV (MOP Effective F & Anderson-Rubin) Stress Tests
# ===========================================================================


def test_weak_iv_identification_failure_and_ar_inversion():
    """Verify Montiel Olea & Pflueger effective F and Anderson-Rubin set inversion on noise IV."""
    rng = np.random.default_rng(777)
    T = 250
    # Pure noise instrument completely uninformative about x
    z_noise = rng.normal(size=T)
    v = rng.normal(size=T)
    x = v  # zero relevance to z_noise
    y = np.cumsum(2.0 * x + rng.normal(size=T))
    w = rng.normal(size=T)

    df = pd.DataFrame({"y": y, "x": x, "z": z_noise, "w": w})
    res = lp_iv(df, y="y", x="x", z="z", controls=["w"], horizons=[1, 2], anderson_rubin=True)

    # MOP effective F should fall far below the 20% bias critical value (6.70)
    for h in [1, 2]:
        mop_f = float(res.loc[h, "mop_f"])
        cv_20 = float(res.loc[h, "mop_cv_20"])
        assert mop_f < cv_20, f"Expected weak IV failure (mop_f={mop_f} < cv_20={cv_20})"
        set_type = res.loc[h, "ar_set_type"]
        assert set_type in ("unbounded_rays", "all_real", "empty"), (
            f"Expected weak identification set type, got {set_type}"
        )


def test_strong_iv_recovery_and_bounded_ar_sets():
    """Verify LP-IV produces bounded Anderson-Rubin confidence sets under strong instrument."""
    rng = np.random.default_rng(555)
    T = 200
    z_strong = rng.normal(size=T)
    v = rng.normal(size=T)
    x = 3.0 * z_strong + v  # strong first stage
    y = np.cumsum(1.8 * x + rng.normal(size=T))
    w = rng.normal(size=T)

    df = pd.DataFrame({"y": y, "x": x, "z": z_strong, "w": w})
    res = lp_iv(df, y="y", x="x", z="z", controls=["w"], horizons=[1, 2], anderson_rubin=True)

    for h in [1, 2]:
        mop_f = float(res.loc[h, "mop_f"])
        cv_10 = float(res.loc[h, "mop_cv_10"])
        assert mop_f > cv_10, f"Expected strong IV (mop_f={mop_f} > cv_10={cv_10})"
        assert res.loc[h, "ar_set_type"] == "bounded"
        ar_lo = float(res.loc[h, "ar_lo"])
        ar_hi = float(res.loc[h, "ar_hi"])
        assert np.isfinite(ar_lo) and np.isfinite(ar_hi)
        assert ar_lo < ar_hi


def test_multi_instrument_mop_and_ar_grid():
    """Verify multi-instrument weak IV diagnostics with k_z = 3 instruments and surface grid boundary bug."""
    rng = np.random.default_rng(432)
    T = 200
    Z = rng.normal(size=(T, 3))
    v = rng.normal(size=T)
    # Weak linear combination
    x = 0.05 * Z[:, 0] - 0.04 * Z[:, 1] + v
    y = np.cumsum(1.2 * x + rng.normal(size=T))
    w = rng.normal(size=T)

    df = pd.DataFrame({"y": y, "x": x, "z1": Z[:, 0], "z2": Z[:, 1], "z3": Z[:, 2], "w": w})
    res = lp_iv(
        df, y="y", x="x", z=["z1", "z2", "z3"], controls=["w"], horizons=[1], anderson_rubin=True
    )

    mop_f = float(res.loc[1, "mop_f"])
    cv_20 = float(res.loc[1, "mop_cv_20"])
    assert mop_f < cv_20, f"mop_f={mop_f} >= cv_20={cv_20}"

    # Verify proper unbounded rays detection and ordering for weak multi-instrument AR
    ar_lo = float(res.loc[1, "ar_lo"])
    ar_hi = float(res.loc[1, "ar_hi"])
    assert np.isfinite(ar_lo) and np.isfinite(ar_hi)
    assert res.loc[1, "ar_set_type"] == "unbounded_rays"
    assert ar_lo > ar_hi

    # Also verify la_lp White-robust multi-instrument AR set
    res_la = la_lp_iv(
        df, y="y", x="x", z=["z1", "z2", "z3"], controls=["w"], horizons=[1], anderson_rubin=True
    )
    assert res_la.loc[1, "ar_set_type"] == "unbounded_rays"
    assert float(res_la.loc[1, "ar_lo"]) > float(res_la.loc[1, "ar_hi"])



# ===========================================================================
# 5. Clean-Subprocess Pyodide Purity Audit
# ===========================================================================


def test_m5_modules_subprocess_zero_forbidden_imports():
    """Confirm in a clean isolated subprocess that importing M3/M5 modules triggers ZERO forbidden deps."""
    code = """
import sys

target_modules = [
    "puremacro.vfi.hjb_achdou",
    "puremacro.causal.dml",
    "puremacro.dsge.occbin",
    "puremacro.lp.iv",
    "puremacro.lp.la_lp",
]
forbidden = ["sklearn", "torch", "statsmodels", "arch"]

for mod in target_modules:
    __import__(mod)

leaked = [f for f in forbidden if any(m == f or m.startswith(f + '.') for m in sys.modules)]
if leaked:
    print(f"LEAKED: {leaked}")
    sys.exit(1)
sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess import leak detected: {result.stdout} {result.stderr}"


def test_vfi_import_does_not_load_matplotlib():
    """puremacro.vfi imports matplotlib lazily (inside .plot methods only), so a clean
    interpreter that imports the subpackage never loads the plotting stack."""
    code = """
import sys
import puremacro.vfi
leaked = sorted(m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib."))
if leaked:
    print(f"LEAKED: {leaked[:5]}")
    sys.exit(1)
sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"matplotlib loaded eagerly: {result.stdout} {result.stderr}"
