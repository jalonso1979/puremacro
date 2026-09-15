"""Tests for canonical implicit upwind HJB solver, adjoint KFE, and continuous Aiyagari GE."""
from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
import numpy as np
import pytest

from puremacro.vfi import (
    AiyagariContinuousHJBResult,
    HJBSolution,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)
from puremacro.vfi.hjb_achdou import _quadrature_weights, _stationary_markov_distribution


def test_hjb_implicit_convergence_and_speed():
    """Verify implicit upwind HJB solver converges in <= 20 iterations to tol 1e-8."""
    sol = solve_hjb_achdou(
        r_rate=0.03,
        w_rate=1.0,
        rho_val=0.05,
        gamma_r=2.0,
        Na=50,
        tol=1e-8,
        max_iter=100,
    )
    assert sol.converged
    assert sol.n_iter <= 20
    assert sol.V.shape == (50, 2)
    assert sol.c_policy.shape == (50, 2)
    assert sol.s_drift.shape == (50, 2)
    # Borrowing constraint: at a=0, drift cannot be negative into the infeasible region
    assert np.all(sol.s_drift[0, :] >= -1e-12)
    # Monotonicity of value function in wealth: V(a) is strictly increasing
    assert np.all(np.diff(sol.V[:, 0]) > 0)
    assert np.all(np.diff(sol.V[:, 1]) > 0)
    # Higher productivity yields higher value
    assert np.all(sol.V[:, 1] > sol.V[:, 0])


def test_hjb_solution_mapping_protocol():
    """Verify HJBSolution satisfies collections.abc.Mapping contract."""
    sol = solve_hjb_achdou(Na=30, max_iter=30)
    assert isinstance(sol, Mapping)
    assert len(sol) == len(sol.keys())
    assert "V" in sol
    assert "c_policy" in sol
    assert "s_drift" in sol
    assert "n_iter" in sol
    assert "g_dist" in sol
    assert "mass_residual" in sol

    # Subscripting
    assert sol["n_iter"] == sol.n_iter
    np.testing.assert_array_equal(sol["V"], sol.V)

    # Conversion to dict
    d = dict(sol)
    assert isinstance(d, dict)
    assert d["converged"] is True

    # Iteration
    keys_list = list(sol)
    assert "V" in keys_list
    assert len(keys_list) == len(sol)


def test_hjb_presentation_methods():
    """Verify HJBSolution presentation methods: summary, to_frame, to_markdown, to_latex, to_typst, plot."""
    sol = solve_hjb_achdou(Na=30, max_iter=30)

    summary_df = sol.summary()
    assert not summary_df.empty
    assert "Asset Grid Points (Na)" in summary_df.index

    frame_df = sol.to_frame()
    assert len(frame_df) == 30 * 2
    assert "asset_a" in frame_df.columns
    assert "consumption_c" in frame_df.columns

    md_str = sol.to_markdown()
    assert "Asset Grid Points (Na)" in md_str

    latex_str = sol.to_latex()
    assert "tabular" in latex_str

    typst_str = sol.to_typst()
    assert "table(" in typst_str

    # Headless plot
    fig = sol.plot(show=False)
    assert fig is not None


def test_kfe_strict_mass_conservation():
    """Verify adjoint continuous-time KFE achieves |sum g_i Delta a_i - 1.0| <= 10^-12 and g >= 0."""
    sol = solve_hjb_achdou(Na=80, tol=1e-8)
    assert sol.g_dist is not None
    assert sol.A_generator is not None

    g = sol.g_dist
    assert g.shape == (80, 2)
    assert np.all(g >= 0.0)

    da = sol.a_grid[1] - sol.a_grid[0]
    total_mass = float(np.sum(g * da))
    assert abs(total_mass - 1.0) <= 1e-12
    assert sol.mass_residual <= 1e-12

    # The stationarity condition itself, not the post-solve normalisation: the node mass
    # pi = g * da is a null vector of A^T, and its marginal over e is the stationary
    # distribution of the default 2-state generator
    pi = (g * da).ravel(order="F")
    assert np.max(np.abs(sol.A_generator.T @ pi)) < 1e-12
    p_z = _stationary_markov_distribution(np.array([[-0.1, 0.1], [0.1, -0.1]]))
    np.testing.assert_allclose(np.sum(g * da, axis=0), p_z, atol=1e-12)

    # Direct call to solve_kfe_achdou
    g_direct, mass_res = solve_kfe_achdou(
        sol.A_generator, sol.a_grid, sol.e_grid, return_residual=True
    )
    assert mass_res <= 1e-12
    np.testing.assert_allclose(g_direct, g, atol=1e-14)


def test_analytical_crra_cake_eating_benchmark():
    """Verify numerical HJB solver converges in < 25 iter and matches analytical CRRA benchmark."""
    # Analytical cake eating / unconstrained consumption benchmark:
    # Under rho=0.05, gamma=2.0, r=0, w=0, exact consumption is c(a) = mu * a where mu = rho / gamma.
    # Value function is V(a) = (mu^(-gamma) / (1 - gamma)) * a^(1 - gamma).
    rho = 0.05
    gamma = 2.0
    mu = rho / gamma
    a_min, a_max = 1.0, 5.0
    Na = 500

    # 1. Execute solve_hjb_achdou directly on cake-eating calibration (r=0, w=0)
    sol = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
    )
    assert sol.converged, "HJB solver failed to converge on cake-eating benchmark"
    assert sol.n_iter < 25, f"HJB solver required {sol.n_iter} iterations (expected < 25)"

    c_exact = mu * sol.a_grid
    K = (mu ** (-gamma)) / (1.0 - gamma)
    V_exact = K * (sol.a_grid ** (1.0 - gamma))

    c_num = sol.c_policy[:, 0]
    V_num = sol.V[:, 0]

    # Verify numerical consumption policy matches analytical closed-form solution
    max_abs_c_err = float(np.max(np.abs(c_num - c_exact)))
    max_rel_c_err = float(np.max(np.abs(c_num - c_exact) / c_exact))
    assert max_abs_c_err < 1e-3, f"Max absolute consumption error {max_abs_c_err} exceeds 1e-3"
    assert max_rel_c_err < 0.01, f"Max relative consumption error {max_rel_c_err} exceeds 1%"

    # Verify interior relative error (Reviewer 2 check) is well below 1%
    rel_err_interior = float(np.max(np.abs(c_num[20:180] - c_exact[20:180]) / c_exact[20:180]))
    assert rel_err_interior < 0.01, f"Interior relative error {rel_err_interior} exceeds 1%"

    # Verify numerical value function matches analytical closed-form value function
    max_rel_V_err = float(np.max(np.abs(V_num - V_exact) / np.abs(V_exact)))
    assert max_rel_V_err < 0.01, f"Max relative value function error {max_rel_V_err} exceeds 1%"

    # 2. v_prime_boundary is live: the exact derivatives reproduce the default policy,
    #    while a perturbed derivative moves it (a higher marginal utility at a_min
    #    lowers consumption at the constraint and propagates along the backward upwind)
    v_bwd_exact = float((mu * a_min) ** (-gamma))
    v_fwd_exact = float((mu * a_max) ** (-gamma))
    sol_custom = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
        v_prime_boundary=(v_bwd_exact, v_fwd_exact),
    )
    assert sol_custom.converged
    assert sol_custom.n_iter < 25
    np.testing.assert_allclose(sol_custom.c_policy, sol.c_policy, atol=1e-12)
    sol_perturbed = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
        v_prime_boundary=(2.0 * v_bwd_exact, v_fwd_exact),
    )
    assert sol_perturbed.converged
    assert sol_perturbed.c_policy[0, 0] < sol.c_policy[0, 0] - 1e-3
    assert float(np.max(np.abs(sol_perturbed.c_policy - sol.c_policy))) > 1e-3

    # 3. Verify analytical unconstrained CRRA with positive interest rate r=0.02
    r_pos = 0.02
    mu_pos = (rho - (1.0 - gamma) * r_pos) / gamma
    sol_r = solve_hjb_achdou(
        r_rate=r_pos,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
    )
    assert sol_r.converged
    assert sol_r.n_iter < 25
    c_exact_r = mu_pos * sol_r.a_grid
    rel_c_r = float(np.max(np.abs(sol_r.c_policy[:, 0] - c_exact_r) / c_exact_r))
    assert rel_c_r < 0.01, f"Positive-r unconstrained relative error {rel_c_r} exceeds 1%"


def test_continuous_aiyagari_general_equilibrium():
    """Verify solve_aiyagari_continuous_hjb converges to market clearing |Ks - Kd| < 10^-4."""
    res = solve_aiyagari_continuous_hjb(
        alpha=0.33,
        delta=0.05,
        rho_val=0.05,
        gamma_r=2.0,
        Na=40,
        a_max=25.0,
        tol_ge=1e-4,
        max_iter_ge=30,
        max_iter_hjb=50,
        tol_hjb=1e-7,
    )
    assert isinstance(res, Mapping)
    assert res.converged
    assert abs(res.excess_capital) < 1e-4
    assert abs(res.Ks_star - res.Kd_star) < 1e-4
    assert res.r_star > 0.005
    assert res.r_star < 0.05
    assert res.w_star > 0
    assert res.K_star > 0
    assert res.L_star > 0

    # Test dictionary indexing
    assert res["r_star"] == res.r_star
    assert res["K_star"] == res.K_star

    # Presentation methods
    assert not res.summary().empty
    assert "tabular" in res.to_latex()
    assert "table(" in res.to_typst()
    assert "Equilibrium Interest Rate" in res.to_markdown()


def test_custom_markov_jump_generator():
    """Verify solve_hjb_achdou supports custom income grids and generator matrix A_z."""
    e_grid = np.array([0.5, 1.0, 1.5])
    Ne = len(e_grid)
    # 3-state Poisson jump generator with zero row sums
    A_z = np.array([
        [-0.2, 0.2, 0.0],
        [0.1, -0.2, 0.1],
        [0.0, 0.2, -0.2],
    ])
    sol = solve_hjb_achdou(
        Na=40,
        e_grid=e_grid,
        A_z=A_z,
        max_iter=50,
        tol=1e-7,
    )
    assert sol.converged
    assert sol.V.shape == (40, 3)
    assert sol.c_policy.shape == (40, 3)
    assert sol.g_dist is not None
    assert sol.g_dist.shape == (40, 3)
    assert sol.mass_residual <= 1e-12


# ---------------------------------------------------------------------------
# 3.4.0 pre-release review regression tests
# ---------------------------------------------------------------------------


def test_kfe_returns_density_on_nonuniform_grid():
    """solve_kfe_achdou returns a density (node mass / cell width): on a non-uniform grid the
    weighted marginal equals the generator's stationary distribution, the node mass solves
    A^T pi = 0, and the capital-supply integral converges to a fine uniform-grid reference
    (regression for the mass-as-density bug that inflated K^s by ~30% on power grids)."""
    p_z = np.array([0.5, 0.5])
    a_ref = np.linspace(0.0, 30.0, 2000)
    w_ref = _quadrature_weights(a_ref)
    np.testing.assert_allclose(w_ref, a_ref[1] - a_ref[0], rtol=1e-12)  # uniform: w == da
    sol_ref = solve_hjb_achdou(a_grid=a_ref)
    Ks_ref = float(np.sum(a_ref[:, None] * sol_ref.g_dist * w_ref[:, None]))

    Ks_err = {}
    for Na in (100, 400):
        a_nu = 30.0 * np.linspace(0.0, 1.0, Na) ** 2
        sol = solve_hjb_achdou(a_grid=a_nu)
        assert sol.converged
        w = _quadrature_weights(a_nu)
        g = sol.g_dist
        assert np.all(g >= 0.0)
        assert sol.mass_residual <= 1e-12
        np.testing.assert_allclose(np.sum(g * w[:, None], axis=0), p_z, atol=1e-12)
        pi = (g * w[:, None]).ravel(order="F")
        assert np.max(np.abs(sol.A_generator.T @ pi)) < 1e-12
        Ks_err[Na] = abs(float(np.sum(a_nu[:, None] * g * w[:, None])) - Ks_ref) / Ks_ref
    assert Ks_err[100] < 0.05, Ks_err
    assert Ks_err[400] < 0.01, Ks_err
    assert Ks_err[400] < Ks_err[100]

    # The GE wrapper integrates K^s with the same weights, so a non-uniform grid yields the
    # same equilibrium as a uniform one (r* was 33% too low before the fix)
    ge_nu = solve_aiyagari_continuous_hjb(a_grid=30.0 * np.linspace(0.0, 1.0, 200) ** 2)
    ge_u = solve_aiyagari_continuous_hjb(Na=400)
    assert ge_nu.converged and ge_u.converged
    assert abs(ge_nu.r_star - ge_u.r_star) < 0.05 * ge_u.r_star


def test_ge_tol_ge_governs_convergence_flag():
    """converged reflects |K^s - K^d| < tol_ge, tol_ge drives the root tolerance, and
    invalid brackets (r_max >= rho, r_min >= r_max, r_min <= -delta) are rejected."""
    res = solve_aiyagari_continuous_hjb(Na=40, a_max=25.0, tol_ge=1e-7, max_iter_ge=30)
    assert res.converged is True
    assert abs(res.excess_capital) < 1e-7
    assert abs(res.Ks_star - res.Kd_star) < 1e-7

    # Starved of iterations, the flag must not report a cleared market
    res_short = solve_aiyagari_continuous_hjb(Na=40, a_max=25.0, max_iter_ge=1)
    assert res_short.converged is False
    assert abs(res_short.excess_capital) > 1e-4

    with pytest.raises(ValueError, match="below rho_val"):
        solve_aiyagari_continuous_hjb(Na=20, r_max=0.06)
    with pytest.raises(ValueError, match="must be < r_max"):
        solve_aiyagari_continuous_hjb(Na=20, r_min=0.04, r_max=0.03)
    with pytest.raises(ValueError, match="-delta"):
        solve_aiyagari_continuous_hjb(Na=20, r_min=-0.06)


def test_income_generator_validation():
    """A_z must be an (Ne, Ne) generator: finite, non-negative off-diagonal, zero row sums;
    a generator without switching is refused for the KFE instead of returning NaN."""
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        solve_hjb_achdou(Na=20, A_z=np.ones((3, 3)))
    with pytest.raises(ValueError, match="transition-probability"):
        solve_hjb_achdou(Na=20, A_z=[[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError, match="non-negative"):
        solve_hjb_achdou(Na=20, A_z=[[-0.1, -0.1], [0.2, -0.2]])
    with pytest.raises(ValueError, match="finite"):
        solve_hjb_achdou(Na=20, A_z=[[np.nan, 0.1], [0.1, -0.1]])
    with pytest.raises(ValueError, match="row sums"):
        solve_hjb_achdou(Na=20, A_z=[[-0.1, 0.1], [0.1, -0.2]])
    with pytest.raises(ValueError, match="no unique stationary distribution"):
        solve_hjb_achdou(Na=20, A_z=np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        solve_aiyagari_continuous_hjb(Na=20, A_z=np.ones((3, 3)))

    # The HJB itself (the v3.3.0 model with independent income states) still solves
    sol = solve_hjb_achdou(Na=20, A_z=np.zeros((2, 2)), compute_kfe=False)
    assert sol.converged
    assert sol.g_dist is None
    assert np.all(np.isfinite(sol.V))


def test_zero_wage_requires_positive_lower_bound():
    """w_rate == 0 with a_grid[0] <= 0 raises a clear ValueError instead of NaN output."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning may leak before the ValueError
        with pytest.raises(ValueError, match="a_min > 0"):
            solve_hjb_achdou(w_rate=0.0, Na=30)
        with pytest.raises(ValueError, match="a_min > 0"):
            solve_hjb_achdou(w_rate=0.0, a_grid=np.linspace(-1.0, 5.0, 30))
    # a_min > 0 remains the supported cake-eating benchmark mode
    sol = solve_hjb_achdou(r_rate=0.0, w_rate=0.0, a_min=1.0, a_max=5.0, Na=30)
    assert sol.converged
    assert np.all(np.isfinite(sol.V))


def test_max_iter_must_be_positive():
    """max_iter < 1 raises ValueError instead of UnboundLocalError."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
            solve_hjb_achdou(Na=10, max_iter=bad)
    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        solve_aiyagari_continuous_hjb(Na=10, max_iter_hjb=0)
    sol = solve_hjb_achdou(Na=10, max_iter=1)
    assert sol.n_iter == 1
    assert sol.converged is False


def test_mapping_protocol_is_field_restricted_and_equality_is_array_aware():
    """Subscript/in/get see dataclass fields only (never methods); keys() keeps the 12 legacy
    keys; == is array-aware field equality; hash is refused like a dict."""
    sol = solve_hjb_achdou(Na=20, max_iter=20)
    assert len(sol.keys()) == 12
    assert len(sol) == 12
    assert len(dict(sol)) == 12
    for k in ("g_dist", "A_generator", "mass_residual"):
        assert k in sol
        assert sol[k] is getattr(sol, k)
    for k in ("summary", "plot", "keys", "__len__", "", "nonexistent"):
        assert k not in sol
        with pytest.raises(KeyError):
            sol[k]
        assert sol.get(k) is None
    for k in (123, None, ()):
        assert k not in sol

    copy_sol = pickle.loads(pickle.dumps(sol))
    assert sol == copy_sol
    assert not (sol != copy_sol)
    assert sol != solve_hjb_achdou(Na=21, max_iter=20)
    with pytest.raises(TypeError):
        hash(sol)

    res = solve_aiyagari_continuous_hjb(Na=20, a_max=20.0)
    assert "summary" not in res
    assert res.get("plot") is None
    with pytest.raises(KeyError):
        res["plot"]
    assert res == pickle.loads(pickle.dumps(res))
    with pytest.raises(TypeError):
        hash(res)
