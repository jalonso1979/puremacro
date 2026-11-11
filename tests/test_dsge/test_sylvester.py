"""Unit tests for the Generalized Schur Sylvester solver (_sylvester.py).

Verifies numerical accuracy, equivalence to dense Kronecker solutions, residual norm,
Schur decomposition stability on complex conjugate eigenvalues, zero-RHS corner cases,
near-singular fallback, memory efficiency, and runtime performance on SW07 dimensions.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from _timing import assert_within, best_of
from puremacro.dsge import load_mod
from puremacro.dsge._sylvester import (
    SylvesterWorkCounts,
    get_last_sylvester_work_counts,
    solve_dense_sylvester_fallback,
    solve_generalized_sylvester_kronecker,
    solve_generalized_sylvester_schur,
    solve_order1_sylvester,
    solve_order3_sylvester_kronecker,
    solve_order3_sylvester_schur,
)
from puremacro.dsge.local_accuracy import diagnose_perturbation


def test_schur_sylvester_2x2_basic():
    """Solve a 2x2 generalized Sylvester system and check dimensions and finiteness."""
    A_hat = np.array([[2.0, 0.5], [0.1, 1.5]])
    A_plus = np.array([[0.2, 0.0], [0.0, 0.1]])
    h_x = np.array([[0.8, 0.1], [0.0, 0.7]])
    K_xx = np.array([[1.0, 0.2, 0.2, 0.5], [0.3, 0.1, 0.1, 0.4]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (2, 4)
    assert np.isfinite(g_xx).all()
    assert np.isrealobj(g_xx)


def test_schur_sylvester_matches_kronecker_exact():
    """Schur Sylvester solution matches full Kronecker expansion to <= 1e-10."""
    rng = np.random.default_rng(12345)
    N, n_x = 3, 3
    A_hat = np.eye(N) * 2.0 + 0.1 * rng.standard_normal((N, N))
    A_plus = 0.1 * rng.standard_normal((N, N))
    h_x = 0.6 * np.eye(n_x) + 0.05 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    g_xx_schur = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)

    # Reference dense Kronecker solve
    C = np.kron(h_x, h_x)
    sys_mat = np.kron(np.eye(n_x**2), A_hat) + np.kron(C.T, A_plus)
    rhs = -K_xx.reshape(-1, order="F")
    vec_gxx = np.linalg.solve(sys_mat, rhs)
    g_xx_kron = vec_gxx.reshape((N, n_x**2), order="F")

    max_diff = np.max(np.abs(g_xx_schur - g_xx_kron))
    assert max_diff <= 1e-10, f"Max absolute difference {max_diff} > 1e-10"
    np.testing.assert_allclose(g_xx_schur, g_xx_kron, atol=1e-10)


def test_schur_sylvester_residual_norm():
    """Check algebraic residual ||A_hat X + A_+ X (h_x kron h_x) + K_xx|| < 1e-10."""
    rng = np.random.default_rng(999)
    N, n_x = 4, 3
    A_hat = np.eye(N) * 3.0 + 0.1 * rng.standard_normal((N, N))
    A_plus = 0.2 * rng.standard_normal((N, N))
    h_x = 0.7 * np.eye(n_x) + 0.03 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    res_norm = np.linalg.norm(residual)
    assert res_norm < 1e-10, f"Residual norm {res_norm} exceeds 1e-10"


def test_schur_sylvester_complex_conjugate_eigenvalues():
    """Operates stably when h_x has complex conjugate eigenvalue pairs."""
    theta = math.pi / 3  # 60 degrees rotation
    h_x = 0.85 * np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    A_hat = np.array([[2.5, 0.2], [0.1, 1.8]])
    A_plus = np.array([[0.1, 0.05], [0.02, 0.15]])
    K_xx = np.array([[1.0, 0.5, 0.5, 2.0], [0.2, 0.8, 0.8, 0.3]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.isrealobj(g_xx)
    assert np.isfinite(g_xx).all()

    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    assert np.linalg.norm(residual) < 1e-10


def test_schur_sylvester_zero_rhs():
    """Sylvester solver with zero RHS K_xx = 0 produces exact zero solution."""
    A_hat = np.eye(3) * 2.0
    A_plus = np.eye(3) * 0.1
    h_x = np.diag([0.5, 0.6])
    K_xx = np.zeros((3, 4))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.allclose(g_xx, 0.0)
    assert g_xx.shape == (3, 4)


def test_schur_sylvester_zero_hx():
    """Sylvester solver with h_x = 0 reduces directly to A_hat X = -K_xx."""
    A_hat = np.array([[2.0, 0.5], [0.0, 3.0]])
    A_plus = np.eye(2)
    h_x = np.zeros((2, 2))
    K_xx = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    expected = -np.linalg.solve(A_hat, K_xx)
    np.testing.assert_allclose(g_xx, expected, atol=1e-12)


def test_schur_sylvester_zero_states():
    """Sylvester solver handles models with zero predetermined states (n_x = 0)."""
    A_hat = np.eye(4)
    A_plus = np.eye(4) * 0.1
    h_x = np.zeros((0, 0))
    K_xx = np.zeros((4, 0))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (4, 0)


def test_schur_sylvester_1x1_state():
    """Sylvester solver on single-state model (n_x = 1)."""
    A_hat = np.array([[2.0, 0.1], [0.2, 1.5]])
    A_plus = np.array([[0.3, 0.0], [0.0, 0.2]])
    h_x = np.array([[0.7]])
    K_xx = np.array([[1.5], [0.8]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (2, 1)
    C = np.kron(h_x, h_x)
    residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
    assert np.linalg.norm(residual) < 1e-12


def test_schur_sylvester_near_singular_fallback():
    """Sylvester solver handles ill-conditioned system gracefully via least squares fallback."""
    A_hat = np.array([[1e-15, 0.0], [0.0, 1.0]])
    A_plus = np.eye(2) * 0.001
    h_x = np.array([[0.5]])
    K_xx = np.array([[1.0], [2.0]])

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert np.isfinite(g_xx).all()
    assert g_xx.shape == (2, 1)


@pytest.mark.benchmark
def test_schur_sylvester_sw07_benchmark_timing():
    """SW07 dimensions (N=40, nx=15) execute within budget(0.030) s (best of 3)."""
    rng = np.random.default_rng(42)
    N, n_x = 40, 15
    A_hat = np.eye(N) + 0.05 * rng.standard_normal((N, N))
    A_plus = 0.1 * rng.standard_normal((N, N))
    h_x = 0.5 * np.eye(n_x) + 0.02 * rng.standard_normal((n_x, n_x))
    K_xx = rng.standard_normal((N, n_x**2))

    g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
    assert g_xx.shape == (N, n_x**2)
    assert np.isfinite(g_xx).all()
    assert np.isrealobj(g_xx)

    elapsed = best_of(
        lambda: solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx))
    assert_within(elapsed, 0.030, "SW07-sized Sylvester solve (best of 3)")

    # Fast verification of sample columns residual
    C = np.kron(h_x, h_x)
    # Check 5 random columns
    cols = [0, 50, 100, 150, 224]
    for col in cols:
        lhs_col = A_hat @ g_xx[:, col] + A_plus @ (g_xx @ C[:, col])
        res_col = lhs_col + K_xx[:, col]
        assert np.linalg.norm(res_col) < 1e-9


def test_exact_cache_keys_preserve_reuse_for_repeated_eigenvalues_and_axis_symmetry():
    """Exact float keys reuse LU factorizations for repeated eigenvalues and symmetric products."""
    A_hat = np.array([[2.2, 0.3, -0.1], [0.1, 1.8, 0.2], [-0.05, 0.15, 2.5]])
    A_plus = np.array([[0.4, -0.05, 0.0], [0.02, 0.3, -0.04], [0.0, 0.03, 0.25]])
    # 3 states: two identical eigenvalues 0.85 and one distinct 0.60
    h_x = np.array([
        [0.85, 0.12, -0.04],
        [0.0, 0.85, 0.07],
        [0.0, 0.0, 0.60],
    ])
    rng = np.random.default_rng(20261024)
    K_x1 = rng.standard_normal((3, 3))
    K_xx = rng.standard_normal((3, 9))
    K_xxx = rng.standard_normal((3, 27))

    g1, w1 = solve_order1_sylvester(A_hat, A_plus, h_x, K_x1, return_work_counts=True)
    assert isinstance(w1, SylvesterWorkCounts)
    assert w1.scalar_solves == 3
    assert w1.lu_factorizations == 2
    assert w1.lu_cache_hits == 1
    np.testing.assert_allclose(A_hat @ g1 + A_plus @ g1 @ h_x + K_x1, 0.0, atol=1e-13)

    g2, w2 = solve_generalized_sylvester_schur(A_hat, A_plus, h_x, K_xx, return_work_counts=True)
    assert isinstance(w2, SylvesterWorkCounts)
    # 9 scalar blocks with unique products {0.85^2, 0.85*0.60, 0.60^2} -> 3 factorizations, 6 hits
    assert w2.scalar_solves == 9
    assert w2.lu_factorizations == 3
    assert w2.lu_cache_hits == 6
    assert w2.lu_cache_evictions == 0
    np.testing.assert_allclose(
        A_hat @ g2 + A_plus @ g2 @ np.kron(h_x, h_x) + K_xx, 0.0, atol=1e-12,
    )

    g3, w3 = solve_order3_sylvester_schur(A_hat, A_plus, h_x, K_xxx, return_work_counts=True)
    assert isinstance(w3, SylvesterWorkCounts)
    # 27 scalar blocks with unique products {0.85^3, 0.85^2*0.6, 0.85*0.6^2, 0.6^3} -> 4 factorizations, 23 hits
    assert w3.scalar_solves == 27
    assert w3.lu_factorizations == 4
    assert w3.lu_cache_hits == 23
    assert w3.lu_cache_evictions == 0
    hx3 = np.kron(np.kron(h_x, h_x), h_x)
    np.testing.assert_allclose(
        A_hat @ g3 + A_plus @ g3 @ hx3 + K_xxx, 0.0, atol=1e-12,
    )
    assert get_last_sylvester_work_counts(3) == w3


def test_nearby_distinct_near_unit_eigenvalue_products_do_not_collide():
    """Distinct eigenvalues differing at 1e-13 near 1.0 do not share LU factorizations."""
    mod_source = """
    var x z y;
    varexo e1 e2;
    parameters r1 r2 beta kappa units;
    r1 = 0.9999998;
    r2 = 0.9999998000001;
    beta = 0.9999998;
    kappa = 0.00000001;
    units = 1;
    model;
        x = r1*x(-1) + e1;
        z = r2*z(-1) + e2;
        y = beta*y(+1) + units*kappa*(
            (x(-1)+z(-1))^2/2 + (x(-1)+z(-1))^3/6);
    end;
    initval;
        x = 0;
        z = 0;
        y = 0;
    end;
    shocks;
        var e1; stderr 0;
        var e2; stderr 0;
    end;
    """
    from itertools import product
    from puremacro.dsge import read_mod

    for scale in (1.0, 1e6):
        model = read_mod(mod_source, params={"units": scale})
        params = model.parameter_values
        persistence = {"x": params["r1"], "z": params["r2"]}
        sol3 = model.solve(order=3, strict=True)
        dr = sol3.decision_rules()
        rho = [persistence[name] for name in dr.state_variables]
        prod2 = np.array([np.prod(idx) for idx in product(rho, repeat=2)])
        prod3 = np.array([np.prod(idx) for idx in product(rho, repeat=3)])
        expected_ghxx = scale * params["kappa"] / (1.0 - params["beta"] * prod2)
        expected_ghxxx = scale * params["kappa"] / (1.0 - params["beta"] * prod3)

        got_ghxx = dr.ghxx.loc["y"].to_numpy(dtype=float)
        got_ghxxx = dr.ghxxx.loc["y"].to_numpy(dtype=float)

        rel_err_2 = float(np.max(np.abs(got_ghxx / expected_ghxx - 1.0)))
        rel_err_3 = float(np.max(np.abs(got_ghxxx / expected_ghxxx - 1.0)))
        assert rel_err_2 < 1e-12, f"order-2 relative error {rel_err_2} too large at scale={scale}"
        assert rel_err_3 < 1e-12, f"order-3 relative error {rel_err_3} too large at scale={scale}"

        # Check work counts: 3 unique products at order 2 (1 hit), 4 unique products at order 3 (4 hits)
        work = sol3._sylvester_work
        assert work[2].lu_factorizations == 3
        assert work[2].lu_cache_hits == 1
        assert work[3].lu_factorizations == 4
        assert work[3].lu_cache_hits == 4

        diag = diagnose_perturbation(sol3)
        assert diag.sylvester_residuals["ghxx"] < 1e-12
        assert diag.sylvester_residuals["ghxxx"] < 1e-12


def test_order1_order2_order3_complex_2x2_schur_blocks_and_mixed_risk_analytic():
    """Verify orders 1, 2, 3 Sylvester and analytic risk terms (ghs2, ghxss, ghuss) with complex 2x2 Schur blocks."""
    # Coupled oscillator state block (complex conjugate eigenvalues) + forward control x
    mod_text = """
    var z1 z2 x;
    varexo e1 e2;
    parameters rho omega b a2 a3;
    rho = 0.8;
    omega = 0.5;
    b = 0.9;
    a2 = 0.3;
    a3 = 0.12;
    model;
    z1 = rho*(cos(omega)*z1(-1) - sin(omega)*z2(-1)) + e1 + 0.1*z1(-1)*e1;
    z2 = rho*(sin(omega)*z1(-1) + cos(omega)*z2(-1)) + e2;
    x  = b*x(+1) + a2*(z1^2 + z2^2) + a3*(z1^3 + z2^3) + 0.15*x(+1)*z1;
    end;
    initval;
    z1 = 0;
    z2 = 0;
    x  = 0;
    end;
    steady;
    """
    sol3 = load_mod(mod_text, order=3)
    diag = diagnose_perturbation(sol3)

    # Confirm complex conjugate roots -> 2x2 Schur block -> quasi-block solves up to m=8 at order 3
    assert bool((np.abs(diag.roots["imaginary"]) > 0.1).any())
    assert sol3._sylvester_work[1].quasi_block_solves == 1
    assert sol3._sylvester_work[1].max_block_size == 2
    assert sol3._sylvester_work[2].quasi_block_solves == 1
    assert sol3._sylvester_work[2].max_block_size == 4
    assert sol3._sylvester_work[3].quasi_block_solves == 1
    assert sol3._sylvester_work[3].max_block_size == 8

    # Check independent operator residuals for ghxx, ghxxx, ghxss, ghs2, ghuss
    for key in ("ghxx", "ghxxx", "ghxss", "ghs2", "ghuss"):
        assert key in diag.sylvester_residuals
        assert diag.sylvester_residuals[key] < 1e-12, f"{key} residual {diag.sylvester_residuals[key]} > 1e-12"

    # Verify analytic policy coefficients on a closed-form cubic risk model:
    # z_t = r z_{t-1} + s e_t
    # x_t = b E_t[x_{t+1}] + c1 z_t + 0.5 c2 z_t^2 + (1/6) c3 z_t^3
    analytic_mod = """
    var z x;
    varexo e;
    parameters r b c1 c2 c3;
    r  = 0.75;
    b  = 0.92;
    c1 = 1.2;
    c2 = 0.8;
    c3 = 0.6;
    model;
    z = r*z(-1) + e;
    x = b*x(+1) + c1*z + 0.5*c2*z^2 + (1/6)*c3*z^3;
    end;
    initval;
    z = 0;
    x = 0;
    end;
    steady;
    """
    r, b, c1, c2, c3 = 0.75, 0.92, 1.2, 0.8, 0.6
    s3 = load_mod(analytic_mod, order=3)
    dr = s3.decision_rules()

    # Exact closed-form coefficients for x_t as function of (z_{t-1}, e_t, sigma):
    # Since z_t = r z_{t-1} + e_t:
    # Linear in z_t: alpha_1 = c1 / (1 - b r)
    # Quadratic in z_t: alpha_2 = c2 / (1 - b r^2)
    # Cubic in z_t: alpha_3 = c3 / (1 - b r^3)
    # Volatility constant ghs2 for x: E_t[x_{t+1}] contributes b * alpha_2 * E[e_{t+1}^2] / (1 - b)
    alpha_2 = c2 / (1.0 - b * r**2)
    alpha_3 = c3 / (1.0 - b * r**3)
    expected_ghxx_x = alpha_2 * (r**2)
    expected_ghuu_x = alpha_2
    expected_ghs2_x = (b * alpha_2) / (1.0 - b)
    expected_ghxxx_x = alpha_3 * (r**3)
    expected_ghxuu_x = alpha_3 * r
    expected_ghxss_x = (b * alpha_3 * (r**2)) / (1.0 - b * r)
    expected_ghuss_x = (b * alpha_3 * r) / (1.0 - b * r)

    np.testing.assert_allclose(dr.ghxx.loc["x"].to_numpy(), [expected_ghxx_x], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghuu.loc["x"].to_numpy(), [expected_ghuu_x], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghs2.loc["x"], expected_ghs2_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghxxx.loc["x"].to_numpy(), [expected_ghxxx_x], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghxuu.loc["x"].to_numpy(), [expected_ghxuu_x], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghxss.loc["x"].to_numpy(), [expected_ghxss_x], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dr.ghuss.loc["x"].to_numpy(), [expected_ghuss_x], rtol=1e-12, atol=1e-12)


def test_singular_inconsistent_sylvester_fails_explicitly_while_consistent_succeeds():
    """Singular inconsistent systems raise LinAlgError; consistent singular systems succeed."""
    A_hat = np.array([[1.0, 0.0], [0.0, 0.0]])
    A_plus = np.zeros((2, 2))
    h_x = np.array([[0.5]])

    # Consistent singular RHS: second row is 0, in range(A_hat)
    K_consistent = np.array([[-2.0], [0.0]])
    sol_ok = solve_generalized_sylvester_schur(A_hat, A_plus, h_x, K_consistent)
    np.testing.assert_allclose(A_hat @ sol_ok + K_consistent, 0.0, atol=1e-14)

    # Inconsistent singular RHS: second row is non-zero, orthogonal to range(A_hat)
    K_inconsistent = np.array([[-2.0], [1.5]])
    with pytest.raises(np.linalg.LinAlgError, match="inconsistent"):
        solve_generalized_sylvester_schur(A_hat, A_plus, h_x, K_inconsistent)
    with pytest.raises(np.linalg.LinAlgError, match="inconsistent"):
        solve_order1_sylvester(A_hat, A_plus, h_x, K_inconsistent)
    with pytest.raises(np.linalg.LinAlgError, match="inconsistent"):
        solve_order3_sylvester_schur(A_hat, A_plus, h_x, K_inconsistent)

    # Also check quasi-block (complex 2x2 Schur block) inconsistent failure
    h_x_complex = np.array([[0.5, -0.5], [0.5, 0.5]])
    K_xx_inconsistent = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 2.0]])
    with pytest.raises(np.linalg.LinAlgError, match="inconsistent"):
        solve_generalized_sylvester_schur(A_hat, A_plus, h_x_complex, K_xx_inconsistent)


def test_dense_sylvester_fallback_enforces_allocation_limits_and_reports_details():
    """Dense Kronecker fallback refuses oversized allocations with an explicit diagnostic report."""
    N, n_x = 20, 10
    A_hat = np.eye(N)
    A_plus = 0.1 * np.eye(N)
    h_x = 0.5 * np.eye(n_x)
    K_xxx = np.ones((N, n_x**3))

    # Order 3 has dim = 20 * 1000 = 20,000 (> max_dense_dim=4096, ~3.05 GiB > 128 MiB)
    with pytest.raises(np.linalg.LinAlgError, match="exceeding allocation limits") as exc_info:
        solve_dense_sylvester_fallback(
            A_hat,
            A_plus,
            h_x,
            K_xxx,
            order=3,
            cause=RuntimeError("synthetic Schur failure"),
        )
    msg = str(exc_info.value)
    assert "20000x20000" in msg
    assert "synthetic Schur failure" in msg


def test_order3_sylvester_mixed_real_and_complex_schur_blocks():
    """Order-3 Real-Schur solver matches dense Kronecker across all mixed (1x1, 2x2) block permutations."""
    rng = np.random.default_rng(20261111)
    # 6 states: two distinct real eigenvalues + two distinct 2x2 complex oscillator blocks,
    # conjugated by a random non-orthogonal similarity transform with off-diagonal coupling.
    base_blocks = np.array(
        [
            [0.75, 0.20, -0.10, 0.05, 0.02, -0.04],
            [0.00, -0.55, 0.15, -0.08, 0.03, 0.01],
            [0.00, 0.00, 0.60, -0.45, 0.12, -0.06],
            [0.00, 0.00, 0.45, 0.60, 0.04, 0.09],
            [0.00, 0.00, 0.00, 0.00, -0.40, -0.50],
            [0.00, 0.00, 0.00, 0.00, 0.50, -0.40],
        ],
        dtype=float,
    )
    P_sim = np.eye(6) + 0.15 * rng.standard_normal((6, 6))
    h_x = P_sim @ base_blocks @ np.linalg.inv(P_sim)

    N = 4
    n_x = 6
    A_hat = np.eye(N) * 1.5 + 0.15 * rng.standard_normal((N, N))
    A_plus = 0.25 * rng.standard_normal((N, N))
    K_xxx = rng.standard_normal((N, n_x**3))

    g_xxx, counts = solve_order3_sylvester_schur(A_hat, A_plus, h_x, K_xxx, return_work_counts=True)
    # 4 Schur blocks (2 real 1x1 + 2 complex 2x2): 2^3 = 8 scalar blocks and 4^3 - 2^3 = 56 quasi-blocks
    assert counts.scalar_solves == 8
    assert counts.quasi_block_solves == 56
    assert counts.max_block_size == 8

    H3 = np.kron(np.kron(h_x, h_x), h_x)
    op_dense = np.kron(np.eye(n_x**3), A_hat) + np.kron(H3.T, A_plus)
    g_dense = np.linalg.solve(op_dense, (-K_xxx).ravel(order="F")).reshape((N, n_x**3), order="F")

    np.testing.assert_allclose(g_xxx, g_dense, rtol=1e-11, atol=1e-12)
    res_inf = float(np.max(np.abs(A_hat @ g_xxx + (A_plus @ g_xxx) @ H3 + K_xxx)))
    assert res_inf < 1e-12

    # End-to-end public order=3 DSGE model with mixed real + complex-conjugate state dynamics
    mixed_mod = """
    var r1 r2 z1 z2 y;
    varexo e_r1 e_r2 e1 e2;
    parameters rho_r1 rho_r2 rho_c omega beta c3;
    rho_r1 = 0.75;
    rho_r2 = -0.55;
    rho_c  = 0.80;
    omega  = 0.60;
    beta   = 0.92;
    c3     = 0.90;
    model;
        r1 = rho_r1 * r1(-1) + e_r1;
        r2 = rho_r2 * r2(-1) + e_r2;
        z1 = rho_c * (cos(omega) * z1(-1) - sin(omega) * z2(-1)) + e1;
        z2 = rho_c * (sin(omega) * z1(-1) + cos(omega) * z2(-1)) + e2;
        y  = beta * y(+1) + c3 * r1 * r2 * z1 + 0.4 * r1^2 * z2;
    end;
    initval;
        r1 = 0; r2 = 0; z1 = 0; z2 = 0; y = 0;
    end;
    steady;
    """
    sol3 = load_mod(mixed_mod, order=3)
    diag = diagnose_perturbation(sol3)
    assert diag.sylvester_residuals["ghxxx"] < 1e-12
    assert not any("Sylvester residual" in w for w in diag.warnings)

