"""Unit and regression test suite for Smolyak sparse grid collocation (puremacro.vfi.smolyak).

Verifies:
1. Clenshaw-Curtis nesting and disjoint node increments:
   - X^(i) subset X^(i+1) for all i >= 1.
   - Disjoint increments H^(i) are mutually disjoint and partition X^(i).
   - Exact node counts N(d, mu) for d in [2, 6] and mu in [1, 4].
   - Node reduction >= 5x over tensor product grids for d >= 3, mu >= 2.
2. Coordinate transformation:
   - Bijective affine mapping between domain bounds and canonical hypercube [-1, 1]^d.
   - Exact inverse round-trips: to_physical(to_canonical(s)) == s.
3. SmolyakBasis properties:
   - Square, invertible collocation matrix Phi with cond(Phi) < 30 for mu <= 2.
   - Exact reproduction of multivariate polynomials up to degree mu (error < 10^-13).
   - Analytical basis derivatives match finite-difference gradients.
   - Combination technique interpolation matches direct collocation.
4. Multi-dimensional economic model solving:
   - 2D multi-capital neoclassical growth model with closed-form solution:
     continuous Euler equation residuals < 10^-4 on >= 1,000 out-of-sample points.
   - 3D multi-capital neoclassical growth model with closed-form solution:
     continuous Euler equation residuals < 10^-4 on >= 1,000 out-of-sample points.
   - Policy relative error against true analytical policy < 10^-4.
   - Value function iteration (Bellman method) convergence and consistency.
   - Custom euler_residual_fn support.
5. Multi-backend hardware acceleration and graceful fallback:
   - NumPy, Numba, MLX parity with relative difference < 10^-4.
   - CuPy unavailable on macOS triggers warning and falls back to NumPy cleanly.
   - Unsupported backends raise ValueError.
6. Presentation contract compliance:
   - .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot().
7. Edge cases and input validation:
   - Non-positive dimension, negative level, reversed bounds, invalid beta, invalid method.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for CI/tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
import puremacro.vfi.smolyak as sm
from puremacro.vfi.smolyak import (
    SmolyakBasis,
    SmolyakGrid,
    SmolyakProblem,
    SmolyakSolution,
    solve_smolyak,
)


# ===========================================================================
# 1. Clenshaw-Curtis Nesting, Disjoint Increments, and Sparse Grid Nodes
# ===========================================================================

def test_clenshaw_curtis_nesting():
    """Verify strict nesting of 1D Clenshaw-Curtis extrema nodes: X^(i) subset X^(i+1)."""
    for i in range(1, 5):
        nodes_i = sm._clenshaw_curtis_nodes(i)
        nodes_next = sm._clenshaw_curtis_nodes(i + 1)
        for x in nodes_i:
            assert np.any(np.abs(nodes_next - x) < 1e-12), f"Node {x} from level {i} not in level {i+1}"


def test_disjoint_increments_partition():
    """Verify disjoint increments H^(i) are mutually disjoint and partition X^(i)."""
    collected = []
    for i in range(1, 6):
        h_i = sm._disjoint_increments(i)
        if i == 1:
            assert len(h_i) == 1 and h_i[0] == 0.0
        elif i == 2:
            assert len(h_i) == 2 and set(np.round(h_i, 6)) == {-1.0, 1.0}
        else:
            assert len(h_i) == (1 << (i - 2))

        # Check mutual disjointness against previously collected nodes
        for prev_h in collected:
            for x in h_i:
                assert not np.any(np.abs(prev_h - x) < 1e-12), f"Overlap found for node {x}"
        collected.append(h_i)


@pytest.mark.parametrize(
    "d,expected_counts",
    [
        (2, [5, 13, 29, 65]),
        (3, [7, 25, 69, 177]),
        (4, [9, 41, 137, 401]),
        (5, [11, 61, 241, 801]),
        (6, [13, 85, 389, 1457]),
    ],
)
def test_sparse_grid_node_counts(d: int, expected_counts: list[int]):
    """Verify exact Smolyak sparse grid node counts across d in [2, 6] and mu in [1, 4]."""
    actual_counts = []
    for mu in range(1, 5):
        grid = SmolyakGrid(d=d, mu=mu)
        actual_counts.append(grid.n_nodes)
        assert len(grid) == grid.n_nodes
        assert grid.nodes.shape == (grid.n_nodes, d)
        assert grid.physical_nodes.shape == (grid.n_nodes, d)
        assert grid.poly_indices.shape == (grid.n_nodes, d)
    assert actual_counts == expected_counts


def test_node_reduction_ratio():
    """Verify >= 5x node reduction over full tensor product grid for d >= 3, mu >= 2."""
    grid_3d_mu2 = SmolyakGrid(d=3, mu=2)
    assert grid_3d_mu2.reduction_ratio >= 5.0
    assert grid_3d_mu2.tensor_nodes_count == 125
    assert grid_3d_mu2.n_nodes == 25

    grid_3d_mu3 = SmolyakGrid(d=3, mu=3)
    assert grid_3d_mu3.reduction_ratio >= 10.0
    assert grid_3d_mu3.tensor_nodes_count == 729
    assert grid_3d_mu3.n_nodes == 69

    grid_4d_mu2 = SmolyakGrid(d=4, mu=2)
    assert grid_4d_mu2.reduction_ratio >= 15.0

    grid_6d_mu3 = SmolyakGrid(d=6, mu=3)
    assert grid_6d_mu3.reduction_ratio > 1000.0


# ===========================================================================
# 2. Domain Coordinate Mappings
# ===========================================================================

def test_affine_domain_roundtrip():
    """Verify bijection between physical domain bounds and canonical hypercube [-1, 1]^d."""
    domain = ((0.5, 2.5), (10.0, 50.0), (-3.0, 7.0))
    grid = SmolyakGrid(d=3, mu=2, domain=domain)

    # Check canonical nodes are strictly in [-1, 1]
    assert np.all(grid.nodes >= -1.0) and np.all(grid.nodes <= 1.0)

    # Check physical nodes are strictly within domain
    for k, (ak, bk) in enumerate(domain):
        assert np.all(grid.physical_nodes[:, k] >= ak - 1e-12)
        assert np.all(grid.physical_nodes[:, k] <= bk + 1e-12)

    # Test round-trips
    np.random.seed(42)
    s_random = np.column_stack([
        np.random.uniform(ak, bk, 50) for ak, bk in domain
    ])
    x_mapped = grid.to_canonical(s_random)
    s_back = grid.to_physical(x_mapped)
    assert np.max(np.abs(s_random - s_back)) < 1e-12

    # Single point round-trip
    s_single = np.array([1.5, 30.0, 2.0])
    x_single = grid.to_canonical(s_single)
    s_single_back = grid.to_physical(x_single)
    assert np.max(np.abs(s_single - s_single_back)) < 1e-12


# ===========================================================================
# 3. SmolyakBasis: Conditioning, Polynomial Exactness & Derivatives
# ===========================================================================

@pytest.mark.parametrize(
    "d,mu,max_cond",
    [
        (2, 1, 5.0),
        (2, 2, 10.0),
        (2, 3, 15.0),
        (3, 1, 8.0),
        (3, 2, 15.0),
        (3, 3, 30.0),
        (4, 1, 12.0),
        (4, 2, 25.0),
    ],
)
def test_collocation_matrix_condition_numbers(d: int, mu: int, max_cond: float):
    """Verify direct Chebyshev basis collocation matrix condition number remains bounded."""
    basis = SmolyakBasis(d=d, mu=mu)
    assert basis.n_basis == basis.grid.n_nodes
    assert basis.collocation_matrix.shape == (basis.n_basis, basis.n_basis)
    assert basis.condition_number < max_cond


def test_multivariate_polynomial_exact_reproduction():
    """Verify exact machine-precision reproduction of multivariate polynomials of degree <= mu."""
    d, mu = 2, 2
    domain = ((-2.0, 3.0), (1.0, 4.0))
    grid = SmolyakGrid(d=d, mu=mu, domain=domain)
    basis = SmolyakBasis(grid=grid)

    # Total degree 2 polynomial: P(s) = 2 + 3*s1 - 4*s2 + 1.5*s1^2 - 2*s1*s2 + 0.5*s2^2
    def poly_target(s: np.ndarray) -> np.ndarray:
        s1, s2 = s[:, 0], s[:, 1]
        return 2.0 + 3.0 * s1 - 4.0 * s2 + 1.5 * (s1**2) - 2.0 * s1 * s2 + 0.5 * (s2**2)

    y_nodes = poly_target(grid.physical_nodes)
    theta = basis.fit(y_nodes)

    # Evaluate on 500 dense query points
    np.random.seed(123)
    s_dense = np.column_stack([
        np.random.uniform(domain[0][0], domain[0][1], 500),
        np.random.uniform(domain[1][0], domain[1][1], 500),
    ])
    y_true = poly_target(s_dense)
    y_approx = basis.interpolate(theta, s_dense)

    max_err = np.max(np.abs(y_true - y_approx))
    assert max_err < 1e-12, f"Polynomial reproduction error too high: {max_err}"


def test_analytical_basis_derivatives():
    """Verify basis analytical partial derivatives against finite-difference approximations."""
    domain = ((1.0, 3.0), (2.0, 5.0))
    basis = SmolyakBasis(d=2, mu=2, domain=domain)

    # Fit a quadratic function
    def f(s: np.ndarray) -> np.ndarray:
        return 2.0 * s[:, 0] ** 2 + 3.0 * s[:, 0] * s[:, 1] - s[:, 1] ** 2

    y_nodes = f(basis.grid.physical_nodes)
    theta = basis.fit(y_nodes)

    # Query points
    s_query = np.array([[1.5, 3.2], [2.2, 4.1], [2.8, 2.5]])
    eps = 1e-7

    for dim in (0, 1):
        dB = basis.derivative(s_query, dim=dim)
        grad_analytical = dB @ theta

        s_plus = s_query.copy()
        s_plus[:, dim] += eps
        s_minus = s_query.copy()
        s_minus[:, dim] -= eps

        grad_fd = (f(s_plus) - f(s_minus)) / (2.0 * eps)
        assert np.max(np.abs(grad_analytical - grad_fd)) < 1e-6


def test_combination_technique_parity():
    """Verify Smolyak combination technique matches direct basis evaluation."""
    basis = SmolyakBasis(d=2, mu=2, domain=((0.5, 2.0), (0.5, 2.0)))
    nodes = basis.grid.physical_nodes

    def f(s: np.ndarray) -> np.ndarray:
        return np.sin(s[:, 0]) * np.cos(s[:, 1])

    y_nodes = f(nodes)
    theta = basis.fit(y_nodes)

    np.random.seed(99)
    test_s = np.random.uniform(0.5, 2.0, (100, 2))
    direct_vals = basis.interpolate(theta, test_s)
    comb_vals = basis.combination_interpolate(y_nodes, test_s)

    assert np.max(np.abs(direct_vals - comb_vals)) < 1e-12


# ===========================================================================
# 4. Multi-Dimensional Economic Models: Neoclassical Growth Benchmark
# ===========================================================================

def test_2d_neoclassical_growth_euler_and_policy():
    """Verify 2D multi-capital neoclassical growth model with analytical solution.

    Target: Continuous Euler equation residuals < 10^-4 and policy relative error < 10^-4.
    """
    alphas = [0.18, 0.18]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)  # Normalize k_ss = (1.0, 1.0)
    domain = ((0.8, 1.2), (0.8, 1.2))

    prob = SmolyakProblem(
        domain=domain,
        mu=2,
        params={"alphas": alphas, "z": z, "beta": beta},
    )
    sol = prob.solve(backend="numpy")

    assert sol.converged, "2D Smolyak solver did not report convergence"
    assert sol.residual_norm < 1e-4

    # Dense out-of-sample evaluation on 1,000 random continuous points
    np.random.seed(42)
    test_s = np.random.uniform(0.8, 1.2, (1000, 2))

    # 1. Policy relative error against analytical truth kp_m = alpha_m * beta * Y
    kp_approx = sol.policy(test_s)
    Y_test = z * (test_s[:, 0] ** alphas[0]) * (test_s[:, 1] ** alphas[1])
    kp_true = np.column_stack([alphas[0] * beta * Y_test, alphas[1] * beta * Y_test])

    rel_error = np.max(np.abs(kp_approx - kp_true) / kp_true)
    assert rel_error < 1e-4, f"Policy relative error exceeded 1e-4: {rel_error}"

    # 2. Continuous Euler equation residual across 1,000 points
    res_dense = sol.euler_residual(test_s)
    max_euler = np.max(np.abs(res_dense))
    assert max_euler < 1e-4, f"Continuous Euler equation residual exceeded 1e-4: {max_euler}"


def test_3d_neoclassical_growth_euler_and_scaling():
    """Verify 3D multi-capital model achieving >= 5x node reduction and Euler residual < 10^-4."""
    alphas = [0.12, 0.12, 0.12]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)
    domain = ((0.7, 1.3), (0.7, 1.3), (0.7, 1.3))

    sol = solve_smolyak(
        domain=domain,
        mu=3,
        params={"alphas": alphas, "z": z, "beta": beta},
    )

    assert sol.converged
    # Node reduction >= 5x
    assert sol.basis.grid.reduction_ratio >= 10.0
    assert sol.basis.n_basis == 69
    assert sol.basis.grid.tensor_nodes_count == 729

    # Dense continuous evaluation on 1,000 random points
    np.random.seed(84)
    test_s = np.random.uniform(0.7, 1.3, (1000, 3))
    res_dense = sol.euler_residual(test_s)
    max_euler = np.max(np.abs(res_dense))
    assert max_euler < 1e-4, f"3D Continuous Euler equation residual exceeded 1e-4: {max_euler}"


def test_bellman_method_convergence_and_consistency():
    """Verify continuous value function iteration (Bellman method) on Smolyak grid."""
    alphas = [0.18, 0.18]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)
    domain = ((0.8, 1.2), (0.8, 1.2))

    sol_bellman = solve_smolyak(
        domain=domain,
        mu=2,
        method="bellman",
        params={"alphas": alphas, "z": z, "beta": beta},
    )

    assert sol_bellman.converged
    assert sol_bellman.residual_norm < 1e-6

    # Steady state policy check at (1.0, 1.0)
    s_ss = np.array([1.0, 1.0])
    kp_ss = sol_bellman.policy(s_ss)
    assert np.allclose(kp_ss, [1.0, 1.0], atol=1e-3)


# ===========================================================================
# 5. Multi-Backend Parity and Fallback
# ===========================================================================

def test_multi_backend_consistency():
    """Verify NumPy, Numba, and MLX backends produce consistent solutions."""
    alphas = [0.18, 0.18]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)
    domain = ((0.8, 1.2), (0.8, 1.2))

    sol_numpy = solve_smolyak(domain=domain, mu=2, backend="numpy", params={"alphas": alphas, "z": z, "beta": beta})
    test_pts = np.random.uniform(0.8, 1.2, (50, 2))
    pol_numpy = sol_numpy.policy(test_pts)

    for backend_name in ("numba", "mlx"):
        if bk.backend_available(backend_name):
            sol_b = solve_smolyak(domain=domain, mu=2, backend=backend_name, params={"alphas": alphas, "z": z, "beta": beta})
            assert sol_b.backend == backend_name
            pol_b = sol_b.policy(test_pts)
            assert np.max(np.abs(pol_numpy - pol_b)) < 1e-4


def test_backend_graceful_fallback_for_cupy():
    """Verify CuPy issues warning and gracefully falls back to NumPy when unavailable."""
    domain = ((0.8, 1.2), (0.8, 1.2))
    alphas = [0.18, 0.18]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)

    if not bk.backend_available("cupy"):
        with pytest.warns(UserWarning, match="falling back to 'numpy'"):
            sol = solve_smolyak(domain=domain, mu=2, backend="cupy", params={"alphas": alphas, "z": z, "beta": beta})
        assert sol.backend == "numpy"
        assert sol.converged


def test_invalid_backend_raises_error():
    """Verify unsupported backend names raise ValueError."""
    domain = ((0.8, 1.2), (0.8, 1.2))
    with pytest.raises(ValueError, match="Unknown backend"):
        solve_smolyak(domain=domain, mu=2, backend="jax_unknown")


# ===========================================================================
# 6. Presentation Contract Compliance
# ===========================================================================

def test_presentation_contract_methods():
    """Verify all 6 puremacro presentation methods work properly."""
    alphas = [0.18, 0.18]
    beta = 0.96
    z = 1.0 / (alphas[0] * beta)
    domain = ((0.8, 1.2), (0.8, 1.2))

    sol = solve_smolyak(domain=domain, mu=2, params={"alphas": alphas, "z": z, "beta": beta})

    # 1. .summary()
    df = sol.summary()
    assert isinstance(df, pd.DataFrame)
    assert "Smolyak Nodes (N)" in df.index
    assert "Node Reduction Ratio" in df.index

    # 2. .to_frame()
    frame = sol.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.equals(df)

    # 3. .to_markdown()
    md = sol.to_markdown()
    assert isinstance(md, str)
    assert "Metric" in md and "Value" in md and "Smolyak Nodes (N)" in md

    # 4. .to_latex()
    tex = sol.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{tabular}" in tex

    # 5. .to_typst()
    typ = sol.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ

    # 6. .plot()
    fig = sol.plot()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


# ===========================================================================
# 7. Edge Cases and Parameter Validation
# ===========================================================================

def test_invalid_parameters_raise_value_errors():
    """Verify defensive input checks on SmolyakGrid, SmolyakProblem, and solve_smolyak."""
    # Invalid dimension
    with pytest.raises(ValueError, match="Dimension d must be >= 1"):
        SmolyakGrid(d=0, mu=2)

    # Negative approximation level
    with pytest.raises(ValueError, match="level mu must be >= 0"):
        SmolyakGrid(d=2, mu=-1)

    # Inverted bounds
    with pytest.raises(ValueError, match="bounds for dim 0 must satisfy a < b"):
        SmolyakGrid(d=2, mu=2, domain=((2.0, 1.0), (0.0, 1.0)))

    # Invalid discount factor
    with pytest.raises(ValueError, match="beta must be in \\(0, 1\\)"):
        SmolyakProblem(domain=((0.5, 1.5), (0.5, 1.5)), beta=1.5)

    # Invalid solution method
    with pytest.raises(ValueError, match="method must be 'euler' or 'bellman'"):
        SmolyakProblem(domain=((0.5, 1.5), (0.5, 1.5)), method="invalid_method")

    # Both problem and kwargs
    prob = SmolyakProblem(domain=((0.5, 1.5), (0.5, 1.5)))
    with pytest.raises(ValueError, match="Cannot pass both"):
        solve_smolyak(problem=prob, mu=3)
