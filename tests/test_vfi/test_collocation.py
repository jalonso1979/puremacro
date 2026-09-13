"""Unit and verification tests for Polynomial Collocation Solver in puremacro.vfi.collocation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from puremacro import _backend as _bk
from puremacro.vfi.collocation import (
    CollocationBasis,
    CollocationProblem,
    CollocationSolution,
    solve_collocation,
)


# ===========================================================================
# 1. CollocationBasis: Nodes, Mapping & Recurrence Relations
# ===========================================================================

def test_basis_lobatto_nodes():
    """Verify Gauss-Lobatto nodes include endpoints and are strictly increasing."""
    basis = CollocationBasis(domain=(1.0, 5.0), orders=4, node_type="lobatto")
    nodes = basis.nodes(squeeze=True)
    assert len(nodes) == 5
    assert np.isclose(nodes[0], 1.0)
    assert np.isclose(nodes[-1], 5.0)
    assert np.all(np.diff(nodes) > 0.0)


def test_basis_gauss_nodes():
    """Verify Gauss nodes lie strictly inside the open interior (a, b)."""
    basis = CollocationBasis(domain=(1.0, 5.0), orders=4, node_type="gauss")
    nodes = basis.nodes(squeeze=True)
    assert len(nodes) == 5
    assert nodes[0] > 1.0
    assert nodes[-1] < 5.0
    assert np.all(np.diff(nodes) > 0.0)


def test_coordinate_mapping_bijective():
    """Verify bijective forward and inverse coordinate mapping between [a, b] and [-1, 1]."""
    domain = (0.2, 4.8)
    basis = CollocationBasis(domain=domain, orders=5)

    # Test points across domain
    s = np.linspace(0.2, 4.8, 50)
    x = basis.to_canonical(s)
    assert np.all(x >= -1.0 - 1e-12)
    assert np.all(x <= 1.0 + 1e-12)

    s_recov = basis.to_state(x)
    assert np.allclose(s, s_recov.ravel(), atol=1e-12)


def test_chebyshev_3term_recurrence_exact():
    """Verify 3-term recurrence T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x) matches explicit formulas."""
    basis = CollocationBasis(domain=(-1.0, 1.0), orders=5)
    x = np.linspace(-1.0, 1.0, 100)
    phi = basis.evaluate(x)

    # Explicit definitions:
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    # T4 = 8x^4 - 8x^2 + 1
    # T5 = 16x^5 - 20x^3 + 5x
    assert np.allclose(phi[:, 0], 1.0)
    assert np.allclose(phi[:, 1], x)
    assert np.allclose(phi[:, 2], 2.0 * x**2 - 1.0)
    assert np.allclose(phi[:, 3], 4.0 * x**3 - 3.0 * x)
    assert np.allclose(phi[:, 4], 8.0 * x**4 - 8.0 * x**2 + 1.0)
    assert np.allclose(phi[:, 5], 16.0 * x**5 - 20.0 * x**3 + 5.0 * x)


def test_chebyshev_derivative_recurrence():
    """Verify derivative recurrence matches central finite differences."""
    basis = CollocationBasis(domain=(0.5, 2.5), orders=6)
    s = np.linspace(0.6, 2.4, 20)
    h = 1e-6

    phi_plus = basis.evaluate(s + h)
    phi_minus = basis.evaluate(s - h)
    num_deriv = (phi_plus - phi_minus) / (2.0 * h)

    analytic_deriv = basis.derivative(s, dim=0)
    assert np.allclose(analytic_deriv, num_deriv, atol=1e-5, rtol=1e-5)


def test_chebyshev_orthogonality():
    r"""Verify orthogonality with Chebyshev weight w(x) = (1 - x^2)^(-1/2).

    Under the transformation x = cos(theta), dx / sqrt(1 - x^2) = -d(theta),
    so \int_{-1}^1 T_m(x) T_n(x) w(x) dx = \int_0^\pi T_m(cos \theta) T_n(cos \theta) d\theta = 0 for m != n.
    """
    basis = CollocationBasis(domain=(-1.0, 1.0), orders=4)

    def integrand(theta, m, n):
        x = np.cos(theta)
        T = basis.evaluate(np.array([x]))
        return T[0, m] * T[0, n]

    val_12, _ = quad(integrand, 0.0, np.pi, args=(1, 2))
    assert np.isclose(val_12, 0.0, atol=1e-12)

    val_24, _ = quad(integrand, 0.0, np.pi, args=(2, 4))
    assert np.isclose(val_24, 0.0, atol=1e-12)

    val_13, _ = quad(integrand, 0.0, np.pi, args=(1, 3))
    assert np.isclose(val_13, 0.0, atol=1e-12)


# ===========================================================================
# 2. Multi-Dimensional Tensor Product Basis
# ===========================================================================

def test_basis_tensor_product_2d():
    """Verify 2D tensor product basis shapes and polynomial interpolation."""
    domain = ((0.0, 2.0), (1.0, 3.0))
    orders = (3, 4)
    basis = CollocationBasis(domain=domain, orders=orders)

    assert basis.n_dims == 2
    assert basis.n_nodes == (3 + 1) * (4 + 1)  # 20
    nodes = basis.nodes()
    assert nodes.shape == (20, 2)

    # Test exact interpolation of 2D polynomial
    # f(x, y) = 3*x^2 - x*y + 2*y - 5
    y_vals = 3.0 * nodes[:, 0]**2 - nodes[:, 0] * nodes[:, 1] + 2.0 * nodes[:, 1] - 5.0
    c = basis.fit(y_vals)

    test_pts = np.array([[0.5, 1.5], [1.2, 2.8], [1.9, 1.1]])
    y_interp = basis.interpolate(c, test_pts)
    y_exact = 3.0 * test_pts[:, 0]**2 - test_pts[:, 0] * test_pts[:, 1] + 2.0 * test_pts[:, 1] - 5.0
    assert np.allclose(y_interp, y_exact, atol=1e-12)


def test_basis_tensor_product_partial_derivatives_2d():
    """Verify partial derivatives along both dimensions in 2D."""
    domain = ((1.0, 4.0), (2.0, 5.0))
    basis = CollocationBasis(domain=domain, orders=(3, 3))
    nodes = basis.nodes()

    # f(x, y) = x^2 * y - 2*x + 3*y^2
    y_vals = (nodes[:, 0]**2) * nodes[:, 1] - 2.0 * nodes[:, 0] + 3.0 * (nodes[:, 1]**2)
    c = basis.fit(y_vals)

    test_pts = np.array([[1.5, 2.5], [3.0, 4.0]])

    # df/dx = 2*x*y - 2
    df_dx_exact = 2.0 * test_pts[:, 0] * test_pts[:, 1] - 2.0
    dphi_dx = basis.derivative(test_pts, dim=0)
    df_dx_calc = dphi_dx @ c
    assert np.allclose(df_dx_calc, df_dx_exact, atol=1e-10)

    # df/dy = x^2 + 6*y
    df_dy_exact = test_pts[:, 0]**2 + 6.0 * test_pts[:, 1]
    dphi_dy = basis.derivative(test_pts, dim=1)
    df_dy_calc = dphi_dy @ c
    assert np.allclose(df_dy_calc, df_dy_exact, atol=1e-10)


def test_basis_validation_errors():
    """Verify defensive input validation on CollocationBasis."""
    with pytest.raises(ValueError, match="domain bounds"):
        CollocationBasis(domain=(2.0, 1.0), orders=3)

    with pytest.raises(ValueError, match="must be >= 1"):
        CollocationBasis(domain=(1.0, 2.0), orders=0)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        CollocationBasis(domain=((1.0, 2.0), (2.0, 3.0)), orders=3)

    with pytest.raises(ValueError, match="Unsupported basis_type"):
        CollocationBasis(domain=(1.0, 2.0), orders=3, basis_type="legendre")

    with pytest.raises(ValueError, match="Unsupported node_type"):
        CollocationBasis(domain=(1.0, 2.0), orders=3, node_type="chebyshev_roots")


# ===========================================================================
# 3. CollocationProblem Validation
# ===========================================================================

def test_problem_validation():
    """Verify defensive parameter checking in CollocationProblem."""
    with pytest.raises(ValueError, match="beta must be in"):
        CollocationProblem(domain=(0.1, 2.0), orders=5, beta=1.05)

    with pytest.raises(ValueError, match="method must be"):
        CollocationProblem(domain=(0.1, 2.0), orders=5, method="unknown_method")

    with pytest.raises(ValueError, match="node_type must be"):
        CollocationProblem(domain=(0.1, 2.0), orders=5, node_type="random")

    with pytest.raises(ValueError, match="requires euler_residual_fn, return_fn, or model parameters"):
        CollocationProblem(domain=(0.1, 2.0), orders=5, method="euler", params={})

    with pytest.raises(ValueError, match="requires return_fn or model parameters"):
        CollocationProblem(domain=(0.1, 2.0), orders=5, method="bellman", params={})


# ===========================================================================
# 4. Canonical Benchmark: Brock-Mirman Neoclassical Growth Model
# ===========================================================================

def test_neoclassical_growth_euler_collocation():
    """Verify Euler collocation solver on canonical Brock-Mirman model.

    Theoretical Closed Form:
      g*(k) = alpha * beta * k^alpha
    Acceptance Criteria:
      - Policy relative error < 10^-4 across dense 1,000 points.
      - Continuous Euler equation residual < 10^-4 across dense 1,000 points.
    """
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_min = 0.5 * k_ss
    k_max = 1.5 * k_ss

    prob = CollocationProblem(
        domain=(k_min, k_max),
        orders=6,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )

    sol = prob.solve(backend="numpy")
    assert isinstance(sol, CollocationSolution)
    assert sol.converged
    assert sol.residual_norm < 1e-8

    # Dense evaluation on 1,000 out-of-sample points
    eval_grid = np.linspace(k_min, k_max, 1000)
    g_approx = sol.policy(eval_grid)
    g_true = alpha * beta * (eval_grid**alpha)

    # Policy relative error < 10^-4
    rel_policy_err = np.max(np.abs(g_approx - g_true) / g_true)
    assert rel_policy_err < 1e-4

    # Continuous Euler equation residual < 10^-4
    c_eval = eval_grid**alpha - g_approx
    kp_eval = g_approx
    kpp_eval = sol.policy(kp_eval)
    cp_eval = kp_eval**alpha - kpp_eval
    euler_res = np.max(np.abs(1.0 - beta * (c_eval / cp_eval) * alpha * (kp_eval ** (alpha - 1.0))))
    assert euler_res < 1e-4


def test_neoclassical_growth_bellman_collocation():
    """Verify continuous Bellman value collocation on Brock-Mirman model.

    Theoretical Closed Form:
      V*(k) = A * ln(k) + B
      where A = alpha / (1 - alpha * beta)
    """
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_min = 0.5 * k_ss
    k_max = 1.5 * k_ss

    prob = CollocationProblem(
        domain=(k_min, k_max),
        orders=8,
        method="bellman",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
        options={"max_iter": 100, "n_howard": 15, "tol": 1e-10},
    )

    sol = prob.solve(backend="numpy")
    assert sol.converged

    # Test value function accuracy
    eval_grid = np.linspace(k_min, k_max, 1000)
    v_approx = sol.value(eval_grid)

    A = alpha / (1.0 - alpha * beta)
    B = (np.log(1.0 - alpha * beta) + (alpha * beta * np.log(alpha * beta)) / (1.0 - alpha * beta)) / (1.0 - beta)
    v_exact = A * np.log(eval_grid) + B

    max_val_err = np.max(np.abs(v_approx - v_exact))
    assert max_val_err < 1e-4

    # Policy accuracy
    g_approx = sol.policy(eval_grid)
    g_true = alpha * beta * (eval_grid**alpha)
    rel_policy_err = np.max(np.abs(g_approx - g_true) / g_true)
    assert rel_policy_err < 1e-4


def test_custom_euler_residual_fn():
    """Verify solving with a user-supplied custom euler_residual_fn."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    def user_euler(policy_fn, s, params):
        a = params["alpha"]
        kp = policy_fn(s)
        c = s**a - kp
        kpp = policy_fn(kp)
        cp = kp**a - kpp
        return 1.0 - 0.96 * (c / cp) * a * (kp ** (a - 1.0))

    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=6,
        method="euler",
        euler_residual_fn=user_euler,
        params={"alpha": alpha},
        beta=beta,
    )
    sol = prob.solve()
    assert sol.converged
    assert sol.residual_norm < 1e-8


def test_custom_return_fn_bellman():
    """Verify Bellman value collocation with a user-supplied return_fn."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    def user_utility(sp, s, **params):
        c = s**0.36 - sp
        return np.log(np.maximum(c, 1e-12))

    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=6,
        method="bellman",
        return_fn=user_utility,
        beta=beta,
        options={"max_iter": 50, "n_howard": 10},
    )
    sol = prob.solve()
    assert sol.converged
    assert sol.n_iter < 60


# ===========================================================================
# 5. Functional Entry Point: solve_collocation
# ===========================================================================

def test_solve_collocation_functional_api():
    """Verify solve_collocation wrapper works with problem instance or kwargs."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    # Calling with kwargs
    sol1 = solve_collocation(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )
    assert isinstance(sol1, CollocationSolution)
    assert sol1.converged

    # Calling with problem instance
    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )
    sol2 = solve_collocation(problem=prob)
    assert isinstance(sol2, CollocationSolution)
    assert np.allclose(sol1.coefficients, sol2.coefficients)

    # Calling with both raises ValueError
    with pytest.raises(ValueError, match="Cannot pass both"):
        solve_collocation(problem=prob, domain=(0.1, 1.0))


# ===========================================================================
# 6. Presentation Interface Compliance
# ===========================================================================

def test_presentation_contract_methods():
    """Verify .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot()."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )
    sol = prob.solve()

    # .summary() -> pd.DataFrame
    df_summary = sol.summary()
    assert isinstance(df_summary, pd.DataFrame)
    assert "Converged" in df_summary.index
    assert "Polynomial Orders" in df_summary.index

    # .to_frame() -> pd.DataFrame
    df_frame = sol.to_frame()
    assert isinstance(df_frame, pd.DataFrame)
    assert df_frame.equals(df_summary)

    # .to_markdown() -> str
    md_str = sol.to_markdown()
    assert isinstance(md_str, str)
    assert "|" in md_str
    assert "Converged" in md_str

    # .to_latex() -> str
    latex_str = sol.to_latex()
    assert isinstance(latex_str, str)
    assert r"\begin{tabular}" in latex_str
    assert r"\end{tabular}" in latex_str

    # .to_typst() -> str
    typst_str = sol.to_typst()
    assert isinstance(typst_str, str)
    assert "#table(" in typst_str

    # .plot() -> matplotlib.figure.Figure
    fig = sol.plot()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


# ===========================================================================
# 7. Multi-Backend Acceleration & Fallback Verification
# ===========================================================================

def test_backend_dispatch_and_consistency():
    """Verify consistency across available backends (numpy, numba, mlx)."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    domain = (0.5 * k_ss, 1.5 * k_ss)

    prob = CollocationProblem(
        domain=domain,
        orders=6,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )

    sol_numpy = prob.solve(backend="numpy")
    eval_grid = np.linspace(domain[0], domain[1], 100)
    pol_numpy = sol_numpy.policy(eval_grid)

    # Numba backend if available
    if _bk.backend_available("numba"):
        sol_numba = prob.solve(backend="numba")
        pol_numba = sol_numba.policy(eval_grid)
        assert np.allclose(pol_numpy, pol_numba, atol=1e-6)

    # MLX backend if available
    if _bk.backend_available("mlx"):
        sol_mlx = prob.solve(backend="mlx")
        pol_mlx = sol_mlx.policy(eval_grid)
        # MLX uses float32 on Apple Silicon Metal, verify <= 10^-4 tolerance
        assert np.allclose(pol_numpy, pol_mlx, atol=1e-4)


def test_unavailable_backend_fallback():
    """Verify requesting an uninstalled optional backend (e.g. cupy on mac) falls back to numpy with warning."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )

    if _bk.backend_available("cupy"):
        pytest.skip("cupy is installed")

    with pytest.warns(UserWarning, match="falling back to 'numpy'"):
        sol = prob.solve(backend="cupy")
    assert sol.backend == "numpy"
    assert sol.metadata["backend"] == "numpy"
    assert sol.converged is True


# ===========================================================================
# 8. Edge Cases, 3D Basis & Immutability
# ===========================================================================

def test_scalar_and_vector_inputs():
    """Verify policy() and value() handle scalar floats, 1D arrays, and lists."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )
    sol = prob.solve()

    # Scalar float input
    pol_scalar = sol.policy(0.20)
    assert isinstance(pol_scalar, float)
    val_scalar = sol.value(0.20)
    assert isinstance(val_scalar, float)

    # 1D array input
    pts_1d = np.array([0.15, 0.20, 0.25])
    pol_1d = sol.policy(pts_1d)
    assert isinstance(pol_1d, np.ndarray)
    assert pol_1d.shape == (3,)

    # List input
    pol_list = sol.policy([0.15, 0.25])
    assert isinstance(pol_list, np.ndarray)
    assert pol_list.shape == (2,)


def test_extrapolation_safety():
    """Verify evaluation slightly outside domain is safely handled without crash."""
    alpha = 0.36
    beta = 0.96
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_min, k_max = 0.5 * k_ss, 1.5 * k_ss

    prob = CollocationProblem(
        domain=(k_min, k_max),
        orders=5,
        method="euler",
        params={"alpha": alpha, "delta": 1.0},
        beta=beta,
    )
    sol = prob.solve()

    # Slightly below k_min and above k_max
    out_pts = np.array([k_min - 0.01, k_max + 0.01])
    pol_out = sol.policy(out_pts)
    assert np.all(np.isfinite(pol_out))


def test_basis_3d_tensor():
    """Verify 3D Chebyshev tensor product basis shapes and operations."""
    domain = ((0.0, 1.0), (-1.0, 2.0), (2.0, 4.0))
    orders = (2, 3, 2)
    basis = CollocationBasis(domain=domain, orders=orders)

    assert basis.n_dims == 3
    assert basis.n_nodes == (2 + 1) * (3 + 1) * (2 + 1)  # 36
    nodes = basis.nodes()
    assert nodes.shape == (36, 3)

    phi = basis.evaluate(nodes)
    assert phi.shape == (36, 36)

    # Differentiate along dim 2
    dphi_d2 = basis.derivative(nodes, dim=2)
    assert dphi_d2.shape == (36, 36)


def test_solution_and_problem_immutability():
    """Verify CollocationProblem and CollocationSolution are immutable frozen dataclasses."""
    prob = CollocationProblem(
        domain=(0.1, 2.0),
        orders=4,
        method="euler",
        params={"alpha": 0.36},
    )
    with pytest.raises(Exception):
        prob.beta = 0.5  # frozen dataclass assignment error

    sol = prob.solve()
    with pytest.raises(Exception):
        sol.converged = False  # frozen dataclass assignment error

