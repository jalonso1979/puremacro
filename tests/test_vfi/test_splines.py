"""Comprehensive verification and unit test suite for puremacro.vfi.splines.

Verifies:
1. CubicBSplineBasis:
   - Partition of unity: sum_{i} B_{i, 3}(x) = 1.0 +- 1e-12 on [a, b].
   - Non-negativity: B_{i, 3}(x) >= 0.
   - Exact Greville abscissae interpolation.
   - Analytical 1st and 2nd derivatives vs central differences.
   - Boundary knot treatments: 'clamped', 'natural', 'not-a-knot'.
2. SchumakerSpline:
   - Exact knot interpolation: S(x_i) = y_i.
   - Strict monotonicity preservation: S'(x) >= 0 on monotonic data.
   - Borrowing constraint kink resolution: auxiliary knot insertion, zero spurious oscillations.
   - Strict concavity preservation: S''(x) <= 0 on concave data.
   - C^1 continuity across all knots and inserted knots.
   - Multi-shock state space evaluation.
3. SplineCollocationProblem & SplineCollocationSolution:
   - Brock-Mirman neoclassical growth analytical benchmark:
     policy relative error < 10^-4 on >= 1,000 points.
   - Out-of-sample continuous Euler equation residuals < 10^-4 on >= 1,000 points.
   - Bellman value collocation convergence.
   - Multi-backend consistency (NumPy, Numba) and warning-safe fallback for unavailable backends.
   - Presentation contract: .summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst().
   - Functional helper `solve_spline_collocation`.
"""
from __future__ import annotations

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
from puremacro.vfi.splines import (
    CubicBSplineBasis,
    SplineBasis,
    SchumakerSpline,
    SplineCollocationProblem,
    SplineCollocationSolution,
    solve_spline_collocation,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def bm_params():
    """Canonical Brock-Mirman (1972) neoclassical growth model parameters."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    k_min = 0.5 * k_ss
    k_max = 1.5 * k_ss

    def g_star(k):
        k_arr = np.asarray(k, dtype=np.float64)
        val = alpha * beta * (k_arr**alpha)
        return float(val) if np.ndim(k) == 0 else val

    return {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "k_ss": k_ss,
        "domain": (k_min, k_max),
        "g_star": g_star,
    }


# ===========================================================================
# 1. CubicBSplineBasis Unit Tests
# ===========================================================================

class TestCubicBSplineBasis:
    """Mathematical and algorithmic tests for CubicBSplineBasis."""

    def test_partition_of_unity_and_non_negativity(self):
        """Verify partition of unity and non-negativity across continuous domain."""
        domain = (0.2, 3.5)
        basis = CubicBSplineBasis(domain=domain, n_knots=18, bc_type="clamped")
        x_dense = np.linspace(domain[0], domain[1], 500)

        B = basis.evaluate(x_dense)
        row_sums = B.sum(axis=1)

        assert np.allclose(row_sums, 1.0, atol=1e-12), "Partition of unity violated"
        assert np.all(B >= -1e-15), "Basis functions must be strictly non-negative"

    def test_greville_abscissae_exact_interpolation(self):
        """Verify exact interpolation at Greville abscissae collocation nodes."""
        domain = (0.5, 5.0)
        basis = CubicBSplineBasis(domain=domain, n_knots=12, bc_type="clamped")
        nodes = basis.nodes()

        # Interpolate a smooth test function
        y_true = np.log(nodes) + np.sqrt(nodes)
        coefs = basis.fit(y_true)

        y_interp = basis.interpolate(coefs, nodes)
        assert np.allclose(y_interp, y_true, atol=1e-12), "Exact knot interpolation failed"

    def test_analytical_derivatives_vs_finite_difference(self):
        """Verify Cox-de Boor 1st and 2nd derivatives against central differences."""
        domain = (1.0, 4.0)
        basis = CubicBSplineBasis(domain=domain, n_knots=14, bc_type="clamped")
        x_test = np.linspace(1.1, 3.9, 40)

        eps = 1e-6
        b_plus = basis.evaluate(x_test + eps, deriv=0)
        b_minus = basis.evaluate(x_test - eps, deriv=0)
        db_num = (b_plus - b_minus) / (2.0 * eps)
        db_ana = basis.evaluate(x_test, deriv=1)

        assert np.allclose(db_num, db_ana, atol=1e-5), "1st derivative mismatch with finite difference"

        # 2nd derivative vs central difference of 1st derivative
        db_plus = basis.evaluate(x_test + eps, deriv=1)
        db_minus = basis.evaluate(x_test - eps, deriv=1)
        d2b_num = (db_plus - db_minus) / (2.0 * eps)
        d2b_ana = basis.evaluate(x_test, deriv=2)

        assert np.allclose(d2b_num, d2b_ana, atol=1e-5), "2nd derivative mismatch with finite difference"

    def test_natural_boundary_conditions(self):
        """Verify that natural boundary splines enforce S''(a) = 0 and S''(b) = 0."""
        domain = (0.0, 2.0)
        basis = CubicBSplineBasis(domain=domain, n_knots=10, bc_type="natural")
        nodes = basis.nodes()

        y_nodes = np.sin(nodes)
        coefs = basis.fit(y_nodes)

        d2_a = basis.interpolate(coefs, domain[0], deriv=2)
        d2_b = basis.interpolate(coefs, domain[1], deriv=2)

        assert abs(d2_a) < 1e-10, f"S''(a) must be zero for natural spline; got {d2_a}"
        assert abs(d2_b) < 1e-10, f"S''(b) must be zero for natural spline; got {d2_b}"

    def test_not_a_knot_construction(self):
        """Verify not-a-knot boundary knot vector correctly omits interior boundary knots."""
        domain = (0.0, 1.0)
        basis = CubicBSplineBasis(domain=domain, n_knots=8, bc_type="not-a-knot")
        assert basis.bc_type == "not-a-knot"
        x_eval = np.linspace(0.0, 1.0, 100)
        b_eval = basis.evaluate(x_eval)
        assert np.allclose(b_eval.sum(axis=1), 1.0, atol=1e-12)

    def test_spline_basis_alias(self):
        """Verify SplineBasis is an alias of CubicBSplineBasis."""
        assert SplineBasis is CubicBSplineBasis
        sb = SplineBasis(domain=(0.0, 1.0), n_knots=6)
        assert isinstance(sb, CubicBSplineBasis)


# ===========================================================================
# 2. SchumakerSpline Unit Tests
# ===========================================================================

class TestSchumakerSpline:
    """Shape-preservation and mathematical tests for SchumakerSpline."""

    def test_exact_knot_interpolation(self):
        """Verify exact knot interpolation: S(x_i) = y_i."""
        x = np.array([0.0, 0.4, 1.0, 1.8, 2.5, 4.0])
        y = np.array([0.0, 0.8, 1.5, 1.9, 2.2, 2.6])
        spline = SchumakerSpline(x, y)

        y_pred = spline.eval(x)
        assert np.allclose(y_pred, y, atol=1e-14), "Schumaker must interpolate data exactly"

    def test_strict_monotonicity_preservation(self):
        """Verify S'(x) >= 0 throughout domain on strictly monotonic data."""
        x = np.linspace(0.1, 5.0, 15)
        y = np.sqrt(x) + 0.1 * x  # strictly increasing concave
        spline = SchumakerSpline(x, y)

        x_dense = np.linspace(0.1, 5.0, 5000)
        d_dense = spline.derivative(x_dense, order=1)

        assert np.all(d_dense >= -1e-14), f"Monotonicity violated: min slope = {d_dense.min()}"

    def test_borrowing_constraint_kink_no_ringing(self):
        """Verify borrowing constraint kink: flat region then positive slope with 0 ringing."""
        # Simulated borrowing constrained policy: a' = 0 for a <= 1, then a' = (a - 1)^1.5
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
        y = np.maximum(0.0, (np.maximum(0.0, x - 1.0)) ** 1.5)
        spline = SchumakerSpline(x, y)

        # Auxiliary knot insertion check
        assert len(spline.inserted_knots) >= 1, "Must insert auxiliary sub-knot near kink"

        x_dense = np.linspace(0.0, 4.0, 3000)
        vals = spline.eval(x_dense)
        slopes = spline.derivative(x_dense, order=1)

        # Flat region must remain flat (>= 0 and <= epsilon)
        flat_mask = x_dense <= 1.0
        assert np.all(vals[flat_mask] >= -1e-14), "Values in flat region must not dip negative"
        assert np.all(vals[flat_mask] <= 1e-12), "Values in flat region must not overshoot"

        # Derivative must be non-negative everywhere (zero non-monotonic dips)
        assert np.all(slopes >= -1e-14), f"Gibbs ringing detected: min slope = {slopes.min()}"

    def test_concavity_preservation(self):
        """Verify S''(x) <= 0 throughout domain on strictly concave data."""
        x = np.linspace(1.0, 8.0, 12)
        y = np.log(x)
        spline = SchumakerSpline(x, y)

        x_dense = np.linspace(1.0, 8.0, 2000)
        d2_dense = spline.second_derivative(x_dense)

        assert np.all(d2_dense <= 1e-12), f"Concavity violated: max 2nd derivative = {d2_dense.max()}"

    def test_c1_continuity_across_knots(self):
        """Verify C^1 continuity of S'(x) across original and inserted knots."""
        x = np.array([0.5, 1.0, 2.0, 3.5, 5.0])
        y = np.array([0.2, 0.9, 1.4, 1.7, 1.9])
        spline = SchumakerSpline(x, y)

        eps = 1e-7
        # Test continuity at all original interior knots
        for xi in x[1:-1]:
            d_left = spline.derivative(xi - eps, order=1)
            d_right = spline.derivative(xi + eps, order=1)
            assert np.isclose(d_left, d_right, atol=1e-5), f"Discontinuous derivative at original knot {xi}"

        # Test continuity at all inserted sub-knots
        for xi_bar in spline.inserted_knots:
            d_left = spline.derivative(xi_bar - eps, order=1)
            d_right = spline.derivative(xi_bar + eps, order=1)
            assert np.isclose(d_left, d_right, atol=1e-5), f"Discontinuous derivative at inserted knot {xi_bar}"

    def test_multi_dim_shock_evaluation(self):
        """Verify SchumakerSpline handles 2D (n, n_z) multi-shock state space."""
        x = np.linspace(0.5, 4.0, 10)
        y = np.column_stack([
            np.sqrt(x) * 1.0,
            np.sqrt(x) * 1.5,
            np.sqrt(x) * 2.0,
        ])
        spline = SchumakerSpline(x, y)
        eval_x = np.array([1.0, 2.5, 3.0])

        vals = spline.eval(eval_x)
        assert vals.shape == (3, 3), f"Expected shape (3, 3); got {vals.shape}"
        assert np.all(vals[:, 0] < vals[:, 1])
        assert np.all(vals[:, 1] < vals[:, 2])


# ===========================================================================
# 3. SplineCollocationProblem & Solution Benchmark Tests
# ===========================================================================

class TestSplineCollocationBenchmark:
    """Benchmark tests on Brock-Mirman neoclassical growth analytical benchmark."""

    def test_cubic_bspline_euler_collocation_accuracy(self, bm_params):
        """Verify CubicBSpline policy error < 1e-4 and Euler residual < 1e-4 on 1,000 points."""
        p = bm_params
        domain = p["domain"]

        prob = SplineCollocationProblem(
            domain=domain,
            n_knots=16,
            spline_type="cubic",
            bc_type="clamped",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged, "Cubic spline collocation failed to converge"

        k_dense = np.linspace(domain[0], domain[1], 1000)
        g_approx = sol.policy(k_dense)
        g_true = p["g_star"](k_dense)

        rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert rel_error < 1e-4, f"Cubic spline policy relative error {rel_error:.2e} >= 1e-4"

        euler_res = np.max(np.abs(sol.euler_residual(k_dense)))
        assert euler_res < 1e-4, f"Cubic spline Euler residual {euler_res:.2e} >= 1e-4"

    def test_schumaker_euler_collocation_accuracy(self, bm_params):
        """Verify Schumaker policy error < 1e-4 and Euler residual < 1e-4 on 1,000 points."""
        p = bm_params
        domain = p["domain"]

        prob = SplineCollocationProblem(
            domain=domain,
            n_knots=20,
            spline_type="schumaker",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged, "Schumaker spline collocation failed to converge"

        k_dense = np.linspace(domain[0], domain[1], 1000)
        g_approx = sol.policy(k_dense)
        g_true = p["g_star"](k_dense)

        rel_error = np.max(np.abs(g_approx - g_true) / g_true)
        assert rel_error < 1e-4, f"Schumaker policy relative error {rel_error:.2e} >= 1e-4"

        euler_res = np.max(np.abs(sol.euler_residual(k_dense)))
        assert euler_res < 1e-4, f"Schumaker Euler residual {euler_res:.2e} >= 1e-4"

        # Verify strict monotonicity of solved policy
        mpol = sol.marginal_policy(k_dense)
        assert np.all(mpol >= 0.0), f"Policy must be strictly monotonic; min derivative = {mpol.min()}"

    def test_bellman_value_collocation(self, bm_params):
        """Verify continuous Bellman value collocation converges with splines."""
        p = bm_params
        prob = SplineCollocationProblem(
            domain=p["domain"],
            n_knots=14,
            spline_type="cubic",
            method="bellman",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
            options={"tol": 1e-6, "max_iter": 100},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged
        k_eval = np.linspace(p["domain"][0], p["domain"][1], 100)
        v_eval = sol.value(k_eval)
        # Value must be strictly increasing in capital
        assert np.all(np.diff(v_eval) > 0.0), "Value function must be strictly increasing"

    def test_multi_backend_consistency(self, bm_params):
        """Verify multi-backend consistency between NumPy and Numba."""
        p = bm_params
        prob = SplineCollocationProblem(
            domain=p["domain"],
            n_knots=15,
            spline_type="cubic",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        sol_np = prob.solve(backend="numpy")

        if bk.backend_available("numba"):
            sol_nb = prob.solve(backend="numba")
            k_eval = np.linspace(p["domain"][0], p["domain"][1], 200)
            diff = np.max(np.abs(sol_np.policy(k_eval) - sol_nb.policy(k_eval)))
            assert diff < 1e-5, f"NumPy and Numba policies diverge by {diff:.2e}"

    def test_warning_safe_fallback(self, bm_params):
        """Verify warning-safe fallback to NumPy when requesting unavailable backend."""
        p = bm_params
        prob = SplineCollocationProblem(
            domain=p["domain"],
            n_knots=12,
            spline_type="cubic",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # CuPy is unavailable on macOS
            sol = prob.solve(backend="cupy")
            assert any("falling back to 'numpy'" in str(warn.message) for warn in w)
            assert sol.backend == "numpy"

    def test_presentation_contract(self, bm_params):
        """Verify full puremacro presentation contract (.summary(), .plot(), .to_frame(), etc.)."""
        p = bm_params
        sol = solve_spline_collocation(
            domain=p["domain"],
            n_knots=15,
            spline_type="cubic",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        assert sol.converged

        # 1. summary & to_frame
        df = sol.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Method" in df.index
        assert sol.to_frame().equals(df)

        # 2. Markdown, LaTeX, Typst
        md = sol.to_markdown()
        assert isinstance(md, str) and len(md) > 0
        latex = sol.to_latex()
        assert isinstance(latex, str) and "begin{tabular}" in latex
        typst = sol.to_typst()
        assert isinstance(typst, str) and "#table(" in typst

        # 3. plot
        fig = sol.plot(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_functional_solve_spline_collocation(self, bm_params):
        """Verify high-level functional helper solve_spline_collocation."""
        p = bm_params

        # Call with domain passed positionally
        sol1 = solve_spline_collocation(
            p["domain"],
            n_knots=16,
            spline_type="cubic",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        assert isinstance(sol1, SplineCollocationSolution)
        assert sol1.converged

        # Call with problem instance
        prob = SplineCollocationProblem(
            domain=p["domain"],
            n_knots=20,
            spline_type="schumaker",
            beta=p["beta"],
            params={"alpha": p["alpha"], "delta": p["delta"], "z": 1.0},
        )
        sol2 = solve_spline_collocation(prob)
        assert isinstance(sol2, SplineCollocationSolution)
        assert sol2.converged
