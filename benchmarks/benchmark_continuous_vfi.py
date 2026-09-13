"""Automated Performance and Accuracy Benchmark CLI for Continuous Projection VFI.

This benchmark compares three dynamic programming and projection paradigms in puremacro.vfi:
1. Discrete Grid VFI (VFIProblem): Discrete choice Howard-accelerated Bellman iteration.
2. Orthogonal Polynomial Collocation (CollocationProblem): Global spectral Chebyshev
   polynomial projection with exponential error convergence O(c^-N) on smooth problems.
3. Finite Element Method (FEMProblem): Localized C^0 piecewise linear Lagrange hat basis
   with exact kink node placement and Fischer-Burmeister complementarity for borrowing
   constraints (algebraic convergence O(h^2)).

Experiments:
- Exp 1: Smooth Neoclassical Growth (Brock-Mirman 1972 analytical benchmark).
- Exp 2: Borrowing-Constrained Savings Model (k' >= k_bar, Gibbs ringing vs kink placement).
- Exp 3: Stochastic Multi-State Model (Markov TFP shocks + continuous capital).
- Exp 4: Multi-Backend Runtime & Scaling Benchmark (NumPy, Numba, MLX, CuPy).

Measurement Engine:
- Standard library time.perf_counter (warmup + timed repetitions).
- Standard library tracemalloc (differential peak heap memory in KiB).
- Zero external profiling dependencies.

Reporting:
- Formats: text (ASCII banner), markdown, latex, typst, json.
- Output artifacts saved to --output-dir when specified.

Exit Codes:
- 0: Success. All requested experiments completed and assertion gates passed.
- 1: Assertion Gate Failure. A numerical validation gate failed in --strict mode.
- 2: Invalid CLI Usage. Argument parsing error or unrecognized options.
- 3: Runtime / Solver Error. Uncaught exception during solver execution.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
import warnings

# Ensure repository root is on sys.path for direct CLI execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from puremacro import _backend as bk
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.vfi import (
    CollocationBasis,
    CollocationProblem,
    CollocationSolution,
    FEMMesh,
    FEMProblem,
    FEMSolution,
    VFIProblem,
)


# ===========================================================================
# 1. Clean Measurement Engine (Standard Library time & tracemalloc)
# ===========================================================================

def measure_runtime_ms(solver_fn: Callable[[], Any], n_runs: int = 5) -> Tuple[float, float, float]:
    """Measure wall-clock execution time in milliseconds.

    Performs a mandatory untimed warmup run (triggering Numba JIT compilation,
    memory allocations, and caches), followed by ``n_runs`` timed repetitions.

    Returns
    -------
    (min_ms, mean_ms, std_ms) : Tuple[float, float, float]
    """
    n_runs = max(1, int(n_runs))
    # 1. Warmup run
    _ = solver_fn()

    times: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = solver_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_arr = np.asarray(times, dtype=np.float64)
    return float(np.min(times_arr)), float(np.mean(times_arr)), float(np.std(times_arr))


def measure_memory_kib(solver_fn: Callable[[], Any]) -> float:
    """Measure peak differential heap memory allocated during solver execution in KiB.

    Uses standard library ``tracemalloc`` to isolate the exact heap footprint
    consumed during a single solver execution.
    """
    # 1. Warmup run so module imports and static lookups are not attributed
    _ = solver_fn()

    tracemalloc.reset_peak()
    tracemalloc.start()
    _ = solver_fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return float(peak_bytes) / 1024.0


def compute_euler_residual_1d(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    eval_grid: np.ndarray,
    alpha: float = 0.36,
    beta: float = 0.96,
    z: float = 1.0,
    delta: float = 1.0,
) -> np.ndarray:
    """Evaluate continuous Euler equation residuals R(k) on an evaluation grid.

    R(k) = 1 - beta * (u'(c') / u'(c)) * [alpha * z' * (k')^(alpha-1) + (1 - delta)]
    For log utility u'(c) = 1/c.
    """
    k = np.asarray(eval_grid, dtype=np.float64)
    kp = policy_fn(k)
    kpp = policy_fn(kp)

    c = np.maximum(z * (k ** alpha) + (1.0 - delta) * k - kp, 1e-12)
    cp = np.maximum(z * (kp ** alpha) + (1.0 - delta) * kp - kpp, 1e-12)
    fkp = alpha * z * (kp ** (alpha - 1.0)) + (1.0 - delta)

    res = 1.0 - beta * (c / cp) * fkp
    return np.asarray(res, dtype=np.float64)


def compute_policy_errors(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    eval_grid: np.ndarray,
    g_star_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[float, float]:
    """Compute maximum relative policy error and normalized relative L2 policy error.

    Returns
    -------
    (max_rel_error, l2_rel_error) : Tuple[float, float]
    """
    g_eval = np.asarray(policy_fn(eval_grid), dtype=np.float64)
    g_true = np.asarray(g_star_fn(eval_grid), dtype=np.float64)

    rel_diff = np.abs(g_eval - g_true) / np.maximum(g_true, 1e-12)
    max_rel_error = float(np.max(rel_diff))

    l2_num = float(np.sum((g_eval - g_true) ** 2))
    l2_den = float(np.sum(g_true ** 2))
    l2_rel_error = float(np.sqrt(l2_num / max(l2_den, 1e-12)))

    return max_rel_error, l2_rel_error


# ===========================================================================
# 2. Canonical Comparative Experiments
# ===========================================================================

def run_exp1_smooth_neoclassical(
    backend: str = "numpy",
    n_runs: int = 5,
    n_eval: int = 2000,
    quiet: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Experiment 1: Smooth Neoclassical Growth (Brock-Mirman 1972 Analytical Benchmark).

    Compares Collocation (spectral O(c^-N)), FEM Galerkin (polynomial O(h^2)),
    and Discrete Grid VFI (discrete O(h)) on the exact closed-form benchmark:
    alpha=0.36, beta=0.96, delta=1.0, g*(k) = alpha * beta * k^alpha.
    """
    if not quiet:
        print("\n" + "=" * 90)
        print("EXPERIMENT 1: SMOOTH NEOCLASSICAL GROWTH (BROCK-MIRMAN 1972)")
        print("=" * 90)
        print(f"Parameters: alpha=0.36, beta=0.96, delta=1.0 | Backend: {backend}")
        print(f"Benchmarking n_runs={n_runs}, n_eval={n_eval} points...")

    alpha = 0.36
    beta = 0.96
    delta = 1.0
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss, 1.5 * k_ss)
    eval_k = np.linspace(domain[0], domain[1], n_eval)

    def g_star(k: np.ndarray) -> np.ndarray:
        return alpha * beta * (np.asarray(k, dtype=np.float64) ** alpha)

    records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. Chebyshev Collocation (Orders N in 4, 8, 12)
    # -----------------------------------------------------------------------
    collocation_orders = [4, 8, 12]
    coll_errs: List[float] = []

    for N in collocation_orders:
        if not quiet:
            print(f"  -> Running Collocation (orders={N})...")

        def solve_coll(order: int = N) -> CollocationSolution:
            prob = CollocationProblem(
                domain=domain,
                orders=order,
                method="euler",
                params={"alpha": alpha, "delta": delta},
                beta=beta,
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_coll, n_runs=n_runs)
        mem_kib = measure_memory_kib(solve_coll)
        sol = solve_coll()

        res = compute_euler_residual_1d(sol.policy, eval_k, alpha=alpha, beta=beta, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        _, l2_err = compute_policy_errors(sol.policy, eval_k, g_star)
        coll_errs.append(l2_err)

        records.append({
            "Solver": "Collocation",
            "Config": f"N={N}",
            "Runtime (ms)": mean_ms,
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "L2 Policy Err": l2_err,
            "Convergence": "Spectral",
        })

    # -----------------------------------------------------------------------
    # 2. Finite Element Method (Elements E in 20, 50, 80)
    # -----------------------------------------------------------------------
    fem_elements = [20, 50, 80]
    fem_errs: List[float] = []

    for E in fem_elements:
        if not quiet:
            print(f"  -> Running FEM Galerkin (elements={E})...")

        def solve_fem_elem(elem: int = E) -> FEMSolution:
            prob = FEMProblem(
                domain=domain,
                elements=elem,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k: k ** alpha,
                beta=beta,
                params={"alpha": alpha, "gamma": 1.0},
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_fem_elem, n_runs=n_runs)
        mem_kib = measure_memory_kib(solve_fem_elem)
        sol = solve_fem_elem()

        res = compute_euler_residual_1d(sol.policy, eval_k, alpha=alpha, beta=beta, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        _, l2_err = compute_policy_errors(sol.policy, eval_k, g_star)
        fem_errs.append(l2_err)

        records.append({
            "Solver": "FEM Galerkin",
            "Config": f"E={E}",
            "Runtime (ms)": mean_ms,
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "L2 Policy Err": l2_err,
            "Convergence": "O(h^2)",
        })

    # -----------------------------------------------------------------------
    # 3. Discrete Grid VFI (Grid sizes Na in 100, 500, 1000)
    # -----------------------------------------------------------------------
    vfi_grid_sizes = [100, 500, 1000]

    for Na in vfi_grid_sizes:
        if not quiet:
            print(f"  -> Running Discrete Grid VFI (Na={Na})...")

        k_grid = np.linspace(domain[0], domain[1], Na)

        def solve_vfi_grid(n_pts: int = Na, grid: np.ndarray = k_grid) -> Any:
            def rf(kp: Any, k: Any, z: Any, xp: Any = np) -> Any:
                c = xp.maximum(k ** alpha - kp, 1e-12)
                return xp.where(c > 1e-12, xp.log(c), -1e10)

            prob = VFIProblem(
                a_grid=grid,
                z_grid=np.array([1.0]),
                P_z=np.array([[1.0]]),
                return_fn=rf,
                beta=beta,
                options={"tol": 1e-7, "howard": True, "n_howard": 40},
            )
            return prob.solve(backend="numpy")

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_vfi_grid, n_runs=n_runs)
        mem_kib = measure_memory_kib(solve_vfi_grid)
        sol = solve_vfi_grid()

        pol_k = k_grid[sol.policy_aprime[:, 0]]
        policy_fn = lambda k, kg=k_grid, pk=pol_k: np.interp(k, kg, pk)

        res = compute_euler_residual_1d(policy_fn, eval_k, alpha=alpha, beta=beta, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        _, l2_err = compute_policy_errors(policy_fn, eval_k, g_star)

        records.append({
            "Solver": "Discrete VFI",
            "Config": f"Na={Na}",
            "Runtime (ms)": mean_ms,
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "L2 Policy Err": l2_err,
            "Convergence": "O(h)",
        })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Strict Numerical Validation Gates
    # -----------------------------------------------------------------------
    if strict:
        if not quiet:
            print("  [Validation] Verifying numerical assertion gates for Experiment 1...")

        # 1. Collocation N=8 and N=12 achieve Euler residual < 1e-4 and L2 error < 1e-4
        assert df.loc[df["Config"] == "N=8", "Max Euler Res"].values[0] < 1e-4, "Collocation N=8 Max Euler Res >= 1e-4"
        assert df.loc[df["Config"] == "N=8", "L2 Policy Err"].values[0] < 1e-4, "Collocation N=8 L2 Policy Err >= 1e-4"
        assert df.loc[df["Config"] == "N=12", "L2 Policy Err"].values[0] < 1e-6, "Collocation N=12 L2 Policy Err >= 1e-6"

        # 2. FEM E=50 and E=80 achieve Euler residual < 1e-4 and L2 error < 1e-4
        assert df.loc[df["Config"] == "E=50", "Max Euler Res"].values[0] < 1e-4, "FEM E=50 Max Euler Res >= 1e-4"
        assert df.loc[df["Config"] == "E=50", "L2 Policy Err"].values[0] < 1e-4, "FEM E=50 L2 Policy Err >= 1e-4"
        assert df.loc[df["Config"] == "E=80", "L2 Policy Err"].values[0] < 5e-5, "FEM E=80 L2 Policy Err >= 5e-5"

        # 3. Spectral convergence ordering: error drops strictly with N
        assert coll_errs[2] < coll_errs[1] < coll_errs[0], "Collocation errors do not decay monotonically with N"

        # 4. Polynomial convergence ordering: error drops strictly with E
        assert fem_errs[2] < fem_errs[1] < fem_errs[0], "FEM errors do not decay monotonically with E"

        # 5. Collocation N=8 achieves higher accuracy than Discrete VFI Na=1000 in less time
        t_coll_8 = df.loc[df["Config"] == "N=8", "Runtime (ms)"].values[0]
        t_vfi_1000 = df.loc[df["Config"] == "Na=1000", "Runtime (ms)"].values[0]
        err_coll_8 = df.loc[df["Config"] == "N=8", "L2 Policy Err"].values[0]
        err_vfi_1000 = df.loc[df["Config"] == "Na=1000", "L2 Policy Err"].values[0]

        assert err_coll_8 < err_vfi_1000, "Collocation N=8 must have strictly lower error than Discrete VFI Na=1000"
        assert t_coll_8 < t_vfi_1000, "Collocation N=8 must execute strictly faster than Discrete VFI Na=1000"

    return df


def run_exp2_borrowing_constrained(
    backend: str = "numpy",
    n_runs: int = 5,
    n_eval: int = 2000,
    quiet: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Experiment 2: Borrowing-Constrained Savings Model (k' >= k_bar).

    Compares Chebyshev Collocation, FEM Uniform Mesh, FEM Kink-Aligned Mesh,
    and Discrete Grid VFI on a model with an occasionally binding constraint:
    k' >= k_bar, where k_bar = alpha * beta * (k^*)^alpha, k^* = 0.5 * k_ss.

    Tracks:
    - Max Boundary Violation: max(0, k_bar - g(k)).
    - Gibbs Ringing Amplitude: max |g(k) - k_bar| on constrained region k <= k^*.
    - Unconstrained Error: max relative policy error on k > k^*.
    """
    if not quiet:
        print("\n" + "=" * 90)
        print("EXPERIMENT 2: BORROWING-CONSTRAINED SAVINGS MODEL (KINK RESOLUTION)")
        print("=" * 90)
        print(f"Parameters: alpha=0.36, beta=0.96, k_bar at 0.5*k_ss | Backend: {backend}")
        print(f"Benchmarking n_runs={n_runs}, n_eval={n_eval} points...")

    alpha = 0.36
    beta = 0.96
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.1 * k_ss, 2.0 * k_ss)
    k_star = 0.5 * k_ss
    k_bar = float(alpha * beta * (k_star ** alpha))

    eval_k = np.linspace(domain[0], domain[1], n_eval)
    g_true = np.where(eval_k <= k_star, k_bar, alpha * beta * (eval_k ** alpha))
    constr_mask = eval_k <= k_star
    unconstr_mask = eval_k > k_star

    records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. Chebyshev Polynomial Collocation (Orders N in 6, 12, 20, 30)
    # -----------------------------------------------------------------------
    cheb_orders = [6, 12, 20, 30]
    cheb_gibbs_list: List[float] = []

    for N in cheb_orders:
        if not quiet:
            print(f"  -> Running Chebyshev Collocation (orders={N})...")

        basis = CollocationBasis(domain=domain, orders=N)
        nodes = basis.nodes(squeeze=True)
        y_nodes = np.where(nodes <= k_star, k_bar, alpha * beta * (nodes ** alpha))

        def solve_cheb(b=basis, y=y_nodes) -> np.ndarray:
            return b.fit(y)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_cheb, n_runs=n_runs)
        coeffs = solve_cheb()

        g_cheb = np.asarray([basis.interpolate(coeffs, float(k)) for k in eval_k], dtype=np.float64)
        violation = float(np.max(np.maximum(0.0, k_bar - g_cheb)))
        gibbs = float(np.max(np.abs(g_cheb[constr_mask] - k_bar)))
        unconstr_err = float(np.max(np.abs(g_cheb[unconstr_mask] - g_true[unconstr_mask]) / g_true[unconstr_mask]))
        cheb_gibbs_list.append(gibbs)

        records.append({
            "Solver": "Collocation",
            "Config": f"N={N}",
            "Runtime (ms)": mean_ms,
            "Max Violation": violation,
            "Gibbs Ringing": gibbs,
            "Unconstr Error": unconstr_err,
            "Constraint Fidelity": "Severe Gibbs Ringing",
        })

    # -----------------------------------------------------------------------
    # 2. FEM Uniform Mesh (No Kink Node, Elements E in 20, 40, 80)
    # -----------------------------------------------------------------------
    fem_elem_uniform = [20, 40, 80]

    for E in fem_elem_uniform:
        if not quiet:
            print(f"  -> Running FEM Uniform Mesh (elements={E})...")

        def solve_fem_uni(elem: int = E) -> FEMSolution:
            prob = FEMProblem(
                domain=domain,
                elements=elem,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k: k ** alpha,
                beta=beta,
                params={"alpha": alpha, "gamma": 1.0},
                borrowing_constraint=k_bar,
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_fem_uni, n_runs=n_runs)
        sol = solve_fem_uni()
        g_fem_uni = sol.policy(eval_k)

        violation = float(np.max(np.maximum(0.0, k_bar - g_fem_uni)))
        gibbs = float(np.max(np.abs(g_fem_uni[constr_mask] - k_bar)))
        unconstr_err = float(np.max(np.abs(g_fem_uni[unconstr_mask] - g_true[unconstr_mask]) / g_true[unconstr_mask]))

        records.append({
            "Solver": "FEM Uniform",
            "Config": f"E={E}",
            "Runtime (ms)": mean_ms,
            "Max Violation": violation,
            "Gibbs Ringing": gibbs,
            "Unconstr Error": unconstr_err,
            "Constraint Fidelity": "Interpolation Mismatch",
        })

    # -----------------------------------------------------------------------
    # 3. FEM Kink-Aligned Mesh (Elements E in 20, 40, 80)
    # -----------------------------------------------------------------------
    fem_elem_kink = [20, 40, 80]
    fem_kink_violations: List[float] = []
    fem_kink_gibbs: List[float] = []

    for E in fem_elem_kink:
        if not quiet:
            print(f"  -> Running FEM Kink-Aligned Mesh (elements={E})...")

        mesh_kink = FEMMesh.from_kinks(domain, n_elements=E, kinks=[k_star])

        def solve_fem_kink(elem: int = E, m=mesh_kink) -> FEMSolution:
            prob = FEMProblem(
                domain=domain,
                elements=elem,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k: k ** alpha,
                beta=beta,
                params={"alpha": alpha, "gamma": 1.0},
                borrowing_constraint=k_bar,
                options={"mesh": m},
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_fem_kink, n_runs=n_runs)
        sol = solve_fem_kink()
        g_fem_kink = sol.policy(eval_k)

        violation = float(np.max(np.maximum(0.0, k_bar - g_fem_kink)))
        gibbs = float(np.max(np.abs(g_fem_kink[constr_mask] - k_bar)))
        unconstr_err = float(np.max(np.abs(g_fem_kink[unconstr_mask] - g_true[unconstr_mask]) / g_true[unconstr_mask]))
        fem_kink_violations.append(violation)
        fem_kink_gibbs.append(gibbs)

        records.append({
            "Solver": "FEM Kink-Aligned",
            "Config": f"E={E}",
            "Runtime (ms)": mean_ms,
            "Max Violation": violation,
            "Gibbs Ringing": gibbs,
            "Unconstr Error": unconstr_err,
            "Constraint Fidelity": "Exact Kink (Zero Ringing)",
        })

    # -----------------------------------------------------------------------
    # 4. Discrete Grid VFI (Grid sizes Na in 100, 500, 1000)
    # -----------------------------------------------------------------------
    vfi_grid_sizes = [100, 500, 1000]

    for Na in vfi_grid_sizes:
        if not quiet:
            print(f"  -> Running Discrete Grid VFI with constraint (Na={Na})...")

        k_grid = np.linspace(domain[0], domain[1], Na)

        def solve_vfi_constr(n_pts: int = Na, grid: np.ndarray = k_grid) -> Any:
            def rf_constr(kp: Any, k: Any, z: Any, xp: Any = np) -> Any:
                c = xp.maximum(k ** alpha - kp, 1e-12)
                penalty = xp.where(kp < k_bar, -1e10, 0.0)
                return xp.where(c > 1e-12, xp.log(c) + penalty, -1e10)

            prob = VFIProblem(
                a_grid=grid,
                z_grid=np.array([1.0]),
                P_z=np.array([[1.0]]),
                return_fn=rf_constr,
                beta=beta,
                options={"tol": 1e-7, "howard": True, "n_howard": 40},
            )
            return prob.solve(backend="numpy")

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_vfi_constr, n_runs=n_runs)
        sol = solve_vfi_constr()
        pol_k = k_grid[sol.policy_aprime[:, 0]]
        g_vfi = np.interp(eval_k, k_grid, pol_k)

        violation = float(np.max(np.maximum(0.0, k_bar - g_vfi)))
        gibbs = float(np.max(np.abs(g_vfi[constr_mask] - k_bar)))
        unconstr_err = float(np.max(np.abs(g_vfi[unconstr_mask] - g_true[unconstr_mask]) / g_true[unconstr_mask]))

        records.append({
            "Solver": "Discrete VFI",
            "Config": f"Na={Na}",
            "Runtime (ms)": mean_ms,
            "Max Violation": violation,
            "Gibbs Ringing": gibbs,
            "Unconstr Error": unconstr_err,
            "Constraint Fidelity": "Grid Discretized",
        })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Strict Numerical Validation Gates
    # -----------------------------------------------------------------------
    if strict:
        if not quiet:
            print("  [Validation] Verifying numerical assertion gates for Experiment 2...")

        # 1. Chebyshev suffers from persistent Gibbs ringing (> 5e-4) across all N
        for gibbs_val in cheb_gibbs_list:
            assert gibbs_val > 5e-4, f"Chebyshev Gibbs ringing {gibbs_val:.2e} unexpectedly <= 5e-4"

        # 2. FEM Kink-Aligned achieves zero boundary violations (<= 1e-12)
        for viol in fem_kink_violations:
            assert viol <= 1e-12, f"FEM Kink-Aligned boundary violation {viol:.2e} > 1e-12"

        # 3. FEM Kink-Aligned achieves near-zero Gibbs ringing (<= 1e-6) on constrained branch
        for g_kink in fem_kink_gibbs:
            assert g_kink <= 1e-6, f"FEM Kink-Aligned Gibbs ringing {g_kink:.2e} > 1e-6"

        # 4. FEM Kink-Aligned maintains accurate unconstrained branch error (< 1e-3)
        fem_unconstr_err = df.loc[
            (df["Solver"] == "FEM Kink-Aligned") & (df["Config"] == "E=80"), "Unconstr Error"
        ].values[0]
        assert fem_unconstr_err < 1e-3, f"FEM Kink-Aligned unconstrained error {fem_unconstr_err:.2e} >= 1e-3"

    return df


def run_exp3_stochastic_multistate(
    backend: str = "numpy",
    n_runs: int = 5,
    n_eval: int = 2000,
    quiet: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Experiment 3: Stochastic Multi-State Model (Markov Shocks + Continuous Capital).

    Solves neoclassical growth with discrete Markov productivity shocks:
    z in [0.90, 1.00, 1.10] with transition matrix P_z and continuous capital k.
    Analytical policy: g*(k, z_i) = alpha * beta * z_i * k^alpha.

    Tracks:
    - Max Euler Residual per regime.
    - Policy relative error against analytical stochastic truth.
    - Strict monotonic state ordering: g(k, z_high) > g(k, z_mid) > g(k, z_low).
    """
    if not quiet:
        print("\n" + "=" * 90)
        print("EXPERIMENT 3: STOCHASTIC MULTI-STATE MODEL (MARKOV SHOCKS)")
        print("=" * 90)
        print(f"Productivity regimes: z in [0.90, 1.00, 1.10] | Backend: {backend}")
        print(f"Benchmarking n_runs={n_runs}, n_eval={n_eval} points...")

    alpha = 0.36
    beta = 0.96
    delta = 1.0
    z_states = [0.90, 1.00, 1.10]
    P_z = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])

    k_ss_mid = float((alpha * beta * 1.0) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss_mid, 1.5 * k_ss_mid)
    eval_k = np.linspace(domain[0], domain[1], n_eval)

    records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. Continuous Collocation (N=8) across Regimes
    # -----------------------------------------------------------------------
    coll_policies: Dict[float, np.ndarray] = {}
    for z in z_states:
        if not quiet:
            print(f"  -> Running Collocation N=8 for regime z={z:.2f}...")

        def user_euler_z(policy_fn: Any, k: Any, params: Any, z_val: float = z) -> Any:
            kp = policy_fn(k)
            kpp = policy_fn(kp)
            c = np.maximum(z_val * (k ** alpha) - kp, 1e-12)
            cp = np.maximum(z_val * (kp ** alpha) - kpp, 1e-12)
            return 1.0 - beta * (c / cp) * alpha * z_val * (kp ** (alpha - 1.0))

        def solve_coll_z(z_val: float = z, euler_fn=user_euler_z) -> CollocationSolution:
            prob = CollocationProblem(
                domain=domain,
                orders=8,
                method="euler",
                euler_residual_fn=euler_fn,
                params={"alpha": alpha, "z": z_val},
                beta=beta,
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_coll_z, n_runs=n_runs)
        mem_kib = measure_memory_kib(solve_coll_z)
        sol = solve_coll_z()

        g_coll = sol.policy(eval_k)
        coll_policies[z] = g_coll

        res = compute_euler_residual_1d(sol.policy, eval_k, alpha=alpha, beta=beta, z=z, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        g_true_z = lambda k, z_val=z: alpha * beta * z_val * (np.asarray(k, dtype=np.float64) ** alpha)
        rel_err, _ = compute_policy_errors(sol.policy, eval_k, g_true_z)

        records.append({
            "Solver": "Collocation",
            "Regime (z)": f"{z:.2f}",
            "Runtime (ms)": mean_ms,
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "Policy Rel Err": rel_err,
            "Monotonic": True,  # verified jointly below
        })

    # Verify Collocation monotonicity across regimes
    coll_mono = bool(np.all(coll_policies[1.10] > coll_policies[1.00]) and np.all(coll_policies[1.00] > coll_policies[0.90]))
    for r in records[-3:]:
        r["Monotonic"] = coll_mono

    # -----------------------------------------------------------------------
    # 2. Continuous FEM Galerkin (E=50) across Regimes
    # -----------------------------------------------------------------------
    fem_policies: Dict[float, np.ndarray] = {}
    for z in z_states:
        if not quiet:
            print(f"  -> Running FEM Galerkin E=50 for regime z={z:.2f}...")

        def solve_fem_z(z_val: float = z) -> FEMSolution:
            prob = FEMProblem(
                domain=domain,
                elements=50,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(c),
                transition_fn=lambda k, zv=z_val: zv * (k ** alpha),
                beta=beta,
                params={"alpha": alpha, "gamma": 1.0, "z": z_val},
            )
            return prob.solve(backend=backend)

        min_ms, mean_ms, std_ms = measure_runtime_ms(solve_fem_z, n_runs=n_runs)
        mem_kib = measure_memory_kib(solve_fem_z)
        sol = solve_fem_z()

        g_fem = sol.policy(eval_k)
        fem_policies[z] = g_fem

        res = compute_euler_residual_1d(sol.policy, eval_k, alpha=alpha, beta=beta, z=z, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        g_true_z = lambda k, z_val=z: alpha * beta * z_val * (np.asarray(k, dtype=np.float64) ** alpha)
        rel_err, _ = compute_policy_errors(sol.policy, eval_k, g_true_z)

        records.append({
            "Solver": "FEM Galerkin",
            "Regime (z)": f"{z:.2f}",
            "Runtime (ms)": mean_ms,
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "Policy Rel Err": rel_err,
            "Monotonic": True,
        })

    # Verify FEM monotonicity across regimes
    fem_mono = bool(np.all(fem_policies[1.10] > fem_policies[1.00]) and np.all(fem_policies[1.00] > fem_policies[0.90]))
    for r in records[-3:]:
        r["Monotonic"] = fem_mono

    # -----------------------------------------------------------------------
    # 3. Discrete Grid VFI (Na=200, Nz=3 Joint Model)
    # -----------------------------------------------------------------------
    if not quiet:
        print("  -> Running Discrete Grid VFI joint 3-state solve (Na=200, Nz=3)...")

    k_grid = np.linspace(domain[0], domain[1], 200)

    def solve_vfi_3state() -> Any:
        def rf_3state(kp: Any, k: Any, z: Any, xp: Any = np) -> Any:
            c = xp.maximum(z * (k ** alpha) - kp, 1e-12)
            return xp.where(c > 1e-12, xp.log(c), -1e10)

        prob = VFIProblem(
            a_grid=k_grid,
            z_grid=np.asarray(z_states),
            P_z=P_z,
            return_fn=rf_3state,
            beta=beta,
            options={"tol": 1e-7, "howard": True, "n_howard": 30},
        )
        return prob.solve(backend="numpy")

    min_ms, mean_ms, std_ms = measure_runtime_ms(solve_vfi_3state, n_runs=n_runs)
    mem_kib = measure_memory_kib(solve_vfi_3state)
    sol_vfi = solve_vfi_3state()

    vfi_policies: Dict[float, np.ndarray] = {}
    for iz, z in enumerate(z_states):
        pol_z = k_grid[sol_vfi.policy_aprime[:, iz]]
        g_vfi_z = np.interp(eval_k, k_grid, pol_z)
        vfi_policies[z] = g_vfi_z

        policy_fn_z = lambda k, kg=k_grid, pz=pol_z: np.interp(k, kg, pz)
        res = compute_euler_residual_1d(policy_fn_z, eval_k, alpha=alpha, beta=beta, z=z, delta=delta)
        max_euler = float(np.max(np.abs(res)))
        g_true_z = lambda k, z_val=z: alpha * beta * z_val * (np.asarray(k, dtype=np.float64) ** alpha)
        rel_err, _ = compute_policy_errors(policy_fn_z, eval_k, g_true_z)

        records.append({
            "Solver": "Discrete VFI",
            "Regime (z)": f"{z:.2f}",
            "Runtime (ms)": mean_ms / len(z_states),  # amortized solve time
            "Peak Mem (KiB)": mem_kib,
            "Max Euler Res": max_euler,
            "Policy Rel Err": rel_err,
            "Monotonic": True,
        })

    vfi_mono = bool(np.all(vfi_policies[1.10] >= vfi_policies[1.00]) and np.all(vfi_policies[1.00] >= vfi_policies[0.90]))
    for r in records[-3:]:
        r["Monotonic"] = vfi_mono

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Strict Numerical Validation Gates
    # -----------------------------------------------------------------------
    if strict:
        if not quiet:
            print("  [Validation] Verifying numerical assertion gates for Experiment 3...")

        # 1. Collocation Euler residual < 1e-4 and relative policy error < 1e-4 for all z
        coll_df = df[df["Solver"] == "Collocation"]
        assert np.all(coll_df["Max Euler Res"] < 1e-4), "Collocation Max Euler Res >= 1e-4 for stochastic regimes"
        assert np.all(coll_df["Policy Rel Err"] < 1e-4), "Collocation Policy Rel Err >= 1e-4 for stochastic regimes"

        # 2. FEM Euler residual < 1e-4 and relative policy error < 1e-4 for all z
        fem_df = df[df["Solver"] == "FEM Galerkin"]
        assert np.all(fem_df["Max Euler Res"] < 1e-4), "FEM Max Euler Res >= 1e-4 for stochastic regimes"
        assert np.all(fem_df["Policy Rel Err"] < 1e-4), "FEM Policy Rel Err >= 1e-4 for stochastic regimes"

        # 3. Monotonicity preserved strictly across regimes
        assert coll_mono, "Collocation policy failed strict monotonicity across TFP regimes"
        assert fem_mono, "FEM policy failed strict monotonicity across TFP regimes"
        assert vfi_mono, "Discrete VFI policy failed monotonicity across TFP regimes"

    return df


def run_exp4_backend_scaling(
    target_backend: str = "all",
    n_runs: int = 5,
    n_eval: int = 2000,
    quiet: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Experiment 4: Multi-Backend Runtime & Scaling Benchmark.

    Measures runtime speedup, memory, and solution parity against the NumPy reference
    across NumPy, Numba (CPU JIT), MLX (Apple Silicon GPU), and CuPy (CUDA GPU).
    Handles uninstalled backends with zero crashes via defensive graceful fallback.
    """
    if not quiet:
        print("\n" + "=" * 90)
        print("EXPERIMENT 4: MULTI-BACKEND RUNTIME & SCALING BENCHMARK")
        print("=" * 90)
        print(f"Target backend: {target_backend} | Installed backends: {bk.available_backends()}")
        print(f"Benchmarking n_runs={n_runs}, n_eval={n_eval} points...")

    alpha = 0.36
    beta = 0.96
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss, 1.5 * k_ss)
    eval_k = np.linspace(domain[0], domain[1], n_eval)

    backends_to_test = ["numpy", "numba", "mlx", "cupy"] if target_backend == "all" else ["numpy", target_backend]
    # Remove duplicates preserving order
    seen = set()
    backends_to_test = [b for b in backends_to_test if not (b in seen or seen.add(b))]

    records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. Baseline Reference Solve (NumPy)
    # -----------------------------------------------------------------------
    prob_coll_ref = CollocationProblem(domain=domain, orders=8, method="euler", params={"alpha": alpha, "delta": 1.0}, beta=beta)
    sol_coll_ref = prob_coll_ref.solve(backend="numpy")
    pol_coll_ref = sol_coll_ref.policy(eval_k)
    _, t_coll_np, _ = measure_runtime_ms(lambda: prob_coll_ref.solve(backend="numpy"), n_runs=n_runs)

    prob_fem_ref = FEMProblem(
        domain=domain, elements=50, method="euler", projection="galerkin",
        return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** alpha,
        beta=beta, params={"alpha": alpha, "gamma": 1.0},
    )
    sol_fem_ref = prob_fem_ref.solve(backend="numpy")
    pol_fem_ref = sol_fem_ref.policy(eval_k)
    _, t_fem_np, _ = measure_runtime_ms(lambda: prob_fem_ref.solve(backend="numpy"), n_runs=n_runs)

    # -----------------------------------------------------------------------
    # 2. Multi-Backend Benchmark Loop
    # -----------------------------------------------------------------------
    for b in backends_to_test:
        is_avail = bk.backend_available(b)
        status = "Native" if is_avail else "Graceful Fallback"

        # A. Collocation under backend b
        if not quiet:
            print(f"  -> Testing Collocation N=8 on backend '{b}' (Status: {status})...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            def solve_c_b(backend_name: str = b) -> CollocationSolution:
                prob = CollocationProblem(domain=domain, orders=8, method="euler", params={"alpha": alpha, "delta": 1.0}, beta=beta)
                return prob.solve(backend=backend_name)

            min_ms, mean_ms, std_ms = measure_runtime_ms(solve_c_b, n_runs=n_runs)
            mem_kib = measure_memory_kib(solve_c_b)
            sol_c_b = solve_c_b()
            pol_c_b = sol_c_b.policy(eval_k)
            parity_err = float(np.max(np.abs(pol_c_b - pol_coll_ref) / np.maximum(pol_coll_ref, 1e-12)))
            speedup = float(t_coll_np / max(mean_ms, 1e-6))

        records.append({
            "Solver": "Collocation",
            "Backend": b,
            "Status": status,
            "Runtime (ms)": mean_ms,
            "Speedup": speedup,
            "Peak Mem (KiB)": mem_kib,
            "Parity Error": parity_err,
        })

        # B. FEM under backend b
        if not quiet:
            print(f"  -> Testing FEM Galerkin E=50 on backend '{b}' (Status: {status})...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            def solve_f_b(backend_name: str = b) -> FEMSolution:
                prob = FEMProblem(
                    domain=domain, elements=50, method="euler", projection="galerkin",
                    return_fn=lambda c: np.log(c), transition_fn=lambda k: k ** alpha,
                    beta=beta, params={"alpha": alpha, "gamma": 1.0},
                )
                return prob.solve(backend=backend_name)

            min_ms, mean_ms, std_ms = measure_runtime_ms(solve_f_b, n_runs=n_runs)
            mem_kib = measure_memory_kib(solve_f_b)
            sol_f_b = solve_f_b()
            pol_f_b = sol_f_b.policy(eval_k)
            parity_err = float(np.max(np.abs(pol_f_b - pol_fem_ref) / np.maximum(pol_fem_ref, 1e-12)))
            speedup = float(t_fem_np / max(mean_ms, 1e-6))

        records.append({
            "Solver": "FEM Galerkin",
            "Backend": b,
            "Status": status,
            "Runtime (ms)": mean_ms,
            "Speedup": speedup,
            "Peak Mem (KiB)": mem_kib,
            "Parity Error": parity_err,
        })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Strict Numerical Validation Gates
    # -----------------------------------------------------------------------
    if strict:
        if not quiet:
            print("  [Validation] Verifying numerical assertion gates for Experiment 4...")

        # 1. For all tested backends, parity error against NumPy reference < 1e-4
        for _, row in df.iterrows():
            b_name = row["Backend"]
            p_err = row["Parity Error"]
            assert p_err < 1e-4, f"Backend '{b_name}' ({row['Solver']}) parity error {p_err:.2e} >= 1e-4"

        # 2. Unavailable backends handled cleanly without crashing
        fallback_rows = df[df["Status"] == "Graceful Fallback"]
        for _, row in fallback_rows.iterrows():
            assert row["Parity Error"] < 1e-12, f"Fallback backend '{row['Backend']}' differed from NumPy reference"

    return df


# ===========================================================================
# 3. Output Table Rendering & Multi-Format Exporters
# ===========================================================================

def format_cell_value(col: str, val: Any) -> str:
    """Format numeric and string values cleanly for terminal display."""
    if val is None:
        return ""
    if isinstance(val, (bool, np.bool_)):
        return str(bool(val))
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(val):
            return str(val)
        if abs(val) == 0.0:
            if any(term in col for term in ("Res", "Err", "Violation", "Ringing")):
                return "0.00e+00"
            return "0.00"
        if any(term in col for term in ("Res", "Err", "Violation", "Ringing")):
            return f"{val:.2e}"
        if abs(val) < 0.01:
            return f"{val:.2e}"
        return f"{val:.2f}"
    return str(val)


def _df_to_ascii_table(df: pd.DataFrame, title: str) -> str:
    """Render a DataFrame as a structured, human-readable ASCII table."""
    cols = list(df.columns)
    formatted_rows: List[List[str]] = []

    for _, row in df.iterrows():
        formatted_rows.append([format_cell_value(col, row[col]) for col in cols])

    # Calculate column widths
    col_widths: List[int] = []
    for i, col in enumerate(cols):
        max_len = len(str(col))
        for row in formatted_rows:
            max_len = max(max_len, len(row[i]))
        col_widths.append(max_len)

    total_width = max(len(title), sum(col_widths) + 3 * (len(cols) - 1) + 2)
    total_width = max(total_width, 80)

    lines: List[str] = []
    lines.append("=" * total_width)
    lines.append(title)
    lines.append("=" * total_width)

    # Header
    header_parts = []
    for i, col in enumerate(cols):
        # Align left for Solver, Config, Backend, Status; right for numbers
        if i < 2 or col in ("Backend", "Status", "Regime (z)", "Convergence", "Constraint Fidelity"):
            header_parts.append(str(col).ljust(col_widths[i]))
        else:
            header_parts.append(str(col).rjust(col_widths[i]))
    lines.append("   ".join(header_parts))
    lines.append("-" * total_width)

    # Rows
    for row in formatted_rows:
        row_parts = []
        for i, val in enumerate(row):
            col = cols[i]
            if i < 2 or col in ("Backend", "Status", "Regime (z)", "Convergence", "Constraint Fidelity"):
                row_parts.append(val.ljust(col_widths[i]))
            else:
                row_parts.append(val.rjust(col_widths[i]))
        lines.append("   ".join(row_parts))

    lines.append("=" * total_width)
    return "\n".join(lines)


def render_experiment_tables(
    exp_key: str,
    df: pd.DataFrame,
    title: str,
    fmt: str,
    output_dir: Optional[Path] = None,
) -> None:
    """Format and output experiment results across requested formats."""
    # 1. ASCII Text Table
    text_table = _df_to_ascii_table(df, title)

    # 2. Markdown Table
    md_table = _df_to_markdown(df, index=False)

    # 3. LaTeX Table
    latex_table = _df_to_latex(df, index=False)

    # 4. Typst Table
    typst_table = _df_to_typst(df, index=False)

    # 5. JSON Table
    json_table = df.to_json(orient="records", indent=2)

    # Print to stdout based on --format
    if fmt == "text":
        print(text_table)
    elif fmt == "markdown":
        print(f"\n### {title}\n\n" + md_table)
    elif fmt == "latex":
        print(f"\n% {title}\n" + latex_table)
    elif fmt == "typst":
        print(f"\n// {title}\n" + typst_table)
    elif fmt == "json":
        print(json_table)
    elif fmt == "all":
        # Standard presentation: ASCII banner table followed by Markdown
        print(text_table)
        print("\nMarkdown Table:\n" + md_table)

    # Save to --output-dir if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fmt in ("text", "all"):
            (output_dir / f"{exp_key}_table.txt").write_text(text_table + "\n", encoding="utf-8")
        if fmt in ("markdown", "all"):
            (output_dir / f"{exp_key}_table.md").write_text(f"### {title}\n\n" + md_table + "\n", encoding="utf-8")
        if fmt in ("latex", "all"):
            (output_dir / f"{exp_key}_table.tex").write_text(f"% {title}\n" + latex_table + "\n", encoding="utf-8")
        if fmt in ("typst", "all"):
            (output_dir / f"{exp_key}_table.typ").write_text(f"// {title}\n" + typst_table + "\n", encoding="utf-8")
        if fmt in ("json", "all"):
            (output_dir / f"{exp_key}_table.json").write_text(json_table + "\n", encoding="utf-8")


# ===========================================================================
# 4. Custom ArgumentParser with Strict Exit Code 2 on Usage Errors
# ===========================================================================

class BenchmarkArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser guaranteeing exit code 2 on invalid CLI options."""

    def error(self, message: str) -> None:
        sys.stderr.write(f"benchmark_continuous_vfi.py error: {message}\n")
        self.print_usage(sys.stderr)
        sys.exit(2)


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = BenchmarkArgumentParser(
        prog="benchmark_continuous_vfi.py",
        description="Comprehensive Comparative Benchmark CLI for Continuous Projection VFI Solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        choices=["all", "exp1", "exp2", "exp3", "exp4", "smooth", "kink", "stochastic", "backend"],
        default="all",
        help="Which comparative benchmark experiment(s) to execute.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["all", "numpy", "numba", "mlx", "cupy"],
        default="all",
        help="Target compute acceleration backend.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["all", "text", "markdown", "latex", "typst", "json"],
        default="all",
        help="Output table formatting.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Optional directory path where output table files will be saved.",
    )
    parser.add_argument(
        "--n-runs",
        "-r",
        type=int,
        default=5,
        help="Number of post-warmup repetitions for statistical runtime stability.",
    )
    parser.add_argument(
        "--n-eval",
        "-m",
        type=int,
        default=2000,
        help="Number of dense continuous evaluation grid points.",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        default=False,
        help="Enforce strict numerical assertion gates.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress progress logs and print only final formatted tables.",
    )

    args = parser.parse_args(argv)

    if args.n_runs < 1:
        parser.error("--n-runs must be an integer >= 1")
    if args.n_eval < 2:
        parser.error("--n-eval must be an integer >= 2")

    return args


# ===========================================================================
# 5. CLI Entry Point
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main execution entry point for benchmark_continuous_vfi.py.

    Exit Codes:
    - 0: Success. All requested benchmarks succeeded and assertions passed.
    - 1: Assertion Gate Failure. A numerical tolerance failed in strict mode.
    - 2: Invalid CLI Usage. Handled automatically by BenchmarkArgumentParser.
    - 3: Runtime / Solver Error. Uncaught computational exception.
    """
    args = parse_cli_args(argv)

    # Map aliases to standard experiment keys
    alias_map = {
        "smooth": "exp1",
        "kink": "exp2",
        "stochastic": "exp3",
        "backend": "exp4",
    }
    exp_target = alias_map.get(args.experiment, args.experiment)
    is_quiet = bool(args.quiet or (args.format == "json"))

    experiments_to_run: List[Tuple[str, str, Callable[[], pd.DataFrame]]] = []

    # Prepare experiment runners
    default_backend = "numpy" if args.backend == "all" else args.backend

    if exp_target in ("all", "exp1"):
        experiments_to_run.append((
            "exp1",
            "EXPERIMENT 1: SMOOTH NEOCLASSICAL GROWTH (BROCK-MIRMAN 1972)",
            lambda: run_exp1_smooth_neoclassical(
                backend=default_backend,
                n_runs=args.n_runs,
                n_eval=args.n_eval,
                quiet=is_quiet,
                strict=args.strict,
            ),
        ))

    if exp_target in ("all", "exp2"):
        experiments_to_run.append((
            "exp2",
            "EXPERIMENT 2: BORROWING-CONSTRAINED SAVINGS MODEL (KINK RESOLUTION)",
            lambda: run_exp2_borrowing_constrained(
                backend=default_backend,
                n_runs=args.n_runs,
                n_eval=args.n_eval,
                quiet=is_quiet,
                strict=args.strict,
            ),
        ))

    if exp_target in ("all", "exp3"):
        experiments_to_run.append((
            "exp3",
            "EXPERIMENT 3: STOCHASTIC MULTI-STATE MODEL (MARKOV SHOCKS)",
            lambda: run_exp3_stochastic_multistate(
                backend=default_backend,
                n_runs=args.n_runs,
                n_eval=args.n_eval,
                quiet=is_quiet,
                strict=args.strict,
            ),
        ))

    if exp_target in ("all", "exp4"):
        experiments_to_run.append((
            "exp4",
            "EXPERIMENT 4: MULTI-BACKEND RUNTIME & SCALING BENCHMARK",
            lambda: run_exp4_backend_scaling(
                target_backend=args.backend,
                n_runs=args.n_runs,
                n_eval=args.n_eval,
                quiet=is_quiet,
                strict=args.strict,
            ),
        ))

    if not is_quiet:
        print("puremacro Continuous Projection VFI Benchmark Suite")
        print(f"Executing {len(experiments_to_run)} experiment(s) [strict={args.strict}, backend={args.backend}]")

    # Run experiments with structured error handling
    for exp_key, title, runner in experiments_to_run:
        try:
            df = runner()
            render_experiment_tables(
                exp_key=exp_key,
                df=df,
                title=title,
                fmt=args.format,
                output_dir=args.output_dir,
            )
        except AssertionError as ae:
            sys.stderr.write(f"\n[STRICT ASSERTION FAILURE] in {exp_key}: {ae}\n")
            sys.exit(1)
        except Exception as exc:
            sys.stderr.write(f"\n[SOLVER RUNTIME ERROR] in {exp_key}: {exc}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(3)

    if not args.quiet and args.output_dir is not None:
        print(f"\nAll benchmark tables successfully saved to: {Path(args.output_dir).resolve()}")

    sys.exit(0)


if __name__ == "__main__":
    main()
