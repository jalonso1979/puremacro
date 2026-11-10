"""Model adapters for the shared finite-equilibrium residual interface.

The initial adapters cover built-in Chebyshev Euler growth models and canonical
continuous Aiyagari EGM equilibria with fixed asset grids and income chains.
They snapshot the supplied state and never polish or re-solve it implicitly.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from puremacro.core.equilibrium import EquilibriumLinearization, EquilibriumSystem, linearize_equilibrium

__all__ = ["linearize_projection", "linearize_aiyagari"]


def _parameter_names(parameters: Sequence[str], allowed: set[str]) -> tuple[str, ...]:
    names = tuple(parameters)
    if not names or len(set(names)) != len(names):
        raise ValueError("parameters must be nonempty and unique")
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"unsupported parameters: {sorted(unknown)}; supported: {sorted(allowed)}")
    return names


def linearize_projection(
    solution: Any, problem: Any, *, parameters: Sequence[str] = ("alpha", "beta"),
    **options: Any,
) -> EquilibriumLinearization:
    """Linearize built-in one-dimensional Chebyshev Euler growth equations.

    The basis/domain remain fixed. Custom Euler/return/transition callbacks,
    Bellman projection, FEM and splines are outside this first adapter. Use
    ``EquilibriumSystem`` for explicit alternative residuals. Physical clipping
    or continuation-policy extrapolation at the supplied point is refused.
    """
    from puremacro.vfi.collocation import CollocationProblem, CollocationSolution, _evaluate_euler_residual

    if not isinstance(problem, CollocationProblem) or not isinstance(solution, CollocationSolution):
        raise TypeError("linearize_projection requires a CollocationProblem and CollocationSolution")
    if problem.method != "euler" or solution.basis.n_dims != 1:
        raise ValueError("only one-dimensional Euler projection is supported")
    if any(fn is not None for fn in (problem.euler_residual_fn, problem.return_fn, problem.transition_fn)):
        raise ValueError("custom model callbacks need an explicit EquilibriumSystem parameter contract")
    if not solution.converged:
        raise ValueError("the supplied projection solution is not converged")
    prob, sol = deepcopy(problem), deepcopy(solution)
    allowed = {"beta"} | (set(prob.params) & {"alpha", "delta", "sigma", "gamma", "z", "A"})
    if "sigma" in prob.params:
        allowed.discard("gamma")
    if "z" in prob.params:
        allowed.discard("A")
    names = _parameter_names(parameters, allowed)
    base = np.array([prob.beta if name == "beta" else prob.params[name] for name in names], dtype=float)
    coefficients = np.asarray(sol.coefficients, dtype=float).copy()
    nodes = sol.basis.nodes()
    k = nodes[:, 0]
    lower, upper = sol.basis.domain[0]

    def varied(theta: np.ndarray) -> Any:
        values, beta = dict(prob.params), prob.beta
        for name, value in zip(names, theta):
            if name == "beta":
                beta = float(value)
            else:
                values[name] = float(value)
        return replace(prob, params=values, beta=beta)

    def regimes(x: np.ndarray, theta: np.ndarray) -> bytes:
        current = varied(theta)
        alpha = current.params.get("alpha", 0.36)
        delta = current.params.get("delta", 1.0)
        z = current.params.get("z", current.params.get("A", 1.0))
        kp = sol.basis.interpolate(x, k, backend="numpy")
        resources = z * k ** alpha + (1 - delta) * k
        safe_kp = np.clip(kp, 1e-8, resources - 1e-8)
        kpp = sol.basis.interpolate(x, safe_kp, backend="numpy")
        resources_next = z * safe_kp ** alpha + (1 - delta) * safe_kp
        return np.concatenate((kp <= 1e-8, kp >= resources - 1e-8,
                               kp <= lower, kp >= upper, kpp <= 1e-8,
                               kpp >= resources_next - 1e-8)).tobytes()

    if any(regimes(coefficients, base)):
        raise ValueError("projection point uses clipping or a continuation policy outside the basis domain")

    def residual(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return _evaluate_euler_residual(varied(theta), sol.basis, x, nodes, backend="numpy")

    system = EquilibriumSystem(
        residual, tuple(f"coefficient[{j}]" for j in range(len(coefficients))), names,
        state_scale=np.full(len(coefficients), max(float(np.max(np.abs(coefficients))), 0.1)),
        parameter_scale=np.maximum(np.abs(base), 0.1), regime=regimes,
        equation_names=tuple(f"euler[{j}]" for j in range(len(coefficients))),
        label="Chebyshev Euler growth model (fixed basis)",
    )
    return linearize_equilibrium(system, coefficients, base, **options)


def linearize_aiyagari(
    solution: Any, *, parameters: Sequence[str] = ("alpha", "beta"),
    **options: Any,
) -> EquilibriumLinearization:
    """Linearize the joint canonical continuous-Aiyagari equilibrium.

    State order is consumption ``consumption[i,z]``, histogram probability
    ``mass[i,z]``, then ``interest_rate``. The fixed income chain is preserved,
    including custom chains; supported parameters are beta, gamma, alpha and
    delta. Grid, income-process and generic ``continuous_stationary_equilibrium``
    derivatives are unsupported. The residual uses a local active-set extension
    of the borrowing constraint; shrinking stencils check that set and the
    interpolation cells. Dense state limits apply before derivative assembly.
    """
    from puremacro.vfi.continuous_distribution import (
        AiyagariContinuousEquilibrium, build_continuous_transition_matrix, young_lottery_weights,
    )
    from puremacro.vfi.discretize import markov_stationary

    if not isinstance(solution, AiyagariContinuousEquilibrium):
        raise TypeError("linearize_aiyagari requires an AiyagariContinuousEquilibrium")
    if solution.metadata.get("solver_type") != "ContinuousEGM":
        raise ValueError("only solve_aiyagari_continuous canonical EGM equilibria are supported")
    if not solution.converged:
        raise ValueError("the supplied Aiyagari equilibrium is not converged")
    params = dict(solution.metadata["params"])
    names = _parameter_names(parameters, {"alpha", "beta", "gamma", "delta"})
    base = np.array([params[name] for name in names], dtype=float)
    hh = solution.household_solution
    consumption = np.asarray(hh.policy_c, dtype=float).copy()
    probability = np.asarray(solution.distribution.pdf, dtype=float).copy()
    histogram = np.asarray(solution.distribution.asset_grid, dtype=float).copy()
    z = np.asarray(hh.z_grid, dtype=float).copy()
    transition = np.asarray(hh.P_z, dtype=float).copy()
    na, nz = consumption.shape
    nk = len(histogram)
    if probability.shape != (nk, nz) or np.any(probability < -1e-12):
        raise ValueError("invalid stationary probability array")
    if not np.allclose(hh.a_grid, histogram, rtol=0, atol=0):
        raise ValueError("household and distribution histogram grids differ")
    if not np.allclose(histogram, np.linspace(0, params["a_max"], nk), rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied histogram grid differs from the canonical solver grid")
    dense_grid = float(params["a_max"]) * np.linspace(0.0, 1.0, na) ** 1.5
    labor = float(markov_stationary(transition) @ z)
    if not np.isclose(labor, solution.L, rtol=1e-10, atol=1e-12):
        raise ValueError("aggregate labor is inconsistent with the supplied income chain")
    nc, nm = consumption.size, probability.size
    point = np.r_[consumption.ravel(), probability.ravel(), solution.r]
    state_names = (tuple(f"consumption[{i},{j}]" for i in range(na) for j in range(nz))
                   + tuple(f"mass[{i},{j}]" for i in range(nk) for j in range(nz))
                   + ("interest_rate",))

    def components(x: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, bytes]:
        values = dict(params)
        values.update(zip(names, theta))
        beta, gamma, alpha, delta = (float(values[name]) for name in ("beta", "gamma", "alpha", "delta"))
        c, mu, rate = x[:nc].reshape(na, nz), x[nc:nc + nm].reshape(nk, nz), float(x[-1])
        if not (0 < beta < 1 and gamma > 0 and 0 < alpha < 1 and delta > 0 and rate > max(-1, -delta)):
            raise ValueError("Aiyagari residual outside its preference/technology/price domain")
        if np.any(c <= 0):
            raise ValueError("Aiyagari consumption must be positive")
        kl = ((rate + delta) / alpha) ** (1 / (alpha - 1))
        wage, capital_demand = (1 - alpha) * kl ** alpha, labor * kl
        cash = (1 + rate) * dense_grid[:, None] + wage * z[None, :]
        marginal = beta * (1 + rate) * (c ** (-gamma) @ transition.T)
        endogenous_c = marginal ** (-1 / gamma)
        c_new = np.empty_like(c)
        binding = np.empty_like(c, dtype=bool)
        cells = np.empty_like(c, dtype=np.int64)
        for j in range(nz):
            endogenous_a = (endogenous_c[:, j] + dense_grid - wage * z[j]) / (1 + rate)
            if np.any(np.diff(endogenous_a) <= 0):
                raise ValueError("nonmonotone endogenous grid in Aiyagari residual")
            binding[:, j] = dense_grid < endogenous_a[0]
            c_new[:, j] = np.interp(dense_grid, endogenous_a, endogenous_c[:, j])
            c_new[binding[:, j], j] = cash[binding[:, j], j]
            cells[:, j] = np.searchsorted(endogenous_a, dense_grid, side="right")
        # A fixed active borrowing equation implies exactly zero saving at its
        # nodes. This avoids differentiating max(cash-c,0) transverse to that
        # equation, while agreeing with the solved policies at equilibrium.
        savings = np.where(binding, 0.0, cash - c)
        if np.any(savings < -1e-10):
            raise ValueError("negative saving outside the declared active borrowing set")
        on_histogram = np.column_stack([np.interp(histogram, dense_grid, savings[:, j]) for j in range(nz)])
        transport = build_continuous_transition_matrix(on_histogram, histogram, transition)
        # Center the affine distribution block at the snapshot. This preserves
        # its exact linear dependence on mass when differencing tiny cells;
        # subtracting two near-unit sums otherwise pollutes the normalization
        # derivative and can be amplified by persistent Markov chains.
        change = mu - probability
        kfe = (probability.ravel() - transport.T @ probability.ravel()
               + change.ravel() - transport.T @ change.ravel())
        kfe[0] = (probability.sum() - 1.0) + change.sum()
        market = float((np.sum(probability * histogram[:, None]) - capital_demand)
                       + np.sum(change * histogram[:, None]))
        result = np.r_[(c - c_new).ravel(), kfe, market]
        lottery, _, _ = young_lottery_weights(on_histogram, histogram)
        signature = (binding.tobytes() + cells.tobytes() + lottery.tobytes()
                     + (on_histogram >= histogram[-1]).tobytes())
        return result, signature

    c_scale = np.maximum(consumption.ravel(), 1.0)
    state_scale = np.r_[c_scale, np.full(nm, 1 / nm), max(abs(solution.r), 0.01)]
    residual_scale = np.r_[c_scale, np.full(nm, 1 / nm), max(solution.K, 1.0)]
    residual_scale[nc] = 1.0  # probability normalization replaces one KFE row
    system = EquilibriumSystem(
        lambda x, theta: components(x, theta)[0], state_names, names,
        state_scale=state_scale, parameter_scale=np.maximum(np.abs(base), 0.1),
        residual_scale=residual_scale, regime=lambda x, theta: components(x, theta)[1],
        equation_names=(tuple(f"coleman[{i},{j}]" for i in range(na) for j in range(nz))
                        + ("mass_normalization",)
                        + tuple(f"stationarity[{i}]" for i in range(1, nm)) + ("capital_market",)),
        label="Continuous Aiyagari (fixed income chain and asset grids)",
    )
    return linearize_equilibrium(system, point, base, **options)
