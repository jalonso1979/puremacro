"""Local HANK equilibrium with reduced distributions and marginal controls.

Household backward steps and probability transport use the original grid.
Only perturbations to marginal values are expanded in a response basis. The
Jacobian and ordered Schur decomposition are assembled in those coordinates;
no full household derivative cache is formed. A solved projected system is
not a claim that the omitted household equations vanish.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import ordqz, solve, solve_discrete_lyapunov
from scipy.optimize import root

from ._hank_one_asset_recursive import _OneAssetKernel
from .hank_recursive import (HANKDistributionBasis, HANKStochasticResult,
                             _array, _integer, _make_basis)
from .hank_two_asset_equilibrium import _TwoAssetKernel

__all__ = ["HANKMarginalBasis", "ReducedStochasticHANKPolicy",
           "solve_hank_reduced_policy", "solve_hank_reduced_kernel"]


class _DirectionalTwoAssetKernel(_TwoAssetKernel):
    """Apply household derivatives without the legacy full-grid cache."""

    def __init__(self, ss):
        super().__init__(ss)
        self.fd_step = 1e-6

    def _tangent(self, dm, dp):
        # Normalize a direction so the step is comparable across basis ranks.
        # In particular zero directions need no household evaluation.
        scale = max(float(np.max(abs(dm)/(1+abs(self.m)))), float(np.max(abs(dp))))
        if scale == 0:
            return np.zeros(6*self.d.size)
        return self._response(dm/scale, dp/scale, self.fd_step)*scale


def _aggregate_state_names(kernel):
    names = tuple(getattr(kernel, "aggregate_state_names", ("lag_nominal", "monetary_shock")))
    if len(set(names)) != len(names) or "monetary_shock" not in names:
        raise ValueError("aggregate_state_names must be unique and contain monetary_shock")
    return names


def _prices_from_state(kernel, controls, aggregate_state, distribution, *, lag_q=None):
    if hasattr(kernel, "prices_from_state"):
        return kernel.prices_from_state(controls, aggregate_state, distribution, lag_q=lag_q)
    return kernel.prices(controls, *aggregate_state, distribution=distribution, lag_q=lag_q)


def _next_aggregate_state(kernel, prices, aggregate_state, rho):
    if hasattr(kernel, "next_aggregate_state"):
        return np.asarray(kernel.next_aggregate_state(prices, aggregate_state, rho), dtype=float)
    return np.array([prices["nominal"]-kernel.r, rho*aggregate_state[1]])


@dataclass(frozen=True)
class HANKMarginalBasis:
    """Orthonormal deviations around full-grid stationary marginal values.

    Marginal values are never clipped to keep a compressed policy feasible.
    The nonlinear backward operator still enforces household asset bounds.
    """
    center: np.ndarray
    vectors: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m, V = _array(self.center, "marginal center"), _array(self.vectors, "marginal vectors")
        if m.ndim != 1 or np.min(m) <= 0 or V.ndim != 2 or V.shape[0] != len(m) or not V.shape[1]:
            raise ValueError("invalid marginal basis dimensions or nonpositive center")
        if not np.allclose(V.T@V, np.eye(V.shape[1]), atol=1e-10, rtol=0):
            raise ValueError("marginal basis must be orthonormal")
        for name, value in (("center", m), ("vectors", V)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[1])

    def project(self, difference: np.ndarray) -> np.ndarray:
        return self.vectors.T@_array(difference, "marginal difference", self.center.shape)

    def decode(self, coefficients: np.ndarray, *, check_positive: bool = True) -> np.ndarray:
        m = self.center+self.vectors@_array(coefficients, "marginal coefficients", (self.rank,))
        if check_positive and np.min(m) <= 0:
            raise ValueError("compressed marginal values left the positive domain")
        return m


def _make_marginal_basis(kernel, basis, rank, snapshots, steps):
    nm, ns, nu = len(kernel.m), basis.rank+len(_aggregate_state_names(kernel)), len(kernel.aggregate_names)
    if rank is None or rank == nm:
        return HANKMarginalBasis(kernel.m, np.eye(nm), dict(method="full_marginal_values", rank=nm))
    rank = _integer(rank, "marginal_rank")
    if rank > nm:
        raise ValueError(f"marginal_rank exceeds the full marginal dimension {nm}")
    zero = np.zeros(ns+nm+nu)
    seeds = []
    for j in [*range(basis.rank, ns), *range(ns+nm, len(zero))]:
        cur = zero.copy()
        cur[j] = 1.
        seeds.append(-kernel.linear_residual(cur, zero, basis, 0.)[ns:ns+nm])
    price_responses = np.column_stack(seeds)
    U, singular, _ = np.linalg.svd(price_responses, full_matrices=False)
    required = int(np.sum(singular > (singular[0] if len(singular) else 1)*1e-10))
    if required == 0 or rank < required:
        raise ValueError(f"marginal_rank must retain at least {required} independent price responses")
    mandatory = U[:, :required]
    if snapshots is None:
        block, columns = mandatory.copy(), []
        for _ in range(steps):
            columns.append(block.copy())
            future = zero.copy()
            new = []
            for v in block.T:
                future[ns:ns+nm] = v
                new.append(-kernel.linear_residual(zero, future, basis, 0.)[ns:ns+nm])
            block = np.column_stack(new)
            norms = np.linalg.norm(block, axis=0)
            block /= np.maximum(norms, 1e-300)
        candidates = np.column_stack(columns)
        method = "backward_response_krylov"
    else:
        data = _array(snapshots, "marginal_snapshots")
        if data.ndim != 2 or data.shape[1] != nm or not len(data) or np.min(data) <= 0:
            raise ValueError("marginal_snapshots must contain positive full marginal values (observations, marginals)")
        candidates = (data-kernel.m).T
        method = "marginal_snapshot_svd"
    training = candidates.copy()
    candidates -= mandatory@(mandatory.T@candidates)
    U, singular, _ = np.linalg.svd(candidates, full_matrices=False)
    available = int(np.sum(singular > max(float(singular[0]) if len(singular) else 0., 1e-300)*1e-12))
    additional = min(rank-required, available)
    extra = U[:, :additional]-mandatory@(mandatory.T@U[:, :additional])
    V, _ = np.linalg.qr(np.column_stack([mandatory, extra]), mode="reduced")
    error = np.linalg.norm(training-V@(V.T@training))/max(np.linalg.norm(training), 1e-300)
    return HANKMarginalBasis(kernel.m, V, dict(method=method, rank=V.shape[1], requested_rank=rank,
        available_directions=available+required, price_response_directions=required, krylov_steps=steps,
        training_relative_error=float(error), singular_values=singular.tolist()))


@dataclass
class ReducedStochasticHANKPolicy:
    """Stable policy for a projected local rational-expectations system.

    ``feedback`` expands marginal controls to the original household grid.
    ``reduced_feedback`` and ``_F0``/``_F1`` use marginal-basis coefficients.
    ``converged`` certifies the projected equations and determinacy only; full
    household residuals are reported separately and checked by ``audit``.
    """
    basis: HANKDistributionBasis
    marginal_basis: HANKMarginalBasis
    transition: np.ndarray
    feedback: np.ndarray
    reduced_feedback: np.ndarray
    rho: float
    shock_std: float
    diagnostics: dict[str, Any]
    _kernel: Any = field(repr=False)
    _F0: np.ndarray = field(repr=False)
    _F1: np.ndarray = field(repr=False)
    _full_residual: np.ndarray = field(repr=False)

    @property
    def converged(self) -> bool:
        return bool(self.diagnostics["equilibrium_residual_max_abs"] < 1e-8)

    @property
    def aggregate_state_names(self) -> tuple[str, ...]:
        return _aggregate_state_names(self._kernel)

    @property
    def shock_index(self) -> int:
        return self.basis.rank+self.aggregate_state_names.index("monetary_shock")

    @property
    def aggregate_feedback(self) -> np.ndarray:
        return self.feedback[-len(self._kernel.aggregate_names):].copy()

    @property
    def steady_state(self):
        from copy import deepcopy
        return deepcopy(self._kernel.ss)

    def state(self, distribution: np.ndarray | None = None, *, lag_nominal: float = 0.,
              monetary_shock: float = 0., **aggregate_states: float) -> np.ndarray:
        names = self.aggregate_state_names
        if set(aggregate_states)-set(names):
            raise ValueError("unknown aggregate state")
        values = dict(lag_nominal=lag_nominal, monetary_shock=monetary_shock, **aggregate_states)
        d = self.basis.center if distribution is None else distribution
        return np.r_[self.basis.encode(d), _array([values.get(n, 0.) for n in names], "aggregate states")]

    def covariance(self) -> np.ndarray:
        """First-order covariance; this does not shift stationary means."""
        Q = np.zeros_like(self.transition)
        Q[self.shock_index, self.shock_index] = self.shock_std**2
        return solve_discrete_lyapunov(self.transition, Q)

    def simulate(self, innovations: np.ndarray, *, initial_state: np.ndarray | None = None) -> HANKStochasticResult:
        """Simulate the projected policy and report omitted full-grid equations."""
        eps = _array(innovations, "innovations")
        if eps.ndim != 1 or not len(eps):
            raise ValueError("innovations must be a nonempty vector")
        ns, k = len(self.transition), self.basis.rank
        x = np.zeros(ns) if initial_state is None else _array(initial_state, "initial_state", (ns,))
        states, distributions, rows = [], [], []
        error = full_error = marginal_error = 0.
        nm = len(self._kernel.m)
        for innovation in eps:
            x[self.shock_index] += innovation
            y, xp = self.reduced_feedback@x, self.transition@x
            vec, nxt = np.r_[x, y], np.r_[xp, self.reduced_feedback@xp]
            error = max(error, float(np.max(abs(self._F0@vec+self._F1@nxt))))
            omitted = self._full_residual@x
            full_error = max(full_error, float(np.max(abs(omitted))))
            marginal_error = max(marginal_error, float(np.max(abs(omitted[ns:ns+nm])/(1+abs(self._kernel.m)))))
            rows.append(dict(zip(self._kernel.aggregate_names, y[-len(self._kernel.aggregate_names):])))
            states.append(x.copy())
            distributions.append(self.basis.decode(x[:k], check_positive=False))
            x = xp
        dist = np.asarray(distributions)
        return HANKStochasticResult(pd.DataFrame(rows), np.asarray(states), dist,
            dict(scope="projected local first-order equilibrium; aggregate paths are log deviations",
                 linear_equilibrium_error=error, full_linear_equation_error=full_error,
                 full_marginal_relative_error=marginal_error,
                 minimum_reconstructed_mass=float(dist.min()),
                 negative_reconstructed_mass=float(np.maximum(-dist, 0).sum(axis=tuple(range(1, dist.ndim))).max()),
                 stochastic_tail_solved=True, distribution_rank=k, marginal_rank=self.marginal_basis.rank,
                 convergence_scope="projected equations only"), error < 1e-8)

    def _forecast(self, next_state, nominal):
        kernel, nm = self._kernel, len(self._kernel.m)
        expected = kernel.m+self.feedback[:nm]@next_state
        if np.min(expected) <= 0 or not np.isfinite(expected).all():
            raise ValueError("terminal household marginal value left the positive domain")
        branches = []
        for innovation in (-self.shock_std, self.shock_std):
            x = next_state.copy()
            x[self.shock_index] += innovation
            if np.min(kernel.m+self.feedback[:nm]@x) <= 0:
                raise ValueError("a terminal shock branch has nonpositive household marginal values")
            u = self.aggregate_feedback@x
            pi, Y = np.expm1(u[0]), kernel.Y*np.exp(u[2])
            gross = (1+nominal)/(1+pi)
            if not np.isfinite([pi, Y, gross]).all() or min(Y, gross) <= 0:
                raise ValueError("terminal aggregate policy left its local domain")
            branches.append((.5, Y, u[0], gross))
        return expected, branches

    def audit(self, innovations: np.ndarray, *, tolerance: float = 1e-5) -> HANKStochasticResult:
        """Transport the full distribution and test every marginal equation.

        Fixed aggregate policy coefficients are held out of the nonlinear
        consistency solve. Acceptance includes full marginal approximation,
        markets, goods, budgets, mass and grid exits. Euler/KKT interpolation
        errors are reported separately. Actual quoted lagged consol prices
        are used even when a market residual makes inferred wealth differ.
        """
        eps = _array(innovations, "innovations")
        if eps.ndim != 1 or not len(eps) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("nonempty innovations and a positive finite tolerance are required")
        kernel, k = self._kernel, self.basis.rank
        ns, nm, rank = len(self.transition), len(kernel.m), self.marginal_basis.rank
        jacobian = np.eye(k)+self._F1[:k, ns:ns+rank]@self.reduced_feedback[:rank, :k]
        d, aggregate = kernel.d.copy(), np.zeros(len(self.aggregate_state_names))
        shock_index = self.aggregate_state_names.index("monetary_shock")
        error = compression = budget = euler = exit_mass = marginal_error = 0.
        lag_q = 1.
        rows, states, distributions, statuses = [], [], [], []
        for innovation in eps:
            aggregate[shock_index] += innovation
            x = np.r_[self.basis.project(d-self.basis.center), aggregate]
            u = self.aggregate_feedback@x
            prices = _prices_from_state(kernel, u, aggregate, d, lag_q=lag_q)
            next_aggregate = _next_aggregate_state(kernel, prices, aggregate, self.rho)

            def evaluate(z):
                xp = np.r_[z, next_aggregate]
                expected, branches = self._forecast(xp, prices["nominal"])
                h = kernel.backward(expected, prices)
                dp = kernel.forward(d, h)
                return z-self.basis.project(dp-self.basis.center), h, dp, expected, branches

            fit = root(lambda z: evaluate(z)[0], (self.transition@x)[:k], jac=lambda z: jacobian, tol=1e-10)
            consistency, h, dp, expected, branches = evaluate(fit.x)
            consistency_error = float(np.max(abs(consistency)))
            statuses.append(dict(success=bool(consistency_error < 1e-9), message=str(fit.message)))
            continuation = sum(prob*Y/prices["Y"]*logpi/gross for prob, Y, logpi, gross in branches)
            residual, row = kernel.markets(d, h, prices, continuation)
            checks = kernel.household_checks(d, h, prices, expected)
            marginal = kernel.m+self.feedback[:nm]@x-h["m"]
            marginal_period = float(np.max(abs(marginal)/(1+abs(kernel.m))))
            marginal_error = max(marginal_error, marginal_period)
            projection = self.basis.decode(self.basis.project(dp-self.basis.center), check_positive=False)
            gap = float(np.sum(abs(dp-projection)))
            error = max(error, float(np.max(abs(residual))), abs(row["goods_mkt"]), consistency_error)
            compression = max(compression, gap)
            budget, euler = max(budget, checks["budget_error"]), max(euler, checks["euler_kkt_max"])
            exit_mass = max(exit_mass, checks["grid_exit_mass"])
            row.update(compression_l1=gap, continuation_error=consistency_error,
                       full_marginal_relative_error=marginal_period,
                       euler_kkt_max=checks["euler_kkt_max"], shock=aggregate[shock_index])
            rows.append(row)
            states.append(x)
            distributions.append(d.copy())
            d, aggregate, lag_q = dp, next_aggregate, prices.get("q", 1.)
        dist = np.asarray([*distributions, d])
        mass_error = float(np.max(abs(dist.reshape(len(dist), -1).sum(axis=1)-1)))
        ok = bool(error <= tolerance and marginal_error <= tolerance and budget < 1e-8
                  and mass_error < 1e-10 and dist.min() >= -1e-12 and exit_mass < 1e-10
                  and all(s["success"] for s in statuses))
        return HANKStochasticResult(pd.DataFrame(rows), np.asarray(states), dist,
            dict(scope="nonlinear full-distribution and full-marginal audit of fixed reduced policy",
                 market_residual_max_abs=error, full_marginal_relative_error=marginal_error,
                 compression_l1_max=compression, mass_error=mass_error, minimum_mass=float(dist.min()),
                 household_budget_error=budget, euler_kkt_max=euler, policy_grid_exit_mass=exit_mass,
                 tolerance=tolerance, continuation_solves=statuses), ok)


def solve_hank_reduced_kernel(kernel: Any, *, distribution_rank: int | None = 12,
        marginal_rank: int | None = 32, snapshots: np.ndarray | None = None,
        marginal_snapshots: np.ndarray | None = None, krylov_steps: int = 48,
        rho: float = .7, shock_std: float = .0001, max_system_size: int = 1800) -> ReducedStochasticHANKPolicy:
    """Solve projected equations for a kernel implementing the HANK protocol.

    ``linear_residual`` accepts full-grid marginal directions and must apply
    household derivatives directionally. The supplied kernel is owned by the
    returned policy; callers should not mutate it afterward. Optional hooks
    ``aggregate_state_names``, ``prices_from_state`` and
    ``next_aggregate_state`` extend the two legacy aggregate states.

    Price-response marginal directions are retained exactly. Further controls
    use a backward-response Krylov/SVD basis, or supplied marginal snapshots.
    A requested rank above the numerically independent training span is capped
    and recorded. ``None`` retains every marginal on small comparison grids.
    """
    if not np.isfinite([rho, shock_std]).all() or abs(rho) >= 1 or shock_std < 0:
        raise ValueError("require abs(rho) < 1 and nonnegative finite shock_std")
    if (getattr(kernel, "_derivatives", None) is not None
            or isinstance(kernel, _TwoAssetKernel) and type(kernel)._tangent is _TwoAssetKernel._tangent):
        raise ValueError("a directional kernel is required; the legacy full household derivative cache is unsupported")
    limit, steps = _integer(max_system_size, "max_system_size"), _integer(krylov_steps, "krylov_steps")
    na, nm, nu = len(_aggregate_state_names(kernel)), len(kernel.m), len(kernel.aggregate_names)
    requested_d = (kernel.d.size-kernel.constraints.shape[1] if distribution_rank is None
                   else _integer(distribution_rank, "distribution_rank"))
    requested_m = nm if marginal_rank is None else _integer(marginal_rank, "marginal_rank")
    anticipated = requested_d+na+requested_m+nu
    if anticipated > limit:
        raise ValueError(f"reduced Schur system dimension {anticipated} exceeds max_system_size={limit}")
    zero_prices = _prices_from_state(kernel, np.zeros(nu), np.zeros(na), kernel.d)
    zero_household = kernel.backward(kernel.m, zero_prices)
    zero_residual, zero_accounts = kernel.markets(kernel.d, zero_household, zero_prices, 0.)
    base_error = max(float(np.max(abs(zero_residual))), abs(zero_accounts["goods_mkt"]),
        float(np.max(abs(zero_household["m"]-kernel.m)/(1+abs(kernel.m)))),
        float(np.max(abs(kernel.forward(kernel.d, zero_household)-kernel.d))))
    if (not np.isfinite(base_error) or base_error > 1e-8 or np.min(kernel.d) < -1e-12
            or abs(kernel.d.sum()-1) > 1e-10 or np.min(kernel.m) <= 0):
        raise ValueError("steady-state equations or distribution are inconsistent")
    basis = _make_basis(kernel, distribution_rank, snapshots)
    marginals = _make_marginal_basis(kernel, basis, marginal_rank, marginal_snapshots, steps)
    ns, nr = basis.rank+na, marginals.rank
    dim, full_dim = ns+nr+nu, ns+nm+nu
    # Only reduced-coordinate square matrices are allocated. Expanded single
    # directions are transient vectors, not a full-grid household Jacobian.
    zero = np.zeros(full_dim)

    def expand(vector):
        return np.r_[vector[:ns], marginals.vectors@vector[ns:ns+nr], vector[-nu:]]

    def project(vector):
        return np.r_[vector[:ns], marginals.vectors.T@vector[ns:ns+nm], vector[-nu:]]

    F0, F1 = np.empty((dim, dim)), np.empty((dim, dim))
    for j in range(dim):
        unit = np.zeros(dim)
        unit[j] = 1.
        direction = expand(unit)
        F0[:, j] = project(kernel.linear_residual(direction, zero, basis, rho))
        F1[:, j] = project(kernel.linear_residual(zero, direction, basis, rho))
    S, T, alpha, beta, _, Z = ordqz(-F0, F1, sort="iuc", output="complex")
    with np.errstate(divide="ignore", invalid="ignore"):
        eigen = alpha/beta
    modulus = abs(eigen)
    if np.isnan(modulus).any() or np.any(abs(modulus-1) < 1e-8) or int(np.sum(modulus < 1-1e-8)) != ns:
        raise ValueError(f"projected stochastic equilibrium fails determinacy: {np.sum(modulus < 1-1e-8)} stable roots for {ns} states")
    Zx = Z[:ns, :ns]
    condition = float(np.linalg.cond(Zx))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("stable Schur subspace does not identify a well-conditioned state policy")
    reduced_G = solve(Zx.T, Z[ns:, :ns].T).T
    H = solve(Zx.T, (Zx@solve(T[:ns, :ns], S[:ns, :ns])).T).T
    if max(float(abs(reduced_G.imag).max()), float(abs(H.imag).max())) > 1e-8:
        raise ValueError("non-real Schur policy")
    reduced_G, H = reduced_G.real, H.real
    G = np.vstack([marginals.vectors@reduced_G[:nr], reduced_G[nr:]])
    stack = np.vstack([np.eye(ns), reduced_G])
    projected = F0@stack+F1@stack@H
    residual = float(np.max(abs(projected)))
    radius = float(np.max(abs(np.linalg.eigvals(H))))
    if residual > 1e-8 or radius >= 1-1e-8:
        raise ValueError("projected stochastic equilibrium failed equation or stability checks")
    full_stack = np.vstack([np.eye(ns), G])
    full_residual = np.column_stack([kernel.linear_residual(full_stack[:, j], full_stack@H[:, j], basis, rho)
                                     for j in range(ns)])
    omitted = full_residual[ns:ns+nm]
    Q = np.zeros_like(H)
    shock_index = basis.rank+_aggregate_state_names(kernel).index("monetary_shock")
    Q[shock_index, shock_index] = shock_std**2
    covariance = solve_discrete_lyapunov(H, Q)
    omitted_variance = np.maximum(np.einsum("ij,ji->i", omitted@covariance, omitted.T), 0.)
    diagnostics = dict(equilibrium_residual_max_abs=residual, projected_equilibrium_residual_max_abs=residual,
        full_equation_residual_max_abs=float(np.max(abs(full_residual))),
        full_marginal_residual_max_abs=float(np.max(abs(omitted))),
        full_marginal_relative_residual_max_abs=float(np.max(abs(omitted)/(1+abs(kernel.m[:, None])))),
        full_marginal_stationary_rms_max=float(np.sqrt(omitted_variance).max()),
        full_marginal_stationary_relative_rms_max=float(np.max(np.sqrt(omitted_variance)/(1+abs(kernel.m)))),
        directional_projection_error=float(np.max(abs(np.column_stack([project(v) for v in full_residual.T])-projected))),
        steady_state_error=base_error, spectral_radius=radius, stable_roots=ns,
        state_projection_condition=condition, system_dimension=dim,
        full_control_system_dimension=full_dim, distribution_rank=basis.rank, marginal_rank=nr,
        full_distribution_dimension=kernel.d.size-kernel.constraints.shape[1], full_marginal_dimension=nm,
        approximation_order=1, stochastic_tail_solved=True, model=kernel.model,
        full_household_derivative_cache=False,
        full_grid_schur_allocated=bool(nr == nm and basis.rank == kernel.d.size-kernel.constraints.shape[1]),
        household_derivatives="analytic directional" if kernel.model == "one_asset" else "centered directional differences",
        household_difference_step=getattr(kernel, "fd_step", None),
        reduced_jacobian_bytes=int(F0.nbytes+F1.nbytes), expanded_feedback_bytes=int(G.nbytes),
        basis_bytes=int(basis.vectors.nbytes+marginals.vectors.nbytes),
        convergence_scope="projected equations and determinacy only; inspect full residuals and nonlinear audit",
        scope="local first-order policy with reduced states and controls, not a global aggregate-risk solution")
    if getattr(kernel, "_derivatives", None) is not None:
        raise ValueError("kernel allocated a full household derivative cache; use a directional kernel")
    for array in (G, reduced_G, H, F0, F1, full_residual):
        array.flags.writeable = False
    return ReducedStochasticHANKPolicy(basis, marginals, H, G, reduced_G, float(rho), float(shock_std),
                                      diagnostics, kernel, F0, F1, full_residual)


def solve_hank_reduced_policy(steady_state: Any, *, distribution_rank: int | None = 12,
        marginal_rank: int | None = 32, snapshots: np.ndarray | None = None,
        marginal_snapshots: np.ndarray | None = None, krylov_steps: int = 48,
        rho: float = .7, shock_std: float = .0001, max_system_size: int = 1800) -> ReducedStochasticHANKPolicy:
    """Reduce distribution states and household marginal controls before QZ.

    Supports one-asset, fixed-supply illiquid-consol and productive-capital
    steady states, preserving each model's aggregate closure.
    ``solve_hank_reduced_kernel`` supplies the extensible aggregate closure.
    Retaining full marginals on a small grid is an independent comparison to
    the existing dense solver; large grids should use an explicit small rank.
    """
    from .one_asset_hank import OneAssetHANKSteadyState
    from .hank_capital import CapitalHANKSteadyState, _CapitalKernel
    kernel = (_CapitalKernel(steady_state) if isinstance(steady_state, CapitalHANKSteadyState)
              else _OneAssetKernel(steady_state) if isinstance(steady_state, OneAssetHANKSteadyState)
              else _DirectionalTwoAssetKernel(steady_state))
    return solve_hank_reduced_kernel(kernel, distribution_rank=distribution_rank, marginal_rank=marginal_rank,
        snapshots=snapshots, marginal_snapshots=marginal_snapshots, krylov_steps=krylov_steps,
        rho=rho, shock_std=shock_std, max_system_size=max_system_size)
