"""Nonlinear household/market corrections with a solved local stochastic tail."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, splu

from puremacro.dsge.stacked_newton import preconditioned_lgmres
from .hank_equilibrium import HANKEquilibriumWorkspace, _compile_tree
from .hank_recursive import StochasticHANKPolicy, HANKStochasticTreeResult, _array, _integer


class _TreeProblem:
    def __init__(self, policy, tree, max_workspace_bytes):
        if not isinstance(policy, StochasticHANKPolicy) or not policy.converged:
            raise ValueError("a solved StochasticHANKPolicy is required")
        self.policy, self.kernel = policy, policy._kernel
        self.nodes, self.parents, self.children, self.prob, self.shocks = _compile_tree(tree)
        self.N, self.k, self.nu = len(tree), policy.basis.rank, len(self.kernel.aggregate_names)
        self.leaves = [i for i, ch in enumerate(self.children) if not ch]
        self.leaf_index = {i: j for j, i in enumerate(self.leaves)}
        self.size = self.N*self.nu+len(self.leaves)*self.k
        scratch = (16*self.kernel.shape[0]*self.kernel.d.size if self.kernel.model == "two_asset_consol" else 0)
        self.estimated_bytes = 8*(32*self.N*self.kernel.d.size+self.size*100+scratch)
        if self.estimated_bytes > _integer(max_workspace_bytes, "max_workspace_bytes"):
            raise ValueError("stochastic tree exceeds estimated working-array budget")
        self.evaluations = 0

    def initial(self):
        states = np.zeros((self.N, self.k+2))
        aggregate = np.zeros((self.nu, self.N))
        for i, parent in enumerate(self.parents):
            if i:
                states[i] = self.policy.transition@states[parent]
            states[i, -1] = self.shocks[i]
            aggregate[:, i] = self.policy.aggregate_feedback@states[i]
        terminal = np.asarray([(self.policy.transition@states[i])[:self.k] for i in self.leaves])
        return np.r_[aggregate.ravel(), terminal.ravel()]

    def evaluate(self, coordinates):
        self.evaluations += 1
        kernel, p = self.kernel, self.policy
        U = coordinates[:self.nu*self.N].reshape(self.nu, self.N)
        terminal_z = coordinates[self.nu*self.N:].reshape(len(self.leaves), self.k)
        prices, tail = [], {}
        for i, parent in enumerate(self.parents):
            lag = prices[parent]["nominal"]-kernel.r if i else 0.
            lagq = prices[parent]["q"] if i else 1.
            prices.append(kernel.prices(U[:, i], lag, self.shocks[i], lag_q=lagq))
        for i in self.leaves:
            xp = np.r_[terminal_z[self.leaf_index[i]], prices[i]["nominal"]-kernel.r, p.rho*self.shocks[i]]
            tail[i] = p._forecast(xp, prices[i]["nominal"])
        household: list[Any] = [None]*self.N
        expected: list[Any] = [None]*self.N
        for i in range(self.N-1, -1, -1):
            ch = self.children[i]
            expected[i] = sum(self.prob[j]*household[j]["m"] for j in ch) if ch else tail[i][0]
            household[i] = kernel.backward(expected[i], prices[i])
        distributions = np.empty((self.N,)+kernel.d.shape)
        distributions[0] = kernel.d
        projected = np.empty_like(terminal_z)
        rows, residuals, post = [], [], []
        for i, ch in enumerate(self.children):
            dp = kernel.forward(distributions[i], household[i])
            post.append(dp)
            if ch:
                distributions[ch] = dp
                continuation = sum(self.prob[j]*prices[j]["Y"]/prices[i]["Y"]*
                    np.log1p(prices[j]["pi"])/(1+prices[j]["r"]) for j in ch)
            else:
                projected[self.leaf_index[i]] = p.basis.project(dp-p.basis.center)
                continuation = sum(prob*Y/prices[i]["Y"]*logpi/gross
                                   for prob, Y, logpi, gross in tail[i][1])
            residual, row = kernel.markets(distributions[i], household[i], prices[i], continuation)
            residuals.append(residual)
            rows.append(row)
        residual = np.r_[np.asarray(residuals).T.ravel(), (terminal_z-projected).ravel()]
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite stochastic equilibrium residual")
        return dict(residual=residual, rows=rows, distributions=distributions, household=household,
                    prices=prices, expected=expected, post=post, terminal=terminal_z)

    def preconditioner(self):
        stop = self.nu*self.N
        if self.kernel.model == "one_asset":
            base = HANKEquilibriumWorkspace(self.kernel.ss, self.nodes)
            base._build_preconditioner()
            factor = base._preconditioner

            def solve_prices(rhs):
                return factor@rhs
        else:
            # Local household responses and unit carry of both asset stocks.
            # Full intertemporal/distribution effects remain in outer products.
            kernel = self.kernel
            local = self.policy._F0[-self.nu:, -self.nu:]
            lag_i = self.policy._F0[-self.nu:, self.k]*kernel.phi
            n, hh, p = kernel.d.size, kernel.hh, kernel.p
            response = kernel._tangent(np.zeros_like(kernel.m), np.array([0., -(1+p["r_a"]), 0.]))
            dc, da, db, _ = [v.reshape(kernel.shape) for v in np.split(response[2*n:], 4)]
            labor = -kernel.w*np.sum(kernel.d*hh.s_grid*hh.gamma*hh.c_ss**(-hh.gamma-1)*dc)
            lag_q = np.array([0., np.sum(kernel.d*db), labor,
                              np.sum(kernel.d*da)+p["consol_units"]])
            rows, cols, values = [], [], []

            def put(row, col, value):
                rows.append(row)
                cols.append(col)
                values.append(float(value))

            for i, parent in enumerate(self.parents):
                for r in range(self.nu):
                    for c in range(self.nu):
                        put(r*self.N+i, c*self.N+i, local[r, c])
                    if i:
                        put(r*self.N+i, parent, lag_i[r])
                        put(r*self.N+i, 3*self.N+parent, lag_q[r])
                for child in self.children[i]:
                    put(i, child, self.prob[child]/(1+kernel.r))
            factor2 = splu(sparse.csc_matrix((values, (rows, cols)), shape=(stop, stop)))

            def solve_prices(rhs):
                transformed = np.array(rhs, copy=True).reshape(self.nu, self.N)
                original = np.asarray(rhs).reshape(self.nu, self.N)
                for r in (1, 3):
                    transformed[r, 1:] -= original[r, self.parents[1:]]
                return factor2.solve(transformed.ravel())

        def apply(rhs):
            return np.r_[solve_prices(rhs[:stop]), rhs[stop:]]

        return LinearOperator((self.size, self.size), matvec=apply, dtype=float)


def _solve_tree(policy, tree, *, tolerance=1e-8, max_iterations=30, initial=None,
                method="krylov", preconditioner="sparse", max_workspace_bytes=1_073_741_824):
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError("tolerance must lie in (0,1e-6]")
    _integer(max_iterations, "max_iterations")
    if method not in ("krylov", "hybr"):
        raise ValueError("method must be 'krylov' or 'hybr'")
    if preconditioner not in ("sparse", "none"):
        raise ValueError("preconditioner must be 'sparse' or 'none'")
    problem = _TreeProblem(policy, tree, max_workspace_bytes)
    x = problem.initial() if initial is None else _array(initial, "initial", (problem.size,))
    state = problem.evaluate(x)
    history = []
    status = "maximum_iterations"
    products = 0
    if method == "hybr":
        # Dense finite differences provide an independent aggregate-solver
        # comparison. Memory is explicitly bounded before MINPACK allocates it.
        if 16*problem.size**2+problem.estimated_bytes > max_workspace_bytes:
            raise ValueError("dense reference Jacobian exceeds max_workspace_bytes")
        fit = root(lambda y: problem.evaluate(y)["residual"], x, method="hybr", tol=tolerance,
                   options={"maxfev": max_iterations*(problem.size+1)})
        x, state = fit.x, problem.evaluate(fit.x)
        status = "converged" if np.max(abs(state["residual"])) <= tolerance else str(fit.message)
    else:
        try:
            linear_preconditioner = problem.preconditioner() if preconditioner == "sparse" else None
        except RuntimeError as exc:
            raise ValueError("singular stochastic-tree preconditioner; try preconditioner='none'") from exc
        for iteration in range(max_iterations+1):
            residual = state["residual"]
            norm = float(np.linalg.norm(residual))
            if np.max(abs(residual)) <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break

            def matvec(direction):
                nonlocal products
                size = np.linalg.norm(direction)
                if size == 0:
                    return np.zeros_like(direction)
                products += 1
                step = 1e-6*max(1., np.linalg.norm(x))/size
                return (problem.evaluate(x+step*direction)["residual"]-
                        problem.evaluate(x-step*direction)["residual"])/(2*step)

            operator = LinearOperator((problem.size, problem.size), matvec=matvec, dtype=float)
            forcing = min(.05, max(1e-6, .5*np.sqrt(norm)))
            try:
                direction, info, error = preconditioned_lgmres(operator, -residual, linear_preconditioner,
                    method="gmres", side="right", rtol=forcing, atol=0., inner_m=min(80, problem.size), maxiter=20)
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                history.append(dict(iteration=iteration, residual_max_abs=float(np.max(abs(residual))),
                                    linear_probe_error=str(exc), step_length=0.))
                status = "linear_probe_failed"
                break
            entry = dict(iteration=iteration, residual_max_abs=float(np.max(abs(residual))),
                         linear_residual_norm=error, krylov_status=info, step_length=0.)
            history.append(entry)
            if not np.isfinite(direction).all() or error > forcing*norm*(1+1e-8):
                status = "linear_solve_failed"
                break
            for backtrack in range(17):
                step_length = .5**backtrack
                try:
                    trial = problem.evaluate(x+step_length*direction)
                except (ValueError, FloatingPointError, RuntimeError):
                    continue
                if np.linalg.norm(trial["residual"]) <= (1-1e-4*step_length)*norm:
                    x, state = x+step_length*direction, trial
                    entry["step_length"] = step_length
                    break
            else:
                status = "line_search_failed"
                break
    kernel = policy._kernel
    checks = [kernel.household_checks(d, h, v, expected) for d, h, v, expected in zip(
        state["distributions"], state["household"], state["prices"], state["expected"])]
    paths = pd.DataFrame(state["rows"], index=problem.nodes.index)
    all_mass = np.asarray([*state["distributions"], *state["post"]])
    mass_error = float(np.max(abs(all_mass.reshape(len(all_mass), -1).sum(axis=1)-1)))
    budget = max(check["budget_error"] for check in checks)
    exit_mass = max(check["grid_exit_mass"] for check in checks)
    residual = max(float(np.max(abs(state["residual"]))), float(abs(paths.goods_mkt).max()))
    terminal_error = float(np.max(abs(state["residual"][problem.nu*problem.N:])))
    compression = max(float(np.sum(abs(state["post"][i]-policy.basis.decode(
        state["terminal"][problem.leaf_index[i]], check_positive=False)))) for i in problem.leaves)
    terminal_minimum_mass = min(float(policy.basis.decode(z, check_positive=False).min())
                                for z in state["terminal"])
    converged = bool(status == "converged" and residual <= tolerance and budget < 1e-8
                     and mass_error < 1e-10 and all_mass.min() >= -1e-12 and exit_mass < 1e-10)
    diagnostics = dict(optimizer_method=method, optimizer_message=status, evaluations=problem.evaluations,
        preconditioner=preconditioner if method == "krylov" else "not_used",
        derivative_scope="centered products of the full nonlinear residual",
        jacobian_vector_products=products, history=history, market_residual_max_abs=residual,
        terminal_consistency_error=terminal_error, terminal_compression_l1=compression,
        terminal_minimum_reconstructed_mass=terminal_minimum_mass,
        mass_error=mass_error, minimum_mass=float(all_mass.min()), household_budget_error=budget,
        policy_grid_exit_mass=exit_mass, euler_kkt_max=max(check["euler_kkt_max"] for check in checks),
        upper_grid_mass=max(check["upper_grid_mass"] for check in checks), tolerance=tolerance,
        estimated_workspace_bytes=problem.estimated_bytes, dense_jacobian_assembled=method == "hybr")
    metadata = dict(scope="nonlinear finite-history equilibrium with solved local stochastic continuation",
        model=kernel.model, terminal_policy_order=1, stochastic_tail_solved=True,
        shock_process="AR(1), two-point Gaussian quadrature in terminal expectations",
        rho=policy.rho, shock_std=policy.shock_std,
        validation_scope="finite-tree residuals do not certify compression, household grids, or global stochastic accuracy")
    policies = {key: np.asarray([h[key] for h in state["household"]])
                for key in state["household"][0] if key != "m"}
    return HANKStochasticTreeResult(problem.nodes, paths, state["distributions"], policies,
                                    state["terminal"], converged, diagnostics, metadata)
