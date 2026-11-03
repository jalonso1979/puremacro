"""Matrix-free, fully coupled one-asset HANK equilibrium on finite histories.

The economic equations and terminal closure are those of hank_scenario_tree.
Analytic tangent sweeps replace dense finite-difference Jacobians. This is an
experimental numerical backend, not a stationary stochastic HANK solution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu

from puremacro.dsge.stacked_newton import preconditioned_lgmres

from ._hank_tangent import _forward_jvp, _forward_tape, _household_jvp, _household_tape
from .hank_scenario_tree import HANKScenarioTreeResult
from .one_asset_hank import OneAssetHANKSteadyState, _backward, _forward

__all__ = ["HANKEquilibriumWorkspace", "solve_hank_equilibrium"]


def _compile_tree(tree):
    # Same economic validation as the dense reference, without its 256-node
    # Jacobian cap. Keep independent so the frozen reference remains unchanged.
    required = ["parent", "conditional_probability", "monetary_shock"]
    if not isinstance(tree, pd.DataFrame) or len(tree) < 2 or any(k not in tree for k in required):
        raise ValueError("tree requires at least two nodes with parent, conditional_probability and monetary_shock")
    if not tree.index.is_unique or any(not isinstance(k, str) or not k.strip() for k in tree.index):
        raise ValueError("node labels must be unique nonempty strings")
    nodes = tree[required].copy(deep=True)
    if np.iscomplexobj(nodes.conditional_probability) or np.iscomplexobj(nodes.monetary_shock):
        raise ValueError("tree probabilities and shocks must be real")
    conditional = nodes.conditional_probability.to_numpy(dtype=float)
    shocks = nodes.monetary_shock.to_numpy(dtype=float)
    if (not np.isfinite(shocks).all() or not np.isfinite(conditional).all()
            or np.any(conditional <= 0) or np.any(conditional > 1)):
        raise ValueError("tree requires finite shocks and probabilities in (0,1]")
    labels = dict(zip(nodes.index, range(len(nodes))))
    parents = np.full(len(nodes), -1, dtype=int)
    depth, probability = np.zeros(len(nodes), dtype=int), np.ones(len(nodes))
    children: list[list[int]] = [[] for _ in nodes.index]
    for j, parent in enumerate(nodes.parent):
        if j == 0:
            if not pd.isna(parent) or conditional[j] != 1:
                raise ValueError("first node must be the unique root with probability one")
        else:
            if not isinstance(parent, str) or parent not in labels or labels[parent] >= j:
                raise ValueError("each nonroot parent must occur earlier; recombining trees are unsupported")
            i = labels[parent]
            parents[j], depth[j] = i, depth[i] + 1
            children[i].append(j)
            probability[j] = probability[i] * conditional[j]
    for ch in children:
        if ch and not np.isclose(conditional[ch].sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("conditional child probabilities must sum to one at every parent")
    if len({int(depth[i]) for i, ch in enumerate(children) if not ch}) != 1:
        raise ValueError("all terminal nodes must share the same horizon")
    nodes["period"], nodes["probability"] = depth, probability
    return nodes, parents, children, conditional, shocks


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _vector(value, size, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real vector of length {size}")
    out = np.array(value, dtype=float, copy=True)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite real vector of length {size}")
    return out


@dataclass
class _Evaluation:
    x: np.ndarray
    levels: dict[str, np.ndarray]
    marginal: np.ndarray
    saving: np.ndarray
    consumption: np.ndarray
    hours: np.ndarray
    distributions: np.ndarray
    residual: np.ndarray


class HANKEquilibriumWorkspace:
    """Prepare reusable topology, household data and a sparse preconditioner.

    The workspace owns copies of its inputs. ``solve`` accepts new monetary
    shocks on the same topology and an explicit warm start. There is no hidden
    reuse of a previous solution. Changed calibration, grid or probabilities
    require a new workspace. Log coordinates are concatenated as
    ``[log(1+pi), log(w/w_ss), log(Y)]``, each block in tree row order.

    ``max_workspace_bytes`` guards an estimated NumPy working-array budget
    (32 float arrays per node/household cell plus 128 floats per node), not
    process RSS. Python, BLAS, and sparse-factor overhead are additional. The
    estimate is conservative for the default Krylov restart; large trees still
    need memory proportional to nodes times household grid size.
    """

    def __init__(self, steady_state: OneAssetHANKSteadyState, tree: pd.DataFrame, *,
                 max_workspace_bytes: int = 1_073_741_824):
        if not isinstance(steady_state, OneAssetHANKSteadyState) or not steady_state.converged:
            raise ValueError("a converged OneAssetHANKSteadyState is required")
        self._nodes, self._parents, self._children, self._prob, self._shocks = _compile_tree(tree)
        self._N = len(self._nodes)
        limit = _positive_int(max_workspace_bytes, "max_workspace_bytes")
        self._memory_limit = limit
        ss = steady_state
        self.estimated_workspace_bytes = int(8 * self._N * (32 * ss.policy_a.size + 128))
        if self.estimated_workspace_bytes > limit:
            raise ValueError(f"estimated workspace arrays require {self.estimated_workspace_bytes} bytes, "
                             f"above max_workspace_bytes={limit}")
        array_keys = ("asset_grid", "e_grid", "transition", "policy_a", "policy_c", "policy_n",
                      "marginal_value", "distribution")
        if any(np.iscomplexobj(getattr(ss, k)) for k in array_keys):
            raise ValueError("steady-state household arrays must be real")
        arrays = {k: np.array(getattr(ss, k), dtype=float, copy=True) for k in array_keys}
        self._ss = replace(ss, calibration=dict(ss.calibration), aggregates=dict(ss.aggregates),
                           diagnostics=deepcopy(ss.diagnostics), _jacobians={},
                           asset_grid=arrays["asset_grid"], e_grid=arrays["e_grid"], transition=arrays["transition"],
                           policy_a=arrays["policy_a"], policy_c=arrays["policy_c"], policy_n=arrays["policy_n"],
                           marginal_value=arrays["marginal_value"], distribution=arrays["distribution"])
        a, e, P, d = arrays["asset_grid"], arrays["e_grid"], arrays["transition"], arrays["distribution"]
        if (a.ndim != 1 or len(a) < 2 or np.any(np.diff(a) <= 0)
                or e.ndim != 1 or not len(e) or np.any(e <= 0)
                or P.shape != (len(e), len(e)) or np.any(P < 0)
                or not np.allclose(P.sum(axis=1), 1, atol=1e-12, rtol=0)
                or any(arrays[k].shape != (len(a), len(e)) for k in
                       ("policy_a", "policy_c", "policy_n", "marginal_value", "distribution"))
                or any(not np.isfinite(v).all() for v in arrays.values())
                or np.any(d < -1e-12) or abs(float(d.sum()) - 1) > 1e-10
                or np.min(arrays["marginal_value"]) <= 0):
            raise ValueError("invalid steady-state household arrays")
        for v in arrays.values():
            v.flags.writeable = False
        p = self._ss.calibration
        keys = ("beta", "eis", "frisch", "vphi", "mu", "kappa", "phi", "w", "r", "B")
        if (any(k not in p or not np.isfinite(p[k]) for k in keys)
                or any(p[k] <= 0 for k in ("beta", "eis", "frisch", "vphi", "kappa", "w"))
                or p["r"] <= -1 or p["mu"] <= 1):
            raise ValueError("invalid steady-state calibration")
        digest = hashlib.sha256(b"one-asset-finite-history-equilibrium-v1")
        digest.update(json.dumps(p, sort_keys=True).encode())
        digest.update(json.dumps(self._ss.aggregates, sort_keys=True).encode())
        digest.update(json.dumps(list(self._nodes.index)).encode())
        for v in [self._parents, self._prob, *arrays.values()]:
            digest.update(str(v.shape).encode())
            digest.update(np.ascontiguousarray(v).tobytes())
        self._signature = digest.hexdigest()
        self._preconditioner: LinearOperator | None = None
        self._preconditioner_nnz = 0

    @property
    def nodes(self) -> pd.DataFrame:
        """A copy of the prepared tree; edits do not change the workspace."""
        return self._nodes.copy(deep=True)

    @property
    def size(self) -> int:
        """Number of aggregate equilibrium unknowns (three per node)."""
        return 3 * self._N

    def _shock_vector(self, monetary_shocks):
        if monetary_shocks is None:
            return self._shocks.copy()
        if isinstance(monetary_shocks, pd.Series) and not monetary_shocks.index.equals(self._nodes.index):
            raise ValueError("monetary_shocks Series index must match tree row order")
        return _vector(monetary_shocks, self._N, "monetary_shocks")

    def _expected(self, marginal, node, *, tangent=False):
        ch = self._children[node]
        if ch:
            return np.einsum("i,ijk->jk", self._prob[ch], marginal[ch])
        return np.zeros_like(self._ss.marginal_value) if tangent else self._ss.marginal_value

    def _evaluate(self, x, shocks):
        ss, N = self._ss, self._N
        p, a, e, P = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition
        logits = x.reshape(3, N)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pi, w, Y = np.expm1(logits[0]), p["w"] * np.exp(logits[1]), np.exp(logits[2])
            nominal = p["r"] + shocks + p["phi"] * pi
            lag = np.r_[p["r"], nominal[self._parents[1:]]]
            r = (1 + lag) * np.exp(-logits[0]) - 1
            if (not np.isfinite(np.r_[pi, w, Y, nominal, r]).all() or np.any(nominal <= -1)
                    or np.any(r <= -1) or np.any(w <= 0) or np.any(Y <= 0)):
                raise ValueError("equilibrium left the positive-price/gross-return domain")
            adj = p["mu"] / (p["mu"] - 1) / (2 * p["kappa"]) * logits[0]**2 * Y
            div, tax = (1 - w) * Y - adj, p["B"] * r
            nkpc = p["kappa"] * (w - p["w"]) - logits[0]
            for i, ch in enumerate(self._children):
                if ch:
                    nkpc[i] += self._prob[ch] @ (Y[ch] / Y[i] * logits[0, ch] / (1 + r[ch]))
            marginal, saving, c, n = [np.empty((N,) + ss.policy_a.shape) for _ in range(4)]
            for i in range(N - 1, -1, -1):
                marginal[i], saving[i], c[i], n[i] = _backward(
                    self._expected(marginal, i), a, e, P, p, r[i], w[i], div[i] - tax[i])
            d = np.empty_like(saving)
            d[0] = ss.distribution
            for i, ch in enumerate(self._children):
                if ch:
                    d[ch] = _forward(d[i], saving[i], a, P)
            A, C, L = (np.sum(d * saving, axis=(1, 2)), np.sum(d * c, axis=(1, 2)),
                       np.sum(d * n * e, axis=(1, 2)))
            residual = np.r_[nkpc, A - p["B"], L - Y]
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite equilibrium residual")
        levels = dict(pi=pi, w=w, Y=Y, r=r, adj=adj, div=div, tax=tax, A=A, C=C, L=L, nkpc=nkpc)
        return _Evaluation(x.copy(), levels, marginal, saving, c, n, d, residual)

    def residual(self, coordinates: np.ndarray, *, monetary_shocks=None) -> np.ndarray:
        """Full nonlinear residual, ordered Phillips / assets / effective labor."""
        return self._evaluate(_vector(coordinates, self.size, "coordinates"),
                              self._shock_vector(monetary_shocks)).residual

    def _tapes(self, state):
        ss, v = self._ss, state.levels
        hh, forward = [], []
        for i in range(self._N):
            hh.append(_household_tape(self._expected(state.marginal, i), ss.asset_grid, ss.e_grid,
                      ss.transition, ss.calibration, v["r"][i], v["w"][i], v["div"][i] - v["tax"][i]))
            forward.append(_forward_tape(state.saving[i], ss.asset_grid))
        return hh, forward

    def _jvp(self, state, tapes, direction):
        ss, N, v = self._ss, self._N, state.levels
        p, a, e, P = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition
        dx = direction.reshape(3, N)
        pi_dot, w_dot, Y_dot = (1 + v["pi"]) * dx[0], v["w"] * dx[1], v["Y"] * dx[2]
        nominal_dot = p["phi"] * pi_dot
        r_dot = np.r_[0., nominal_dot[self._parents[1:]]] / (1 + v["pi"]) - (1 + v["r"]) * dx[0]
        log_pi = state.x[:N]
        adj_dot = (p["mu"] / (p["mu"] - 1) / (2 * p["kappa"]) * v["Y"]
                   * (2 * log_pi * dx[0] + log_pi**2 * dx[2]))
        transfer_dot = (1 - v["w"]) * Y_dot - v["Y"] * w_dot - adj_dot - p["B"] * r_dot
        nkpc_dot = p["kappa"] * w_dot - dx[0]
        for i, ch in enumerate(self._children):
            if ch:
                scale = v["Y"][ch] / v["Y"][i] / (1 + v["r"][ch])
                nkpc_dot[i] += self._prob[ch] @ (scale * (dx[0, ch] + log_pi[ch] * (
                    dx[2, ch] - dx[2, i] - r_dot[ch] / (1 + v["r"][ch]))))
        dm, da, dn = [np.empty_like(state.marginal) for _ in range(3)]
        for i in range(N - 1, -1, -1):
            dm[i], da[i], _, dn[i] = _household_jvp(
                tapes[0][i], self._expected(dm, i, tangent=True), a, e, P, p,
                v["r"][i], v["w"][i], state.consumption[i], state.hours[i],
                r_dot[i], w_dot[i], transfer_dot[i])
        dd = np.zeros_like(state.distributions)
        asset_dot, labor_dot = np.empty(N), np.empty(N)
        for i, ch in enumerate(self._children):
            d = state.distributions[i]
            asset_dot[i] = np.sum(dd[i] * state.saving[i] + d * da[i])
            labor_dot[i] = np.sum((dd[i] * state.hours[i] + d * dn[i]) * e)
            if ch:
                dd[ch] = _forward_jvp(d, dd[i], da[i], tapes[1][i], P)
        return np.r_[nkpc_dot, asset_dot, labor_dot - Y_dot]

    def jacobian_vector_product(self, coordinates: np.ndarray, direction: np.ndarray, *,
                                monetary_shocks=None) -> np.ndarray:
        """Analytic branch derivative including household and distribution feedback.

        At interpolation/borrowing kinks the current branch is used. Directional
        finite differences crossing those kinks need not match this linear map.
        ``solve`` caches the tapes across Krylov products at the same iterate.
        """
        state = self._evaluate(_vector(coordinates, self.size, "coordinates"),
                               self._shock_vector(monetary_shocks))
        return self._jvp(state, self._tapes(state), _vector(direction, self.size, "direction"))

    def _build_preconditioner(self):
        # Local household price derivatives plus approximate unit carry of
        # aggregate assets. Only a preconditioner: the outer JVP retains all
        # expectations and distribution-composition effects.
        ss, N = self._ss, self._N
        p, a, e, P = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition
        transfer = ss.aggregates["Div"] - ss.aggregates["Tax"]
        _, _, c, n = _backward(ss.marginal_value, a, e, P, p, p["r"], p["w"], transfer)
        tape = _household_tape(ss.marginal_value, a, e, P, p, p["r"], p["w"], transfer)
        local = np.empty((2, 3))
        for k in range(3):
            _, da, _, dn = _household_jvp(tape, np.zeros_like(c), a, e, P, p, p["r"], p["w"],
                                         c, n, *np.eye(3)[k])
            local[:, k] = np.sum(ss.distribution * da), np.sum(ss.distribution * dn * e)
        rows, cols, data = [], [], []

        def put(row, col, value):
            rows.append(row)
            cols.append(col)
            data.append(float(value))

        for i, ch in enumerate(self._children):
            put(i, i, -1.)
            put(i, N + i, p["kappa"] * p["w"])
            for j in ch:
                put(i, j, self._prob[j] / (1 + p["r"]))
            for k, (fr, fw, ft) in enumerate(local):
                row = (k + 1) * N + i
                real_coefficient = fr - p["B"] * ft
                put(row, i, -(1 + p["r"]) * real_coefficient)
                if i:
                    put(row, self._parents[i], p["phi"] * real_coefficient)
                put(row, N + i, (fw - ft) * p["w"])
                put(row, 2 * N + i, ft * (1 - p["w"]) - k)
        matrix = sparse.csc_matrix((data, (rows, cols)), shape=(self.size, self.size))
        try:
            factor = splu(matrix)
        except RuntimeError as exc:
            raise ValueError("HANK sparse preconditioner is singular; try preconditioner='none'") from exc

        def apply(rhs):
            transformed = np.array(rhs, copy=True).reshape(3, N)
            transformed[1, 1:] -= np.asarray(rhs).reshape(3, N)[1, self._parents[1:]]
            return factor.solve(transformed.ravel())

        self._preconditioner = LinearOperator((self.size, self.size), matvec=apply, dtype=float)
        self._preconditioner_nnz = int(factor.L.nnz + factor.U.nnz)

    def _initial(self, initial):
        if initial is None:
            return np.zeros(self.size)
        if isinstance(initial, HANKScenarioTreeResult):
            if initial.metadata.get("equilibrium_workspace_signature") != self._signature:
                raise ValueError("warm-start result must match this workspace's calibration, grid and topology")
            if not initial.nodes.index.equals(self._nodes.index):
                raise ValueError("warm-start result node order does not match")
            p = initial.paths
            with np.errstate(invalid="ignore", divide="ignore"):
                initial = np.r_[np.log1p(p.pi), np.log1p(p.w / self._ss.calibration["w"]), np.log1p(p.Y)]
        return _vector(initial, self.size, "initial")

    def solve(self, *, monetary_shocks=None, initial=None, tolerance: float = 1e-8,
              max_iterations: int = 30, preconditioner: str = "sparse",
              krylov_restart: int = 40, krylov_max_cycles: int = 20) -> HANKScenarioTreeResult:
        """Damped inexact Newton-GMRES with full nonlinear market checks.

        Invalid inputs or initial economic domains raise. Iteration, linear-solve
        and line-search failures return ``converged=False`` and retain the last
        feasible iterate and diagnostic history. A solver exit alone never
        certifies convergence. ``initial`` is a coordinate vector or a result
        from a matching workspace; new shocks use the original node row order.
        """
        if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
            raise ValueError("tolerance must lie in (0,1e-6]")
        max_iterations = _positive_int(max_iterations, "max_iterations")
        restart = min(self.size, _positive_int(krylov_restart, "krylov_restart"))
        cycles = _positive_int(krylov_max_cycles, "krylov_max_cycles")
        estimated_bytes = self.estimated_workspace_bytes + 8 * self.size * max(0, restart - 40)
        if estimated_bytes > self._memory_limit:
            raise ValueError("Krylov restart exceeds max_workspace_bytes working-array budget")
        if preconditioner not in ("sparse", "none"):
            raise ValueError("preconditioner must be 'sparse' or 'none'")
        shocks, x = self._shock_vector(monetary_shocks), self._initial(initial)
        state = self._evaluate(x, shocks)
        evaluations, products, rejected = 1, 0, 0
        reused = preconditioner == "sparse" and self._preconditioner is not None
        history: list[dict[str, Any]] = []
        status = "maximum_iterations"
        for iteration in range(max_iterations + 1):
            norm = float(np.linalg.norm(state.residual))
            goods_error = np.max(np.abs(state.levels["Y"] - state.levels["C"] - state.levels["adj"]))
            if max(float(np.max(np.abs(state.residual))), float(goods_error)) <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            if preconditioner == "sparse" and self._preconditioner is None:
                self._build_preconditioner()
            tapes = self._tapes(state)

            def matvec(v):
                nonlocal products
                products += 1
                return self._jvp(state, tapes, v)

            operator = LinearOperator((self.size, self.size), matvec=matvec, dtype=float)
            forcing = float(min(.1, max(1e-8, .5 * np.sqrt(norm))))
            direction, info, linear_error = preconditioned_lgmres(
                operator, -state.residual, self._preconditioner if preconditioner == "sparse" else None,
                method="gmres", side="right", rtol=forcing, atol=0., inner_m=restart, maxiter=cycles)
            entry = dict(iteration=iteration, residual_max_abs=float(np.max(np.abs(state.residual))),
                         krylov_status=info, linear_residual_norm=linear_error, forcing=forcing,
                         step_length=0.)
            history.append(entry)
            if not np.isfinite(direction).all() or linear_error > forcing * norm * (1 + 1e-10):
                status = "linear_solve_failed"
                break
            accepted = False
            for backtrack in range(17):
                alpha = .5**backtrack
                evaluations += 1
                try:
                    trial = self._evaluate(state.x + alpha * direction, shocks)
                except (ValueError, FloatingPointError, RuntimeError):
                    rejected += 1
                    continue
                if float(np.linalg.norm(trial.residual)) <= (1 - 1e-4 * alpha) * norm:
                    state, accepted = trial, True
                    entry["step_length"] = alpha
                    break
                rejected += 1
            del tapes
            if not accepted:
                status = "line_search_failed"
                break
        diagnostics = dict(optimizer_method="analytic Newton-GMRES", optimizer_success=status == "converged",
                           optimizer_message=status, evaluations=evaluations, jacobian_vector_products=products,
                           newton_iterations=len(history), rejected_trials=rejected, history=history,
                           preconditioner=preconditioner, preconditioner_reused=reused,
                           preconditioner_factor_nnz=self._preconditioner_nnz if preconditioner == "sparse" else 0,
                           dense_jacobian_assembled=False, tolerance=tolerance,
                           estimated_workspace_bytes=estimated_bytes)
        return self._result(state, shocks, diagnostics)

    def _result(self, state, shocks, diagnostics):
        ss, v = self._ss, state.levels
        p, a, e, P = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition
        mass_error = min_mass = top_mass = budget_error = exit_mass = euler_max = euler_mean = 0.
        for i in range(self._N):
            d, saving, c, n = state.distributions[i], state.saving[i], state.consumption[i], state.hours[i]
            post = _forward(d, saving, a, P)
            for mass in (d, post):
                mass_error = max(mass_error, abs(float(mass.sum()) - 1))
                min_mass = min(min_mass, float(mass.min()))
                top_mass = max(top_mass, float(mass[-1].sum()))
            exit_mass = max(exit_mass, float(np.sum(d[(saving < a[0] - 1e-12) | (saving > a[-1])])))
            budget = c + saving - (1 + v["r"][i]) * a[:, None] - v["w"][i] * e * n - (v["div"][i] - v["tax"][i]) * e
            budget_error = max(budget_error, float(np.max(np.abs(budget))))
            rhs_grid = p["beta"] * (self._expected(state.marginal, i) @ P.T)
            rhs = np.empty_like(rhs_grid)
            for z in range(len(e)):
                rhs[:, z] = np.interp(saving[:, z], a, rhs_grid[:, z])
            lhs = c**(-1 / p["eis"])
            violation = np.where(saving <= a[0] + 1e-10, np.maximum(rhs - lhs, 0), np.abs(lhs - rhs)) / lhs
            euler_max = max(euler_max, float(np.max(violation)))
            euler_mean = max(euler_mean, float(np.sum(d * violation)))
        paths = pd.DataFrame(dict(pi=v["pi"], w=v["w"] - p["w"], Y=v["Y"] - 1, C=v["C"] - ss.aggregates["C"],
            A=v["A"] - ss.aggregates["A"], NE=v["L"] - ss.aggregates["NE"], r=v["r"] - p["r"],
            Div=v["div"] - ss.aggregates["Div"], Tax=v["tax"] - ss.aggregates["Tax"],
            goods_mkt=v["Y"] - v["C"] - v["adj"], asset_mkt=v["A"] - p["B"],
            labor_mkt=v["L"] - v["Y"], nkpc_res=v["nkpc"]), index=self._nodes.index.copy())
        residual = float(np.max(np.abs(paths[["asset_mkt", "labor_mkt", "goods_mkt", "nkpc_res"]].to_numpy())))
        converged = bool(diagnostics["optimizer_success"] and residual <= diagnostics["tolerance"]
                         and mass_error < 1e-10 and min_mass >= -1e-12 and budget_error < 1e-9 and exit_mass < 1e-10)
        diagnostics.update(market_residual_max_abs=residual, mass_error=mass_error, minimum_mass=min_mass,
            upper_grid_mass=top_mass, household_budget_error=budget_error, policy_grid_exit_mass=exit_mass,
            euler_kkt_max_relative=euler_max, euler_kkt_max_node_mean=euler_mean,
            euler_scope="linear interpolation at continuous policies; grid exits reported separately",
            minimum_consumption=float(state.consumption.min()), minimum_saving=float(state.saving.min()))
        nodes = self.nodes
        nodes["monetary_shock"] = shocks
        metadata = dict(scope="finite-tree aggregate-risk one-asset HANK with fully coupled household and market equations",
            firm_valuation_kernel="M_child=1/(1+r_child); risk-neutral nominal valuation",
            terminal_condition="steady household marginal value and zero terminal inflation; no solved stochastic tail",
            information="households condition on history and average successor marginal values before choosing",
            income_transition="idiosyncratic chain independent of aggregate branch",
            fiscal_rule="constant real government debt; taxes equal realized net interest times debt",
            validation_scope="equation convergence is separate from asset, income, horizon and aggregate-state approximation",
            derivative_scope="analytic current interpolation and borrowing branch; nonsmooth switches require fresh tapes",
            household_distribution_solved=True, general_stationary_aggregate_risk_solver=False,
            equilibrium_workspace_signature=self._signature)
        return HANKScenarioTreeResult(nodes, paths, state.saving, state.consumption, state.hours,
                                      state.distributions, deepcopy(ss), converged, diagnostics, metadata)


def solve_hank_equilibrium(steady_state: OneAssetHANKSteadyState, tree: pd.DataFrame, *,
                          tolerance: float = 1e-8, max_iterations: int = 30) -> HANKScenarioTreeResult:
    """One-shot matrix-free equilibrium; use a workspace for repeated experiments."""
    return HANKEquilibriumWorkspace(steady_state, tree).solve(tolerance=tolerance, max_iterations=max_iterations)
