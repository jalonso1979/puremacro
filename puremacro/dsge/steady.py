"""Robust block-triangular steady-state solver and homotopy continuation.

Implements Dynare-parity deterministic steady-state solving under the zero-dependency
Pyodide runtime contract:
- Hopcroft-Karp maximum bipartite matching between equations and variables (O(|E|*sqrt(|V|))).
- Tarjan strongly connected components (SCC) on the matching-induced dependency graph (O(|V|+|E|)).
- Dulmage-Mendelsohn structural singularity decomposition diagnosing over- and under-determined
  subsets when a complete matching fails.
- Topological block-by-block solver: 1D scalar root (Brent / secant) for singletons,
  vector root (hybr / lm / df-sane) for coupled blocks.
- solve_algo menu: "block" (default), "hybr", "lm", "df-sane".
- Homotopy continuation across parameter paths with adaptive step bisection on solver failure.
"""
from __future__ import annotations

import inspect
from collections import deque
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.optimize

from .build import SteadyStateError


class StructuralSingularityError(SteadyStateError):
    """Raised when a DSGE model is structurally singular in steady state.

    Attributes
    ----------
    overdetermined_equations : tuple[str, ...]
        Equations that overconstrain variables.
    overdetermined_variables : tuple[str, ...]
        Variables involved in the overdetermined block.
    underdetermined_equations : tuple[str, ...]
        Equations involved in the underdetermined block.
    underdetermined_variables : tuple[str, ...]
        Variables that cannot be uniquely determined.
    well_determined_equations : tuple[str, ...]
        Equations in the well-determined square core.
    well_determined_variables : tuple[str, ...]
        Variables in the well-determined square core.
    matching_size : int
        Size of the maximum matching found.
    n_equations : int
        Total number of equations.
    n_variables : int
        Total number of variables.
    """

    def __init__(
        self,
        message: str,
        *,
        overdetermined_equations: Sequence[str] = (),
        overdetermined_variables: Sequence[str] = (),
        underdetermined_equations: Sequence[str] = (),
        underdetermined_variables: Sequence[str] = (),
        well_determined_equations: Sequence[str] = (),
        well_determined_variables: Sequence[str] = (),
        matching_size: int = 0,
        n_equations: int = 0,
        n_variables: int = 0,
    ):
        super().__init__(message)
        self.overdetermined_equations = tuple(overdetermined_equations)
        self.overdetermined_variables = tuple(overdetermined_variables)
        self.underdetermined_equations = tuple(underdetermined_equations)
        self.underdetermined_variables = tuple(underdetermined_variables)
        self.well_determined_equations = tuple(well_determined_equations)
        self.well_determined_variables = tuple(well_determined_variables)
        self.matching_size = int(matching_size)
        self.n_equations = int(n_equations)
        self.n_variables = int(n_variables)


class _Vec:
    """Lightweight named vector supporting attribute, key, and index access."""
    __slots__ = ("_names", "_values", "_index", "_what")

    def __init__(self, names: Sequence[str], values: np.ndarray | Sequence[float], what: str = "variable"):
        object.__setattr__(self, "_names", tuple(names))
        object.__setattr__(self, "_values", np.asarray(values, dtype=float))
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(names)})
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[self._index[name]]
        except KeyError:
            raise AttributeError(f"no {self._what} named {name!r}; declared: {list(self._names)}") from None

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            try:
                return self._values[self._index[key]]
            except KeyError:
                raise KeyError(f"no {self._what} named {key!r}; declared: {list(self._names)}") from None
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        body = ", ".join(f"{n}={v:.4g}" for n, v in zip(self._names, self._values))
        return f"<{self._what}s {body}>"


def hopcroft_karp(
    n_u: int,
    n_v: int,
    adj: list[list[int]],
) -> tuple[dict[int, int | None], dict[int, int | None]]:
    """Hopcroft-Karp maximum cardinality bipartite matching in O(|E| * sqrt(|V|))."""
    pair_u: dict[int, int | None] = {u: None for u in range(n_u)}
    pair_v: dict[int, int | None] = {v: None for v in range(n_v)}
    dist: dict[int | None, float] = {}
    inf = float("inf")

    def bfs() -> bool:
        queue: deque[int] = deque()
        for u in range(n_u):
            if pair_u[u] is None:
                dist[u] = 0.0
                queue.append(u)
            else:
                dist[u] = inf
        dist[None] = inf

        while queue:
            u = queue.popleft()
            if dist[u] < dist[None]:
                for v in adj[u]:
                    pu = pair_v[v]
                    if dist.get(pu, inf) == inf:
                        dist[pu] = dist[u] + 1.0
                        if pu is not None:
                            queue.append(pu)
        return dist[None] != inf

    def dfs(u: int | None) -> bool:
        if u is not None:
            for v in adj[u]:
                pu = pair_v[v]
                if dist.get(pu, inf) == dist[u] + 1.0:
                    if dfs(pu):
                        pair_v[v] = u
                        pair_u[u] = v
                        return True
            dist[u] = inf
            return False
        return True

    while bfs():
        for u in range(n_u):
            if pair_u[u] is None:
                dfs(u)

    return pair_u, pair_v


def dulmage_mendelsohn(
    n_u: int,
    n_v: int,
    adj: list[list[int]],
    pair_u: dict[int, int | None],
    pair_v: dict[int, int | None],
) -> dict[str, list[int]]:
    """Dulmage-Mendelsohn decomposition for bipartite graph structural singularity."""
    rev_adj: list[list[int]] = [[] for _ in range(n_v)]
    for u in range(n_u):
        for v in adj[u]:
            rev_adj[v].append(u)

    unmatched_u = [u for u in range(n_u) if pair_u[u] is None]
    unmatched_v = [v for v in range(n_v) if pair_v[v] is None]

    # 1. Overdetermined block: traverse alternating paths from unmatched equations
    visited_u_over: set[int] = set(unmatched_u)
    visited_v_over: set[int] = set()
    queue_u: deque[int] = deque(unmatched_u)
    while queue_u:
        u = queue_u.popleft()
        for v in adj[u]:
            if v not in visited_v_over:
                visited_v_over.add(v)
                pu = pair_v[v]
                if pu is not None and pu not in visited_u_over:
                    visited_u_over.add(pu)
                    queue_u.append(pu)

    # 2. Underdetermined block: traverse alternating paths from unmatched variables
    visited_v_under: set[int] = set(unmatched_v)
    visited_u_under: set[int] = set()
    queue_v: deque[int] = deque(unmatched_v)
    while queue_v:
        v = queue_v.popleft()
        for u in rev_adj[v]:
            if u not in visited_u_under:
                visited_u_under.add(u)
                pv = pair_u[u]
                if pv is not None and pv not in visited_v_under:
                    visited_v_under.add(pv)
                    queue_v.append(pv)

    # 3. Well-determined block: remaining vertices
    all_u = set(range(n_u))
    all_v = set(range(n_v))
    e_well = all_u - visited_u_over - visited_u_under
    v_well = all_v - visited_v_over - visited_v_under

    return {
        "overdetermined_equations": sorted(visited_u_over),
        "overdetermined_variables": sorted(visited_v_over),
        "underdetermined_equations": sorted(visited_u_under),
        "underdetermined_variables": sorted(visited_v_under),
        "well_determined_equations": sorted(e_well),
        "well_determined_variables": sorted(v_well),
    }


def tarjan_scc(n: int, graph: list[list[int]]) -> list[list[int]]:
    """Tarjan strongly connected components algorithm in topological execution order."""
    indices = [-1] * n
    lowlink = [-1] * n
    on_stack = [False] * n
    stack: list[int] = []
    sccs: list[list[int]] = []
    idx = 0

    for root in range(n):
        if indices[root] != -1:
            continue
        call_stack = [(root, 0)]
        indices[root] = lowlink[root] = idx
        idx += 1
        stack.append(root)
        on_stack[root] = True

        while call_stack:
            v, edge_idx = call_stack[-1]
            neighbors = graph[v]
            if edge_idx < len(neighbors):
                call_stack[-1] = (v, edge_idx + 1)
                w = neighbors[edge_idx]
                if indices[w] == -1:
                    indices[w] = lowlink[w] = idx
                    idx += 1
                    stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, 0))
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])
            else:
                call_stack.pop()
                if call_stack:
                    p = call_stack[-1][0]
                    lowlink[p] = min(lowlink[p], lowlink[v])
                if lowlink[v] == indices[v]:
                    scc: list[int] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)

    return sccs


def numeric_incidence_matrix(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x_base: np.ndarray,
    n_points: int = 5,
    tol: float = 1e-8,
    seed: int = 42,
) -> np.ndarray:
    """Construct union numeric incidence matrix across random neighborhood points."""
    x0 = np.asarray(x_base, dtype=float)
    n_v = len(x0)
    r0 = np.asarray(residual_fn(x0), dtype=float)
    n_u = len(r0)

    rng = np.random.RandomState(seed)
    incidence = np.zeros((n_u, n_v), dtype=bool)

    points = [x0]
    for _ in range(max(0, n_points - 1)):
        pert = rng.uniform(-0.05, 0.05, size=n_v) * np.maximum(1.0, np.abs(x0))
        pt = x0 + pert
        points.append(pt)

    for pt in points:
        try:
            r_center = np.asarray(residual_fn(pt), dtype=float)
            if not np.all(np.isfinite(r_center)):
                continue
        except Exception:
            continue

        for j in range(n_v):
            h = 1e-6 * max(1.0, abs(pt[j]))
            pt_plus = pt.copy()
            pt_plus[j] += h
            try:
                r_plus = np.asarray(residual_fn(pt_plus), dtype=float)
            except Exception:
                continue

            pt_minus = pt.copy()
            pt_minus[j] -= h
            try:
                r_minus = np.asarray(residual_fn(pt_minus), dtype=float)
                diff = np.abs(r_plus - r_minus) / (2.0 * h)
            except Exception:
                diff = np.abs(r_plus - r_center) / h

            if np.any(np.isnan(diff)):
                continue
            incidence[:, j] |= (diff > tol)

    # Fallback for any equation with no detected variables: mark variable with max derivative
    for i in range(n_u):
        if not np.any(incidence[i, :]):
            for j in range(n_v):
                h = 1e-6 * max(1.0, abs(x0[j]))
                x_p = x0.copy()
                x_p[j] += h
                try:
                    d = abs(residual_fn(x_p)[i] - r0[i]) / h
                    if d > 1e-12:
                        incidence[i, j] = True
                except Exception:
                    pass

    return incidence


def _params_to_dict(params: Any) -> dict[str, float]:
    if hasattr(params, "_names") and hasattr(params, "_values"):
        return {n: float(v) for n, v in zip(params._names, params._values)}
    if isinstance(params, Mapping):
        return {k: float(v) for k, v in params.items()}
    return {}


def _create_residual_evaluator(
    equations: Callable,
    variables: Sequence[str],
    shocks: Sequence[str] | None,
    default_params: Any,
) -> tuple[Callable[[np.ndarray, Mapping[str, float] | None], np.ndarray], int]:
    """Wrap equations into a standard (x_arr, params) -> resid callable."""
    var_tuple = tuple(variables)
    shock_tuple = tuple(shocks or ())
    def_params = _params_to_dict(default_params)

    # Probe signature
    x_test = np.ones(len(var_tuple))
    x_v = _Vec(var_tuple, x_test, "variable")
    e_v = _Vec(shock_tuple, np.zeros(len(shock_tuple)), "shock")
    p_v = _Vec(list(def_params.keys()), list(def_params.values()), "parameter")

    call_mode = 0  # 5: dynare, 4: build, 2: (x, p), 1: (x)

    try:
        res = equations(x_v, x_v, x_v, e_v, p_v)
        if hasattr(res, "__len__"):
            call_mode = 5
    except TypeError:
        pass

    if call_mode == 0:
        try:
            res = equations(x_v, x_v, e_v, p_v)
            if hasattr(res, "__len__"):
                call_mode = 4
        except TypeError:
            pass

    if call_mode == 0:
        try:
            res = equations(x_test, def_params)
            if hasattr(res, "__len__"):
                call_mode = 2
        except TypeError:
            pass

    if call_mode == 0:
        call_mode = 1

    def evaluate(x_arr: np.ndarray, params: Mapping[str, float] | None = None) -> np.ndarray:
        p_dict = def_params if params is None else _params_to_dict(params)
        if call_mode == 5:
            x_vec = _Vec(var_tuple, x_arr, "variable")
            e_vec = _Vec(shock_tuple, np.zeros(len(shock_tuple)), "shock")
            p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), "parameter")
            return np.asarray(equations(x_vec, x_vec, x_vec, e_vec, p_vec), dtype=float)
        elif call_mode == 4:
            x_vec = _Vec(var_tuple, x_arr, "variable")
            e_vec = _Vec(shock_tuple, np.zeros(len(shock_tuple)), "shock")
            p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), "parameter")
            return np.asarray(equations(x_vec, x_vec, e_vec, p_vec), dtype=float)
        elif call_mode == 2:
            return np.asarray(equations(x_arr, p_dict), dtype=float)
        else:
            return np.asarray(equations(x_arr), dtype=float)

    return evaluate, call_mode


def _solve_singleton_block(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x_curr: np.ndarray,
    var_idx: int,
    eq_idx: int,
    tol: float,
) -> float:
    """Solve a 1D singleton equation for one unknown variable."""
    x0 = float(x_curr[var_idx])

    def g(val: float) -> float:
        x_tmp = x_curr.copy()
        x_tmp[var_idx] = val
        res = residual_fn(x_tmp)
        return float(res[eq_idx])

    g0 = g(x0)
    if abs(g0) <= tol:
        return x0

    # Try bracket search for brentq
    bracket = None
    deltas = [0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    for d in deltas:
        for val_other in (x0 - d, x0 + d):
            try:
                g_other = g(val_other)
                if np.isfinite(g_other) and g0 * g_other < 0:
                    a, b = min(x0, val_other), max(x0, val_other)
                    bracket = (a, b)
                    break
            except Exception:
                continue
        if bracket is not None:
            break

    if bracket is not None:
        try:
            sol = scipy.optimize.root_scalar(g, bracket=bracket, method="brentq", xtol=1e-12, rtol=1e-12)
            if sol.converged:
                return float(sol.root)
        except Exception:
            pass

    # Try secant
    x1 = x0 + 0.01 if abs(x0) < 1.0 else x0 * 1.01
    try:
        sol = scipy.optimize.root_scalar(g, x0=x0, x1=x1, method="secant", xtol=1e-12)
        if sol.converged and abs(g(sol.root)) <= max(tol, 1e-6):
            return float(sol.root)
    except Exception:
        pass

    # Fallback to 1D hybr
    sol_root = scipy.optimize.root(lambda v: [g(v[0])], [x0], method="hybr", tol=1e-12)
    if sol_root.success:
        return float(sol_root.x[0])

    # Fallback to relaxed tol hybr
    sol_root2 = scipy.optimize.root(lambda v: [g(v[0])], [x0], method="hybr", tol=tol)
    if sol_root2.success:
        return float(sol_root2.x[0])

    return x0


def _solve_coupled_block(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x_curr: np.ndarray,
    var_indices: list[int],
    eq_indices: list[int],
    sub_algo: str,
    tol: float,
) -> np.ndarray:
    """Solve a coupled block of m equations in m variables."""
    m_vars = list(var_indices)
    m_eqs = list(eq_indices)
    xi_0 = x_curr[m_vars].copy()

    def block_res(v: np.ndarray) -> np.ndarray:
        x_tmp = x_curr.copy()
        x_tmp[m_vars] = v
        r = residual_fn(x_tmp)
        return r[m_eqs]

    method = "hybr" if sub_algo == "block" else sub_algo
    sol = scipy.optimize.root(block_res, xi_0, method=method, tol=1e-12)
    if not sol.success or np.max(np.abs(sol.fun)) > tol:
        relaxed = scipy.optimize.root(block_res, xi_0, method=method, tol=tol)
        if relaxed.success and np.max(np.abs(relaxed.fun)) <= tol:
            sol = relaxed

    if not sol.success and method != "lm":
        lm_sol = scipy.optimize.root(block_res, xi_0, method="lm")
        if lm_sol.success and np.max(np.abs(lm_sol.fun)) <= tol:
            sol = lm_sol

    return np.asarray(sol.x, dtype=float)


def _solve_block_system(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    variables: Sequence[str],
    equation_names: Sequence[str] | None,
    tol: float,
    max_iter: int = 100,
) -> tuple[np.ndarray, dict]:
    """Execute Hopcroft-Karp + Tarjan block-triangular steady-state solve."""
    n_vars = len(variables)
    eq_names = list(equation_names) if equation_names else [f"eq_{i+1}" for i in range(n_vars)]

    # 0. Check initial residual finiteness
    try:
        r_init = np.asarray(residual_fn(x_init), dtype=float)
        if not np.all(np.isfinite(r_init)):
            return _solve_direct_system(residual_fn, x_init, method="hybr", tol=tol)
    except Exception:
        return _solve_direct_system(residual_fn, x_init, method="hybr", tol=tol)

    # 1. Build 5-point numeric incidence matrix
    inc = numeric_incidence_matrix(residual_fn, x_init, n_points=5)
    adj = [[j for j in range(n_vars) if inc[i, j]] for i in range(n_vars)]

    # 2. Hopcroft-Karp maximum bipartite matching
    pair_u, pair_v = hopcroft_karp(n_vars, n_vars, adj)
    matching_size = sum(1 for u in pair_u if pair_u[u] is not None)

    # 3. Check for structural singularity
    if matching_size < n_vars:
        # Fallback to direct solver (e.g. hybr) if system can be solved directly
        # (for instance, models with unit roots whose steady state level is pinned by guess).
        try:
            sol_direct, info_direct = _solve_direct_system(residual_fn, x_init, method="hybr", tol=tol)
            if info_direct.get("converged", False) and float(info_direct.get("max_residual", 1.0)) <= tol:
                info_direct["fallback_from_block"] = True
                return sol_direct, info_direct
        except Exception:
            pass

        dm = dulmage_mendelsohn(n_vars, n_vars, adj, pair_u, pair_v)
        over_eqs = [eq_names[i] for i in dm["overdetermined_equations"]]
        over_vars = [variables[j] for j in dm["overdetermined_variables"]]
        under_eqs = [eq_names[i] for i in dm["underdetermined_equations"]]
        under_vars = [variables[j] for j in dm["underdetermined_variables"]]
        well_eqs = [eq_names[i] for i in dm["well_determined_equations"]]
        well_vars = [variables[j] for j in dm["well_determined_variables"]]

        lines = [
            f"Model is structurally singular in steady state (maximum bipartite matching size: {matching_size} / {n_vars})."
        ]
        if over_eqs:
            lines.append(
                f"- Over-determined block ({len(over_eqs)} equations constrain {len(over_vars)} variables):"
            )
            lines.append(f"  Equations: {over_eqs}")
            lines.append(f"  Variables: {over_vars}")
        if under_vars:
            lines.append(
                f"- Under-determined block ({len(under_vars)} variables appear in only {len(under_eqs)} equations):"
            )
            lines.append(f"  Variables: {under_vars}")
            lines.append(f"  Equations: {under_eqs}")
        if well_eqs:
            lines.append(
                f"- Well-determined core: {len(well_eqs)} equations in {len(well_vars)} variables."
            )

        raise StructuralSingularityError(
            "\n".join(lines),
            overdetermined_equations=over_eqs,
            overdetermined_variables=over_vars,
            underdetermined_equations=under_eqs,
            underdetermined_variables=under_vars,
            well_determined_equations=well_eqs,
            well_determined_variables=well_vars,
            matching_size=matching_size,
            n_equations=n_vars,
            n_variables=n_vars,
        )

    # 4. Dependency graph on variables: variable j depends on variable k if k enters pair_v[j]
    dep_graph = [[] for _ in range(n_vars)]
    for j in range(n_vars):
        eq_idx = pair_v[j]
        assert eq_idx is not None
        for k in adj[eq_idx]:
            if k != j:
                dep_graph[j].append(k)

    # 5. Tarjan SCC in topological execution order
    sccs = tarjan_scc(n_vars, dep_graph)

    # 6. Block-by-block sequential solve
    x_curr = np.asarray(x_init, dtype=float).copy()
    blocks_info = []
    for block in sccs:
        b_vars = list(block)
        b_eqs = [pair_v[j] for j in b_vars]
        blocks_info.append(tuple(variables[j] for j in b_vars))

        if len(b_vars) == 1:
            j = b_vars[0]
            i = b_eqs[0]
            assert i is not None
            x_curr[j] = _solve_singleton_block(residual_fn, x_curr, j, i, tol)
        else:
            eq_list = [i for i in b_eqs if i is not None]
            x_curr[b_vars] = _solve_coupled_block(residual_fn, x_curr, b_vars, eq_list, "hybr", tol)

    # Final residual check
    r_final = residual_fn(x_curr)
    max_res = float(np.max(np.abs(r_final)))
    if max_res > tol:
        # Full-system polish attempt
        polish = scipy.optimize.root(residual_fn, x_curr, method="hybr", tol=tol)
        if polish.success and np.max(np.abs(polish.fun)) <= tol:
            x_curr = np.asarray(polish.x, dtype=float)
            r_final = polish.fun
            max_res = float(np.max(np.abs(r_final)))

    if max_res > tol:
        raise SteadyStateError(
            f"steady state did not converge from the supplied guess: "
            f"max residual {max_res:.3e} > tol {tol:.3e}."
        )

    info = {
        "solve_algo": "block",
        "converged": True,
        "max_residual": max_res,
        "blocks": blocks_info,
        "n_blocks": len(sccs),
        "matching": {eq_names[i]: variables[j] for i, j in pair_u.items() if j is not None},
    }
    return x_curr, info


def _solve_direct_system(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    method: str,
    tol: float,
) -> tuple[np.ndarray, dict]:
    """Execute direct full-system root solve via hybr, lm, or df-sane."""
    x0 = np.asarray(x_init, dtype=float)
    if method == "hybr":
        sol = scipy.optimize.root(residual_fn, x0, method="hybr", tol=1e-12)
        if not sol.success:
            relaxed = scipy.optimize.root(residual_fn, x0, method="hybr", tol=tol)
            if relaxed.success:
                sol = relaxed
    else:
        sol = scipy.optimize.root(residual_fn, x0, method=method, tol=tol)

    max_res = float(np.max(np.abs(sol.fun))) if (hasattr(sol, "fun") and np.all(np.isfinite(sol.fun))) else float("inf")
    if max_res > tol:
        raise SteadyStateError(
            f"steady state did not converge from the supplied guess: {getattr(sol, 'message', 'tolerance not reached')}"
        )

    info = {
        "solve_algo": method,
        "converged": True,
        "max_residual": max_res,
        "iterations": getattr(sol, "nfev", 0),
    }
    return np.asarray(sol.x, dtype=float), info


def steady(
    equations: Callable,
    variables: Sequence[str],
    guess: Sequence[float] | Mapping[str, float] | np.ndarray,
    params: Mapping[str, float] | None = None,
    *,
    shocks: Sequence[str] | None = None,
    solve_algo: str = "block",
    homotopy: Mapping[str, tuple[float, float]] | None = None,
    homotopy_steps: int = 10,
    tol: float = 1e-8,
    max_iter: int = 100,
    equation_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve for the deterministic steady state of a DSGE model."""
    valid_algos = ("block", "hybr", "lm", "df-sane")
    if solve_algo not in valid_algos:
        raise ValueError(
            f"unknown solve_algo {solve_algo!r}; must be one of {list(valid_algos)}"
        )

    var_list = list(variables)
    n_vars = len(var_list)

    if equation_names is not None and len(equation_names) != n_vars:
        raise ValueError(
            f"equation_names length {len(equation_names)} does not match number of variables {n_vars}"
        )

    # Initial guess array
    if isinstance(guess, Mapping):
        missing = [v for v in var_list if v not in guess]
        if missing:
            raise SteadyStateError(f"guess is missing values for {missing}")
        x0 = np.array([float(guess[v]) for v in var_list], dtype=float)
    else:
        x0 = np.asarray(guess, dtype=float)
        if len(x0) != n_vars:
            raise SteadyStateError(
                f"guess length {len(x0)} does not match number of variables {n_vars}"
            )

    evaluator, _ = _create_residual_evaluator(equations, var_list, shocks, params)

    def _solve_at_p(p_current: Mapping[str, float] | None, x_start: np.ndarray) -> tuple[np.ndarray, dict]:
        res_fn = lambda x: evaluator(x, p_current)
        if solve_algo == "block":
            return _solve_block_system(
                res_fn,
                x_start,
                variables=var_list,
                equation_names=equation_names,
                tol=tol,
                max_iter=max_iter,
            )
        else:
            return _solve_direct_system(
                res_fn,
                x_start,
                method=solve_algo,
                tol=tol,
            )

    # Path 1: Direct solve (no homotopy)
    if not homotopy:
        return _solve_at_p(params, x0)

    # Path 2: Homotopy continuation with adaptive step bisection
    p_base = _params_to_dict(params)
    for k in homotopy:
        if k not in p_base:
            p_base[k] = homotopy[k][0]

    # Initial solve at s = 0.0
    p_start = dict(p_base)
    for k, (v_start, _) in homotopy.items():
        p_start[k] = float(v_start)

    try:
        x_curr, info_start = _solve_at_p(p_start, x0)
    except SteadyStateError as e:
        raise SteadyStateError(
            f"Homotopy initialization failed at s=0.0: {e}"
        ) from e

    s = 0.0
    step = 1.0 / max(1, int(homotopy_steps))
    min_step = 1e-4
    max_step = 0.5
    bisections = 0
    total_steps = 0

    while s < 1.0 - 1e-12:
        s_try = min(1.0, s + step)
        p_try = dict(p_base)
        for k, (v_start, v_target) in homotopy.items():
            p_try[k] = float((1.0 - s_try) * v_start + s_try * v_target)

        try:
            x_new, _ = _solve_at_p(p_try, x_curr)
            success = True
        except SteadyStateError:
            success = False

        if success:
            s = s_try
            x_curr = x_new
            total_steps += 1
            step = min(step * 2.0, max_step, 1.0 - s)
        else:
            step /= 2.0
            bisections += 1
            if step < min_step:
                raise SteadyStateError(
                    f"Homotopy continuation stalled at s={s:.4f} with step size {step:.2e} < min_step {min_step:.2e}. "
                    f"Failed to reach target parameters."
                )

    # Verify at target parameters
    p_target = dict(p_base)
    for k, (_, v_target) in homotopy.items():
        p_target[k] = float(v_target)

    r_final = evaluator(x_curr, p_target)
    max_res = float(np.max(np.abs(r_final)))

    info = {
        "solve_algo": solve_algo,
        "converged": True,
        "max_residual": max_res,
        "homotopy_steps": total_steps,
        "bisections": bisections,
    }
    return x_curr, info
