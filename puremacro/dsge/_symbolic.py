r"""Symbolic differentiation engine with Common Subexpression Elimination (CSE) for DSGE models.

Computes exact analytical first-order Jacobians (A_+, A_0, A_-, B_u) and second-order
dynamic Hessian (H_f) over the Expression DAG with respect to dynamic coordinates:
    y_{t+1} (leads), y_t (current), y_{t-1} (lags), and u_t (shocks).

Derives analytical structural sparsity patterns, applies algebraic simplification rules,
discovers repeated sub-DAGs via topological Common Subexpression Elimination (CSE), and
compiles executable Python callables achieving sub-millisecond evaluation under the
zero-dependency Pyodide 4-package contract.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var


def _is_zero(node: Node) -> bool:
    """Check if node is structurally / algebraically identically zero."""
    return isinstance(node, Const) and node.value == 0


def _node_size(n: Node) -> int:
    """Compute number of AST nodes in subtree rooted at n."""
    if isinstance(n, (Var, Param, Const)):
        return 1
    elif isinstance(n, UnaryOp):
        return 1 + _node_size(n.expr)
    elif isinstance(n, BinOp):
        return 1 + _node_size(n.left) + _node_size(n.right)
    elif isinstance(n, Call):
        return 1 + sum(_node_size(a) for a in n.args)
    return 1


def _count_composite_subexprs(node: Node, counter: Counter[Node]) -> None:
    """Count occurrences of composite subexpressions (BinOp, UnaryOp, Call)."""
    if isinstance(node, (BinOp, UnaryOp, Call)):
        counter[node] += 1
        if isinstance(node, UnaryOp):
            _count_composite_subexprs(node.expr, counter)
        elif isinstance(node, BinOp):
            _count_composite_subexprs(node.left, counter)
            _count_composite_subexprs(node.right, counter)
        elif isinstance(node, Call):
            for a in node.args:
                _count_composite_subexprs(a, counter)


def _node_to_py(
    node: Node,
    replacements: Mapping[Node, str],
    var_indices: Mapping[str, int],
    shock_indices: Mapping[str, int],
    param_indices: Mapping[str, int],
    is_defining: bool = False,
) -> str:
    """Translate AST Node into executable Python code string, utilizing CSE replacements."""
    if not is_defining and node in replacements:
        return replacements[node]

    if isinstance(node, Const):
        val = node.value
        if isinstance(val, float):
            if math.isnan(val):
                return "float('nan')"
            if math.isinf(val):
                return "float('inf')" if val > 0 else "float('-inf')"
        return repr(val)

    if isinstance(node, Param):
        if node.name in param_indices:
            idx = param_indices[node.name]
            return f"params[{idx}]"
        return f"params['{node.name}']"

    if isinstance(node, Var):
        if node.name in shock_indices and node.lead == 0:
            idx = shock_indices[node.name]
            return f"shocks[{idx}]"
        if node.name in var_indices:
            idx = var_indices[node.name]
            if node.lead == 1:
                return f"lead[{idx}]"
            elif node.lead == 0:
                return f"curr[{idx}]"
            elif node.lead == -1:
                return f"lag[{idx}]"
        # Fallback for general timing
        if node.lead == 1:
            return f"lead.{node.name}"
        elif node.lead == 0:
            return f"curr.{node.name}"
        elif node.lead == -1:
            return f"lag.{node.name}"
        elif node.lead == 0 and node.name in shock_indices:
            return f"shocks.{node.name}"
        return f"var_{node.name}_{node.lead}"

    if isinstance(node, UnaryOp):
        expr_py = _node_to_py(
            node.expr,
            replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        return f"({node.op}{expr_py})"

    if isinstance(node, BinOp):
        l_py = _node_to_py(
            node.left,
            replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        r_py = _node_to_py(
            node.right,
            replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        py_op = "**" if node.op == "^" else node.op
        return f"({l_py} {py_op} {r_py})"

    if isinstance(node, Call):
        fn = node.func.lower()
        args_py = [
            _node_to_py(
                a,
                replacements,
                var_indices,
                shock_indices,
                param_indices,
                is_defining=False,
            )
            for a in node.args
        ]
        if fn == "exp":
            return f"math.exp({args_py[0]})"
        if fn in ("log", "ln"):
            return f"math.log({args_py[0]})"
        if fn == "log10":
            return f"math.log10({args_py[0]})"
        if fn == "sqrt":
            return f"math.sqrt({args_py[0]})"
        if fn == "cbrt":
            return f"math.pow({args_py[0]}, 1.0 / 3.0)"
        if fn == "sin":
            return f"math.sin({args_py[0]})"
        if fn == "cos":
            return f"math.cos({args_py[0]})"
        if fn == "tan":
            return f"math.tan({args_py[0]})"
        if fn == "asin":
            return f"math.asin({args_py[0]})"
        if fn == "acos":
            return f"math.acos({args_py[0]})"
        if fn == "atan":
            return f"math.atan({args_py[0]})"
        if fn == "sinh":
            return f"math.sinh({args_py[0]})"
        if fn == "cosh":
            return f"math.cosh({args_py[0]})"
        if fn == "tanh":
            return f"math.tanh({args_py[0]})"
        if fn == "erf":
            return f"math.erf({args_py[0]})"
        if fn == "erfc":
            return f"math.erfc({args_py[0]})"
        if fn == "abs":
            return f"abs({args_py[0]})"
        if fn == "sign":
            return f"(1.0 if ({args_py[0]}) > 0.0 else (-1.0 if ({args_py[0]}) < 0.0 else 0.0))"
        if fn == "normpdf":
            return f"(math.exp(-0.5 * ({args_py[0]})**2) * 0.3989422804014327)"
        if fn == "normcdf":
            return f"(0.5 * (1.0 + math.erf(({args_py[0]}) * 0.7071067811865475)))"
        if fn == "steady_state":
            return args_py[0]
        if fn == "expectation":
            return args_py[-1]
        if fn == "max":
            return f"max({', '.join(args_py)})"
        if fn == "min":
            return f"min({', '.join(args_py)})"
        return f"{node.func}({', '.join(args_py)})"

    return str(node)


def _perform_cse(
    expressions: Sequence[Node],
    var_indices: Mapping[str, int],
    shock_indices: Mapping[str, int],
    param_indices: Mapping[str, int],
) -> tuple[list[tuple[str, str]], dict[Node, str]]:
    """Discover repeated sub-DAGs (appearing >= 2 times) and topologically order them."""
    counter: Counter[Node] = Counter()
    for expr in expressions:
        _count_composite_subexprs(expr, counter)

    candidates = [node for node, count in counter.items() if count >= 2]
    # Topological sort: strictly ascending order of node_size ensures all child subexpressions
    # are evaluated before any parent subexpression referencing them.
    candidates.sort(key=lambda n: (_node_size(n), repr(n)))

    replacements: dict[Node, str] = {}
    temp_definitions: list[tuple[str, str]] = []

    for idx, cand in enumerate(candidates):
        t_var = f"_t{idx}"
        rhs_py = _node_to_py(
            cand,
            replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=True,
        )
        temp_definitions.append((t_var, rhs_py))
        replacements[cand] = t_var

    return temp_definitions, replacements


def _normalize_vec(
    v: Any,
    expected_len: int,
    names: Sequence[str],
    defaults: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Normalize input vector to 1D float NumPy array without unnecessary copying."""
    if v is None:
        return np.zeros(expected_len, dtype=float)
    if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == expected_len:
        return v.astype(float, copy=False)
    if isinstance(v, dict):
        return np.array(
            [
                float(
                    v.get(
                        name,
                        v.get(
                            (name, 0),
                            defaults.get(name, 0.0) if defaults else 0.0,
                        ),
                    )
                )
                for name in names
            ],
            dtype=float,
        )
    if hasattr(v, "_values"):
        return np.asarray(v._values, dtype=float)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    return np.array(
        [
            float(getattr(v, name, defaults.get(name, 0.0) if defaults else 0.0))
            for name in names
        ],
        dtype=float,
    )


@dataclass
class SparseDynamicTensor3D:
    """Sparse coordinate representation of 3rd dynamic tensor derivatives T_f.

    Avoids allocating dense N x K^3 tensor (which would consume >650 MB for SW07
    and crash Pyodide). Coordinates are stored for non-decreasing index triplets
    (0 <= p <= q <= r < K) by permutation symmetry.
    """

    shape: tuple[int, int, int, int]
    entries: dict[tuple[int, int, int, int], float]

    def __getitem__(self, key: tuple[int, int, int, int]) -> float:
        i, p, q, r = key
        p, q, r = sorted((p, q, r))
        return self.entries.get((i, p, q, r), 0.0)

    def __len__(self) -> int:
        return len(self.entries)

    def contract(
        self,
        M1: np.ndarray,
        M2: np.ndarray,
        M3: np.ndarray,
        flatten: bool = True,
    ) -> np.ndarray:
        """Contract tensor T_f with matrices M1 (K, d1), M2 (K, d2), M3 (K, d3).

        Computes:
            C_{i, a, b, c} = sum_{p, q, r} T_{f, i, p, q, r} M1_{p, a} M2_{q, b} M3_{r, c}
        summing over all distinct permutations of (p, q, r) without allocating
        any intermediate 4D dense tensors.
        """
        import itertools

        N, K1, K2, K3 = self.shape
        d1 = M1.shape[1]
        d2 = M2.shape[1]
        d3 = M3.shape[1]

        C = np.zeros((N, d1 * d2 * d3), dtype=float)
        if not self.entries:
            return C if flatten else C.reshape((N, d1, d2, d3))

        for (i, p, q, r), v in self.entries.items():
            perms = set(itertools.permutations([p, q, r]))
            for jp, jq, jr in perms:
                v1 = M1[jp]
                v2 = M2[jq]
                v3 = M3[jr]
                C[i] += v * np.kron(np.kron(v1, v2), v3)

        return C if flatten else C.reshape((N, d1, d2, d3))

    def to_dense(self, max_elements: int = 5_000_000) -> np.ndarray:
        """Materialize dense array if within safety limit (raises MemoryError if too large)."""
        import itertools

        N, K1, K2, K3 = self.shape
        total_elements = N * K1 * K2 * K3
        if total_elements > max_elements:
            raise MemoryError(
                f"Dense tensor size {total_elements} elements ({total_elements * 8 / 1e6:.1f} MB) "
                f"exceeds safe threshold {max_elements} ({max_elements * 8 / 1e6:.1f} MB). "
                "Use SparseDynamicTensor3D.contract() instead."
            )
        dense = np.zeros(self.shape, dtype=float)
        for (i, p, q, r), v in self.entries.items():
            for jp, jq, jr in set(itertools.permutations([p, q, r])):
                dense[i, jp, jq, jr] = v
        return dense


@dataclass
class CompiledDerivatives:
    """Compiled analytical derivatives engine with CSE and structural sparsity.

    Provides high-speed analytical evaluation of first-order Jacobians:
        (A_+, A_0, A_-, B_u)
    second-order dynamic Hessian tensor:
        H_f of shape (N, K, K) where K = 3N + n_e,
    and third-order dynamic derivatives:
        T_f as a memory-efficient SparseDynamicTensor3D.
    """

    variables: list[str]
    shocks: list[str]
    parameters: list[str]
    parameter_defaults: dict[str, float]
    sparsity_pattern: dict[str, np.ndarray]
    is_linear: bool
    model_dag: Any
    _eval_first_order_impl: Callable
    _eval_second_order_impl: Callable
    _eval_derivatives_impl: Callable
    _eval_third_order_impl: Callable | None = None

    @property
    def N(self) -> int:
        return len(self.variables)

    @property
    def n_e(self) -> int:
        return len(self.shocks)

    @property
    def K(self) -> int:
        return 3 * self.N + self.n_e

    def eval_first_order(
        self,
        lead: Any = None,
        curr: Any = None,
        lag: Any = None,
        shocks: Any = None,
        params: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate first-order Jacobians A_+, A_0, A_-, B_u at dynamic coordinate point."""
        lead_arr = _normalize_vec(lead, self.N, self.variables)
        curr_arr = _normalize_vec(curr, self.N, self.variables)
        lag_arr = _normalize_vec(lag, self.N, self.variables)
        shk_arr = _normalize_vec(shocks, self.n_e, self.shocks)
        p_arr = _normalize_vec(
            params, len(self.parameters), self.parameters, self.parameter_defaults
        )
        return self._eval_first_order_impl(
            lead_arr, curr_arr, lag_arr, shk_arr, p_arr
        )

    def eval_second_order(
        self,
        lead: Any = None,
        curr: Any = None,
        lag: Any = None,
        shocks: Any = None,
        params: Any = None,
    ) -> np.ndarray:
        """Evaluate second-order dynamic Hessian H_f of shape (N, K, K)."""
        if self.is_linear:
            return np.zeros((self.N, self.K, self.K), dtype=float)
        lead_arr = _normalize_vec(lead, self.N, self.variables)
        curr_arr = _normalize_vec(curr, self.N, self.variables)
        lag_arr = _normalize_vec(lag, self.N, self.variables)
        shk_arr = _normalize_vec(shocks, self.n_e, self.shocks)
        p_arr = _normalize_vec(
            params, len(self.parameters), self.parameters, self.parameter_defaults
        )
        return self._eval_second_order_impl(
            lead_arr, curr_arr, lag_arr, shk_arr, p_arr
        )

    def eval_derivatives(
        self,
        lead: Any = None,
        curr: Any = None,
        lag: Any = None,
        shocks: Any = None,
        params: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate all first-order and second-order derivatives in a single call."""
        lead_arr = _normalize_vec(lead, self.N, self.variables)
        curr_arr = _normalize_vec(curr, self.N, self.variables)
        lag_arr = _normalize_vec(lag, self.N, self.variables)
        shk_arr = _normalize_vec(shocks, self.n_e, self.shocks)
        p_arr = _normalize_vec(
            params, len(self.parameters), self.parameters, self.parameter_defaults
        )
        return self._eval_derivatives_impl(
            lead_arr, curr_arr, lag_arr, shk_arr, p_arr
        )

    def eval_third_order(
        self,
        lead: Any = None,
        curr: Any = None,
        lag: Any = None,
        shocks: Any = None,
        params: Any = None,
    ) -> SparseDynamicTensor3D:
        """Evaluate third-order dynamic derivatives T_f as SparseDynamicTensor3D."""
        if self.is_linear or self._eval_third_order_impl is None:
            return SparseDynamicTensor3D(shape=(self.N, self.K, self.K, self.K), entries={})
        lead_arr = _normalize_vec(lead, self.N, self.variables)
        curr_arr = _normalize_vec(curr, self.N, self.variables)
        lag_arr = _normalize_vec(lag, self.N, self.variables)
        shk_arr = _normalize_vec(shocks, self.n_e, self.shocks)
        p_arr = _normalize_vec(
            params, len(self.parameters), self.parameters, self.parameter_defaults
        )
        return self._eval_third_order_impl(
            lead_arr, curr_arr, lag_arr, shk_arr, p_arr
        )

    @property
    def has_third_order(self) -> bool:
        """Whether third-order dynamic derivatives are compiled."""
        return self._eval_third_order_impl is not None


def compile_derivatives(model_dag: Any) -> CompiledDerivatives:
    """Differentiate model equations symbolically and compile into high-speed CSE callables.

    Parameters
    ----------
    model_dag : ParsedModelDAG
        The parsed model DAG containing equations, variable declarations, and parameters.

    Returns
    -------
    CompiledDerivatives
        Compiled derivatives container providing eval_first_order, eval_second_order,
        and eval_derivatives.
    """
    variables = list(model_dag.variables)
    shocks = list(model_dag.shocks)
    parameters = list(model_dag.parameters)
    parameter_defaults = dict(model_dag.parameter_values)

    N = len(variables)
    n_e = len(shocks)
    K = 3 * N + n_e

    var_indices = {v: idx for idx, v in enumerate(variables)}
    shock_indices = {s: idx for idx, s in enumerate(shocks)}
    param_indices = {p: idx for idx, p in enumerate(parameters)}

    # Dynamic coordinates list z: [y_{t+1}, y_t, y_{t-1}, u_t]
    coords: list[tuple[str, int]] = []
    for v in variables:
        coords.append((v, 1))
    for v in variables:
        coords.append((v, 0))
    for v in variables:
        coords.append((v, -1))
    for s in shocks:
        coords.append((s, 0))
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    # 1. First-order differentiation
    a_plus_entries: list[tuple[int, int, Node]] = []
    a_0_entries: list[tuple[int, int, Node]] = []
    a_minus_entries: list[tuple[int, int, Node]] = []
    b_u_entries: list[tuple[int, int, Node]] = []

    a_plus_sparsity: list[tuple[int, int]] = []
    a_0_sparsity: list[tuple[int, int]] = []
    a_minus_sparsity: list[tuple[int, int]] = []
    b_u_sparsity: list[tuple[int, int]] = []

    for i, eq in enumerate(model_dag.equations):
        eq_vars = eq.variables()

        # Lead variables (+1) -> A_+
        for j, v in enumerate(variables):
            if (v, 1) in eq_vars:
                d = eq.diff(v, 1).simplify()
                a_plus_sparsity.append((i, j))
                if not _is_zero(d):
                    a_plus_entries.append((i, j, d))

        # Current variables (0) -> A_0
        for j, v in enumerate(variables):
            if (v, 0) in eq_vars:
                d = eq.diff(v, 0).simplify()
                a_0_sparsity.append((i, j))
                if not _is_zero(d):
                    a_0_entries.append((i, j, d))

        # Lag variables (-1) -> A_-
        for j, v in enumerate(variables):
            if (v, -1) in eq_vars:
                d = eq.diff(v, -1).simplify()
                a_minus_sparsity.append((i, j))
                if not _is_zero(d):
                    a_minus_entries.append((i, j, d))

        # Shock variables (0) -> B_u
        for m, s in enumerate(shocks):
            if (s, 0) in eq_vars:
                d = eq.diff(s, 0).simplify()
                b_u_sparsity.append((i, m))
                if not _is_zero(d):
                    b_u_entries.append((i, m, d))

    # 2. Second-order dynamic Hessian H_f[i, p, q] = d^2 f_i / (dz_p dz_q)
    h_f_entries: list[tuple[int, int, int, Node]] = []
    h_f_sparsity: list[tuple[int, int, int]] = []

    for i, eq in enumerate(model_dag.equations):
        eq_vars = eq.variables()
        for p, cp in enumerate(coords):
            if cp in eq_vars:
                d1 = eq.diff(cp[0], cp[1]).simplify()
                d1_vars = d1.variables()
                for cq in d1_vars:
                    if cq in coord_to_idx:
                        q = coord_to_idx[cq]
                        if q >= p:
                            d2 = d1.diff(cq[0], cq[1]).simplify()
                            if not _is_zero(d2):
                                h_f_entries.append((i, p, q, d2))
                                h_f_sparsity.append((i, p, q))
                                if p != q:
                                    h_f_sparsity.append((i, q, p))

    # 2b. Third-order dynamic tensor T_f[i, p, q, r] = d^3 f_i / (dz_p dz_q dz_r)
    # Guided by DAG incidence over non-zero branches for p <= q <= r (6x permutation symmetry)
    t_f_entries: list[tuple[int, int, int, int, Node]] = []
    t_f_sparsity: list[tuple[int, int, int, int]] = []

    for i, p, q, d2 in h_f_entries:
        d2_vars = d2.variables()
        for cr in d2_vars:
            if cr in coord_to_idx:
                r = coord_to_idx[cr]
                if r >= q:
                    d3 = d2.diff(cr[0], cr[1]).simplify()
                    if not _is_zero(d3):
                        t_f_entries.append((i, p, q, r, d3))
                        t_f_sparsity.append((i, p, q, r))

    sparsity_pattern = {
        "A_plus": (
            np.array(a_plus_sparsity, dtype=int).reshape(-1, 2)
            if a_plus_sparsity
            else np.zeros((0, 2), dtype=int)
        ),
        "A_0": (
            np.array(a_0_sparsity, dtype=int).reshape(-1, 2)
            if a_0_sparsity
            else np.zeros((0, 2), dtype=int)
        ),
        "A_minus": (
            np.array(a_minus_sparsity, dtype=int).reshape(-1, 2)
            if a_minus_sparsity
            else np.zeros((0, 2), dtype=int)
        ),
        "B_u": (
            np.array(b_u_sparsity, dtype=int).reshape(-1, 2)
            if b_u_sparsity
            else np.zeros((0, 2), dtype=int)
        ),
        "H_f": (
            np.array(h_f_sparsity, dtype=int).reshape(-1, 3)
            if h_f_sparsity
            else np.zeros((0, 3), dtype=int)
        ),
        "T_f": (
            np.array(t_f_sparsity, dtype=int).reshape(-1, 4)
            if t_f_sparsity
            else np.zeros((0, 4), dtype=int)
        ),
    }

    is_linear = len(h_f_entries) == 0 and len(t_f_entries) == 0

    # 3. Common Subexpression Elimination (CSE) across first-, second-, and third-order
    first_order_nodes = (
        [d for _, _, d in a_plus_entries]
        + [d for _, _, d in a_0_entries]
        + [d for _, _, d in a_minus_entries]
        + [d for _, _, d in b_u_entries]
    )
    second_order_nodes = [d2 for _, _, _, d2 in h_f_entries]
    third_order_nodes = [d3 for _, _, _, _, d3 in t_f_entries]
    all_nodes = first_order_nodes + second_order_nodes + third_order_nodes

    # CSE for first order alone (for fastest order-1 Klein solves)
    fo_temps, fo_replacements = _perform_cse(
        first_order_nodes, var_indices, shock_indices, param_indices
    )
    # CSE across all expressions
    all_temps, all_replacements = _perform_cse(
        all_nodes, var_indices, shock_indices, param_indices
    )

    # 4. Code generation for _eval_first_order
    fo_code_lines = [
        "def _compiled_eval_first_order(lead, curr, lag, shocks, params):",
        f"    A_plus = np.zeros(({N}, {N}), dtype=float)",
        f"    A_0 = np.zeros(({N}, {N}), dtype=float)",
        f"    A_minus = np.zeros(({N}, {N}), dtype=float)",
        f"    B_u = np.zeros(({N}, {n_e}), dtype=float)",
    ]
    for tvar, rhs in fo_temps:
        fo_code_lines.append(f"    {tvar} = {rhs}")

    for i, j, d in a_plus_entries:
        val_py = _node_to_py(
            d,
            fo_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        fo_code_lines.append(f"    A_plus[{i}, {j}] = {val_py}")
    for i, j, d in a_0_entries:
        val_py = _node_to_py(
            d,
            fo_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        fo_code_lines.append(f"    A_0[{i}, {j}] = {val_py}")
    for i, j, d in a_minus_entries:
        val_py = _node_to_py(
            d,
            fo_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        fo_code_lines.append(f"    A_minus[{i}, {j}] = {val_py}")
    for i, m, d in b_u_entries:
        val_py = _node_to_py(
            d,
            fo_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        fo_code_lines.append(f"    B_u[{i}, {m}] = {val_py}")
    fo_code_lines.append("    return A_plus, A_0, A_minus, B_u\n")

    # 5. Code generation for _eval_second_order
    so_temps, so_replacements = _perform_cse(
        second_order_nodes, var_indices, shock_indices, param_indices
    )
    so_code_lines = [
        "def _compiled_eval_second_order(lead, curr, lag, shocks, params):",
        f"    H_f = np.zeros(({N}, {K}, {K}), dtype=float)",
    ]
    for tvar, rhs in so_temps:
        so_code_lines.append(f"    {tvar} = {rhs}")

    for i, p, q, d2 in h_f_entries:
        val_py = _node_to_py(
            d2,
            so_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        so_code_lines.append(f"    H_f[{i}, {p}, {q}] = {val_py}")
        if p != q:
            so_code_lines.append(f"    H_f[{i}, {q}, {p}] = H_f[{i}, {p}, {q}]")
    so_code_lines.append("    return H_f\n")

    # 6. Code generation for _eval_derivatives (joint first + second order)
    all_code_lines = [
        "def _compiled_eval_derivatives(lead, curr, lag, shocks, params):",
        f"    A_plus = np.zeros(({N}, {N}), dtype=float)",
        f"    A_0 = np.zeros(({N}, {N}), dtype=float)",
        f"    A_minus = np.zeros(({N}, {N}), dtype=float)",
        f"    B_u = np.zeros(({N}, {n_e}), dtype=float)",
        f"    H_f = np.zeros(({N}, {K}, {K}), dtype=float)",
    ]
    for tvar, rhs in all_temps:
        all_code_lines.append(f"    {tvar} = {rhs}")

    for i, j, d in a_plus_entries:
        val_py = _node_to_py(
            d,
            all_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        all_code_lines.append(f"    A_plus[{i}, {j}] = {val_py}")
    for i, j, d in a_0_entries:
        val_py = _node_to_py(
            d,
            all_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        all_code_lines.append(f"    A_0[{i}, {j}] = {val_py}")
    for i, j, d in a_minus_entries:
        val_py = _node_to_py(
            d,
            all_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        all_code_lines.append(f"    A_minus[{i}, {j}] = {val_py}")
    for i, m, d in b_u_entries:
        val_py = _node_to_py(
            d,
            all_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        all_code_lines.append(f"    B_u[{i}, {m}] = {val_py}")
    for i, p, q, d2 in h_f_entries:
        val_py = _node_to_py(
            d2,
            all_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        all_code_lines.append(f"    H_f[{i}, {p}, {q}] = {val_py}")
        if p != q:
            all_code_lines.append(f"    H_f[{i}, {q}, {p}] = H_f[{i}, {p}, {q}]")
    all_code_lines.append("    return A_plus, A_0, A_minus, B_u, H_f\n")

    # 7. Code generation for _eval_third_order (sparse dynamic 3rd-order tensor)
    to_temps, to_replacements = _perform_cse(
        third_order_nodes, var_indices, shock_indices, param_indices
    )
    to_code_lines = [
        "def _compiled_eval_third_order(lead, curr, lag, shocks, params):",
        "    entries = {}",
    ]
    for tvar, rhs in to_temps:
        to_code_lines.append(f"    {tvar} = {rhs}")

    for i, p, q, r, d3 in t_f_entries:
        val_py = _node_to_py(
            d3,
            to_replacements,
            var_indices,
            shock_indices,
            param_indices,
            is_defining=False,
        )
        to_code_lines.append(f"    _v = float({val_py})")
        to_code_lines.append(f"    if _v != 0.0:")
        to_code_lines.append(f"        entries[({i}, {p}, {q}, {r})] = _v")
    to_code_lines.append(f"    return SparseDynamicTensor3D(shape=({N}, {K}, {K}, {K}), entries=entries)\n")

    compiled_src = "\n".join(
        fo_code_lines + ["\n"] + so_code_lines + ["\n"] + all_code_lines + ["\n"] + to_code_lines
    )
    scope: dict[str, Any] = {
        "np": np,
        "math": math,
        "SparseDynamicTensor3D": SparseDynamicTensor3D,
    }
    exec(compiled_src, scope)

    return CompiledDerivatives(
        variables=variables,
        shocks=shocks,
        parameters=parameters,
        parameter_defaults=parameter_defaults,
        sparsity_pattern=sparsity_pattern,
        is_linear=is_linear,
        model_dag=model_dag,
        _eval_first_order_impl=scope["_compiled_eval_first_order"],
        _eval_second_order_impl=scope["_compiled_eval_second_order"],
        _eval_derivatives_impl=scope["_compiled_eval_derivatives"],
        _eval_third_order_impl=scope["_compiled_eval_third_order"],
    )
