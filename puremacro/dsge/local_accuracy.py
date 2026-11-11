"""Evaluate local decision rules and expose their numerical conditioning.

These are raw Taylor policies, suitable for initial guesses and independent
equilibrium-residual comparisons. They are not a replacement for pruned
simulation: pruning retains separate order components of the state history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse.linalg

from ._policy_reporting import _PolicyReport
from ._sylvester import (
    SylvesterWorkCounts,
    _LAST_WORK_COUNTS,
    _schur_blocks,
    solve_generalized_sylvester_schur,
    solve_order1_sylvester,
    solve_order3_sylvester_schur,
)


def evaluate_perturbation(solution: Any, lagged_states: Any, shocks: Any = None,
                          *, sigma: float = 1.0, order: int | None = None,
                          deviations: bool = False) -> np.ndarray:
    """Evaluate an order-1/2/3 Taylor rule in Dynare decision-rule timing.

    ``lagged_states`` contains end-of-previous-period states, in
    ``solution.decision_rules().state_variables`` order. It contains levels
    unless ``deviations=True``. ``shocks`` contains current realized innovations
    in ``shock_names`` order; these are not multiplied by ``sigma``. ``sigma``
    scales anticipated innovation risk in the constant and third-order mixed
    risk terms. Re-solving is necessary to change relative shock covariance.

    Returns endogenous levels in ``variable_names`` order, shape ``(n_vars,)``
    for a single point or ``(n_points,n_vars)`` for a batch. Dense coefficient
    tensors are contracted directly: no Kronecker vectors are allocated.
    """
    rules = solution.decision_rules()
    available = 3 if hasattr(rules, "ghxxx") else 2 if hasattr(rules, "ghxx") else 1
    if order is None:
        order = available
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2, 3) or order > available:
        raise ValueError(f"order must be in 1..{available} for this solution")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    states = np.asarray(lagged_states, dtype=float)
    single = states.ndim == 1
    if single:
        states = states.reshape(1, -1)
    n_states, n_shocks = len(rules.state_variables), len(rules.shock_names)
    variables = list(rules.variable_names)
    n_vars = len(variables)
    if states.ndim != 2 or states.shape[1] != n_states or not np.isfinite(states).all():
        raise ValueError(f"lagged_states must have {n_states} finite columns")
    innovations = np.zeros((len(states), n_shocks)) if shocks is None else np.asarray(shocks, dtype=float)
    if innovations.ndim == 1:
        innovations = np.broadcast_to(innovations, (len(states), n_shocks))
    if innovations.shape != (len(states), n_shocks) or not np.isfinite(innovations).all():
        raise ValueError(f"shocks must have {n_shocks} finite columns and match state rows")
    steady = rules.ys.loc[variables].to_numpy(dtype=float)
    state_steady = rules.ys.loc[list(rules.state_variables)].to_numpy(dtype=float)
    x = states if deviations else states - state_steady
    u = innovations

    def matrix(name: str) -> np.ndarray:
        return getattr(rules, name).loc[variables].to_numpy(dtype=float)

    output = steady + x @ matrix("ghx").T + u @ matrix("ghu").T
    if order >= 2:
        output += 0.5 * np.einsum("ijk,bj,bk->bi", matrix("ghxx").reshape(n_vars, n_states, n_states), x, x, optimize=True)
        output += np.einsum("ijk,bj,bk->bi", matrix("ghxu").reshape(n_vars, n_states, n_shocks), x, u, optimize=True)
        output += 0.5 * np.einsum("ijk,bj,bk->bi", matrix("ghuu").reshape(n_vars, n_shocks, n_shocks), u, u, optimize=True)
        output += 0.5 * sigma ** 2 * matrix("ghs2")
    if order >= 3:
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxxx").reshape(n_vars, n_states, n_states, n_states), x, x, x, optimize=True) / 6
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxxu").reshape(n_vars, n_states, n_states, n_shocks), x, x, u, optimize=True) / 2
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghxuu").reshape(n_vars, n_states, n_shocks, n_shocks), x, u, u, optimize=True) / 2
        output += np.einsum("ijkl,bj,bk,bl->bi", matrix("ghuuu").reshape(n_vars, n_shocks, n_shocks, n_shocks), u, u, u, optimize=True) / 6
        output += 0.5 * sigma ** 2 * (x @ matrix("ghxss").T + u @ matrix("ghuss").T)
    return output[0] if single else output


@dataclass(frozen=True)
class LocalDiagnostics(_PolicyReport):
    """Conditioning evidence for an already solved local approximation.

    Distinguishes small diagonal Schur-block conditioning
    (``schur_block_condition``) from conditioning of the full higher-order
    Sylvester curvature operator (``curvature_operator_condition``), which
    additionally reflects off-diagonal Schur coupling and non-normality of the
    state transition matrix.
    """

    roots: pd.DataFrame
    jacobian_rank: int | None
    equation_count: int | None
    jacobian_condition: float | None
    warnings: tuple[str, ...]
    schur_block_condition: Mapping[int, float] = field(default_factory=dict)
    curvature_operator_condition: Mapping[int, float] = field(default_factory=dict)
    curvature_coupling_amplification: Mapping[int, float] = field(default_factory=dict)
    curvature_condition_method: Mapping[int, str] = field(default_factory=dict)
    risk_operator_condition: float | None = None
    innovation_operator_condition: float | None = None
    sylvester_residuals: Mapping[str, float] = field(default_factory=dict)
    sylvester_work: Mapping[int, SylvesterWorkCounts] = field(default_factory=dict)
    curvature_conditioning: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "order",
                "schur_block_condition",
                "curvature_operator_condition",
                "coupling_amplification",
                "condition_method",
            ]
        )
    )

    def summary(self) -> str:
        warning = "; ".join(self.warnings) if self.warnings else "no conditioning warning detected"
        extra = ""
        if self.schur_block_condition or self.curvature_operator_condition:
            orders = sorted(set(self.schur_block_condition) | set(self.curvature_operator_condition))
            parts = []
            for k in orders:
                sb = self.schur_block_condition.get(k, float("nan"))
                op = self.curvature_operator_condition.get(k, float("nan"))
                parts.append(f"order-{k} Schur-block cond={sb:.3e}, full-operator cond={op:.3e}")
            extra = "; " + "; ".join(parts)
        return (
            f"Local diagnostics: {warning}; dynamic Jacobian rank="
            f"{self.jacobian_rank}/{self.equation_count}{extra}"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return an independent copy of the transition-root diagnostics."""
        return self.roots.copy()

    def curvature_frame(self) -> pd.DataFrame:
        """Return an independent copy of the per-order Schur-block vs full-operator conditioning table."""
        return self.curvature_conditioning.copy()


def _extract_sylvester_pencils(
    solution: Any,
    linear: Any,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract (A_hat, A_plus) for the higher-order Sylvester operator if available."""
    a_hat_direct = getattr(solution, "_A_hat", None)
    if a_hat_direct is None:
        a_hat_direct = getattr(solution, "A_hat", None)
    a_plus_direct = getattr(solution, "_A_plus", None)
    if a_plus_direct is None:
        a_plus_direct = getattr(solution, "A_plus", None)
    if a_hat_direct is not None and a_plus_direct is not None:
        return np.asarray(a_hat_direct, dtype=float), np.asarray(a_plus_direct, dtype=float)

    a_0 = getattr(linear, "_A_0", None)
    a_plus = getattr(linear, "_A_plus", None)
    if a_0 is None or a_plus is None:
        return None
    a_0_arr = np.asarray(a_0, dtype=float)
    a_plus_arr = np.asarray(a_plus, dtype=float)
    n_vars = a_0_arr.shape[0]
    n_states = transition.shape[0]

    ghx = getattr(solution, "ghx", None)
    var_names = getattr(solution, "variable_names", None) or getattr(linear, "variables", None)
    state_names = getattr(solution, "state_names", None) or getattr(linear, "states", None)

    if ghx is None and hasattr(solution, "decision_rules"):
        try:
            rules = solution.decision_rules()
            var_names = list(rules.variable_names)
            state_names = list(rules.state_variables)
            ghx = rules.ghx.loc[var_names].to_numpy(dtype=float)
        except Exception:
            ghx = None

    if ghx is not None and var_names is not None and state_names is not None:
        vars_list = list(var_names)
        states_list = list(state_names)
        ghx_arr = np.asarray(ghx, dtype=float)
        if ghx_arr.shape == (n_vars, n_states) and len(vars_list) == n_vars and len(states_list) == n_states:
            p_s = np.zeros((n_states, n_vars), dtype=float)
            for j, s_name in enumerate(states_list):
                if s_name in vars_list:
                    p_s[j, vars_list.index(s_name)] = 1.0
            a_hat_arr = a_0_arr + a_plus_arr @ ghx_arr @ p_s
            return a_hat_arr, a_plus_arr

    if n_vars == n_states:
        return a_0_arr + a_plus_arr @ transition, a_plus_arr

    return None


def _max_schur_block_condition(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    T_h: np.ndarray,
    blocks: list[tuple[int, int]],
    order: int,
) -> float:
    """Compute the maximum 2-norm condition number across all diagonal Bartels-Stewart Schur blocks."""
    n_x = T_h.shape[0]
    if n_x == 0:
        return float(np.linalg.cond(A_hat)) if A_hat.size > 0 else 1.0

    scalar_cond_cache: dict[float, float] = {}
    max_cond = 1.0

    def eval_block_cond(S: np.ndarray) -> float:
        m = S.shape[0]
        if m == 1:
            tau = 0.0 if float(S[0, 0]) == 0.0 else float(S[0, 0])
            if tau in scalar_cond_cache:
                return scalar_cond_cache[tau]
            cond = float(np.linalg.cond(A_hat + tau * A_plus))
            scalar_cond_cache[tau] = cond
            return cond
        lhs = np.kron(np.eye(m), A_hat) + np.kron(S.T, A_plus)
        return float(np.linalg.cond(lhs))

    if order == 1:
        for i1, e1 in blocks:
            c = eval_block_cond(T_h[i1:e1, i1:e1])
            if c > max_cond or not np.isfinite(c):
                max_cond = c
    elif order == 2:
        for idx1, (i1, e1) in enumerate(blocks):
            T1 = T_h[i1:e1, i1:e1]
            for i2, e2 in blocks[idx1:]:
                T2 = T_h[i2:e2, i2:e2]
                c = eval_block_cond(np.kron(T1, T2))
                if c > max_cond or not np.isfinite(c):
                    max_cond = c
    elif order == 3:
        for idx1, (i1, e1) in enumerate(blocks):
            T1 = T_h[i1:e1, i1:e1]
            for idx2 in range(idx1, len(blocks)):
                i2, e2 = blocks[idx2]
                T2 = T_h[i2:e2, i2:e2]
                T12 = np.kron(T1, T2)
                for idx3 in range(idx2, len(blocks)):
                    i3, e3 = blocks[idx3]
                    T3 = T_h[i3:e3, i3:e3]
                    c = eval_block_cond(np.kron(T12, T3))
                    if c > max_cond or not np.isfinite(c):
                        max_cond = c
    return max_cond


def _full_curvature_operator_condition(
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
    T_h: np.ndarray,
    order: int,
    *,
    max_exact_dim: int = 1024,
) -> tuple[float, str]:
    """Compute or estimate conditioning of the full higher-order Sylvester operator L_k(X) = A_hat X + A_+ X h_x^{\\otimes k}."""
    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    m = int(n_x**order)
    dim = N * m
    if dim == 0:
        return (float(np.linalg.cond(A_hat)) if N > 0 else 1.0), "exact_2norm"

    if dim <= max_exact_dim:
        t_kron = np.array([[1.0]], dtype=float)
        for _ in range(order):
            t_kron = np.kron(t_kron, T_h)
        op_mat = np.kron(np.eye(m), A_hat) + np.kron(t_kron.T, A_plus)
        return float(np.linalg.cond(op_mat)), "exact_2norm"

    # Matrix-free 1-norm condition estimate via LinearOperator + Bartels-Stewart inverse
    def matvec_forward(v: np.ndarray) -> np.ndarray:
        X = v.reshape((N, m), order="F")
        out = A_hat @ X
        if order == 1:
            out = out + (A_plus @ X) @ h_x
        elif order == 2:
            X3 = (A_plus @ X).reshape(N, n_x, n_x)
            t1 = np.tensordot(X3, h_x, axes=(2, 0))
            t2 = np.tensordot(t1, h_x, axes=(1, 0)).transpose(0, 2, 1)
            out = out + t2.reshape(N, m)
        else:
            X4 = (A_plus @ X).reshape(N, n_x, n_x, n_x)
            t1 = np.tensordot(X4, h_x, axes=(3, 0))
            t2 = np.tensordot(t1, h_x, axes=(2, 0))
            t3 = np.tensordot(t2, h_x, axes=(1, 0)).transpose(0, 3, 2, 1)
            out = out + t3.reshape(N, m)
        return out.reshape(dim, order="F")

    def rmatvec_forward(v: np.ndarray) -> np.ndarray:
        X = v.reshape((N, m), order="F")
        out = A_hat.T @ X
        if order == 1:
            out = out + (A_plus.T @ X) @ h_x.T
        elif order == 2:
            X3 = (A_plus.T @ X).reshape(N, n_x, n_x)
            t1 = np.tensordot(X3, h_x.T, axes=(2, 0))
            t2 = np.tensordot(t1, h_x.T, axes=(1, 0)).transpose(0, 2, 1)
            out = out + t2.reshape(N, m)
        else:
            X4 = (A_plus.T @ X).reshape(N, n_x, n_x, n_x)
            t1 = np.tensordot(X4, h_x.T, axes=(3, 0))
            t2 = np.tensordot(t1, h_x.T, axes=(2, 0))
            t3 = np.tensordot(t2, h_x.T, axes=(1, 0)).transpose(0, 3, 2, 1)
            out = out + t3.reshape(N, m)
        return out.reshape(dim, order="F")

    def matvec_inverse(v: np.ndarray) -> np.ndarray:
        K = -v.reshape((N, m), order="F")
        if order == 1:
            sol = solve_order1_sylvester(A_hat, A_plus, h_x, K)
        elif order == 2:
            sol = solve_generalized_sylvester_schur(A_hat, A_plus, h_x, K)
        else:
            sol = solve_order3_sylvester_schur(A_hat, A_plus, h_x, K)
        assert isinstance(sol, np.ndarray)
        return sol.reshape(dim, order="F")

    def rmatvec_inverse(v: np.ndarray) -> np.ndarray:
        K = -v.reshape((N, m), order="F")
        if order == 1:
            sol = solve_order1_sylvester(A_hat.T, A_plus.T, h_x.T, K)
        elif order == 2:
            sol = solve_generalized_sylvester_schur(A_hat.T, A_plus.T, h_x.T, K)
        else:
            sol = solve_order3_sylvester_schur(A_hat.T, A_plus.T, h_x.T, K)
        assert isinstance(sol, np.ndarray)
        return sol.reshape(dim, order="F")

    op_fwd = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec_forward, rmatvec=rmatvec_forward, dtype=float,
    )
    op_inv = scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec_inverse, rmatvec=rmatvec_inverse, dtype=float,
    )
    saved_work = dict(_LAST_WORK_COUNTS)
    try:
        norm_fwd = float(scipy.sparse.linalg.onenormest(op_fwd))
        norm_inv = float(scipy.sparse.linalg.onenormest(op_inv))
        return norm_fwd * norm_inv, "onenormest"
    except np.linalg.LinAlgError:
        return float("inf"), "onenormest"
    finally:
        _LAST_WORK_COUNTS.clear()
        _LAST_WORK_COUNTS.update(saved_work)


def _compute_sylvester_residuals(
    solution: Any,
    A_hat: np.ndarray,
    A_plus: np.ndarray,
    h_x: np.ndarray,
) -> dict[str, float]:
    """Compute normalized algebraic residuals of solved higher-order Sylvester and risk equations."""
    residuals: dict[str, float] = {}
    N = A_hat.shape[0]
    n_x = h_x.shape[0]
    a_hat_norm = float(np.linalg.norm(A_hat, ord=np.inf)) if N > 0 else 0.0
    a_plus_norm = float(np.linalg.norm(A_plus, ord=np.inf)) if N > 0 else 0.0

    K_xx = getattr(solution, "_K_xx", None)
    ghxx = getattr(solution, "ghxx", None)
    if K_xx is not None and ghxx is not None and n_x > 0:
        gxx = np.asarray(ghxx, dtype=float)
        kxx = np.asarray(K_xx, dtype=float)
        gxx_3d = (A_plus @ gxx).reshape(N, n_x, n_x)
        t1 = np.tensordot(gxx_3d, h_x, axes=(2, 0))
        t2 = np.tensordot(t1, h_x, axes=(1, 0)).transpose(0, 2, 1).reshape(N, n_x * n_x)
        res = A_hat @ gxx + t2 + kxx
        scale = max(
            float(np.linalg.norm(kxx, ord=np.inf)),
            (a_hat_norm + a_plus_norm) * float(np.linalg.norm(gxx, ord=np.inf)),
            1.0,
        )
        residuals["ghxx"] = float(np.linalg.norm(res, ord=np.inf)) / scale

    rhs_sig = getattr(solution, "_rhs_sig", None)
    ghs2 = getattr(solution, "ghs2", None)
    if rhs_sig is not None and ghs2 is not None:
        gs2 = np.asarray(ghs2, dtype=float)
        rsig = np.asarray(rhs_sig, dtype=float)
        res = (A_hat + A_plus) @ gs2 - rsig
        scale = max(
            float(np.linalg.norm(rsig, ord=np.inf)),
            (a_hat_norm + a_plus_norm) * float(np.linalg.norm(gs2, ord=np.inf)),
            1.0,
        )
        residuals["ghs2"] = float(np.linalg.norm(res, ord=np.inf)) / scale

    K_xxx = getattr(solution, "_K_xxx", None)
    ghxxx = getattr(solution, "ghxxx", None)
    if K_xxx is not None and ghxxx is not None and n_x > 0:
        gxxx = np.asarray(ghxxx, dtype=float)
        kxxx = np.asarray(K_xxx, dtype=float)
        gxxx_4d = (A_plus @ gxxx).reshape(N, n_x, n_x, n_x)
        t1 = np.tensordot(gxxx_4d, h_x, axes=(3, 0))
        t2 = np.tensordot(t1, h_x, axes=(2, 0))
        t3 = np.tensordot(t2, h_x, axes=(1, 0)).transpose(0, 3, 2, 1).reshape(N, n_x**3)
        res = A_hat @ gxxx + t3 + kxxx
        scale = max(
            float(np.linalg.norm(kxxx, ord=np.inf)),
            (a_hat_norm + a_plus_norm) * float(np.linalg.norm(gxxx, ord=np.inf)),
            1.0,
        )
        residuals["ghxxx"] = float(np.linalg.norm(res, ord=np.inf)) / scale

    K_x_ss = getattr(solution, "_K_x_ss", None)
    ghxss = getattr(solution, "ghxss", None)
    if K_x_ss is not None and ghxss is not None and n_x > 0:
        gxss = np.asarray(ghxss, dtype=float)
        kxss = np.asarray(K_x_ss, dtype=float)
        res = A_hat @ gxss + (A_plus @ gxss) @ h_x + kxss
        scale = max(
            float(np.linalg.norm(kxss, ord=np.inf)),
            (a_hat_norm + a_plus_norm) * float(np.linalg.norm(gxss, ord=np.inf)),
            1.0,
        )
        residuals["ghxss"] = float(np.linalg.norm(res, ord=np.inf)) / scale

    rhs_uss = getattr(solution, "_rhs_uss", None)
    ghuss = getattr(solution, "ghuss", None)
    if rhs_uss is not None and ghuss is not None and np.asarray(ghuss).size > 0:
        guss = np.asarray(ghuss, dtype=float)
        russ = np.asarray(rhs_uss, dtype=float)
        res = A_hat @ guss - russ
        scale = max(
            float(np.linalg.norm(russ, ord=np.inf)),
            a_hat_norm * float(np.linalg.norm(guss, ord=np.inf)),
            1.0,
        )
        residuals["ghuss"] = float(np.linalg.norm(res, ord=np.inf)) / scale

    return residuals


def diagnose_perturbation(
    solution: Any,
    *,
    unit_root_band: float = 1e-5,
    condition_warning: float = 1e10,
    residual_warning: float = 1e-8,
    max_curvature_order: int | None = None,
    max_exact_curvature_dim: int = 1024,
) -> LocalDiagnostics:
    """Report near-unit transition roots, dynamic Jacobian conditioning, and Sylvester curvature conditioning.

    This report does not redo a Blanchard-Kahn classification or establish
    uniqueness of a nonlinear equilibrium. The transition spectrum describes
    the selected local solution; the original Jacobian is inspected only when
    retained by that solution. When higher-order Sylvester pencils are
    available, small diagonal Schur-block conditioning is reported separately
    from the conditioning of the full higher-order curvature operator.
    """
    if not np.isfinite(unit_root_band) or unit_root_band <= 0:
        raise ValueError("unit_root_band must be finite and positive")
    if not np.isfinite(condition_warning) or condition_warning <= 1:
        raise ValueError("condition_warning must exceed one")
    if not np.isfinite(residual_warning) or residual_warning <= 0:
        raise ValueError("residual_warning must be finite and positive")
    if max_curvature_order is not None and (
        isinstance(max_curvature_order, (bool, np.bool_)) or max_curvature_order not in (1, 2, 3)
    ):
        raise ValueError("max_curvature_order must be in {1, 2, 3}")

    linear = getattr(solution, "first_order", None)
    if linear is None:
        linear = solution
    transition = getattr(solution, "G", None)
    if transition is None:
        solved = getattr(linear, "solution", None)
        transition = getattr(solved, "G", None)
    if transition is None:
        raise ValueError("solution does not retain a state transition matrix")
    transition = np.asarray(transition, dtype=float)
    eigenvalues = np.linalg.eigvals(transition)
    modulus = np.abs(eigenvalues)
    roots = pd.DataFrame(
        dict(
            real=eigenvalues.real,
            imaginary=eigenvalues.imag,
            modulus=modulus,
            distance_to_unit_circle=np.abs(modulus - 1),
            near_unit=np.abs(modulus - 1) <= unit_root_band,
            stable=modulus < 1,
        )
    )
    messages = []
    if bool(roots.near_unit.any()):
        messages.append("transition roots near the unit circle; persistence and horizon accuracy are sensitive")
    if bool((~roots.stable).any()):
        messages.append("the selected transition has nonstationary roots")
    rank = count = None
    condition = None
    blocks = [getattr(linear, name, None) for name in ("_A_minus", "_A_0", "_A_plus")]
    if all(block is not None for block in blocks):
        jacobian = np.hstack([np.asarray(block, dtype=float) for block in blocks])
        rank = int(np.linalg.matrix_rank(jacobian))
        count = jacobian.shape[0]
        condition = float(np.linalg.cond(jacobian))
        if rank < count:
            messages.append("dynamic equation Jacobian is row-rank deficient")
        if condition > condition_warning:
            messages.append("dynamic equation Jacobian is ill-conditioned in its supplied units")
    else:
        messages.append("original dynamic Jacobian was not retained; rank/conditioning not checked")

    schur_block_cond: dict[int, float] = {}
    curvature_op_cond: dict[int, float] = {}
    amplification: dict[int, float] = {}
    cond_method: dict[int, str] = {}
    risk_op_cond: float | None = None
    innov_op_cond: float | None = None
    residuals: dict[str, float] = {}
    work_map: dict[int, SylvesterWorkCounts] = dict(getattr(solution, "_sylvester_work", {}) or {})

    pencils = _extract_sylvester_pencils(solution, linear, transition)
    if pencils is not None:
        A_hat, A_plus = pencils
        if max_curvature_order is None:
            if hasattr(solution, "ghxxx"):
                max_k = 3
            elif hasattr(solution, "ghxx"):
                max_k = 2
            else:
                max_k = 2
        else:
            max_k = int(max_curvature_order)

        if A_hat.size > 0:
            innov_op_cond = float(np.linalg.cond(A_hat))
            risk_op_cond = float(np.linalg.cond(A_hat + A_plus))
            if risk_op_cond > condition_warning:
                messages.append("volatility-risk operator (A_hat + A_plus) is ill-conditioned")

        if transition.shape[0] > 0 and A_hat.size > 0:
            T_h, _ = scipy.linalg.schur(transition, output="real")
            schur_blocks = _schur_blocks(T_h)
            for k in range(1, max_k + 1):
                sb_c = _max_schur_block_condition(A_hat, A_plus, T_h, schur_blocks, k)
                op_c, method_name = _full_curvature_operator_condition(
                    A_hat,
                    A_plus,
                    transition,
                    T_h,
                    k,
                    max_exact_dim=max_exact_curvature_dim,
                )
                schur_block_cond[k] = sb_c
                curvature_op_cond[k] = op_c
                cond_method[k] = method_name
                amp = (op_c / sb_c) if (sb_c > 0.0 and np.isfinite(sb_c)) else float("inf")
                amplification[k] = amp
                if sb_c > condition_warning:
                    messages.append(
                        f"order-{k} diagonal Schur block is ill-conditioned (cond={sb_c:.3e})"
                    )
                if op_c > condition_warning:
                    if sb_c <= condition_warning:
                        messages.append(
                            f"order-{k} full curvature operator is ill-conditioned (cond={op_c:.3e}) "
                            f"despite well-conditioned diagonal Schur blocks (cond={sb_c:.3e}); "
                            "off-diagonal Schur coupling amplifies sensitivity"
                        )
                    else:
                        messages.append(
                            f"order-{k} full curvature operator is ill-conditioned (cond={op_c:.3e})"
                        )

        residuals = _compute_sylvester_residuals(solution, A_hat, A_plus, transition)
        for r_name, r_val in residuals.items():
            if not np.isfinite(r_val) or r_val > residual_warning:
                messages.append(
                    f"perturbation Sylvester residual for {r_name} exceeds warning threshold "
                    f"(residual={r_val:.3e} > {residual_warning:.3e})"
                )

    curv_rows = [
        {
            "order": k,
            "schur_block_condition": schur_block_cond[k],
            "curvature_operator_condition": curvature_op_cond[k],
            "coupling_amplification": amplification[k],
            "condition_method": cond_method[k],
        }
        for k in sorted(schur_block_cond)
    ]
    curvature_df = pd.DataFrame(
        curv_rows,
        columns=[
            "order",
            "schur_block_condition",
            "curvature_operator_condition",
            "coupling_amplification",
            "condition_method",
        ],
    )

    return LocalDiagnostics(
        roots=roots,
        jacobian_rank=rank,
        equation_count=count,
        jacobian_condition=condition,
        warnings=tuple(messages),
        schur_block_condition=schur_block_cond,
        curvature_operator_condition=curvature_op_cond,
        curvature_coupling_amplification=amplification,
        curvature_condition_method=cond_method,
        risk_operator_condition=risk_op_cond,
        innovation_operator_condition=innov_op_cond,
        sylvester_residuals=residuals,
        sylvester_work=work_map,
        curvature_conditioning=curvature_df,
    )
