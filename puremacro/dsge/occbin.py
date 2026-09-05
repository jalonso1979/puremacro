"""Piecewise-Linear Solution for Occasionally Binding Constraints (OccBin).

Implementation of Guerrieri & Iacoviello (2015), "OccBin: A Toolkit for
Solving Dynamic Models with Occasionally Binding Constraints Easily",
Journal of Monetary Economics 70, 22-38.

Solves models with occasionally binding constraints (such as the Zero Lower
Bound on nominal interest rates, borrowing limits, or irreversible investment)
under perfect foresight via backward recursion over piecewise-linear regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge.build import LinearModel, _Vec, _CSTEP, _FDSTEP


@dataclass
class OccBinConstraint:
    """Occasionally binding constraint definition.

    Parameters
    ----------
    variable : str
        Name of the endogenous variable subject to the constraint (e.g. 'r').
    threshold : float
        Numerical boundary value (e.g. -r_ss for nominal interest rate floor).
    operator : {'<', '<=', '>', '>='}, default '<'
        Inequality operator defining when the constraint becomes binding.
        For example, 'r < -r_ss' binds when the nominal rate falls below floor.
    relax_variable : str, optional
        Auxiliary variable name tracking the shadow/notional rate or multiplier
        to test when the constraint relaxes (e.g. 'rnot'). If None, the shadow
        value is automatically deduced from the reference regime equation.
    relax_threshold : float, optional
        Threshold for the relax condition. Defaults to ``threshold``.
    relax_operator : {'>', '>=', '<', '<='}, optional
        Operator for relaxation. Defaults to opposite of ``operator``.
    """

    variable: str
    threshold: float
    operator: str = "<"
    relax_variable: str | None = None
    relax_threshold: float | None = None
    relax_operator: str | None = None

    def __post_init__(self):
        valid_ops = {"<", "<=", ">", ">="}
        if self.operator not in valid_ops:
            raise ValueError(f"invalid operator {self.operator!r}; expected one of {valid_ops}")
        if self.relax_threshold is None:
            self.relax_threshold = self.threshold
        if self.relax_operator is None:
            inv_map = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}
            self.relax_operator = inv_map[self.operator]

    def evaluate(self, val: float | np.ndarray) -> bool | np.ndarray:
        """Check if the binding condition is satisfied (constraint binds)."""
        if self.operator == "<":
            return val < self.threshold
        elif self.operator == "<=":
            return val <= self.threshold
        elif self.operator == ">":
            return val > self.threshold
        elif self.operator == ">=":
            return val >= self.threshold
        return False

    def evaluate_relax(self, val: float | np.ndarray) -> bool | np.ndarray:
        """Check if the relaxation condition is satisfied (constraint relaxes)."""
        op = self.relax_operator
        thresh = self.relax_threshold
        if thresh is None or op is None:
            return False
        if op == ">=":
            return val >= thresh
        elif op == ">":
            return val > thresh
        elif op == "<=":
            return val <= thresh
        elif op == "<":
            return val < thresh
        return False

    def __repr__(self) -> str:
        return f"OccBinConstraint({self.variable} {self.operator} {self.threshold})"


@dataclass(frozen=True)
class OccBinResult:
    """Result of an OccBin simulation for occasionally binding constraints.

    Attributes
    ----------
    simulated_path : pd.DataFrame
        Simulated trajectory of all endogenous variables over the horizon.
    regimes : list[int]
        Regime indicator for each period (0 = reference regime, 1 = constrained regime).
    binding_periods : int
        Number of periods during which the constraint is binding.
    converged : bool
        Whether the Guerrieri & Iacoviello backward-recursion algorithm converged.
    iterations : int
        Number of iterations executed until convergence or termination.
    reference_model : Any
        Underlying unconstrained reference model.
    constrained_model : Any
        Underlying model representing the constrained regime.
    constraint : OccBinConstraint, optional
        The constraint definition.
    shadow_path : pd.DataFrame, optional
        Simulated path including shadow/notional variables during the binding spell.
    """

    simulated_path: pd.DataFrame
    regimes: list[int]
    binding_periods: int
    converged: bool
    iterations: int
    reference_model: Any
    constrained_model: Any
    constraint: OccBinConstraint | None = None
    shadow_path: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return simulated path as a DataFrame."""
        return self.simulated_path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        """Allow subscript access to simulated variable series."""
        if key in self.simulated_path.columns:
            return self.simulated_path[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"OccBinResult has no variable or attribute {key!r}")

    def summary(self) -> str:
        """Render a formatted summary of the OccBin simulation."""
        status = "Converged" if self.converged else "Did NOT converge"
        horizon = len(self.simulated_path)
        c_desc = repr(self.constraint) if self.constraint else "Unspecified"

        lines = [
            "OCCASIONALLY BINDING CONSTRAINTS REPORT (OccBin - Guerrieri & Iacoviello 2015)",
            "=" * 78,
            f"Constraint         : {c_desc}",
            f"Algorithm status   : {status} in {self.iterations} iteration(s)",
            f"Binding duration   : {self.binding_periods} period(s) out of {horizon}",
            f"Regime sequence    : {self.regimes[:min(horizon, 20)]}{'...' if horizon > 20 else ''}",
            "-" * 78,
            "TRAJECTORY SUMMARY STATISTICS",
            "-" * 78,
        ]

        stats_df = pd.DataFrame(
            {
                "Impact (t=1)": self.simulated_path.iloc[0],
                "Min": self.simulated_path.min(),
                "Max": self.simulated_path.max(),
                "Mean": self.simulated_path.mean(),
                "Final (t=H)": self.simulated_path.iloc[-1],
            }
        )
        lines.append(stats_df.round(6).to_string())
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export simulated path to Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export simulated path to LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export simulated path to Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        style: str = "publication",
    ):
        """Plot the simulated trajectories with highlighted binding regimes.

        Parameters
        ----------
        variables : Sequence[str], optional
            Variables to include. If None, plots all endogenous variables.
        style : {'publication', 'default'}, default 'publication'
            Plot style. If 'publication', uses black-and-white publication styling
            from puremacro.plotting.bw_style.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        import matplotlib.pyplot as plt

        if variables is None:
            variables = list(self.simulated_path.columns)
        else:
            variables = [v for v in variables if v in self.simulated_path.columns]

        n_vars = len(variables)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        if style == "publication":
            from puremacro.plotting.bw_style import apply_bw_style, bw_colors, bw_linestyles

            apply_bw_style()
            colors: list[str | None] = list(bw_colors(n_vars))
            styles: list[str] = list(bw_linestyles(n_vars))
        else:
            colors = [None] * n_vars
            styles = ["-"] * n_vars

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.2 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        horizon = len(self.simulated_path)
        time_grid = np.arange(1, horizon + 1)

        for i, var in enumerate(variables):
            ax = axes_flat[i]
            c = colors[i] if colors[i] is not None else "black"
            ls = styles[i] if styles[i] is not None else "-"
            ax.plot(time_grid, self.simulated_path[var], label=var, color=c, linestyle=ls, linewidth=1.5)

            # Highlight binding regime periods
            binding_mask = np.array(self.regimes[:horizon]) == 1
            if np.any(binding_mask):
                diff = np.diff(np.pad(binding_mask.astype(int), (1, 1), "constant"))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0]
                for s, e in zip(starts, ends):
                    ax.axvspan(s - 0.5, e + 0.5, color="0.85", alpha=0.3, label="Constrained" if i == 0 else None)

            # Threshold line for constrained variable
            if self.constraint is not None and var == self.constraint.variable:
                ax.axhline(
                    self.constraint.threshold,
                    color="0.4",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Threshold ({self.constraint.threshold:.3g})",
                )

            ax.axhline(0, color="0.6", linestyle=":", linewidth=0.6)
            ax.set_title(var)
            ax.set_xlabel("Period")
            ax.legend(loc="best", frameon=False, fontsize=8)

        # Hide extra unused subplots
        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Internal helpers for Jacobian and decision rule extraction
# ---------------------------------------------------------------------------


def _extract_model_matrices(model: Any, ref_model: Any | None = None) -> tuple:
    """Extract canonical lead-lag system matrices: A_plus, A_0, A_minus, B_u, const, steady_state.

    Canonical representation:
        A_+ E_t X_{t+1} + A_0 X_t + A_- X_{t-1} + B_u u_t + c = 0
    """
    if isinstance(model, LinearModel):
        variables = list(model.variables)
        shocks = list(model.shocks)
        ss_arr = np.array([float(model.steady_state[v]) for v in variables])
        n_vars = len(variables)
        n_shocks = len(shocks)

        # If canonical lead-lag Jacobians were already cached:
        if (
            model._A_plus is not None
            and model._A_0 is not None
            and model._A_minus is not None
            and model._B_u is not None
        ):
            if ref_model is not None and model._dynare_equations is not None:
                ref_ss = np.array([float(ref_model.steady_state[v]) for v in variables])
                par_dict = model._params or {}
                par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")
                y_ss_v = _Vec(variables, ref_ss)
                e_0_v = _Vec(shocks, np.zeros(n_shocks))
                res = model._dynare_equations(y_ss_v, y_ss_v, y_ss_v, e_0_v, par_vec)
                c = np.asarray(res, dtype=float)
            else:
                c = np.zeros(n_vars)
            return (
                model._A_plus.copy(),
                model._A_0.copy(),
                model._A_minus.copy(),
                model._B_u.copy(),
                c,
                ss_arr,
                variables,
                shocks,
            )

        # Re-differentiate if _dynare_equations is available
        if model._dynare_equations is not None:
            return _differentiate_dynare_eqs(
                model._dynare_equations,
                variables=variables,
                shocks=shocks,
                params=model._params or {},
                steady_state=ss_arr,
                ref_model=ref_model,
            )

        # Fallback to Klein form
        A_p = model.A.copy()
        A_0 = -model.B.copy()
        A_m = np.zeros_like(A_0)
        B_u = -model.C.copy()
        c = np.zeros(n_vars)
        return A_p, A_0, A_m, B_u, c, ss_arr, variables, shocks

    elif callable(model):
        if ref_model is None or not isinstance(ref_model, LinearModel):
            raise ValueError("When constrained_model is a callable, reference_model must be a LinearModel")
        variables = list(ref_model.variables)
        shocks = list(ref_model.shocks)
        ref_ss = np.array([float(ref_model.steady_state[v]) for v in variables])
        return _differentiate_dynare_eqs(
            model,
            variables=variables,
            shocks=shocks,
            params=ref_model._params or {},
            steady_state=ref_ss,
            ref_model=ref_model,
        )

    raise TypeError(f"unsupported model type: {type(model)}; expected LinearModel or callable")


def _differentiate_dynare_eqs(
    equations: Callable,
    variables: list[str],
    shocks: list[str],
    params: Mapping[str, float],
    steady_state: np.ndarray,
    ref_model: Any | None = None,
) -> tuple:
    """Differentiate lead-lag equations via complex-step differentiation."""
    n_vars = len(variables)
    n_shocks = len(shocks)
    par_dict = dict(params or {})
    par_vec = _Vec(list(par_dict.keys()), list(par_dict.values()), what="parameter")

    eval_ss = (
        np.array([float(ref_model.steady_state[v]) for v in variables])
        if ref_model is not None
        else steady_state.copy()
    )

    step = _CSTEP
    base_ss = np.asarray(eval_ss, dtype=complex)
    base_e = np.zeros(n_shocks, dtype=complex)

    A_plus = np.zeros((n_vars, n_vars))
    A_0 = np.zeros((n_vars, n_vars))
    A_minus = np.zeros((n_vars, n_vars))
    B_u = np.zeros((n_vars, n_shocks))

    for j in range(n_vars):
        pert_p = base_ss.copy()
        pert_p[j] += 1j * step
        out_p = equations(
            _Vec(variables, pert_p),
            _Vec(variables, base_ss),
            _Vec(variables, base_ss),
            _Vec(shocks, base_e),
            par_vec,
        )
        A_plus[:, j] = np.asarray(out_p, dtype=complex).imag / step

        pert_0 = base_ss.copy()
        pert_0[j] += 1j * step
        out_0 = equations(
            _Vec(variables, base_ss),
            _Vec(variables, pert_0),
            _Vec(variables, base_ss),
            _Vec(shocks, base_e),
            par_vec,
        )
        A_0[:, j] = np.asarray(out_0, dtype=complex).imag / step

        pert_m = base_ss.copy()
        pert_m[j] += 1j * step
        out_m = equations(
            _Vec(variables, base_ss),
            _Vec(variables, base_ss),
            _Vec(variables, pert_m),
            _Vec(shocks, base_e),
            par_vec,
        )
        A_minus[:, j] = np.asarray(out_m, dtype=complex).imag / step

    for j in range(n_shocks):
        pert_e = base_e.copy()
        pert_e[j] += 1j * step
        out_e = equations(
            _Vec(variables, base_ss),
            _Vec(variables, base_ss),
            _Vec(variables, base_ss),
            _Vec(shocks, pert_e),
            par_vec,
        )
        B_u[:, j] = np.asarray(out_e, dtype=complex).imag / step

    # Constant / residual evaluated at reference steady state
    y_ss_real = _Vec(variables, eval_ss)
    e_zero_real = _Vec(shocks, np.zeros(n_shocks))
    res_ss = equations(y_ss_real, y_ss_real, y_ss_real, e_zero_real, par_vec)
    const = np.asarray(res_ss, dtype=float)

    return A_plus, A_0, A_minus, B_u, const, eval_ss, variables, shocks


def _safe_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B safely with fallback to least squares if singular."""
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        return sol


# ---------------------------------------------------------------------------
# OccBin Solver
# ---------------------------------------------------------------------------


def solve_occbin(
    reference_model: Any,
    constrained_model: Any,
    constraint: OccBinConstraint,
    shock_sequence: np.ndarray,
    max_iter: int = 50,
    horizon: int = 40,
) -> OccBinResult:
    """Solve dynamic models with occasionally binding constraints (Guerrieri & Iacoviello 2015).

    Finds the piecewise-linear perfect-foresight transition path between the
    constrained and unconstrained regimes using backward recursion.

    Parameters
    ----------
    reference_model : LinearModel
        The unconstrained baseline model (e.g., standard Taylor rule regime).
    constrained_model : LinearModel | Callable
        The model under the binding constraint (e.g., nominal interest rate held at floor).
        Can be a solved ``LinearModel`` or a callable ``eqs(lead, curr, lag, shocks, params)``.
    constraint : OccBinConstraint
        Constraint definition specifying variable, threshold, and direction.
    shock_sequence : np.ndarray
        Array of unforeseen structural shocks. Can be 1D (shape ``(n_shocks,)``)
        for a one-shot shock at t=1, or 2D (shape ``(n_periods, n_shocks)``).
    max_iter : int, default 50
        Maximum number of iterations for guessing the binding duration T*.
    horizon : int, default 40
        Simulation horizon (number of periods to simulate).

    Returns
    -------
    OccBinResult
        Container holding simulated path, regime timeline, binding periods,
        convergence diagnostics, and visualization/export methods.
    """
    # 1. Extract system matrices for both regimes
    A_p_0, A_0_0, A_m_0, B_u_0, c_0, ss_0, variables, shocks = _extract_model_matrices(reference_model)
    A_p_1, A_0_1, A_m_1, B_u_1, c_1, ss_1, _, _ = _extract_model_matrices(
        constrained_model, ref_model=reference_model
    )

    n_vars = len(variables)
    n_shocks = len(shocks)

    # Standardize shock sequence to shape (horizon, n_shocks)
    sh_arr = np.asarray(shock_sequence, dtype=float)
    if sh_arr.ndim == 1:
        if len(sh_arr) != n_shocks:
            raise ValueError(f"1D shock sequence has length {len(sh_arr)}, expected {n_shocks} shocks: {shocks}")
        shocks_mat = np.zeros((horizon, n_shocks))
        shocks_mat[0, :] = sh_arr
    elif sh_arr.ndim == 2:
        n_rows, n_cols = sh_arr.shape
        if n_cols != n_shocks:
            raise ValueError(f"shock sequence columns {n_cols} != model shocks {n_shocks}")
        shocks_mat = np.zeros((horizon, n_shocks))
        shocks_mat[: min(n_rows, horizon), :] = sh_arr[: min(n_rows, horizon), :]
    else:
        raise ValueError("shock_sequence must be 1D or 2D array")

    # 2. Extract reference linear decision rules: X_t = P_0 X_{t-1} + Q_0 u_t
    dr = reference_model.decision_rules()
    P_0 = np.zeros((n_vars, n_vars))
    for s in reference_model.states:
        idx_s = variables.index(s)
        P_0[:, idx_s] = dr.ghx[s].values
    Q_0 = dr.ghu.values

    # Find the row index of the constrained variable
    if constraint.variable not in variables:
        raise ValueError(f"constraint variable {constraint.variable!r} not found in model variables: {variables}")
    idx_var = variables.index(constraint.variable)

    # Identify the equation row that determines the constrained variable in the reference regime
    diff_rows = np.where(
        (np.linalg.norm(A_0_0 - A_0_1, axis=1) > 1e-8)
        | (np.abs(c_0 - c_1) > 1e-8)
        | (np.linalg.norm(B_u_0 - B_u_1, axis=1) > 1e-8)
    )[0]
    if len(diff_rows) > 0:
        eq_row = diff_rows[0]
    else:
        eq_row = idx_var

    # 3. Backward recursion engine for a conjectured duration T*
    def compute_decision_rules(T_star: int):
        if T_star == 0:
            return (
                [P_0] * horizon,
                [np.zeros(n_vars)] * horizon,
                [Q_0] + [np.zeros((n_vars, n_shocks))] * (horizon - 1),
            )

        P_seq: list[np.ndarray] = [np.zeros((n_vars, n_vars)) for _ in range(T_star + 1)]
        D_seq: list[np.ndarray] = [np.zeros(n_vars) for _ in range(T_star + 1)]

        # Terminal period of binding spell: t = T*
        M_T = A_0_1 + A_p_1 @ P_0
        P_seq[T_star] = _safe_solve(M_T, -A_m_1)
        D_seq[T_star] = _safe_solve(M_T, -c_1)

        # Iterate backward from T* - 1 down to 1
        for t in range(T_star - 1, 0, -1):
            M_t = A_0_1 + A_p_1 @ P_seq[t + 1]
            P_seq[t] = _safe_solve(M_t, -A_m_1)
            D_seq[t] = _safe_solve(M_t, -(A_p_1 @ D_seq[t + 1] + c_1))

        # Contemporaneous shock loading at t = 1
        M_1 = A_0_1 + A_p_1 @ (P_seq[2] if T_star > 1 else P_0)
        Q_1 = _safe_solve(M_1, -B_u_1)

        # Assemble full horizon sequences
        full_P: list[np.ndarray] = [P_0] * horizon
        full_D: list[np.ndarray] = [np.zeros(n_vars)] * horizon
        full_Q: list[np.ndarray] = [Q_0] * horizon
        for t in range(1, horizon + 1):
            if t <= T_star:
                full_P[t - 1] = P_seq[t]
                full_D[t - 1] = D_seq[t]
            else:
                full_P[t - 1] = P_0
                full_D[t - 1] = np.zeros(n_vars)

            if t == 1:
                full_Q[t - 1] = Q_1
            else:
                full_Q[t - 1] = Q_0

        return full_P, full_D, full_Q

    # 4. Forward simulation under conjectured rules
    def simulate_path(T_star: int) -> tuple[np.ndarray, np.ndarray]:
        full_P, full_D, full_Q = compute_decision_rules(T_star)
        X = np.zeros((horizon + 1, n_vars))
        for t in range(1, horizon + 1):
            u_t = shocks_mat[t - 1]
            X[t] = full_P[t - 1] @ X[t - 1] + full_D[t - 1] + full_Q[t - 1] @ u_t

        sim_X = X[1:]

        # Compute shadow / notional value of the constrained variable
        shadow_vals = np.zeros(horizon)
        a_var = A_0_0[eq_row, idx_var] if abs(A_0_0[eq_row, idx_var]) > 1e-12 else 1.0

        for t in range(horizon):
            x_prev = X[t]
            x_curr = sim_X[t]
            x_next = sim_X[t + 1] if t + 1 < horizon else P_0 @ x_curr
            u_t = shocks_mat[t]

            # Shadow value from reference equation row:
            other_curr = np.sum([A_0_0[eq_row, j] * x_curr[j] for j in range(n_vars) if j != idx_var])
            term_next = A_p_0[eq_row, :] @ x_next
            term_prev = A_m_0[eq_row, :] @ x_prev
            term_shock = B_u_0[eq_row, :] @ u_t
            shadow_vals[t] = - (other_curr + term_next + term_prev + term_shock + c_0[eq_row]) / a_var

        return sim_X, shadow_vals

    # 5. Backward recursion iteration loop over duration T*
    T_star = 0
    converged = False
    history: list[int] = []
    final_X = None
    final_shadow = None

    for iteration in range(1, max_iter + 1):
        history.append(T_star)
        sim_X, shadow_vals = simulate_path(T_star)

        # Evaluate binding condition for each period
        bind_vec = []
        for t in range(horizon):
            is_constrained_regime = t < T_star
            if is_constrained_regime:
                if constraint.relax_variable is not None and constraint.relax_variable in variables:
                    r_idx = variables.index(constraint.relax_variable)
                    val = sim_X[t, r_idx]
                else:
                    val = shadow_vals[t]
                binds = constraint.evaluate(val)
            else:
                val = sim_X[t, idx_var]
                binds = constraint.evaluate(val)

            bind_vec.append(bool(binds))

        # Number of consecutive periods constraint binds from t=0
        T_new = 0
        while T_new < horizon and bind_vec[T_new]:
            T_new += 1

        # Check convergence
        if T_new == T_star:
            converged = True
            final_X = sim_X
            final_shadow = shadow_vals
            break

        # Cycle detection
        if T_new in history:
            candidate_T = max(T_star, T_new)
            final_X, final_shadow = simulate_path(candidate_T)
            T_star = candidate_T
            converged = True
            break

        T_star = T_new

    if not converged:
        final_X, final_shadow = simulate_path(T_star)

    # 6. Construct OccBinResult
    sim_df = pd.DataFrame(final_X, columns=variables)
    shadow_df = sim_df.copy()
    shadow_df[f"{constraint.variable}_shadow"] = final_shadow

    regimes = [1 if t < T_star else 0 for t in range(horizon)]

    return OccBinResult(
        simulated_path=sim_df,
        regimes=regimes,
        binding_periods=T_star,
        converged=converged,
        iterations=iteration,
        reference_model=reference_model,
        constrained_model=constrained_model,
        constraint=constraint,
        shadow_path=shadow_df,
    )


__all__ = [
    "OccBinConstraint",
    "OccBinResult",
    "solve_occbin",
]
