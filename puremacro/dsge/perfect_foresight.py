r"""Deterministic Non-Linear Simulation / Perfect Foresight & MCP Solver for DSGE models.

Implements:
1. The stacked Newton-Raphson relaxation algorithm (Boucekkine 1995, Juillard 1996;
   also known as the Laffargue-Boucekkine-Juillard / LBJ algorithm) for solving
   non-linear dynamic rational-expectations models under perfect foresight.
2. Two-point boundary transitions across distinct steady states (histval -> endval)
   with permanent exogenous shocks and anticipated paths (varexo_det).
3. Rolling unanticipated surprise shock sequence simulation (simulate_surprise_shocks).
4. Mixed Complementarity Problem (MCP) solver using Semismooth Newton with the
   Fischer-Burmeister complementarity function:
       Phi(a, b) = a + b - sqrt(a^2 + b^2) = 0
   directly embedded into the sparse block-tridiagonal stacked Jacobian without
   destroying its O(T) sparse structure or violating the Pyodide 4-package contract.

Solves the stacked non-linear system:
    f(y_{t+1}, y_t, y_{t-1}, \epsilon_t) = 0,   t = 1, ..., T
subject to boundary conditions:
    y_0 = y_init
    y_{T+1} = y_end
and optional complementarity constraints y_{j, t} >= lb_j perp f_{i, t} >= 0.

Exploits the sparse block-tridiagonal structure of the stacked Jacobian using
scipy.sparse and SuperLU sparse direct linear solvers (scipy.sparse.linalg.spsolve)
for fast, memory-efficient O(T) performance without dense O(T^3) complexity.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Relative tolerance on the Newton STEP.
_STEP_TOL = 1e-10
# Armijo sufficient-decrease constant for backtracking line search.
_ARMIJO_C = 1e-4


@dataclass(frozen=True)
class PerfectForesightResult:
    """Result of deterministic non-linear perfect foresight simulation.

    Attributes
    ----------
    path : pd.DataFrame, shape (T, n_vars)
        Simulated trajectory of endogenous variables from t=1 to t=T.
    converged : bool
        Whether the stacked Newton-Raphson solver converged. Convergence
        requires BOTH a small residual and a negligible Newton step.
    iterations : int
        Number of Newton systems solved.
    residual_norm : float
        Maximum absolute equation residual across all periods and equations.
    terminal_error : float
        Maximum absolute distance between terminal state y_T and steady state y_ss.
    variable_names : tuple[str, ...]
        Names of endogenous variables.
    initial_residual_norm : float
        Residual infinity norm of the initial guess, before any Newton step.
    """

    path: pd.DataFrame
    converged: bool
    iterations: int
    residual_norm: float
    terminal_error: float
    variable_names: tuple[str, ...] = ()
    initial_residual_norm: float = float("nan")

    def to_frame(self) -> pd.DataFrame:
        """Return the simulated trajectory as a DataFrame."""
        return self.path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        """Access a variable's simulated path by column name."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.path.columns:
            return self.path[key]
        raise KeyError(f"Variable or attribute {key!r} not found in PerfectForesightResult.")

    def summary(self, as_dataframe: bool = False) -> str | pd.DataFrame:
        """Render summary of convergence and variable trajectory statistics.

        Parameters
        ----------
        as_dataframe : bool, default False
            If True, returns a pandas DataFrame with summary statistics.
            If False, returns a publication-formatted string report.
        """
        stats = self.path.describe().T[["mean", "std", "min", "max"]].copy()
        if not self.path.empty:
            stats["initial"] = self.path.iloc[0].values
            stats["terminal"] = self.path.iloc[-1].values
        if as_dataframe:
            return stats

        lines = [
            "PERFECT FORESIGHT SIMULATION RESULT",
            "=" * 72,
            f"Convergence status  : {'CONVERGED' if self.converged else 'FAILED'}",
            f"Iterations          : {self.iterations}",
            f"Residual norm       : {self.residual_norm:.4e}"
            + (
                f"  (initial guess: {self.initial_residual_norm:.4e})"
                if np.isfinite(self.initial_residual_norm)
                else ""
            ),
            f"Terminal error      : {self.terminal_error:.4e}",
            f"Simulation horizon  : {len(self.path)} periods",
            f"Endogenous variables: {', '.join(map(str, self.path.columns))}",
            "-" * 72,
            "TRAJECTORY SUMMARY (t=1..T):",
            stats.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a Markdown table."""
        from puremacro.reports import _df_to_markdown

        df = self.path.head(head) if head is not None else self.path
        return _df_to_markdown(df, index=index)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        df = self.path.head(head) if head is not None else self.path
        return _df_to_latex(df, index=index)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a Typst table."""
        from puremacro.reports import _df_to_typst

        df = self.path.head(head) if head is not None else self.path
        return _df_to_typst(df, index=index)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        style: str = "publication",
        *,
        ax=None,
        title: str | None = None,
        xlabel: str = "Period (t)",
        ylabel: str = "Level",
        **kwargs,
    ):
        """Plot simulated variable trajectories."""
        from puremacro.plot import _new_ax

        if variables is not None:
            cols = [v for v in variables if v in self.path.columns]
            if not cols:
                raise ValueError(
                    f"None of requested variables {variables} found in {list(self.path.columns)}"
                )
        else:
            cols = list(self.path.columns)

        fig, ax = _new_ax(ax)

        if style == "publication":
            from puremacro.plotting.bw_style import bw_colors, bw_linestyles

            colors = bw_colors(len(cols))
            linestyles = bw_linestyles(len(cols))
            for i, col in enumerate(cols):
                ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=1.3,
                    **kwargs,
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle=":", linewidth=0.5, color="0.7", alpha=0.7)
        else:
            for col in cols:
                ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    linewidth=1.3,
                    **kwargs,
                )
            ax.grid(True, alpha=0.3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Deterministic Simulation (Perfect Foresight)")
        ax.legend(loc="best", frameon=False)
        return fig


@dataclass(frozen=True)
class MCPResult:
    """Result of Mixed Complementarity Problem (MCP) non-linear simulation.

    Attributes
    ----------
    converged : bool
        Whether the Semismooth Newton solver converged.
    iterations : int
        Number of Newton iterations executed.
    residuals : np.ndarray
        Array of equation residuals at the solution.
    binding_periods : dict[str, list[int]]
        Dictionary mapping constrained variable names to lists of 1-based period
        indices where the bound is active.
    path : pd.DataFrame
        Simulated trajectory of endogenous variables from t=1 to t=T.
    residual_norm : float
        Maximum absolute Fischer-Burmeister residual across all periods and equations.
    terminal_error : float
        Maximum absolute distance between terminal state y_T and terminal steady state.
    complementarity_residual : float
        Maximum complementarity violation max_{t, k} |(y_{k, t} - bound) * f_k|.
    variable_names : tuple[str, ...]
        Names of endogenous variables.
    initial_residual_norm : float
        Residual infinity norm of the initial guess.
    """

    converged: bool = True
    iterations: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    binding_periods: dict[str, list[int]] = field(default_factory=dict)
    path: pd.DataFrame = field(default_factory=pd.DataFrame)
    residual_norm: float = 0.0
    terminal_error: float = 0.0
    complementarity_residual: float = 0.0
    variable_names: tuple[str, ...] = ()
    initial_residual_norm: float = float("nan")

    def __post_init__(self):
        if not self.variable_names and not self.path.empty:
            object.__setattr__(self, "variable_names", tuple(map(str, self.path.columns)))
        if self.residual_norm == 0.0 and self.residuals.size > 0:
            object.__setattr__(self, "residual_norm", float(np.max(np.abs(self.residuals))))

    def to_frame(self) -> pd.DataFrame:
        """Return the simulated trajectory as a DataFrame."""
        return self.path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        """Access a variable's simulated path by column name."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.path.columns:
            return self.path[key]
        raise KeyError(f"Variable or attribute {key!r} not found in MCPResult.")

    def summary(self, as_dataframe: bool = False) -> str | pd.DataFrame:
        """Render summary of MCP convergence and constraint regimes."""
        stats = self.path.describe().T[["mean", "std", "min", "max"]].copy()
        if not self.path.empty:
            stats["initial"] = self.path.iloc[0].values
            stats["terminal"] = self.path.iloc[-1].values
        if as_dataframe:
            return stats

        binding_lines = []
        if self.binding_periods:
            for v, p_list in self.binding_periods.items():
                if p_list:
                    sample_p = p_list[:10]
                    suffix = "..." if len(p_list) > 10 else ""
                    binding_lines.append(f"  {v}: {len(p_list)} periods (periods {sample_p}{suffix})")
                else:
                    binding_lines.append(f"  {v}: never binding (0 periods)")
        else:
            binding_lines.append("  (None)")

        lines = [
            "MIXED COMPLEMENTARITY PROBLEM (MCP) SIMULATION RESULT",
            "=" * 72,
            f"Convergence status      : {'CONVERGED' if self.converged else 'FAILED'}",
            f"Semismooth Newton iter  : {self.iterations}",
            f"Residual norm (FB MCP)  : {self.residual_norm:.4e}"
            + (
                f"  (initial guess: {self.initial_residual_norm:.4e})"
                if np.isfinite(self.initial_residual_norm)
                else ""
            ),
            f"Complementarity error   : {self.complementarity_residual:.4e}",
            f"Terminal error          : {self.terminal_error:.4e}",
            f"Simulation horizon      : {len(self.path)} periods",
            f"Endogenous variables    : {', '.join(map(str, self.path.columns))}",
            "Active constraint periods:",
            *binding_lines,
            "-" * 72,
            "TRAJECTORY SUMMARY (t=1..T):",
            stats.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a Markdown table."""
        from puremacro.reports import _df_to_markdown

        df = self.path.head(head) if head is not None else self.path
        return _df_to_markdown(df, index=index)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        df = self.path.head(head) if head is not None else self.path
        return _df_to_latex(df, index=index)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a Typst table."""
        from puremacro.reports import _df_to_typst

        df = self.path.head(head) if head is not None else self.path
        return _df_to_typst(df, index=index)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        style: str = "publication",
        *,
        ax=None,
        title: str | None = None,
        xlabel: str = "Period (t)",
        ylabel: str = "Level",
        **kwargs,
    ):
        """Plot simulated variable trajectories with shaded constraint binding spans."""
        from puremacro.plot import _new_ax

        if variables is not None:
            cols = [v for v in variables if v in self.path.columns]
            if not cols:
                raise ValueError(
                    f"None of requested variables {variables} found in {list(self.path.columns)}"
                )
        else:
            cols = list(self.path.columns)

        fig, ax = _new_ax(ax)

        # Shade binding periods
        if self.binding_periods:
            all_bound_t = set()
            for v in cols:
                if v in self.binding_periods:
                    all_bound_t.update(self.binding_periods[v])
            for t_idx in sorted(all_bound_t):
                ax.axvspan(t_idx - 0.45, t_idx + 0.45, color="0.88", alpha=0.4, zorder=-1)

        if style == "publication":
            from puremacro.plotting.bw_style import bw_colors, bw_linestyles

            colors = bw_colors(len(cols))
            linestyles = bw_linestyles(len(cols))
            for i, col in enumerate(cols):
                ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=1.3,
                    **kwargs,
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle=":", linewidth=0.5, color="0.7", alpha=0.7)
        else:
            for col in cols:
                ax.plot(
                    self.path.index,
                    self.path[col],
                    label=str(col),
                    linewidth=1.3,
                    **kwargs,
                )
            ax.grid(True, alpha=0.3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Mixed Complementarity Problem Simulation (MCP)")
        ax.legend(loc="best", frameon=False)
        return fig


def find_steady_state(
    model_or_equations: Any,
    *,
    variables: Sequence[str] | None = None,
    shocks: Sequence[str] | None = None,
    params: Mapping[str, float] | None = None,
    exo_values: Mapping[str, float] | Sequence[float] | None = None,
    guess: Mapping[str, float] | Sequence[float] | None = None,
    tol: float = 1e-10,
    solve_algo: str = "block",
) -> np.ndarray:
    """Statically compute steady state under permanent exogenous shocks or parameter values."""
    from puremacro.dsge.build import _Vec

    if hasattr(model_or_equations, "_dynare_equations") and model_or_equations._dynare_equations is not None:
        eqs = model_or_equations._dynare_equations
        v_list = list(variables or model_or_equations.variables)
        s_list = list(shocks or model_or_equations.shocks)
        p_dict = dict(model_or_equations._params or {})
        if params:
            p_dict.update(params)
    elif hasattr(model_or_equations, "compile_equations") and hasattr(model_or_equations, "variables"):
        eqs = model_or_equations.compile_equations()
        v_list = list(variables or model_or_equations.variables)
        s_list = list(shocks or model_or_equations.shocks)
        p_dict = dict(getattr(model_or_equations, "parameter_values", {}))
        if params:
            p_dict.update(params)
    elif callable(model_or_equations):
        eqs = model_or_equations
        v_list = list(variables) if variables is not None else []
        s_list = list(shocks) if shocks is not None else []
        p_dict = dict(params or {})
    else:
        raise TypeError(f"Unsupported model_or_equations type: {type(model_or_equations)}")

    # Exogenous values
    if isinstance(exo_values, Mapping):
        exo_arr = np.array([float(exo_values.get(s, 0.0)) for s in s_list], dtype=float)
    elif exo_values is not None:
        exo_arr = np.asarray(exo_values, dtype=float).ravel()
    else:
        exo_arr = np.zeros(len(s_list), dtype=float)

    # Initial guess
    if isinstance(guess, Mapping):
        guess_arr = np.array([float(guess.get(v, 0.0)) for v in v_list], dtype=float)
    elif guess is not None:
        guess_arr = np.asarray(guess, dtype=float).ravel()
    elif hasattr(model_or_equations, "steady_state") and model_or_equations.steady_state is not None:
        guess_arr = np.array([float(model_or_equations.steady_state[v]) for v in v_list], dtype=float)
    else:
        guess_arr = np.ones(len(v_list), dtype=float)

    p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), what="parameter")
    e_vec = _Vec(s_list, exo_arr, what="shock")

    def static_fn(y_val: np.ndarray) -> np.ndarray:
        try:
            y_v = _Vec(v_list, y_val, what="variable")
            out = eqs(y_v, y_v, y_v, e_vec, p_vec)
        except TypeError:
            eps_val = exo_arr if len(s_list) > 1 else (exo_arr[0] if len(exo_arr) else 0.0)
            out = eqs(y_val, y_val, y_val, eps_val)
        return np.asarray(out, dtype=float).ravel()

    # Try hybr, then lm, then df-sane
    sol = scipy.optimize.root(static_fn, guess_arr, method="hybr", tol=tol)
    if not sol.success or np.max(np.abs(static_fn(sol.x))) > max(tol, 1e-6):
        sol_lm = scipy.optimize.root(static_fn, guess_arr, method="lm", tol=tol)
        if sol_lm.success and np.max(np.abs(static_fn(sol_lm.x))) < np.max(np.abs(static_fn(sol.x))):
            sol = sol_lm
    if not sol.success or np.max(np.abs(static_fn(sol.x))) > max(tol, 1e-6):
        sol_df = scipy.optimize.root(static_fn, guess_arr, method="df-sane", tol=tol)
        if sol_df.success and np.max(np.abs(static_fn(sol_df.x))) < np.max(np.abs(static_fn(sol.x))):
            sol = sol_df

    return np.asarray(sol.x, dtype=float)


def simulate_surprise_shocks(
    model_or_equations: Any,
    surprise_shocks: np.ndarray | Sequence[Any],
    *,
    horizon: int = 100,
    y_init: np.ndarray | None = None,
    y_ss: np.ndarray | None = None,
    baseline_shocks: np.ndarray | Sequence[Any] | None = None,
    mcp: bool = False,
    mcp_bounds: Mapping[str | int, tuple[float | None, float | None]] | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    variable_names: Sequence[str] | None = None,
    **kwargs,
) -> PerfectForesightResult | MCPResult:
    """Solve rolling sequence of unanticipated surprise shocks (MIT shocks)."""
    s_arr = np.asarray(surprise_shocks, dtype=float)
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(-1, 1)
    T_surprise, n_shocks = s_arr.shape

    if hasattr(model_or_equations, "variables"):
        v_list = list(variable_names or model_or_equations.variables)
        ss_val = np.array([float(model_or_equations.steady_state[v]) for v in v_list]) if y_ss is None else np.asarray(y_ss, dtype=float).ravel()
    else:
        ss_val = np.asarray(y_ss, dtype=float).ravel() if y_ss is not None else np.zeros(1)
        v_list = list(variable_names or [f"y_{i}" for i in range(len(ss_val))])

    n_vars = len(ss_val) if y_init is None else len(np.asarray(y_init).ravel())
    current_state = np.asarray(y_init, dtype=float).ravel() if y_init is not None else ss_val.copy()

    if baseline_shocks is not None:
        base_exo = np.asarray(baseline_shocks, dtype=float)
        if base_exo.ndim == 1 and n_shocks == 1:
            base_exo = base_exo.reshape(-1, 1)
        if base_exo.shape[0] < horizon:
            last_row = base_exo[-1:, :]
            pad = np.repeat(last_row, horizon - base_exo.shape[0], axis=0)
            base_exo = np.vstack([base_exo, pad])
        else:
            base_exo = base_exo[:horizon].copy()
    else:
        base_exo = np.zeros((horizon, n_shocks), dtype=float)

    realized_path = np.zeros((T_surprise, n_vars), dtype=float)
    total_iter = 0
    max_res = 0.0
    all_binding: dict[str, list[int]] = {}

    for t in range(T_surprise):
        exo_path_t = base_exo.copy()
        exo_path_t[0, :] = s_arr[t, :]

        res_t = solve_perfect_foresight(
            model_or_equations,
            y_init=current_state,
            y_ss=ss_val,
            exogenous_path=exo_path_t,
            n_periods=horizon,

            mcp=mcp,
            mcp_bounds=mcp_bounds,
            tol=tol,
            max_iter=max_iter,
            variable_names=v_list,
            **kwargs,
        )

        current_state = res_t.path.iloc[0].to_numpy()
        realized_path[t, :] = current_state
        total_iter += res_t.iterations
        max_res = max(max_res, getattr(res_t, "residual_norm", 0.0))

        if mcp and hasattr(res_t, "binding_periods"):
            for v_name, b_periods in res_t.binding_periods.items():
                if 1 in b_periods:
                    all_binding.setdefault(v_name, []).append(t + 1)

    path_df = pd.DataFrame(
        realized_path,
        index=pd.RangeIndex(1, T_surprise + 1, name="t"),
        columns=v_list,
    )

    if mcp:
        return MCPResult(
            path=path_df,
            converged=True,
            iterations=total_iter,
            residuals=np.zeros((T_surprise, n_vars)),
            binding_periods=all_binding,
            residual_norm=max_res,
            terminal_error=float(np.max(np.abs(realized_path[-1] - ss_val))),
            variable_names=tuple(v_list),
        )

    return PerfectForesightResult(
        path=path_df,
        converged=True,
        iterations=total_iter,
        residual_norm=max_res,
        terminal_error=float(np.max(np.abs(realized_path[-1] - ss_val))),
        variable_names=tuple(v_list),
    )


def solve_perfect_foresight(
    model_or_equations: Any = None,
    y_init: np.ndarray | None = None,
    y_ss: np.ndarray | None = None,
    exogenous_path: np.ndarray | None = None,
    n_periods: int | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    dampening: float = 1.0,
    *,
    periods: int | None = None,
    shocks: Any | None = None,
    y_end: np.ndarray | None = None,
    mcp: bool = False,
    mcp_bounds: Mapping[str | int, tuple[float | None, float | None]] | None = None,
    variable_names: Sequence[str] | None = None,
    initial_path: np.ndarray | None = None,
    method: str = "auto",
    verbose: bool = False,
    equations_fn: Callable | None = None,
    surprise: bool = False,
    histval: Mapping[str, float] | None = None,
    endval: Mapping[str, float] | None = None,
) -> PerfectForesightResult | MCPResult:
    r"""Solve dynamic non-linear model under perfect foresight using stacked Newton-Raphson or Semismooth Newton MCP."""
    from puremacro.dsge.build import _Vec

    # 1. Resolve horizon
    if periods is not None:
        eff_periods = int(periods)
    elif n_periods is not None:
        eff_periods = int(n_periods)
    else:
        eff_periods = 100

    if eff_periods < 1:
        raise ValueError(f"n_periods must be at least 1, got {eff_periods}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"solve_perfect_foresight: max_iter must be an integer >= 1, got {max_iter!r}")
    max_iter = int(max_iter)
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"solve_perfect_foresight: tol must be a finite positive number, got {tol!r}")
    dampening = float(dampening)
    if not np.isfinite(dampening) or not (0.0 < dampening <= 1.0):
        raise ValueError(f"solve_perfect_foresight: dampening must lie in (0, 1], got {dampening!r}")

    if method not in ("auto", "central", "complex"):
        raise ValueError(
            f"solve_perfect_foresight: unknown method {method!r}; "
            "expected one of 'auto', 'central', 'complex'"
        )

    # 2. Resolve Model vs Equations
    model = None
    if equations_fn is not None:
        eq_fn = equations_fn
    elif hasattr(model_or_equations, "_dynare_equations") and model_or_equations._dynare_equations is not None:
        model = model_or_equations
    elif hasattr(model_or_equations, "compile_equations") and hasattr(model_or_equations, "variables"):
        model = model_or_equations
    elif callable(model_or_equations):
        eq_fn = model_or_equations
    else:
        raise ValueError("Must provide model or equations_fn to solve_perfect_foresight")

    if model is not None:
        v_names_list = list(variable_names or model.variables)
        s_names_list = list(model.shocks)
        p_dict = model._params or getattr(model, "parameter_values", {}) or {}
        p_vec = _Vec(list(p_dict.keys()), list(p_dict.values()), what="parameter")

        dyn_eqs = getattr(model, "_dynare_equations", None)
        if dyn_eqs is None and hasattr(model, "compile_equations"):
            dyn_eqs = model.compile_equations()

        def eq_wrapper(yp, yc, yl, eps):
            yp_v = _Vec(v_names_list, yp, what="variable")
            yc_v = _Vec(v_names_list, yc, what="variable")
            yl_v = _Vec(v_names_list, yl, what="variable")
            eps_v = _Vec(s_names_list, np.atleast_1d(eps), what="shock")
            return dyn_eqs(yp_v, yc_v, yl_v, eps_v, p_vec)

        eq_fn = eq_wrapper
        ss_dict = model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else dict(model.steady_state or {})
        base_ss = np.array([float(ss_dict.get(v, 0.0)) for v in v_names_list], dtype=float)

        # Handle histval override
        eff_histval = histval or getattr(model, "_histval", None)
        if y_init is None:
            y_init_arr = base_ss.copy()
            if eff_histval:
                for v, val in eff_histval.items():
                    if v in v_names_list:
                        y_init_arr[v_names_list.index(v)] = float(val)
        else:
            y_init_arr = np.asarray(y_init, dtype=float).ravel()

        # Handle endval override / terminal steady state solve
        eff_endval = endval or getattr(model, "_endval", None)
        if y_end is not None:
            y_ss_arr = np.asarray(y_end, dtype=float).ravel()
        elif y_ss is not None:
            y_ss_arr = np.asarray(y_ss, dtype=float).ravel()
        elif eff_endval:
            y_ss_arr = find_steady_state(model, exo_values=eff_endval, params=eff_endval, tol=tol)
        else:
            y_ss_arr = base_ss.copy()

        # Shocks
        raw_shocks = shocks if shocks is not None else exogenous_path
        if raw_shocks is None:
            exo_sim = np.zeros((eff_periods, len(s_names_list)), dtype=float)
        else:
            exo_arr = np.asarray(raw_shocks, dtype=float)
            if exo_arr.ndim == 1 and len(s_names_list) > 1:
                exo_arr = exo_arr.reshape(-1, 1)
            if exo_arr.shape[0] == eff_periods + 2:
                exo_sim = exo_arr[1 : eff_periods + 1]
            elif exo_arr.shape[0] >= eff_periods:
                exo_sim = exo_arr[:eff_periods]
            else:
                raise ValueError(f"shocks length ({exo_arr.shape[0]}) must be at least n_periods ({eff_periods})")
        v_names = tuple(v_names_list)
    else:
        # Pure callable path
        y_init_arr = np.asarray(y_init, dtype=float).ravel()
        y_end_target = y_end if y_end is not None else y_ss
        y_ss_arr = np.asarray(y_end_target, dtype=float).ravel()
        raw_shocks = shocks if shocks is not None else exogenous_path
        if raw_shocks is None:
            exo_sim = np.zeros(eff_periods, dtype=float)
        else:
            exo_arr = np.asarray(raw_shocks, dtype=float)
            if exo_arr.shape[0] == eff_periods + 2:
                exo_sim = exo_arr[1 : eff_periods + 1]
            elif exo_arr.shape[0] >= eff_periods:
                exo_sim = exo_arr[:eff_periods]
            else:
                raise ValueError(f"exogenous_path length ({exo_arr.shape[0]}) must be at least n_periods ({eff_periods})")


        if variable_names is not None:
            v_names = tuple(str(v) for v in variable_names)
        else:
            v_names = tuple(f"y_{i}" for i in range(len(y_init_arr)))

    n_vars = len(y_init_arr)
    if len(y_ss_arr) != n_vars:
        raise ValueError(f"Dimension mismatch: len(y_ss)={len(y_ss_arr)} != len(y_init)={n_vars}")

    if surprise:
        return simulate_surprise_shocks(
            model_or_equations=model if model is not None else eq_fn,
            surprise_shocks=exo_sim,
            horizon=eff_periods,
            y_init=y_init_arr,
            y_ss=y_ss_arr,
            mcp=mcp,
            mcp_bounds=mcp_bounds,
            tol=tol,
            max_iter=max_iter,
            variable_names=v_names,
        )

    # Initial guess for Y (T x n_vars) via linear interpolation
    if initial_path is not None:
        Y = np.asarray(initial_path, dtype=float).copy()
        if Y.shape != (eff_periods, n_vars):
            raise ValueError(f"initial_path shape {Y.shape} does not match ({eff_periods}, {n_vars})")
    else:
        Y = np.zeros((eff_periods, n_vars), dtype=float)
        for t in range(eff_periods):
            w = (t + 1.0) / (eff_periods + 1.0)
            Y[t] = (1.0 - w) * y_init_arr + w * y_ss_arr

    # Auto-detect differentiation method
    step_fd = 1e-7
    use_complex = False
    if method == "complex":
        use_complex = True
    elif method == "auto":
        try:
            pert = y_init_arr.astype(complex)
            pert[0] += 1j * 1e-20
            out = np.asarray(eq_fn(y_ss_arr.astype(complex), pert, y_init_arr.astype(complex), exo_sim[0]), dtype=complex)
            if not np.all(out.imag == 0) and not np.isnan(out.imag).any():
                use_complex = True
        except Exception:
            use_complex = False

    # Setup MCP constraints if mcp is True or mcp_bounds provided
    is_mcp = bool(mcp or (mcp_bounds is not None and len(mcp_bounds) > 0))
    mcp_specs = []  # list of (var_idx, eq_idx, s, kind, lb, ub)
    if is_mcp and mcp_bounds:
        # Determine baseline static equation loadings to map var_idx -> eq_idx and determine sign
        base_f = np.asarray(eq_fn(y_ss_arr, y_ss_arr, y_ss_arr, exo_sim[0]), dtype=float).ravel()
        N = n_vars
        B_test = np.zeros((N, N))
        for j in range(N):
            yp_c = y_ss_arr.copy(); yp_c[j] += 1e-6
            ym_c = y_ss_arr.copy(); ym_c[j] -= 1e-6
            fp = np.asarray(eq_fn(y_ss_arr, yp_c, y_ss_arr, exo_sim[0]), dtype=float).ravel()
            fm = np.asarray(eq_fn(y_ss_arr, ym_c, y_ss_arr, exo_sim[0]), dtype=float).ravel()
            B_test[:, j] = (fp - fm) / (2.0 * 1e-6)

        for k, b_tuple in mcp_bounds.items():
            if isinstance(k, int):
                v_idx = k
            else:
                k_str = str(k)
                if k_str in v_names:
                    v_idx = v_names.index(k_str)
                else:
                    raise ValueError(f"MCP variable {k!r} not in model variables {v_names}")

            # Map to equation index
            if abs(B_test[v_idx, v_idx]) > 1e-5:
                e_idx = v_idx
            else:
                e_idx = int(np.argmax(np.abs(B_test[:, v_idx])))

            s_sign = 1.0 if B_test[e_idx, v_idx] >= 0 else -1.0
            lb_val = None if b_tuple[0] is None else float(b_tuple[0])
            ub_val = None if b_tuple[1] is None else float(b_tuple[1])

            if lb_val is not None and ub_val is not None and abs(lb_val - ub_val) < 1e-12:
                kind = "PINNED"
            elif lb_val is not None and ub_val is None:
                kind = "LOWER"
            elif lb_val is None and ub_val is not None:
                kind = "UPPER"
            elif lb_val is not None and ub_val is not None:
                kind = "TWO_SIDED"
            else:
                kind = "NONE"

            mcp_specs.append((v_idx, e_idx, s_sign, kind, lb_val, ub_val, str(k)))

    def _eval_stacked(Y_curr: np.ndarray) -> np.ndarray:
        R = np.zeros(n_vars * eff_periods, dtype=float)
        for t in range(eff_periods):
            yl = y_init_arr if t == 0 else Y_curr[t - 1]
            yc = Y_curr[t]
            yp = y_ss_arr if t == eff_periods - 1 else Y_curr[t + 1]
            eps = exo_sim[t]
            f_val = np.asarray(eq_fn(yp, yc, yl, eps), dtype=float).ravel()
            if len(f_val) != n_vars:
                raise ValueError(f"equations_fn returned {len(f_val)} equations at t={t+1}, expected {n_vars}")

            if is_mcp and mcp_specs:
                f_mcp = f_val.copy()
                for (v_idx, e_idx, s_sign, kind, lb_val, ub_val, _) in mcp_specs:
                    if kind == "PINNED":
                        f_mcp[e_idx] = yc[v_idx] - lb_val
                    elif kind == "LOWER":
                        a = yc[v_idx] - lb_val
                        b = s_sign * f_val[e_idx]
                        f_mcp[e_idx] = a + b - math.hypot(a, b)
                    elif kind == "UPPER":
                        a = ub_val - yc[v_idx]
                        b = -s_sign * f_val[e_idx]
                        f_mcp[e_idx] = a + b - math.hypot(a, b)
                    elif kind == "TWO_SIDED":
                        a_u = ub_val - yc[v_idx]
                        b_u = -s_sign * f_val[e_idx]
                        phi_u = a_u + b_u - math.hypot(a_u, b_u)
                        a_l = yc[v_idx] - lb_val
                        b_l = -phi_u
                        f_mcp[e_idx] = a_l + b_l - math.hypot(a_l, b_l)
                R[t * n_vars : (t + 1) * n_vars] = f_mcp
            else:
                R[t * n_vars : (t + 1) * n_vars] = f_val
        return R

    # Pre-allocate sparse block-tridiagonal Jacobian structure
    total_entries = (3 * eff_periods - 2) * n_vars * n_vars
    rows = np.zeros(total_entries, dtype=np.int32)
    cols = np.zeros(total_entries, dtype=np.int32)
    data = np.zeros(total_entries, dtype=float)

    idx = 0
    for t in range(eff_periods):
        for j in range(n_vars):
            for i in range(n_vars):
                rows[idx] = t * n_vars + i
                cols[idx] = t * n_vars + j
                idx += 1
        if t < eff_periods - 1:
            for j in range(n_vars):
                for i in range(n_vars):
                    rows[idx] = t * n_vars + i
                    cols[idx] = (t + 1) * n_vars + j
                    idx += 1
        if t > 0:
            for j in range(n_vars):
                for i in range(n_vars):
                    rows[idx] = t * n_vars + i
                    cols[idx] = (t - 1) * n_vars + j
                    idx += 1

    R = _eval_stacked(Y)
    res_norm = float(np.max(np.abs(R)))
    initial_res_norm = res_norm

    converged = False
    iterations = 0
    res_ok = False
    step_ok = False
    step_norm = float("inf")
    jac_scale = 1.0
    stalled = False

    for it in range(max_iter):
        iterations = it + 1

        # Populate Jacobian entries
        data_idx = 0
        for t in range(eff_periods):
            y_l = y_init_arr if t == 0 else Y[t - 1]
            y_c = Y[t]
            y_p = y_ss_arr if t == eff_periods - 1 else Y[t + 1]
            eps = exo_sim[t]

            # B_t: w.r.t y_curr
            B_t = np.zeros((n_vars, n_vars), dtype=float)
            for j in range(n_vars):
                if use_complex:
                    pert = y_c.astype(complex)
                    pert[j] += 1j * 1e-20
                    df = np.asarray(eq_fn(y_p.astype(complex), pert, y_l.astype(complex), eps), dtype=complex).imag / 1e-20
                else:
                    h = step_fd * max(1.0, abs(y_c[j]))
                    yp_c = y_c.copy(); yp_c[j] += h
                    ym_c = y_c.copy(); ym_c[j] -= h
                    fp = np.asarray(eq_fn(y_p, yp_c, y_l, eps), dtype=float).ravel()
                    fm = np.asarray(eq_fn(y_p, ym_c, y_l, eps), dtype=float).ravel()
                    df = (fp - fm) / (2.0 * h)
                B_t[:, j] = df

            # A_t: w.r.t y_plus
            A_t = np.zeros((n_vars, n_vars), dtype=float)
            if t < eff_periods - 1:
                for j in range(n_vars):
                    if use_complex:
                        pert = y_p.astype(complex)
                        pert[j] += 1j * 1e-20
                        df = np.asarray(eq_fn(pert, y_c.astype(complex), y_l.astype(complex), eps), dtype=complex).imag / 1e-20
                    else:
                        h = step_fd * max(1.0, abs(y_p[j]))
                        yp_p = y_p.copy(); yp_p[j] += h
                        ym_p = y_p.copy(); ym_p[j] -= h
                        fp = np.asarray(eq_fn(yp_p, y_c, y_l, eps), dtype=float).ravel()
                        fm = np.asarray(eq_fn(ym_p, y_c, y_l, eps), dtype=float).ravel()
                        df = (fp - fm) / (2.0 * h)
                    A_t[:, j] = df

            # C_t: w.r.t y_lag
            C_t = np.zeros((n_vars, n_vars), dtype=float)
            if t > 0:
                for j in range(n_vars):
                    if use_complex:
                        pert = y_l.astype(complex)
                        pert[j] += 1j * 1e-20
                        df = np.asarray(eq_fn(y_p.astype(complex), y_c.astype(complex), pert, eps), dtype=complex).imag / 1e-20
                    else:
                        h = step_fd * max(1.0, abs(y_l[j]))
                        yp_l = y_l.copy(); yp_l[j] += h
                        ym_l = y_l.copy(); ym_l[j] -= h
                        fp = np.asarray(eq_fn(y_p, y_c, yp_l, eps), dtype=float).ravel()
                        fm = np.asarray(eq_fn(y_p, y_c, ym_l, eps), dtype=float).ravel()
                        df = (fp - fm) / (2.0 * h)
                    C_t[:, j] = df

            # Apply MCP chain rule updates to constrained rows
            if is_mcp and mcp_specs:
                f_val = np.asarray(eq_fn(y_p, y_c, y_l, eps), dtype=float).ravel()
                for (v_idx, e_idx, s_sign, kind, lb_val, ub_val, _) in mcp_specs:
                    if kind == "PINNED":
                        B_t[e_idx, :] = 0.0
                        B_t[e_idx, v_idx] = 1.0
                        if t < eff_periods - 1:
                            A_t[e_idx, :] = 0.0
                        if t > 0:
                            C_t[e_idx, :] = 0.0
                    elif kind == "LOWER":
                        a = y_c[v_idx] - lb_val
                        b = s_sign * f_val[e_idx]
                        hyp = math.hypot(a, b)
                        da = 1.0 - a / hyp if hyp >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        db = 1.0 - b / hyp if hyp >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        B_t[e_idx, :] = db * s_sign * B_t[e_idx, :]
                        B_t[e_idx, v_idx] += da
                        if t < eff_periods - 1:
                            A_t[e_idx, :] = db * s_sign * A_t[e_idx, :]
                        if t > 0:
                            C_t[e_idx, :] = db * s_sign * C_t[e_idx, :]
                    elif kind == "UPPER":
                        a = ub_val - y_c[v_idx]
                        b = -s_sign * f_val[e_idx]
                        hyp = math.hypot(a, b)
                        da = 1.0 - a / hyp if hyp >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        db = 1.0 - b / hyp if hyp >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        B_t[e_idx, :] = -db * s_sign * B_t[e_idx, :]
                        B_t[e_idx, v_idx] -= da
                        if t < eff_periods - 1:
                            A_t[e_idx, :] = -db * s_sign * A_t[e_idx, :]
                        if t > 0:
                            C_t[e_idx, :] = -db * s_sign * C_t[e_idx, :]
                    elif kind == "TWO_SIDED":
                        a_u = ub_val - y_c[v_idx]
                        b_u = -s_sign * f_val[e_idx]
                        hyp_u = math.hypot(a_u, b_u)
                        dau = 1.0 - a_u / hyp_u if hyp_u >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        dbu = 1.0 - b_u / hyp_u if hyp_u >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        phi_u = a_u + b_u - hyp_u
                        a_l = y_c[v_idx] - lb_val
                        b_l = -phi_u
                        hyp_l = math.hypot(a_l, b_l)
                        dal = 1.0 - a_l / hyp_l if hyp_l >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        dbl = 1.0 - b_l / hyp_l if hyp_l >= 1e-15 else 1.0 - 1.0 / math.sqrt(2.0)
                        c_f = dbl * dbu * s_sign
                        c_v = dal + dbl * dau
                        B_t[e_idx, :] = c_f * B_t[e_idx, :]
                        B_t[e_idx, v_idx] += c_v
                        if t < eff_periods - 1:
                            A_t[e_idx, :] = c_f * A_t[e_idx, :]
                        if t > 0:
                            C_t[e_idx, :] = c_f * C_t[e_idx, :]

            # Copy B_t into data
            for j in range(n_vars):
                for i in range(n_vars):
                    data[data_idx] = B_t[i, j]
                    data_idx += 1

            # Copy A_t into data
            if t < eff_periods - 1:
                for j in range(n_vars):
                    for i in range(n_vars):
                        data[data_idx] = A_t[i, j]
                        data_idx += 1

            # Copy C_t into data
            if t > 0:
                for j in range(n_vars):
                    for i in range(n_vars):
                        data[data_idx] = C_t[i, j]
                        data_idx += 1

        J = sp.csc_matrix((data, (rows, cols)), shape=(n_vars * eff_periods, n_vars * eff_periods))
        jac_scale = max(1.0, float(np.max(np.abs(data))))

        try:
            dY_flat = spla.spsolve(J, -R)
        except Exception as exc:
            warnings.warn(f"Singular Jacobian encountered at iteration {it+1}: {exc}")
            break

        dY = dY_flat.reshape(eff_periods, n_vars)

        y_scale = max(1.0, float(np.max(np.abs(Y))))
        step_norm = float(np.max(np.abs(dampening * dY))) if np.all(np.isfinite(dY)) else float("inf")
        res_ok = res_norm < tol * jac_scale
        step_ok = step_norm <= _STEP_TOL * y_scale
        if res_ok and step_ok:
            converged = True
            break

        # Armijo line search on merit function Psi = 0.5 * ||R||_2^2
        psi_curr = 0.5 * float(np.sum(R**2))
        step_scale = float(dampening)
        accepted = False
        saw_finite_trial = False

        for _ in range(15):
            Y_trial = Y + step_scale * dY
            try:
                R_trial = _eval_stacked(Y_trial)
                trial_res_norm = float(np.max(np.abs(R_trial)))
                trial_psi = 0.5 * float(np.sum(R_trial**2))
                if np.isfinite(trial_res_norm) and trial_res_norm < 1e12:
                    saw_finite_trial = True
                    # Check both merit decrease and infinity-norm decrease
                    if (trial_psi <= (1.0 - 2.0 * _ARMIJO_C * step_scale) * psi_curr) or (
                        trial_res_norm <= (1.0 - _ARMIJO_C * step_scale) * res_norm
                    ):
                        Y = Y_trial
                        R = R_trial
                        res_norm = trial_res_norm
                        accepted = True
                        break
            except Exception:
                pass
            step_scale *= 0.5

        if not accepted:
            if saw_finite_trial:
                stalled = True
                warnings.warn(
                    f"solve_perfect_foresight: the Newton line search found no step that "
                    f"reduces the residual at iteration {it+1}; stalled at {res_norm:.4e}.",
                    stacklevel=2,
                )
                break
            warnings.warn(f"Newton-Raphson line search failed at iteration {it+1}")
            break

        step_norm = float(np.max(np.abs(step_scale * dY)))
        if verbose:
            print(f"Iter {iterations}: max residual = {res_norm:.4e}, step = {step_norm:.4e}")

    if not converged and not stalled:
        if res_ok and step_norm <= _STEP_TOL * y_scale:
            converged = True
        elif res_norm < tol:
            converged = True
        else:
            warnings.warn(
                f"solve_perfect_foresight did not converge in {max_iter} iterations "
                f"(residual norm {res_norm:.4e}, step norm {step_norm:.4e})"
            )

    # Clean up machine-precision numerical noise on active bounds for MCP
    binding_dict: dict[str, list[int]] = {}
    comp_residual = 0.0
    if is_mcp and mcp_specs:
        for (v_idx, e_idx, s_sign, kind, lb_val, ub_val, k_name) in mcp_specs:
            bound_periods = []
            for t in range(eff_periods):
                val = Y[t, v_idx]
                if kind == "PINNED":
                    Y[t, v_idx] = lb_val
                    bound_periods.append(t + 1)
                elif kind == "LOWER":
                    if val <= lb_val + 1e-6:
                        bound_periods.append(t + 1)
                    if val < lb_val:
                        Y[t, v_idx] = lb_val
                elif kind == "UPPER":
                    if val >= ub_val - 1e-6:
                        bound_periods.append(t + 1)
                    if val > ub_val:
                        Y[t, v_idx] = ub_val
                elif kind == "TWO_SIDED":
                    if val <= lb_val + 1e-6 or val >= ub_val - 1e-6:
                        bound_periods.append(t + 1)
                    if val < lb_val:
                        Y[t, v_idx] = lb_val
                    elif val > ub_val:
                        Y[t, v_idx] = ub_val

            binding_dict[k_name] = bound_periods
            # Complementarity violation: max_t |(y_t - lb) * f_t|
            for t in range(eff_periods):
                yl = y_init_arr if t == 0 else Y[t - 1]
                yc = Y[t]
                yp = y_ss_arr if t == eff_periods - 1 else Y[t + 1]
                eps = exo_sim[t]
                f_v = np.asarray(eq_fn(yp, yc, yl, eps), dtype=float).ravel()
                if kind == "LOWER":
                    comp_residual = max(comp_residual, abs((yc[v_idx] - lb_val) * min(0.0, s_sign * f_v[e_idx])))
                elif kind == "UPPER":
                    comp_residual = max(comp_residual, abs((ub_val - yc[v_idx]) * min(0.0, -s_sign * f_v[e_idx])))

    term_err = float(np.max(np.abs(Y[-1] - y_ss_arr)))
    path_df = pd.DataFrame(
        Y,
        index=pd.RangeIndex(1, eff_periods + 1, name="t"),
        columns=list(v_names),
    )

    residuals_out = R.reshape(eff_periods, n_vars)

    if is_mcp:
        return MCPResult(
            path=path_df,
            converged=converged,
            iterations=iterations,
            residuals=residuals_out,
            binding_periods=binding_dict,
            residual_norm=res_norm,
            terminal_error=term_err,
            complementarity_residual=comp_residual,
            variable_names=v_names,
            initial_residual_norm=initial_res_norm,
        )

    return PerfectForesightResult(
        path=path_df,
        converged=converged,
        iterations=iterations,
        residual_norm=res_norm,
        terminal_error=term_err,
        variable_names=v_names,
        initial_residual_norm=initial_res_norm,
    )


__all__ = [
    "PerfectForesightResult",
    "MCPResult",
    "find_steady_state",
    "simulate_surprise_shocks",
    "solve_perfect_foresight",
]
