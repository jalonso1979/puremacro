r"""Deterministic Non-Linear Simulation / Perfect Foresight Solver for DSGE models.

Implements the stacked Newton-Raphson relaxation algorithm (Boucekkine 1995,
Juillard 1996; also known as the Laffargue-Boucekkine-Juillard / LBJ algorithm)
for solving non-linear dynamic rational-expectations models under perfect foresight.

Solves the stacked non-linear system:
    f(y_{t+1}, y_t, y_{t-1}, \epsilon_t) = 0,   t = 1, ..., T
subject to boundary conditions:
    y_0 = y_init
    y_{T+1} = y_ss

Exploits the sparse block-tridiagonal structure of the stacked Jacobian using
scipy.sparse and SuperLU sparse direct linear solvers (scipy.sparse.linalg.spsolve)
for fast, memory-efficient O(T) performance without dense O(T^3) complexity.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Relative tolerance on the Newton STEP.  Unlike the residual, the Newton step
# dY = -J^{-1} f is invariant to rescaling all equations by a common factor, so
# it is the scale-free half of the convergence test.
_STEP_TOL = 1e-10
# Armijo sufficient-decrease constant for the backtracking line search.
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
        requires BOTH a small residual and a negligible Newton step: an
        absolute residual test alone is not scale-aware, and would declare
        success on the untouched initial guess of a model whose equations are
        written in small units.
    iterations : int
        Number of Newton systems solved (the last one is the step whose
        negligible size certified convergence).
    residual_norm : float
        Maximum absolute equation residual across all periods and equations
        (infinity norm: max_{t, i} |f_{i, t}|).
    terminal_error : float
        Maximum absolute distance between terminal state y_T and steady state y_ss
        (infinity norm: max_i |y_{T, i} - y_{ss, i}|).
    variable_names : tuple[str, ...]
        Names of endogenous variables.
    initial_residual_norm : float
        Residual infinity norm of the *initial guess*, before any Newton step.
        Compare it against ``residual_norm``: if the two are equal the solver
        made no progress, which on a badly scaled model is the signature of an
        absolute residual tolerance that is uninformative for this model.
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
        """Render simulation path as a Markdown table.

        Parameters
        ----------
        head : int, optional
            Number of initial rows to include. If None, includes all periods.
        index : bool, default True
            Whether to include the period index column.
        """
        from puremacro.reports import _df_to_markdown

        df = self.path.head(head) if head is not None else self.path
        return _df_to_markdown(df, index=index)

    def to_latex(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a LaTeX tabular environment.

        Parameters
        ----------
        head : int, optional
            Number of initial rows to include. If None, includes all periods.
        index : bool, default True
            Whether to include the period index column.
        """
        from puremacro.reports import _df_to_latex

        df = self.path.head(head) if head is not None else self.path
        return _df_to_latex(df, index=index)

    def to_typst(self, *, head: int | None = None, index: bool = True, **kwargs) -> str:
        """Render simulation path as a Typst table.

        Parameters
        ----------
        head : int, optional
            Number of initial rows to include. If None, includes all periods.
        index : bool, default True
            Whether to include the period index column.
        """
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
        """Plot simulated variable trajectories.

        Parameters
        ----------
        variables : Sequence[str], optional
            Subset of variables to plot. Defaults to all variables.
        style : {'publication', 'default'}, default 'publication'
            Plotting style. 'publication' produces high-contrast grayscale
            figures suitable for academic journals and monochrome print.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes to plot on.
        title : str, optional
            Custom figure title.
        xlabel : str, default 'Period (t)'
            X-axis label.
        ylabel : str, default 'Level'
            Y-axis label.
        **kwargs
            Passed to ax.plot().

        Returns
        -------
        matplotlib.figure.Figure
        """
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


def solve_perfect_foresight(
    equations_fn: Callable[..., Sequence[float] | np.ndarray],
    y_init: np.ndarray,
    y_ss: np.ndarray,
    exogenous_path: np.ndarray,
    n_periods: int = 100,
    tol: float = 1e-8,
    max_iter: int = 50,
    dampening: float = 1.0,
    *,
    variable_names: Sequence[str] | None = None,
    initial_path: np.ndarray | None = None,
    method: str = "auto",
    verbose: bool = False,
) -> PerfectForesightResult:
    r"""Solve dynamic non-linear model under perfect foresight using stacked Newton-Raphson.

    Solves the non-linear system:
        f(y_{t+1}, y_t, y_{t-1}, \epsilon_t) = 0,   t = 1, ..., T
    with boundary conditions y_0 = y_init and y_{T+1} = y_ss.

    Uses stacked Newton-Raphson (Boucekkine 1995, Juillard 1996) with a sparse
    block-tridiagonal Jacobian matrix inverted via scipy.sparse.linalg.spsolve
    for O(T) computational and memory scaling.

    Parameters
    ----------
    equations_fn : Callable(y_plus, y_curr, y_lag, eps) -> array_like of shape (n_vars,)
        Non-linear equilibrium conditions returning the residual vector:
        - y_plus: 1D array of endogenous variables at t+1 (lead)
        - y_curr: 1D array of endogenous variables at t (current)
        - y_lag: 1D array of endogenous variables at t-1 (lag)
        - eps: exogenous variable(s) at t
    y_init : np.ndarray, shape (n_vars,)
        Initial condition vector at t=0 (boundary condition y_0).
    y_ss : np.ndarray, shape (n_vars,)
        Terminal steady-state vector at t=T+1 (boundary condition y_{T+1}).
    exogenous_path : np.ndarray, shape (T,) or (T, n_shocks) or (T+2, ...)
        Path of exogenous variables/shocks. Supports pre-announced future shocks
        occurring at any period t in 1..T.
    n_periods : int, default 100
        Simulation horizon T.
    tol : float, default 1e-8
        Convergence tolerance on the residual infinity norm, scaled by the
        largest absolute Jacobian entry: ``max_{t,i} |f_{i,t}| < tol *
        max(1, max|J|)``. The Jacobian scaling makes the test invariant to
        writing the same model's equations in different units. Must be > 0.
    max_iter : int, default 50
        Maximum number of Newton-Raphson iterations. Must be >= 1.
    dampening : float, default 1.0
        Newton step dampening parameter in (0, 1]; the initial step length
        offered to the backtracking line search.
    variable_names : Sequence[str], optional
        Names of the endogenous variables. Defaults to ("y_0", "y_1", ...).
    initial_path : np.ndarray, optional, shape (T, n_vars)
        Custom initial guess for the path. Defaults to linear interpolation
        between y_init and y_ss.
    method : {'auto', 'central', 'complex'}, default 'auto'
        Jacobian differentiation method.
    verbose : bool, default False
        Whether to print iteration-by-iteration progress.

    Returns
    -------
    PerfectForesightResult
        Frozen dataclass with fields `path`, `converged`, `iterations`,
        `residual_norm`, `terminal_error`, `initial_residual_norm`.

    Notes
    -----
    Convergence requires BOTH criteria:

    * a small residual, ``max|f| < tol * max(1, max|J|)``, and
    * a negligible Newton step, ``max|s*dY| <= 1e-10 * max(1, max|Y|)``.

    The residual alone is not a scale-aware test: multiplying every equation of
    a model by 1e-9 shrinks every residual by 1e-9, and a solver that stopped on
    the residual alone would return the untouched initial guess as if it were
    the solution. The Newton step ``dY = -J^{-1} f`` is invariant to that
    rescaling, so at least one Newton system is always solved and its step must
    be negligible before ``converged=True`` is returned. The line search
    likewise requires an Armijo sufficient decrease, not merely a finite
    residual, so a Newton cycle is escaped rather than run to ``max_iter``.
    """
    y_init_arr = np.asarray(y_init, dtype=float).ravel()
    y_ss_arr = np.asarray(y_ss, dtype=float).ravel()
    n_vars = len(y_init_arr)

    if len(y_ss_arr) != n_vars:
        raise ValueError(
            f"Dimension mismatch: len(y_ss)={len(y_ss_arr)} != len(y_init)={n_vars}"
        )
    if n_periods < 1:
        raise ValueError(f"n_periods must be at least 1, got {n_periods}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(
            f"solve_perfect_foresight: max_iter must be an integer >= 1, got {max_iter!r}"
        )
    max_iter = int(max_iter)
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            f"solve_perfect_foresight: tol must be a finite positive number, got {tol!r}"
        )
    dampening = float(dampening)
    if not np.isfinite(dampening) or not (0.0 < dampening <= 1.0):
        raise ValueError(
            f"solve_perfect_foresight: dampening must lie in (0, 1], got {dampening!r}; "
            "it is the initial Newton step length, so a zero or negative value makes no "
            "progress and a value above 1 overshoots the Newton step."
        )
    if method not in ("auto", "central", "complex"):
        raise ValueError(
            f"solve_perfect_foresight: unknown method {method!r}; "
            "expected one of 'auto', 'central', 'complex'"
        )

    # Handle exogenous path
    exo_arr = np.asarray(exogenous_path, dtype=float)
    if exo_arr.shape[0] == n_periods + 2:
        exo_sim = exo_arr[1 : n_periods + 1]
    elif exo_arr.shape[0] >= n_periods:
        exo_sim = exo_arr[:n_periods]
    else:
        raise ValueError(
            f"exogenous_path length ({exo_arr.shape[0]}) must be at least n_periods ({n_periods})"
        )

    # Variable names
    if variable_names is not None:
        v_names = tuple(str(v) for v in variable_names)
        if len(v_names) != n_vars:
            raise ValueError(
                f"len(variable_names)={len(v_names)} does not match n_vars={n_vars}"
            )
    else:
        v_names = tuple(f"y_{i}" for i in range(n_vars))

    # Initial guess for Y (T x n_vars)
    if initial_path is not None:
        Y = np.asarray(initial_path, dtype=float).copy()
        if Y.shape != (n_periods, n_vars):
            raise ValueError(
                f"initial_path shape {Y.shape} does not match (n_periods, n_vars)={(n_periods, n_vars)}"
            )
    else:
        # Linear interpolation between boundary conditions
        Y = np.zeros((n_periods, n_vars), dtype=float)
        for t in range(n_periods):
            w = (t + 1.0) / (n_periods + 1.0)
            Y[t] = (1.0 - w) * y_init_arr + w * y_ss_arr

    def _eval_stacked_residuals(Y_curr: np.ndarray) -> np.ndarray:
        R = np.zeros(n_vars * n_periods, dtype=float)
        for t in range(n_periods):
            y_l = y_init_arr if t == 0 else Y_curr[t - 1]
            y_c = Y_curr[t]
            y_p = y_ss_arr if t == n_periods - 1 else Y_curr[t + 1]
            eps = exo_sim[t]
            r = np.asarray(equations_fn(y_p, y_c, y_l, eps), dtype=float).ravel()
            if len(r) != n_vars:
                raise ValueError(
                    f"equations_fn returned {len(r)} equations at period t={t+1}, expected {n_vars}"
                )
            R[t * n_vars : (t + 1) * n_vars] = r
        return R

    # Pre-allocate sparse block-tridiagonal Jacobian indices
    # Non-zero blocks:
    #   B_t for t = 0..T-1 (T blocks)
    #   A_t for t = 0..T-2 (T-1 blocks)
    #   C_t for t = 1..T-1 (T-1 blocks)
    total_entries = (3 * n_periods - 2) * n_vars * n_vars
    rows = np.zeros(total_entries, dtype=np.int32)
    cols = np.zeros(total_entries, dtype=np.int32)
    data = np.zeros(total_entries, dtype=float)

    idx = 0
    for t in range(n_periods):
        # Diagonal block B_t: row block t, col block t
        for j in range(n_vars):
            for i in range(n_vars):
                rows[idx] = t * n_vars + i
                cols[idx] = t * n_vars + j
                idx += 1
        # Superdiagonal block A_t: row block t, col block t + 1
        if t < n_periods - 1:
            for j in range(n_vars):
                for i in range(n_vars):
                    rows[idx] = t * n_vars + i
                    cols[idx] = (t + 1) * n_vars + j
                    idx += 1
        # Subdiagonal block C_t: row block t, col block t - 1
        if t > 0:
            for j in range(n_vars):
                for i in range(n_vars):
                    rows[idx] = t * n_vars + i
                    cols[idx] = (t - 1) * n_vars + j
                    idx += 1

    # Auto-detect differentiation method
    step_fd = 1e-7
    use_complex = False
    if method == "complex":
        use_complex = True
    elif method == "auto":
        try:
            pert = y_init_arr.astype(complex)
            pert[0] += 1j * 1e-20
            out = np.asarray(
                equations_fn(
                    y_ss_arr.astype(complex),
                    pert,
                    y_init_arr.astype(complex),
                    exo_sim[0],
                ),
                dtype=complex,
            )
            if not np.all(out.imag == 0) and not np.isnan(out.imag).any():
                use_complex = True
        except Exception:
            use_complex = False

    # Evaluate initial residuals
    R = _eval_stacked_residuals(Y)
    res_norm = float(np.max(np.abs(R)))
    initial_res_norm = res_norm

    if verbose:
        print(f"Iter 0: initial max residual = {res_norm:.4e}")

    # NOTE: there is deliberately no "residual already below tol -> return the
    # initial guess with iterations=0" shortcut here.  An unscaled absolute
    # residual test is not a statement about the accuracy of the path: on a
    # model whose equations are written in small units every residual is small,
    # and that shortcut returned the raw linear-interpolation guess flagged
    # converged=True.  At least one Newton system is always solved, and its
    # step has to be negligible before convergence is claimed.
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
        for t in range(n_periods):
            y_l = y_init_arr if t == 0 else Y[t - 1]
            y_c = Y[t]
            y_p = y_ss_arr if t == n_periods - 1 else Y[t + 1]
            eps = exo_sim[t]

            # B_t: w.r.t y_curr
            for j in range(n_vars):
                if use_complex:
                    pert = y_c.astype(complex)
                    pert[j] += 1j * 1e-20
                    df = np.asarray(
                        equations_fn(
                            y_p.astype(complex),
                            pert,
                            y_l.astype(complex),
                            eps,
                        ),
                        dtype=complex,
                    ).imag / 1e-20
                else:
                    h = step_fd * max(1.0, abs(y_c[j]))
                    yp = y_c.copy()
                    yp[j] += h
                    ym = y_c.copy()
                    ym[j] -= h
                    fp = np.asarray(equations_fn(y_p, yp, y_l, eps), dtype=float).ravel()
                    fm = np.asarray(equations_fn(y_p, ym, y_l, eps), dtype=float).ravel()
                    df = (fp - fm) / (2.0 * h)
                for i in range(n_vars):
                    data[data_idx] = df[i]
                    data_idx += 1

            # A_t: w.r.t y_plus (lead) if t < n_periods - 1
            if t < n_periods - 1:
                for j in range(n_vars):
                    if use_complex:
                        pert = y_p.astype(complex)
                        pert[j] += 1j * 1e-20
                        df = np.asarray(
                            equations_fn(
                                pert,
                                y_c.astype(complex),
                                y_l.astype(complex),
                                eps,
                            ),
                            dtype=complex,
                        ).imag / 1e-20
                    else:
                        h = step_fd * max(1.0, abs(y_p[j]))
                        yp = y_p.copy()
                        yp[j] += h
                        ym = y_p.copy()
                        ym[j] -= h
                        fp = np.asarray(equations_fn(yp, y_c, y_l, eps), dtype=float).ravel()
                        fm = np.asarray(equations_fn(ym, y_c, y_l, eps), dtype=float).ravel()
                        df = (fp - fm) / (2.0 * h)
                    for i in range(n_vars):
                        data[data_idx] = df[i]
                        data_idx += 1

            # C_t: w.r.t y_lag (lag) if t > 0
            if t > 0:
                for j in range(n_vars):
                    if use_complex:
                        pert = y_l.astype(complex)
                        pert[j] += 1j * 1e-20
                        df = np.asarray(
                            equations_fn(
                                y_p.astype(complex),
                                y_c.astype(complex),
                                pert,
                                eps,
                            ),
                            dtype=complex,
                        ).imag / 1e-20
                    else:
                        h = step_fd * max(1.0, abs(y_l[j]))
                        yp = y_l.copy()
                        yp[j] += h
                        ym = y_l.copy()
                        ym[j] -= h
                        fp = np.asarray(equations_fn(y_p, y_c, yp, eps), dtype=float).ravel()
                        fm = np.asarray(equations_fn(y_p, y_c, ym, eps), dtype=float).ravel()
                        df = (fp - fm) / (2.0 * h)
                    for i in range(n_vars):
                        data[data_idx] = df[i]
                        data_idx += 1

        # Construct sparse CSC Jacobian and solve linear Newton system
        J = sp.csc_matrix(
            (data, (rows, cols)),
            shape=(n_vars * n_periods, n_vars * n_periods),
        )

        # Scale of the equations, used to make the residual test invariant to
        # writing the same model in different units.
        jac_scale = max(1.0, float(np.max(np.abs(data))))

        try:
            dY_flat = spla.spsolve(J, -R)
        except Exception as exc:
            warnings.warn(f"Singular Jacobian encountered at iteration {it+1}: {exc}")
            break

        dY = dY_flat.reshape(n_periods, n_vars)

        # Convergence test, evaluated BEFORE the step is taken: the residual is
        # small in the model's own units AND the Newton step this iterate calls
        # for is negligible relative to the path itself.
        y_scale = max(1.0, float(np.max(np.abs(Y))))
        step_norm = float(np.max(np.abs(dampening * dY))) if np.all(np.isfinite(dY)) else float("inf")
        res_ok = res_norm < tol * jac_scale
        step_ok = step_norm <= _STEP_TOL * y_scale
        if res_ok and step_ok:
            converged = True
            break

        # Apply update with dampening and Armijo backtracking line search.
        # Acceptance requires a sufficient DECREASE of the residual norm, not
        # merely a finite one -- otherwise the "line search" is only a NaN
        # guard and Newton can oscillate between two points forever.
        step_scale = float(dampening)
        accepted = False
        saw_finite_trial = False
        for _ in range(10):
            Y_trial = Y + step_scale * dY
            try:
                R_trial = _eval_stacked_residuals(Y_trial)
                trial_norm = float(np.max(np.abs(R_trial)))
                if np.isfinite(trial_norm) and trial_norm < 1e12:
                    saw_finite_trial = True
                    if trial_norm <= (1.0 - _ARMIJO_C * step_scale) * res_norm:
                        Y = Y_trial
                        R = R_trial
                        res_norm = trial_norm
                        accepted = True
                        break
            except Exception:
                pass
            step_scale *= 0.5

        if not accepted:
            if saw_finite_trial:
                # Every trial step was finite but none reduced the residual:
                # the merit function has stalled (typically a local minimum of
                # |f| that is not a root, or a Newton cycle).  Keep the current
                # iterate -- the lowest-residual one seen -- rather than moving
                # to a point the line search just rejected.
                stalled = True
                warnings.warn(
                    f"solve_perfect_foresight: the Newton line search found no step that "
                    f"reduces the residual at iteration {it+1}; the iteration has stalled at "
                    f"residual norm {res_norm:.4e} (scaled tolerance {tol * jac_scale:.4e}), "
                    f"which is a stationary point of |f| but not a solution. Returning the "
                    f"last accepted iterate with converged=False. Try a different "
                    f"initial_path, a smaller dampening, or a shorter n_periods.",
                    stacklevel=2,
                )
                break
            # Could not find a valid step at all (non-finite residuals)
            warnings.warn(f"Newton-Raphson line search failed at iteration {it+1}")
            break

        step_norm = float(np.max(np.abs(step_scale * dY)))

        if verbose:
            print(f"Iter {iterations}: max residual = {res_norm:.4e}, step = {step_norm:.4e}")

    if not converged and not stalled:
        y_scale = max(1.0, float(np.max(np.abs(Y))))
        if res_ok and not step_ok:
            warnings.warn(
                f"solve_perfect_foresight did not converge in {max_iter} iterations: the "
                f"residual norm {res_norm:.4e} is below the scaled tolerance "
                f"{tol * jac_scale:.4e}, but the Newton step {step_norm:.4e} is still large "
                f"relative to the path (scale {y_scale:.4e}), so the residual test is not "
                f"informative for this model -- its equations are probably written in units "
                f"that make every residual small. The returned path is NOT a solution; "
                f"rescale the equations or raise max_iter."
            )
        elif step_ok and not res_ok:
            warnings.warn(
                f"solve_perfect_foresight did not converge in {max_iter} iterations: the "
                f"Newton step {step_norm:.4e} has stalled while the residual norm "
                f"{res_norm:.4e} is still above the scaled tolerance {tol * jac_scale:.4e}. "
                f"The Jacobian is probably ill-conditioned at the returned path."
            )
        else:
            warnings.warn(
                f"solve_perfect_foresight did not converge in {max_iter} iterations "
                f"(residual norm {res_norm:.4e} > {tol * jac_scale:.4e}, "
                f"Newton step {step_norm:.4e})"
            )

    term_err = float(np.max(np.abs(Y[-1] - y_ss_arr)))
    path_df = pd.DataFrame(
        Y,
        index=pd.RangeIndex(1, n_periods + 1, name="t"),
        columns=list(v_names),
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
    "solve_perfect_foresight",
]
