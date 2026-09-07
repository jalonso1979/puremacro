"""Piecewise-Linear Solution for Occasionally Binding Constraints (OccBin).

Implementation of Guerrieri & Iacoviello (2015), "OccBin: A Toolkit for
Solving Dynamic Models with Occasionally Binding Constraints Easily",
Journal of Monetary Economics 70, 22-38.

Solves models with occasionally binding constraints (such as the Zero Lower
Bound on nominal interest rates, borrowing limits, or irreversible investment)
under perfect foresight via backward recursion over piecewise-linear regimes.

Scope and honest limitations
----------------------------
* **Two regimes only.** A reference (unconstrained) regime and one alternative
  (constrained) regime. Models with several simultaneously occasionally
  binding constraints are out of scope.
* **Perfect foresight.** Every row of ``shock_sequence`` is *anticipated at
  t = 1*: agents in period 1 know the whole shock path. The backward
  recursion folds the shocks into the period-by-period drift
  ``D_t = M_t^{-1}(-(A_+^{r_t} D_{t+1} + c^{r_t} + B_u^{r_t} u_t))``, so a
  shock dated t > 1 moves the path from period 1 onwards. To simulate an
  *unanticipated* shock at date s, run the solver from date s with the state
  reached at s - 1 as the initial condition.
* **Arbitrary regime sequences.** The regime is a full boolean vector over
  the horizon, not a leading spell, so spells that start after t = 1, and
  several disjoint spells, are represented exactly.
* **Terminal condition.** The recursion is seeded with the reference-regime
  decision rule at t = horizon + 1. If the constraint still binds in the last
  simulated period that assumption is *not* verified: the solver then returns
  ``converged=False`` and warns rather than pretending to have a solution.
* **``converged=True`` means verified.** It is set only when the regime
  iteration reached a fixed point *and* the returned path satisfies the
  constraint where that is meaningful for the constraint's style (see below)
  *and* the terminal condition was tested. Every
  other outcome returns ``converged=False`` together with a warning naming
  the reason.
* **Two constraint styles.** If the alternative regime replaces the equation
  that determines the constrained variable (a ZLB-style peg), the test for
  whether a spell keeps binding uses the variable's notional value, solved
  out of the *reference* equation that the alternative regime replaced. If no
  switching equation contains the constrained variable (a trigger-style
  constraint, e.g. "credit policy activates once the spread exceeds x"), the
  variable stays endogenous in both regimes and its simulated value is used
  directly. ``OccBinConstraint.relax_variable`` overrides both.
  What ``converged=True`` verifies differs by style, necessarily. A **peg**
  pins the variable AT the bound while binding, so the whole path must respect
  the bound and every period is checked. A **trigger** fires *because* the
  variable has passed the threshold, so while the constraint is active the
  variable is supposed to sit beyond it — checking those periods against the
  bound would reject every correct solution. For trigger-style constraints the
  verified property is therefore that no period declared *slack* has tripped
  the trigger; that the binding periods did trip it is already guaranteed by
  the regime sequence being a fixed point, which is checked separately and is
  the stronger condition.

  There is a third shape that is neither, and it is refused rather than
  guessed at: the alternative regime pegs the constrained variable to a
  constant in a row that does *not* determine it in the reference model (for
  instance a leverage cap written into the public-credit rule rather than into
  the bank's incentive constraint). There is then no reference equation to
  solve a notional value out of, and the default relax test would compare the
  pegged value against the very bound it is pegged to -- vacuously "relaxed"
  in every binding period. Such a constraint must supply
  ``relax_variable``: the multiplier, or the instrument that enforces the peg,
  tested against zero. Complementary slackness then reads "the constraint
  binds while the instrument still has to push in the direction that enforces
  it", which is testable.
"""
from __future__ import annotations

import warnings
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
        It must name a model variable, and it is **required** when the
        alternative regime pegs ``variable`` to a constant in an equation that
        does not determine ``variable`` in the reference regime --
        :func:`solve_occbin` raises rather than run a vacuous relax test.
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
        Simulated trajectory of all endogenous variables over the horizon,
        indexed by period ``t = 1 .. horizon`` (matching :meth:`plot` and
        :meth:`summary`).
    regimes : list[int]
        Regime indicator for each period (0 = reference regime, 1 = constrained
        regime). This is the regime sequence the returned path was actually
        solved under, whether or not it is a verified fixed point.
    binding_periods : int
        Total number of periods in which the constraint binds, i.e.
        ``sum(regimes)``. Spells need not start at t = 1 and need not be
        contiguous, so this is a count, not a spell length.
    converged : bool
        ``True`` only if the regime iteration reached a fixed point, the
        returned path passes the style-appropriate bound check (peg-style:
        every period; trigger-style: every period declared slack has not
        tripped the trigger), and the
        constraint is slack in the final period so that the terminal condition
        is verified. Otherwise ``False``, and a warning naming the reason was
        emitted by :func:`solve_occbin`.
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
        Structural shocks, **all anticipated at t = 1** (perfect foresight).
        Either 1D (shape ``(n_shocks,)``) for a one-shot shock at t=1, or 2D
        (shape ``(n_periods, n_shocks)``) for a whole announced path. A shock
        dated t > 1 therefore already moves the period-1 response; to simulate
        an unanticipated shock at date s, re-run the solver from date s.
    max_iter : int, default 50
        Maximum number of regime-guess iterations.
    horizon : int, default 40
        Simulation horizon (number of periods to simulate), must be >= 1.

    Returns
    -------
    OccBinResult
        Container holding simulated path, regime timeline, binding periods,
        convergence diagnostics, and visualization/export methods.

    Notes
    -----
    ``result.converged`` is ``True`` only for a *verified* solution: the regime
    iteration reached a fixed point, the returned path satisfies the constraint
    where that is meaningful for the constraint's style, and the constraint is
    slack in the final period
    so that the terminal condition (the reference regime resumes after the
    horizon) is actually tested. When any of those fails the function returns
    ``converged=False`` and emits a ``UserWarning`` naming the reason; the
    returned path is then the one solved under the regime sequence reported in
    ``result.regimes``, kept only as a diagnostic.

    Raises
    ------
    ValueError
        If ``horizon`` or ``max_iter`` is not a positive integer, if the
        constrained variable is not a model variable, or if it appears in no
        equation of the reference model (so neither its notional value nor the
        binding test is defined).
    """
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"solve_occbin: horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"solve_occbin: max_iter must be an integer >= 1, got {max_iter!r}")
    horizon = int(horizon)
    max_iter = int(max_iter)

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

    # 2. Reference-regime decision rule X_t = P_0 X_{t-1}, used to seed the
    #    backward recursion at t = horizon + 1.
    dr = reference_model.decision_rules()
    P_0 = np.zeros((n_vars, n_vars))
    for s in reference_model.states:
        idx_s = variables.index(s)
        P_0[:, idx_s] = dr.ghx[s].values

    # Find the row index of the constrained variable
    if constraint.variable not in variables:
        raise ValueError(f"constraint variable {constraint.variable!r} not found in model variables: {variables}")
    idx_var = variables.index(constraint.variable)

    # Resolve the relax (shadow / multiplier) variable up front: a name that is
    # not a model variable used to be dropped silently, which turns an explicit
    # relax rule into the default one without telling anybody.
    relax_idx = None
    if constraint.relax_variable is not None:
        if constraint.relax_variable not in variables:
            raise ValueError(
                f"solve_occbin: relax_variable {constraint.relax_variable!r} is not a model "
                f"variable; expected one of {variables}"
            )
        relax_idx = variables.index(constraint.relax_variable)

    # Identify the equation row that determines the constrained variable in the
    # reference regime.  It must (a) differ between the two regimes -- that is
    # what makes it the switching equation -- and (b) actually contain the
    # constrained variable, otherwise the shadow (notional) value solved out of
    # it below is meaningless.  Scanning only A_0/c/B_u, or taking the first
    # differing row without checking (b), makes the answer depend on the order
    # in which the model's equations happen to be written.
    diff_rows = np.where(
        (np.linalg.norm(A_0_0 - A_0_1, axis=1) > 1e-8)
        | (np.linalg.norm(A_p_0 - A_p_1, axis=1) > 1e-8)
        | (np.linalg.norm(A_m_0 - A_m_1, axis=1) > 1e-8)
        | (np.linalg.norm(B_u_0 - B_u_1, axis=1) > 1e-8)
        | (np.abs(c_0 - c_1) > 1e-8)
    )[0]
    has_var = np.abs(A_0_0[:, idx_var]) > 1e-12
    switching_rows = [int(r) for r in diff_rows if has_var[r]]
    if switching_rows:
        # The constrained regime replaces the equation that determines the
        # constrained variable (a ZLB-style peg): its notional value has to be
        # recovered from the reference-regime equation.
        eq_row = switching_rows[0]
    elif np.any(has_var):
        # No switching equation pins the constrained variable, i.e. the
        # constrained regime leaves that variable endogenously determined by
        # the same equation (a trigger-style constraint such as "public credit
        # policy kicks in once the spread exceeds x"). Its simulated value is
        # then already its notional value, and no shadow needs solving out.
        eq_row = None
    else:
        raise ValueError(
            f"solve_occbin: the constrained variable {constraint.variable!r} does not appear "
            f"contemporaneously in any equation of the reference model (every entry of column "
            f"{idx_var} of its contemporaneous Jacobian is negligible), so neither its notional "
            f"value nor the binding test is defined. Check the reference model, or constrain a "
            f"variable the model actually determines."
        )

    # A trap that the eq_row rule above cannot see. If the alternative regime
    # pegs the constrained variable to a constant in a row that does NOT
    # determine it in the reference regime, then eq_row is None (nothing to
    # solve the notional value out of) and the default relax test compares the
    # simulated value -- pinned AT the bound by that very peg -- against the
    # bound. That test is vacuous: it answers "the constraint has just
    # relaxed" in every binding period, whatever the economics, so the spell
    # can never be longer than the iteration's own transient. Such a
    # constraint needs an explicit `relax_variable` (the multiplier, or the
    # instrument that enforces the peg, tested against zero); refuse to guess.
    if eq_row is None and relax_idx is None:
        pegged = []
        for r in diff_rows:
            if has_var[r]:
                continue
            a_r = float(A_0_1[r, idx_var])
            if abs(a_r) <= 1e-12:
                continue
            others = A_0_1[r].copy()
            others[idx_var] = 0.0
            scale = 1e-10 * max(1.0, abs(a_r))
            if (
                np.max(np.abs(others)) <= scale
                and np.max(np.abs(A_p_1[r])) <= scale
                and np.max(np.abs(A_m_1[r])) <= scale
                and np.max(np.abs(B_u_1[r])) <= scale
            ):
                pegged.append(int(r))
        if pegged:
            raise ValueError(
                f"solve_occbin: the alternative regime pegs {constraint.variable!r} to a "
                f"constant in equation row(s) {pegged}, but that row does not determine "
                f"{constraint.variable!r} in the reference model, so there is no reference "
                f"equation to solve its notional value out of. The relax test would then "
                f"compare the pegged value against the very bound it is pegged to and "
                f"relax in every period. Set OccBinConstraint.relax_variable to the "
                f"multiplier or the instrument that enforces the peg (with "
                f"relax_threshold=0.0 and the sign that means 'the constraint would have "
                f"to push the wrong way'), so the exit condition is testable."
            )

    # 3. Backward recursion engine for an arbitrary regime sequence.
    #
    #    For a regime vector r = (r_1, ..., r_H) the perfect-foresight path
    #    obeys X_t = P_t X_{t-1} + D_t with
    #        M_t = A_0^{r_t} + A_+^{r_t} P_{t+1}
    #        P_t = M_t^{-1} (-A_-^{r_t})
    #        D_t = M_t^{-1} (-(A_+^{r_t} D_{t+1} + c^{r_t} + B_u^{r_t} u_t))
    #    seeded at t = H + 1 with the reference decision rule (P_0, 0).  The
    #    anticipated shocks live in the drift D_t, so every u_t is loaded with
    #    the matrices of the regime actually in force at t, and a shock dated
    #    t > 1 propagates backwards to period 1 as perfect foresight requires.
    regime_matrices = (
        (A_p_0, A_0_0, A_m_0, B_u_0, c_0),
        (A_p_1, A_0_1, A_m_1, B_u_1, c_1),
    )

    def compute_decision_rules(regime: np.ndarray):
        P_seq: list[np.ndarray | None] = [None] * (horizon + 1)
        D_seq: list[np.ndarray | None] = [None] * (horizon + 1)
        P_next = P_0
        D_next = np.zeros(n_vars)
        for t in range(horizon, 0, -1):
            A_p_r, A_0_r, A_m_r, B_u_r, c_r = regime_matrices[int(regime[t - 1])]
            M_t = A_0_r + A_p_r @ P_next
            P_t = _safe_solve(M_t, -A_m_r)
            D_t = _safe_solve(M_t, -(A_p_r @ D_next + c_r + B_u_r @ shocks_mat[t - 1]))
            P_seq[t] = P_t
            D_seq[t] = D_t
            P_next, D_next = P_t, D_t
        return P_seq, D_seq

    # Reference-regime row used to solve out the shadow (notional) value of the
    # constrained variable; `a_var` is guaranteed non-negligible by the eq_row
    # selection above, so there is no silent unit fallback here.
    if eq_row is not None:
        a_var = float(A_0_0[eq_row, idx_var])
        row_others = A_0_0[eq_row].copy()
        row_others[idx_var] = 0.0

    # 4. Forward simulation under a conjectured regime sequence
    def simulate_path(regime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P_seq, D_seq = compute_decision_rules(regime)
        X = np.zeros((horizon + 1, n_vars))
        for t in range(1, horizon + 1):
            X[t] = P_seq[t] @ X[t - 1] + D_seq[t]

        sim_X = X[1:]

        if eq_row is None:
            # No switching equation pins the constrained variable, so it is
            # its own notional value.
            return sim_X, sim_X[:, idx_var].copy()

        shadow_vals = np.zeros(horizon)
        for t in range(horizon):
            x_prev = X[t]
            x_curr = sim_X[t]
            x_next = sim_X[t + 1] if t + 1 < horizon else P_0 @ x_curr
            u_t = shocks_mat[t]

            # Shadow value from reference equation row:
            other_curr = row_others @ x_curr
            term_next = A_p_0[eq_row, :] @ x_next
            term_prev = A_m_0[eq_row, :] @ x_prev
            term_shock = B_u_0[eq_row, :] @ u_t
            shadow_vals[t] = - (other_curr + term_next + term_prev + term_shock + c_0[eq_row]) / a_var

        return sim_X, shadow_vals

    def next_regime(regime: np.ndarray, sim_X: np.ndarray, shadow_vals: np.ndarray) -> np.ndarray:
        """Guerrieri-Iacoviello regime update, period by period."""
        upd = np.zeros(horizon, dtype=int)
        for t in range(horizon):
            if regime[t] == 1:
                # Inside a conjectured spell: does the constraint still bind?
                val = sim_X[t, relax_idx] if relax_idx is not None else shadow_vals[t]
                binds = not bool(constraint.evaluate_relax(val))
            else:
                # Outside: would the unconstrained path violate the bound?
                binds = bool(constraint.evaluate(sim_X[t, idx_var]))
            upd[t] = 1 if binds else 0
        return upd

    # 5. Regime-guess iteration over the whole horizon
    regime = np.zeros(horizon, dtype=int)
    history: set[tuple[int, ...]] = {tuple(regime.tolist())}
    fixed_point = False
    cycled_to: np.ndarray | None = None
    iteration = 0

    for iteration in range(1, max_iter + 1):  # max_iter >= 1 is validated above
        sim_X, shadow_vals = simulate_path(regime)
        upd = next_regime(regime, sim_X, shadow_vals)

        if np.array_equal(upd, regime):
            fixed_point = True
            break

        key = tuple(upd.tolist())
        if key in history:
            # The guess cycles.  The path in hand was solved under `regime`,
            # which the solver's own binding test has just rejected, so this is
            # NOT a solution -- report it as such instead of picking a winner.
            cycled_to = upd
            break

        history.add(key)
        regime = upd
    else:
        # max_iter exhausted; `regime` holds the last update, re-simulate it so
        # that the returned path and the reported regime sequence agree.
        sim_X, shadow_vals = simulate_path(regime)

    converged = fixed_point
    reasons: list[str] = []

    if cycled_to is not None:
        solved_for = [int(t) + 1 for t in np.flatnonzero(regime)]
        rejected_to = [int(t) + 1 for t in np.flatnonzero(cycled_to)]
        reasons.append(
            f"the regime iteration cycled at iteration {iteration}: the path was solved with "
            f"the constraint binding in period(s) {solved_for[:12]}"
            f"{' ...' if len(solved_for) > 12 else ''}, but the solver's own binding test on "
            f"that very path returns {rejected_to[:12]}"
            f"{' ...' if len(rejected_to) > 12 else ''}, a guess already visited, so no regime "
            f"sequence is a fixed point"
        )
    elif not fixed_point:
        reasons.append(
            f"the regime iteration did not reach a fixed point within max_iter={max_iter}"
        )

    # 6. Post-hoc verification of the returned path against the constraint.
    #    `converged=True` must mean "this path satisfies the constraint and the
    #    regime sequence it was solved under", so the bound is tested here even
    #    when the regime iteration reported a fixed point.
    #
    #    What "satisfies the bound" means depends on the constraint style. For a
    #    ZLB-style peg (``eq_row is not None``) the alternative regime pins the
    #    constrained variable AT the bound, so no period of the returned path may
    #    lie beyond it. For a trigger-style constraint (``eq_row is None``, e.g.
    #    "credit policy activates once the spread exceeds x") the variable stays
    #    endogenous in both regimes and is *meant* to sit beyond the threshold
    #    while the alternative regime is in force -- that is what tripping the
    #    trigger means, not a violation. There the testable property is that no
    #    period declared slack has in fact tripped the trigger. Testing the whole
    #    path in that case would make every trigger-style constraint that ever
    #    binds report ``converged=False``, however clean its fixed point.
    bound_tol = 1e-8 * max(1.0, abs(float(constraint.threshold)))
    thresh = float(constraint.threshold)
    checked = np.arange(horizon) if eq_row is not None else np.flatnonzero(regime == 0)
    col = sim_X[checked, idx_var]
    if constraint.operator in ("<", "<="):
        violated = checked[col < thresh - bound_tol]
    else:
        violated = checked[col > thresh + bound_tol]
    if violated.size:
        vals = sim_X[violated, idx_var]
        worst = float(vals.min() if constraint.operator in ("<", "<=") else vals.max())
        reasons.append(
            f"the returned path violates {constraint!r} in period(s) "
            f"{[int(t) + 1 for t in violated[:12]]}"
            f"{' ...' if violated.size > 12 else ''} "
            f"(worst {constraint.variable}={worst:.6g} against threshold {thresh:.6g})"
        )

    if regime[-1] == 1:
        reasons.append(
            f"the constraint still binds in the last simulated period (t={horizon}), so the "
            f"terminal condition -- that the reference regime resumes after the horizon -- is "
            f"assumed rather than verified; increase horizon"
        )

    if reasons:
        converged = False
        warnings.warn(
            "solve_occbin did not produce a verified solution (converged=False): "
            + "; ".join(reasons)
            + ". The returned path is the one solved under `result.regimes` and is a "
            "diagnostic, not a solution.",
            UserWarning,
            stacklevel=2,
        )

    # 7. Construct OccBinResult
    period_index = pd.RangeIndex(1, horizon + 1, name="t")
    sim_df = pd.DataFrame(sim_X, columns=variables, index=period_index)
    shadow_df = sim_df.copy()
    shadow_df[f"{constraint.variable}_shadow"] = shadow_vals

    regimes = [int(v) for v in regime]

    return OccBinResult(
        simulated_path=sim_df,
        regimes=regimes,
        binding_periods=int(regime.sum()),
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
