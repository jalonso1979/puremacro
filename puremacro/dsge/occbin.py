"""Piecewise-Linear Solution for Occasionally Binding Constraints (OccBin).

Implementation of Guerrieri & Iacoviello (2015), "OccBin: A Toolkit for
Solving Dynamic Models with Occasionally Binding Constraints Easily",
Journal of Monetary Economics 70, 22-38.

Solves models with occasionally binding constraints (such as the Zero Lower
Bound on nominal interest rates, borrowing limits, or irreversible investment)
under perfect foresight via backward recursion over piecewise-linear regimes.

Scope and honest limitations
----------------------------
* **One constraint in** :func:`solve_occbin`, **K constraints in**
  :func:`solve_multiconstraint_occbin`. The single-constraint solver works
  with a reference (unconstrained) regime and one alternative (constrained)
  regime. The multi-constraint solver takes one constrained model per
  constraint and builds the ``2**K`` regimes itself: regime ``r`` is a
  **bitmask**, bit ``k`` of ``r`` is set exactly when constraint ``k`` (the
  ``k``-th entry of the model mapping/sequence, ``k = 0`` first) binds, so
  with ``K = 2`` the regimes are ``0`` (both slack), ``1`` (first only),
  ``2`` (second only) and ``3`` (both). ``OccBinResult.regimes`` reports that
  bitmask per period. The system for regime ``r`` is the reference system
  with **every** row in which constrained model ``k`` differs from the
  reference spliced in, for each set bit ``k``; two constrained models must
  therefore not rewrite the same equation row, and the solver raises when
  they do. :func:`solve_occbin` forwards a mapping or sequence of models and
  constraints to the multi-constraint solver, whose ``converged`` contract
  and warnings are the same as below.
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

import sys
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
        :func:`solve_occbin` and :func:`solve_multiconstraint_occbin` raise
        rather than run a vacuous relax test.
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


# ---------------------------------------------------------------------------
# Smooth Operators & Fischer-Burmeister Relaxation
# ---------------------------------------------------------------------------


def smin_tau(x: float | np.ndarray, y: float | np.ndarray, tau: float = 0.01) -> float | np.ndarray:
    """Smooth-min operator with temperature parameter tau > 0.

    smin_tau(x, y) = 0.5 * (x + y - sqrt((x - y)^2 + tau^2))
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr - y_arr
    res = 0.5 * (x_arr + y_arr - np.sqrt(diff**2 + tau**2))
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(res)
    return res


def smax_tau(x: float | np.ndarray, y: float | np.ndarray, tau: float = 0.01) -> float | np.ndarray:
    """Smooth-max operator with temperature parameter tau > 0.

    smax_tau(x, y) = 0.5 * (x + y + sqrt((x - y)^2 + tau^2))
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr - y_arr
    res = 0.5 * (x_arr + y_arr + np.sqrt(diff**2 + tau**2))
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(res)
    return res


def d_smax_tau(x: float | np.ndarray, y: float | np.ndarray, tau: float = 0.01, wrt: str = "x") -> float | np.ndarray:
    """Smooth C^inf derivative of smax_tau(x, y).

    d/dx smax_tau(x, y) = 0.5 * (1 + (x - y) / sqrt((x - y)^2 + tau^2))
    d/dy smax_tau(x, y) = 0.5 * (1 - (x - y) / sqrt((x - y)^2 + tau^2))
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr - y_arr
    root = np.sqrt(diff**2 + tau**2)
    if wrt == "x":
        res = 0.5 * (1.0 + diff / root)
    elif wrt == "y":
        res = 0.5 * (1.0 - diff / root)
    else:
        raise ValueError(f"wrt must be 'x' or 'y', got {wrt!r}")
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(res)
    return res


def d_smin_tau(x: float | np.ndarray, y: float | np.ndarray, tau: float = 0.01, wrt: str = "x") -> float | np.ndarray:
    """Smooth C^inf derivative of smin_tau(x, y).

    d/dx smin_tau(x, y) = 0.5 * (1 - (x - y) / sqrt((x - y)^2 + tau^2))
    d/dy smin_tau(x, y) = 0.5 * (1 + (x - y) / sqrt((x - y)^2 + tau^2))
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = x_arr - y_arr
    root = np.sqrt(diff**2 + tau**2)
    if wrt == "x":
        res = 0.5 * (1.0 - diff / root)
    elif wrt == "y":
        res = 0.5 * (1.0 + diff / root)
    else:
        raise ValueError(f"wrt must be 'x' or 'y', got {wrt!r}")
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(res)
    return res


smooth_min = smin_tau
smooth_max = smax_tau
d_smooth_max = d_smax_tau
d_smooth_min = d_smin_tau


def fischer_burmeister(a: float | np.ndarray, b: float | np.ndarray, tau: float = 0.01) -> float | np.ndarray:
    """Fischer-Burmeister complementary condition relaxation.

    Phi_tau(a, b) = a + b - sqrt(a^2 + b^2 + 2 * tau^2)
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    res = a_arr + b_arr - np.sqrt(a_arr**2 + b_arr**2 + 2.0 * tau**2)
    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return float(res)
    return res


Phi_tau = fischer_burmeister


def grad_fischer_burmeister(
    a: float | np.ndarray, b: float | np.ndarray, tau: float = 0.01
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Smooth gradient of Fischer-Burmeister relaxation w.r.t (a, b)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    root = np.sqrt(a_arr**2 + b_arr**2 + 2.0 * tau**2)
    d_da = 1.0 - a_arr / root
    d_db = 1.0 - b_arr / root
    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return float(d_da), float(d_db)
    return d_da, d_db


@dataclass(frozen=True)
class DifferentiableOccBinResult:
    """Result of Differentiable OccBin simulation with smooth relaxation (tau > 0).

    Attributes
    ----------
    simulated_path : pd.DataFrame
        Simulated trajectory of all endogenous variables over the horizon.
    weights : np.ndarray
        Continuous regime weights w_t in (0, 1) over the horizon.
    regimes : list[int]
        Discretized regime indicator (w_t >= 0.5) for comparison with discrete OccBin.
    binding_periods : int
        Number of periods where w_t >= 0.5.
    converged : bool
        Whether the fixed-point relaxation iteration reached convergence.
    iterations : int
        Number of fixed-point iterations.
    reference_model : Any
        Underlying unconstrained reference model.
    constrained_model : Any
        Underlying constrained regime model.
    constraint : OccBinConstraint
        Constraint definition.
    tau : float
        Temperature parameter tau.
    shadow_path : pd.DataFrame | None
        Simulated path with shadow / notional values.
    loss : float | None
        Objective loss value if target data provided.
    gradient : dict[str, float] | np.ndarray | None
        Gradient of simulated path or loss w.r.t model parameters.
    param_sensitivities : pd.DataFrame | None
        Sensitivities dX/dtheta if computed.
    """

    simulated_path: pd.DataFrame
    weights: np.ndarray
    regimes: list[int]
    binding_periods: int
    converged: bool
    iterations: int
    reference_model: Any
    constrained_model: Any
    constraint: OccBinConstraint
    tau: float = 0.01
    shadow_path: pd.DataFrame | None = None
    loss: float | None = None
    gradient: dict[str, float] | np.ndarray | None = None
    param_sensitivities: pd.DataFrame | None = None

    @property
    def path(self) -> pd.DataFrame:
        """Alias for simulated_path matching puremacro convention."""
        return self.simulated_path

    @property
    def regime_history(self) -> np.ndarray:
        """Array representation of regime sequence."""
        return np.asarray(self.regimes)

    def to_frame(self) -> pd.DataFrame:
        """Return simulated path as a DataFrame."""
        return self.simulated_path.copy()

    def __getitem__(self, key: str) -> pd.Series:
        """Allow subscript access to simulated variable series."""
        if key in self.simulated_path.columns:
            return self.simulated_path[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"DifferentiableOccBinResult has no variable or attribute {key!r}")

    def summary(self) -> str:
        """Render a formatted summary of the differentiable OccBin simulation."""
        status = "Converged" if self.converged else "Did NOT converge"
        horizon = len(self.simulated_path)
        c_desc = repr(self.constraint) if self.constraint else "Unspecified"

        lines = [
            "DIFFERENTIABLE OCCBIN SIMULATION REPORT (Smooth Relaxation)",
            "=" * 78,
            f"Constraint         : {c_desc}",
            f"Temperature tau    : {self.tau:.6g}",
            f"Algorithm status   : {status} in {self.iterations} iteration(s)",
            f"Binding duration   : {self.binding_periods} period(s) out of {horizon} (w_t >= 0.5)",
            f"Mean regime weight : {float(np.mean(self.weights)):.4f}",
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
        """Plot the simulated trajectories with continuous weight shading."""
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

            # Continuous weight shading
            for t in range(horizon):
                w_t = float(self.weights[t])
                if w_t > 0.05:
                    ax.axvspan(t + 0.5, t + 1.5, color="0.8", alpha=min(0.5, w_t * 0.5))

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

        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        return fig


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
        solved under, whether or not it is a verified fixed point. For a
        result of :func:`solve_multiconstraint_occbin` the entry is a bitmask:
        bit ``k`` is set exactly when constraint ``k`` (in the order of the
        constrained-model mapping/sequence) binds in that period, so with two
        constraints ``3`` means both bind and ``2`` means only the second one.
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
    constrained_models: dict[str, Any] | None = None
    constraints: dict[str, OccBinConstraint] | None = None

    @property
    def path(self) -> pd.DataFrame:
        """Alias for simulated_path matching puremacro convention."""
        return self.simulated_path

    @property
    def regime_history(self) -> np.ndarray:
        """Array representation of regime sequence."""
        return np.asarray(self.regimes)

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
        if self.constraints:
            c_desc = ", ".join(repr(c) for c in self.constraints.values())
        elif self.constraint:
            c_desc = repr(self.constraint)
        else:
            c_desc = "Unspecified"

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
            binding_mask = np.array(self.regimes[:horizon]) > 0
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
            elif self.constraints is not None:
                for c_name, c_obj in self.constraints.items():
                    if var == c_obj.variable:
                        ax.axhline(
                            c_obj.threshold,
                            color="0.4",
                            linestyle="--",
                            linewidth=1.0,
                            label=f"Threshold ({c_obj.threshold:.3g})",
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


@dataclass(frozen=True)
class PiecewiseKalmanResult:
    """Result of Piecewise Kalman Filter likelihood evaluation and state filtering.

    Attributes
    ----------
    log_likelihood : float
        Total log-likelihood ln L(Y | theta).
    filtered_states : pd.DataFrame, shape (T_obs, n_states)
        Filtered state estimates x_{t|t}.
    regime_history : pd.Series | pd.DataFrame
        Realized regime sequence r_t in {0, ..., 2^K - 1}.
    filtered_covariances : list[np.ndarray]
        Filtered state error covariances P_{t|t}.
    imputed_shocks : pd.DataFrame
        Imputed contemporaneous innovations u_{t|t}.
    forecast_errors : pd.DataFrame
        One-step prediction errors v_t.
    converged : bool
        Whether regime consistency succeeded at all observation dates.
    iterations : int
        Average/total regime consistency iterations.
    reference_model : Any, optional
        Reference DSGE model.
    constraints : dict[str, OccBinConstraint] | None, optional
        OccBin constraints applied during filtering.
    """

    log_likelihood: float
    filtered_states: pd.DataFrame
    regime_history: pd.Series | pd.DataFrame
    filtered_covariances: list[np.ndarray]
    imputed_shocks: pd.DataFrame
    forecast_errors: pd.DataFrame
    converged: bool = True
    iterations: int = 1
    reference_model: Any = None
    constraints: dict[str, OccBinConstraint] | None = None

    @property
    def regimes(self) -> list[int]:
        """List of integer regime indices matching sample length."""
        if isinstance(self.regime_history, pd.Series):
            return [int(x) for x in self.regime_history.tolist()]
        elif isinstance(self.regime_history, pd.DataFrame):
            return [int(x) for x in self.regime_history.iloc[:, 0].tolist()]
        return [int(x) for x in np.asarray(self.regime_history).ravel()]

    def __iter__(self):
        """Enable tuple unpacking: ll, filtered_states, regimes = pkf(...)"""
        yield self.log_likelihood
        yield self.filtered_states
        yield self.regime_history

    def to_frame(self) -> pd.DataFrame:
        """Return filtered states as a DataFrame."""
        return self.filtered_states.copy()

    def summary(self, as_dataframe: bool = False) -> str | pd.DataFrame:
        """Render a publication-quality report of the Piecewise Kalman Filter results."""
        t_obs = len(self.filtered_states)
        reg_arr = np.asarray(self.regimes)
        n_binding = int(np.sum(reg_arr > 0))
        binding_pct = 100.0 * n_binding / max(1, t_obs)

        if as_dataframe:
            return pd.DataFrame(
                {
                    "Log-Likelihood": [self.log_likelihood],
                    "Observations": [t_obs],
                    "Binding Periods": [n_binding],
                    "Binding Pct (%)": [binding_pct],
                    "Converged": [self.converged],
                }
            )

        lines = [
            "PIECEWISE KALMAN FILTER REPORT (Giovannini, Pfeiffer & Ratto 2021)",
            "=" * 78,
            f"Log-likelihood     : {self.log_likelihood:.6f}",
            f"Observations       : {t_obs}",
            f"Constraint regimes : {n_binding} period(s) binding ({binding_pct:.1f}%)",
            f"Regime convergence : {'Verified' if self.converged else 'Failed'}",
            f"Recent regimes     : {self.regimes[-min(t_obs, 20):]}",
            "-" * 78,
            "FILTERED STATE SUMMARY STATISTICS",
            "-" * 78,
        ]
        stats_df = pd.DataFrame(
            {
                "Initial (t=1)": self.filtered_states.iloc[0],
                "Min": self.filtered_states.min(),
                "Max": self.filtered_states.max(),
                "Mean": self.filtered_states.mean(),
                "Final (t=T)": self.filtered_states.iloc[-1],
            }
        )
        lines.append(stats_df.round(6).to_string())
        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(self, variables: Sequence[str] | None = None, style: str = "publication", ax=None):
        """Plot filtered state trajectories with shaded constrained regimes."""
        import matplotlib.pyplot as plt

        if variables is None:
            variables = list(self.filtered_states.columns)
        else:
            variables = [v for v in variables if v in self.filtered_states.columns]

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

        t_total = len(self.filtered_states)
        time_grid = np.arange(1, t_total + 1)
        reg_arr = np.asarray(self.regimes)

        for i, var in enumerate(variables):
            a = axes_flat[i]
            c = colors[i] if colors[i] is not None else "black"
            ls = styles[i] if styles[i] is not None else "-"
            a.plot(time_grid, self.filtered_states[var], label=var, color=c, linestyle=ls, linewidth=1.5)

            # Highlight binding regime periods
            binding_mask = reg_arr > 0
            if np.any(binding_mask):
                diff = np.diff(np.pad(binding_mask.astype(int), (1, 1), "constant"))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0]
                for s, e in zip(starts, ends):
                    a.axvspan(s - 0.5, e + 0.5, color="0.85", alpha=0.3, label="Constrained" if i == 0 else None)

            if self.constraints:
                for c_name, c_obj in self.constraints.items():
                    if var == c_obj.variable:
                        a.axhline(
                            c_obj.threshold,
                            color="0.4",
                            linestyle="--",
                            linewidth=1.0,
                            label=f"Threshold ({c_obj.threshold:.3g})",
                        )

            a.axhline(0, color="0.6", linestyle=":", linewidth=0.6)
            a.set_title(var)
            a.set_xlabel("Period")
            a.legend(loc="best", frameon=False, fontsize=8)

        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        return fig

    def to_markdown(self, **kwargs) -> str:
        """Export filtered states to Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export filtered states to LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export filtered states to Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)



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


def _stacklevel_outside_module() -> int:
    """``stacklevel`` that attributes a warning to the first frame outside this module.

    :func:`solve_occbin` forwards mapping/sequence inputs to
    :func:`solve_multiconstraint_occbin`; with a fixed ``stacklevel=2`` the
    non-convergence warning raised there would point at the dispatch line
    inside this file rather than at the user's call. Walking past every frame
    whose globals belong to this module gives the right level for a direct
    call (2) and for the forwarded one (3) alike.
    """
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        level += 1
        frame = frame.f_back
    return level


# ---------------------------------------------------------------------------
# Switching-equation resolution shared by the single- and multi-constraint solvers
# ---------------------------------------------------------------------------


_ROW_DIFF_TOL = 1e-8


def _differing_rows(ref_mats: tuple, cons_mats: tuple, tol: float = _ROW_DIFF_TOL) -> np.ndarray:
    """Rows of the canonical system in which the constrained regime differs from the reference.

    Both arguments are ``(A_+, A_0, A_-, B_u, c, ...)`` tuples; only the first
    five entries are read. A row counts as differing when any of its five
    blocks moves by more than ``tol`` (row norm, or absolute value for ``c``).
    """
    A_p_0, A_0_0, A_m_0, B_u_0, c_0 = ref_mats[:5]
    A_p_1, A_0_1, A_m_1, B_u_1, c_1 = cons_mats[:5]
    return np.where(
        (np.linalg.norm(A_0_0 - A_0_1, axis=1) > tol)
        | (np.linalg.norm(A_p_0 - A_p_1, axis=1) > tol)
        | (np.linalg.norm(A_m_0 - A_m_1, axis=1) > tol)
        | (np.linalg.norm(B_u_0 - B_u_1, axis=1) > tol)
        | (np.abs(np.asarray(c_0, dtype=float) - np.asarray(c_1, dtype=float)) > tol)
    )[0]


def _pegged_rows(cons_mats: tuple, rows: Sequence[int], idx_var: int) -> list[int]:
    """Rows among ``rows`` in which the constrained regime pegs variable ``idx_var`` to a constant.

    A peg is a row whose contemporaneous block has a single non-negligible
    entry, on ``idx_var``, and whose lead, lag and shock blocks are all
    negligible, i.e. the equation reads ``a * x_t + c = 0``.
    """
    A_p_1, A_0_1, A_m_1, B_u_1 = cons_mats[:4]
    pegged: list[int] = []
    for r in rows:
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
    return pegged


def _resolve_switching_row(
    ref_mats: tuple,
    cons_mats: tuple,
    constraint: OccBinConstraint,
    variables: Sequence[str],
    caller: str,
    label: str = "",
    relax_hint: str = "",
) -> tuple[int | None, np.ndarray]:
    """Identify the reference-regime row the shadow value of ``constraint`` is solved out of.

    Returns ``(eq_row, diff_rows)``. ``eq_row`` is the index of the reference
    equation that determines ``constraint.variable`` and that the constrained
    regime replaces (a ZLB-style peg), or ``None`` when no differing row
    contains the variable in the reference regime (a trigger-style constraint,
    whose simulated value is its own notional value). ``diff_rows`` is the
    array of all rows in which the two regimes differ.

    The rules are the ones :func:`solve_occbin` documents, and both solvers
    call this helper so that ``solve_occbin(ref, m, c)`` and
    ``solve_occbin(ref, [m], [c])`` resolve a constraint identically:

    * ``constraint.variable`` and ``constraint.relax_variable`` must name model
      variables;
    * among the differing rows that contain the variable in the reference,
      the one the constrained regime turns into a peg of that variable is
      preferred (that is the equation the peg replaces); otherwise the first
      such row is used, so a constrained model that rewrites several rows
      still gets its notional value from the right equation;
    * when no differing row contains the variable, the variable must still
      appear somewhere in the reference system, and if the constrained regime
      pegs it to a constant in a row that does not determine it in the
      reference, ``relax_variable`` is required (the default relax test would
      compare the pegged value against its own bound).

    ``label`` is inserted after "the alternative regime" in error messages
    (e.g. ``" for constrained model 'zlb'"``) and ``relax_hint`` is appended
    to the pegged-row message, so the multi-constraint solver can name the
    offending model.
    """
    variables = list(variables)
    if constraint.variable not in variables:
        raise ValueError(
            f"{caller}: constraint variable {constraint.variable!r} not found in model variables: {variables}"
        )
    idx_var = variables.index(constraint.variable)

    # Resolve the relax (shadow / multiplier) variable up front: a name that is
    # not a model variable used to be dropped silently, which turns an explicit
    # relax rule into the default one without telling anybody.
    if constraint.relax_variable is not None and constraint.relax_variable not in variables:
        raise ValueError(
            f"{caller}: relax_variable {constraint.relax_variable!r} of the constraint on "
            f"{constraint.variable!r}{label} is not a model variable; expected one of {variables}"
        )

    A_0_0 = ref_mats[1]
    diff_rows = _differing_rows(ref_mats, cons_mats)
    has_var = np.abs(A_0_0[:, idx_var]) > 1e-12
    switching_rows = [int(r) for r in diff_rows if has_var[r]]

    if switching_rows:
        # The constrained regime replaces the equation that determines the
        # constrained variable (a ZLB-style peg): its notional value has to be
        # recovered from the reference-regime equation. When several
        # differing rows contain the variable, the one the alternative regime
        # pegs it in is the equation the peg replaced; scanning only the
        # first such row would make the answer depend on the order in which
        # the model's equations happen to be written.
        pegged_switching = _pegged_rows(cons_mats, switching_rows, idx_var)
        return (pegged_switching[0] if pegged_switching else switching_rows[0]), diff_rows

    if not np.any(has_var):
        raise ValueError(
            f"{caller}: the constrained variable {constraint.variable!r} does not appear "
            f"contemporaneously in any equation of the reference model (every entry of column "
            f"{idx_var} of its contemporaneous Jacobian is negligible), so neither its notional "
            f"value nor the binding test is defined. Check the reference model, or constrain a "
            f"variable the model actually determines."
        )

    # No switching equation pins the constrained variable, i.e. the
    # constrained regime leaves that variable endogenously determined by the
    # same equation (a trigger-style constraint such as "public credit policy
    # kicks in once the spread exceeds x"). Its simulated value is then already
    # its notional value, and no shadow needs solving out.
    #
    # A trap that the rule above cannot see. If the alternative regime pegs
    # the constrained variable to a constant in a row that does NOT determine
    # it in the reference regime, then there is nothing to solve the notional
    # value out of and the default relax test compares the simulated value --
    # pinned AT the bound by that very peg -- against the bound. That test is
    # vacuous: it answers "the constraint has just relaxed" in every binding
    # period, whatever the economics, so the spell can never be longer than
    # the iteration's own transient. Such a constraint needs an explicit
    # `relax_variable` (the multiplier, or the instrument that enforces the
    # peg, tested against zero); refuse to guess.
    if constraint.relax_variable is None:
        pegged = _pegged_rows(cons_mats, [int(r) for r in diff_rows], idx_var)
        if pegged:
            raise ValueError(
                f"{caller}: the alternative regime{label} pegs {constraint.variable!r} to a "
                f"constant in equation row(s) {pegged}, but that row does not determine "
                f"{constraint.variable!r} in the reference model, so there is no reference "
                f"equation to solve its notional value out of. The relax test would then "
                f"compare the pegged value against the very bound it is pegged to and "
                f"relax in every period. Set OccBinConstraint.relax_variable to the "
                f"multiplier or the instrument that enforces the peg (with "
                f"relax_threshold=0.0 and the sign that means 'the constraint would have "
                f"to push the wrong way'), so the exit condition is testable.{relax_hint}"
            )
    return None, diff_rows


# ---------------------------------------------------------------------------
# OccBin Solver
# ---------------------------------------------------------------------------


def solve_occbin(
    reference_model: Any,
    constrained_model: Any,
    constraint: OccBinConstraint | Mapping[str, OccBinConstraint] | Sequence[OccBinConstraint],
    shock_sequence: np.ndarray,
    max_iter: int = 50,
    horizon: int = 40,
) -> OccBinResult:
    """Solve dynamic models with occasionally binding constraints (Guerrieri & Iacoviello 2015).

    Finds the piecewise-linear perfect-foresight transition path between the
    constrained and unconstrained regimes using backward recursion.
    Supports polymorphic dispatch to multi-constraint solver when multiple
    constraints or models are provided.

    Parameters
    ----------
    reference_model : LinearModel
        The unconstrained baseline model (e.g., standard Taylor rule regime).
    constrained_model : LinearModel | Callable | Mapping | Sequence
        The model under the binding constraint (e.g., nominal interest rate held at floor),
        or a mapping/sequence of models for multi-constraint regimes.
    constraint : OccBinConstraint | Mapping[str, OccBinConstraint] | Sequence[OccBinConstraint]
        Constraint definition specifying variable, threshold, and direction,
        or a mapping/sequence of constraints.
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
        constrained variable (or ``relax_variable``) is not a model variable,
        if the constrained variable appears in no equation of the reference
        model (so neither its notional value nor the binding test is defined),
        or if the alternative regime pegs it in a row that does not determine
        it in the reference model and no ``relax_variable`` is given. When a
        mapping or sequence is passed, :func:`solve_multiconstraint_occbin`
        additionally requires one constraint per constrained model.
    """
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"solve_occbin: horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"solve_occbin: max_iter must be an integer >= 1, got {max_iter!r}")
    horizon = int(horizon)
    max_iter = int(max_iter)

    # Polymorphic routing: dispatch to multi-constraint solver if multiple constraints or multiple models are passed
    if (
        (isinstance(constraint, (Mapping, Sequence)) and not isinstance(constraint, OccBinConstraint))
        or isinstance(constrained_model, Mapping)
        or (isinstance(constrained_model, (list, tuple)) and not callable(constrained_model))
    ):
        return solve_multiconstraint_occbin(
            m_unconstrained=reference_model,
            m_constrained_dict=constrained_model,
            shock_seq=shock_sequence,
            constraints=constraint,
            horizon=horizon,
            max_iter=max_iter,
        )

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

    # Identify the equation row that determines the constrained variable in the
    # reference regime (the switching equation), or None for a trigger-style
    # constraint; validate the constraint's variable names on the way. The
    # rules live in _resolve_switching_row so that the multi-constraint solver
    # resolves a constraint exactly the same way.
    eq_row, _ = _resolve_switching_row(
        (A_p_0, A_0_0, A_m_0, B_u_0, c_0),
        (A_p_1, A_0_1, A_m_1, B_u_1, c_1),
        constraint,
        variables,
        caller="solve_occbin",
    )
    idx_var = variables.index(constraint.variable)
    relax_idx = variables.index(constraint.relax_variable) if constraint.relax_variable is not None else None

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


# ---------------------------------------------------------------------------
# Differentiable OccBin Solver (Smooth Relaxation)
# ---------------------------------------------------------------------------


def solve_differentiable_occbin(
    reference_model: Any,
    constrained_model: Any,
    constraint: OccBinConstraint,
    shock_sequence: np.ndarray,
    tau: float = 0.01,
    horizon: int = 40,
    max_iter: int = 50,
    tol: float = 1e-6,
    damping: float = 0.5,
    compute_gradient: bool = False,
    param_names: Sequence[str] | None = None,
    target_data: np.ndarray | pd.DataFrame | None = None,
    observed_vars: Sequence[str] | None = None,
    init_weights: np.ndarray | None = None,
) -> DifferentiableOccBinResult:
    """Solve dynamic models with occasionally binding constraints via smooth relaxation.

    Replaces discrete piecewise regime switching with continuous regime weights
    w_t in (0, 1) smoothly interpolating system matrices:
        A_{p, t}(w_t) = (1 - w_t) A_{p, 0} + w_t A_{p, 1}
        A_{0, t}(w_t) = (1 - w_t) A_{0, 0} + w_t A_{0, 1}
        A_{m, t}(w_t) = (1 - w_t) A_{m, 0} + w_t A_{m, 1}
        B_{u, t}(w_t) = (1 - w_t) B_{u, 0} + w_t B_{u, 1}
        c_t(w_t) = (1 - w_t) c_0 + w_t c_1

    This produces C^inf smooth, differentiable simulated trajectories and likelihood
    surfaces enabling gradient-guided MCMC (Hamiltonian Monte Carlo / NUTS) without
    divergences caused by non-differentiable threshold kinks.

    Parameters
    ----------
    reference_model : LinearModel | Callable
        The unconstrained baseline model.
    constrained_model : LinearModel | Callable
        The model under the binding constraint.
    constraint : OccBinConstraint
        Constraint definition specifying variable, threshold, and direction.
    shock_sequence : np.ndarray
        Structural shock sequence, shape (horizon, n_shocks) or (n_shocks,).
    tau : float, default 0.01
        Temperature parameter controlling the softness of the relaxation.
        As tau -> 0, the smooth relaxation converges to discrete OccBin.
    horizon : int, default 40
        Simulation horizon.
    max_iter : int, default 50
        Maximum fixed-point iterations for continuous regime weights.
    tol : float, default 1e-6
        Tolerance for convergence of continuous weights ||w_{k+1} - w_k||_inf.
    damping : float, default 0.5
        Damping factor in (0, 1] for fixed-point iteration.
    compute_gradient : bool, default False
        Whether to compute parameter sensitivities dX/dtheta and loss gradient.
    param_names : Sequence[str], optional
        List of parameter names for gradient computation.
    target_data : np.ndarray | pd.DataFrame, optional
        Target observations for squared loss or likelihood gradient evaluation.
    observed_vars : Sequence[str], optional
        List of observed variable names matching columns of target_data.
    init_weights : np.ndarray, optional
        Initial guess for weights vector, shape (horizon,).

    Returns
    -------
    DifferentiableOccBinResult
        Container holding simulated path, continuous weights, convergence stats,
        shadow values, loss, and parameter sensitivities / gradients.
    """
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"solve_differentiable_occbin: horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"solve_differentiable_occbin: max_iter must be an integer >= 1, got {max_iter!r}")
    if float(tau) <= 0.0:
        raise ValueError(f"solve_differentiable_occbin: tau must be positive, got {tau!r}")
    horizon = int(horizon)
    max_iter = int(max_iter)
    tau = float(tau)

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

    # 2. Reference-regime decision rule X_t = P_0 X_{t-1}
    dr = reference_model.decision_rules() if hasattr(reference_model, "decision_rules") else None
    P_0 = np.zeros((n_vars, n_vars))
    if dr is not None and hasattr(reference_model, "states"):
        for s in reference_model.states:
            idx_s = variables.index(s)
            P_0[:, idx_s] = dr.ghx[s].values
    else:
        from puremacro.dsge.klein import klein_solve
        try:
            sol = klein_solve(A_0_0, -A_m_0, n_pre=n_vars)
            P_0 = sol.P
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError, Exception):
            P_0 = np.zeros((n_vars, n_vars))

    # Find the row index of the constrained variable
    if constraint.variable not in variables:
        raise ValueError(f"constraint variable {constraint.variable!r} not found in model variables: {variables}")
    idx_var = variables.index(constraint.variable)

    relax_idx = None
    if constraint.relax_variable is not None:
        if constraint.relax_variable not in variables:
            raise ValueError(
                f"solve_differentiable_occbin: relax_variable {constraint.relax_variable!r} is not a model "
                f"variable; expected one of {variables}"
            )
        relax_idx = variables.index(constraint.relax_variable)

    # Identify switching equation row
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
        eq_row = switching_rows[0]
    elif np.any(has_var):
        eq_row = None
    else:
        raise ValueError(
            f"solve_differentiable_occbin: constrained variable {constraint.variable!r} does not appear "
            f"contemporaneously in any equation of the reference model."
        )

    if eq_row is not None:
        a_var = float(A_0_0[eq_row, idx_var])
        row_others = A_0_0[eq_row].copy()
        row_others[idx_var] = 0.0

    # 3. Forward-backward simulation under given continuous weights w in (0, 1)^H
    def simulate_with_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_cl = np.clip(weights, 0.0, 1.0)
        P_seq: list[np.ndarray | None] = [None] * (horizon + 1)
        D_seq: list[np.ndarray | None] = [None] * (horizon + 1)
        P_next = P_0
        D_next = np.zeros(n_vars)
        for t in range(horizon, 0, -1):
            w_t = float(w_cl[t - 1])
            A_p_t = (1.0 - w_t) * A_p_0 + w_t * A_p_1
            A_0_t = (1.0 - w_t) * A_0_0 + w_t * A_0_1
            A_m_t = (1.0 - w_t) * A_m_0 + w_t * A_m_1
            B_u_t = (1.0 - w_t) * B_u_0 + w_t * B_u_1
            c_t = (1.0 - w_t) * c_0 + w_t * c_1

            M_t = A_0_t + A_p_t @ P_next
            P_t = _safe_solve(M_t, -A_m_t)
            D_t = _safe_solve(M_t, -(A_p_t @ D_next + c_t + B_u_t @ shocks_mat[t - 1]))
            P_seq[t] = P_t
            D_seq[t] = D_t
            P_next, D_next = P_t, D_t

        X = np.zeros((horizon + 1, n_vars))
        for t in range(1, horizon + 1):
            X[t] = P_seq[t] @ X[t - 1] + D_seq[t]

        sim_X = X[1:]

        if eq_row is None:
            shadow_vals = sim_X[:, idx_var].copy()
        else:
            shadow_vals = np.zeros(horizon)
            for t in range(horizon):
                x_prev = X[t]
                x_curr = sim_X[t]
                x_next = sim_X[t + 1] if t + 1 < horizon else P_0 @ x_curr
                u_t = shocks_mat[t]
                other_curr = row_others @ x_curr
                term_next = A_p_0[eq_row, :] @ x_next
                term_prev = A_m_0[eq_row, :] @ x_prev
                term_shock = B_u_0[eq_row, :] @ u_t
                shadow_vals[t] = -(other_curr + term_next + term_prev + term_shock + c_0[eq_row]) / a_var

        return sim_X, shadow_vals

    def compute_target_weights(sim_X: np.ndarray, shadow_vals: np.ndarray) -> np.ndarray:
        if relax_idx is not None:
            v = sim_X[:, relax_idx]
        elif eq_row is not None:
            v = shadow_vals
        else:
            v = sim_X[:, idx_var]

        thresh = float(constraint.threshold)
        if constraint.operator in ("<", "<="):
            gap = thresh - v
        else:
            gap = v - thresh

        # Smooth relaxation via derivative of smax_tau(gap, 0):
        root = np.sqrt(gap**2 + tau**2)
        target_w = 0.5 * (1.0 + gap / root)
        return np.clip(target_w, 0.0, 1.0)

    # 4. Fixed-point iteration
    if init_weights is not None:
        weights = np.asarray(init_weights, dtype=float).copy()
    else:
        weights = np.zeros(horizon, dtype=float)

    converged = False
    sim_X = np.zeros((horizon, n_vars))
    shadow_vals = np.zeros(horizon)

    for it in range(1, max_iter + 1):
        sim_X, shadow_vals = simulate_with_weights(weights)
        target_w = compute_target_weights(sim_X, shadow_vals)
        diff = float(np.max(np.abs(target_w - weights)))
        weights = (1.0 - damping) * weights + damping * target_w
        if diff < tol:
            converged = True
            break
    else:
        sim_X, shadow_vals = simulate_with_weights(weights)

    # 5. Build result frames
    period_index = pd.RangeIndex(1, horizon + 1, name="t")
    sim_df = pd.DataFrame(sim_X, columns=variables, index=period_index)
    shadow_df = sim_df.copy()
    shadow_df[f"{constraint.variable}_shadow"] = shadow_vals

    regimes = [int(w >= 0.5) for w in weights]
    binding_periods = int(np.sum(np.asarray(weights) >= 0.5))

    # Loss computation if target data provided
    loss_val = None
    t_vars = list(observed_vars) if observed_vars is not None else variables
    if target_data is not None:
        if isinstance(target_data, pd.DataFrame):
            t_vars = [c for c in target_data.columns if c in variables]
            y_target = target_data[t_vars].to_numpy(dtype=float)
            y_sim = sim_df[t_vars].to_numpy(dtype=float)
        else:
            y_target = np.asarray(target_data, dtype=float)
            y_sim = sim_X[:, : y_target.shape[1]]
        loss_val = float(0.5 * np.sum((y_sim - y_target) ** 2))

    # 6. Sensitivity / Gradient computation if requested
    gradient_dict: dict[str, float] | None = None
    sens_df: pd.DataFrame | None = None

    if compute_gradient or param_names is not None:
        gradient_dict = {}
        sensitivities = {}
        p_names = list(param_names) if param_names is not None else list(getattr(reference_model, "_params", {}).keys())

        base_p = dict(getattr(reference_model, "_params", {}) or {})
        from puremacro.dsge.dynare import build_dynare

        for p_name in p_names:
            if p_name not in base_p:
                continue
            val = float(base_p[p_name])
            h = 1e-5 * max(1.0, abs(val))

            p_plus = dict(base_p)
            p_plus[p_name] = val + h
            p_minus = dict(base_p)
            p_minus[p_name] = val - h

            try:
                if getattr(reference_model, "_dynare_equations", None) is not None:
                    ref_plus = build_dynare(
                        reference_model._dynare_equations,
                        variables=reference_model.variables,
                        shocks=reference_model.shocks,
                        params=p_plus,
                        steady_state=reference_model.steady_state,
                        check_steady_state=False,
                        strict=False,
                    )
                    ref_minus = build_dynare(
                        reference_model._dynare_equations,
                        variables=reference_model.variables,
                        shocks=reference_model.shocks,
                        params=p_minus,
                        steady_state=reference_model.steady_state,
                        check_steady_state=False,
                        strict=False,
                    )
                else:
                    ref_plus, ref_minus = reference_model, reference_model

                if getattr(constrained_model, "_dynare_equations", None) is not None:
                    cons_base = dict(getattr(constrained_model, "_params", {}) or base_p)
                    cp_plus = dict(cons_base)
                    cp_plus[p_name] = val + h
                    cp_minus = dict(cons_base)
                    cp_minus[p_name] = val - h
                    cons_plus = build_dynare(
                        constrained_model._dynare_equations,
                        variables=constrained_model.variables,
                        shocks=constrained_model.shocks,
                        params=cp_plus,
                        steady_state=constrained_model.steady_state,
                        check_steady_state=False,
                        strict=False,
                    )
                    cons_minus = build_dynare(
                        constrained_model._dynare_equations,
                        variables=constrained_model.variables,
                        shocks=constrained_model.shocks,
                        params=cp_minus,
                        steady_state=constrained_model.steady_state,
                        check_steady_state=False,
                        strict=False,
                    )
                else:
                    cons_plus, cons_minus = constrained_model, constrained_model

                res_p = solve_differentiable_occbin(
                    ref_plus, cons_plus, constraint, shock_sequence,
                    tau=tau, horizon=horizon, max_iter=20, tol=tol,
                    init_weights=weights,
                )
                res_m = solve_differentiable_occbin(
                    ref_minus, cons_minus, constraint, shock_sequence,
                    tau=tau, horizon=horizon, max_iter=20, tol=tol,
                    init_weights=weights,
                )

                dX_dp = (res_p.simulated_path.to_numpy() - res_m.simulated_path.to_numpy()) / (2.0 * h)
                sensitivities[p_name] = dX_dp

                if target_data is not None:
                    lp_p = 0.5 * np.sum((res_p.simulated_path[t_vars].to_numpy() - y_target) ** 2)
                    lp_m = 0.5 * np.sum((res_m.simulated_path[t_vars].to_numpy() - y_target) ** 2)
                    gradient_dict[p_name] = float((lp_p - lp_m) / (2.0 * h))
                else:
                    gradient_dict[p_name] = float(dX_dp[0, idx_var])
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError, Exception):
                gradient_dict[p_name] = 0.0

        if sensitivities:
            records = []
            for p_name, mat in sensitivities.items():
                for v_idx, var in enumerate(variables):
                    records.append({
                        "parameter": p_name,
                        "variable": var,
                        "impact_sensitivity": float(mat[0, v_idx]),
                        "mean_sensitivity": float(np.mean(mat[:, v_idx])),
                        "max_sensitivity": float(np.max(np.abs(mat[:, v_idx]))),
                    })
            sens_df = pd.DataFrame(records)

    return DifferentiableOccBinResult(
        simulated_path=sim_df,
        weights=weights,
        regimes=regimes,
        binding_periods=binding_periods,
        converged=converged,
        iterations=it,
        reference_model=reference_model,
        constrained_model=constrained_model,
        constraint=constraint,
        tau=tau,
        shadow_path=shadow_df,
        loss=loss_val,
        gradient=gradient_dict,
        param_sensitivities=sens_df,
    )


# ---------------------------------------------------------------------------
# Multi-Constraint OccBin & Piecewise Kalman Filter
# ---------------------------------------------------------------------------


def _resolve_user_constraints(
    model_items: Sequence[tuple[Any, Any]],
    constraints: OccBinConstraint | Mapping[Any, OccBinConstraint] | Sequence[OccBinConstraint] | None,
    caller: str,
) -> dict[Any, OccBinConstraint]:
    """Pair user-supplied constraint definitions with the constrained models, key by key.

    Returns ``{model_key: OccBinConstraint}``, empty when ``constraints`` is
    ``None`` (every constraint is then auto-detected from its model). Any other
    input must pair one-to-one with the models: a mapping must carry exactly
    the model keys, a sequence exactly one entry per model in the same order,
    and a bare :class:`OccBinConstraint` is accepted only with a single model.
    A mismatch raises ``ValueError``: silently handing the unmatched models to
    the auto-detector used to replace the user's threshold by a guessed one
    without any notice.
    """
    keys = [k for k, _ in model_items]
    if constraints is None:
        return {}
    if isinstance(constraints, OccBinConstraint):
        if len(keys) != 1:
            raise ValueError(
                f"{caller}: a single OccBinConstraint was given for {len(keys)} constrained "
                f"models {keys}; pass one constraint per model, as a mapping with the same keys "
                f"or a sequence in the same order."
            )
        resolved = {keys[0]: constraints}
    elif isinstance(constraints, Mapping):
        missing = [k for k in keys if k not in constraints]
        unmatched = [k for k in constraints if k not in keys]
        if missing or unmatched:
            raise ValueError(
                f"{caller}: the keys of `constraints` {list(constraints)} do not match the "
                f"constrained-model keys {keys} (models without a constraint: {missing}; "
                f"constraints without a model: {unmatched}). Pass one OccBinConstraint per model "
                f"under the model's key, or constraints=None to auto-detect all of them."
            )
        resolved = {k: constraints[k] for k in keys}
    elif isinstance(constraints, (list, tuple)):
        if len(constraints) != len(keys):
            raise ValueError(
                f"{caller}: {len(constraints)} constraint(s) were given for {len(keys)} constrained "
                f"model(s) {keys}; pass exactly one OccBinConstraint per model, in the same order."
            )
        resolved = dict(zip(keys, constraints))
    else:
        raise TypeError(
            f"{caller}: `constraints` must be an OccBinConstraint, a mapping or sequence of "
            f"them, or None; got {type(constraints).__name__}"
        )
    for key, c_obj in resolved.items():
        if not isinstance(c_obj, OccBinConstraint):
            raise TypeError(
                f"{caller}: the constraint for model {key!r} must be an OccBinConstraint, "
                f"got {type(c_obj).__name__}"
            )
    return resolved


def _auto_detect_constraint(
    ref_model: Any,
    cons_model: Any,
    default_name: Any = "constraint",
    user_constraint: OccBinConstraint | None = None,
    caller: str = "solve_multiconstraint_occbin",
) -> tuple[OccBinConstraint, int | None, tuple]:
    """Resolve the constraint definition and switching row for one constrained model.

    Returns ``(constraint, eq_row, cons_mats)``: the constraint (the user's,
    or one deduced from the model when ``user_constraint`` is None), the
    reference-regime row its shadow value is solved out of (``None`` for a
    trigger-style constraint, whose simulated value is its own notional
    value) and the constrained model's ``(A_+, A_0, A_-, B_u, c)`` matrices.
    Every row in which ``cons_model`` differs from ``ref_model`` is spliced
    into the binding regimes by :func:`_build_multi_regime_matrices`;
    ``eq_row`` only serves the shadow-value computation. The switching row
    and the validation follow :func:`_resolve_switching_row`, i.e. the rules
    of :func:`solve_occbin`, including its refusal of a pegged variable whose
    relax test would be vacuous.

    Auto-detection reads the constraint off the constrained model: the first
    differing row that pegs a variable to a constant (or, failing that, the
    first differing row) names the variable, ``-c / a`` is the threshold, and
    the operator follows the model key ('zlb'/'floor' -> '<', 'borrow'/'cap'/
    'collateral' -> '>') or else the sign of the threshold. It is a heuristic
    for the common peg layouts; pass explicit constraints for anything else.
    """
    ref_mats = _extract_model_matrices(ref_model)
    cons_mats = tuple(_extract_model_matrices(cons_model, ref_model=ref_model)[:5])
    A_0_0 = ref_mats[1]
    variables = list(ref_mats[6])
    A_0_1, c_1 = cons_mats[1], cons_mats[4]

    diff_rows = _differing_rows(ref_mats, cons_mats)
    if len(diff_rows) == 0:
        raise ValueError(
            f"No differing equations found between reference model and constrained model '{default_name}'."
        )
    label = f" for constrained model '{default_name}'"

    if user_constraint is not None:
        eq_row, _ = _resolve_switching_row(
            ref_mats, cons_mats, user_constraint, variables, caller=caller, label=label
        )
        return user_constraint, eq_row, cons_mats

    # Auto-detect: prefer the differing row in which the constrained regime
    # pegs a variable to a constant. A constrained model that also rewrites
    # other equations (an IS curve that changes at the ZLB, say) must not have
    # its constraint read off whichever of those rows happens to come first.
    eq_row = int(diff_rows[0])
    for r in diff_rows:
        row_abs = np.abs(A_0_1[r, :])
        if np.max(row_abs) > 1e-12 and _pegged_rows(cons_mats, [int(r)], int(np.argmax(row_abs))):
            eq_row = int(r)
            break
    row_abs = np.abs(A_0_1[eq_row, :])
    if np.max(row_abs) > 1e-12:
        idx_var = int(np.argmax(row_abs))
        var_name = variables[idx_var]
        coeff = A_0_1[eq_row, idx_var]
        const_val = c_1[eq_row]
        thresh = -float(const_val) / float(coeff) if abs(coeff) > 1e-12 else 0.0
    else:
        row_abs_ref = np.abs(A_0_0[eq_row, :])
        idx_var = int(np.argmax(row_abs_ref))
        var_name = variables[idx_var]
        thresh = 0.0

    name_lower = str(default_name).lower()
    if "zlb" in name_lower or "floor" in name_lower or thresh < 0:
        op = "<"
    elif "borrow" in name_lower or "cap" in name_lower or "collateral" in name_lower or thresh > 0:
        op = ">"
    else:
        op = "<"

    constraint = OccBinConstraint(variable=var_name, threshold=thresh, operator=op)
    eq_row, _ = _resolve_switching_row(
        ref_mats,
        cons_mats,
        constraint,
        variables,
        caller=caller,
        label=label,
        relax_hint=(
            f" Auto-detection cannot name that variable: pass an explicit OccBinConstraint "
            f"for '{default_name}' instead of constraints=None."
        ),
    )
    return constraint, eq_row, cons_mats


def _build_multi_regime_matrices(
    ref_matrices: tuple,
    constraint_info: Sequence[tuple[OccBinConstraint, int | None, tuple]],
) -> dict[int, tuple]:
    """Construct ``(A_+, A_0, A_-, B_u, c)`` for all ``2**K`` regimes.

    Regime ``r`` is a bitmask over the ``K`` constraints (bit ``k`` set means
    constraint ``k`` binds). Its system is the reference system with EVERY
    row in which constrained model ``k`` differs from the reference replaced
    by that model's row, for each set bit ``k``. Splicing the switching row
    alone is not enough: a constrained model may rewrite several equations,
    and dropping the others silently changes the economics (or, when the peg
    is not the first differing row, drops the peg itself). Two constrained
    models that rewrite the same row are incompatible and raise ``ValueError``.
    """
    A_p_0, A_0_0, A_m_0, B_u_0, c_0 = ref_matrices[:5]
    K = len(constraint_info)
    n_regimes = 2 ** K

    row_sets: list[list[int]] = []
    for _, eq_row, cons_mats in constraint_info:
        rows = {int(r) for r in _differing_rows(ref_matrices, cons_mats)}
        if eq_row is not None:
            rows.add(int(eq_row))
        row_sets.append(sorted(rows))
    for i in range(K):
        for j in range(i + 1, K):
            overlap = sorted(set(row_sets[i]) & set(row_sets[j]))
            if overlap:
                raise ValueError(
                    f"Incompatible constraint regimes: multiple constraints modify the same "
                    f"equation row(s) {overlap} (constraints {i} and {j})."
                )

    regime_matrices = {}
    for r in range(n_regimes):
        Ap_r = A_p_0.copy()
        A0_r = A_0_0.copy()
        Am_r = A_m_0.copy()
        Bu_r = B_u_0.copy()
        c_r = np.array(c_0, dtype=float, copy=True)

        for k in range(K):
            if (r >> k) & 1:
                Ap_k, A0_k, Am_k, Bu_k, c_k = constraint_info[k][2][:5]
                rows = row_sets[k]
                Ap_r[rows, :] = Ap_k[rows, :]
                A0_r[rows, :] = A0_k[rows, :]
                Am_r[rows, :] = Am_k[rows, :]
                Bu_r[rows, :] = Bu_k[rows, :]
                c_r[rows] = np.asarray(c_k, dtype=float)[rows]

        regime_matrices[r] = (Ap_r, A0_r, Am_r, Bu_r, c_r)

    return regime_matrices


def solve_multiconstraint_occbin(
    m_unconstrained: Any,
    m_constrained_dict: Mapping[str, Any] | Sequence[Any],
    shock_seq: np.ndarray,
    constraints: Mapping[str, OccBinConstraint] | Sequence[OccBinConstraint] | None = None,
    horizon: int = 40,
    max_iter: int = 50,
) -> OccBinResult:
    """Solve a model with K occasionally binding constraints across its ``2**K`` regimes.

    Parameters
    ----------
    m_unconstrained : LinearModel
        The unconstrained reference model.
    m_constrained_dict : Mapping[str, LinearModel | Callable] or Sequence
        One constrained model per constraint: the model in which that
        constraint (and only that one) binds. Every row in which a
        constrained model differs from the reference is spliced into the
        regimes where its constraint binds, so two constrained models must
        not rewrite the same equation. A sequence is keyed ``constraint_0,
        constraint_1, ...``.
    shock_seq : np.ndarray
        Anticipated structural shock sequence of shape (n_shocks,) or (horizon, n_shocks).
    constraints : Mapping[str, OccBinConstraint] or Sequence, optional
        One :class:`OccBinConstraint` per constrained model: a mapping with
        exactly the model keys, a sequence in the model order, or a single
        constraint when there is a single model. If None, every constraint is
        deduced from its model (the pegged variable, ``-c / a`` as threshold,
        an operator read off the key or the threshold's sign).
    horizon : int, default 40
        Simulation horizon.
    max_iter : int, default 50
        Maximum fixed-point regime iterations.

    Returns
    -------
    OccBinResult
        ``result.regimes[t]`` is a bitmask: bit ``k`` is set when constraint
        ``k`` (the ``k``-th model) binds in period ``t``, so with two
        constraints ``3`` means both bind and ``2`` only the second.
        ``result.constraints`` maps each model key to the constraint actually
        used and ``result.shadow_path`` adds one ``<variable>_shadow`` column
        per constraint.

    Notes
    -----
    ``result.converged`` follows the :func:`solve_occbin` contract: it is
    ``True`` only when the regime iteration reached a fixed point, the
    returned path respects every constraint where that is meaningful for its
    style (a peg in every period, a trigger in every period declared slack
    for it) and every constraint is slack in the final period, so that the
    terminal condition is verified. Otherwise the function returns
    ``converged=False`` and emits one ``UserWarning`` naming every reason
    (``max_iter`` exhausted, a regime cycle that damping could not break, a
    violated bound, a constraint still binding at ``T``); the path is then
    the one solved under ``result.regimes``, a diagnostic rather than a
    solution.

    Raises
    ------
    ValueError
        If ``horizon``/``max_iter`` are not positive integers, no constrained
        model is given, ``constraints`` cannot be paired one-to-one with the
        models (mismatched keys, wrong length, one constraint for several
        models), a constraint names an unknown variable or ``relax_variable``,
        a constrained model does not differ from the reference, two
        constrained models rewrite the same equation row, or the alternative
        regime pegs a variable in a row that does not determine it in the
        reference model without a ``relax_variable`` (the relax test would be
        vacuous, exactly as in :func:`solve_occbin`).
    """
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"solve_multiconstraint_occbin: horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"solve_multiconstraint_occbin: max_iter must be an integer >= 1, got {max_iter!r}")
    horizon = int(horizon)
    max_iter = int(max_iter)

    ref_matrices = _extract_model_matrices(m_unconstrained)
    A_p_0, A_0_0, A_m_0, B_u_0, c_0, ss_0, variables, shocks = ref_matrices
    n_vars = len(variables)
    n_shocks = len(shocks)

    # Standardize m_constrained_dict
    if isinstance(m_constrained_dict, Mapping):
        model_items = list(m_constrained_dict.items())
    elif isinstance(m_constrained_dict, (list, tuple)):
        model_items = [(f"constraint_{i}", m) for i, m in enumerate(m_constrained_dict)]
    else:
        model_items = [("constraint_0", m_constrained_dict)]

    K = len(model_items)
    if K == 0:
        raise ValueError("solve_multiconstraint_occbin requires at least one constrained model.")

    # Pair the user's constraints with the models, key by key; anything that
    # does not pair one-to-one is an error, never a silent auto-detection.
    user_constraints_dict = _resolve_user_constraints(
        model_items, constraints, "solve_multiconstraint_occbin"
    )

    # Build constraint info for each constraint
    constraint_info = []
    final_constraints: dict[str, OccBinConstraint] = {}
    for name, c_model in model_items:
        c_obj, eq_row, cons_mats = _auto_detect_constraint(
            m_unconstrained,
            c_model,
            default_name=name,
            user_constraint=user_constraints_dict.get(name),
            caller="solve_multiconstraint_occbin",
        )
        constraint_info.append((c_obj, eq_row, cons_mats))
        final_constraints[name] = c_obj

    # Splice every differing row of each constrained model into its regimes;
    # raises when two constrained models rewrite the same equation row.
    regime_matrices = _build_multi_regime_matrices(ref_matrices, constraint_info)

    # Standardize shock sequence
    sh_arr = np.asarray(shock_seq, dtype=float)
    if sh_arr.ndim == 1:
        if len(sh_arr) != n_shocks:
            raise ValueError(f"1D shock sequence has length {len(sh_arr)}, expected {n_shocks}: {shocks}")
        shocks_mat = np.zeros((horizon, n_shocks))
        shocks_mat[0, :] = sh_arr
    elif sh_arr.ndim == 2:
        n_rows, n_cols = sh_arr.shape
        if n_cols != n_shocks:
            raise ValueError(f"shock sequence columns {n_cols} != model shocks {n_shocks}")
        shocks_mat = np.zeros((horizon, n_shocks))
        shocks_mat[: min(n_rows, horizon), :] = sh_arr[: min(n_rows, horizon), :]
    else:
        raise ValueError("shock_seq must be 1D or 2D array")

    # Reference decision rule
    dr = m_unconstrained.decision_rules()
    P_0 = np.zeros((n_vars, n_vars))
    for s in m_unconstrained.states:
        idx_s = variables.index(s)
        P_0[:, idx_s] = dr.ghx[s].values

    # Pre-extract variable indices and relax parameters
    c_indices = []
    for k in range(K):
        c_obj, eq_row, _ = constraint_info[k]
        idx_var = variables.index(c_obj.variable)
        relax_idx = variables.index(c_obj.relax_variable) if c_obj.relax_variable else None
        c_indices.append((c_obj, eq_row, idx_var, relax_idx))

    # Check if shocks are zero: degenerate case
    if np.all(np.abs(shocks_mat) < 1e-14):
        zero_path = np.zeros((horizon, n_vars))
        period_index = pd.RangeIndex(1, horizon + 1, name="t")
        sim_df = pd.DataFrame(zero_path, columns=variables, index=period_index)
        return OccBinResult(
            simulated_path=sim_df,
            regimes=[0] * horizon,
            binding_periods=0,
            converged=True,
            iterations=1,
            reference_model=m_unconstrained,
            constrained_model=model_items[0][1],
            constrained_models=dict(model_items),
            constraint=constraint_info[0][0],
            constraints=final_constraints,
            shadow_path=sim_df.copy(),
        )

    # Backward recursion engine
    def compute_decision_rules(regime_seq: np.ndarray):
        P_seq = [None] * (horizon + 1)
        D_seq = [None] * (horizon + 1)
        P_next = P_0
        D_next = np.zeros(n_vars)
        for t in range(horizon, 0, -1):
            r_t = int(regime_seq[t - 1])
            Ap, A0, Am, Bu, c = regime_matrices[r_t]
            M_t = A0 + Ap @ P_next
            P_t = _safe_solve(M_t, -Am)
            D_t = _safe_solve(M_t, -(Ap @ D_next + c + Bu @ shocks_mat[t - 1]))
            P_seq[t] = P_t
            D_seq[t] = D_t
            P_next, D_next = P_t, D_t
        return P_seq, D_seq

    # Forward simulation and shadow calculation
    def simulate_path(regime_seq: np.ndarray):
        P_seq, D_seq = compute_decision_rules(regime_seq)
        X = np.zeros((horizon + 1, n_vars))
        for t in range(1, horizon + 1):
            X[t] = P_seq[t] @ X[t - 1] + D_seq[t]
        sim_X = X[1:]

        shadow_vals = np.zeros((K, horizon))
        for k in range(K):
            c_obj, eq_row, idx_var, relax_idx = c_indices[k]
            # eq_row is None for a trigger-style constraint: no switching
            # equation pins the variable, so its simulated value is its own
            # notional value (the else branch below).
            a_var = float(A_0_0[eq_row, idx_var]) if eq_row is not None else 0.0
            if abs(a_var) > 1e-12:
                row_others = A_0_0[eq_row].copy()
                row_others[idx_var] = 0.0
                for t in range(horizon):
                    x_prev = X[t]
                    x_curr = sim_X[t]
                    x_next = sim_X[t + 1] if t + 1 < horizon else P_0 @ x_curr
                    u_t = shocks_mat[t]
                    term_other = row_others @ x_curr
                    term_next = A_p_0[eq_row, :] @ x_next
                    term_prev = A_m_0[eq_row, :] @ x_prev
                    term_shock = B_u_0[eq_row, :] @ u_t
                    shadow_vals[k, t] = -(term_other + term_next + term_prev + term_shock + c_0[eq_row]) / a_var
            else:
                shadow_vals[k, :] = sim_X[:, idx_var]

        return sim_X, shadow_vals

    def next_regime(regime_seq: np.ndarray, sim_X: np.ndarray, shadow_vals: np.ndarray):
        upd = np.zeros(horizon, dtype=int)
        for t in range(horizon):
            r_t = int(regime_seq[t])
            r_new = 0
            for k in range(K):
                c_obj, eq_row, idx_var, relax_idx = c_indices[k]
                bit_k = (r_t >> k) & 1
                if bit_k == 1:
                    val = sim_X[t, relax_idx] if relax_idx is not None else shadow_vals[k, t]
                    binds = not bool(c_obj.evaluate_relax(val))
                else:
                    val = sim_X[t, idx_var]
                    binds = bool(c_obj.evaluate(val))
                if binds:
                    r_new |= (1 << k)
            upd[t] = r_new
        return upd

    # Fixed-point iteration with damping
    regime = np.zeros(horizon, dtype=int)
    history: set[tuple[int, ...]] = {tuple(regime.tolist())}
    fixed_point = False
    cycled_to: np.ndarray | None = None
    iteration = 0

    for iteration in range(1, max_iter + 1):
        sim_X, shadow_vals = simulate_path(regime)
        upd = next_regime(regime, sim_X, shadow_vals)

        if np.array_equal(upd, regime):
            fixed_point = True
            break

        key = tuple(upd.tolist())
        if key in history:
            # Oscillatory chattering detected: apply damping (accept only the
            # first period that changed) if that yields an unvisited guess.
            diff_t = np.where(upd != regime)[0]
            if len(diff_t) > 0:
                damped = regime.copy()
                damped[diff_t[0]] = upd[diff_t[0]]
                damped_key = tuple(damped.tolist())
                if damped_key not in history:
                    history.add(damped_key)
                    regime = damped
                    continue
            # No unvisited guess left. The path in hand was solved under
            # `regime`, which the solver's own binding test has just rejected,
            # so this is NOT a solution -- report it instead of picking a winner.
            cycled_to = upd
            break

        history.add(key)
        regime = upd
    else:
        # max_iter exhausted; `regime` holds the last update, re-simulate it so
        # that the returned path and the reported regime sequence agree.
        sim_X, shadow_vals = simulate_path(regime)

    regimes_list = [int(v) for v in regime]
    reasons: list[str] = []

    if cycled_to is not None:
        solved_for = [int(t) + 1 for t in np.flatnonzero(regime)]
        rejected_to = [int(t) + 1 for t in np.flatnonzero(cycled_to)]
        reasons.append(
            f"the regime iteration cycled at iteration {iteration}: the path was solved with "
            f"a constraint binding in period(s) {solved_for[:12]}"
            f"{' ...' if len(solved_for) > 12 else ''}, but the solver's own binding test on "
            f"that very path returns {rejected_to[:12]}"
            f"{' ...' if len(rejected_to) > 12 else ''}, a guess already visited that damping "
            f"could not move away from, so no regime sequence is a fixed point"
        )
    elif not fixed_point:
        reasons.append(
            f"the regime iteration did not reach a fixed point within max_iter={max_iter}"
        )

    # Post-hoc verification of the returned path against every constraint,
    # mirroring solve_occbin: a peg-style constraint (eq_row is not None, its
    # reference equation determines the variable and the alternative regime
    # pins it AT the bound) must hold in every period; a trigger-style one
    # (eq_row is None) is meant to sit beyond its threshold while active, so
    # only the periods declared slack for it are tested.
    regime_arr = np.asarray(regimes_list)
    for k in range(K):
        c_obj, eq_row, idx_var, _ = c_indices[k]
        thresh = float(c_obj.threshold)
        bound_tol = 1e-8 * max(1.0, abs(thresh))
        checked = (
            np.arange(horizon) if eq_row is not None else np.flatnonzero(((regime_arr >> k) & 1) == 0)
        )
        col = sim_X[checked, idx_var]
        if c_obj.operator in ("<", "<="):
            violated = checked[col < thresh - bound_tol]
        else:
            violated = checked[col > thresh + bound_tol]
        if violated.size:
            vals = sim_X[violated, idx_var]
            worst = float(vals.min() if c_obj.operator in ("<", "<=") else vals.max())
            reasons.append(
                f"the returned path violates {c_obj!r} in period(s) "
                f"{[int(t) + 1 for t in violated[:12]]}"
                f"{' ...' if violated.size > 12 else ''} "
                f"(worst {c_obj.variable}={worst:.6g} against threshold {thresh:.6g})"
            )

    if regimes_list[-1] != 0:
        reasons.append(
            f"constraint still binds at terminal period T={horizon} (regime={regimes_list[-1]}), "
            f"so the terminal condition P_{{T+1}} = P_0 -- that the reference regime resumes "
            f"after the horizon -- is assumed rather than verified; increase horizon"
        )

    converged = bool(fixed_point) and not reasons
    if reasons:
        # Attributed to the user's call whether they called this function or
        # reached it through solve_occbin's mapping/sequence dispatch.
        warnings.warn(
            "solve_multiconstraint_occbin did not produce a verified solution (converged=False): "
            + "; ".join(reasons)
            + ". The returned path is the one solved under `result.regimes` and is a "
            "diagnostic, not a solution.",
            UserWarning,
            stacklevel=_stacklevel_outside_module(),
        )

    period_index = pd.RangeIndex(1, horizon + 1, name="t")
    sim_df = pd.DataFrame(sim_X, columns=variables, index=period_index)
    shadow_df = sim_df.copy()
    for k in range(K):
        c_obj = constraint_info[k][0]
        shadow_df[f"{c_obj.variable}_shadow"] = shadow_vals[k]

    binding_periods = int(np.sum(np.asarray(regimes_list) > 0))

    return OccBinResult(
        simulated_path=sim_df,
        regimes=regimes_list,
        binding_periods=binding_periods,
        converged=converged,
        iterations=iteration,
        reference_model=m_unconstrained,
        constrained_model=model_items[0][1],
        constrained_models=dict(model_items),
        constraint=constraint_info[0][0],
        constraints=final_constraints,
        shadow_path=shadow_df,
    )


def piecewise_kalman_filter(
    m_unconstrained: Any,
    m_constrained_dict: Mapping[str, Any] | Sequence[Any],
    data: pd.DataFrame | np.ndarray,
    varobs: Sequence[str] | None = None,
    constraints: Mapping[str, OccBinConstraint] | Sequence[OccBinConstraint] | None = None,
    horizon: int = 30,
    max_regime_iter: int = 20,
    H: np.ndarray | None = None,
    Q: np.ndarray | None = None,
) -> PiecewiseKalmanResult:
    """Piecewise Kalman Filter (Giovannini, Pfeiffer & Ratto 2021).

    Combines OccBin backward recursions with forward Kalman updates to evaluate
    exact Gaussian log-likelihoods in models with occasionally binding constraints.

    Parameters
    ----------
    m_unconstrained : LinearModel
        The unconstrained reference model.
    m_constrained_dict : Mapping[str, LinearModel | Callable] or Sequence
        Dictionary or sequence of constrained regime models.
    data : pd.DataFrame or np.ndarray
        Observed macroeconomic time series.
    varobs : Sequence[str], optional
        List of observable variable names. If None, inferred from DataFrame columns.
    constraints : Mapping[str, OccBinConstraint] or Sequence, optional
        Constraint definitions. If None, auto-detected from model equations.
    horizon : int, default 30
        Regime anticipation horizon H.
    max_regime_iter : int, default 20
        Maximum iterations for regime consistency at each observation date.
    H : np.ndarray, optional
        Measurement error covariance matrix (n_obs, n_obs).
    Q : np.ndarray, optional
        Structural innovation covariance matrix (n_shocks, n_shocks).

    Returns
    -------
    PiecewiseKalmanResult
        Object containing log-likelihood, filtered states, regime history,
        imputed innovations, and presentation methods. Can be unpacked directly
        as (log_likelihood, filtered_states, regime_history).
    """
    ref_matrices = _extract_model_matrices(m_unconstrained)
    A_p_0, A_0_0, A_m_0, B_u_0, c_0, ss_0, variables, shocks = ref_matrices
    n_vars = len(variables)
    n_shocks = len(shocks)

    # Standardize data and varobs
    if isinstance(data, pd.DataFrame):
        data_df = data
        t_index = data.index
        if varobs is None:
            varobs = list(data.columns)
    else:
        data_mat = np.asarray(data, dtype=float)
        if varobs is None:
            varobs = [variables[i] for i in range(min(data_mat.shape[1], n_vars))]
        t_index = pd.RangeIndex(len(data_mat), name="t")
        data_df = pd.DataFrame(data_mat, columns=varobs, index=t_index)

    obs_vars = list(varobs)
    n_obs = len(obs_vars)
    T_obs = len(data_df)

    # Observation selection matrix Z: shape (n_obs, n_vars)
    Z = np.zeros((n_obs, n_vars))
    for i, v in enumerate(obs_vars):
        if v not in variables:
            raise ValueError(f"Observable variable '{v}' not in model variables: {variables}")
        Z[i, variables.index(v)] = 1.0

    # Covariance matrices
    if H is None:
        H_mat = np.zeros((n_obs, n_obs))
    else:
        H_mat = np.asarray(H, dtype=float)

    if Q is None:
        Q_mat = np.eye(n_shocks)
    else:
        Q_mat = np.asarray(Q, dtype=float)

    # Standardize models and constraints
    if isinstance(m_constrained_dict, Mapping):
        model_items = list(m_constrained_dict.items())
    elif isinstance(m_constrained_dict, (list, tuple)):
        model_items = [(f"c_{i}", m) for i, m in enumerate(m_constrained_dict)]
    else:
        model_items = [("c_0", m_constrained_dict)]

    K = len(model_items)
    # Pair the user's constraints with the models, key by key; a mismatch is
    # an error rather than a silent auto-detection with a guessed threshold.
    user_constraints_dict = _resolve_user_constraints(model_items, constraints, "piecewise_kalman_filter")

    constraint_info = []
    final_constraints: dict[str, OccBinConstraint] = {}
    for name, c_model in model_items:
        c_obj, eq_row, cons_mats = _auto_detect_constraint(
            m_unconstrained,
            c_model,
            default_name=name,
            user_constraint=user_constraints_dict.get(name),
            caller="piecewise_kalman_filter",
        )
        constraint_info.append((c_obj, eq_row, cons_mats))
        final_constraints[name] = c_obj

    # Splice every differing row of each constrained model into its regimes;
    # raises when two constrained models rewrite the same equation row.
    regime_matrices = _build_multi_regime_matrices(ref_matrices, constraint_info)

    # Reference decision rule
    dr = m_unconstrained.decision_rules()
    P_0 = np.zeros((n_vars, n_vars))
    for s in m_unconstrained.states:
        P_0[:, variables.index(s)] = dr.ghx[s].values

    # Pre-extract variable indices and relax parameters
    c_indices = []
    for k in range(K):
        c_obj, eq_row, _ = constraint_info[k]
        idx_var = variables.index(c_obj.variable)
        relax_idx = variables.index(c_obj.relax_variable) if c_obj.relax_variable else None
        c_indices.append((c_obj, eq_row, idx_var, relax_idx))

    # Seed state and covariance
    x_filt = np.zeros(n_vars)
    # Solve discrete Lyapunov for stationary P_0
    M_0 = A_0_0 + A_p_0 @ P_0
    R_0 = _safe_solve(M_0, -B_u_0)
    RQR = R_0 @ Q_mat @ R_0.T
    try:
        P_filt = scipy.linalg.solve_discrete_lyapunov(P_0, RQR)
        P_filt = 0.5 * (P_filt + P_filt.T)
        if not np.all(np.isfinite(P_filt)) or np.any(np.linalg.eigvalsh(P_filt) < -1e-8):
            P_filt = 10.0 * np.eye(n_vars)
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError, Exception):
        P_filt = 10.0 * np.eye(n_vars)

    # Storage arrays
    filtered_states_mat = np.zeros((T_obs, n_vars))
    filtered_covs = []
    imputed_shocks_mat = np.zeros((T_obs, n_shocks))
    forecast_errors_mat = np.zeros((T_obs, n_obs))
    realized_regimes = np.zeros(T_obs, dtype=int)
    total_loglik = 0.0
    all_converged = True
    total_iters = 0

    prev_regime_seq = np.zeros(horizon, dtype=int)

    for t in range(T_obs):
        y_t = data_df.iloc[t][obs_vars].to_numpy(dtype=float)
        obs_mask = np.isfinite(y_t)
        n_valid = int(np.sum(obs_mask))

        # Initialize conjectured regime sequence: warm-start from previous period
        conjectured_R = np.zeros(horizon, dtype=int)
        if t > 0:
            conjectured_R[:-1] = prev_regime_seq[1:]
            conjectured_R[-1] = 0

        converged_t = False
        reg_history: set[tuple[int, ...]] = set()

        for reg_it in range(1, max_regime_iter + 1):
            total_iters += 1
            # 1. OccBin backward recursion under conjectured_R with future shocks = 0
            P_seq = [None] * (horizon + 1)
            D_seq = [None] * (horizon + 1)
            R_seq = [None] * (horizon + 1)
            P_next = P_0
            D_next = np.zeros(n_vars)

            for s in range(horizon, 0, -1):
                r_s = int(conjectured_R[s - 1])
                Ap_s, A0_s, Am_s, Bu_s, c_s = regime_matrices[r_s]
                M_s = A0_s + Ap_s @ P_next
                P_s = _safe_solve(M_s, -Am_s)
                D_s = _safe_solve(M_s, -(Ap_s @ D_next + c_s))
                R_s = _safe_solve(M_s, -Bu_s)
                P_seq[s], D_seq[s], R_seq[s] = P_s, D_s, R_s
                P_next, D_next = P_s, D_s

            T_t = P_seq[1]
            C_t = D_seq[1]
            R_t = R_seq[1]

            # 2. Kalman Prediction
            x_pred = T_t @ x_filt + C_t
            P_pred = T_t @ P_filt @ T_t.T + R_t @ Q_mat @ R_t.T
            P_pred = 0.5 * (P_pred + P_pred.T)

            # 3. Kalman Update
            if n_valid == 0:
                # All variables missing at period t
                x_upd = x_pred.copy()
                P_upd = P_pred.copy()
                u_hat = np.zeros(n_shocks)
                v_t_full = np.zeros(n_obs)
                ll_t = 0.0
            else:
                Z_v = Z[obs_mask, :]
                y_v = y_t[obs_mask]
                H_v = H_mat[obs_mask, :][:, obs_mask]
                v_v = y_v - Z_v @ x_pred

                F_v = Z_v @ P_pred @ Z_v.T + H_v
                F_v = 0.5 * (F_v + F_v.T)
                # Safeguard against zero measurement error boundary / rank deficiency
                jitter = 1e-11 * max(1.0, float(np.trace(F_v) / len(F_v)))
                F_v_safe = F_v + jitter * np.eye(len(F_v))

                try:
                    L_F = scipy.linalg.cho_factor(F_v_safe, lower=True)
                    F_inv_v = scipy.linalg.cho_solve(L_F, v_v)
                    log_det_F = 2.0 * float(np.sum(np.log(np.diag(L_F[0]))))
                    K_gain = scipy.linalg.cho_solve(L_F, Z_v @ P_pred.T).T
                except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
                    evals = np.linalg.eigvalsh(F_v_safe)
                    log_det_F = float(np.sum(np.log(np.maximum(evals, 1e-12))))
                    F_inv = np.linalg.pinv(F_v_safe)
                    F_inv_v = F_inv @ v_v
                    K_gain = P_pred @ Z_v.T @ F_inv

                x_upd = x_pred + K_gain @ v_v
                P_upd = (np.eye(n_vars) - K_gain @ Z_v) @ P_pred
                P_upd = 0.5 * (P_upd + P_upd.T)

                # Impute contemporaneous shock: u_hat = Q R_t^T Z_v^T F_v^{-1} v_v
                u_hat = Q_mat @ R_t.T @ Z_v.T @ F_inv_v

                quad = float(v_v @ F_inv_v)
                ll_t = -0.5 * (n_valid * np.log(2.0 * np.pi) + log_det_F + quad)

                v_t_full = np.zeros(n_obs)
                v_t_full[obs_mask] = v_v

            # 4. Forward simulate over horizon H to verify regime consistency
            X_sim = np.zeros((horizon + 1, n_vars))
            X_sim[0] = x_filt
            X_sim[1] = T_t @ x_filt + C_t + R_t @ u_hat
            for s in range(2, horizon + 1):
                X_sim[s] = P_seq[s] @ X_sim[s - 1] + D_seq[s]

            sim_X_h = X_sim[1:]

            # Compute shadow values along simulated trajectory
            shadow_vals_h = np.zeros((K, horizon))
            for k in range(K):
                c_obj, eq_row, idx_var, relax_idx = c_indices[k]
                # eq_row is None for a trigger-style constraint: the simulated
                # value is then its own notional value (else branch below).
                a_var = float(A_0_0[eq_row, idx_var]) if eq_row is not None else 0.0
                if abs(a_var) > 1e-12:
                    row_others = A_0_0[eq_row].copy()
                    row_others[idx_var] = 0.0
                    for s in range(horizon):
                        x_p = X_sim[s]
                        x_c = sim_X_h[s]
                        x_n = sim_X_h[s + 1] if s + 1 < horizon else P_0 @ x_c
                        u_s = u_hat if s == 0 else np.zeros(n_shocks)
                        term_other = row_others @ x_c
                        term_next = A_p_0[eq_row, :] @ x_n
                        term_prev = A_m_0[eq_row, :] @ x_p
                        term_shock = B_u_0[eq_row, :] @ u_s
                        shadow_vals_h[k, s] = -(term_other + term_next + term_prev + term_shock + c_0[eq_row]) / a_var
                else:
                    shadow_vals_h[k, :] = sim_X_h[:, idx_var]

            # Update regime sequence
            new_R = np.zeros(horizon, dtype=int)
            for s in range(horizon):
                r_s = int(conjectured_R[s])
                r_new = 0
                for k in range(K):
                    c_obj, eq_row, idx_var, relax_idx = c_indices[k]
                    bit_k = (r_s >> k) & 1
                    if bit_k == 1:
                        val = sim_X_h[s, relax_idx] if relax_idx is not None else shadow_vals_h[k, s]
                        binds = not bool(c_obj.evaluate_relax(val))
                    else:
                        val = sim_X_h[s, idx_var]
                        binds = bool(c_obj.evaluate(val))
                    if binds:
                        r_new |= (1 << k)
                new_R[s] = r_new

            if np.array_equal(new_R, conjectured_R):
                converged_t = True
                conjectured_R = new_R
                break

            key = tuple(new_R.tolist())
            if key in reg_history:
                # Oscillating cycle: damp by updating first divergent period
                diff_s = np.where(new_R != conjectured_R)[0]
                if len(diff_s) > 0:
                    damped_R = conjectured_R.copy()
                    damped_R[diff_s[0]] = new_R[diff_s[0]]
                    d_key = tuple(damped_R.tolist())
                    if d_key not in reg_history:
                        reg_history.add(d_key)
                        conjectured_R = damped_R
                        continue
                converged_t = False
                break

            reg_history.add(key)
            conjectured_R = new_R

        if not converged_t:
            all_converged = False

        # Advance state to t
        x_filt = x_upd
        P_filt = P_upd

        filtered_states_mat[t, :] = x_filt
        filtered_covs.append(P_filt)
        imputed_shocks_mat[t, :] = u_hat
        forecast_errors_mat[t, :] = v_t_full
        realized_regimes[t] = int(conjectured_R[0])
        total_loglik += ll_t
        prev_regime_seq = conjectured_R

    filtered_states_df = pd.DataFrame(filtered_states_mat, columns=variables, index=t_index)
    imputed_shocks_df = pd.DataFrame(imputed_shocks_mat, columns=shocks, index=t_index)
    forecast_errors_df = pd.DataFrame(forecast_errors_mat, columns=obs_vars, index=t_index)
    regime_series = pd.Series(realized_regimes, index=t_index, name="regime")

    return PiecewiseKalmanResult(
        log_likelihood=float(total_loglik),
        filtered_states=filtered_states_df,
        regime_history=regime_series,
        filtered_covariances=filtered_covs,
        imputed_shocks=imputed_shocks_df,
        forecast_errors=forecast_errors_df,
        converged=all_converged,
        iterations=max(1, total_iters // max(1, T_obs)),
        reference_model=m_unconstrained,
        constraints=final_constraints,
    )


__all__ = [
    "OccBinConstraint",
    "OccBinResult",
    "PiecewiseKalmanResult",
    "DifferentiableOccBinResult",
    "solve_occbin",
    "solve_differentiable_occbin",
    "solve_multiconstraint_occbin",
    "piecewise_kalman_filter",
    "smin_tau",
    "smax_tau",
    "d_smax_tau",
    "d_smin_tau",
    "smooth_min",
    "smooth_max",
    "d_smooth_max",
    "d_smooth_min",
    "fischer_burmeister",
    "grad_fischer_burmeister",
    "Phi_tau",
]

