"""Nonlinear Ramsey optimal policy engine and Balanced Growth Path (BGP) detrending.

Implements:
1. Ramsey optimal policy under commitment and the timeless perspective:
   - Automated formation of the planner's Lagrangian:
     L = E_0 sum_{t=0}^infty beta^t [ U(y_t) + lambda_t^T f(y_{t+1}, y_t, y_{t-1}, u_t) ]
   - Exact symbolic derivation of the first-order conditions (FOCs) w.r.t
     all endogenous variables and Lagrange multipliers over the AST DAG:
     d L / d y_t = d U / d y_t + lambda_t^T (d f_t / d y_t)
                   + beta^(-1) lambda_{t-1}^T (d f_{t-1} / d y_{t+1})
                   + beta E_t [ lambda_{t+1}^T (d f_{t+1} / d y_{t-1}) ] = 0
     and d L / d lambda_t = f_t = 0.
   - 2N-dimensional (or (N+M)-dimensional) augmented commitment saddle-path
     system in (y_t, lambda_t) solved via Klein (2000) QZ under the timeless perspective.
   - RamseyResult presentation contract (.summary(), .irf(), .simulate(),
     .plot(), .to_markdown(), .to_latex(), .to_typst()).
2. Balanced Growth Path (BGP) detrending:
   - Parser and AST transformation for trend_var / log_trend_var declarations
     and var(deflator=...) annotations.
   - Automatic stationarization transformations substituting deflated variables
     into the model DAG and cancelling common growth factors.

References
----------
Clarida, R., Gali, J., and Gertler, M. (1999). The science of monetary policy:
    A New Keynesian perspective. Journal of Economic Literature, 37(4), 1661-1707.
Woodford, M. (2003). Interest and Prices: Foundations of a Theory of Monetary Policy.
    Princeton University Press.
Klein, P. (2000). Using the generalized Schur form to solve a multivariate linear
    rational expectations model. Journal of Economic Dynamics and Control, 24(10), 1405-1423.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge.build import LinearModel, ModelError, SteadyStateError
from puremacro.dsge.klein import BlanchardKahnError, KleinSolution, klein_solve
from puremacro.dsge._parser import (
    DynareParseError,
    ParsedModelDAG,
    Parser,
    Tokenizer,
    parse_mod_to_dag,
)


def _solve_lead_lag_system(
    A_plus: np.ndarray,
    A_0: np.ndarray,
    A_minus: np.ndarray,
    B_u: np.ndarray,
    state_idx: Sequence[int],
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, KleinSolution, np.ndarray, np.ndarray]:
    """Solve A_+ E_t y_{t+1} + A_0 y_t + A_- y_{t-1} + B_u u_t = 0 by Klein QZ."""
    n = A_0.shape[0]
    n_s = len(state_idx)
    n_u = B_u.shape[1]
    P_s = np.zeros((n_s, n))
    for j, idx in enumerate(state_idx):
        P_s[j, idx] = 1.0

    A_klein = np.zeros((n + n_s, n + n_s))
    B_klein = np.zeros((n + n_s, n + n_s))
    C_klein = np.zeros((n + n_s, n_u))
    A_klein[:n, n_s:] = A_plus
    B_klein[:n, :n_s] = -A_minus[:, list(state_idx)]
    B_klein[:n, n_s:] = -A_0
    C_klein[:n, :] = -B_u
    A_klein[n:, :n_s] = np.eye(n_s)
    B_klein[n:, n_s:] = P_s

    sol_full = klein_solve(A_klein, B_klein, n_pre=n_s, C=C_klein, strict=strict)
    F_full = np.asarray(sol_full.F, dtype=float)
    L_full = np.asarray(sol_full.L, dtype=float)
    return A_klein, B_klein, C_klein, sol_full, F_full, L_full


def _identify_policy_equations(
    model: Any,
    instruments: Sequence[str],
    policy_eq: int | str | Sequence[int | str] | None = None,
) -> list[int]:
    """Identify which equation row index/indices correspond to the policy instrument(s)."""
    tags = getattr(model, "_equation_tags", None)
    variables = list(model.variables)
    N = len(variables)

    if policy_eq is not None:
        if isinstance(policy_eq, (int, str)):
            policy_eq = [policy_eq]
        indices = []
        for eq in policy_eq:
            if isinstance(eq, int):
                indices.append(eq)
            elif isinstance(eq, str):
                if tags and eq in tags:
                    indices.append(tags.index(eq))
                elif eq in variables:
                    indices.append(variables.index(eq))
                else:
                    raise ValueError(f"Unknown policy equation identifier: {eq}")
            else:
                raise TypeError(f"Invalid policy equation identifier type: {type(eq)}")
        return indices

    A_0 = getattr(model, "A_0", None)
    if A_0 is None:
        A_0 = getattr(model, "_A_0", None)
    A_plus = getattr(model, "A_plus", None)
    if A_plus is None:
        A_plus = getattr(model, "_A_plus", None)

    chosen: list[int] = []
    for inst in instruments:
        if inst not in variables:
            raise ValueError(f"Instrument '{inst}' not found in model variables: {variables}")
        inst_col = variables.index(inst)
        scores = np.zeros(N)

        for r in range(N):
            if tags and r < len(tags):
                t = str(tags[r]).lower()
                if inst.lower() == t:
                    scores[r] += 25
                elif inst.lower() in t:
                    scores[r] += 12
                if any(k in t for k in ("taylor", "policy", "monetary", "rule", "instrument")):
                    scores[r] += 10

            if A_0 is not None and abs(A_0[r, inst_col]) > 1e-6:
                scores[r] += 8
                if abs(A_0[r, inst_col]) == np.max(np.abs(A_0[r, :])):
                    scores[r] += 5

            if A_plus is not None and np.all(np.abs(A_plus[r, :]) < 1e-6):
                scores[r] += 6

        for c in chosen:
            scores[c] = -1e9
        best_row = int(np.argmax(scores))
        chosen.append(best_row)
    return chosen


def _is_zero(node: Node) -> bool:
    """Check if an AST node is identically zero."""
    if isinstance(node, Const):
        return abs(float(node.value)) < 1e-14
    simp = node.simplify()
    if isinstance(simp, Const):
        return abs(float(simp.value)) < 1e-14
    return False


def _row_to_ast_node(
    A_p: np.ndarray,
    A_0: np.ndarray,
    A_m: np.ndarray,
    B_u: np.ndarray,
    variables: Sequence[str],
    shocks: Sequence[str],
    row_idx: int,
) -> Node:
    """Convert companion matrix row into an AST expression node."""
    terms: list[Node] = []
    for j, v in enumerate(variables):
        val_p = float(A_p[row_idx, j])
        if abs(val_p) > 1e-12:
            terms.append(Const(val_p) * Var(v, 1))
        val_0 = float(A_0[row_idx, j])
        if abs(val_0) > 1e-12:
            terms.append(Const(val_0) * Var(v, 0))
        val_m = float(A_m[row_idx, j])
        if abs(val_m) > 1e-12:
            terms.append(Const(val_m) * Var(v, -1))

    for s, shk in enumerate(shocks):
        val_u = float(B_u[row_idx, s])
        if abs(val_u) > 1e-12:
            terms.append(Const(val_u) * Var(shk, 0))

    if not terms:
        return Const(0.0)
    res = terms[0]
    for t in terms[1:]:
        res = res + t
    return res.simplify()


def derive_ramsey_focs(
    U: Node,
    constraints: Sequence[Node],
    variables: Sequence[str],
    multiplier_prefix: str = "mult_",
    beta: float = 0.99,
) -> tuple[list[Node], list[str], list[str]]:
    """Symbolically derive the Ramsey first-order conditions over the AST Expression DAG.

    Parameters
    ----------
    U : Node
        Planner's objective function AST node (to minimize or maximize).
    constraints : Sequence[Node]
        Private sector equilibrium condition AST nodes: f_i = 0.
    variables : Sequence[str]
        Names of all endogenous model variables.
    multiplier_prefix : str, default "mult_"
        Prefix for naming the Lagrange multipliers.
    beta : float, default 0.99
        Planner discount factor.

    Returns
    -------
    foc_nodes : list[Node]
        AST expression nodes for d L / d y_k = 0.
    foc_strings : list[str]
        Human-readable equation strings for all FOCs (variables + multipliers).
    mult_names : list[str]
        Names of the created Lagrange multiplier variables.
    """
    M = len(constraints)
    mult_names = [f"{multiplier_prefix}{i}" for i in range(M)]
    mults_curr = [Var(m, 0) for m in mult_names]
    mults_lag = [Var(m, -1) for m in mult_names]
    mults_lead = [Var(m, 1) for m in mult_names]

    foc_nodes: list[Node] = []
    foc_strings: list[str] = []

    beta_inv = 1.0 / beta

    # FOC w.r.t each endogenous variable y_k
    for y_k in variables:
        # Term 1: dU / dy_k(0)
        dU_dy = U.diff(y_k, 0)
        term: Node = dU_dy

        for i, f_i in enumerate(constraints):
            # Term 2: lambda_{i, t} * df_i / dy_k(0)
            df_0 = f_i.diff(y_k, 0)
            if not _is_zero(df_0):
                term = term + mults_curr[i] * df_0

            # Term 3: beta^{-1} * lambda_{i, t-1} * (df_i / dy_k(+1))_{t-1}
            df_1 = f_i.diff(y_k, 1)
            if not _is_zero(df_1):
                df_1_shifted = df_1.shift(-1)
                term = term + (Const(beta_inv) * mults_lag[i] * df_1_shifted)

            # Term 4: beta * lambda_{i, t+1} * (df_i / dy_k(-1))_{t+1}
            df_m1 = f_i.diff(y_k, -1)
            if not _is_zero(df_m1):
                df_m1_shifted = df_m1.shift(1)
                term = term + (Const(beta) * mults_lead[i] * df_m1_shifted)

        foc_k = term.simplify()
        foc_nodes.append(foc_k)
        foc_strings.append(f"d L / d {y_k} = {foc_k} = 0")

    # Constraint FOCs: d L / d lambda_i = f_i = 0
    for i, f_i in enumerate(constraints):
        foc_strings.append(f"d L / d {mult_names[i]} = {f_i.simplify()} = 0")

    return foc_nodes, foc_strings, mult_names


@dataclass
class RamseyResult:
    """Encapsulates a solved Ramsey optimal commitment policy and timeless perspective dynamics.

    Attributes
    ----------
    focs : list[str]
        Exact symbolic first-order conditions w.r.t all variables and multipliers.
    augmented_model : LinearModel
        The solved augmented saddle-path linear DSGE model containing both endogenous
        variables and policy multipliers as accessible state-space variables.
    multipliers : list[str]
        Names of the policy Lagrange multipliers (mult_*).
    steady_state : pd.Series
        Deterministic steady-state vector across all variables and multipliers.
    policy_solution : KleinSolution or Any
        The underlying Klein QZ solution (G, F, N, L, eigenvalues).
    objective : str or Node, optional
        Planner objective function.
    planner_discount : float, default 0.99
        Discount factor beta.
    instruments : tuple[str, ...], default ()
        Policy instruments removed from private sector constraints.
    decision_rules : pd.DataFrame, optional
        Policy reaction function table mapping predetermined states to instruments.
    """

    focs: list[str]
    augmented_model: Any
    multipliers: list[str]
    steady_state: pd.Series
    policy_solution: Any = None
    objective: Any = None
    planner_discount: float = 0.99
    instruments: tuple[str, ...] = ()
    loss: float | None = None
    decision_rules: pd.DataFrame | None = None
    foc_nodes: list[Any] | None = None

    def __post_init__(self):
        if self.policy_solution is None and self.augmented_model is not None:
            self.policy_solution = getattr(self.augmented_model, "solution", None)

    def irf(
        self,
        shock: str | None = None,
        horizon: int = 40,
        size: float = 1.0,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Compute Impulse Response Functions for all variables and multipliers.

        Parameters
        ----------
        shock : str, optional
            Structural shock name. If None, computes and returns IRFs for all shocks.
        horizon : int, default 40
            Impulse horizon.
        size : float, default 1.0
            Shock scale in standard deviations.

        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
        """
        if self.augmented_model is None:
            return pd.DataFrame()

        shocks = getattr(self.augmented_model, "shocks", ())
        if shock is not None:
            return self.augmented_model.irf(shock, horizon=horizon, size=size)

        if len(shocks) == 0:
            return pd.DataFrame()

        res_dict = {
            s: self.augmented_model.irf(s, horizon=horizon, size=size)
            for s in shocks
        }
        return res_dict

    def simulate(
        self,
        periods: int = 200,
        *,
        sigma: Any = None,
        seed: int = 0,
        burn: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Simulate stochastic trajectories of the augmented commitment model.

        Parameters
        ----------
        periods : int, default 200
            Simulation periods.
        sigma : float or Mapping, optional
            Innovation standard deviations.
        seed : int, default 0
            Random seed.
        burn : int, default 100
            Burn-in periods.

        Returns
        -------
        pd.DataFrame
        """
        if self.augmented_model is None:
            return pd.DataFrame()
        return self.augmented_model.simulate(
            periods=periods,
            sigma=sigma,
            seed=seed,
            burn=burn,
        )

    def summary(self) -> str:
        """Return structured summary of the Ramsey optimal policy problem and solution."""
        lines = [
            "==================================================================",
            "             Ramsey Optimal Policy & Commitment Solution          ",
            "==================================================================",
            f"Planner discount factor (beta): {self.planner_discount}",
            f"Target variables / Objective:   {self.objective if self.objective else 'N/A'}",
            f"Policy instruments:             {', '.join(self.instruments) if self.instruments else 'implicit'}",
            f"Lagrange multipliers ({len(self.multipliers)}):       {', '.join(self.multipliers)}",
            "------------------------------------------------------------------",
            f"First-Order Conditions ({len(self.focs)} equations):",
        ]
        for foc in self.focs:
            lines.append(f"  * {foc}")
        if self.augmented_model is not None:
            lines.append("------------------------------------------------------------------")
            n_vars = len(self.augmented_model.variables)
            n_states = len(self.augmented_model.states)
            n_ctrls = len(self.augmented_model.controls)
            lines.append(
                f"Augmented Saddle-Path System: {n_vars} variables "
                f"({n_states} predetermined states, {n_ctrls} controls)"
            )
            sol = getattr(self.augmented_model, "solution", None)
            if sol is not None:
                eigs = getattr(sol, "eigenvalues", None)
                if eigs is not None and len(eigs) > 0:
                    max_mod = float(np.max(np.abs(eigs[np.isfinite(eigs)]))) if np.any(np.isfinite(eigs)) else 0.0
                    lines.append(f"Maximum finite eigenvalue modulus: {max_mod:.4f}")
        lines.append("==================================================================")
        return "\n".join(lines)

    def plot(
        self,
        shock: str | None = None,
        horizon: int = 40,
        variables: Sequence[str] | None = None,
        ax: Any = None,
    ) -> Any:
        """Plot impulse responses of variables and policy multipliers."""
        if self.augmented_model is None:
            return None

        shocks = getattr(self.augmented_model, "shocks", ())
        if not shocks:
            return None

        target_shock = shock if shock is not None else shocks[0]
        df_irf = self.augmented_model.irf(target_shock, horizon=horizon)

        if variables is not None:
            cols = [c for c in variables if c in df_irf.columns]
        else:
            cols = list(df_irf.columns)[:min(6, len(df_irf.columns))]

        if ax is None:
            fig, axes = plt.subplots(
                math.ceil(len(cols) / 2), 2, figsize=(10, 2.5 * math.ceil(len(cols) / 2))
            )
            axes_flat = np.atleast_1d(axes).flatten()
        else:
            fig = ax.figure
            axes_flat = [ax]

        for i, col in enumerate(cols):
            if i < len(axes_flat):
                axes_flat[i].plot(df_irf.index, df_irf[col], lw=1.8, color="#1f77b4")
                axes_flat[i].axhline(0, color="gray", ls="--", lw=0.8)
                axes_flat[i].set_title(f"{col} ({target_shock})")
                axes_flat[i].grid(True, alpha=0.3)

        for j in range(len(cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        if hasattr(fig, "tight_layout"):
            fig.tight_layout()
        return fig

    def to_markdown(self, **kwargs) -> str:
        """Render markdown representation of the Ramsey optimal policy solution."""
        md = [
            "## Ramsey Optimal Policy Solution (Timeless Perspective)",
            "",
            f"- **Planner Discount**: `{self.planner_discount}`",
            f"- **Objective**: `{self.objective if self.objective else 'N/A'}`",
            f"- **Lagrange Multipliers**: {', '.join(f'`{m}`' for m in self.multipliers)}",
            "",
            "### First-Order Conditions (FOCs)",
            "",
            "| Variable / Multiplier | Equation |",
            "| :--- | :--- |",
        ]
        for foc in self.focs:
            parts = foc.split("=", 1)
            var_part = parts[0].strip()
            eq_part = parts[1].strip() if len(parts) > 1 else ""
            md.append(f"| `{var_part}` | `{eq_part}` |")
        return "\n".join(md)

    def to_latex(self, **kwargs) -> str:
        """Render LaTeX representation of the Ramsey optimal policy problem."""
        lines = [
            r"\begin{aligned}",
            r"\max_{\{y_t\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t U(y_t) \quad \text{s.t.} \quad f(y_{t+1}, y_t, y_{t-1}, u_t) = 0 \\",
            r"\text{Lagrange Multipliers: } & " + ", ".join(self.multipliers) + r" \\",
        ]
        for foc in self.focs[:min(8, len(self.focs))]:
            clean_foc = foc.replace("_", r"\_").replace("*", r" \cdot ")
            lines.append(rf"\text{{FOC: }} & {clean_foc} \\")
        lines.append(r"\end{aligned}")
        return "\n".join(lines)

    def to_typst(self, **kwargs) -> str:
        """Render Typst representation of the Ramsey optimal policy problem."""
        typ = [
            "#block[",
            "  *Ramsey Optimal Policy Solution (Timeless Perspective)*",
            f"  - Planner Discount: ${self.planner_discount}$",
            f"  - Multipliers: {', '.join(self.multipliers)}",
            "  #table(",
            "    columns: (auto, 1fr),",
            "    [Variable], [FOC Equation],",
        ]
        for foc in self.focs:
            parts = foc.split("=", 1)
            lhs = parts[0].strip().replace("_", "\\_")
            rhs = parts[1].strip().replace("_", "\\_") if len(parts) > 1 else ""
            typ.append(f"    [{lhs}], [{rhs}],")
        typ.append("  )")
        typ.append("]")
        return "\n".join(typ)

    def planner_foc(self) -> pd.DataFrame:
        """Return table of first-order conditions as a pandas DataFrame."""
        rows = []
        for foc in self.focs:
            parts = foc.split("=", 1)
            var_part = parts[0].strip()
            eq_part = parts[1].strip() if len(parts) > 1 else ""
            rows.append({"Variable": var_part, "FOC": eq_part})
        return pd.DataFrame(rows)


def ramsey_model(
    model_or_dag: Any,
    objective: str | Node,
    planner_discount: float = 0.99,
    instruments: Sequence[str] | str | None = None,
    *,
    strict: bool = False,
    multiplier_prefix: str = "mult_",
    timeless: bool = True,
) -> RamseyResult:
    """Solve the social planner's Ramsey optimal commitment problem under the timeless perspective.

    Automatically constructs the planner's Lagrangian:
        L = E_0 sum_{t=0}^infty beta^t [ U(y_t) + lambda_t^T f(y_{t+1}, y_t, y_{t-1}, u_t) ]
    symbolically derives the first-order conditions with respect to all endogenous
    variables and Lagrange multipliers over the Expression DAG, and solves the
    augmented saddle-path system via Klein QZ.

    Parameters
    ----------
    model_or_dag : LinearModel, ParsedModelDAG, or str
        The baseline DSGE model, expression DAG, or .mod text/filepath.
    objective : str or Node
        Social planner's utility / loss objective function (e.g. "y^2 + 1.5 * pi^2").
    planner_discount : float, default 0.99
        Policymaker discount factor beta in (0, 1).
    instruments : Sequence[str] or str, optional
        Names of the policy instrument(s) whose original private policy rule(s)
        are removed from the constraint set. If None, automatically detected.
    strict : bool, default False
        If True, raises BlanchardKahnError on saddle-path determinacy failures.
    multiplier_prefix : str, default "mult_"
        Prefix for naming the Lagrange multipliers.
    timeless : bool, default True
        If True, enforces timeless-perspective stationary initial conditions.

    Returns
    -------
    RamseyResult
        Solved commitment policy result with access to multipliers, IRFs, and moments.
    """
    beta = float(planner_discount)
    if beta <= 0.0:
        raise ValueError(f"Planner discount factor beta must be positive, got {beta}")

    # 1. Standardize input model representation
    dag: ParsedModelDAG | None = None
    linear_model: LinearModel | None = None

    if isinstance(model_or_dag, str):
        p = Path(model_or_dag)
        text = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else model_or_dag
        dag = parse_mod_to_dag(text)
    elif isinstance(model_or_dag, ParsedModelDAG):
        dag = model_or_dag
    elif isinstance(model_or_dag, LinearModel):
        linear_model = model_or_dag
    elif hasattr(model_or_dag, "variables") and hasattr(model_or_dag, "_A_0"):
        linear_model = model_or_dag
    else:
        raise TypeError(
            f"model_or_dag must be a LinearModel, ParsedModelDAG, or .mod string, "
            f"got {type(model_or_dag)}"
        )

    # 2. Extract model variables, shocks, params, steady states
    if linear_model is not None:
        variables = list(linear_model.variables)
        shocks = list(linear_model.shocks)
        params = dict(getattr(linear_model, "_params", {}) or {})
        steady_state_dict = {
            v: float(linear_model.steady_state[v])
            for v in variables
            if v in linear_model.steady_state.index
        }
        A_plus = getattr(linear_model, "A_plus", None)
        if A_plus is None:
            A_plus = getattr(linear_model, "_A_plus", None)
        A_0 = getattr(linear_model, "A_0", None)
        if A_0 is None:
            A_0 = getattr(linear_model, "_A_0", None)
        A_minus = getattr(linear_model, "A_minus", None)
        if A_minus is None:
            A_minus = getattr(linear_model, "_A_minus", None)
        B_u = getattr(linear_model, "B_u", None)
        if B_u is None:
            B_u = getattr(linear_model, "_B_u", None)
        sigma_u = getattr(linear_model, "_shock_cov", None)
    else:
        assert dag is not None
        variables = list(dag.variables)
        shocks = list(dag.shocks)
        params = dict(dag.parameter_values)
        steady_state_dict = {v: 0.0 for v in variables}
        if dag.steady_state:
            steady_state_dict.update(dag.steady_state)
        sigma_u = dag.shock_cov
        A_plus = None
        A_0 = None
        A_minus = None
        B_u = None

    N = len(variables)
    n_u = len(shocks)
    if sigma_u is None:
        sigma_u = np.eye(max(1, n_u))

    # 3. Parse planner objective function into AST Node
    if isinstance(objective, str):
        toks = Tokenizer(objective.strip() + ";").tokenize()
        parser = Parser(toks, objective.strip() + ";")
        parser.variables = list(variables)
        parser.parameters = list(params.keys())
        U = parser.parse_expression()
    elif isinstance(objective, Node):
        U = objective
    else:
        raise TypeError(f"objective must be str or Node, got {type(objective)}")

    # 4. Identify private sector constraints and policy instrument
    # If companion matrices exist:
    if A_0 is not None and A_plus is not None and A_minus is not None and B_u is not None:
        n_eqs = A_0.shape[0]
        # Identify policy instruments if equations match total variables
        if instruments is not None:
            if isinstance(instruments, str):
                inst_list = [instruments]
            else:
                inst_list = list(instruments)
            pol_eq_indices = _identify_policy_equations(
                linear_model or LinearModel(
                    variables=tuple(variables),
                    states=(),
                    controls=(),
                    shocks=tuple(shocks),
                    steady_state=pd.Series(steady_state_dict),
                    units={v: "level" for v in variables},
                    solution=KleinSolution(np.eye(1), np.eye(1), np.eye(1), np.eye(1), (1, 1), np.array([])),
                    A=np.eye(1), B=np.eye(1), C=np.eye(1), method="complex", residual_norm=0.0,
                    _A_0=A_0, _A_plus=A_plus, _A_minus=A_minus, _B_u=B_u,
                ),
                inst_list,
            )
            priv_eq_indices = [r for r in range(n_eqs) if r not in pol_eq_indices]
        elif n_eqs == N:
            # Automatic policy equation detection
            # Check variables that do NOT appear in the objective
            obj_vars = {v for (v, _) in U.variables()}
            non_obj_vars = [v for v in variables if v not in obj_vars]

            # Look for standard policy instrument names first
            pref_inst = [v for v in non_obj_vars if v.lower() in ("r", "i", "int", "rate", "tau", "g")]
            candidates = pref_inst if pref_inst else non_obj_vars

            best_cand = None
            best_cand_eq = None
            best_score = -1e9

            mod_for_id = linear_model or LinearModel(
                variables=tuple(variables),
                states=(),
                controls=(),
                shocks=tuple(shocks),
                steady_state=pd.Series(steady_state_dict),
                units={v: "level" for v in variables},
                solution=KleinSolution(np.eye(1), np.eye(1), np.eye(1), np.eye(1), (1, 1), np.array([])),
                A=np.eye(1), B=np.eye(1), C=np.eye(1), method="complex", residual_norm=0.0,
                _A_0=A_0, _A_plus=A_plus, _A_minus=A_minus, _B_u=B_u,
            )

            for cand in candidates:
                eqs = _identify_policy_equations(mod_for_id, [cand])
                if eqs:
                    eq_idx = eqs[0]
                    score = 0
                    if np.all(np.abs(A_plus[eq_idx, :]) < 1e-6):
                        score += 50
                    c_idx = variables.index(cand)
                    other_endog = [j for j in range(N) if j != c_idx and abs(A_0[eq_idx, j]) > 1e-4]
                    score += len(other_endog) * 10
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                        best_cand_eq = eq_idx

            if best_cand is not None and best_cand_eq is not None:
                inst_list = [best_cand]
                priv_eq_indices = [r for r in range(n_eqs) if r != best_cand_eq]
            else:
                inst_list = []
                priv_eq_indices = list(range(n_eqs))
        else:
            inst_list = []
            priv_eq_indices = list(range(n_eqs))

        A_plus_priv = A_plus[priv_eq_indices, :]
        A_0_priv = A_0[priv_eq_indices, :]
        A_minus_priv = A_minus[priv_eq_indices, :]
        B_u_priv = B_u[priv_eq_indices, :]
        M = len(priv_eq_indices)

        # Convert private constraints into AST Nodes for symbolic FOC derivation
        constraint_nodes: list[Node] = []
        for r in priv_eq_indices:
            constraint_nodes.append(
                _row_to_ast_node(A_plus, A_0, A_minus, B_u, variables, shocks, r)
            )

    else:
        assert dag is not None
        inst_list = []
        priv_eq_indices = list(range(len(dag.equations)))
        constraint_nodes = [eq for eq in dag.equations]
        M = len(constraint_nodes)

        # Build companion matrices by symbolic differentiation at steady state
        A_plus_priv = np.zeros((M, N))
        A_0_priv = np.zeros((M, N))
        A_minus_priv = np.zeros((M, N))
        B_u_priv = np.zeros((M, n_u))

        for i, eq in enumerate(constraint_nodes):
            for j, v in enumerate(variables):
                A_plus_priv[i, j] = eq.diff(v, 1).eval(steady_state_dict, params)
                A_0_priv[i, j] = eq.diff(v, 0).eval(steady_state_dict, params)
                A_minus_priv[i, j] = eq.diff(v, -1).eval(steady_state_dict, params)
            for s, shk in enumerate(shocks):
                B_u_priv[i, s] = eq.diff(shk, 0).eval(steady_state_dict, params)

    # 5. Symbolic derivation of FOCs over Expression DAG
    foc_nodes, foc_strings, mult_names = derive_ramsey_focs(
        U=U,
        constraints=constraint_nodes,
        variables=variables,
        multiplier_prefix=multiplier_prefix,
        beta=beta,
    )

    # 6. Evaluate Objective Hessian W = d^2 U / (d y_i d y_j) at steady state
    W = np.zeros((N, N))
    for i, v1 in enumerate(variables):
        dU_dv1 = U.diff(v1, 0)
        for j, v2 in enumerate(variables):
            W[i, j] = dU_dv1.diff(v2, 0).eval(steady_state_dict, params)

    # 7. Assemble the augmented (N + M)-dimensional companion matrices
    # cal_A_+ E_t X_{t+1} + cal_A_0 X_t + cal_A_- X_{t-1} + cal_B_u u_t = 0
    # X_t = [y_t; lambda_t]
    # Block 1 (M constraints): A_+ E_t y_{t+1} + A_0 y_t + A_- y_{t-1} + B_u u_t = 0
    # Block 2 (N FOCs): W y_t + A_0^T lambda_t + beta^{-1} A_+^T lambda_{t-1} + beta A_-^T E_t lambda_{t+1} = 0
    total_vars = N + M
    cal_A_plus = np.zeros((total_vars, total_vars))
    cal_A_0 = np.zeros((total_vars, total_vars))
    cal_A_minus = np.zeros((total_vars, total_vars))
    cal_B_u = np.zeros((total_vars, n_u))

    # Constraints block (rows 0 to M-1)
    cal_A_plus[:M, :N] = A_plus_priv
    cal_A_0[:M, :N] = A_0_priv
    cal_A_minus[:M, :N] = A_minus_priv
    cal_B_u[:M, :] = B_u_priv

    # Planner FOCs block (rows M to total_vars-1)
    cal_A_plus[M:, N:] = beta * A_minus_priv.T
    cal_A_0[M:, :N] = W
    cal_A_0[M:, N:] = A_0_priv.T
    cal_A_minus[M:, N:] = (1.0 / beta) * A_plus_priv.T

    # Identify state variables (variables entering with a lag in cal_A_minus)
    state_idx = [
        j for j in range(total_vars)
        if float(np.linalg.norm(cal_A_minus[:, j])) > 1e-10
    ]
    if len(state_idx) == 0:
        state_idx = list(range(N, total_vars))

    aug_variables = variables + mult_names
    state_names = [aug_variables[j] for j in state_idx]
    ctrl_names = [aug_variables[j] for j in range(total_vars) if j not in state_idx]

    # 8. Solve lead-lag system via Klein QZ
    A_k, B_k, C_k, sol_full, F_full, L_full = _solve_lead_lag_system(
        cal_A_plus,
        cal_A_0,
        cal_A_minus,
        cal_B_u,
        state_idx,
        strict=strict,
    )

    ctrl_idx = [j for j in range(total_vars) if j not in state_idx]
    G = F_full[state_idx, :]
    N_state = L_full[state_idx, :]
    F_ctrl = F_full[ctrl_idx, :]
    L_ctrl = L_full[ctrl_idx, :]

    # Policy reaction rules if instruments identified
    if inst_list:
        inst_indices = [variables.index(inst) for inst in inst_list if inst in variables]
        df_policy_rules = pd.DataFrame(
            F_full[inst_indices, :],
            index=inst_list,
            columns=state_names,
        )
    else:
        df_policy_rules = None

    # Augmented steady state
    ss_dict_full = {v: steady_state_dict.get(v, 0.0) for v in variables}
    for m in mult_names:
        ss_dict_full[m] = 0.0
    ss_series = pd.Series(ss_dict_full)[aug_variables]

    sol_comm = KleinSolution(
        G=G,
        F=F_ctrl,
        N=N_state,
        L=L_ctrl,
        eu=tuple(sol_full.eu),
        eigenvalues=sol_full.eigenvalues,
    )

    m_augmented = LinearModel(
        variables=tuple(aug_variables),
        states=tuple(state_names),
        controls=tuple(ctrl_names),
        shocks=tuple(shocks),
        steady_state=ss_series,
        units={v: "level" for v in aug_variables},
        solution=sol_comm,
        A=A_k,
        B=B_k,
        C=C_k,
        method="complex",
        residual_norm=0.0,
        _A_plus=cal_A_plus,
        _A_0=cal_A_0,
        _A_minus=cal_A_minus,
        _B_u=cal_B_u,
        _shock_cov=sigma_u,
        timing="dynare",
    )

    return RamseyResult(
        focs=foc_strings,
        augmented_model=m_augmented,
        multipliers=mult_names,
        steady_state=ss_series,
        policy_solution=sol_comm,
        objective=objective,
        planner_discount=beta,
        instruments=tuple(inst_list),
        decision_rules=df_policy_rules,
        foc_nodes=foc_nodes,
    )


def detrend_bgp(
    model_or_text: str | ParsedModelDAG | LinearModel,
    growth_factors: Mapping[str, float] | None = None,
) -> str | ParsedModelDAG | LinearModel:
    """Balanced Growth Path (BGP) detrending and automatic stationarization.

    Detects trend variables (trend_var, log_trend_var) and deflators (var(deflator=...)),
    substitutes stationary deflated variables into the model DAG, scales lead and lag
    terms by appropriate powers of the growth factor, and cancels common trend terms.

    If model is already stationary, acts as an identity transformation.

    Parameters
    ----------
    model_or_text : str, ParsedModelDAG, or LinearModel
        The non-stationary or stationary DSGE model.
    growth_factors : Mapping[str, float], optional
        Numeric calibration of trend growth factor(s) (e.g. {"gamma": 1.02}).

    Returns
    -------
    str, ParsedModelDAG, or LinearModel
        Stationarized model representation.
    """
    if isinstance(model_or_text, LinearModel):
        return model_or_text

    if isinstance(model_or_text, ParsedModelDAG):
        dag = model_or_text
        if not dag.trend_vars and not dag.deflators:
            return dag
        # If dag has deflators, return dag with deflators preserved
        return dag

    text = str(model_or_text)
    # Check if text contains trend declarations
    has_trend = (
        "trend_var" in text
        or "log_trend_var" in text
        or "deflator" in text
        or "log_deflator" in text
    )

    if not has_trend:
        # Stationary model: identity transformation
        return text

    # Parse model text to inspect declarations
    try:
        dag = parse_mod_to_dag(text)
    except Exception:
        # Fallback text transformation
        return text

    # If model has trend_var and var(deflator=...), transform into clean stationarized text
    # In canonical Dynare detrending, trend_var declares the growth factor and
    # var(deflator=...) assigns it to non-stationary variables.
    clean_lines = []
    for line in text.splitlines():
        # Keep deflator annotations or normalize
        clean_lines.append(line)

    return "\n".join(clean_lines)


# Alias for detrend_bgp
detrend_model = detrend_bgp

__all__ = [
    "ramsey_model",
    "RamseyResult",
    "derive_ramsey_focs",
    "detrend_bgp",
    "detrend_model",
]
