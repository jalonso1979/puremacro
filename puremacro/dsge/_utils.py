"""Dynare Language Extensions and Model Utilities for puremacro DSGE models.

Provides:
- `ModelInfoResult`: Frozen dataclass containing structural model summary metadata,
  lead/lag incidence matrix, dynamic variable classifications, and Jacobian sparsity.
  Adheres to puremacro presentation contract (.summary(), .to_markdown(), .to_latex(), .to_typst()).
- `model_info(model_or_dag) -> ModelInfoResult`: Computes comprehensive model summary.
- `write_latex_dynamic_model(model_or_dag, filepath=None, write_equation_tags=True) -> str`:
  Emits publication-grade LaTeX equations from parsed model DAG.
- `detect_linear_model(model_dag) -> bool`:
  Determines whether a model is linear by inspecting AST DAG derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

# Standard Greek symbols for LaTeX parameter rendering
GREEK_SYMBOLS: dict[str, str] = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "varepsilon": r"\varepsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "vartheta": r"\vartheta",
    "iota": r"\iota",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "pi": r"\pi",
    "varpi": r"\varpi",
    "rho": r"\rho",
    "varrho": r"\varrho",
    "sigma": r"\sigma",
    "varsigma": r"\varsigma",
    "tau": r"\tau",
    "upsilon": r"\upsilon",
    "phi": r"\phi",
    "varphi": r"\varphi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
    "Gamma": r"\Gamma",
    "Delta": r"\Delta",
    "Theta": r"\Theta",
    "Lambda": r"\Lambda",
    "Xi": r"\Xi",
    "Pi": r"\Pi",
    "Sigma": r"\Sigma",
    "Upsilon": r"\Upsilon",
    "Phi": r"\Phi",
    "Psi": r"\Psi",
    "Omega": r"\Omega",
}


@dataclass(frozen=True)
class ModelInfoResult:
    """Frozen dataclass containing structural model summary metadata and diagnostics."""

    num_equations: int
    num_variables: int
    num_shocks: int
    num_parameters: int
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    states: tuple[str, ...]
    controls: tuple[str, ...]
    purely_static: tuple[str, ...]
    purely_forward: tuple[str, ...]
    purely_backward: tuple[str, ...]
    mixed: tuple[str, ...]
    lead_lag_incidence: np.ndarray
    jacobian_sparsity: dict[str, float]
    is_linear: bool
    equation_tags: dict[int, dict[str, str]] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Tabulate high-level model metrics into a pandas DataFrame."""
        rows = [
            ("Number of equations", self.num_equations),
            ("Number of variables", self.num_variables),
            ("Number of shocks", self.num_shocks),
            ("Number of parameters", self.num_parameters),
            ("Linear model", self.is_linear),
            ("Predetermined states", len(self.states)),
            ("Forward controls", len(self.controls)),
            ("Purely static variables", len(self.purely_static)),
            ("Purely forward variables", len(self.purely_forward)),
            ("Purely backward variables", len(self.purely_backward)),
            ("Mixed variables", len(self.mixed)),
            (
                "Jacobian density (A_+)",
                f"{self.jacobian_sparsity.get('A_plus', 0.0):.4f}",
            ),
            (
                "Jacobian density (A_0)",
                f"{self.jacobian_sparsity.get('A_0', 0.0):.4f}",
            ),
            (
                "Jacobian density (A_-)",
                f"{self.jacobian_sparsity.get('A_minus', 0.0):.4f}",
            ),
            (
                "Jacobian density (B_u)",
                f"{self.jacobian_sparsity.get('B_u', 0.0):.4f}",
            ),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")

    def summary(self) -> str:
        """Generate formatted publication-grade ASCII summary."""
        lines = [
            "=" * 78,
            "Dynare Model Information & Structural Diagnostics",
            "=" * 78,
            f"Equations                  : {self.num_equations}",
            f"Variables                  : {self.num_variables}",
            f"Shocks                     : {self.num_shocks}",
            f"Parameters                 : {self.num_parameters}",
            f"Linear model               : {self.is_linear}",
            "",
            "Dynamic Classification:",
            f"  States (predetermined)   : {', '.join(self.states) if self.states else 'none'}",
            f"  Controls (forward)       : {', '.join(self.controls) if self.controls else 'none'}",
            f"  Purely static            : {', '.join(self.purely_static) if self.purely_static else 'none'}",
            f"  Purely forward           : {', '.join(self.purely_forward) if self.purely_forward else 'none'}",
            f"  Purely backward          : {', '.join(self.purely_backward) if self.purely_backward else 'none'}",
            f"  Mixed (lead & lag)       : {', '.join(self.mixed) if self.mixed else 'none'}",
            "",
            "Jacobian Sparsity (Non-zero Density):",
            f"  A_+ (lead)               : {self.jacobian_sparsity.get('A_plus', 0.0):.4f}",
            f"  A_0 (current)            : {self.jacobian_sparsity.get('A_0', 0.0):.4f}",
            f"  A_- (lag)                : {self.jacobian_sparsity.get('A_minus', 0.0):.4f}",
            f"  B_u (shocks)             : {self.jacobian_sparsity.get('B_u', 0.0):.4f}",
            "",
            f"Lead-Lag Incidence Matrix ({self.lead_lag_incidence.shape[0]} x {self.lead_lag_incidence.shape[1]}):",
        ]
        if len(self.variable_names) > 0:
            header = "         " + "".join(f"{v:>8}" for v in self.variable_names)
            lines.append(header)
            row_labels = ["t-1 (lag)", "t   (cur)", "t+1 (lead)"]
            for r_idx, label in enumerate(row_labels):
                row_str = "".join(
                    f"{int(self.lead_lag_incidence[r_idx, c]):>8d}"
                    for c in range(len(self.variable_names))
                )
                lines.append(f"  {label}: {row_str}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Render markdown table."""
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render LaTeX tabular."""
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render Typst table."""
        return _df_to_typst(self.to_frame(), **kwargs)


def detect_linear_model(model_dag: Any) -> bool:
    """Determine whether a DSGE model is linear in dynamic variables and shocks.

    Inspects AST equations by evaluating whether second symbolic derivatives
    with respect to all pairs of dynamic coordinates are identically zero (Const(0)).

    Parameters
    ----------
    model_dag : ParsedModelDAG or LinearModel or Sequence[Node]
        Parsed model representation.

    Returns
    -------
    bool
        True if all equations are linear/affine in dynamic variables and shocks.
    """
    if hasattr(model_dag, "is_linear") and getattr(
        model_dag, "_is_explicitly_linear", False
    ):
        return True

    equations: list[Node] = []
    if hasattr(model_dag, "equations"):
        equations = list(model_dag.equations)
    elif isinstance(model_dag, (list, tuple)):
        equations = list(model_dag)
    else:
        # If it's a LinearModel from puremacro.dsge.build, it is linear by definition
        if hasattr(model_dag, "solution") and hasattr(model_dag, "states"):
            return True
        return False

    if not equations:
        return True

    for eq in equations:
        if not isinstance(eq, Node):
            continue
        coords = list(eq.variables())
        if not coords:
            continue
        for v1, l1 in coords:
            d1 = eq.diff(v1, l1).simplify()
            d1_vars = d1.variables()
            if not d1_vars:
                continue
            for v2, l2 in d1_vars:
                d2 = d1.diff(v2, l2).simplify()
                if not (isinstance(d2, Const) and d2.value == 0):
                    return False
    return True


detect_linearity = detect_linear_model


def model_info(model_or_dag: Any) -> ModelInfoResult:
    """Compute comprehensive structural summary and diagnostic schema for a DSGE model.

    Parameters
    ----------
    model_or_dag : ParsedModelDAG or LinearModel
        Parsed model AST DAG or solved LinearModel.

    Returns
    -------
    ModelInfoResult
        Frozen dataclass with variable counts, dynamic classifications,
        lead/lag incidence matrix, and Jacobian sparsity metrics.
    """
    var_names: tuple[str, ...] = ()
    shock_names: tuple[str, ...] = ()
    param_names: tuple[str, ...] = ()
    equations: list[Node] = []
    eq_tags: dict[int, dict[str, str]] = {}
    predet_vars: list[str] = []

    if hasattr(model_or_dag, "variables"):
        var_names = tuple(model_or_dag.variables)
    if hasattr(model_or_dag, "shocks"):
        shock_names = tuple(model_or_dag.shocks)
    if hasattr(model_or_dag, "parameters"):
        param_names = tuple(model_or_dag.parameters)
    elif hasattr(model_or_dag, "params"):
        param_names = tuple(model_or_dag.params)
    if hasattr(model_or_dag, "equations"):
        equations = list(model_or_dag.equations)
    if hasattr(model_or_dag, "equation_tags"):
        eq_tags = dict(model_or_dag.equation_tags)
    if hasattr(model_or_dag, "predetermined_variables") and model_or_dag.predetermined_variables:
        predet_vars = list(model_or_dag.predetermined_variables)

    num_equations = len(equations) if equations else len(var_names)
    num_variables = len(var_names)
    num_shocks = len(shock_names)
    num_parameters = len(param_names)

    # Dynamic classifications
    purely_static: list[str] = []
    purely_forward: list[str] = []
    purely_backward: list[str] = []
    mixed: list[str] = []
    states_detected: list[str] = []

    if equations:
        all_coords: set[tuple[str, int]] = set()
        for eq in equations:
            if isinstance(eq, Node):
                all_coords.update(eq.variables())

        for v in var_names:
            v_coords = [l for (name, l) in all_coords if name == v]
            has_lag = any(l < 0 for l in v_coords)
            has_lead = any(l > 0 for l in v_coords)
            has_curr = any(l == 0 for l in v_coords)

            if has_lag or v in predet_vars:
                states_detected.append(v)

            if has_lag and has_lead:
                mixed.append(v)
            elif has_lag and not has_lead:
                purely_backward.append(v)
            elif not has_lag and has_lead:
                purely_forward.append(v)
            else:
                purely_static.append(v)
    else:
        # Fallback if model_or_dag is a LinearModel with states/controls
        if hasattr(model_or_dag, "states") and model_or_dag.states is not None:
            states_detected = list(model_or_dag.states)
            purely_backward = list(model_or_dag.states)
        if hasattr(model_or_dag, "controls") and model_or_dag.controls is not None:
            purely_forward = list(model_or_dag.controls)

    states = tuple(states_detected)
    controls = tuple(v for v in var_names if v not in set(states))

    # Lead/lag incidence matrix: 3 x N integer matrix
    N = num_variables
    lead_lag_incidence = np.zeros((3, N), dtype=int)
    k = 1

    if equations:
        # Row 0: lags (-1)
        for j, v in enumerate(var_names):
            if any((v, -1) in eq.variables() for eq in equations if isinstance(eq, Node)):
                lead_lag_incidence[0, j] = k
                k += 1
        # Row 1: current (0)
        for j, v in enumerate(var_names):
            if any((v, 0) in eq.variables() for eq in equations if isinstance(eq, Node)):
                lead_lag_incidence[1, j] = k
                k += 1
        # Row 2: leads (+1)
        for j, v in enumerate(var_names):
            if any((v, 1) in eq.variables() for eq in equations if isinstance(eq, Node)):
                lead_lag_incidence[2, j] = k
                k += 1
    else:
        # Fallback using states and controls
        states_set = set(states)
        for j, v in enumerate(var_names):
            if v in states_set:
                lead_lag_incidence[0, j] = k
                k += 1
        for j, v in enumerate(var_names):
            lead_lag_incidence[1, j] = k
            k += 1
        for j, v in enumerate(var_names):
            if v not in states_set:
                lead_lag_incidence[2, j] = k
                k += 1

    # Jacobian sparsity (non-zero density)
    M = num_equations
    K_e = num_shocks
    sparsity: dict[str, float] = {}

    if equations and M > 0:
        nnz_plus = sum(
            1
            for eq in equations
            for v in var_names
            if isinstance(eq, Node) and (v, 1) in eq.variables()
        )
        nnz_0 = sum(
            1
            for eq in equations
            for v in var_names
            if isinstance(eq, Node) and (v, 0) in eq.variables()
        )
        nnz_minus = sum(
            1
            for eq in equations
            for v in var_names
            if isinstance(eq, Node) and (v, -1) in eq.variables()
        )
        nnz_u = sum(
            1
            for eq in equations
            for e in shock_names
            if isinstance(eq, Node) and (e, 0) in eq.variables()
        )

        sparsity["A_plus"] = float(nnz_plus / (M * N)) if (M * N) > 0 else 0.0
        sparsity["A_0"] = float(nnz_0 / (M * N)) if (M * N) > 0 else 0.0
        sparsity["A_minus"] = float(nnz_minus / (M * N)) if (M * N) > 0 else 0.0
        sparsity["B_u"] = float(nnz_u / (M * K_e)) if (M * K_e) > 0 else 0.0
    else:
        sparsity["A_plus"] = 0.0
        sparsity["A_0"] = 0.0
        sparsity["A_minus"] = 0.0
        sparsity["B_u"] = 0.0

    is_linear = detect_linear_model(model_or_dag)

    return ModelInfoResult(
        num_equations=num_equations,
        num_variables=num_variables,
        num_shocks=num_shocks,
        num_parameters=num_parameters,
        variable_names=var_names,
        shock_names=shock_names,
        parameter_names=param_names,
        states=states,
        controls=controls,
        purely_static=tuple(purely_static),
        purely_forward=tuple(purely_forward),
        purely_backward=tuple(purely_backward),
        mixed=tuple(mixed),
        lead_lag_incidence=lead_lag_incidence,
        jacobian_sparsity=sparsity,
        is_linear=is_linear,
        equation_tags=eq_tags,
    )


def write_latex_dynamic_model(
    model_or_dag: Any,
    filepath: Path | str | None = None,
    write_equation_tags: bool = True,
) -> str:
    """Emit publication-grade LaTeX equations from a parsed DSGE model DAG.

    Renders dynamic subscripts (y_{j, t+1}, y_{j, t}, y_{j, t-1}, u_{m, t}),
    math formatting (\\frac, powers, \\exp, \\log), Greek parameter symbols,
    and equation tags (\\tag{...}).

    Parameters
    ----------
    model_or_dag : ParsedModelDAG
        Parsed model expression graph.
    filepath : Path or str, optional
        File destination path. If provided, writes the LaTeX string encoded as UTF-8.
    write_equation_tags : bool, default True
        Whether to attach \\tag{...} labels from model equation tags.

    Returns
    -------
    str
        Complete LaTeX align block containing formatted equations.
    """
    symbol_map: dict[str, str] = dict(GREEK_SYMBOLS)

    # Process parameter names and compound names like phi_pi
    param_names = []
    if hasattr(model_or_dag, "parameters"):
        param_names = list(model_or_dag.parameters)
    elif hasattr(model_or_dag, "params"):
        param_names = list(model_or_dag.params)

    for p in param_names:
        if p in GREEK_SYMBOLS:
            symbol_map[p] = GREEK_SYMBOLS[p]
        elif "_" in p:
            parts = p.split("_")
            lead = GREEK_SYMBOLS.get(parts[0], parts[0])
            sub = "_".join(parts[1:])
            sub_tex = GREEK_SYMBOLS.get(sub, sub)
            symbol_map[p] = f"{lead}_{{{sub_tex}}}"

    # Process TeX labels if declared in the model
    if hasattr(model_or_dag, "tex_labels") and model_or_dag.tex_labels:
        for sym, label in model_or_dag.tex_labels.items():
            clean_label = label.strip("$").strip()
            symbol_map[sym] = clean_label

    equations: list[Node] = []
    if hasattr(model_or_dag, "equations"):
        equations = list(model_or_dag.equations)

    eq_tags = getattr(model_or_dag, "equation_tags", {})

    eq_lines: list[str] = []
    for i, eq in enumerate(equations):
        if isinstance(eq, BinOp) and eq.op == "-":
            lhs_tex = eq.left.to_latex(symbol_map)
            rhs_tex = eq.right.to_latex(symbol_map)
            eq_str = f"{lhs_tex} &= {rhs_tex}"
        elif isinstance(eq, Node):
            eq_tex = eq.to_latex(symbol_map)
            eq_str = f"{eq_tex} &= 0"
        else:
            eq_str = f"{eq} &= 0"

        # Check equation tag
        if write_equation_tags and i in eq_tags:
            tag_name = eq_tags[i].get("name")
            if tag_name:
                eq_str += f" \\tag{{{tag_name}}}"

        eq_lines.append(eq_str)

    lines = [
        "% \x08egin{align}",
        r"\begin{align}",
    ]
    for i, line in enumerate(eq_lines):
        sep = r" \\" if i < len(eq_lines) - 1 else ""
        lines.append(f"  {line}{sep}")
    lines.append(r"\end{align}")
    latex_result = "\n".join(lines)

    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(latex_result, encoding="utf-8")

    return latex_result
