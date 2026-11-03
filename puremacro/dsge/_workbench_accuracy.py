"""Presentation of separate, scoped DSGE accuracy checks."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from puremacro.validation._audit_report import _jsonable
from ._workbench_state import DSGEFailure
from ._workbench_html import badge, panel

_DIMENSIONS = ("solver", "equations", "determinacy", "feasibility", "household_grid",
               "horizon", "quadrature", "compression", "branching", "derivatives")
_ACTIONS = {
    "solver": "Inspect the stopping reason and retained iteration history.",
    "equations": "Inspect the largest residual in its original units and at its reported location.",
    "determinacy": "Inspect the roots and variable timing at this calibration.",
    "feasibility": "Inspect budgets, admissible choices and probability mass.",
    "household_grid": "Compare household grids and off-grid Euler/KKT errors.",
    "horizon": "Compare longer horizons at the same branching/integration settings.",
    "quadrature": "Compare integration orders at the same domain and horizon.",
    "compression": "Compare retained ranks and omitted equations on the same experiment.",
    "branching": "Compare branch depths at a fixed horizon and quadrature order.",
    "derivatives": "Compare derivatives against an independent directional calculation.",
}


@dataclass(frozen=True)
class DSGEAccuracyDashboard:
    """Immutable evidence snapshot; deliberately has no overall pass flag."""

    _json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict()["checks"])

    def summary(self) -> str:
        frame = self.to_frame()
        counts = frame.status.value_counts().to_dict()
        return "Accuracy evidence (separate scopes): " + ", ".join(f"{k}={v}" for k, v in counts.items())

    def _repr_html_(self) -> str:
        return panel(self._html_body())

    def _html_body(self) -> str:
        data = self.to_dict()
        rows = []
        def number(value):
            if value is None:
                return "—"
            if isinstance(value, float):
                return f"{value:.5g}"
            return str(value)
        for row in data["checks"]:
            details = ('<details><summary>' + escape(row['check'].replace('_', ' '))
                       + '</summary><p class="pm-meta">Evidence: ' + escape(row['evidence'])
                       + '</p><p>' + escape(row['detail']) + '</p><p>'
                       + escape(row['next_step']) + '</p></details>')
            cells = [escape(row['dimension'].replace('_', ' ')), details, badge(row['status']),
                     escape(number(row['value'])), escape(number(row['tolerance'])), escape(row['units'])]
            rows.append('<tr>' + ''.join('<td>' + cell + '</td>' for cell in cells) + '</tr>')
        return ('<h3>Accuracy evidence</h3><p>' + escape(data["scope"])
                + '</p><p class="pm-meta">Expand a check for its evidence and next diagnostic. '
                'Reported values and unchecked dimensions are not accuracy certificates.</p>'
                '<div class="pm-scroll"><table><thead><tr>'
                + ''.join('<th>' + name + '</th>' for name in ('Dimension', 'Check', 'Status', 'Value', 'Tolerance', 'Units'))
                + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')

    def plot(self, ax=None):
        """A labeled status plot; color is redundant with text and symbols."""
        import matplotlib.pyplot as plt
        frame = self.to_frame()
        if ax is None:
            _, ax = plt.subplots(figsize=(11, max(4, .34 * len(frame))))
        styles = {"passed": ("#28754b", "o"), "failed": ("#b42b31", "x"),
                  "reported": ("#8b6508", "D"), "not_checked": ("#666666", "s"),
                  "unsupported": ("#666666", "s")}
        for i, row in frame.iterrows():
            color, marker = styles.get(row.status, ("#666666", "s"))
            ax.scatter([0], [i], c=color, marker=marker)
            ax.text(.03, i, f"{row.status}: {row['check']}", va="center", fontsize=9)
        ax.set_yticks(range(len(frame)), frame.dimension)
        ax.invert_yaxis()
        ax.set_xlim(-.02, 1)
        ax.set_xticks([])
        ax.set_title("Accuracy checks — each row has a separate scope")
        return ax


def solution_accuracy(result: Any, *, model: Any = None,
                      tolerances: Mapping[str, float] | None = None,
                      failure: DSGEFailure | None = None) -> DSGEAccuracyDashboard:
    """Inspect existing evidence, reusing ``audit_result`` where supported.

    No re-solving, differentiation or refinement is performed. For recursive
    HANK objects, diagnostics are explicitly labeled recorded solver evidence.
    Euler and compression values without an explicit tolerance stay reported.
    Higher-order objects expose first-order checks with that scope stated.
    """
    from puremacro.validation import audit_result
    from .build import LinearModel
    from .perfect_foresight import PerfectForesightResult, MCPResult
    from .transitions import TransitionResult
    from .global_solution import TimeIterationResult

    thresholds = dict(tolerances or {})
    allowed = {"equations", "euler", "compression", "mass", "budget"}
    if set(thresholds) - allowed or any(not np.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError(f"tolerances require finite nonnegative values for {sorted(allowed)}")
    rows: list[dict[str, Any]] = []
    scope = "No current solved result."

    def add(dimension, check, *, value=None, tolerance=None, status="not_checked",
            units="", evidence="not_evaluated", detail=""):
        if status == "measure":
            status = ("passed" if value is not None and np.isfinite(value) and value <= tolerance else "failed")
        rows.append(dict(dimension=dimension, check=check, value=_jsonable(value),
                         tolerance=tolerance, units=units, status=status, evidence=evidence,
                         detail=str(detail), next_step=_ACTIONS.get(dimension, "Inspect the original failure.")))

    if result is not None:
        linear = result if isinstance(result, LinearModel) else getattr(result, "first_order", None)
        audited = linear if isinstance(linear, LinearModel) else result
        if isinstance(result, TransitionResult) and result.solution is not None:
            audited = result.solution
        try:
            report = audit_result(audited, model=model,
                                  tolerances={k: v for k, v in thresholds.items() if k in {"equations", "euler", "mass"}})
            for row in report.records:
                dimension = {"distribution": "feasibility", "refinement": "horizon"}.get(row.category, row.category)
                if row.status == "unsupported":
                    continue
                add(dimension, row.check, value=row.value, tolerance=row.tolerance,
                    status=row.status, units=row.units, evidence=row.evidence_source,
                    detail=json.dumps(_jsonable(dict(row.details)), sort_keys=True))
        except Exception as exc:
            add("equations", "audit_evaluation", status="failed", evidence="audit_result",
                detail=f"{type(exc).__name__}: {exc}")
        if isinstance(linear, LinearModel):
            scope = "First-order decision rule and deterministic steady state; no global stochastic accuracy claim."
            if linear is not result:
                scope += " Only the embedded first-order model was audited; higher-order accuracy is unchecked."
            add("determinacy", "existence_uniqueness", value=bool(linear.is_determinate),
                status="passed" if linear.is_determinate else "failed", evidence="Klein solution eu",
                detail=str(linear.solution.eu))
        elif isinstance(result, (PerfectForesightResult, MCPResult)):
            scope = "Finite deterministic boundary problem; terminal distance is not a horizon-refinement test."
            if isinstance(result, MCPResult):
                add("solver", "mcp_converged", value=bool(result.converged),
                    status="passed" if result.converged else "failed", evidence="solver_record")
                add("feasibility", "physical_complementarity_residual", value=result.physical_complementarity_residual,
                    status="reported", evidence="solver_record", units="variable times equation units")
            add("horizon", "terminal_distance", value=result.terminal_error, status="reported",
                units="endogenous level units", evidence="solver_record")
        elif isinstance(result, TransitionResult):
            scope = "Deterministic transition with separately recorded continuation and horizon comparisons."
            add("solver", "target_transition_converged", value=bool(result.converged),
                status="passed" if result.converged else "failed", evidence="transition_record", detail=result.reason)
            for i, comparison in enumerate(result.horizon_history):
                add("horizon", f"horizon_comparison_{i + 1}", status="reported", evidence="transition_record",
                    detail=json.dumps(_jsonable(comparison), sort_keys=True))
        elif isinstance(result, TimeIterationResult):
            scope = "Global policy iteration on a declared finite domain; update convergence alone is not Euler accuracy."
            add("solver", "iteration_converged", value=bool(result.converged),
                status="passed" if result.converged else "failed", evidence="solver_record")
            add("equations", "policy_kkt_residual", value=result.audit.max_kkt_residual,
                tolerance=result.audit.tolerance, status="measure", units="scaled KKT residual",
                evidence="saved_policy_audit", detail="Audit points and integration rule are those of the saved policy audit; no grid refinement performed.")
            add("feasibility", "policy_audit_accepted", value=bool(result.audit.passed),
                status="passed" if result.audit.passed else "failed", evidence="saved_policy_audit",
                detail=f"require_domain={result.audit.require_domain}; includes pointwise feasibility and domain criteria.")
        elif type(result).__module__.startswith("puremacro.models.hank"):
            scope = "Recorded HANK solver evidence; local policies, household grids and finite trees have separate accuracy limits."

        diagnostics = getattr(result, "diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            diagnostics = getattr(result, "metadata", {})
        # Only interpret documented HANK diagnostics, not arbitrary attributes
        # on caller objects that happen to have similarly named fields.
        if type(result).__module__.startswith("puremacro.models.hank") and isinstance(diagnostics, Mapping):
            d = diagnostics
            if isinstance(getattr(result, "converged", None), (bool, np.bool_)):
                add("solver", "solver_converged", value=bool(result.converged),
                    status="passed" if result.converged else "failed", evidence="solver_record",
                    detail="Solver declaration only; inspect all other dimensions.")
            for key, dim in (("equation_converged", "equations"), ("horizon_converged", "horizon"),
                             ("branching_converged", "branching"), ("quadrature_order_converged", "quadrature")):
                value = d.get(key)
                if value is not None:
                    add(dim, key, value=bool(value), status="passed" if value else "failed",
                        evidence="solver_record", detail="Recorded acceptance criterion, not an independent new check.")
            metrics = (
                ("market_residual_max_abs", "equations", "equations", "market equation units"),
                ("projected_equilibrium_residual_max_abs", "equations", "equations", "projected equation units"),
                ("euler_kkt_max", "household_grid", "euler", "relative marginal-utility residual"),
                ("household_budget_error", "feasibility", "budget", "household budget units"),
                ("mass_error", "feasibility", "mass", "probability mass"),
                ("terminal_compression_l1", "compression", "compression", "distribution L1"),
                ("full_marginal_stationary_relative_rms_max", "compression", "compression", "normalized marginal residual RMS"),
            )
            for key, dim, threshold, units in metrics:
                value = d.get(key)
                if value is not None:
                    tol = thresholds.get(threshold)
                    add(dim, key, value=float(value), tolerance=tol,
                        status="reported" if tol is None else "measure", units=units,
                        evidence="solver_record", detail="An explicit displayed tolerance is required for a numerical pass.")
            if "completed_periods" in d:
                add("solver", "completed_periods", value=d["completed_periods"], status="reported",
                    units="dates", evidence="solver_record", detail=f"Requested: {d.get('requested_periods', 'unknown')}")

    if failure is not None:
        add("solver", failure.category, status="failed", evidence=failure.exception_type,
            detail=failure.message + " " + failure.next_step)
    represented = {row["dimension"] for row in rows}
    for dimension in _DIMENSIONS:
        if dimension not in represented:
            add(dimension, dimension, detail="No check for this dimension was supplied or performed.")
    payload = dict(schema_version=1, scope=scope, checks=rows)
    return DSGEAccuracyDashboard(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False))
