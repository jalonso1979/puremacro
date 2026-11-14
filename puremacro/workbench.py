"""Pyodide-safe unified empirical-to-structural research workbench.

Composes panel preflight diagnostics, empirical Local Projection estimation and
specification sensitivity grids, multi-model IRF comparisons, optional
structural moment-target scoring, prior-posterior diagnostics, and explicit
solver convergence/accuracy/support diagnostics into an immutable
:class:`EmpiricalStructuralDossier`.

Module top-level imports are restricted to the standard library and the core
numerical stack so importing this module triggers zero optional or solver
dependencies under Pyodide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import math
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from puremacro.data import PreflightReport
    from puremacro.dsge.validation_scorecard import (
        IRFScorecardResult,
        PriorPosteriorComparisonResult,
    )
    from puremacro.reports import IRFComparisonResult
    from puremacro.solvers.diagnostics import (
        AccuracyStatus,
        ConvergenceStatus,
        SolverDiagnosticRecord,
    )
    from puremacro.structural.bridge import MomentTargets


_VALID_LANGS: tuple[str, str] = ("en", "es")


def _resolve_lang(lang: str) -> Literal["en", "es"]:
    if lang not in _VALID_LANGS:
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    return "es" if lang == "es" else "en"


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/pandas/dataclass structures into RFC 8259 JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, np.ndarray):
        return [_json_sanitize(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _json_sanitize(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_sanitize(value.tolist())
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_sanitize(value.to_dict())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _fmt_num(val: float, digits: int = 3) -> str:
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{digits}f}"


def _comparison_to_dict(cmp: IRFComparisonResult) -> dict[str, Any]:
    return {
        "baseline": cmp.baseline,
        "horizons": list(cmp.horizons),
        "metrics": _json_sanitize(cmp.metrics),
        "table": _json_sanitize(cmp.table),
        "warnings": list(cmp.warnings),
        "errors": dict(cmp.errors),
    }


def _scorecard_to_dict(sc: IRFScorecardResult) -> dict[str, Any]:
    return {
        "rmse": _json_sanitize(sc.rmse),
        "mae": _json_sanitize(sc.mae),
        "wse": _json_sanitize(sc.wse),
        "chi2_stat": _json_sanitize(sc.chi2_stat),
        "chi2_df": int(sc.chi2_df),
        "chi2_pvalue": _json_sanitize(sc.chi2_pvalue),
        "sign_concordance": _json_sanitize(sc.sign_concordance),
        "band_coverage": _json_sanitize(sc.band_coverage),
        "determinacy_rate": _json_sanitize(sc.determinacy_rate),
        "weight_mode_used": sc.weight_mode_used,
        "covariance_condition_number": _json_sanitize(sc.covariance_condition_number),
        "metrics": _json_sanitize(sc.metrics),
        "table": _json_sanitize(sc.table),
        "warnings": list(sc.warnings),
    }


def _prior_post_to_dict(pp: PriorPosteriorComparisonResult) -> dict[str, Any]:
    df = pp.table
    n_weak = int(df["weakly_identified"].sum()) if "weakly_identified" in df.columns else 0
    n_bnd = int(df["near_boundary"].sum()) if "near_boundary" in df.columns else 0
    n_unconv = (
        int(df["unconverged_or_divergent"].sum())
        if "unconverged_or_divergent" in df.columns
        else 0
    )
    return {
        "model_name": pp.model_name,
        "ci_level": float(pp.ci_level),
        "n_chains": int(pp.n_chains),
        "n_draws": int(pp.n_draws),
        "n_params": int(len(df)),
        "flags_summary": {
            "weakly_identified": n_weak,
            "near_boundary": n_bnd,
            "unconverged_or_divergent": n_unconv,
        },
        "table": _json_sanitize(df),
        "warnings": list(pp.warnings),
    }


@dataclass(frozen=True)
class EmpiricalStructuralDossier:
    """Immutable end-to-end empirical and structural validation dossier.

    Bundles panel preflight checks, empirical LP estimates, sensitivity grid
    comparisons, optional structural moment targets, IRF validation scorecards,
    prior-posterior diagnostics, and explicit solver diagnostics with bilingual
    (``'en'`` / ``'es'``) summaries and Markdown/LaTeX/Typst/JSON exports.
    """

    y: str
    x: str
    controls: tuple[str, ...]
    horizons: tuple[int, ...]
    lags: int
    ci_level: float
    lang: Literal["en", "es"]
    title: str
    preflight: PreflightReport
    empirical_lp: Any
    sensitivity: IRFComparisonResult | None
    irf_comparison: IRFComparisonResult
    moment_targets: MomentTargets | None = None
    prior_posterior: PriorPosteriorComparisonResult | None = None
    irf_scorecard: IRFScorecardResult | None = None
    solver_diagnostics: SolverDiagnosticRecord | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    digits: int = 3

    def __post_init__(self) -> None:
        _resolve_lang(self.lang)
        if not isinstance(self.controls, tuple):
            object.__setattr__(self, "controls", tuple(str(c) for c in self.controls))
        if not isinstance(self.horizons, tuple):
            object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))
        if int(self.digits) < 0:
            raise ValueError(f"digits must be non-negative, got {self.digits!r}")
        object.__setattr__(self, "digits", int(self.digits))

    @property
    def passed(self) -> bool:
        """True only when preflight passed and any solver diagnostics are PASS."""
        solver_ok = (
            self.solver_diagnostics.is_pass
            if self.solver_diagnostics is not None
            else True
        )
        return bool(self.preflight.passed and solver_ok)

    @property
    def overall_status(self) -> str:
        """Coarse dossier verdict separating preflight, solver accuracy, and warnings."""
        if not self.preflight.passed:
            return "PREFLIGHT_FAILED"
        if self.solver_diagnostics is not None and not self.solver_diagnostics.is_pass:
            return str(self.solver_diagnostics.overall_status)
        if self.preflight.status == "warn" or bool(self.warnings):
            return "PASS_WITH_WARNINGS"
        return "PASS"

    @property
    def convergence_status(self) -> ConvergenceStatus | None:
        """Numerical solver convergence record when solver diagnostics are attached."""
        return (
            self.solver_diagnostics.convergence
            if self.solver_diagnostics is not None
            else None
        )

    @property
    def accuracy_status(self) -> AccuracyStatus | None:
        """Economic equation accuracy record when solver diagnostics are attached."""
        return (
            self.solver_diagnostics.accuracy
            if self.solver_diagnostics is not None
            else None
        )

    def _baseline_metric_row(self) -> Mapping[str, Any]:
        metrics = self.irf_comparison.metrics
        if metrics.empty:
            return {}
        matched = metrics[metrics["baseline"] == True]  # noqa: E712
        row = matched.iloc[0] if not matched.empty else metrics.iloc[0]
        return row.to_dict()

    def _overview_rows(self, lang: Literal["en", "es"]) -> list[dict[str, str]]:
        is_es = lang == "es"
        base_row = self._baseline_metric_row()
        peak_h = str(base_row.get("peak_h", "-"))
        peak_val = _fmt_num(float(base_row.get("peak_response", float("nan"))), self.digits)
        n_sig = str(base_row.get("n_sig", "-"))
        rows: list[dict[str, str]] = [
            {
                "stage": "Preflight" if not is_es else "Diagnóstico previo",
                "status": self.preflight.status.upper(),
                "summary": (
                    f"shape={self.preflight.panel_shape}, units={self.preflight.n_units}, "
                    f"periods={self.preflight.n_periods}, errors={self.preflight.n_errors}, "
                    f"warnings={self.preflight.n_warnings}"
                ),
            },
            {
                "stage": "Empirical LP" if not is_es else "LP empírica",
                "status": "OK",
                "summary": (
                    f"{self.y} <- {self.x} (lags={self.lags}, peak_h={peak_h}, "
                    f"peak={peak_val}, n_sig={n_sig})"
                ),
            },
            {
                "stage": "IRF Comparison" if not is_es else "Comparación IRF",
                "status": f"{len(self.irf_comparison.metrics)} specs",
                "summary": (
                    f"baseline={self.irf_comparison.baseline}, "
                    f"horizons=[{min(self.irf_comparison.horizons)}..{max(self.irf_comparison.horizons)}]"
                    if self.irf_comparison.horizons
                    else f"baseline={self.irf_comparison.baseline}"
                ),
            },
        ]
        if self.sensitivity is not None:
            rows.append(
                {
                    "stage": "LP Sensitivity" if not is_es else "Sensibilidad LP",
                    "status": f"{len(self.sensitivity.metrics)} specs",
                    "summary": (
                        f"baseline={self.sensitivity.baseline}, "
                        f"skipped={len(self.sensitivity.errors)}"
                    ),
                }
            )
        if self.irf_scorecard is not None:
            sc = self.irf_scorecard
            rows.append(
                {
                    "stage": "Structural Scorecard" if not is_es else "Evaluación estructural",
                    "status": sc.weight_mode_used.upper(),
                    "summary": (
                        f"RMSE={_fmt_num(sc.rmse, self.digits)}, "
                        f"MAE={_fmt_num(sc.mae, self.digits)}, "
                        f"sign={_fmt_num(sc.sign_concordance, self.digits)}, "
                        f"cov={_fmt_num(sc.band_coverage, self.digits)}"
                    ),
                }
            )
        if self.prior_posterior is not None:
            pp = self.prior_posterior
            n_flag = int(
                (
                    pp.table["weakly_identified"]
                    | pp.table["near_boundary"]
                    | pp.table["unconverged_or_divergent"]
                ).sum()
            )
            rows.append(
                {
                    "stage": "Prior-Posterior" if not is_es else "Previo-Posterior",
                    "status": "FLAGGED" if n_flag > 0 else "OK",
                    "summary": (
                        f"params={len(pp.table)}, chains={pp.n_chains}x{pp.n_draws}, flagged={n_flag}"
                    ),
                }
            )
        if self.solver_diagnostics is not None:
            sd = self.solver_diagnostics
            rows.append(
                {
                    "stage": "Solver Diagnostics" if not is_es else "Diagnóstico del solucionador",
                    "status": sd.overall_status,
                    "summary": (
                        f"converged={sd.convergence.converged}, "
                        f"accurate={sd.accuracy.is_accurate} (max_res={sd.accuracy.max_residual:.3e}), "
                        f"supported={sd.support.is_supported} ({sd.support.regime})"
                    ),
                }
            )
        return rows

    def summary(self, lang: str | None = None) -> str:
        """Render a concise bilingual (``'en'`` or ``'es'``) dossier summary."""
        use_lang = self.lang if lang is None else _resolve_lang(lang)
        pf = self.preflight
        base_row = self._baseline_metric_row()
        peak_h = base_row.get("peak_h", "-")
        peak_val = _fmt_num(float(base_row.get("peak_response", float("nan"))), self.digits)
        n_sig = base_row.get("n_sig", 0)
        h_span = (
            f"[{min(self.horizons)}..{max(self.horizons)}]" if self.horizons else "[]"
        )

        lines: list[str] = []
        if use_lang == "es":
            lines.append(
                f"{self.title} [{self.overall_status}] | sha256={pf.data_sha256[:12]}..."
            )
            lines.append(
                f"- Diagnóstico previo: estado={pf.status.upper()}, filas={pf.panel_shape[0]}, "
                f"unidades={pf.n_units}, periodos={pf.n_periods}, frecuencia={pf.inferred_freq or 'n/a'}, "
                f"balanceado={pf.balanced}, errores={pf.n_errors}, advertencias={pf.n_warnings}."
            )
            lines.append(
                f"- LP empírica y comparación: {self.y} <- {self.x}, rezagos={self.lags}, "
                f"horizontes={h_span}, IC={int(round(100 * self.ci_level))}%, "
                f"pico en h={peak_h} ({peak_val}), horizontes significativos={n_sig}, "
                f"especificaciones comparadas={len(self.irf_comparison.metrics)}."
            )
            if self.sensitivity is not None:
                lines.append(
                    f"- Malla de sensibilidad LP: {len(self.sensitivity.metrics)} especificaciones "
                    f"(referencia={self.sensitivity.baseline!r}, omitidas={len(self.sensitivity.errors)})."
                )
            if self.irf_scorecard is not None:
                sc = self.irf_scorecard
                wald = (
                    f"chi2({sc.chi2_df})={_fmt_num(sc.chi2_stat, self.digits)}, p={sc.chi2_pvalue:.4g}"
                    if sc.chi2_df > 0 and math.isfinite(sc.chi2_stat) and math.isfinite(sc.chi2_pvalue)
                    else f"WSE={_fmt_num(sc.wse, self.digits)}"
                )
                lines.append(
                    f"- Evaluación estructural: momentos={len(sc.table)}, modo={sc.weight_mode_used}, "
                    f"RMSE={_fmt_num(sc.rmse, self.digits)}, MAE={_fmt_num(sc.mae, self.digits)}, "
                    f"{wald}, concordancia_signo={_fmt_num(sc.sign_concordance, self.digits)}, "
                    f"cobertura_banda={_fmt_num(sc.band_coverage, self.digits)}."
                )
            if self.prior_posterior is not None:
                pp = self.prior_posterior
                n_weak = int(pp.table["weakly_identified"].sum())
                n_bnd = int(pp.table["near_boundary"].sum())
                n_unc = int(pp.table["unconverged_or_divergent"].sum())
                lines.append(
                    f"- Previo-posterior: parámetros={len(pp.table)}, cadenas={pp.n_chains}x{pp.n_draws}, "
                    f"identificación_débil={n_weak}, cerca_frontera={n_bnd}, no_convergente={n_unc}."
                )
            if self.solver_diagnostics is not None:
                sd = self.solver_diagnostics
                lines.append(
                    f"- Diagnóstico del solucionador: estado={sd.overall_status}, "
                    f"convergió={sd.convergence.converged} ({sd.convergence.iterations} iters), "
                    f"preciso={sd.accuracy.is_accurate} (res_max={sd.accuracy.max_residual:.3e}), "
                    f"soportado={sd.support.is_supported} (régimen={sd.support.regime})."
                )
            if self.warnings:
                lines.append(f"- Advertencias ({len(self.warnings)}): " + "; ".join(self.warnings))
        else:
            lines.append(
                f"{self.title} [{self.overall_status}] | sha256={pf.data_sha256[:12]}..."
            )
            lines.append(
                f"- Preflight: status={pf.status.upper()}, rows={pf.panel_shape[0]}, "
                f"units={pf.n_units}, periods={pf.n_periods}, freq={pf.inferred_freq or 'n/a'}, "
                f"balanced={pf.balanced}, errors={pf.n_errors}, warnings={pf.n_warnings}."
            )
            lines.append(
                f"- Empirical LP & comparison: {self.y} <- {self.x}, lags={self.lags}, "
                f"horizons={h_span}, CI={int(round(100 * self.ci_level))}%, "
                f"peak at h={peak_h} ({peak_val}), significant horizons={n_sig}, "
                f"compared specs={len(self.irf_comparison.metrics)}."
            )
            if self.sensitivity is not None:
                lines.append(
                    f"- LP sensitivity grid: {len(self.sensitivity.metrics)} specs "
                    f"(baseline={self.sensitivity.baseline!r}, skipped={len(self.sensitivity.errors)})."
                )
            if self.irf_scorecard is not None:
                sc = self.irf_scorecard
                wald = (
                    f"chi2({sc.chi2_df})={_fmt_num(sc.chi2_stat, self.digits)}, p={sc.chi2_pvalue:.4g}"
                    if sc.chi2_df > 0 and math.isfinite(sc.chi2_stat) and math.isfinite(sc.chi2_pvalue)
                    else f"WSE={_fmt_num(sc.wse, self.digits)}"
                )
                lines.append(
                    f"- Structural scorecard: moments={len(sc.table)}, mode={sc.weight_mode_used}, "
                    f"RMSE={_fmt_num(sc.rmse, self.digits)}, MAE={_fmt_num(sc.mae, self.digits)}, "
                    f"{wald}, sign_concordance={_fmt_num(sc.sign_concordance, self.digits)}, "
                    f"band_coverage={_fmt_num(sc.band_coverage, self.digits)}."
                )
            if self.prior_posterior is not None:
                pp = self.prior_posterior
                n_weak = int(pp.table["weakly_identified"].sum())
                n_bnd = int(pp.table["near_boundary"].sum())
                n_unc = int(pp.table["unconverged_or_divergent"].sum())
                lines.append(
                    f"- Prior-posterior: params={len(pp.table)}, chains={pp.n_chains}x{pp.n_draws}, "
                    f"weakly_identified={n_weak}, near_boundary={n_bnd}, unconverged={n_unc}."
                )
            if self.solver_diagnostics is not None:
                sd = self.solver_diagnostics
                lines.append(
                    f"- Solver diagnostics: status={sd.overall_status}, "
                    f"converged={sd.convergence.converged} ({sd.convergence.iterations} iters), "
                    f"accurate={sd.accuracy.is_accurate} (max_res={sd.accuracy.max_residual:.3e}), "
                    f"supported={sd.support.is_supported} (regime={sd.support.regime})."
                )
            if self.warnings:
                lines.append(f"- Warnings ({len(self.warnings)}): " + "; ".join(self.warnings))
        return "\n".join(lines)

    def to_frame(
        self,
        section: Literal[
            "overview",
            "preflight",
            "irf_metrics",
            "irf_table",
            "scorecard",
            "scorecard_metrics",
            "prior_posterior",
        ] = "overview",
        *,
        lang: str | None = None,
    ) -> pd.DataFrame:
        """Return one dossier section as a tidy pandas DataFrame."""
        use_lang = self.lang if lang is None else _resolve_lang(lang)
        if section == "overview":
            return pd.DataFrame(self._overview_rows(use_lang))
        if section == "preflight":
            return self.preflight.to_frame("variables")
        if section == "irf_metrics":
            return self.irf_comparison.to_frame("metrics")
        if section == "irf_table":
            return self.irf_comparison.to_frame("coef_se", digits=self.digits)
        if section == "scorecard":
            if self.irf_scorecard is None:
                raise ValueError("No structural IRF scorecard is attached to this dossier.")
            return self.irf_scorecard.to_frame("table")
        if section == "scorecard_metrics":
            if self.irf_scorecard is None:
                raise ValueError("No structural IRF scorecard is attached to this dossier.")
            return self.irf_scorecard.to_frame("metrics")
        if section == "prior_posterior":
            if self.prior_posterior is None:
                raise ValueError("No prior-posterior comparison is attached to this dossier.")
            return self.prior_posterior.to_frame("compact")
        raise ValueError(
            f"Unknown section {section!r}; expected one of "
            "'overview', 'preflight', 'irf_metrics', 'irf_table', "
            "'scorecard', 'scorecard_metrics', 'prior_posterior'."
        )

    def to_dataframe(
        self,
        section: Literal[
            "overview",
            "preflight",
            "irf_metrics",
            "irf_table",
            "scorecard",
            "scorecard_metrics",
            "prior_posterior",
        ] = "overview",
        *,
        lang: str | None = None,
    ) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(section=section, lang=lang)

    def _section_tables(
        self, lang: Literal["en", "es"]
    ) -> list[tuple[str, pd.DataFrame, bool]]:
        """Collect ``(heading, dataframe, include_index)`` pairs for exports."""
        is_es = lang == "es"
        sections: list[tuple[str, pd.DataFrame, bool]] = [
            (
                "Resumen del flujo" if is_es else "Pipeline Overview",
                pd.DataFrame(self._overview_rows(lang)),
                False,
            ),
            (
                "Diagnóstico previo de variables" if is_es else "Panel Preflight Variables",
                self.preflight.to_frame("variables"),
                False,
            ),
            (
                "Métricas de comparación IRF" if is_es else "IRF Comparison Metrics",
                self.irf_comparison.to_frame("metrics"),
                False,
            ),
            (
                "Tabla IRF (coef / ee)" if is_es else "IRF Comparison Table (coef / se)",
                self.irf_comparison.to_frame("coef_se", digits=self.digits),
                False,
            ),
        ]
        if self.sensitivity is not None and self.sensitivity is not self.irf_comparison:
            sections.append(
                (
                    "Malla de sensibilidad LP" if is_es else "LP Sensitivity Grid Metrics",
                    self.sensitivity.to_frame("metrics"),
                    False,
                )
            )
        if self.irf_scorecard is not None:
            sections.append(
                (
                    "Evaluación estructural IRF" if is_es else "Structural IRF Scorecard Metrics",
                    self.irf_scorecard.to_frame("metrics"),
                    False,
                )
            )
        if self.prior_posterior is not None:
            sections.append(
                (
                    "Diagnóstico previo-posterior" if is_es else "Prior-Posterior Comparison",
                    self.prior_posterior.to_frame("compact"),
                    False,
                )
            )
        if self.solver_diagnostics is not None:
            sd = self.solver_diagnostics
            sd_df = pd.DataFrame(
                [
                    {
                        "overall_status": sd.overall_status,
                        "converged": sd.convergence.converged,
                        "iterations": sd.convergence.iterations,
                        "solver_method": sd.convergence.solver_method,
                        "is_accurate": sd.accuracy.is_accurate,
                        "max_residual": sd.accuracy.max_residual,
                        "residual_norm": sd.accuracy.residual_norm,
                        "is_supported": sd.support.is_supported,
                        "regime": sd.support.regime,
                    }
                ]
            )
            sections.append(
                (
                    "Diagnósticos del solucionador" if is_es else "Solver Diagnostics",
                    sd_df,
                    False,
                )
            )
        return sections

    def to_markdown(
        self,
        *,
        lang: str | None = None,
        digits: int | None = None,
    ) -> str:
        """Render the complete dossier as GitHub-Flavored Markdown."""
        from puremacro.reports import df_to_markdown

        use_lang = self.lang if lang is None else _resolve_lang(lang)
        d = self.digits if digits is None else int(digits)
        lines: list[str] = [
            f"# {self.title}",
            "",
            self.summary(lang=use_lang),
            "",
        ]
        for heading, df, use_idx in self._section_tables(use_lang):
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(df_to_markdown(df, index=use_idx, digits=d))
            lines.append("")
        if self.preflight.recommendations:
            rec_head = (
                "Recomendaciones de preflight"
                if use_lang == "es"
                else "Preflight Recommendations"
            )
            lines.append(f"## {rec_head}")
            lines.append("")
            for rec in self.preflight.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_latex(
        self,
        *,
        lang: str | None = None,
        digits: int | None = None,
    ) -> str:
        """Render the complete dossier as LaTeX sections and tabular tables."""
        from puremacro.reports import df_to_latex, latex_escape

        use_lang = self.lang if lang is None else _resolve_lang(lang)
        d = self.digits if digits is None else int(digits)
        parts: list[str] = [
            f"\\section*{{{latex_escape(self.title)}}}",
            "",
            latex_escape(self.summary(lang=use_lang)).replace("\n", " \\\\\n"),
            "",
        ]
        for heading, df, use_idx in self._section_tables(use_lang):
            parts.append(f"\\subsection*{{{latex_escape(heading)}}}")
            parts.append(df_to_latex(df, index=use_idx, digits=d))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def to_typst(
        self,
        *,
        lang: str | None = None,
        digits: int | None = None,
    ) -> str:
        """Render the complete dossier as Typst headings and ``#table`` blocks."""
        from puremacro.reports import df_to_typst, typst_escape

        use_lang = self.lang if lang is None else _resolve_lang(lang)
        d = self.digits if digits is None else int(digits)
        parts: list[str] = [
            f"= {typst_escape(self.title)}",
            "",
            typst_escape(self.summary(lang=use_lang)),
            "",
        ]
        for heading, df, use_idx in self._section_tables(use_lang):
            parts.append(f"== {typst_escape(heading)}")
            parts.append(df_to_typst(df, index=use_idx, digits=d))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dossier into a portable JSON-safe dictionary."""
        import puremacro

        mt_dict: dict[str, Any] | None = None
        if self.moment_targets is not None:
            mt = self.moment_targets
            mt_dict = {
                "labels": list(mt.labels),
                "values": _json_sanitize(mt.values),
                "covariance": _json_sanitize(mt.covariance),
                "units": list(mt.units) if isinstance(mt.units, tuple) else str(mt.units),
                "metadata": _json_sanitize(dict(mt.metadata)),
            }

        return {
            "puremacro_version": puremacro.__version__,
            "title": self.title,
            "lang": self.lang,
            "overall_status": self.overall_status,
            "passed": self.passed,
            "specification": {
                "y": self.y,
                "x": self.x,
                "controls": list(self.controls),
                "horizons": list(self.horizons),
                "lags": int(self.lags),
                "ci_level": float(self.ci_level),
            },
            "summary": self.summary(lang=self.lang),
            "summaries": {
                "en": self.summary(lang="en"),
                "es": self.summary(lang="es"),
            },
            "preflight": json.loads(self.preflight.to_json()),
            "empirical_lp": _json_sanitize(
                pd.DataFrame(self.empirical_lp)
                if "h" in getattr(self.empirical_lp, "columns", ())
                else pd.DataFrame(self.empirical_lp).reset_index()
            ),
            "sensitivity": (
                _comparison_to_dict(self.sensitivity)
                if self.sensitivity is not None
                else None
            ),
            "irf_comparison": _comparison_to_dict(self.irf_comparison),
            "moment_targets": mt_dict,
            "irf_scorecard": (
                _scorecard_to_dict(self.irf_scorecard)
                if self.irf_scorecard is not None
                else None
            ),
            "prior_posterior": (
                _prior_post_to_dict(self.prior_posterior)
                if self.prior_posterior is not None
                else None
            ),
            "solver_diagnostics": (
                _json_sanitize(self.solver_diagnostics.to_dict())
                if self.solver_diagnostics is not None
                else None
            ),
            "warnings": list(self.warnings),
            "metadata": _json_sanitize(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the complete dossier to valid RFC 8259 JSON (NaN/Inf -> null)."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False, ensure_ascii=False)


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a sequence of non-negative integers, not a string.")
    hs = [int(h) for h in horizons]
    if not hs:
        raise ValueError("horizons must be non-empty.")
    if any(isinstance(h, bool) or h < 0 for h in horizons):
        raise ValueError(f"horizons must contain non-negative integers, got {list(horizons)!r}.")
    if len(set(hs)) != len(hs):
        raise ValueError(f"horizons must be unique, got {hs!r}.")
    return tuple(hs)


def _prepare_estimation_frame(
    data: pd.DataFrame,
    *,
    unit_col: str | None,
    time_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare a non-mutated estimation DataFrame and panel coordinate kwargs."""
    work = data.copy()
    if unit_col is not None:
        if isinstance(work.index, pd.MultiIndex) and work.index.nlevels >= 2:
            idx_names = list(work.index.names)
            if idx_names[0] is None or idx_names[1] is None:
                work.index = work.index.set_names([unit_col, time_col])
                idx_names = [unit_col, time_col]
            return work, {
                "entity_level": str(idx_names[0]),
                "time_level": str(idx_names[1]),
            }
        return work, {"unit_col": unit_col, "time_col": time_col}

    if isinstance(work.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return work, {}

    if time_col in work.columns:
        ser = work[time_col]
        is_num = (
            pd.api.types.is_integer_dtype(ser)
            or pd.api.types.is_float_dtype(ser)
            or pd.api.types.is_bool_dtype(ser)
        )
        if not is_num:
            num_try = pd.to_numeric(ser, errors="coerce")
            n_present = int(ser.notna().sum())
            is_num = n_present > 0 and int(num_try.notna().sum()) == n_present
        if not is_num:
            try:
                if isinstance(ser.dtype, pd.PeriodDtype):
                    dt = ser.dt.to_timestamp()
                else:
                    dt = pd.to_datetime(ser, format="mixed", errors="coerce")
                if dt.notna().all() and dt.is_unique:
                    order = np.argsort(dt.to_numpy())
                    work = work.iloc[order].copy()
                    dti = pd.DatetimeIndex(dt.iloc[order])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        inferred = pd.infer_freq(dti)
                    if inferred is not None:
                        work.index = dti
                    else:
                        work = work.reset_index(drop=True)
                    return work, {}
            except Exception:
                pass
        work = work.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    return work, {}


def _call_estimator_cleanly(
    estimator: Callable[..., Any],
    data: pd.DataFrame,
    y: str,
    x: str,
    *,
    horizons: tuple[int, ...],
    lags: int,
    controls: tuple[str, ...],
    ci: float,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke an LP estimator while respecting alias exclusivity and signatures."""
    sig = inspect.signature(estimator)
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs: dict[str, Any] = {}
    if extra_kwargs:
        for k, v in extra_kwargs.items():
            if accepts_var_kw or k in params:
                kwargs[k] = v

    if "x" in params:
        kwargs["x"] = x
    elif "treat" in params:
        kwargs["treat"] = x
    elif "x_endog" in params:
        kwargs["x_endog"] = x
    else:
        kwargs["x"] = x

    if "horizons" in params:
        kwargs["horizons"] = list(horizons)
    elif "horizon" in params:
        if list(horizons) == list(range(0, max(horizons) + 1)):
            kwargs["horizon"] = int(max(horizons))
        else:
            raise ValueError(
                f"{getattr(estimator, '__name__', estimator)} only accepts scalar horizon=."
            )

    if "n_lags" in params:
        kwargs["n_lags"] = int(lags)
    elif "lags" in params:
        kwargs["lags"] = int(lags)

    if "alpha" in params:
        kwargs["alpha"] = 1.0 - float(ci)
    elif "ci" in params:
        kwargs["ci"] = float(ci)

    if controls and ("controls" in params or accepts_var_kw):
        kwargs["controls"] = list(controls)

    return estimator(data, y=y, **kwargs)


def _default_sensitivity_grid(
    *,
    lags: int,
    estimator: Callable[..., Any],
) -> tuple[dict[str, Sequence[Any]], dict[str, Any]]:
    lag_choices = sorted({max(0, int(lags) - 1), int(lags), int(lags) + 1})
    grid: dict[str, Sequence[Any]] = {"lags": lag_choices, "estimator": [estimator]}
    return grid, {
        "lags": int(lags),
        "estimator": estimator,
    }


def _adapt_model_irf_for_compare(
    model_irf: Any,
    *,
    target: str,
    shock: str | None,
) -> Any:
    """Adapt structural IRF containers into forms accepted by ``compare_irfs``."""
    # HANKResult carries transition_paths and steady_state
    if hasattr(model_irf, "transition_paths") and hasattr(model_irf, "steady_state"):
        df_paths = model_irf.transition_paths
        if target not in df_paths.columns:
            raise ValueError(
                f"Target variable {target!r} not found in HANKResult.transition_paths "
                f"columns {list(df_paths.columns)}."
            )
        h_vals = (
            df_paths["h"].to_numpy(dtype=int)
            if "h" in df_paths.columns
            else np.arange(len(df_paths), dtype=int)
        )
        out = pd.DataFrame({"h": h_vals, "point": df_paths[target].to_numpy(dtype=float)})
        out.attrs["method"] = str(getattr(model_irf, "model_name", "HANK"))
        return out

    # Mapping[shock, DataFrame] (used by score_irf_against_targets)
    if isinstance(model_irf, Mapping) and not (
        "point" in model_irf and "lower" in model_irf and "upper" in model_irf
    ):
        keys = list(model_irf.keys())
        if not keys:
            raise ValueError("model_irf mapping is empty.")
        shk = shock if (shock is not None and shock in model_irf) else (keys[0] if len(keys) == 1 else None)
        if shk is None:
            raise ValueError(f"model_irf mapping has multiple shocks {keys}; pass model_shock=...")
        df_s = model_irf[shk]
        if not isinstance(df_s, pd.DataFrame):
            return df_s
        model_irf = df_s

    if isinstance(model_irf, pd.DataFrame):
        cols = [str(c) for c in model_irf.columns]
        if any(c in cols for c in ("beta", "point", "response")) or any(
            c.startswith("beta_") for c in cols
        ):
            return model_irf
        if target in cols:
            h_vals = (
                model_irf["h"].to_numpy(dtype=int)
                if "h" in cols
                else np.arange(len(model_irf), dtype=int)
            )
            out_df = pd.DataFrame(
                {"h": h_vals, "point": model_irf[target].to_numpy(dtype=float)}
            )
            if "se" in cols:
                out_df["se"] = model_irf["se"].to_numpy(dtype=float)
            if "lower" in cols and "upper" in cols:
                out_df["lower"] = model_irf["lower"].to_numpy(dtype=float)
                out_df["upper"] = model_irf["upper"].to_numpy(dtype=float)
            return out_df
    return model_irf


def _adapt_model_irf_for_scorecard(
    model_irf: Any,
    *,
    target: str | None = None,
) -> Any:
    """Adapt long/point IRF tables into wide variable-column frames for scorecard scoring."""
    if hasattr(model_irf, "median") and hasattr(model_irf, "bands") and hasattr(model_irf, "variables"):
        return model_irf
    if hasattr(model_irf, "transition_paths") and hasattr(model_irf, "steady_state"):
        return model_irf
    if isinstance(model_irf, Mapping):
        return {
            str(k): _adapt_model_irf_for_scorecard(v, target=target)
            for k, v in model_irf.items()
        }
    if not isinstance(model_irf, pd.DataFrame):
        return model_irf

    meta_cols = {
        "h",
        "horizon",
        "point",
        "beta",
        "response",
        "value",
        "se",
        "lo",
        "hi",
        "lower",
        "upper",
        "variable",
        "var",
        "shock",
        "spec",
        "t",
        "p",
        "pvalue",
        "n_obs",
        "ci_level",
        "band_kind",
        "is_cumulative",
        "method",
    }
    cols = [str(c) for c in model_irf.columns]
    if any(c not in meta_cols for c in cols):
        return model_irf

    pt_col = next((c for c in ("point", "beta", "response", "value") if c in cols), None)
    if pt_col is None:
        return model_irf

    work = model_irf.copy()
    if "h" not in cols:
        if "horizon" in cols:
            work["h"] = work["horizon"]
        else:
            work["h"] = np.arange(len(work), dtype=int)

    var_col = "variable" if "variable" in cols else ("var" if "var" in cols else None)
    if var_col is not None:
        pivoted = work.pivot_table(
            index="h",
            columns=var_col,
            values=pt_col,
            aggfunc="first",
        )
        pivoted.columns = [str(c) for c in pivoted.columns]
        return pivoted.reset_index()

    if target is not None:
        return pd.DataFrame(
            {
                "h": work["h"].to_numpy(dtype=int),
                str(target): work[pt_col].to_numpy(dtype=float),
            }
        )
    return model_irf


def _normalize_multiindex_panel_data(
    data: pd.DataFrame,
    *,
    unit_col: str | None,
    time_col: str | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Promote MultiIndex panel coordinates into explicit columns when needed."""
    if not (isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2):
        return data, unit_col, time_col
    if (
        unit_col is not None
        and unit_col in data.columns
        and (time_col is None or time_col in data.columns)
    ):
        return data, unit_col, time_col

    idx_names = list(data.index.names)
    lvl0 = str(idx_names[0]) if idx_names[0] is not None else "unit"
    lvl1 = str(idx_names[1]) if idx_names[1] is not None else "date"
    use_unit = unit_col if unit_col is not None else lvl0
    use_time = time_col if time_col is not None else lvl1

    work = data.copy()
    new_names = [
        use_unit if i == 0 else (use_time if i == 1 else (str(n) if n is not None else f"level_{i}"))
        for i, n in enumerate(idx_names)
    ]
    work.index = work.index.set_names(new_names)
    # Drop any level already duplicated as a column before reset_index
    drop_levels = [nm for nm in new_names if nm in work.columns]
    if drop_levels:
        work = work.reset_index(level=[nm for nm in new_names if nm not in drop_levels])
    else:
        work = work.reset_index()
    return work, use_unit, use_time


def _coerce_solver_diagnostics(
    *,
    solver_result: Any = None,
    solver_diagnostics: Any = None,
    model_irf: Any = None,
    prior_posterior: PriorPosteriorComparisonResult | None = None,
) -> SolverDiagnosticRecord | None:
    """Normalize explicit or attached solver diagnostics into SolverDiagnosticRecord."""
    from puremacro.solvers.diagnostics import (
        AccuracyStatus,
        ConvergenceStatus,
        SolverDiagnosticRecord,
        SupportStatus,
        determine_overall_status,
    )

    def _from_mapping(raw: Mapping[str, Any]) -> SolverDiagnosticRecord:
        if "convergence" in raw and "accuracy" in raw and isinstance(raw["convergence"], Mapping):
            c_map = raw["convergence"]
            raw_acc = raw["accuracy"]
            a_map: Mapping[str, Any] = raw_acc if isinstance(raw_acc, Mapping) else {}
            raw_supp = raw.get("support")
            s_map: Mapping[str, Any] = raw_supp if isinstance(raw_supp, Mapping) else {}
            conv = ConvergenceStatus(
                converged=bool(c_map.get("converged", False)),
                iterations=int(c_map.get("iterations", 0)),
                stopping_reason=str(c_map.get("stopping_reason", "")),
                solver_method=str(c_map.get("solver_method", "")),
                tolerance=float(c_map.get("tolerance", 0.0)),
            )
            acc = AccuracyStatus(
                is_accurate=bool(a_map.get("is_accurate", True)),
                max_residual=float(a_map.get("max_residual", 0.0)),
                residual_norm=float(a_map.get("residual_norm", a_map.get("max_residual", 0.0))),
                residual_type=str(a_map.get("residual_type", "")),
                tolerance=float(a_map.get("tolerance", 0.0)),
                residuals=dict(a_map.get("residuals", {}) or {}),
            )
            supp = SupportStatus(
                is_supported=bool(s_map.get("is_supported", True)),
                regime=str(s_map.get("regime", "interior")),
                warnings=tuple(s_map.get("warnings", ()) or ()),
                boundary_metrics=dict(s_map.get("boundary_metrics", {}) or {}),
            )
            ev = dict(raw.get("evidence", {}) or {})
            status = raw.get("overall_status") or determine_overall_status(conv, acc, supp)
            return SolverDiagnosticRecord(
                convergence=conv,
                accuracy=acc,
                support=supp,
                overall_status=str(status),
                evidence=ev,
            )

        max_res = float(
            raw.get(
                "max_residual",
                raw.get("residual", raw.get("euler_kkt_max", 0.0)),
            )
        )
        res_tol = float(raw.get("accuracy_tolerance", raw.get("tolerance", 1e-4)))
        is_acc = raw.get("is_accurate")
        if is_acc is None:
            is_acc = math.isfinite(max_res) and (
                max_res <= res_tol if res_tol > 0.0 else True
            )
        conv = ConvergenceStatus(
            converged=bool(raw.get("converged", True)),
            iterations=int(raw.get("iterations", raw.get("n_iter", 0))),
            stopping_reason=str(raw.get("stopping_reason", raw.get("reason", ""))),
            solver_method=str(raw.get("solver_method", raw.get("solver", raw.get("method", "")))),
            tolerance=float(raw.get("tolerance", 0.0)),
        )
        acc = AccuracyStatus(
            is_accurate=bool(is_acc),
            max_residual=max_res,
            residual_norm=float(raw.get("residual_norm", max_res)),
            residual_type=str(raw.get("residual_type", "")),
            tolerance=res_tol,
            residuals=dict(raw.get("residuals", {}) or {}),
        )
        supp = SupportStatus(
            is_supported=bool(raw.get("is_supported", True)),
            regime=str(raw.get("regime", "interior")),
            warnings=tuple(raw.get("warnings", ()) or ()),
            boundary_metrics=dict(raw.get("boundary_metrics", {}) or {}),
        )
        status = raw.get("overall_status") or determine_overall_status(conv, acc, supp)
        return SolverDiagnosticRecord(
            convergence=conv,
            accuracy=acc,
            support=supp,
            overall_status=str(status),
            evidence=dict(raw.get("evidence", {}) or {}),
        )

    if solver_diagnostics is not None:
        if isinstance(solver_diagnostics, SolverDiagnosticRecord):
            return solver_diagnostics
        if isinstance(solver_diagnostics, Mapping):
            return _from_mapping(solver_diagnostics)
        if (
            isinstance(solver_diagnostics, tuple)
            and len(solver_diagnostics) >= 2
            and isinstance(solver_diagnostics[0], ConvergenceStatus)
            and isinstance(solver_diagnostics[1], AccuracyStatus)
        ):
            conv = solver_diagnostics[0]
            acc = solver_diagnostics[1]
            supp = (
                solver_diagnostics[2]
                if len(solver_diagnostics) >= 3 and isinstance(solver_diagnostics[2], SupportStatus)
                else SupportStatus(is_supported=True, regime="interior")
            )
            return SolverDiagnosticRecord(
                convergence=conv,
                accuracy=acc,
                support=supp,
                overall_status=determine_overall_status(conv, acc, supp),
            )
        raise TypeError(
            "solver_diagnostics must be a SolverDiagnosticRecord, Mapping, or "
            f"(ConvergenceStatus, AccuracyStatus[, SupportStatus]) tuple; got {type(solver_diagnostics).__name__}."
        )

    candidate = solver_result if solver_result is not None else model_irf
    if candidate is not None:
        attached = getattr(candidate, "solver_diagnostics", None)
        if isinstance(attached, SolverDiagnosticRecord):
            return attached
        if isinstance(attached, Mapping) and attached:
            return _from_mapping(attached)

        # HANKResult or custom result carrying converged / diagnostics / determinacy_rate
        has_conv = hasattr(candidate, "converged")
        diag_map = getattr(candidate, "diagnostics", None)
        det_rate = getattr(candidate, "determinacy_rate", None)
        if has_conv or (isinstance(diag_map, Mapping) and diag_map) or det_rate is not None:
            diag_dict = dict(diag_map) if isinstance(diag_map, Mapping) else {}
            conv_flag = bool(getattr(candidate, "converged", True))
            res_val = float(
                diag_dict.get(
                    "max_residual",
                    diag_dict.get(
                        "market_residual_max_abs",
                        diag_dict.get("euler_kkt_max", getattr(candidate, "max_residual", 0.0)),
                    ),
                )
            )
            tol_val = float(diag_dict.get("tolerance", 1e-4))
            is_acc = bool(
                diag_dict.get(
                    "is_accurate",
                    math.isfinite(res_val) and res_val <= tol_val,
                )
            )
            is_supp = True
            regime = "interior"
            supp_warns: list[str] = []
            b_metrics: dict[str, Any] = {}
            if det_rate is not None and math.isfinite(float(det_rate)):
                b_metrics["determinacy_rate"] = float(det_rate)
                if float(det_rate) <= 0.0:
                    is_supp = False
                    regime = "indeterminate_or_explosive"
                    supp_warns.append("Model posterior draws yielded 0% Blanchard-Kahn determinacy.")
            conv = ConvergenceStatus(
                converged=conv_flag,
                iterations=int(diag_dict.get("iterations", getattr(candidate, "iterations", 0))),
                stopping_reason=str(diag_dict.get("stopping_reason", "converged" if conv_flag else "failed")),
                solver_method=str(getattr(candidate, "model_name", type(candidate).__name__)),
                tolerance=tol_val,
            )
            acc = AccuracyStatus(
                is_accurate=is_acc,
                max_residual=res_val,
                residual_norm=float(diag_dict.get("residual_norm", res_val)),
                residual_type=str(diag_dict.get("residual_type", "model_equations")),
                tolerance=tol_val,
            )
            supp = SupportStatus(
                is_supported=is_supp,
                regime=regime,
                warnings=tuple(supp_warns),
                boundary_metrics=b_metrics,
            )
            return SolverDiagnosticRecord(
                convergence=conv,
                accuracy=acc,
                support=supp,
                overall_status=determine_overall_status(conv, acc, supp),
            )

    if prior_posterior is not None and not prior_posterior.table.empty:
        tbl = prior_posterior.table
        any_unconv = bool(tbl["unconverged_or_divergent"].any())
        rhats = tbl["rhat"].to_numpy(dtype=float) if "rhat" in tbl.columns else np.array([], dtype=float)
        fin_rhat = rhats[np.isfinite(rhats)]
        max_rhat_dev = float(np.max(np.abs(fin_rhat - 1.0))) if fin_rhat.size else 0.0
        conv = ConvergenceStatus(
            converged=not any_unconv,
            iterations=int(prior_posterior.n_draws),
            stopping_reason="rhat_ess_check" if not any_unconv else "unconverged_or_divergent",
            solver_method="mcmc_posterior",
            tolerance=0.05,
        )
        acc = AccuracyStatus(
            is_accurate=max_rhat_dev <= 0.05 if fin_rhat.size else True,
            max_residual=max_rhat_dev,
            residual_norm=max_rhat_dev,
            residual_type="max_abs_rhat_minus_1",
            tolerance=0.05,
        )
        supp = SupportStatus(
            is_supported=not bool(tbl["near_boundary"].any()),
            regime="boundary_concentration" if bool(tbl["near_boundary"].any()) else "interior",
        )
        return SolverDiagnosticRecord(
            convergence=conv,
            accuracy=acc,
            support=supp,
            overall_status=determine_overall_status(conv, acc, supp),
        )

    return None


def _resolve_moment_freq(est_df: pd.DataFrame, requested_freq: str | None, inferred_freq: str | None) -> str:
    idx_freq: str | None = None
    if isinstance(est_df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        fstr = getattr(est_df.index, "freqstr", None)
        if fstr:
            idx_freq = str(fstr)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inferred = pd.infer_freq(est_df.index)
            if inferred:
                idx_freq = str(inferred)
    if requested_freq is not None and requested_freq.strip():
        req = requested_freq.strip()
        req_up = req.upper()
        if idx_freq is not None and req_up in ("Q", "M", "A", "Y"):
            idx_up = idx_freq.upper()
            if idx_up.startswith(req_up) or (req_up in ("A", "Y") and idx_up.startswith(("A", "Y"))):
                return idx_freq
        return req
    if idx_freq is not None:
        return idx_freq
    if inferred_freq and not str(inferred_freq).startswith(("step:", "numeric:")):
        return str(inferred_freq)
    return "Q"


def run_empirical_structural_workbench(
    data: pd.DataFrame,
    y: str,
    x: str,
    *,
    unit_col: str | None = None,
    time_col: str | None = None,
    variables: Sequence[str] | None = None,
    freq: str | None = None,
    response_unit: str = "percent",
    shock_unit: str = "unit",
    shock_size: float = 1.0,
    run_stationarity: bool = True,
    run_seasonality: bool = True,
    strict_preflight: bool = True,
    preflight_kwargs: Mapping[str, Any] | None = None,
    horizons: Sequence[int] = tuple(range(9)),
    lags: int = 2,
    controls: Sequence[str] = (),
    ci: float = 0.90,
    estimator: Callable[..., Any] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    sensitivity_grid: Mapping[str, Sequence[Any]] | None = None,
    baseline_spec: Mapping[str, Any] | None = None,
    run_sensitivity: bool = True,
    sensitivity_on_error: Literal["raise", "skip"] = "skip",
    extra_irfs: Mapping[str, Any] | None = None,
    baseline_label: str = "Baseline LP",
    horizon_align: Literal["exact", "intersection"] = "intersection",
    band_compat: Literal["warn", "raise", "ignore"] = "warn",
    model_irf: Any | None = None,
    model_label: str = "Structural Model",
    model_shock: str | None = None,
    targets: Any | None = None,
    build_moment_targets: bool = True,
    target_bandwidth: int | None = None,
    weight_mode: Literal["auto", "covariance", "diagonal"] = "auto",
    posterior: Any | None = None,
    priors: Mapping[str, Any] | Any | None = None,
    solver_result: Any | None = None,
    solver_diagnostics: Any | None = None,
    lang: Literal["en", "es"] = "en",
    title: str | None = None,
    digits: int = 3,
) -> EmpiricalStructuralDossier:
    """Run the unified preflight -> LP/sensitivity -> IRF -> structural dossier pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series or panel input DataFrame.
    y, x : str
        Outcome and shock/treatment column names.
    unit_col, time_col : str, optional
        Optional cross-sectional unit and time column identifiers.
    variables : Sequence[str], optional
        Subset of numeric columns audited in preflight (defaults to all numeric columns).
    freq : str, optional
        Frequency declaration passed to preflight and ``lp_moment_targets``.
    strict_preflight : bool, default True
        Raise ``ValueError`` immediately when preflight detects error-severity issues.
    horizons : Sequence[int], default (0, ..., 8)
        Impulse-response horizons to estimate and compare.
    lags : int, default 2
        Baseline lag length for LP estimation and moment targets.
    controls : Sequence[str], default ()
        Optional control column names.
    ci : float, default 0.90
        Two-sided confidence/credible interval coverage in ``(0, 1)``.
    estimator : Callable, optional
        Baseline LP estimator (defaults to ``panel_lp`` for panels, ``lp_hac`` otherwise).
    sensitivity_grid : Mapping[str, Sequence[Any]], optional
        Custom specification grid forwarded to :func:`puremacro.lp.sensitivity.lp_sensitivity_grid`.
    extra_irfs : Mapping[str, Any], optional
        Additional IRF results included in the unified IRF comparison.
    model_irf : Any, optional
        Structural model IRF (``BayesianIRFResult``, ``HANKResult``, DataFrame, or mapping).
    targets : Any, optional
        Explicit empirical targets for :func:`score_irf_against_targets`.
    posterior : Any, optional
        Bayesian posterior object evaluated via :func:`prior_posterior_comparison`.
    solver_result, solver_diagnostics : Any, optional
        Solved model object or explicit convergence/accuracy/support diagnostic record.
    lang : {'en', 'es'}, default 'en'
        Default output language for dossier summaries and table headings.
    """
    use_lang = _resolve_lang(lang)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("data must not be empty.")
    if not isinstance(y, str) or not y.strip() or y not in data.columns:
        raise ValueError(f"Outcome column y={y!r} not found in DataFrame columns {list(data.columns)!r}.")
    if not isinstance(x, str) or not x.strip() or x not in data.columns:
        raise ValueError(f"Shock column x={x!r} not found in DataFrame columns {list(data.columns)!r}.")
    if isinstance(controls, str):
        raise ValueError("controls must be a sequence of column names, not a string.")
    ctrl_tuple = tuple(str(c) for c in controls)
    missing_ctrl = [c for c in ctrl_tuple if c not in data.columns]
    if missing_ctrl:
        raise ValueError(f"Control column(s) {missing_ctrl!r} not found in DataFrame columns.")
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or int(lags) < 0:
        raise ValueError(f"lags must be a non-negative integer, got {lags!r}.")
    lags_int = int(lags)
    if not (0.0 < float(ci) < 1.0):
        raise ValueError(f"ci must lie strictly in (0, 1), got {ci!r}.")
    ci_val = float(ci)
    horizons_tup = _validate_horizons(horizons)
    if horizon_align not in ("exact", "intersection"):
        raise ValueError(f"Invalid horizon_align={horizon_align!r}.")
    if band_compat not in ("warn", "raise", "ignore"):
        raise ValueError(f"Invalid band_compat={band_compat!r}.")
    if sensitivity_on_error not in ("raise", "skip"):
        raise ValueError(f"Invalid sensitivity_on_error={sensitivity_on_error!r}.")
    if weight_mode not in ("auto", "covariance", "diagonal"):
        raise ValueError(f"Invalid weight_mode={weight_mode!r}.")

    from puremacro.data import preflight_panel
    from puremacro.dsge.validation_scorecard import (
        prior_posterior_comparison,
        score_irf_against_targets,
    )
    from puremacro.lp.jorda import lp_hac
    from puremacro.lp.panel import panel_lp
    from puremacro.lp.sensitivity import lp_sensitivity_grid
    from puremacro.reports import compare_irfs
    from puremacro.structural.bridge import MomentTargets
    from puremacro.structural.lp import lp_moment_targets

    dossier_warnings: list[str] = []

    # 1. Preflight
    pf_data, pf_unit_col, pf_time_col = _normalize_multiindex_panel_data(
        data,
        unit_col=unit_col,
        time_col=time_col,
    )
    pf_extra = dict(preflight_kwargs or {})
    preflight = preflight_panel(
        pf_data,
        unit_col=pf_unit_col,
        time_col=pf_time_col,
        variables=variables,
        freq=freq,
        run_stationarity=run_stationarity,
        run_seasonality=run_seasonality,
        digits=digits,
        **pf_extra,
    )
    if strict_preflight and not preflight.passed:
        err_msgs = "; ".join(
            f.message for f in preflight.findings if f.severity == "error"
        )
        raise ValueError(
            f"Panel preflight failed with {preflight.n_errors} error(s): {err_msgs}"
        )
    if not preflight.passed:
        for f in preflight.findings:
            if f.severity == "error":
                dossier_warnings.append(f"Preflight error [{f.check}]: {f.message}")

    # 2. Estimation DataFrame normalization + baseline empirical LP
    est_df, panel_kwargs = _prepare_estimation_frame(
        pf_data,
        unit_col=preflight.unit_col,
        time_col=preflight.time_col,
    )
    est_fn = estimator if estimator is not None else (
        panel_lp if preflight.unit_col is not None else lp_hac
    )

    merged_base: dict[str, Any] = {**panel_kwargs, **dict(base_kwargs or {})}
    empirical_lp = _call_estimator_cleanly(
        est_fn,
        est_df,
        y,
        x,
        horizons=horizons_tup,
        lags=lags_int,
        controls=ctrl_tuple,
        ci=ci_val,
        extra_kwargs=merged_base,
    )

    # 3. LP Sensitivity Grid
    sensitivity: IRFComparisonResult | None = None
    if sensitivity_grid is not None or run_sensitivity:
        if sensitivity_grid is not None:
            grid_to_run = dict(sensitivity_grid)
            base_spec_use = baseline_spec
        else:
            grid_to_run, base_spec_use = _default_sensitivity_grid(
                lags=lags_int,
                estimator=est_fn,
            )

        grid_base_kwargs: dict[str, Any] = dict(merged_base)
        if "horizons" not in grid_to_run and "horizon" not in grid_to_run:
            grid_base_kwargs.setdefault("horizons", list(horizons_tup))
        if "ci" not in grid_to_run and "alpha" not in grid_to_run:
            grid_base_kwargs.setdefault("ci", ci_val)
        if ctrl_tuple and "controls" not in grid_to_run:
            grid_base_kwargs.setdefault("controls", list(ctrl_tuple))
        if "lags" not in grid_to_run and "n_lags" not in grid_to_run:
            grid_base_kwargs.setdefault("lags", lags_int)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sensitivity = lp_sensitivity_grid(
                    est_df,
                    y,
                    x,
                    grid=grid_to_run,
                    estimator=est_fn,
                    baseline_spec=base_spec_use,
                    base_kwargs=grid_base_kwargs,
                    on_error=sensitivity_on_error,
                    horizon_align=horizon_align,
                    band_compat=band_compat,
                )
            dossier_warnings.extend(sensitivity.warnings)
        except Exception as exc:
            if sensitivity_grid is not None or sensitivity_on_error == "raise":
                raise
            dossier_warnings.append(
                f"Default LP sensitivity grid skipped ({type(exc).__name__}: {exc})."
            )

    # 4. IRF Comparison
    if extra_irfs is None and model_irf is None and sensitivity is not None:
        irf_comparison = sensitivity
    else:
        cmp_map: dict[str, Any] = {baseline_label: empirical_lp}
        if extra_irfs:
            for lbl, obj in extra_irfs.items():
                cmp_map[str(lbl)] = obj
        if model_irf is not None:
            cmp_map[model_label] = _adapt_model_irf_for_compare(
                model_irf,
                target=y,
                shock=model_shock,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            irf_comparison = compare_irfs(
                cmp_map,
                baseline=baseline_label,
                horizon_align=horizon_align,
                target=y,
                shock=model_shock,
                band_compat=band_compat,
                digits=digits,
            )
        dossier_warnings.extend(irf_comparison.warnings)

    # 5. Structural Moment Targets + IRF Scorecard + Prior-Posterior
    moment_targets: MomentTargets | None = (
        targets if isinstance(targets, MomentTargets) else None
    )
    if moment_targets is None and targets is None and build_moment_targets:
        if preflight.unit_col is None:
            try:
                mt_freq = _resolve_moment_freq(est_df, freq, preflight.inferred_freq)
                moment_targets = lp_moment_targets(
                    est_df,
                    responses=(y,),
                    shock=x,
                    horizons=horizons_tup,
                    response_units={y: response_unit},
                    shock_unit=shock_unit,
                    frequency=mt_freq,
                    lags=lags_int,
                    controls=ctrl_tuple,
                    bandwidth=target_bandwidth,
                    shock_size=shock_size,
                )
            except ValueError as exc:
                dossier_warnings.append(
                    f"Joint moment targets unavailable ({exc}); scoring uses marginal LP estimates if requested."
                )
        else:
            dossier_warnings.append(
                "Joint moment targets via lp_moment_targets apply to single time series; "
                "panel workflow uses marginal LP estimates for structural scoring."
            )

    irf_scorecard: IRFScorecardResult | None = None
    if model_irf is not None:
        score_targets = (
            targets
            if targets is not None
            else (moment_targets if moment_targets is not None else {y: empirical_lp})
        )
        score_vars = None if targets is not None else [y]
        score_model_irf = _adapt_model_irf_for_scorecard(model_irf, target=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            irf_scorecard = score_irf_against_targets(
                score_model_irf,
                score_targets,
                shock=model_shock,
                variables=score_vars,
                horizons=horizons_tup,
                weight_mode=weight_mode,
                ci_level=ci_val,
                digits=digits,
            )
        dossier_warnings.extend(irf_scorecard.warnings)

    prior_posterior: PriorPosteriorComparisonResult | None = None
    if posterior is not None:
        prior_posterior = prior_posterior_comparison(
            posterior,
            priors=priors,
            ci_level=ci_val,
            digits=digits,
        )
        dossier_warnings.extend(prior_posterior.warnings)

    # 6. Solver Diagnostics
    coerced_diag = _coerce_solver_diagnostics(
        solver_result=solver_result,
        solver_diagnostics=solver_diagnostics,
        model_irf=model_irf,
        prior_posterior=prior_posterior,
    )
    if coerced_diag is not None and coerced_diag.support.warnings:
        dossier_warnings.extend(coerced_diag.support.warnings)

    default_title = (
        f"Expediente empírico-estructural: {y} <- {x}"
        if use_lang == "es"
        else f"Empirical-Structural Dossier: {y} <- {x}"
    )

    return EmpiricalStructuralDossier(
        y=y,
        x=x,
        controls=ctrl_tuple,
        horizons=horizons_tup,
        lags=lags_int,
        ci_level=ci_val,
        lang=use_lang,
        title=title if title is not None else default_title,
        preflight=preflight,
        empirical_lp=empirical_lp,
        sensitivity=sensitivity,
        irf_comparison=irf_comparison,
        moment_targets=moment_targets,
        prior_posterior=prior_posterior,
        irf_scorecard=irf_scorecard,
        solver_diagnostics=coerced_diag,
        warnings=tuple(dict.fromkeys(dossier_warnings)),
        metadata={
            "estimator": getattr(est_fn, "__name__", str(est_fn)),
            "is_panel": preflight.unit_col is not None,
            "data_sha256": preflight.data_sha256,
        },
        digits=int(digits),
    )


__all__ = [
    "EmpiricalStructuralDossier",
    "run_empirical_structural_workbench",
]
