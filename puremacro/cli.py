"""Unified command-line interface for puremacro workflows (``python -m puremacro.cli``).

Provides subcommands for panel preflight checks (``preflight``), LP
sensitivity grids (``lp-grid``), multi-specification IRF comparison
(``compare-irfs``), structural IRF scorecards (``scorecard``), connector
freshness checks (``freshness``), and environment diagnostics (``doctor``).

Top-level imports use the Python standard library exclusively so CLI parsing
and Pyodide module sweeps stay lightweight and free of optional dependencies.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Callable, Mapping, Sequence
import warnings


_RANGE_RE = re.compile(r"^\s*(-?\d+)\s*(?:\.\.|:)\s*(-?\d+)\s*$")

_ESTIMATOR_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("lp_hac", "puremacro.lp.jorda", "lp_hac"),
    ("la_lp", "puremacro.lp.la_lp", "la_lp"),
    ("smooth_lp", "puremacro.lp.smooth", "smooth_lp"),
    ("panel_lp", "puremacro.lp.panel", "panel_lp"),
    ("panel_lp_dk", "puremacro.lp.panel_dk", "panel_lp_dk"),
    ("lp_did", "puremacro.lp.lp_did", "lp_did"),
    ("lp_iv", "puremacro.lp.iv", "lp_iv"),
)


class _CLIUsageError(ValueError):
    """Raised when CLI arguments or input schemas are invalid (exit code 2)."""


def _safe_print(text: str, stream: Any = None) -> None:
    target = sys.stdout if stream is None else stream
    enc = getattr(target, "encoding", None) or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    target.write(safe)
    if not safe.endswith("\n"):
        target.write("\n")


def _write_or_print(
    rendered: str,
    *,
    output: str | None,
    quiet: bool,
) -> None:
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = rendered if rendered.endswith("\n") else rendered + "\n"
        out_path.write_text(payload, encoding="utf-8")
    if not quiet:
        _safe_print(rendered)


def _json_sanitize(value: Any) -> Any:
    import numpy as np
    import pandas as pd

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
    if isinstance(value, Path):
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


def _dumps_json(payload: Any) -> str:
    return json.dumps(_json_sanitize(payload), indent=2, allow_nan=False, ensure_ascii=False)


def _parse_csv_tokens(raw_values: Sequence[str] | None) -> list[str]:
    if not raw_values:
        return []
    tokens: list[str] = []
    for chunk in raw_values:
        for part in str(chunk).split(","):
            cleaned = part.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _parse_int_sequence(spec: str, *, name: str) -> list[int]:
    text = str(spec).strip()
    if not text:
        raise _CLIUsageError(f"{name} specification cannot be empty.")
    m = _RANGE_RE.match(text)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        if hi < lo:
            raise _CLIUsageError(f"{name} range {text!r} has end < start.")
        return list(range(lo, hi + 1))

    vals: list[int] = []
    for part in text.split(","):
        tok = part.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError as exc:
            raise _CLIUsageError(
                f"Invalid integer {tok!r} in {name}={spec!r}; expected e.g. '0..8' or '0,1,2'."
            ) from exc
    if not vals:
        raise _CLIUsageError(f"{name} produced no integers from {spec!r}.")
    if len(set(vals)) != len(vals):
        raise _CLIUsageError(f"{name} contains duplicate values: {vals!r}.")
    return vals


def _load_json_raw(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _CLIUsageError(f"Invalid JSON in {path}: {exc}") from exc


def _dataframe_from_json_obj(obj: Any, *, source: str) -> Any:
    import pandas as pd

    if isinstance(obj, list):
        if not obj:
            raise _CLIUsageError(f"JSON list in {source} is empty.")
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if "data" in obj and "columns" in obj and isinstance(obj["data"], list):
            return pd.DataFrame(obj["data"], columns=obj["columns"])
        try:
            return pd.DataFrame(obj)
        except ValueError as exc:
            raise _CLIUsageError(f"Cannot interpret JSON object in {source} as a table: {exc}") from exc
    raise _CLIUsageError(f"Expected JSON array or column mapping in {source}, got {type(obj).__name__}.")


def _load_dataframe(path_str: str) -> Any:
    import pandas as pd

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.is_dir():
        raise _CLIUsageError(f"Input path is a directory, expected a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        obj = _load_json_raw(path)
        return _dataframe_from_json_obj(obj, source=str(path))
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise _CLIUsageError(
        f"Unsupported file extension {suffix!r} for {path}; expected .csv, .tsv, .json, or .parquet."
    )


def _resolve_estimator(name: str) -> Callable[..., Any]:
    key = name.strip().lower()
    for est_name, mod_path, attr in _ESTIMATOR_REGISTRY:
        if key == est_name:
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)  # type: ignore[no-any-return]
    valid = [item[0] for item in _ESTIMATOR_REGISTRY]
    raise _CLIUsageError(f"Unknown estimator {name!r}; supported: {valid}")


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format ('json' or 'markdown'). Default: markdown.",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "es"),
        default="en",
        help="Report language ('en' or 'es'). Default: en.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Decimal precision for Markdown tables. Default: 3.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to write rendered output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output (useful with --output).",
    )


def _run_preflight(args: argparse.Namespace) -> int:
    from puremacro.data import preflight_panel
    from puremacro.reports import df_to_markdown

    df = _load_dataframe(args.input)
    vars_list = _parse_csv_tokens(args.variables) or None
    report = preflight_panel(
        df,
        unit_col=args.unit_col,
        time_col=args.time_col,
        variables=vars_list,
        freq=args.freq,
        min_obs=args.min_obs,
        max_missing_share=args.max_missing_share,
        alpha=args.alpha,
        run_stationarity=not args.no_stationarity,
        run_seasonality=not args.no_seasonality,
        digits=args.digits,
    )

    if args.format == "json":
        payload = json.loads(report.to_json())
        payload["lang"] = args.lang
        rendered = _dumps_json(payload)
    else:
        is_es = args.lang == "es"
        heading = "# Diagnóstico previo del panel" if is_es else "# Panel Preflight Report"
        var_heading = "## Variables" if not is_es else "## Variables"
        find_heading = "## Hallazgos" if is_es else "## Findings"
        rec_heading = "## Recomendaciones" if is_es else "## Recommendations"
        dr = f"{report.date_range[0]}..{report.date_range[1]}" if report.date_range else "n/a"
        meta_line = (
            f"Estado: **{report.status.upper()}** | filas={report.panel_shape[0]}, "
            f"columnas={report.panel_shape[1]}, unidades={report.n_units}, periodos={report.n_periods}, "
            f"frecuencia={report.inferred_freq or 'n/a'}, rango={dr}, balanceado={report.balanced}, "
            f"errores={report.n_errors}, advertencias={report.n_warnings}"
            if is_es
            else
            f"Status: **{report.status.upper()}** | rows={report.panel_shape[0]}, "
            f"cols={report.panel_shape[1]}, units={report.n_units}, periods={report.n_periods}, "
            f"freq={report.inferred_freq or 'n/a'}, range={dr}, balanced={report.balanced}, "
            f"errors={report.n_errors}, warnings={report.n_warnings}"
        )
        lines = [
            heading,
            "",
            meta_line,
            "",
            var_heading,
            "",
            report.to_markdown(layout="variables", digits=args.digits),
        ]
        if report.findings:
            lines.extend([
                "",
                find_heading,
                "",
                df_to_markdown(report.to_frame("findings"), index=False, digits=args.digits),
            ])
        if report.recommendations:
            lines.extend(["", rec_heading, ""])
            lines.extend(f"- {rec}" for rec in report.recommendations)
        rendered = "\n".join(lines) + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    if report.n_errors > 0 and not args.allow_errors:
        return 1
    if args.strict and report.n_warnings > 0:
        return 1
    return 0


def _run_lp_grid(args: argparse.Namespace) -> int:
    from puremacro.lp.sensitivity import lp_sensitivity_grid
    from puremacro.reports import df_to_markdown
    from puremacro.workbench import (
        _prepare_lp_grid_context,
        run_empirical_structural_workbench,
    )

    df = _load_dataframe(args.input)
    if args.y not in df.columns:
        raise _CLIUsageError(f"Outcome column {args.y!r} not found in {list(df.columns)!r}.")
    if args.x not in df.columns:
        raise _CLIUsageError(f"Shock column {args.x!r} not found in {list(df.columns)!r}.")

    horizons_list = _parse_int_sequence(args.horizons, name="horizons")
    lags_list = _parse_int_sequence(args.lags, name="lags")
    if any(lv < 0 for lv in lags_list):
        raise _CLIUsageError(f"lags must be non-negative integers, got {lags_list!r}.")
    if any(hv < 0 for hv in horizons_list):
        raise _CLIUsageError(f"horizons must be non-negative integers, got {horizons_list!r}.")
    if not (0.0 < float(args.ci) < 1.0):
        raise _CLIUsageError(f"ci must lie strictly in (0, 1), got {args.ci!r}.")

    controls = _parse_csv_tokens(args.controls)
    missing_ctrl = [c for c in controls if c not in df.columns]
    if missing_ctrl:
        raise _CLIUsageError(f"Control column(s) {missing_ctrl!r} not found in {list(df.columns)!r}.")

    est_tokens = _parse_csv_tokens(args.estimators)
    if not est_tokens:
        est_tokens = ["panel_lp"] if args.unit_col else ["lp_hac"]
    est_fns = [_resolve_estimator(tok) for tok in est_tokens]

    grid: dict[str, list[Any]] = {"lags": lags_list}
    if len(est_fns) > 1:
        grid["estimator"] = est_fns

    n_grid_points = len(lags_list) * len(est_fns)
    if n_grid_points < 2 and not args.workbench:
        raise _CLIUsageError(
            "lp-grid requires at least 2 grid points (pass multiple --lags or --estimators)."
        )

    baseline_spec: dict[str, Any] | None = None
    if args.baseline_lags is not None or args.baseline_estimator is not None:
        baseline_spec = {}
        if args.baseline_lags is not None:
            baseline_spec["lags"] = int(args.baseline_lags)
        if args.baseline_estimator is not None:
            baseline_spec["estimator"] = _resolve_estimator(args.baseline_estimator)

    raw_base_kwargs: dict[str, Any] = {
        "horizons": horizons_list,
        "ci": float(args.ci),
    }
    if controls:
        raw_base_kwargs["controls"] = controls

    try:
        est_df, adapted_panel_kwargs, grid_base_kwargs = _prepare_lp_grid_context(
            df,
            estimators=est_fns,
            unit_col=args.unit_col,
            time_col=args.time_col,
            extra_kwargs=raw_base_kwargs,
        )
    except ValueError as exc:
        raise _CLIUsageError(str(exc)) from exc

    if args.workbench:
        base_lags = int(args.baseline_lags) if args.baseline_lags is not None else lags_list[0]
        wb_unit = args.unit_col or adapted_panel_kwargs.get("unit") or adapted_panel_kwargs.get("unit_col")
        wb_time = args.time_col or adapted_panel_kwargs.get("time") or adapted_panel_kwargs.get("time_col")
        dossier = run_empirical_structural_workbench(
            df,
            args.y,
            args.x,
            unit_col=wb_unit,
            time_col=wb_time,
            horizons=horizons_list,
            lags=base_lags,
            controls=controls,
            ci=float(args.ci),
            estimator=est_fns[0],
            sensitivity_grid=grid if n_grid_points >= 2 else None,
            baseline_spec=baseline_spec,
            run_sensitivity=n_grid_points >= 2,
            sensitivity_on_error=args.on_error,
            horizon_align=args.horizon_align,
            band_compat=args.band_compat,
            lang=args.lang,
            digits=args.digits,
        )
        rendered = (
            dossier.to_json()
            if args.format == "json"
            else dossier.to_markdown(lang=args.lang, digits=args.digits)
        )
        _write_or_print(rendered, output=args.output, quiet=args.quiet)
        return 0 if dossier.passed else 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp = lp_sensitivity_grid(
            est_df,
            args.y,
            args.x,
            grid=grid,
            estimator=est_fns[0],
            baseline_spec=baseline_spec,
            base_kwargs=grid_base_kwargs,
            on_error=args.on_error,
            horizon_align=args.horizon_align,
            band_compat=args.band_compat,
        )

    if args.format == "json":
        rendered = _dumps_json(
            {
                "y": args.y,
                "x": args.x,
                "controls": controls,
                "baseline": cmp.baseline,
                "horizons": list(cmp.horizons),
                "metrics": cmp.metrics,
                "table": cmp.table,
                "warnings": list(cmp.warnings),
                "errors": dict(cmp.errors),
            }
        )
    else:
        is_es = args.lang == "es"
        title = (
            f"# Malla de sensibilidad LP: {args.y} <- {args.x}"
            if is_es
            else f"# LP Sensitivity Grid: {args.y} <- {args.x}"
        )
        m_head = "## Métricas por especificación" if is_es else "## Specification Metrics"
        t_head = "## Respuestas al impulso (coef / ee)" if is_es else "## Impulse Responses (coef / se)"
        summary_line = (
            f"Referencia: **{cmp.baseline}** | especificaciones={len(cmp.metrics)} | "
            f"horizontes=[{min(cmp.horizons)}..{max(cmp.horizons)}]"
            if is_es
            else
            f"Baseline: **{cmp.baseline}** | specs={len(cmp.metrics)} | "
            f"horizons=[{min(cmp.horizons)}..{max(cmp.horizons)}]"
        )
        lines = [
            title,
            "",
            summary_line,
            "",
            m_head,
            "",
            df_to_markdown(cmp.to_frame("metrics"), index=False, digits=args.digits),
            "",
            t_head,
            "",
            cmp.to_markdown(layout="coef_se", digits=args.digits),
        ]
        if cmp.warnings:
            w_head = "## Advertencias" if is_es else "## Warnings"
            lines.extend(["", w_head, ""])
            lines.extend(f"- {w}" for w in cmp.warnings)
        rendered = "\n".join(lines) + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    return 0


def _split_label_path(raw: str, seen_labels: dict[str, int]) -> tuple[str, str]:
    if "=" in raw:
        label, path_str = raw.split("=", 1)
        label = label.strip()
        path_str = path_str.strip()
        if not label or not path_str:
            raise _CLIUsageError(f"Invalid spec {raw!r}; expected LABEL=PATH.")
    else:
        path_str = raw.strip()
        label = Path(path_str).stem or "spec"
    count = seen_labels.get(label, 0) + 1
    seen_labels[label] = count
    unique_label = label if count == 1 else f"{label} #{count}"
    return unique_label, path_str


def _run_compare_irfs(args: argparse.Namespace) -> int:
    import pandas as pd
    from puremacro.reports import compare_irfs, df_to_markdown

    spec_items = list(args.specs) + list(args.spec)
    if not spec_items and args.bundle is None:
        raise _CLIUsageError(
            "compare-irfs requires at least one spec file (positional PATH / LABEL=PATH, --spec, or --bundle)."
        )

    irf_map: dict[str, Any] = {}
    seen_labels: dict[str, int] = {}

    if args.bundle is not None:
        b_path = Path(args.bundle)
        if b_path.suffix.lower() == ".json":
            raw_bundle = _load_json_raw(b_path)
            if isinstance(raw_bundle, dict) and all(
                isinstance(v, (list, dict)) for v in raw_bundle.values()
            ) and "h" not in raw_bundle:
                for k, v in raw_bundle.items():
                    irf_map[str(k)] = _dataframe_from_json_obj(v, source=f"{b_path}:{k}")
            else:
                df_b = _dataframe_from_json_obj(raw_bundle, source=str(b_path))
                if "spec" in df_b.columns:
                    for sp, sub in df_b.groupby("spec", sort=False):
                        irf_map[str(sp)] = sub.drop(columns=["spec"]).reset_index(drop=True)
                else:
                    raise _CLIUsageError(f"Bundle {b_path} lacks spec mapping or 'spec' column.")
        else:
            df_b = _load_dataframe(str(b_path))
            if "spec" not in df_b.columns:
                raise _CLIUsageError(f"Bundle table {b_path} must contain a 'spec' column.")
            for sp, sub in df_b.groupby("spec", sort=False):
                irf_map[str(sp)] = sub.drop(columns=["spec"]).reset_index(drop=True)

    for raw in spec_items:
        label, path_str = _split_label_path(raw, seen_labels)
        df_s = _load_dataframe(path_str)
        if "spec" in df_s.columns and "=" not in raw and df_s["spec"].nunique() > 1:
            for sp, sub in df_s.groupby("spec", sort=False):
                irf_map[str(sp)] = sub.drop(columns=["spec"]).reset_index(drop=True)
        else:
            irf_map[label] = df_s

    if not irf_map:
        raise _CLIUsageError("No IRF specifications found to compare.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp = compare_irfs(
            irf_map,
            baseline=args.baseline,
            horizon_align=args.horizon_align,
            band_compat=args.band_compat,
            cumulative=args.cumulative,
            allow_mixed_cumulative=args.allow_mixed_cumulative,
            expand_groups=args.expand_groups,
            digits=args.digits,
        )

    if args.format == "json":
        rendered = _dumps_json(
            {
                "baseline": cmp.baseline,
                "horizons": list(cmp.horizons),
                "metrics": cmp.metrics,
                "table": cmp.table,
                "warnings": list(cmp.warnings),
                "errors": dict(cmp.errors),
            }
        )
    else:
        is_es = args.lang == "es"
        title = "# Comparación de funciones de impulso-respuesta" if is_es else "# IRF Comparison"
        m_head = "## Métricas" if is_es else "## Metrics"
        t_head = "## Tabla alineada (coef / ee)" if is_es else "## Aligned Table (coef / se)"
        summary_line = (
            f"Referencia: **{cmp.baseline}** | especificaciones={len(cmp.metrics)} | "
            f"horizontes=[{min(cmp.horizons)}..{max(cmp.horizons)}]"
            if is_es
            else
            f"Baseline: **{cmp.baseline}** | specs={len(cmp.metrics)} | "
            f"horizons=[{min(cmp.horizons)}..{max(cmp.horizons)}]"
        )
        lines = [
            title,
            "",
            summary_line,
            "",
            m_head,
            "",
            df_to_markdown(cmp.to_frame("metrics"), index=False, digits=args.digits),
            "",
            t_head,
            "",
            cmp.to_markdown(layout="coef_se", digits=args.digits),
        ]
        if cmp.warnings:
            w_head = "## Advertencias" if is_es else "## Warnings"
            lines.extend(["", w_head, ""])
            lines.extend(f"- {w}" for w in cmp.warnings)
        rendered = "\n".join(lines) + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    return 0


def _load_scorecard_model_irf(path_str: str) -> Any:
    import pandas as pd

    path = Path(path_str)
    if path.suffix.lower() == ".json":
        raw = _load_json_raw(path)
        if isinstance(raw, dict) and raw and all(
            isinstance(v, (list, dict)) for v in raw.values()
        ) and "h" not in raw:
            return {
                str(k): _dataframe_from_json_obj(v, source=f"{path}:{k}")
                for k, v in raw.items()
            }
        return _dataframe_from_json_obj(raw, source=str(path))
    return _load_dataframe(path_str)


def _load_scorecard_targets(
    path_str: str,
    *,
    covariance_path: str | None,
    vars_filter: list[str] | None,
) -> Any:
    import numpy as np
    import pandas as pd
    from puremacro.structural.bridge import MomentTargets

    path = Path(path_str)
    cov_arr: np.ndarray | None = None
    if covariance_path is not None:
        c_path = Path(covariance_path)
        if c_path.suffix.lower() == ".json":
            c_raw = _load_json_raw(c_path)
            cov_arr = np.asarray(c_raw, dtype=float)
        else:
            cov_df = _load_dataframe(covariance_path)
            cov_arr = cov_df.to_numpy(dtype=float)

    if path.suffix.lower() == ".json":
        raw = _load_json_raw(path)
        if (
            isinstance(raw, dict)
            and "values" in raw
            and "labels" in raw
            and "covariance" in raw
        ):
            cov_use = cov_arr if cov_arr is not None else np.asarray(raw["covariance"], dtype=float)
            units = raw.get("units", "unspecified")
            meta = raw.get("metadata") or {}
            return MomentTargets(
                values=np.asarray(raw["values"], dtype=float),
                covariance=cov_use,
                labels=tuple(str(lbl) for lbl in raw["labels"]),
                units=tuple(str(u) for u in units) if isinstance(units, list) else str(units),
                metadata=dict(meta) if isinstance(meta, Mapping) else {},
            )
        df_t = _dataframe_from_json_obj(raw, source=str(path))
    else:
        df_t = _load_dataframe(path_str)

    if isinstance(df_t, pd.DataFrame) and cov_arr is not None and "label" in df_t.columns:
        val_col = next(
            (c for c in ("estimate", "target", "value", "point", "beta") if c in df_t.columns),
            "estimate",
        )
        return MomentTargets.from_frame(df_t, covariance=cov_arr, value=val_col)

    if (
        isinstance(df_t, pd.DataFrame)
        and ("beta" in df_t.columns or "point" in df_t.columns)
        and "variable" not in df_t.columns
        and "var" not in df_t.columns
        and vars_filter
        and len(vars_filter) == 1
    ):
        df_t = df_t.copy()
        df_t.y_name = vars_filter[0]  # type: ignore[attr-defined]
    return df_t


def _run_scorecard(args: argparse.Namespace) -> int:
    from puremacro.dsge.validation_scorecard import score_irf_against_targets
    from puremacro.reports import df_to_markdown
    from puremacro.workbench import _adapt_model_irf_for_scorecard

    if not (0.0 < float(args.ci) < 1.0):
        raise _CLIUsageError(f"ci must lie strictly in (0, 1), got {args.ci!r}.")

    vars_filter = _parse_csv_tokens(args.variables) or None
    horizons_filter = (
        _parse_int_sequence(args.horizons, name="horizons")
        if args.horizons is not None
        else None
    )
    model_irf = _load_scorecard_model_irf(args.model_irf)
    targets = _load_scorecard_targets(
        args.targets,
        covariance_path=args.covariance,
        vars_filter=vars_filter,
    )
    inferred_target: str | None = (
        vars_filter[0] if (vars_filter is not None and len(vars_filter) == 1) else None
    )
    if inferred_target is None and hasattr(targets, "labels"):
        lbl_vars = {str(lbl).split(":", 1)[0] for lbl in getattr(targets, "labels", ())}
        if len(lbl_vars) == 1:
            inferred_target = next(iter(lbl_vars))
    model_irf = _adapt_model_irf_for_scorecard(model_irf, target=inferred_target)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sc = score_irf_against_targets(
            model_irf,
            targets,
            shock=args.shock,
            variables=vars_filter,
            horizons=horizons_filter,
            weight_mode=args.weight_mode,
            ci_level=float(args.ci),
            digits=args.digits,
        )

    if args.format == "json":
        rendered = _dumps_json(
            {
                "rmse": sc.rmse,
                "mae": sc.mae,
                "wse": sc.wse,
                "chi2_stat": sc.chi2_stat,
                "chi2_df": sc.chi2_df,
                "chi2_pvalue": sc.chi2_pvalue,
                "sign_concordance": sc.sign_concordance,
                "band_coverage": sc.band_coverage,
                "determinacy_rate": sc.determinacy_rate,
                "weight_mode_used": sc.weight_mode_used,
                "covariance_condition_number": sc.covariance_condition_number,
                "metrics": sc.metrics,
                "table": sc.table,
                "warnings": list(sc.warnings),
            }
        )
    else:
        is_es = args.lang == "es"
        title = (
            "# Evaluación de IRF estructural frente a objetivos empíricos"
            if is_es
            else "# Structural IRF Validation Scorecard"
        )
        m_head = "## Métricas agregadas y por variable" if is_es else "## Aggregate & Variable Metrics"
        t_head = "## Comparación horizonte a horizonte" if is_es else "## Horizon-by-Horizon Comparison"
        wald = (
            f"chi2({sc.chi2_df})={sc.chi2_stat:.{args.digits}f}, p={sc.chi2_pvalue:.4g}"
            if sc.chi2_df > 0 and math.isfinite(sc.chi2_stat) and math.isfinite(sc.chi2_pvalue)
            else (f"WSE={sc.wse:.{args.digits}f}" if math.isfinite(sc.wse) else "unweighted")
        )
        summary_line = (
            f"Momentos={len(sc.table)} | modo_ponderación={sc.weight_mode_used} | "
            f"RMSE={sc.rmse:.{args.digits}f} | MAE={sc.mae:.{args.digits}f} | {wald} | "
            f"signo={sc.sign_concordance:.{args.digits}f} | cobertura={sc.band_coverage:.{args.digits}f}"
            if is_es
            else
            f"Moments={len(sc.table)} | weight_mode={sc.weight_mode_used} | "
            f"RMSE={sc.rmse:.{args.digits}f} | MAE={sc.mae:.{args.digits}f} | {wald} | "
            f"sign_concordance={sc.sign_concordance:.{args.digits}f} | "
            f"band_coverage={sc.band_coverage:.{args.digits}f}"
        )
        lines = [
            title,
            "",
            summary_line,
            "",
            m_head,
            "",
            df_to_markdown(sc.to_frame("metrics"), index=False, digits=args.digits),
            "",
            t_head,
            "",
            sc.to_markdown(layout="table", digits=args.digits),
        ]
        if sc.warnings:
            w_head = "## Advertencias" if is_es else "## Warnings"
            lines.extend(["", w_head, ""])
            lines.extend(f"- {w}" for w in sc.warnings)
        rendered = "\n".join(lines) + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    return 0


def _run_freshness(args: argparse.Namespace) -> int:
    from puremacro.fetch.freshness import RateBudget, connector_freshness_report

    sources = _parse_csv_tokens(args.sources) or None
    budget = RateBudget(
        max_requests=args.max_requests,
        min_interval_seconds=args.min_interval,
        timeout_seconds=args.timeout,
    )
    report = connector_freshness_report(
        mode=args.mode,
        sources=sources,
        as_of=args.as_of,
        db_path=args.db_path,
        rate_budget=budget,
    )
    rendered = (
        _dumps_json(report.to_dict())
        if args.format == "json"
        else report.to_markdown(lang=args.lang)
    )
    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    if args.fail_on_outage and report.has_outages:
        return 1
    return 0


def _probe_module(mod_name: str, *, required: bool) -> dict[str, Any]:
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        return {
            "name": mod_name,
            "required": required,
            "status": "missing" if required else "not_installed",
            "version": None,
            "error": None,
        }
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", None)
        return {
            "name": mod_name,
            "required": required,
            "status": "ok",
            "version": str(ver) if ver is not None else "installed",
            "error": None,
        }
    except Exception as exc:
        return {
            "name": mod_name,
            "required": required,
            "status": "error",
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_doctor(args: argparse.Namespace) -> int:
    import pandas as pd
    import puremacro
    from puremacro import _cache_db
    from puremacro.reports import df_to_markdown

    core_pkgs = ("numpy", "scipy", "pandas", "matplotlib", "requests")
    optional_pkgs = ("pyarrow", "openpyxl", "statsmodels", "linearmodels", "arch", "numba")
    unvendored_stdlib = ("sqlite3", "ssl")
    core_submodules = (
        "puremacro.data",
        "puremacro.reports",
        "puremacro.lp.jorda",
        "puremacro.lp.sensitivity",
        "puremacro.structural.lp",
        "puremacro.dsge.validation_scorecard",
        "puremacro.solvers.diagnostics",
        "puremacro.fetch.freshness",
        "puremacro.workbench",
    )

    core_checks = [_probe_module(pkg, required=True) for pkg in core_pkgs]
    optional_checks = [_probe_module(pkg, required=False) for pkg in optional_pkgs]
    stdlib_checks = [
        {
            "name": mod_name,
            "required": False,
            "status": "ok" if importlib.util.find_spec(mod_name) is not None else "missing",
        }
        for mod_name in unvendored_stdlib
    ]
    submod_checks = [_probe_module(sub, required=True) for sub in core_submodules]

    db_path = _cache_db.default_db_path()
    db_info = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "parent_exists": db_path.parent.exists(),
    }

    required_failures = [
        item["name"]
        for item in (*core_checks, *submod_checks)
        if item["status"] != "ok"
    ]
    io_missing = [
        item["name"]
        for item in optional_checks
        if item["name"] in ("pyarrow", "openpyxl") and item["status"] != "ok"
    ]
    if required_failures:
        overall = "fail"
    elif io_missing:
        overall = "warn"
    else:
        overall = "ok"

    payload: dict[str, Any] = {
        "puremacro_version": puremacro.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_root": str(Path(puremacro.__file__).resolve().parent),
        "status": overall,
        "required_failures": required_failures,
        "core_dependencies": core_checks,
        "optional_extras": optional_checks,
        "pyodide_unvendored_stdlib": stdlib_checks,
        "core_submodules": submod_checks,
        "cache_db": db_info,
    }

    if args.format == "json":
        rendered = _dumps_json(payload)
    else:
        is_es = args.lang == "es"
        heading = "# Diagnóstico de entorno puremacro" if is_es else "# puremacro Environment Doctor"
        summary_line = (
            f"Estado: **{overall.upper()}** | puremacro={puremacro.__version__} | "
            f"Python={platform.python_version()} | plataforma={platform.system()}"
            if is_es
            else
            f"Status: **{overall.upper()}** | puremacro={puremacro.__version__} | "
            f"Python={platform.python_version()} | platform={platform.system()}"
        )
        dep_rows = [
            {
                "component": item["name"],
                "tier": "required" if item["required"] else "optional",
                "status": item["status"],
                "version": item["version"] or "-",
            }
            for item in (*core_checks, *optional_checks)
        ]
        mod_rows = [
            {
                "submodule": item["name"],
                "status": item["status"],
                "error": item["error"] or "-",
            }
            for item in submod_checks
        ]
        dep_head = "## Dependencias" if is_es else "## Dependencies"
        mod_head = "## Módulos principales" if is_es else "## Core Submodules"
        cache_head = "## Caché y stdlib opcional" if is_es else "## Cache & Optional Stdlib"
        std_summary = ", ".join(f"{c['name']}={c['status']}" for c in stdlib_checks)
        lines = [
            heading,
            "",
            summary_line,
            "",
            dep_head,
            "",
            df_to_markdown(pd.DataFrame(dep_rows), index=False, digits=args.digits),
            "",
            mod_head,
            "",
            df_to_markdown(pd.DataFrame(mod_rows), index=False, digits=args.digits),
            "",
            cache_head,
            "",
            f"- cache_db: `{db_path}` (exists={db_info['exists']})",
            f"- pyodide_unvendored_stdlib: {std_summary}",
        ]
        rendered = "\n".join(lines) + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    if required_failures:
        return 1
    if args.strict and io_missing:
        return 1
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Construct the ``puremacro.cli`` argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.cli",
        description="Unified command-line interface for puremacro empirical and structural workflows.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")

    # 1. preflight
    p_pre = sub.add_parser(
        "preflight",
        help="Audit panel or time-series data readiness before estimation.",
    )
    p_pre.add_argument("input", help="Path to CSV, TSV, JSON, or Parquet input file.")
    p_pre.add_argument("--unit-col", default=None, help="Cross-sectional unit column name.")
    p_pre.add_argument("--time-col", default=None, help="Time column name.")
    p_pre.add_argument(
        "--variables",
        action="append",
        default=[],
        help="Comma-separated variable names to audit (repeatable).",
    )
    p_pre.add_argument("--freq", default=None, help="Explicit frequency override (e.g. Q, M, A).")
    p_pre.add_argument("--min-obs", type=int, default=12, help="Minimum obs per unit for ADF/KPSS. Default: 12.")
    p_pre.add_argument(
        "--max-missing-share",
        type=float,
        default=0.25,
        help="Missingness warning threshold in [0, 1]. Default: 0.25.",
    )
    p_pre.add_argument("--alpha", type=float, default=0.05, help="Significance level in (0, 1). Default: 0.05.")
    p_pre.add_argument("--no-stationarity", action="store_true", help="Skip ADF/KPSS stationarity tests.")
    p_pre.add_argument("--no-seasonality", action="store_true", help="Skip residual seasonality checks.")
    p_pre.add_argument("--strict", action="store_true", help="Exit 1 on warnings as well as errors.")
    p_pre.add_argument("--allow-errors", action="store_true", help="Exit 0 even when errors are reported.")
    _add_common_output_args(p_pre)
    p_pre.set_defaults(_handler=_run_preflight)

    # 2. lp-grid
    p_lp = sub.add_parser(
        "lp-grid",
        help="Run a Local Projection specification sensitivity grid (or full workbench dossier).",
    )
    p_lp.add_argument("input", help="Path to CSV, TSV, JSON, or Parquet dataset.")
    p_lp.add_argument("--y", required=True, help="Outcome column name.")
    p_lp.add_argument("--x", required=True, help="Shock or treatment column name.")
    p_lp.add_argument(
        "--controls",
        action="append",
        default=[],
        help="Comma-separated control columns (repeatable).",
    )
    p_lp.add_argument("--horizons", default="0..8", help="Horizon range/list (e.g. '0..8' or '0,2,4').")
    p_lp.add_argument("--lags", default="1,2,3", help="Lag grid (e.g. '1,2,4' or '1..3').")
    p_lp.add_argument(
        "--estimators",
        action="append",
        default=[],
        help="Comma-separated LP estimators (lp_hac, la_lp, smooth_lp, panel_lp, panel_lp_dk, lp_did, lp_iv).",
    )
    p_lp.add_argument("--unit-col", default=None, help="Panel unit column name.")
    p_lp.add_argument("--time-col", default=None, help="Time column name.")
    p_lp.add_argument("--ci", type=float, default=0.90, help="Confidence interval coverage in (0, 1). Default: 0.90.")
    p_lp.add_argument("--baseline-lags", type=int, default=None, help="Reference lag coordinate in grid.")
    p_lp.add_argument("--baseline-estimator", default=None, help="Reference estimator name in grid.")
    p_lp.add_argument(
        "--horizon-align",
        choices=("exact", "intersection"),
        default="exact",
        help="Horizon alignment policy. Default: exact.",
    )
    p_lp.add_argument(
        "--band-compat",
        choices=("warn", "raise", "ignore"),
        default="warn",
        help="Band compatibility policy. Default: warn.",
    )
    p_lp.add_argument(
        "--on-error",
        choices=("raise", "skip"),
        default="raise",
        help="How to handle failing grid points. Default: raise.",
    )
    p_lp.add_argument(
        "--workbench",
        action="store_true",
        help="Run the full EmpiricalStructuralDossier pipeline (preflight + LP + moment targets).",
    )
    _add_common_output_args(p_lp)
    p_lp.set_defaults(_handler=_run_lp_grid)

    # 3. compare-irfs
    p_cmp = sub.add_parser(
        "compare-irfs",
        help="Compare multiple IRF tables/files across specifications.",
    )
    p_cmp.add_argument(
        "specs",
        nargs="*",
        help="IRF CSV/JSON inputs as PATH or LABEL=PATH.",
    )
    p_cmp.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Named IRF input LABEL=PATH (repeatable).",
    )
    p_cmp.add_argument(
        "--bundle",
        default=None,
        help="Path to multi-spec JSON mapping or long table with a 'spec' column.",
    )
    p_cmp.add_argument("--baseline", default=None, help="Baseline specification label.")
    p_cmp.add_argument(
        "--horizon-align",
        choices=("exact", "intersection"),
        default="exact",
        help="Horizon alignment policy. Default: exact.",
    )
    p_cmp.add_argument(
        "--band-compat",
        choices=("warn", "raise", "ignore"),
        default="warn",
        help="Confidence band compatibility policy. Default: warn.",
    )
    p_cmp.add_argument("--cumulative", action="store_true", help="Accumulate non-cumulative IRFs.")
    p_cmp.add_argument(
        "--allow-mixed-cumulative",
        action="store_true",
        help="Allow mixing cumulative and level IRFs.",
    )
    p_cmp.add_argument("--expand-groups", action="store_true", help="Expand multi-regime/quantile specs.")
    _add_common_output_args(p_cmp)
    p_cmp.set_defaults(_handler=_run_compare_irfs)

    # 4. scorecard
    p_sc = sub.add_parser(
        "scorecard",
        help="Score a structural model IRF against empirical targets.",
    )
    p_sc.add_argument("--model-irf", required=True, help="Path to model IRF CSV or JSON.")
    p_sc.add_argument("--targets", required=True, help="Path to empirical targets CSV/JSON or MomentTargets JSON.")
    p_sc.add_argument("--covariance", default=None, help="Optional target covariance matrix CSV/JSON.")
    p_sc.add_argument("--shock", default=None, help="Shock name selector when multiple shocks exist.")
    p_sc.add_argument(
        "--variables",
        action="append",
        default=[],
        help="Comma-separated response variable subset (repeatable).",
    )
    p_sc.add_argument("--horizons", default=None, help="Optional horizon subset (e.g. '0..8').")
    p_sc.add_argument(
        "--weight-mode",
        choices=("auto", "covariance", "diagonal"),
        default="auto",
        help="Wald weighting mode. Default: auto.",
    )
    p_sc.add_argument("--ci", type=float, default=0.90, help="Target band coverage in (0, 1). Default: 0.90.")
    _add_common_output_args(p_sc)
    p_sc.set_defaults(_handler=_run_scorecard)

    # 5. freshness
    p_fr = sub.add_parser(
        "freshness",
        help="Evaluate bounded connector provenance, identity checks, and freshness.",
    )
    p_fr.add_argument(
        "--mode",
        choices=("cache", "recorded", "live"),
        default="cache",
        help="Execution mode: cache (offline), recorded (fixtures), or live. Default: cache.",
    )
    p_fr.add_argument(
        "--sources",
        action="append",
        default=[],
        help="Comma-separated cohort source IDs (repeatable).",
    )
    p_fr.add_argument("--as-of", default=None, help="Reference date (YYYY-MM-DD) for staleness.")
    p_fr.add_argument("--db-path", default=None, help="Explicit SQLite cache database path for cache mode.")
    p_fr.add_argument("--max-requests", type=int, default=5, help="Rate budget cap for live mode. Default: 5.")
    p_fr.add_argument("--min-interval", type=float, default=0.5, help="Min seconds between live requests. Default: 0.5.")
    p_fr.add_argument("--timeout", type=float, default=20.0, help="Live request timeout in seconds. Default: 20.0.")
    p_fr.add_argument(
        "--fail-on-outage",
        action="store_true",
        help="Exit 1 if any connector reports an outage status.",
    )
    _add_common_output_args(p_fr)
    p_fr.set_defaults(_handler=_run_freshness)

    # 6. doctor
    p_doc = sub.add_parser(
        "doctor",
        help="Check Python environment, runtime dependencies, and puremacro health.",
    )
    p_doc.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if optional io extras (pyarrow/openpyxl) are missing.",
    )
    _add_common_output_args(p_doc)
    p_doc.set_defaults(_handler=_run_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning an explicit integer exit status."""
    parser = create_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        return int(exc.code) if isinstance(exc.code, int) else 2

    handler: Callable[[argparse.Namespace], int] = args._handler
    try:
        return int(handler(args))
    except _CLIUsageError as exc:
        _safe_print(f"puremacro-cli error: {exc}", stream=sys.stderr)
        return 2
    except (FileNotFoundError, OSError) as exc:
        _safe_print(f"puremacro-cli io error: {exc}", stream=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        _safe_print(f"puremacro-cli error: {exc}", stream=sys.stderr)
        return 1


__all__ = ["create_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
