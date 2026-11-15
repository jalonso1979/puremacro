"""Unified command-line interface for puremacro workflows (``python -m puremacro.cli``).

Provides subcommands for panel preflight checks (``preflight``), LP
sensitivity grids (``lp-grid``), end-to-end empirical-structural dossiers
(``workbench``), multi-specification IRF comparison (``compare-irfs``),
structural IRF scorecards (``scorecard``), HANK fiscal closure analysis
(``hank-fiscal``), identification-robust moment sets (``robust-set``),
connector freshness checks (``freshness``), and environment diagnostics
(``doctor``).

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


def _add_common_output_args(
    parser: argparse.ArgumentParser,
    *,
    formats: tuple[str, ...] = ("json", "markdown"),
) -> None:
    fmt_desc = " or ".join(f"'{f}'" for f in formats)
    parser.add_argument(
        "--format",
        choices=formats,
        default="markdown",
        help=f"Output format ({fmt_desc}). Default: markdown.",
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


def _render_and_export_workbench(dossier: Any, args: argparse.Namespace) -> int:
    bundle_dir = getattr(args, "bundle_dir", None)
    bundle_info: dict[str, Any] | None = None
    if bundle_dir is not None:
        langs_raw = getattr(args, "bundle_langs", None)
        langs = _parse_csv_tokens([langs_raw]) if langs_raw else None
        bundle_info = dossier.to_bundle(
            bundle_dir,
            languages=langs,
            digits=args.digits,
        )

    if args.format == "html":
        rendered = dossier.to_html(lang=args.lang, digits=args.digits)
    elif args.format == "json":
        if bundle_info is not None:
            payload = dossier.to_dict()
            payload["bundle"] = {
                "languages": bundle_info["languages"],
                "checksums": bundle_info["checksums"],
                "manifest_sha256": bundle_info["manifest_sha256"],
                "schema": bundle_info["manifest"]["schema"],
            }
            rendered = _dumps_json(payload)
        else:
            rendered = dossier.to_json()
    else:
        rendered = dossier.to_markdown(lang=args.lang, digits=args.digits)

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    return 0 if dossier.passed else 1


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

    run_workbench = bool(
        args.workbench
        or getattr(args, "bundle_dir", None) is not None
    )
    if args.format == "html" and not run_workbench:
        raise _CLIUsageError("--format html requires --workbench or --bundle-dir.")

    if run_workbench:
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
        return _render_and_export_workbench(dossier, args)

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


def _parse_robust_spec_payload(
    spec_path: str,
    *,
    alpha_override: float | None = None,
    calibration_override: str | None = None,
) -> dict[str, Any]:
    """Load and schema-validate a JSON spec for moment testing / confidence inversion."""
    import numpy as np
    from puremacro.structural.bridge import MomentTargets
    from puremacro.structural.robust import QuadraticMomentCurve
    from puremacro.structural.simulation import SimulatedMoments, SimulationPlan

    raw = _load_json_raw(Path(spec_path))
    if not isinstance(raw, dict):
        raise _CLIUsageError(f"Robust spec in {spec_path} must be a JSON object.")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, dict):
        raise _CLIUsageError("Robust spec requires a 'targets' object.")
    if "values" not in raw_targets or "covariance" not in raw_targets:
        raise _CLIUsageError("Robust spec 'targets' requires 'values' and 'covariance'.")

    try:
        vals = raw_targets["values"]
        cov = raw_targets["covariance"]
        labels_raw = raw_targets.get("labels")
        if labels_raw is None:
            labels_tuple: tuple[str, ...] = tuple(f"m{i}" for i in range(len(vals)))
        else:
            if isinstance(labels_raw, (str, bytes)) or not isinstance(labels_raw, Sequence):
                raise _CLIUsageError("Robust spec 'targets.labels' must be an array of strings.")
            labels_tuple = tuple(str(lbl) for lbl in labels_raw)
        mt = MomentTargets(
            values=vals,
            covariance=cov,
            labels=labels_tuple,
            units=raw_targets.get("units", "unspecified"),
            metadata=raw_targets.get("metadata") or {},
        )
    except (ValueError, TypeError) as exc:
        raise _CLIUsageError(f"Invalid 'targets' schema: {exc}") from exc

    alpha_raw = alpha_override if alpha_override is not None else raw.get("alpha", 0.05)
    try:
        alpha = float(alpha_raw)
    except (TypeError, ValueError) as exc:
        raise _CLIUsageError(f"Invalid alpha={alpha_raw!r}: {exc}") from exc
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise _CLIUsageError(f"alpha must lie strictly in (0, 1), got {alpha!r}.")

    calibration = str(
        calibration_override if calibration_override is not None else raw.get("calibration", "asymptotic")
    )

    cov_override = raw.get("covariance")
    model_moments_raw = raw.get("model_moments")
    model_moments: Any | None = model_moments_raw
    if isinstance(model_moments_raw, dict):
        missing_labels = [lbl for lbl in mt.labels if lbl not in model_moments_raw]
        extra_labels = [k for k in model_moments_raw if k not in mt.labels]
        if missing_labels or extra_labels:
            raise _CLIUsageError(
                "Mapping 'model_moments' keys must match target labels "
                f"{list(mt.labels)!r}; missing={missing_labels!r}, extra={extra_labels!r}."
            )
        model_moments = [float(model_moments_raw[lbl]) for lbl in mt.labels]

    simulation_obj: SimulatedMoments | None = None
    assume_independent = False
    cross_covariance: Any | None = None
    if "simulation" in raw and raw["simulation"] is not None:
        raw_sim = raw["simulation"]
        if not isinstance(raw_sim, dict):
            raise _CLIUsageError("'simulation' must be a JSON object.")
        raw_plan = raw_sim.get("plan")
        if not isinstance(raw_plan, dict):
            raise _CLIUsageError("'simulation.plan' must be a JSON object.")
        try:
            plan = SimulationPlan(
                seed=int(raw_plan["seed"]),
                replications=int(raw_plan["replications"]),
                periods=int(raw_plan["periods"]),
                burn_in=int(raw_plan.get("burn_in", 0)),
                stream=int(raw_plan.get("stream", 0)),
                role=str(raw_plan.get("role", "fit")),
            )
            replicates = np.asarray(raw_sim["replicates"], dtype=float)
            sim_labels = tuple(raw_sim.get("labels", mt.labels))
            counts = (
                np.asarray(raw_sim["counts"], dtype=float)
                if raw_sim.get("counts") is not None
                else np.full(replicates.shape, float("nan"), dtype=float)
            )
            statuses = tuple(
                raw_sim.get("statuses", ["ok"] * plan.replications)
            )
            simulation_obj = SimulatedMoments(
                theta=raw_sim["theta"],
                plan=plan,
                labels=sim_labels,
                units=raw_sim.get("units", "unspecified"),
                definition_id=str(raw_sim.get("definition_id", "cli_simulation")),
                replicates=replicates,
                counts=counts,
                statuses=statuses,
                metadata=raw_sim.get("metadata") or {},
            )
            assume_independent = bool(raw_sim.get("assume_independent", False))
            cross_covariance = raw_sim.get("cross_covariance")
        except (KeyError, ValueError, TypeError) as exc:
            raise _CLIUsageError(f"Invalid 'simulation' schema: {exc}") from exc

    curve_obj: QuadraticMomentCurve | None = None
    if "curve" in raw and raw["curve"] is not None:
        raw_curve = raw["curve"]
        if not isinstance(raw_curve, dict):
            raise _CLIUsageError("'curve' must be a JSON object.")
        if "offset" not in raw_curve or "loading" not in raw_curve:
            raise _CLIUsageError("'curve' requires 'offset' and 'loading'.")
        try:
            curve_obj = QuadraticMomentCurve(
                offset=np.asarray(raw_curve["offset"], dtype=float),
                loading=np.asarray(raw_curve["loading"], dtype=float),
                labels=tuple(raw_curve.get("labels", mt.labels)),
                definition_id=str(raw_curve.get("definition_id", "cli_quadratic_curve")),
                linear=float(raw_curve.get("linear", 1.0)),
                quadratic=float(raw_curve.get("quadratic", 0.0)),
                center=float(raw_curve.get("center", 0.0)),
                scale=float(raw_curve.get("scale", 1.0)),
                parameter_name=str(raw_curve.get("parameter_name", "theta")),
                units=raw_curve.get("units", "unspecified"),
            )
        except (ValueError, TypeError) as exc:
            raise _CLIUsageError(f"Invalid 'curve' schema: {exc}") from exc

    domain: tuple[float, float] = (-float("inf"), float("inf"))
    if "domain" in raw and raw["domain"] is not None:
        raw_dom = raw["domain"]
        if not isinstance(raw_dom, list) or len(raw_dom) != 2:
            raise _CLIUsageError("'domain' must be a 2-element JSON array [lower, upper].")
        try:
            lo = -float("inf") if raw_dom[0] is None else float(raw_dom[0])
            hi = float("inf") if raw_dom[1] is None else float(raw_dom[1])
        except (TypeError, ValueError) as exc:
            raise _CLIUsageError(f"Invalid 'domain' bounds {raw_dom!r}: {exc}") from exc
        if math.isnan(lo) or math.isnan(hi) or lo >= hi:
            raise _CLIUsageError(f"Invalid 'domain' bounds {raw_dom!r}: lower must be < upper without NaN.")
        domain = (lo, hi)

    grid_values: Any | None = None
    if "grid" in raw and raw["grid"] is not None:
        raw_grid = raw["grid"]
        if not isinstance(raw_grid, list) or len(raw_grid) < 2:
            raise _CLIUsageError("'grid' must be a numeric array with at least 2 points.")
        try:
            grid_values = np.asarray(raw_grid, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _CLIUsageError(f"Invalid 'grid' values: {exc}") from exc
        if not np.all(np.isfinite(grid_values)):
            raise _CLIUsageError("'grid' values must be finite real numbers.")

    grid_moments: list[Any] | None = None
    if "grid_moments" in raw and raw["grid_moments"] is not None:
        if grid_values is None:
            raise _CLIUsageError("'grid_moments' requires 'grid'.")
        gm = raw["grid_moments"]
        if not isinstance(gm, list) or len(gm) != len(grid_values):
            raise _CLIUsageError("'grid_moments' length must match 'grid' length.")
        grid_moments = gm

    parameter_name = str(
        raw.get(
            "parameter_name",
            curve_obj.parameter_name if curve_obj is not None else "theta",
        )
    )

    return {
        "targets": mt,
        "covariance": cov_override,
        "alpha": alpha,
        "calibration": calibration,
        "model_moments": model_moments,
        "simulation": simulation_obj,
        "assume_independent": assume_independent,
        "cross_covariance": cross_covariance,
        "curve": curve_obj,
        "domain": domain,
        "grid": grid_values,
        "grid_moments": grid_moments,
        "parameter_name": parameter_name,
    }


def _run_workbench(args: argparse.Namespace) -> int:
    from puremacro.workbench import run_empirical_structural_workbench

    df = _load_dataframe(args.input)
    horizons_list = _parse_int_sequence(args.horizons, name="horizons")
    if any(h < 0 for h in horizons_list):
        raise _CLIUsageError("horizons must contain non-negative integers.")
    if not math.isfinite(float(args.ci)) or not (0.0 < float(args.ci) < 1.0):
        raise _CLIUsageError(f"--ci must lie in (0, 1), got {args.ci!r}.")
    if args.lags < 0:
        raise _CLIUsageError(f"--lags must be non-negative, got {args.lags!r}.")

    for col_name, flag in ((args.y, "--y"), (args.x, "--x")):
        if col_name not in df.columns:
            raise _CLIUsageError(f"Column {col_name!r} ({flag}) not found in {list(df.columns)!r}.")

    controls = _parse_csv_tokens(args.controls)
    missing_ctrl = [c for c in controls if c not in df.columns]
    if missing_ctrl:
        raise _CLIUsageError(f"Control column(s) {missing_ctrl!r} not found in {list(df.columns)!r}.")

    est_fn = _resolve_estimator(args.estimator) if args.estimator else None
    wb_kwargs: dict[str, Any] = {
        "unit_col": args.unit_col,
        "time_col": args.time_col,
        "freq": args.freq,
        "horizons": horizons_list,
        "lags": int(args.lags),
        "controls": controls,
        "ci": float(args.ci),
        "estimator": est_fn,
        "run_sensitivity": not args.no_sensitivity,
        "sensitivity_on_error": args.on_error,
        "horizon_align": args.horizon_align,
        "band_compat": args.band_compat,
        "lang": args.lang,
        "digits": args.digits,
    }
    if args.robust_spec is not None:
        spec = _parse_robust_spec_payload(args.robust_spec)
        wb_kwargs["moment_test_targets"] = spec["targets"]
        wb_kwargs["moment_test_covariance"] = spec["covariance"]
        wb_kwargs["moment_test_alpha"] = spec["alpha"]
        wb_kwargs["moment_test_calibration"] = spec["calibration"]
        wb_kwargs["model_moments"] = spec["model_moments"]
        wb_kwargs["simulation"] = spec["simulation"]
        wb_kwargs["moment_assume_independent"] = spec["assume_independent"]
        wb_kwargs["moment_cross_covariance"] = spec["cross_covariance"]
        wb_kwargs["moment_curve"] = spec["curve"]
        wb_kwargs["moment_domain"] = spec["domain"]
        wb_kwargs["moment_grid_values"] = spec["grid"]
        wb_kwargs["moment_parameter_name"] = spec["parameter_name"]

    dossier = run_empirical_structural_workbench(df, args.y, args.x, **wb_kwargs)
    return _render_and_export_workbench(dossier, args)


def _parse_shock_sequence(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    candidate = Path(spec)
    if candidate.exists():
        raw = _load_json_raw(candidate)
        if not isinstance(raw, list) or not raw:
            raise _CLIUsageError(f"Shock JSON file {spec} must contain a non-empty JSON array.")
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError) as exc:
            raise _CLIUsageError(f"Non-numeric entry in shock JSON {spec}: {exc}") from exc
    else:
        tokens = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
        if not tokens:
            raise _CLIUsageError("--shock cannot be empty.")
        try:
            vals = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise _CLIUsageError(f"Invalid float in --shock={spec!r}: {exc}") from exc
    if not all(math.isfinite(v) for v in vals):
        raise _CLIUsageError("--shock values must all be finite.")
    return vals


def _run_hank_fiscal(args: argparse.Namespace) -> int:
    from puremacro.models.hank_fiscal import compare_fiscal_closures, solve_hank_fiscal
    from puremacro.reports import df_to_markdown

    if args.horizon < 1:
        raise _CLIUsageError(f"--horizon must be >= 1, got {args.horizon!r}.")
    if args.n_a < 10:
        raise _CLIUsageError(f"--n-a must be >= 10, got {args.n_a!r}.")
    for nm, val in (
        ("--phi-b", args.phi_b),
        ("--phi-g", args.phi_g),
        ("--g-ss", args.g_ss),
        ("--a-max", args.a_max),
        ("--tol", args.tol),
    ):
        if not math.isfinite(float(val)):
            raise _CLIUsageError(f"{nm} must be finite, got {val!r}.")
    if float(args.a_max) <= 0.0:
        raise _CLIUsageError(f"--a-max must be positive, got {args.a_max!r}.")
    if float(args.tol) <= 0.0:
        raise _CLIUsageError(f"--tol must be positive, got {args.tol!r}.")

    shock_seq = _parse_shock_sequence(args.shock)
    is_es = args.lang == "es"

    if args.mode == "compare":
        try:
            df = compare_fiscal_closures(
                shock_seq=shock_seq,
                horizon=int(args.horizon),
                phi_B=float(args.phi_b),
                phi_G=float(args.phi_g),
                include_unfinanced=not args.no_unfinanced,
                tol=float(args.tol),
                g_ss=float(args.g_ss),
                n_a=int(args.n_a),
                a_max=float(args.a_max),
            )
        except ValueError as exc:
            raise _CLIUsageError(str(exc)) from exc

        reset_df = df.reset_index()
        payload = {
            "schema": "puremacro.hank_fiscal/1",
            "mode": "compare",
            "parameters": {
                "horizon": int(args.horizon),
                "phi_B": float(args.phi_b),
                "phi_G": float(args.phi_g),
                "g_ss": float(args.g_ss),
                "n_a": int(args.n_a),
                "a_max": float(args.a_max),
                "tol": float(args.tol),
                "include_unfinanced": not args.no_unfinanced,
            },
            "scenarios": _json_sanitize(reset_df),
        }
        if args.format == "json":
            rendered = _dumps_json(payload)
        else:
            title = (
                "Comparación fiscal HANK en espacio de secuencias"
                if is_es
                else "Sequence-Space HANK Fiscal Closure Comparison"
            )
            rendered = (
                f"# {title}\n\n"
                f"- horizon={args.horizon}, phi_B={args.phi_b}, phi_G={args.phi_g}, "
                f"n_a={args.n_a}, a_max={args.a_max}\n\n"
                + df_to_markdown(reset_df, index=False, digits=args.digits)
                + "\n"
            )
        _write_or_print(rendered, output=args.output, quiet=args.quiet)
        financed = df[df["closure"] != "unfinanced_goods_only"]
        ok = bool(financed["status"].astype(str).str.startswith("PASS").all()) if not financed.empty else False
        return 0 if ok else 1

    try:
        res = solve_hank_fiscal(
            shock_seq=shock_seq,
            closure=args.closure,
            tax_type=args.tax_type,
            horizon=int(args.horizon),
            phi_B=float(args.phi_b),
            phi_G=float(args.phi_g),
            g_ss=float(args.g_ss),
            tol=float(args.tol),
            n_a=int(args.n_a),
            a_max=float(args.a_max),
            allow_unstable_rule=bool(args.allow_unstable_rule),
        )
    except ValueError as exc:
        raise _CLIUsageError(str(exc)) from exc

    import pandas as pd

    paths_df = pd.DataFrame(
        {
            "t": list(range(int(res.horizon))),
            "dY": res.dY,
            "dC": res.dC,
            "dG": res.dG,
            "dT": res.dT,
            "dB": res.dB,
            "residuals": res.residuals,
        }
    )
    metrics_row = {
        "closure": res.closure,
        "tax_type": res.tax_type,
        "horizon": int(res.horizon),
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "status": str(res.solver_diagnostics.overall_status),
        "impact_multiplier": float(res.impact_multiplier),
        "cumulative_multiplier": float(res.cumulative_multiplier),
        "present_value_multiplier": float(res.present_value_multiplier),
        "goods_market_residual": float(res.goods_market_residual),
        "gov_budget_residual": float(res.gov_budget_residual),
        "gov_intertemporal_residual": float(res.gov_intertemporal_residual),
        "asset_market_residual": float(res.asset_market_residual),
        "terminal_debt_gap": float(res.terminal_debt_gap),
        "terminal_discounted_debt_gap": float(res.terminal_discounted_debt_gap),
    }
    payload = {
        "schema": "puremacro.hank_fiscal/1",
        "mode": "single",
        "metrics": _json_sanitize(metrics_row),
        "steady_state": _json_sanitize(res.steady_state),
        "solver_diagnostics": _json_sanitize(res.solver_diagnostics.to_dict()),
        "paths": _json_sanitize(paths_df),
    }
    if args.format == "json":
        rendered = _dumps_json(payload)
    else:
        title = (
            f"Transición fiscal HANK: {res.closure} ({res.tax_type})"
            if is_es
            else f"HANK Fiscal Transition: {res.closure} ({res.tax_type})"
        )
        summary_df = pd.DataFrame([metrics_row])
        rendered = (
            f"# {title}\n\n"
            f"```\n{res.summary()}\n```\n\n"
            + df_to_markdown(summary_df, index=False, digits=args.digits)
            + "\n"
        )
    _write_or_print(rendered, output=args.output, quiet=args.quiet)
    return 0 if bool(res.solver_diagnostics.is_pass) else 1


def _run_robust_set(args: argparse.Namespace) -> int:
    from puremacro.reports import df_to_markdown
    from puremacro.structural.robust import (
        evaluate_moment_grid,
        invert_quadratic_moments,
        moment_restriction_test,
        simulated_moment_test,
    )
    from puremacro.workbench import (
        _confidence_set_report_frame,
        _moment_confidence_set_to_dict,
        _moment_grid_to_dict,
        _moment_targets_to_dict,
        _moment_test_to_dict,
    )

    spec = _parse_robust_spec_payload(
        args.spec,
        alpha_override=args.alpha,
        calibration_override=args.calibration,
    )
    targets = spec["targets"]
    cov = spec["covariance"]
    alpha = spec["alpha"]
    cal = spec["calibration"]
    model_moments = spec["model_moments"]
    sim = spec["simulation"]
    curve = spec["curve"]
    grid = spec["grid"]
    grid_moments = spec["grid_moments"]

    if (
        model_moments is None
        and sim is None
        and curve is None
        and (grid is None or grid_moments is None)
    ):
        raise _CLIUsageError(
            "Robust spec must specify at least one of 'model_moments', 'simulation', "
            "'curve', or ('grid' with 'grid_moments')."
        )

    mt_res = None
    if model_moments is not None and sim is not None:
        raise _CLIUsageError("Specify either 'model_moments' or 'simulation' in spec, not both.")
    if model_moments is not None:
        try:
            mt_res = moment_restriction_test(
                targets,
                model_moments,
                covariance=cov,
                alpha=alpha,
                calibration=cal,
            )
        except ValueError as exc:
            raise _CLIUsageError(str(exc)) from exc
    elif sim is not None:
        mt_res = simulated_moment_test(
            targets,
            sim,
            assume_independent=spec["assume_independent"],
            cross_covariance=spec["cross_covariance"],
            alpha=alpha,
        )

    cs_res = None
    if curve is not None:
        try:
            cs_res = invert_quadratic_moments(
                curve,
                targets,
                covariance=cov,
                domain=spec["domain"],
                alpha=alpha,
                calibration=cal,
            )
        except ValueError as exc:
            raise _CLIUsageError(str(exc)) from exc

    grid_res = None
    if grid is not None:
        if grid_moments is not None:
            moments_by_node = {
                float(g): m for g, m in zip(grid, grid_moments)
            }

            def _eval_node(theta: float) -> Any:
                return moment_restriction_test(
                    targets,
                    moments_by_node[float(theta)],
                    covariance=cov,
                    alpha=alpha,
                    calibration=cal,
                )

            try:
                grid_res = evaluate_moment_grid(
                    _eval_node,
                    grid,
                    parameter_name=spec["parameter_name"],
                )
            except ValueError as exc:
                raise _CLIUsageError(str(exc)) from exc
        elif curve is not None:
            def _eval_curve_node(theta: float) -> Any:
                return moment_restriction_test(
                    targets,
                    curve(theta),
                    covariance=cov,
                    alpha=alpha,
                    calibration=cal,
                )

            try:
                grid_res = evaluate_moment_grid(
                    _eval_curve_node,
                    grid,
                    parameter_name=curve.parameter_name,
                )
            except ValueError as exc:
                raise _CLIUsageError(str(exc)) from exc
        else:
            raise _CLIUsageError("'grid' requires either 'curve' or 'grid_moments'.")

    payload: dict[str, Any] = {
        "schema": "puremacro.robust_set/1",
        "alpha": float(alpha),
        "calibration": str(cal),
        "targets": _moment_targets_to_dict(targets),
        "moment_test": _moment_test_to_dict(mt_res) if mt_res is not None else None,
        "confidence_set": _moment_confidence_set_to_dict(cs_res) if cs_res is not None else None,
        "moment_grid": _moment_grid_to_dict(grid_res) if grid_res is not None else None,
    }
    if args.format == "json":
        rendered = _dumps_json(payload)
    else:
        is_es = args.lang == "es"
        parts: list[str] = [
            f"# {'Conjunto robusto de momentos' if is_es else 'Identification-Robust Moment Analysis'}",
            "",
            f"- alpha={alpha}, calibration={cal}",
            "",
        ]
        if mt_res is not None:
            parts.append(
                f"## {'Prueba de restricción de momentos' if is_es else 'Moment Restriction Test'}"
            )
            parts.append("")
            parts.append(df_to_markdown(mt_res.to_frame(), index=False, digits=args.digits))
            parts.append("")
        if cs_res is not None:
            parts.append(
                f"## {'Conjunto de confianza robusto' if is_es else 'Inverted Confidence Set'}"
            )
            parts.append("")
            parts.append(
                df_to_markdown(
                    _confidence_set_report_frame(cs_res),
                    index=False,
                    digits=args.digits,
                )
            )
            parts.append("")
        if grid_res is not None:
            parts.append(
                f"## {'Malla de prueba de momentos (solo nodos)' if is_es else 'Moment Grid Tests (Sampled Nodes Only)'}"
            )
            parts.append("")
            parts.append(
                "- continuous_complete=False (unsampled interiors are not certified)"
            )
            parts.append("")
            parts.append(df_to_markdown(grid_res.to_frame(), index=False, digits=args.digits))
            parts.append("")
            parts.append(
                df_to_markdown(grid_res.unresolved_cells(), index=False, digits=args.digits)
            )
            parts.append("")
        rendered = "\n".join(parts).rstrip() + "\n"

    _write_or_print(rendered, output=args.output, quiet=args.quiet)
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
    p_lp.add_argument(
        "--bundle-dir",
        default=None,
        help="Export bilingual HTML/Markdown/LaTeX/JSON dossier bundle with SHA-256 manifest to this directory.",
    )
    p_lp.add_argument(
        "--bundle-langs",
        default=None,
        help="Comma-separated bundle languages (default: 'en,es').",
    )
    _add_common_output_args(p_lp, formats=("json", "markdown", "html"))
    p_lp.set_defaults(_handler=_run_lp_grid)

    # 2b. workbench
    p_wb = sub.add_parser(
        "workbench",
        help="Run the unified empirical-structural workbench dossier pipeline.",
    )
    p_wb.add_argument("input", help="Path to CSV, TSV, JSON, or Parquet dataset.")
    p_wb.add_argument("--y", required=True, help="Outcome column name.")
    p_wb.add_argument("--x", required=True, help="Shock or treatment column name.")
    p_wb.add_argument(
        "--controls",
        action="append",
        default=[],
        help="Comma-separated control columns (repeatable).",
    )
    p_wb.add_argument("--horizons", default="0..8", help="Horizon range/list (e.g. '0..8' or '0,2,4').")
    p_wb.add_argument("--lags", type=int, default=2, help="Baseline LP lag length. Default: 2.")
    p_wb.add_argument("--estimator", default=None, help="LP estimator name (default: panel_lp or lp_hac).")
    p_wb.add_argument("--unit-col", default=None, help="Panel unit column name.")
    p_wb.add_argument("--time-col", default=None, help="Time column name.")
    p_wb.add_argument("--freq", default=None, help="Frequency declaration (e.g. Q, M, A).")
    p_wb.add_argument("--ci", type=float, default=0.90, help="Confidence level in (0, 1). Default: 0.90.")
    p_wb.add_argument("--no-sensitivity", action="store_true", help="Disable default lag sensitivity grid.")
    p_wb.add_argument(
        "--horizon-align",
        choices=("exact", "intersection"),
        default="intersection",
        help="Horizon alignment policy. Default: intersection.",
    )
    p_wb.add_argument(
        "--band-compat",
        choices=("warn", "raise", "ignore"),
        default="warn",
        help="Band compatibility policy. Default: warn.",
    )
    p_wb.add_argument(
        "--on-error",
        choices=("raise", "skip"),
        default="skip",
        help="How to handle failing sensitivity points. Default: skip.",
    )
    p_wb.add_argument(
        "--robust-spec",
        default=None,
        help="Optional JSON specification for identification-robust moment testing / confidence inversion.",
    )
    p_wb.add_argument(
        "--bundle-dir",
        default=None,
        help="Export bilingual HTML/Markdown/LaTeX/JSON dossier bundle with SHA-256 manifest to this directory.",
    )
    p_wb.add_argument(
        "--bundle-langs",
        default=None,
        help="Comma-separated bundle languages (default: 'en,es').",
    )
    _add_common_output_args(p_wb, formats=("json", "markdown", "html"))
    p_wb.set_defaults(_handler=_run_workbench)

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

    # 7. hank-fiscal
    p_hf = sub.add_parser(
        "hank-fiscal",
        help="Solve or compare financed HANK sequence-space fiscal closures.",
    )
    p_hf.add_argument(
        "--mode",
        choices=("compare", "single"),
        default="compare",
        help="Execution mode: compare all closures or run a single solve. Default: compare.",
    )
    p_hf.add_argument(
        "--closure",
        choices=("balanced_budget", "debt_rule"),
        default="balanced_budget",
        help="Fiscal financing closure for single mode. Default: balanced_budget.",
    )
    p_hf.add_argument(
        "--tax-type",
        choices=("proportional", "lump_sum"),
        default="proportional",
        help="Tax adjustment channel for single mode. Default: proportional.",
    )
    p_hf.add_argument("--horizon", type=int, default=40, help="Transition horizon T (>= 1). Default: 40.")
    p_hf.add_argument("--phi-b", type=float, default=0.08, help="Debt feedback coefficient phi_B. Default: 0.08.")
    p_hf.add_argument("--phi-g", type=float, default=0.0, help="Contemporaneous spending-tax offset phi_G. Default: 0.0.")
    p_hf.add_argument("--g-ss", type=float, default=0.0, help="Steady-state government spending level. Default: 0.0.")
    p_hf.add_argument("--n-a", type=int, default=100, help="Asset grid size (>= 10). Default: 100.")
    p_hf.add_argument("--a-max", type=float, default=100.0, help="Upper asset grid bound. Default: 100.0.")
    p_hf.add_argument("--tol", type=float, default=1e-8, help="Convergence/residual tolerance. Default: 1e-8.")
    p_hf.add_argument("--shock", default=None, help="Comma-separated dG shock values or path to JSON array.")
    p_hf.add_argument(
        "--no-unfinanced",
        action="store_true",
        help="Omit unfinanced goods-market reference row in compare mode.",
    )
    p_hf.add_argument(
        "--allow-unstable-rule",
        action="store_true",
        help="Return non-stationary explosive rule diagnostic instead of raising error (single mode).",
    )
    _add_common_output_args(p_hf)
    p_hf.set_defaults(_handler=_run_hank_fiscal)

    # 8. robust-set
    p_rs = sub.add_parser(
        "robust-set",
        help="Run identification-robust moment restriction tests and confidence set inversions from a JSON spec.",
    )
    p_rs.add_argument("spec", help="Path to JSON specification file defining targets, moments/curve/grid.")
    p_rs.add_argument("--alpha", type=float, default=None, help="Override significance level alpha in (0, 1).")
    p_rs.add_argument(
        "--calibration",
        choices=("asymptotic", "gaussian_known"),
        default=None,
        help="Override test critical-value calibration.",
    )
    _add_common_output_args(p_rs)
    p_rs.set_defaults(_handler=_run_robust_set)

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
