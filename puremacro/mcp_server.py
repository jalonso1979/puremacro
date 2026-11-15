"""Zero-dependency stdio JSON-RPC 2.0 MCP server and tool registry for puremacro.

Exposes bounded, read-only computational workflows over the Model Context
Protocol (MCP) using either newline-delimited JSON or ``Content-Length``
framing on ``stdin``/``stdout`` (``python -m puremacro.mcp_server``).

Pyodide import invariant: module top-level imports are strictly Python stdlib;
all ``puremacro.*``, ``numpy``, and ``pandas`` imports happen lazily inside tool
handlers and serialization helpers.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
from datetime import date, datetime
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, BinaryIO, TextIO

MCP_PROTOCOL_VERSION = "2024-11-05"
_SUPPORTED_PROTOCOL_VERSIONS = ("2025-03-26", "2024-11-05")

_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_TABLE_ROWS = 50_000
_MAX_TABLE_COLS = 200
_MAX_HORIZON = 120
_MAX_HANK_NA = 150
_MAX_MOMENT_DIM = 64
_MAX_GRID_NODES = 250
_MAX_LAGS = 40
_MAX_GRID_SPECS = 64
_MAX_TARIFF_SCENARIOS = 16
_MAX_COMPOSITE_DEPTH = 5
_MAX_TARIFF_RATE_PREVIEW = 500

_DATA_EXTENSIONS = frozenset({".csv", ".json"})
_DB_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3"})


class UnknownToolError(KeyError, ValueError):
    """Raised when an unknown MCP tool name is requested."""

    def __str__(self) -> str:
        return ValueError.__str__(self)


def _puremacro_version() -> str:
    import puremacro

    return str(puremacro.__version__)


def _to_json_safe(value: Any) -> Any:
    """Recursively convert Python, NumPy, pandas, and dataclass values to JSON-safe types."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(v) for v in value]

    # Lazy check for numpy / pandas / dataclasses only on non-builtin objects.
    import numpy as np
    import pandas as pd

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return None if pd.isna(value) else str(value)
    if isinstance(value, pd.DataFrame):
        df = value
        has_named_index = not (isinstance(df.index, pd.RangeIndex) and df.index.name is None)
        if has_named_index:
            col_set = {str(c) for c in df.columns}
            idx_names = [n for n in df.index.names]
            if all(n is not None and str(n) not in col_set for n in idx_names):
                df = df.reset_index()
        records = df.to_dict(orient="records")
        return [_to_json_safe(rec) for rec in records]
    if isinstance(value, pd.Series):
        return [_to_json_safe(v) for v in value.tolist()]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_json_safe(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(_to_json_safe(payload), allow_nan=False)


def _resolve_allowed_root(allowed_root: str | os.PathLike[str] | None) -> Path | None:
    raw = allowed_root
    if raw is None:
        env_val = os.environ.get("PUREMACRO_MCP_ALLOWED_ROOT", "").strip()
        if not env_val:
            return None
        raw = env_val
    root = Path(raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"allowed_root must be an existing directory, got {str(raw)!r}")
    return root


def _resolve_allowed_path(
    raw_path: Any,
    *,
    allowed_root: Path | None,
    allowed_extensions: frozenset[str],
    param_name: str = "data_path",
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{param_name} must be a non-empty string path")
    if "\x00" in raw_path:
        raise ValueError(f"{param_name} contains an invalid NUL byte")
    if allowed_root is None:
        raise PermissionError(
            f"{param_name} requires an explicit allowed_root (--allowed-root or PUREMACRO_MCP_ALLOWED_ROOT)"
        )
    rel_or_abs = Path(raw_path).expanduser()
    candidate = (rel_or_abs if rel_or_abs.is_absolute() else (allowed_root / rel_or_abs)).resolve()
    if not candidate.is_relative_to(allowed_root):
        raise PermissionError(f"{param_name} {raw_path!r} escapes allowed_root {str(allowed_root)!r}")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"{param_name} {raw_path!r} does not exist or is not a regular file")
    suffix = candidate.suffix.lower()
    if suffix not in allowed_extensions:
        raise ValueError(
            f"{param_name} extension {suffix!r} not permitted; allowed: {sorted(allowed_extensions)}"
        )
    if candidate.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"{param_name} exceeds maximum allowed size of {_MAX_FILE_BYTES} bytes"
        )
    return candidate


def _read_json_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc


def _frame_from_tabular_payload(payload: Any, *, source_name: str = "data") -> Any:
    import pandas as pd

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{source_name} rows list must be non-empty")
        if len(payload) > _MAX_TABLE_ROWS:
            raise ValueError(f"{source_name} exceeds maximum of {_MAX_TABLE_ROWS} rows")
        if not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"{source_name} rows must be JSON objects")
        df = pd.DataFrame(payload)
    elif isinstance(payload, Mapping):
        if "rows" in payload and len(payload) == 1:
            return _frame_from_tabular_payload(payload["rows"], source_name=source_name)
        if "columns" in payload and len(payload) == 1:
            return _frame_from_tabular_payload(payload["columns"], source_name=source_name)
        if not payload:
            raise ValueError(f"{source_name} columns mapping must be non-empty")
        lengths: set[int] = set()
        for col, vals in payload.items():
            if not isinstance(col, str) or not col:
                raise ValueError(f"{source_name} column names must be non-empty strings")
            if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
                raise ValueError(f"{source_name} column {col!r} must be a sequence")
            lengths.add(len(vals))
        if len(lengths) != 1:
            raise ValueError(f"{source_name} columns must all have equal length")
        n_rows = next(iter(lengths))
        if n_rows == 0:
            raise ValueError(f"{source_name} must contain at least one row")
        if n_rows > _MAX_TABLE_ROWS:
            raise ValueError(f"{source_name} exceeds maximum of {_MAX_TABLE_ROWS} rows")
        df = pd.DataFrame(dict(payload))
    else:
        raise ValueError(f"{source_name} must be a list of row objects or a column mapping")

    if df.empty:
        raise ValueError(f"{source_name} produced an empty DataFrame")
    if len(df) > _MAX_TABLE_ROWS:
        raise ValueError(f"{source_name} exceeds maximum of {_MAX_TABLE_ROWS} rows")
    if len(df.columns) > _MAX_TABLE_COLS:
        raise ValueError(f"{source_name} exceeds maximum of {_MAX_TABLE_COLS} columns")
    if not df.columns.is_unique:
        raise ValueError(f"{source_name} contains duplicate column names")
    return df


def _read_bounded_csv(path: Path, *, param_name: str = "data_path") -> Any:
    import pandas as pd

    try:
        df = pd.read_csv(path, nrows=_MAX_TABLE_ROWS + 1)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{param_name} CSV file is empty") from exc
    if len(df) > _MAX_TABLE_ROWS:
        raise ValueError(f"{param_name} exceeds maximum of {_MAX_TABLE_ROWS} rows")
    if len(df.columns) > _MAX_TABLE_COLS:
        raise ValueError(f"{param_name} exceeds maximum of {_MAX_TABLE_COLS} columns")
    if df.empty:
        raise ValueError(f"{param_name} CSV file is empty")
    return df


def _load_dataframe_from_args(
    args: Mapping[str, Any],
    *,
    allowed_root: Path | None,
) -> Any:
    has_rows = args.get("rows") is not None
    has_columns = args.get("columns") is not None
    has_path = args.get("data_path") is not None
    n_sources = int(has_rows) + int(has_columns) + int(has_path)
    if n_sources == 0:
        raise ValueError("Provide exactly one of 'rows', 'columns', or 'data_path'")
    if n_sources > 1:
        raise ValueError("Pass only one of 'rows', 'columns', or 'data_path'")

    if has_rows:
        return _frame_from_tabular_payload(args["rows"], source_name="rows")
    if has_columns:
        return _frame_from_tabular_payload(args["columns"], source_name="columns")

    path = _resolve_allowed_path(
        args["data_path"],
        allowed_root=allowed_root,
        allowed_extensions=_DATA_EXTENSIONS,
        param_name="data_path",
    )
    if path.suffix.lower() == ".csv":
        return _read_bounded_csv(path, param_name="data_path")

    payload = _read_json_file(path)
    return _frame_from_tabular_payload(payload, source_name="data_path")


def _apply_time_index(df: Any, time_col: str | None) -> Any:
    if time_col is None:
        return df
    if time_col not in df.columns:
        raise ValueError(f"time_col {time_col!r} not found in columns {list(df.columns)}")
    from puremacro.data import _coerce_time_column
    import pandas as pd

    parsed, is_dt = _coerce_time_column(df[time_col])
    if not is_dt:
        raise ValueError(f"time_col {time_col!r} could not be parsed as dates or periods")
    out = df.copy()
    out.index = pd.DatetimeIndex(parsed, name=time_col)
    return out


def _check_json_type(val: Any, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if val is None and "null" in expected:
            return
        non_null = [t for t in expected if t != "null"]
        if len(non_null) == 1:
            expected = non_null[0]
        else:
            errors: list[str] = []
            for branch_type in non_null:
                try:
                    _check_json_type(val, {**schema, "type": branch_type}, path)
                    return
                except ValueError as exc:
                    errors.append(str(exc))
            raise ValueError(f"{path}: value did not match allowed types {expected}: {'; '.join(errors)}")

    if val is None:
        if expected == "null":
            return
        raise ValueError(f"{path} must not be null")

    if expected == "string":
        if not isinstance(val, str):
            raise ValueError(f"{path} must be a string, got {type(val).__name__}")
    elif expected == "boolean":
        if not isinstance(val, bool):
            raise ValueError(f"{path} must be a boolean, got {type(val).__name__}")
    elif expected == "integer":
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{path} must be an integer, got {type(val).__name__}")
    elif expected == "number":
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"{path} must be a finite number, got {val!r}")
    elif expected == "array":
        if not isinstance(val, list):
            raise ValueError(f"{path} must be an array, got {type(val).__name__}")
        if "minItems" in schema and len(val) < int(schema["minItems"]):
            raise ValueError(f"{path} must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(val) > int(schema["maxItems"]):
            raise ValueError(f"{path} must contain at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, item in enumerate(val):
                _check_json_type(item, item_schema, f"{path}[{idx}]")
    elif expected == "object":
        if not isinstance(val, Mapping):
            raise ValueError(f"{path} must be an object, got {type(val).__name__}")
        _validate_object(val, schema, path)

    if "enum" in schema and val not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}, got {val!r}")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if "minimum" in schema and val < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}, got {val}")
        if "exclusiveMinimum" in schema and val <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path} must be > {schema['exclusiveMinimum']}, got {val}")
        if "maximum" in schema and val > schema["maximum"]:
            raise ValueError(f"{path} must be <= {schema['maximum']}, got {val}")
        if "exclusiveMaximum" in schema and val >= schema["exclusiveMaximum"]:
            raise ValueError(f"{path} must be < {schema['exclusiveMaximum']}, got {val}")


def _validate_object(obj: Mapping[str, Any], schema: Mapping[str, Any], path: str = "arguments") -> None:
    props = schema.get("properties")
    if isinstance(props, Mapping):
        if schema.get("additionalProperties") is False:
            unknown = sorted(str(k) for k in obj.keys() if k not in props)
            if unknown:
                raise ValueError(f"{path} contains unsupported field(s): {unknown}")
        for req_key in schema.get("required", ()):
            if req_key not in obj or obj[req_key] is None:
                raise ValueError(f"{path}.{req_key} is required")
        for key, val in obj.items():
            if key in props and isinstance(props[key], Mapping):
                _check_json_type(val, props[key], f"{path}.{key}")


def _validate_lp_horizons(
    horizons: Sequence[int] | None,
    horizon: int | None,
    n_lags: int | None,
    lags: int | None,
) -> None:
    if horizons is not None:
        h_list = [int(h) for h in horizons]
        if not h_list:
            raise ValueError("horizons must be non-empty")
        if len(h_list) > _MAX_HORIZON + 1 or any(h < 0 or h > _MAX_HORIZON for h in h_list):
            raise ValueError(f"horizons entries must lie in [0, {_MAX_HORIZON}]")
    if horizon is not None and (horizon < 0 or horizon > _MAX_HORIZON):
        raise ValueError(f"horizon must lie in [0, {_MAX_HORIZON}]")
    for label, lag_val in (("n_lags", n_lags), ("lags", lags)):
        if lag_val is not None and (lag_val < 0 or lag_val > _MAX_LAGS):
            raise ValueError(f"{label} must lie in [0, {_MAX_LAGS}]")


def _handle_preflight_panel(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.data import preflight_panel

    df = _load_dataframe_from_args(args, allowed_root=allowed_root)
    report = preflight_panel(
        df,
        unit_col=args.get("unit_col"),
        time_col=args.get("time_col"),
        variables=args.get("variables"),
        freq=args.get("freq"),
        min_obs=int(args.get("min_obs", 12)),
        max_missing_share=float(args.get("max_missing_share", 0.25)),
        alpha=float(args.get("alpha", 0.05)),
        run_stationarity=bool(args.get("run_stationarity", True)),
        run_seasonality=bool(args.get("run_seasonality", True)),
        digits=int(args.get("digits", 3)),
    )
    payload = json.loads(report.to_json())
    payload["summary"] = report.summary()
    if args.get("include_markdown"):
        payload["markdown"] = report.to_markdown()
    return payload


def _handle_lp_hac(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.lp import lp_hac

    df = _load_dataframe_from_args(args, allowed_root=allowed_root)
    df = _apply_time_index(df, args.get("time_col"))
    horizons = args.get("horizons")
    horizon = args.get("horizon")
    n_lags = args.get("n_lags")
    lags = args.get("lags")
    _validate_lp_horizons(horizons, horizon, n_lags, lags)

    kwargs: dict[str, Any] = {
        "y": str(args["y"]),
        "x": str(args["x"]),
        "controls": args.get("controls"),
        "allow_index_gaps": bool(args.get("allow_index_gaps", False)),
    }
    if horizons is not None:
        kwargs["horizons"] = [int(h) for h in horizons]
    if horizon is not None:
        kwargs["horizon"] = int(horizon)
    if n_lags is not None:
        kwargs["n_lags"] = int(n_lags)
    if lags is not None:
        kwargs["lags"] = int(lags)
    if args.get("alpha") is not None:
        kwargs["alpha"] = float(args["alpha"])
    if args.get("ci") is not None:
        kwargs["ci"] = float(args["ci"])

    res = lp_hac(df, **kwargs)
    horizons_out = [int(h) for h in res["h"].tolist()] if "h" in res.columns else []
    return {
        "y_name": getattr(res, "y_name", str(args["y"])),
        "x_name": getattr(res, "x_name", str(args["x"])),
        "method": getattr(res, "method", "LP-HAC"),
        "ci_level": getattr(res, "ci_level", None),
        "horizons": horizons_out,
        "table": _to_json_safe(res.reset_index(drop=True)),
        "specification": _to_json_safe(res.attrs.get("specification", {})),
        "summary": res.summary(),
    }


def _lp_estimator_map() -> dict[str, Callable[..., Any]]:
    from puremacro.lp import (
        la_lp,
        lp_did,
        lp_hac,
        lp_iv,
        panel_lp,
        panel_lp_dk,
        smooth_lp,
    )

    return {
        "lp_hac": lp_hac,
        "la_lp": la_lp,
        "smooth_lp": smooth_lp,
        "lp_iv": lp_iv,
        "panel_lp": panel_lp,
        "panel_lp_dk": panel_lp_dk,
        "lp_did": lp_did,
    }


def _resolve_lp_estimator(name: Any, est_map: Mapping[str, Callable[..., Any]]) -> Callable[..., Any]:
    if not isinstance(name, str) or name not in est_map:
        raise ValueError(
            f"Unknown LP estimator {name!r}; allowed estimators: {sorted(est_map)}"
        )
    return est_map[name]


def _handle_lp_sensitivity_grid(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.lp.sensitivity import lp_sensitivity_grid

    df = _load_dataframe_from_args(args, allowed_root=allowed_root)
    est_map = _lp_estimator_map()
    default_est_name = str(args.get("estimator", "lp_hac"))
    default_est = _resolve_lp_estimator(default_est_name, est_map)

    raw_grid = args["grid"]
    if not isinstance(raw_grid, Mapping) or not raw_grid:
        raise ValueError("grid must be a non-empty object mapping axis -> list of candidate values")

    n_combos = 1
    grid: dict[str, list[Any]] = {}
    for axis, vals in raw_grid.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"grid[{axis!r}] must be a non-empty array")
        n_combos *= len(vals)
        if n_combos > _MAX_GRID_SPECS:
            raise ValueError(f"grid size exceeds maximum of {_MAX_GRID_SPECS} specifications")
        if axis == "estimator":
            grid[axis] = [_resolve_lp_estimator(v, est_map) for v in vals]
        else:
            grid[axis] = list(vals)

    base_kwargs = dict(args.get("base_kwargs") or {})
    unit_col = args.get("unit_col")
    time_col = args.get("time_col")

    baseline_spec: dict[str, Any] | None = None
    selected_est_names: set[str] = set()
    if "estimator" in raw_grid:
        selected_est_names.update(str(v) for v in raw_grid["estimator"])
    else:
        selected_est_names.add(default_est_name)
    if args.get("baseline_spec") is not None:
        baseline_spec = dict(args["baseline_spec"])
        if "estimator" in baseline_spec:
            selected_est_names.add(str(baseline_spec["estimator"]))
            baseline_spec["estimator"] = _resolve_lp_estimator(baseline_spec["estimator"], est_map)

    uses_panel = bool(selected_est_names & {"panel_lp", "panel_lp_dk"})
    uses_lp_did = "lp_did" in selected_est_names
    uses_other_est = bool(selected_est_names - {"lp_did"})
    if uses_lp_did and uses_other_est:
        raise ValueError(
            "Cannot combine 'lp_did' (flat panel with unit/time columns) with "
            "other LP estimators in the same sensitivity grid"
        )

    if uses_panel and unit_col and time_col:
        if unit_col not in df.columns or time_col not in df.columns:
            raise ValueError("unit_col and time_col must exist in data for panel estimators")
        from puremacro.data import _coerce_time_column

        parsed_t, _ = _coerce_time_column(df[time_col])
        df = df.copy()
        df[time_col] = parsed_t
        df = df.set_index([unit_col, time_col])
        base_kwargs.setdefault("entity_level", unit_col)
        base_kwargs.setdefault("time_level", time_col)
    elif uses_lp_did:
        unit_name = unit_col or base_kwargs.get("unit")
        time_name = time_col or base_kwargs.get("time")
        if not unit_name or not time_name:
            raise ValueError(
                "estimator='lp_did' requires 'unit_col' and 'time_col' "
                "(or 'unit' and 'time' in base_kwargs)"
            )
        if unit_name not in df.columns or time_name not in df.columns:
            raise ValueError("unit_col and time_col must exist in data for lp_did")
        base_kwargs.setdefault("unit", unit_name)
        base_kwargs.setdefault("time", time_name)
    elif time_col:
        df = _apply_time_index(df, time_col)

    for axis_name in ("horizons", "horizon", "n_lags", "lags"):
        vals_to_check = grid.get(axis_name, [base_kwargs[axis_name]] if axis_name in base_kwargs else [])
        for v in vals_to_check:
            if axis_name == "horizons":
                _validate_lp_horizons(v, None, None, None)
            elif axis_name == "horizon":
                _validate_lp_horizons(None, int(v), None, None)
            elif axis_name == "n_lags":
                _validate_lp_horizons(None, None, int(v), None)
            elif axis_name == "lags":
                _validate_lp_horizons(None, None, None, int(v))

    cmp = lp_sensitivity_grid(
        df,
        y=str(args["y"]),
        x=str(args["x"]),
        grid=grid,
        estimator=default_est,
        baseline_spec=baseline_spec,
        base_kwargs=base_kwargs or None,
        on_error=args.get("on_error", "raise"),
        horizon_align=args.get("horizon_align", "exact"),
        band_compat=args.get("band_compat", "warn"),
    )
    return {
        "baseline": cmp.baseline,
        "horizons": list(cmp.horizons),
        "warnings": list(cmp.warnings),
        "errors": dict(cmp.errors),
        "metrics": _to_json_safe(cmp.metrics),
        "table": _to_json_safe(cmp.table),
        "summary": cmp.summary(),
    }


def _normalize_irf_frame(df: Any) -> Any:
    renames: dict[str, str] = {}
    if "h" not in df.columns and "horizon" in df.columns:
        renames["horizon"] = "h"
    if not any(col in df.columns for col in ("beta", "point", "response")) and "irf" in df.columns:
        renames["irf"] = "point"
    return df.rename(columns=renames) if renames else df


def _coerce_irf_spec_container(spec_payload: Any, label: str) -> Any:
    import numpy as np

    if isinstance(spec_payload, list):
        return _normalize_irf_frame(
            _frame_from_tabular_payload(spec_payload, source_name=f"results[{label!r}]")
        )
    if isinstance(spec_payload, Mapping):
        if "rows" in spec_payload:
            df = _normalize_irf_frame(
                _frame_from_tabular_payload(spec_payload["rows"], source_name=f"results[{label!r}]")
            )
            if spec_payload.get("method") is not None:
                df.attrs["method"] = str(spec_payload["method"])
            if spec_payload.get("ci_level") is not None:
                df.attrs["ci_level"] = float(spec_payload["ci_level"])
            if spec_payload.get("alpha") is not None:
                df.attrs["alpha"] = float(spec_payload["alpha"])
            if spec_payload.get("is_cumulative") is not None:
                df.attrs["is_cumulative"] = bool(spec_payload["is_cumulative"])
            return df
        if "beta" in spec_payload or "response" in spec_payload or (
            "se" in spec_payload and ("point" in spec_payload or "irf" in spec_payload)
        ):
            return _normalize_irf_frame(
                _frame_from_tabular_payload(spec_payload, source_name=f"results[{label!r}]")
            )
        out_dict = dict(spec_payload)
        if "h" not in out_dict and "horizon" in out_dict:
            out_dict["h"] = out_dict.pop("horizon")
        if "point" not in out_dict and "irf_point" not in out_dict and "irf" in out_dict:
            out_dict["point"] = out_dict.pop("irf")
        for arr_key in ("point", "irf_point", "lower", "irf_lower", "upper", "irf_upper", "h"):
            if arr_key in out_dict and out_dict[arr_key] is not None:
                out_dict[arr_key] = np.asarray(out_dict[arr_key], dtype=int if arr_key == "h" else float)
        return out_dict
    raise ValueError(f"results[{label!r}] must be a list of records or an object")


def _handle_compare_irfs(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.reports import compare_irfs

    raw_results = args.get("results")
    raw_path = args.get("data_path")
    if (raw_results is None) == (raw_path is None):
        raise ValueError("Provide exactly one of 'results' or 'data_path'")

    prepared: dict[str, Any] = {}
    if raw_path is not None:
        path = _resolve_allowed_path(
            raw_path,
            allowed_root=allowed_root,
            allowed_extensions=_DATA_EXTENSIONS,
            param_name="data_path",
        )
        if path.suffix.lower() == ".csv":
            df = _read_bounded_csv(path, param_name="data_path")
            if "spec" not in df.columns:
                raise ValueError("CSV input for compare_irfs requires a 'spec' column")
            for spec_name in dict.fromkeys(df["spec"].astype(str).tolist()):
                prepared[spec_name] = _normalize_irf_frame(
                    df[df["spec"].astype(str) == spec_name].drop(columns=["spec"])
                )
        else:
            loaded = _read_json_file(path)
            if not isinstance(loaded, Mapping) or not loaded:
                raise ValueError("JSON file for compare_irfs must be a non-empty object keyed by spec label")
            for k, v in loaded.items():
                prepared[str(k)] = _coerce_irf_spec_container(v, str(k))
    else:
        if not isinstance(raw_results, Mapping) or not raw_results:
            raise ValueError("results must be a non-empty mapping of spec label -> IRF data")
        for k, v in raw_results.items():
            prepared[str(k)] = _coerce_irf_spec_container(v, str(k))

    cmp = compare_irfs(
        prepared,
        baseline=args.get("baseline"),
        horizon_align=args.get("horizon_align", "exact"),
        target=args.get("target"),
        shock=args.get("shock"),
        regime=args.get("regime"),
        quantile=args.get("quantile"),
        ci=args.get("ci"),
        expand_groups=bool(args.get("expand_groups", False)),
        cumulative=bool(args.get("cumulative", False)),
        allow_mixed_cumulative=bool(args.get("allow_mixed_cumulative", False)),
        band_compat=args.get("band_compat", "warn"),
        digits=int(args.get("digits", 3)),
    )
    out: dict[str, Any] = {
        "baseline": cmp.baseline,
        "horizons": list(cmp.horizons),
        "warnings": list(cmp.warnings),
        "errors": dict(cmp.errors),
        "metrics": _to_json_safe(cmp.metrics),
        "table": _to_json_safe(cmp.table),
        "wide_table": _to_json_safe(cmp.to_frame("wide")),
        "summary": cmp.summary(),
    }
    if args.get("include_markdown"):
        out["markdown"] = cmp.to_markdown(digits=int(args.get("digits", 3)))
    return out


def _normalize_model_irf_frame(df: Any) -> Any:
    if "h" not in df.columns and "horizon" in df.columns:
        df = df.rename(columns={"horizon": "h"})
    if "variable" in df.columns and "h" in df.columns:
        val_col = next(
            (c for c in ("irf", "value", "response", "point", "beta") if c in df.columns),
            None,
        )
        if val_col is not None:
            if "shock" in df.columns and df["shock"].nunique() > 1:
                return {
                    str(shk): sub.pivot(index="h", columns="variable", values=val_col).reset_index()
                    for shk, sub in df.groupby("shock", sort=False)
                }
            return df.pivot(index="h", columns="variable", values=val_col).reset_index()
    return df


def _normalize_target_frame(df: Any) -> Any:
    renames: dict[str, str] = {}
    if "h" not in df.columns and "horizon" in df.columns:
        renames["horizon"] = "h"
    if (
        "variable" in df.columns
        and not any(c in df.columns for c in ("target", "value", "estimate", "point", "beta"))
        and "irf" in df.columns
    ):
        renames["irf"] = "value"
    return df.rename(columns=renames) if renames else df


def _coerce_model_irf(payload: Any) -> Any:
    if isinstance(payload, list):
        return _normalize_model_irf_frame(_frame_from_tabular_payload(payload, source_name="model_irf"))
    if isinstance(payload, Mapping):
        if any(isinstance(v, (list, Mapping)) and isinstance(v, Mapping) for v in payload.values()):
            return {
                str(shk): _normalize_model_irf_frame(
                    _frame_from_tabular_payload(val, source_name=f"model_irf[{shk!r}]")
                )
                for shk, val in payload.items()
            }
        # Mapping[str, list_of_row_dicts] vs single column-dict {h: [...], y: [...]}
        if all(isinstance(v, list) and v and isinstance(v[0], Mapping) for v in payload.values()):
            return {
                str(shk): _normalize_model_irf_frame(
                    _frame_from_tabular_payload(val, source_name=f"model_irf[{shk!r}]")
                )
                for shk, val in payload.items()
            }
        return _normalize_model_irf_frame(_frame_from_tabular_payload(payload, source_name="model_irf"))
    raise ValueError("model_irf must be a list of row records, column object, or shock->table mapping")


def _coerce_targets(payload: Any) -> Any:
    import numpy as np

    if isinstance(payload, list):
        return _normalize_target_frame(_frame_from_tabular_payload(payload, source_name="targets"))
    if isinstance(payload, Mapping):
        if "values" in payload and "labels" in payload and "covariance" in payload:
            from puremacro.structural.bridge import MomentTargets

            return MomentTargets(
                values=np.asarray(payload["values"], dtype=float),
                covariance=np.asarray(payload["covariance"], dtype=float),
                labels=tuple(str(lbl) for lbl in payload["labels"]),
                units=payload.get("units", "unspecified"),
                metadata=dict(payload.get("metadata") or {}),
            )
        if "rows" in payload and len(payload) == 1:
            return _normalize_target_frame(
                _frame_from_tabular_payload(payload["rows"], source_name="targets")
            )
        if all(isinstance(v, (list, Mapping)) for v in payload.values()) and any(
            (isinstance(v, list) and v and isinstance(v[0], Mapping)) or isinstance(v, Mapping)
            for v in payload.values()
        ):
            return {
                str(k): _normalize_target_frame(
                    _frame_from_tabular_payload(v, source_name=f"targets[{k!r}]")
                )
                for k, v in payload.items()
            }
        return _normalize_target_frame(_frame_from_tabular_payload(payload, source_name="targets"))
    raise ValueError("targets must be a MomentTargets object, long table, or variable->LP table mapping")


def _handle_score_irf_against_targets(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.dsge.validation_scorecard import score_irf_against_targets

    has_m_inline = args.get("model_irf") is not None
    has_m_path = args.get("model_irf_path") is not None
    if has_m_inline == has_m_path:
        raise ValueError("Provide exactly one of 'model_irf' or 'model_irf_path'")

    has_t_inline = args.get("targets") is not None
    has_t_path = args.get("targets_path") is not None
    if has_t_inline == has_t_path:
        raise ValueError("Provide exactly one of 'targets' or 'targets_path'")

    if has_m_path:
        m_path = _resolve_allowed_path(
            args["model_irf_path"],
            allowed_root=allowed_root,
            allowed_extensions=_DATA_EXTENSIONS,
            param_name="model_irf_path",
        )
        model_irf: Any = (
            _normalize_model_irf_frame(_read_bounded_csv(m_path, param_name="model_irf_path"))
            if m_path.suffix.lower() == ".csv"
            else _coerce_model_irf(_read_json_file(m_path))
        )
    else:
        model_irf = _coerce_model_irf(args["model_irf"])

    if has_t_path:
        t_path = _resolve_allowed_path(
            args["targets_path"],
            allowed_root=allowed_root,
            allowed_extensions=_DATA_EXTENSIONS,
            param_name="targets_path",
        )
        targets: Any = (
            _normalize_target_frame(_read_bounded_csv(t_path, param_name="targets_path"))
            if t_path.suffix.lower() == ".csv"
            else _coerce_targets(_read_json_file(t_path))
        )
    else:
        targets = _coerce_targets(args["targets"])

    res = score_irf_against_targets(
        model_irf,
        targets,
        shock=args.get("shock"),
        variables=args.get("variables"),
        horizons=args.get("horizons"),
        weight_mode=args.get("weight_mode", "auto"),
        ci_level=float(args.get("ci_level", 0.90)),
        cond_warn=float(args.get("cond_warn", 1e6)),
        hank_unit_mode=args.get("hank_unit_mode", "warn"),
        digits=int(args.get("digits", 3)),
    )
    out: dict[str, Any] = {
        "rmse": _to_json_safe(res.rmse),
        "mae": _to_json_safe(res.mae),
        "wse": _to_json_safe(res.wse),
        "chi2_stat": _to_json_safe(res.chi2_stat),
        "chi2_df": int(res.chi2_df),
        "chi2_pvalue": _to_json_safe(res.chi2_pvalue),
        "sign_concordance": _to_json_safe(res.sign_concordance),
        "band_coverage": _to_json_safe(res.band_coverage),
        "determinacy_rate": _to_json_safe(res.determinacy_rate),
        "weight_mode_used": res.weight_mode_used,
        "covariance_condition_number": _to_json_safe(res.covariance_condition_number),
        "warnings": list(res.warnings),
        "metrics": _to_json_safe(res.metrics),
        "table": _to_json_safe(res.table),
        "summary": res.summary(),
    }
    if args.get("include_markdown"):
        out["markdown"] = res.to_markdown(digits=int(args.get("digits", 3)))
    return out


def _parse_tariff_entries(raw_tariffs: Any) -> dict[tuple[str, str], float]:
    rates: dict[tuple[str, str], float] = {}
    if isinstance(raw_tariffs, list):
        if not raw_tariffs:
            raise ValueError("tariffs list must be non-empty")
        for idx, entry in enumerate(raw_tariffs):
            if not isinstance(entry, Mapping):
                raise ValueError(f"tariffs[{idx}] must be an object with exporter, sector, rate")
            exp = entry.get("exporter")
            sec = entry.get("sector", "*")
            rate = entry.get("rate")
            if not isinstance(exp, str) or not exp:
                raise ValueError(f"tariffs[{idx}].exporter must be a non-empty string")
            if not isinstance(sec, str) or not sec:
                raise ValueError(f"tariffs[{idx}].sector must be a non-empty string")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError(f"tariffs[{idx}].rate must be a non-negative number")
            rates[(exp, sec)] = float(rate)
        return rates

    if isinstance(raw_tariffs, Mapping):
        if not raw_tariffs:
            raise ValueError("tariffs mapping must be non-empty")
        for key, val in raw_tariffs.items():
            if isinstance(val, Mapping):
                for sec, rate in val.items():
                    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                        raise ValueError(f"tariff rate for ({key!r}, {sec!r}) must be numeric")
                    rates[(str(key), str(sec))] = float(rate)
            else:
                k_str = str(key)
                if ":" in k_str:
                    exp, sec = k_str.split(":", 1)
                else:
                    exp, sec = k_str, "*"
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError(f"tariff rate for {k_str!r} must be numeric")
                rates[(exp, sec)] = float(val)
        return rates

    raise ValueError("tariffs must be a list of {exporter, sector, rate} objects or a mapping")


def _build_tariff_scenario(spec: Any, *, depth: int = 0) -> Any:
    from puremacro.trade.scenarios import CompositeTariffScenario, TariffScenario

    if depth > _MAX_COMPOSITE_DEPTH:
        raise ValueError(f"CompositeTariffScenario nesting exceeds maximum depth {_MAX_COMPOSITE_DEPTH}")
    if not isinstance(spec, Mapping):
        raise ValueError("Each tariff scenario specification must be an object")

    name = str(spec.get("name", f"scenario_{depth}"))
    desc = str(spec.get("description", ""))
    if "scenarios" in spec:
        children = spec["scenarios"]
        if not isinstance(children, list) or not children:
            raise ValueError(f"Composite scenario {name!r} requires a non-empty 'scenarios' array")
        built_children = tuple(_build_tariff_scenario(ch, depth=depth + 1) for ch in children)
        return CompositeTariffScenario(
            scenarios=built_children,
            name=name,
            on_conflict=spec.get("on_conflict", "last_wins"),
            description=desc,
        )

    importer = spec.get("importer")
    if not isinstance(importer, str) or not importer:
        raise ValueError(f"Single tariff scenario {name!r} requires a non-empty 'importer' string")
    if "tariffs" not in spec:
        raise ValueError(f"Single tariff scenario {name!r} requires 'tariffs'")
    rates = _parse_tariff_entries(spec["tariffs"])
    return TariffScenario(
        importer=importer,
        tariffs=rates,
        name=name,
        description=desc,
    )


def _load_trade_calibration(args: Mapping[str, Any], allowed_root: Path | None) -> tuple[Any, str]:
    from puremacro.trade.calibration import calibrate_trade_model
    from puremacro.trade.data import ICIOData, load_icio_data, synthetic_codes, synthetic_icio_table
    import numpy as np

    source = str(args.get("model_source", "oecd_icio"))
    if source == "synthetic":
        nc = int(args.get("n_countries", 3))
        ns = int(args.get("n_sectors", 2))
        seed = int(args.get("seed", 0))
        tbl = synthetic_icio_table(nc, ns, seed=seed)
        c_codes, s_codes = synthetic_codes(nc, ns)
        c_override = args.get("country_codes")
        s_override = args.get("sector_codes")
        calib = calibrate_trade_model(
            tbl,
            country_codes=c_override or c_codes,
            sector_codes=s_override or s_codes,
        )
        return calib, "synthetic"

    if source == "oecd_icio":
        icio = load_icio_data(return_structured=True)
        if not isinstance(icio, ICIOData):
            raise TypeError("Expected structured ICIOData from load_icio_data")
        calib = calibrate_trade_model(
            icio.matrix,
            country_codes=icio.country_codes,
            sector_codes=icio.sector_codes,
        )
        return calib, "oecd_icio"

    if source == "file":
        if not args.get("data_path"):
            raise ValueError("model_source='file' requires 'data_path'")
        path = _resolve_allowed_path(
            args["data_path"],
            allowed_root=allowed_root,
            allowed_extensions=_DATA_EXTENSIONS,
            param_name="data_path",
        )
        if path.suffix.lower() == ".json":
            raw_json = _read_json_file(path)
            c_file_codes: list[str] | None
            s_file_codes: list[str] | None
            if isinstance(raw_json, Mapping):
                mat = np.asarray(raw_json["matrix"], dtype=float)
                c_file_codes = raw_json.get("country_codes")
                s_file_codes = raw_json.get("sector_codes")
            else:
                mat = np.asarray(raw_json, dtype=float)
                c_file_codes = args.get("country_codes")
                s_file_codes = args.get("sector_codes")
            calib = calibrate_trade_model(
                mat,
                country_codes=c_file_codes,
                sector_codes=s_file_codes,
            )
        else:
            icio = load_icio_data(
                path,
                return_structured=True,
                country_codes=args.get("country_codes"),
                sector_codes=args.get("sector_codes"),
            )
            if not isinstance(icio, ICIOData):
                raise TypeError("Expected structured ICIOData from load_icio_data")
            calib = calibrate_trade_model(
                icio.matrix,
                country_codes=icio.country_codes,
                sector_codes=icio.sector_codes,
            )
        return calib, "file"

    raise ValueError(f"Unsupported model_source {source!r}; expected 'oecd_icio', 'synthetic', or 'file'")


def _handle_tariff_scenario_batch(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.trade.scenarios import (
        CompositeTariffScenario,
        run_scenario_batch,
        summarize_scenario_batch,
    )

    raw_scenarios = args["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty array of scenario specifications")
    if len(raw_scenarios) > _MAX_TARIFF_SCENARIOS:
        raise ValueError(f"scenarios exceeds maximum of {_MAX_TARIFF_SCENARIOS}")

    calib, resolved_source = _load_trade_calibration(args, allowed_root)
    built = [_build_tariff_scenario(spec) for spec in raw_scenarios]

    composite_conflicts: dict[str, list[dict[str, Any]]] = {}
    resolved_tariffs: dict[str, list[dict[str, Any]]] = {}
    for scen in built:
        if isinstance(scen, CompositeTariffScenario):
            conflicts_df = scen.conflicts(calib.country_codes, calib.sector_codes)
            composite_conflicts[scen.name] = _to_json_safe(conflicts_df)
            rates_df = scen.to_frame(calib.country_codes, calib.sector_codes, nonzero_only=True)
            resolved_tariffs[scen.name] = _to_json_safe(rates_df.head(_MAX_TARIFF_RATE_PREVIEW))
        else:
            # Validate codes/rates eagerly even in resolve_only mode.
            _ = scen.rate_matrix(calib.country_codes, calib.sector_codes)

    resolve_only = bool(args.get("resolve_only", False))
    base_out: dict[str, Any] = {
        "model_source": resolved_source,
        "country_codes": list(calib.country_codes),
        "sector_codes": list(calib.sector_codes),
        "resolve_only": resolve_only,
        "composite_conflicts": composite_conflicts,
        "resolved_tariffs": resolved_tariffs,
    }
    if resolve_only:
        return base_out

    method = str(args.get("method", "equilibrated_newton"))
    allowed_methods = {"equilibrated_newton", "newton", "krylov", "sparse_lu"}
    if method not in allowed_methods:
        raise ValueError(f"Unsupported trade solver method {method!r}; allowed: {sorted(allowed_methods)}")

    batch = run_scenario_batch(
        calib,
        built,
        baseline_name=str(args.get("baseline_name", "baseline")),
        warm_start=bool(args.get("warm_start", True)),
        numeraire=args.get("numeraire"),
        sigma=float(args.get("sigma", 0.0)),
        method=method,
        tol=float(args.get("tol", 1e-5)),
        foreign_saving_units=str(args.get("foreign_saving_units", "world_income")),
    )
    cmp = summarize_scenario_batch(
        batch,
        calib,
        countries=args.get("countries"),
        include_baseline=bool(args.get("include_baseline", True)),
        digits=int(args.get("digits", 3)),
    )
    base_out.update(
        {
            "baseline": cmp.baseline,
            "countries": list(cmp.countries),
            "scenarios": list(cmp.scenarios),
            "numeraire": cmp.numeraire,
            "warnings": list(cmp.warnings),
            "scenario_summary": _to_json_safe(cmp.scenario_summary),
            "table": _to_json_safe(cmp.table),
            "summary": cmp.summary(),
        }
    )
    if args.get("include_markdown"):
        base_out["markdown"] = cmp.to_markdown(digits=int(args.get("digits", 3)))
    return base_out


def _handle_connector_freshness_report(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.fetch.freshness import RateBudget, connector_freshness_report

    mode = str(args.get("mode", "cache")).strip().lower()
    db_path_arg = args.get("db_path")
    resolved_db: Path | None = None
    if db_path_arg is not None:
        resolved_db = _resolve_allowed_path(
            db_path_arg,
            allowed_root=allowed_root,
            allowed_extensions=_DB_EXTENSIONS,
            param_name="db_path",
        )

    budget: RateBudget | None = None
    if mode == "live":
        budget = RateBudget(
            max_requests=min(int(args.get("max_requests", 5)), 5),
            min_interval_seconds=float(args.get("min_interval_seconds", 0.5)),
            timeout_seconds=float(args.get("timeout_seconds", 10.0)),
        )

    lang = str(args.get("lang", "en"))
    report = connector_freshness_report(
        mode=mode,
        sources=args.get("sources"),
        as_of=args.get("as_of"),
        db_path=resolved_db,
        rate_budget=budget,
    )
    payload = _to_json_safe(report.to_dict())
    payload["summary"] = report.summary(lang=lang)
    if args.get("include_markdown"):
        payload["markdown"] = report.to_markdown(lang=lang)
    return payload


def _handle_validation_scorecard(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    del allowed_root
    from puremacro.validation._model import run as run_case
    from puremacro.validation.runner import _discover_cases, scorecard

    cases = _discover_cases()
    available_subsystems = sorted({c.subsystem for c in cases})
    subsystem_req = args.get("subsystem")
    resolved_subsystem: str | None = None
    if subsystem_req is not None:
        s_str = str(subsystem_req).strip()
        if s_str in available_subsystems:
            resolved_subsystem = s_str
        else:
            matches = [sub for sub in available_subsystems if sub.lower() == s_str.lower()]
            if not matches:
                raise ValueError(
                    f"Unknown validation subsystem {subsystem_req!r}; available: {available_subsystems}"
                )
            resolved_subsystem = matches[0]
        cases = [c for c in cases if c.subsystem == resolved_subsystem]

    case_ids_req = args.get("case_ids")
    if case_ids_req is not None:
        wanted = [str(cid) for cid in case_ids_req]
        by_id = {c.id: c for c in cases}
        unknown_ids = [cid for cid in wanted if cid not in by_id]
        if unknown_ids:
            raise ValueError(f"Unknown validation case_id(s): {unknown_ids}")
        cases = [by_id[cid] for cid in wanted]

    if args.get("list_only"):
        return {
            "subsystem": resolved_subsystem,
            "available_subsystems": available_subsystems,
            "n_cases": len(cases),
            "cases": [
                {
                    "id": c.id,
                    "subsystem": c.subsystem,
                    "mechanism": c.mechanism.value,
                    "tol": c.tol.value,
                    "title": c.title,
                    "citation": c.citation,
                }
                for c in cases
            ],
        }

    results = [run_case(c) for c in cases]
    df = scorecard(results)
    err_by_id = {r.id: r.error for r in results}
    records = _to_json_safe(df)
    for rec in records:
        rec["error"] = err_by_id.get(rec["id"], "")

    n_passed = sum(1 for r in results if r.passed)
    n_cases = len(results)
    return {
        "subsystem": resolved_subsystem,
        "available_subsystems": available_subsystems,
        "n_cases": n_cases,
        "n_passed": n_passed,
        "n_failed": n_cases - n_passed,
        "all_passed": n_passed == n_cases,
        "records": records,
    }


def _coerce_moment_targets_obj(payload: Mapping[str, Any]) -> Any:
    import numpy as np
    from puremacro.structural.bridge import MomentTargets

    vals = payload.get("values")
    cov = payload.get("covariance")
    if not isinstance(vals, list) or not vals:
        raise ValueError("targets.values must be a non-empty list of finite numbers")
    if len(vals) > _MAX_MOMENT_DIM:
        raise ValueError(f"targets.values exceeds maximum dimension of {_MAX_MOMENT_DIM}")
    if not isinstance(cov, list) or len(cov) != len(vals):
        raise ValueError("targets.covariance must be a square q x q matrix matching values")
    labels = payload.get("labels")
    if labels is None:
        labels = [f"m{i}" for i in range(len(vals))]
    return MomentTargets(
        values=np.asarray(vals, dtype=float),
        covariance=np.asarray(cov, dtype=float),
        labels=tuple(str(lbl) for lbl in labels),
        units=payload.get("units", "unspecified"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _coerce_quadratic_curve_obj(payload: Mapping[str, Any], *, default_labels: tuple[str, ...]) -> Any:
    import numpy as np
    from puremacro.structural.robust import QuadraticMomentCurve

    offset = payload.get("offset")
    loading = payload.get("loading")
    if not isinstance(offset, list) or not isinstance(loading, list):
        raise ValueError("curve.offset and curve.loading must be numeric arrays")
    if len(offset) > _MAX_MOMENT_DIM or len(loading) > _MAX_MOMENT_DIM:
        raise ValueError(f"curve vectors exceed maximum dimension of {_MAX_MOMENT_DIM}")
    return QuadraticMomentCurve(
        offset=np.asarray(offset, dtype=float),
        loading=np.asarray(loading, dtype=float),
        labels=tuple(str(x) for x in payload.get("labels", default_labels)),
        definition_id=str(payload.get("definition_id", "mcp_quadratic_curve")),
        linear=float(payload.get("linear", 1.0)),
        quadratic=float(payload.get("quadratic", 0.0)),
        center=float(payload.get("center", 0.0)),
        scale=float(payload.get("scale", 1.0)),
        parameter_name=str(payload.get("parameter_name", "theta")),
        units=payload.get("units", "unspecified"),
    )


def _coerce_domain_pair(raw_domain: Any) -> tuple[float, float]:
    if raw_domain is None:
        return (-float("inf"), float("inf"))
    if not isinstance(raw_domain, list) or len(raw_domain) != 2:
        raise ValueError("domain must be a 2-element array [lower, upper]")
    lo = -float("inf") if raw_domain[0] is None else float(raw_domain[0])
    hi = float("inf") if raw_domain[1] is None else float(raw_domain[1])
    if math.isnan(lo) or math.isnan(hi) or lo >= hi:
        raise ValueError("domain bounds require lower < upper without NaN")
    return (lo, hi)


def _normalize_model_moments_payload(raw_mm: Any, *, target_labels: tuple[str, ...]) -> Any:
    if isinstance(raw_mm, Mapping):
        missing = [lbl for lbl in target_labels if lbl not in raw_mm]
        extra = [k for k in raw_mm if k not in target_labels]
        if missing or extra:
            raise ValueError(
                f"model_moments keys must match targets labels {list(target_labels)!r}"
            )
        return [float(raw_mm[lbl]) for lbl in target_labels]
    return raw_mm


def _handle_hank_fiscal_closures(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    del allowed_root
    from puremacro.models.hank_fiscal import compare_fiscal_closures
    from puremacro.reports import df_to_markdown

    horizon = int(args.get("horizon", 40))
    if horizon < 1 or horizon > _MAX_HORIZON:
        raise ValueError(f"horizon must lie in [1, {_MAX_HORIZON}], got {horizon}")
    n_a = int(args.get("n_a", 100))
    if n_a < 20 or n_a > _MAX_HANK_NA:
        raise ValueError(f"n_a must lie in [20, {_MAX_HANK_NA}], got {n_a}")
    a_max = float(args.get("a_max", 100.0))
    if not math.isfinite(a_max) or a_max <= 0.0:
        raise ValueError(f"a_max must be finite and positive, got {a_max}")
    phi_B = float(args.get("phi_B", 0.08))
    phi_G = float(args.get("phi_G", 0.0))
    g_ss = float(args.get("g_ss", 0.0))
    tol = float(args.get("tol", 1e-8))
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive, got {tol}")
    include_unfinanced = bool(args.get("include_unfinanced", True))

    shock_seq = args.get("shock_seq")
    if shock_seq is not None:
        if not isinstance(shock_seq, list) or not shock_seq or len(shock_seq) > _MAX_HORIZON:
            raise ValueError(f"shock_seq must be a list with 1..{_MAX_HORIZON} entries")
        shock_seq = [float(v) for v in shock_seq]

    df = compare_fiscal_closures(
        shock_seq=shock_seq,
        horizon=horizon,
        phi_B=phi_B,
        phi_G=phi_G,
        include_unfinanced=include_unfinanced,
        tol=tol,
        g_ss=g_ss,
        n_a=n_a,
        a_max=a_max,
    )
    reset_df = df.reset_index()
    out: dict[str, Any] = {
        "parameters": {
            "horizon": horizon,
            "n_a": n_a,
            "a_max": a_max,
            "phi_B": phi_B,
            "phi_G": phi_G,
            "g_ss": g_ss,
            "tol": tol,
            "include_unfinanced": include_unfinanced,
        },
        "table": _to_json_safe(reset_df),
    }
    if args.get("include_markdown"):
        out["markdown"] = df_to_markdown(
            reset_df,
            index=False,
            digits=int(args.get("digits", 3)),
        )
    return out


def _handle_cce_mg_panel_lp(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    from puremacro.data import _coerce_time_column
    from puremacro.lp.cce import cce_panel_lp
    from puremacro.reports import df_to_markdown

    df = _load_dataframe_from_args(args, allowed_root=allowed_root)
    unit_col = str(args["unit_col"])
    time_col = str(args["time_col"])
    if unit_col not in df.columns or time_col not in df.columns:
        raise ValueError(
            f"unit_col ({unit_col!r}) and time_col ({time_col!r}) must exist in columns {list(df.columns)}"
        )

    horizons = args.get("horizons")
    horizon = args.get("horizon")
    n_lags = args.get("n_lags")
    lags = args.get("lags")
    if horizons is not None and horizon is not None:
        raise ValueError("Pass either 'horizons' or 'horizon', not both")
    if n_lags is not None and lags is not None:
        raise ValueError("Pass either 'n_lags' or 'lags', not both")
    if args.get("alpha") is not None and args.get("ci") is not None:
        raise ValueError("Pass either 'alpha' or 'ci', not both")
    _validate_lp_horizons(horizons, horizon, n_lags, lags)

    parsed_t, is_dt = _coerce_time_column(df[time_col])
    work = df.copy()
    if is_dt:
        work[time_col] = parsed_t
    work = work.set_index([unit_col, time_col])

    kwargs: dict[str, Any] = {
        "y": str(args["y"]),
        "x": str(args["x"]),
        "controls": tuple(str(c) for c in (args.get("controls") or ())),
        "entity_level": unit_col,
        "time_level": time_col,
        "estimator": str(args.get("estimator", "mean_group")),
        "allow_index_gaps": bool(args.get("allow_index_gaps", False)),
    }
    if horizons is not None:
        kwargs["horizons"] = [int(h) for h in horizons]
    elif horizon is not None:
        kwargs["horizon"] = int(horizon)
    if n_lags is not None:
        kwargs["n_lags"] = int(n_lags)
    elif lags is not None:
        kwargs["lags"] = int(lags)
    if args.get("min_obs") is not None:
        kwargs["min_obs"] = int(args["min_obs"])
    if args.get("alpha") is not None:
        kwargs["alpha"] = float(args["alpha"])
    elif args.get("ci") is not None:
        kwargs["ci"] = float(args["ci"])

    res = cce_panel_lp(work, **kwargs)
    horizons_out = [int(h) for h in res["h"].tolist()] if "h" in res.columns else []
    out: dict[str, Any] = {
        "estimator": str(kwargs["estimator"]),
        "y_name": getattr(res, "y_name", str(args["y"])),
        "x_name": getattr(res, "x_name", str(args["x"])),
        "method": getattr(res, "method", "CCE-Panel-LP"),
        "ci_level": getattr(res, "ci_level", None),
        "horizons": horizons_out,
        "table": _to_json_safe(res.reset_index(drop=True)),
        "specification": _to_json_safe(res.attrs.get("specification", {})),
        "summary": res.summary(),
    }
    if args.get("include_markdown"):
        out["markdown"] = df_to_markdown(
            res.to_frame(),
            index=False,
            digits=int(args.get("digits", 3)),
        )
    return out


def _handle_identification_robust_set(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    del allowed_root
    import numpy as np
    from puremacro.reports import df_to_markdown
    from puremacro.structural.robust import (
        evaluate_moment_grid,
        invert_quadratic_moments,
        moment_restriction_test,
    )
    from puremacro.workbench import (
        _confidence_set_report_frame,
        _moment_confidence_set_to_dict,
        _moment_grid_to_dict,
        _moment_targets_to_dict,
        _moment_test_to_dict,
    )

    targets = _coerce_moment_targets_obj(args["targets"])
    cov_override = None
    if args.get("covariance") is not None:
        cov_override = np.asarray(args["covariance"], dtype=float)
    alpha = float(args.get("alpha", 0.05))
    calibration = str(args.get("calibration", "asymptotic"))

    raw_mm = args.get("model_moments")
    raw_curve = args.get("curve")
    raw_grid = args.get("grid")
    raw_grid_moments = args.get("grid_moments")

    if raw_mm is None and raw_curve is None and (raw_grid is None or raw_grid_moments is None):
        raise ValueError(
            "Provide at least one of 'model_moments', 'curve', or ('grid' with 'grid_moments')"
        )

    mt_res = None
    if raw_mm is not None:
        mm_norm = _normalize_model_moments_payload(raw_mm, target_labels=targets.labels)
        mt_res = moment_restriction_test(
            targets,
            mm_norm,
            covariance=cov_override,
            alpha=alpha,
            calibration=calibration,
        )

    curve_obj = None
    cs_res = None
    if raw_curve is not None:
        curve_obj = _coerce_quadratic_curve_obj(raw_curve, default_labels=targets.labels)
        domain = _coerce_domain_pair(args.get("domain"))
        cs_res = invert_quadratic_moments(
            curve_obj,
            targets,
            covariance=cov_override,
            domain=domain,
            alpha=alpha,
            calibration=calibration,
        )

    grid_res = None
    if raw_grid is not None:
        if len(raw_grid) > _MAX_GRID_NODES:
            raise ValueError(f"grid exceeds maximum of {_MAX_GRID_NODES} nodes")
        grid_arr = np.asarray(raw_grid, dtype=float)
        if raw_grid_moments is not None:
            if len(raw_grid_moments) != len(grid_arr):
                raise ValueError("grid_moments length must match grid length")
            moments_map = {
                float(g): _normalize_model_moments_payload(m, target_labels=targets.labels)
                for g, m in zip(grid_arr, raw_grid_moments)
            }
            pname = str(args.get("parameter_name", curve_obj.parameter_name if curve_obj is not None else "theta"))
            grid_res = evaluate_moment_grid(
                lambda th: moment_restriction_test(
                    targets,
                    moments_map[float(th)],
                    covariance=cov_override,
                    alpha=alpha,
                    calibration=calibration,
                ),
                grid_arr,
                parameter_name=pname,
            )
        elif curve_obj is not None:
            pname = str(args.get("parameter_name", curve_obj.parameter_name))
            grid_res = evaluate_moment_grid(
                lambda th: moment_restriction_test(
                    targets,
                    curve_obj(th),
                    covariance=cov_override,
                    alpha=alpha,
                    calibration=calibration,
                ),
                grid_arr,
                parameter_name=pname,
            )
        else:
            raise ValueError("grid requires either 'curve' or 'grid_moments'")

    out: dict[str, Any] = {
        "alpha": alpha,
        "calibration": calibration,
        "targets": _moment_targets_to_dict(targets),
        "moment_test": _moment_test_to_dict(mt_res) if mt_res is not None else None,
        "confidence_set": _moment_confidence_set_to_dict(cs_res) if cs_res is not None else None,
        "moment_grid": _moment_grid_to_dict(grid_res) if grid_res is not None else None,
    }
    if args.get("include_markdown"):
        digits = int(args.get("digits", 3))
        parts: list[str] = []
        if mt_res is not None:
            parts.append("## Moment Restriction Test\n\n" + df_to_markdown(mt_res.to_frame(), index=False, digits=digits))
        if cs_res is not None:
            parts.append("## Robust Confidence Set\n\n" + df_to_markdown(_confidence_set_report_frame(cs_res), index=False, digits=digits))
        if grid_res is not None:
            parts.append("## Moment Grid Tests (Sampled Nodes Only)\n\n" + df_to_markdown(grid_res.to_frame(), index=False, digits=digits))
        out["markdown"] = "\n\n".join(parts)
    return _to_json_safe(out)


def _handle_run_workbench_dossier(args: Mapping[str, Any], allowed_root: Path | None) -> dict[str, Any]:
    import numpy as np
    from puremacro.workbench import run_empirical_structural_workbench

    df = _load_dataframe_from_args(args, allowed_root=allowed_root)
    horizons = args.get("horizons")
    lags = args.get("lags", 2)
    _validate_lp_horizons(horizons, None, None, int(lags))

    est_fn = None
    if args.get("estimator") is not None:
        est_fn = _resolve_lp_estimator(str(args["estimator"]), _lp_estimator_map())

    model_irf = None
    if args.get("model_irf") is not None:
        model_irf = _coerce_model_irf(args["model_irf"])

    targets = None
    if args.get("targets") is not None:
        targets = _coerce_targets(args["targets"])

    wb_kwargs: dict[str, Any] = {
        "unit_col": args.get("unit_col"),
        "time_col": args.get("time_col"),
        "freq": args.get("freq"),
        "strict_preflight": bool(args.get("strict_preflight", True)),
        "run_stationarity": bool(args.get("run_stationarity", True)),
        "run_seasonality": bool(args.get("run_seasonality", True)),
        "horizons": [int(h) for h in horizons] if horizons is not None else tuple(range(9)),
        "lags": int(lags),
        "controls": tuple(str(c) for c in (args.get("controls") or ())),
        "ci": float(args.get("ci", 0.90)),
        "estimator": est_fn,
        "run_sensitivity": bool(args.get("run_sensitivity", True)),
        "model_irf": model_irf,
        "targets": targets,
        "lang": str(args.get("lang", "en")),
        "digits": int(args.get("digits", 3)),
    }

    if args.get("moment_test_targets") is not None:
        mt_obj = _coerce_moment_targets_obj(args["moment_test_targets"])
        wb_kwargs["moment_test_targets"] = mt_obj
    else:
        mt_obj = None

    if args.get("model_moments") is not None:
        raw_mm = args["model_moments"]
        if isinstance(raw_mm, Mapping) and mt_obj is not None:
            raw_mm = _normalize_model_moments_payload(raw_mm, target_labels=mt_obj.labels)
        wb_kwargs["model_moments"] = raw_mm
    if args.get("moment_covariance") is not None:
        wb_kwargs["moment_test_covariance"] = np.asarray(args["moment_covariance"], dtype=float)
    if args.get("moment_alpha") is not None:
        wb_kwargs["moment_test_alpha"] = float(args["moment_alpha"])
    if args.get("moment_calibration") is not None:
        wb_kwargs["moment_test_calibration"] = str(args["moment_calibration"])
    if args.get("moment_curve") is not None:
        default_lbls = mt_obj.labels if mt_obj is not None else ()
        wb_kwargs["moment_curve"] = _coerce_quadratic_curve_obj(
            args["moment_curve"],
            default_labels=default_lbls,
        )
    if args.get("moment_domain") is not None:
        wb_kwargs["moment_domain"] = _coerce_domain_pair(args["moment_domain"])
    if args.get("moment_grid") is not None:
        raw_g = args["moment_grid"]
        if len(raw_g) > _MAX_GRID_NODES:
            raise ValueError(f"moment_grid exceeds maximum of {_MAX_GRID_NODES} nodes")
        wb_kwargs["moment_grid_values"] = np.asarray(raw_g, dtype=float)

    dossier = run_empirical_structural_workbench(
        df,
        str(args["y"]),
        str(args["x"]),
        **wb_kwargs,
    )
    lang = str(args.get("lang", "en"))
    digits = int(args.get("digits", 3))
    out = _to_json_safe(dossier.to_dict())
    out["summary"] = dossier.summary(lang=lang)
    if args.get("include_markdown"):
        out["markdown"] = dossier.to_markdown(lang=lang, digits=digits)
    return out


_READONLY_ANNOTATIONS: dict[str, Any] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

_RAW_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "preflight_panel",
        "description": (
            "Audit a panel or time-series dataset for schema, duplicates, time gaps, "
            "missingness, zero variance, ADF/KPSS stationarity, and residual seasonality."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "object"},
                "data_path": {"type": "string"},
                "unit_col": {"type": ["string", "null"]},
                "time_col": {"type": ["string", "null"]},
                "variables": {"type": ["array", "null"], "items": {"type": "string"}},
                "freq": {"type": ["string", "null"]},
                "min_obs": {"type": "integer", "minimum": 6},
                "max_missing_share": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "alpha": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "run_stationarity": {"type": "boolean"},
                "run_seasonality": {"type": "boolean"},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "lp_hac",
        "description": (
            "Estimate a single-country Jorda (2005) local projection with Newey-West "
            "Bartlett HAC standard errors (bandwidth h+1)."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["y", "x"],
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "object"},
                "data_path": {"type": "string"},
                "y": {"type": "string"},
                "x": {"type": "string"},
                "controls": {"type": ["array", "null"], "items": {"type": "string"}},
                "time_col": {"type": ["string", "null"]},
                "horizons": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 0, "maximum": _MAX_HORIZON}},
                "horizon": {"type": ["integer", "null"], "minimum": 0, "maximum": _MAX_HORIZON},
                "n_lags": {"type": ["integer", "null"], "minimum": 0, "maximum": _MAX_LAGS},
                "lags": {"type": ["integer", "null"], "minimum": 0, "maximum": _MAX_LAGS},
                "alpha": {"type": ["number", "null"], "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "ci": {"type": ["number", "null"], "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "allow_index_gaps": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "lp_sensitivity_grid",
        "description": (
            "Run a bounded Local Projection specification grid across lag lengths, "
            "horizons, CI levels, or allowlisted LP estimators and align IRF comparisons."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["y", "x", "grid"],
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "object"},
                "data_path": {"type": "string"},
                "y": {"type": "string"},
                "x": {"type": "string"},
                "grid": {"type": "object"},
                "estimator": {
                    "type": "string",
                    "enum": ["lp_hac", "la_lp", "smooth_lp", "lp_iv", "panel_lp", "panel_lp_dk", "lp_did"],
                },
                "baseline_spec": {"type": ["object", "null"]},
                "base_kwargs": {"type": ["object", "null"]},
                "unit_col": {"type": ["string", "null"]},
                "time_col": {"type": ["string", "null"]},
                "on_error": {"type": "string", "enum": ["raise", "skip"]},
                "horizon_align": {"type": "string", "enum": ["exact", "intersection"]},
                "band_compat": {"type": "string", "enum": ["warn", "raise", "ignore"]},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "compare_irfs",
        "description": (
            "Align and compare impulse-response series across multiple specifications "
            "(point estimates, bands, peak horizons, cumulative sums, RMSE, sign agreement)."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {"type": "object"},
                "data_path": {"type": "string"},
                "baseline": {"type": ["string", "null"]},
                "horizon_align": {"type": "string", "enum": ["exact", "intersection"]},
                "target": {"type": ["string", "integer", "object", "null"]},
                "shock": {"type": ["string", "integer", "object", "null"]},
                "regime": {"type": ["string", "object", "null"]},
                "quantile": {"type": ["number", "object", "null"]},
                "ci": {"type": ["number", "null"], "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "expand_groups": {"type": "boolean"},
                "cumulative": {"type": "boolean"},
                "allow_mixed_cumulative": {"type": "boolean"},
                "band_compat": {"type": "string", "enum": ["warn", "raise", "ignore"]},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "score_irf_against_targets",
        "description": (
            "Score model impulse responses against empirical moment targets with "
            "rank-aware covariance or diagonal Wald chi2, RMSE, sign concordance, and band coverage."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_irf": {"type": ["array", "object"]},
                "model_irf_path": {"type": "string"},
                "targets": {"type": ["array", "object"]},
                "targets_path": {"type": "string"},
                "shock": {"type": ["string", "null"]},
                "variables": {"type": ["array", "null"], "items": {"type": "string"}},
                "horizons": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 0, "maximum": _MAX_HORIZON}},
                "weight_mode": {"type": "string", "enum": ["auto", "covariance", "diagonal"]},
                "ci_level": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "cond_warn": {"type": "number", "exclusiveMinimum": 0.0},
                "hank_unit_mode": {"type": "string", "enum": ["warn", "raw", "scale_to_target"]},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "tariff_scenario_batch",
        "description": (
            "Compose single and multi-importer CompositeTariffScenario packages, audit "
            "bilateral conflicts, solve general equilibrium, and compute Hicksian EV/CV welfare summaries."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["scenarios"],
            "properties": {
                "scenarios": {"type": "array", "minItems": 1, "maxItems": _MAX_TARIFF_SCENARIOS, "items": {"type": "object"}},
                "model_source": {"type": "string", "enum": ["oecd_icio", "synthetic", "file"]},
                "n_countries": {"type": "integer", "minimum": 2, "maximum": 12},
                "n_sectors": {"type": "integer", "minimum": 1, "maximum": 10},
                "seed": {"type": "integer", "minimum": 0},
                "data_path": {"type": "string"},
                "country_codes": {"type": ["array", "null"], "items": {"type": "string"}},
                "sector_codes": {"type": ["array", "null"], "items": {"type": "string"}},
                "baseline_name": {"type": "string"},
                "warm_start": {"type": "boolean"},
                "countries": {"type": ["array", "null"], "items": {"type": "string"}},
                "include_baseline": {"type": "boolean"},
                "sigma": {"type": "number", "minimum": 0.0, "maximum": 50.0},
                "method": {"type": "string", "enum": ["equilibrated_newton", "newton", "krylov", "sparse_lu"]},
                "tol": {"type": "number", "minimum": 1e-12, "maximum": 1e-2},
                "foreign_saving_units": {"type": "string", "enum": ["world_income", "numeraire"]},
                "numeraire": {"type": ["string", "null"]},
                "resolve_only": {"type": "boolean"},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "connector_freshness_report",
        "description": (
            "Audit provenance, domain identities, revisions, and staleness for the "
            "supported connector cohort in cache, recorded, or rate-budgeted live mode."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["cache", "recorded", "live"]},
                "sources": {"type": ["array", "null"], "items": {"type": "string"}},
                "as_of": {"type": ["string", "null"]},
                "db_path": {"type": ["string", "null"]},
                "max_requests": {"type": "integer", "minimum": 1, "maximum": 5},
                "min_interval_seconds": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                "timeout_seconds": {"type": "number", "minimum": 0.5, "maximum": 30.0},
                "lang": {"type": "string", "enum": ["en", "es"]},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "validation_scorecard",
        "description": (
            "Discover or execute puremacro validation gallery cases by subsystem or case_id "
            "and return the tolerance-margin scorecard."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subsystem": {"type": ["string", "null"]},
                "case_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                "list_only": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "hank_fiscal_closures",
        "description": (
            "Compare non-linear sequence-space HANK fiscal transitions across "
            "balanced-budget and debt-rule closures with proportional vs lump-sum taxation."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "horizon": {"type": "integer", "minimum": 1, "maximum": _MAX_HORIZON},
                "n_a": {"type": "integer", "minimum": 20, "maximum": _MAX_HANK_NA},
                "a_max": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 10000.0},
                "phi_B": {"type": "number"},
                "phi_G": {"type": "number"},
                "g_ss": {"type": "number", "minimum": 0.0},
                "tol": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
                "include_unfinanced": {"type": "boolean"},
                "shock_seq": {
                    "type": ["array", "null"],
                    "minItems": 1,
                    "maxItems": _MAX_HORIZON,
                    "items": {"type": "number"},
                },
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "cce_mg_panel_lp",
        "description": (
            "Estimate Pesaran Common Correlated Effects Mean Group or Pooled panel "
            "local projections (cce_panel_lp) to control for unobserved common factors."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["y", "x", "unit_col", "time_col"],
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "object"},
                "data_path": {"type": "string"},
                "y": {"type": "string"},
                "x": {"type": "string"},
                "unit_col": {"type": "string"},
                "time_col": {"type": "string"},
                "controls": {"type": ["array", "null"], "items": {"type": "string"}},
                "estimator": {"type": "string", "enum": ["mean_group", "pooled"]},
                "horizons": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_HORIZON + 1,
                    "items": {"type": "integer", "minimum": 0, "maximum": _MAX_HORIZON},
                },
                "horizon": {"type": "integer", "minimum": 0, "maximum": _MAX_HORIZON},
                "n_lags": {"type": "integer", "minimum": 0, "maximum": _MAX_LAGS},
                "lags": {"type": "integer", "minimum": 0, "maximum": _MAX_LAGS},
                "min_obs": {"type": ["integer", "null"], "minimum": 2},
                "alpha": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "ci": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "allow_index_gaps": {"type": "boolean"},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "identification_robust_set",
        "description": (
            "Run weak-identification-robust moment restriction tests, exact quadratic "
            "confidence set inversions, or sampled grid moment evaluations."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["targets"],
            "properties": {
                "targets": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["values", "covariance"],
                    "properties": {
                        "values": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_MOMENT_DIM,
                            "items": {"type": "number"},
                        },
                        "covariance": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_MOMENT_DIM,
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                        "labels": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "units": {"type": ["string", "array"]},
                        "metadata": {"type": ["object", "null"]},
                    },
                },
                "model_moments": {
                    "type": ["array", "object", "null"],
                },
                "covariance": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "curve": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["offset", "loading"],
                    "properties": {
                        "offset": {"type": "array", "items": {"type": "number"}},
                        "loading": {"type": "array", "items": {"type": "number"}},
                        "labels": {"type": ["array", "null"], "items": {"type": "string"}},
                        "definition_id": {"type": "string"},
                        "linear": {"type": "number"},
                        "quadratic": {"type": "number"},
                        "center": {"type": "number"},
                        "scale": {"type": "number"},
                        "parameter_name": {"type": "string"},
                        "units": {"type": ["string", "array"]},
                    },
                },
                "domain": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": ["number", "null"]},
                },
                "grid": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": _MAX_GRID_NODES,
                    "items": {"type": "number"},
                },
                "grid_moments": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": _MAX_GRID_NODES,
                },
                "parameter_name": {"type": "string"},
                "alpha": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "calibration": {"type": "string", "enum": ["asymptotic", "gaussian_known"]},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
    {
        "name": "run_workbench_dossier",
        "description": (
            "Run the unified preflight, LP/sensitivity, IRF comparison, optional structural "
            "scorecard, and identification-robust workbench pipeline returning a JSON/Markdown dossier."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["y", "x"],
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "object"},
                "data_path": {"type": "string"},
                "y": {"type": "string"},
                "x": {"type": "string"},
                "unit_col": {"type": ["string", "null"]},
                "time_col": {"type": ["string", "null"]},
                "freq": {"type": ["string", "null"]},
                "controls": {"type": ["array", "null"], "items": {"type": "string"}},
                "horizons": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_HORIZON + 1,
                    "items": {"type": "integer", "minimum": 0, "maximum": _MAX_HORIZON},
                },
                "lags": {"type": "integer", "minimum": 0, "maximum": _MAX_LAGS},
                "ci": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "estimator": {"type": ["string", "null"]},
                "strict_preflight": {"type": "boolean"},
                "run_stationarity": {"type": "boolean"},
                "run_seasonality": {"type": "boolean"},
                "run_sensitivity": {"type": "boolean"},
                "model_irf": {"type": ["array", "object", "null"]},
                "targets": {"type": ["array", "object", "null"]},
                "moment_test_targets": {"type": ["object", "null"]},
                "model_moments": {"type": ["array", "object", "null"]},
                "moment_covariance": {"type": ["array", "null"]},
                "moment_alpha": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
                "moment_calibration": {"type": "string", "enum": ["asymptotic", "gaussian_known"]},
                "moment_curve": {"type": ["object", "null"]},
                "moment_domain": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": ["number", "null"]},
                },
                "moment_grid": {
                    "type": ["array", "null"],
                    "minItems": 2,
                    "maxItems": _MAX_GRID_NODES,
                    "items": {"type": "number"},
                },
                "lang": {"type": "string", "enum": ["en", "es"]},
                "digits": {"type": "integer", "minimum": 0, "maximum": 12},
                "include_markdown": {"type": "boolean"},
            },
        },
        "annotations": dict(_READONLY_ANNOTATIONS),
    },
)

PUREMACRO_MCP_TOOLS: Mapping[str, dict[str, Any]] = MappingProxyType(
    {spec["name"]: spec for spec in _RAW_TOOLS}
)

_TOOL_HANDLERS: dict[str, Callable[[Mapping[str, Any], Path | None], dict[str, Any]]] = {
    "preflight_panel": _handle_preflight_panel,
    "lp_hac": _handle_lp_hac,
    "lp_sensitivity_grid": _handle_lp_sensitivity_grid,
    "compare_irfs": _handle_compare_irfs,
    "score_irf_against_targets": _handle_score_irf_against_targets,
    "tariff_scenario_batch": _handle_tariff_scenario_batch,
    "connector_freshness_report": _handle_connector_freshness_report,
    "validation_scorecard": _handle_validation_scorecard,
    "hank_fiscal_closures": _handle_hank_fiscal_closures,
    "cce_mg_panel_lp": _handle_cce_mg_panel_lp,
    "identification_robust_set": _handle_identification_robust_set,
    "run_workbench_dossier": _handle_run_workbench_dossier,
}


def list_tools() -> list[dict[str, Any]]:
    """Return deep copies of the declarative MCP tool descriptors."""
    return [copy.deepcopy(spec) for spec in PUREMACRO_MCP_TOOLS.values()]


def call_tool(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate arguments and invoke a registered puremacro MCP tool.

    Returns a JSON-safe dictionary containing no ``NaN`` or ``Inf`` values.
    """
    if not isinstance(name, str) or name not in PUREMACRO_MCP_TOOLS:
        raise UnknownToolError(
            f"Unknown MCP tool {name!r}; available tools: {sorted(PUREMACRO_MCP_TOOLS)}"
        )
    if arguments is None:
        args: Mapping[str, Any] = {}
    elif isinstance(arguments, Mapping):
        args = arguments
    else:
        raise ValueError(f"arguments must be an object/mapping, got {type(arguments).__name__}")

    schema = PUREMACRO_MCP_TOOLS[name]["inputSchema"]
    _validate_object(args, schema, path=f"tools.{name}.arguments")

    resolved_root = _resolve_allowed_root(allowed_root)
    handler = _TOOL_HANDLERS[name]
    raw_result = handler(args, resolved_root)
    return _to_json_safe(raw_result)


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = _to_json_safe(data)
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": _to_json_safe(result)}


def _handle_jsonrpc_message(
    msg: Any,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(msg, Mapping):
        return _jsonrpc_error(None, -32600, "Invalid Request: expected JSON object")

    has_id = "id" in msg
    req_id = msg.get("id")
    if has_id and (isinstance(req_id, bool) or (req_id is not None and not isinstance(req_id, (str, int)))):
        return _jsonrpc_error(None, -32600, "Invalid Request: 'id' must be string, integer, or null")

    if msg.get("jsonrpc") != "2.0":
        return _jsonrpc_error(req_id if has_id else None, -32600, "Invalid Request: jsonrpc must be '2.0'")

    method = msg.get("method")
    if not isinstance(method, str) or not method:
        return _jsonrpc_error(req_id if has_id else None, -32600, "Invalid Request: missing method string")

    params = msg.get("params")
    is_notification = not has_id or method.startswith("notifications/")

    if method == "notifications/initialized" or method.startswith("notifications/"):
        return None if not has_id else _jsonrpc_result(req_id, {})

    if not has_id:
        return None

    if method == "initialize":
        req_proto = params.get("protocolVersion") if isinstance(params, Mapping) else None
        negotiated = (
            req_proto
            if isinstance(req_proto, str) and req_proto in _SUPPORTED_PROTOCOL_VERSIONS
            else MCP_PROTOCOL_VERSION
        )
        return _jsonrpc_result(
            req_id,
            {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "puremacro", "version": _puremacro_version()},
                "instructions": (
                    "Bounded, read-only computational MCP server for puremacro econometric "
                    "diagnostics, local projections, CCE panel LP, IRF comparison, DSGE validation "
                    "scorecards, HANK fiscal closures, identification-robust confidence sets, "
                    "unified workbench dossiers, tariff scenarios, connector freshness, and library validation."
                ),
            },
        )

    if method == "ping":
        return _jsonrpc_result(req_id, {})

    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": list_tools()})

    if method == "tools/call":
        if not isinstance(params, Mapping):
            return _jsonrpc_error(req_id, -32602, "Invalid params: expected object for tools/call")
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return _jsonrpc_error(req_id, -32602, "Invalid params: 'name' must be a non-empty string")
        if tool_name not in PUREMACRO_MCP_TOOLS:
            return _jsonrpc_error(req_id, -32602, f"Unknown tool: {tool_name!r}")
        tool_args = params.get("arguments")
        if tool_args is not None and not isinstance(tool_args, Mapping):
            return _jsonrpc_error(req_id, -32602, "Invalid params: 'arguments' must be an object")

        try:
            payload = call_tool(tool_name, tool_args, allowed_root=allowed_root)
            return _jsonrpc_result(
                req_id,
                {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2, allow_nan=False)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )
        except Exception as exc:
            err_payload = {"error": str(exc), "error_type": type(exc).__name__}
            return _jsonrpc_result(
                req_id,
                {
                    "content": [{"type": "text", "text": json.dumps(err_payload, allow_nan=False)}],
                    "structuredContent": err_payload,
                    "isError": True,
                },
            )

    if is_notification:
        return None
    return _jsonrpc_error(req_id, -32601, f"Method not found: {method!r}")


class _FramingError(ValueError):
    """Internal sentinel for malformed transport framing or invalid UTF-8."""


def _is_binary_stream(stream: Any) -> bool:
    return not isinstance(stream, io.TextIOBase) and not hasattr(stream, "encoding")


def _read_exact_text_utf8_bytes(stream: Any, n_bytes: int) -> str:
    if n_bytes == 0:
        return ""
    chars: list[str] = []
    total_bytes = 0
    while total_bytes < n_bytes:
        ch = stream.read(1)
        if not ch:
            raise _FramingError("Unexpected EOF while reading Content-Length payload")
        chars.append(ch)
        total_bytes += len(ch.encode("utf-8"))
    if total_bytes != n_bytes:
        raise _FramingError("Content-Length byte boundary split a multi-byte UTF-8 character")
    return "".join(chars)


def _read_exact_bytes(stream: Any, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise _FramingError("Unexpected EOF while reading Content-Length payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_next_frame(stdin: Any, binary_in: bool) -> tuple[str | None, str]:
    while True:
        raw_line = stdin.readline()
        if raw_line == (b"" if binary_in else ""):
            return None, "ndjson"
        if binary_in:
            if not isinstance(raw_line, (bytes, bytearray)):
                raise _FramingError("Binary stream returned non-bytes")
            try:
                line_text = bytes(raw_line).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _FramingError(f"Invalid UTF-8 input: {exc}") from exc
        else:
            line_text = str(raw_line)

        stripped = line_text.strip()
        if not stripped:
            continue
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped, "ndjson"

        # Possible Content-Length header block
        if ":" in stripped:
            headers: dict[str, str] = {}
            cur_line = line_text
            while True:
                k, sep, v = cur_line.partition(":")
                if not sep:
                    raise _FramingError(f"Malformed header line: {cur_line!r}")
                headers[k.strip().lower()] = v.strip()
                nxt = stdin.readline()
                if nxt == (b"" if binary_in else ""):
                    raise _FramingError("Unexpected EOF inside header block")
                nxt_text = bytes(nxt).decode("utf-8") if binary_in else str(nxt)
                if nxt_text.strip() == "":
                    break
                cur_line = nxt_text

            if "content-length" not in headers:
                raise _FramingError("Missing Content-Length header")
            try:
                length = int(headers["content-length"])
            except ValueError as exc:
                raise _FramingError(f"Invalid Content-Length: {headers['content-length']!r}") from exc
            if length < 0 or length > _MAX_MESSAGE_BYTES:
                raise _FramingError(f"Content-Length out of bounds: {length}")

            if binary_in:
                raw_body = _read_exact_bytes(stdin, length)
                try:
                    return raw_body.decode("utf-8"), "content-length"
                except UnicodeDecodeError as exc:
                    raise _FramingError(f"Invalid UTF-8 body: {exc}") from exc
            return _read_exact_text_utf8_bytes(stdin, length), "content-length"

        return stripped, "ndjson"


def _write_frame(stdout: Any, payload: Any, framing: str, binary_out: bool) -> None:
    text = _json_dumps(payload)
    if framing == "content-length":
        body_bytes = text.encode("utf-8")
        header = f"Content-Length: {len(body_bytes)}\r\n\r\n"
        if binary_out:
            stdout.write(header.encode("ascii") + body_bytes)
        else:
            stdout.write(header + text)
    else:
        if binary_out:
            stdout.write((text + "\n").encode("utf-8"))
        else:
            stdout.write(text + "\n")
    stdout.flush()


def serve_stdio(
    stdin: TextIO | BinaryIO | None = None,
    stdout: TextIO | BinaryIO | None = None,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> int:
    """Serve JSON-RPC 2.0 MCP requests over stdio until EOF.

    Automatically detects and preserves per-message framing for both
    newline-delimited JSON and ``Content-Length`` headers.
    """
    in_stream: Any = (
        stdin
        if stdin is not None
        else (getattr(sys.stdin, "buffer", None) or sys.stdin)
    )
    out_stream: Any = (
        stdout
        if stdout is not None
        else (getattr(sys.stdout, "buffer", None) or sys.stdout)
    )
    binary_in = _is_binary_stream(in_stream)
    binary_out = _is_binary_stream(out_stream)

    while True:
        try:
            raw_payload, framing = _read_next_frame(in_stream, binary_in)
        except _FramingError as exc:
            _write_frame(
                out_stream,
                _jsonrpc_error(None, -32700, f"Parse error: {exc}"),
                "content-length",
                binary_out,
            )
            continue

        if raw_payload is None:
            break

        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            _write_frame(
                out_stream,
                _jsonrpc_error(None, -32700, f"Parse error: {exc}"),
                framing,
                binary_out,
            )
            continue

        if isinstance(parsed, list):
            if not parsed:
                _write_frame(
                    out_stream,
                    _jsonrpc_error(None, -32600, "Invalid Request: empty batch"),
                    framing,
                    binary_out,
                )
                continue
            responses = [
                resp
                for item in parsed
                if (resp := _handle_jsonrpc_message(item, allowed_root=allowed_root)) is not None
            ]
            if responses:
                _write_frame(out_stream, responses, framing, binary_out)
            continue

        response = _handle_jsonrpc_message(parsed, allowed_root=allowed_root)
        if response is not None:
            _write_frame(out_stream, response, framing, binary_out)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m puremacro.mcp_server``."""
    parser = argparse.ArgumentParser(
        prog="puremacro.mcp_server",
        description="Read-only computational MCP stdio server for puremacro.",
    )
    parser.add_argument(
        "--allowed-root",
        type=str,
        default=None,
        help="Directory root permitted for read-only CSV/JSON/SQLite path inputs.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print available MCP tools as JSON and exit.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport protocol (only 'stdio' is supported).",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if parsed.allowed_root is not None:
        _resolve_allowed_root(parsed.allowed_root)
    if parsed.list_tools:
        print(json.dumps({"tools": list_tools()}, indent=2, allow_nan=False))
        return 0
    return serve_stdio(allowed_root=parsed.allowed_root)


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "PUREMACRO_MCP_TOOLS",
    "UnknownToolError",
    "call_tool",
    "list_tools",
    "main",
    "serve_stdio",
]


if __name__ == "__main__":
    raise SystemExit(main())
