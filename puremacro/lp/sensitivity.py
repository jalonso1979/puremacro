"""Sensitivity grid runner for Local Projection impulse-response estimators."""
from __future__ import annotations

import inspect
import itertools
import warnings
from dataclasses import replace
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd

from puremacro.reports import IRFComparisonResult, compare_irfs
from .jorda import lp_hac

_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("lags", "n_lags"),
    ("horizon", "horizons"),
    ("ci", "alpha"),
)


def _format_axis_value(val: Any) -> str:
    if callable(val):
        return str(getattr(val, "__name__", type(val).__name__))
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        items = list(val)
        if len(items) <= 3:
            return "[" + ",".join(str(v) for v in items) + "]"
        return f"({len(items)} items)"
    return str(val)


def _format_spec_label(combo: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, val in combo.items():
        if key == "estimator":
            parts.append(_format_axis_value(val))
        else:
            parts.append(f"{key}={_format_axis_value(val)}")
    return " | ".join(parts)


def _normalize_estimator_call(
    estimator: Callable[..., Any],
    base_kwargs: Mapping[str, Any] | None,
    combo_without_est: Mapping[str, Any],
    y: str,
    x: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build signature-compatible call kwargs and target/treatment kwargs."""
    sig_params = inspect.signature(estimator).parameters
    merged: dict[str, Any] = dict(base_kwargs or {})

    # Drop conflicting sibling aliases from base_kwargs when overridden by grid combo
    for alias_a, alias_b in _ALIAS_PAIRS:
        if alias_a in combo_without_est and alias_b not in combo_without_est:
            merged.pop(alias_b, None)
        if alias_b in combo_without_est and alias_a not in combo_without_est:
            merged.pop(alias_a, None)

    merged.update(combo_without_est)

    # Signature-aware alias normalization
    if "ci" in merged and "ci" not in sig_params and "alpha" in sig_params and "alpha" not in merged:
        merged["alpha"] = 1.0 - float(merged.pop("ci"))
    elif "alpha" in merged and "alpha" not in sig_params and "ci" in sig_params and "ci" not in merged:
        merged["ci"] = 1.0 - float(merged.pop("alpha"))

    if "horizon" in merged and "horizon" not in sig_params and "horizons" in sig_params and "horizons" not in merged:
        h_val = merged.pop("horizon")
        merged["horizons"] = list(range(0, int(h_val) + 1))
    elif "horizons" in merged and "horizons" not in sig_params and "horizon" in sig_params and "horizon" not in merged:
        h_seq = [int(v) for v in merged.pop("horizons")]
        if h_seq and h_seq == list(range(0, h_seq[-1] + 1)):
            merged["horizon"] = int(h_seq[-1])
        else:
            raise ValueError(
                f"{getattr(estimator, '__name__', estimator)} expects scalar 'horizon', "
                f"cannot represent arbitrary horizons={h_seq}."
            )

    if "lags" in merged and "lags" not in sig_params and "n_lags" in sig_params and "n_lags" not in merged:
        merged["n_lags"] = merged.pop("lags")
    elif "n_lags" in merged and "n_lags" not in sig_params and "lags" in sig_params and "lags" not in merged:
        merged["lags"] = merged.pop("n_lags")

    # Map x -> estimator's treatment/shock parameter (e.g. lp_did uses treat=, Lewbel uses x_endog=)
    target_kwargs: dict[str, Any] = {"y": y}
    if "x" in sig_params:
        target_kwargs["x"] = x
    elif "treat" in sig_params:
        target_kwargs["treat"] = x
    elif "x_endog" in sig_params:
        target_kwargs["x_endog"] = x
    else:
        target_kwargs["x"] = x

    return target_kwargs, merged


def _matches_baseline(raw_combo: Mapping[str, Any], baseline_spec: Mapping[str, Any]) -> bool:
    for key, expected in baseline_spec.items():
        if key not in raw_combo:
            return False
        if raw_combo[key] != expected:
            return False
    return True


def lp_sensitivity_grid(
    data: pd.DataFrame,
    y: str,
    x: str,
    *,
    grid: Mapping[str, Sequence[Any]],
    estimator: Callable[..., Any] = lp_hac,
    baseline_spec: Mapping[str, Any] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    on_error: Literal["raise", "skip"] = "raise",
    horizon_align: Literal["exact", "intersection"] = "exact",
    band_compat: Literal["warn", "raise", "ignore"] = "warn",
) -> IRFComparisonResult:
    """Run a Local Projection specification grid across parameters and/or estimators.

    Parameters
    ----------
    data : pd.DataFrame
        Input time-series or panel dataset accepted by the chosen estimator(s).
    y, x : str
        Outcome and shock/treatment column names. For :func:`~puremacro.lp.lp_did`,
        ``x`` is automatically bound to ``treat`` (pass ``unit=`` and ``time=``
        via ``base_kwargs``).
    grid : Mapping[str, Sequence[Any]]
        Mapping from parameter names to sequences of candidate values. May include
        an ``"estimator"`` key holding candidate LP callables.
    estimator : Callable, default lp_hac
        Default LP estimator when ``"estimator"`` is not present in ``grid``.
    baseline_spec : Mapping[str, Any], optional
        Partial or full grid coordinate identifying the reference specification
        in the returned :class:`~puremacro.reports.IRFComparisonResult`.
    base_kwargs : Mapping[str, Any], optional
        Baseline keyword arguments forwarded to each estimator run before applying
        grid overrides and signature-aware alias normalization.
    on_error : {"raise", "skip"}, default "raise"
        Whether to abort on estimator failure or skip failing specs and require at
        least 2 surviving specifications.
    horizon_align : {"exact", "intersection"}, default "exact"
        Passed through to :func:`~puremacro.reports.compare_irfs`.
    band_compat : {"warn", "raise", "ignore"}, default "warn"
        Passed through to :func:`~puremacro.reports.compare_irfs`.
    """
    if not isinstance(grid, Mapping) or len(grid) == 0:
        raise ValueError("grid must be a non-empty mapping of axis name -> sequence of values.")
    if on_error not in ("raise", "skip"):
        raise ValueError(f"Unknown on_error={on_error!r}; expected 'raise' or 'skip'.")

    axis_names = list(grid.keys())
    axis_values: list[list[Any]] = []
    for k in axis_names:
        vals = grid[k]
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise ValueError(f"grid[{k!r}] must be a non-string Sequence of values.")
        val_list = list(vals)
        if not val_list:
            raise ValueError(f"grid[{k!r}] must be non-empty.")
        axis_values.append(val_list)

    results_by_label: dict[str, Any] = {}
    resolved_by_label: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    seen_labels: dict[str, int] = {}
    baseline_candidates: list[str] = []
    baseline_raw_match_found = False

    for combo_vals in itertools.product(*axis_values):
        raw_combo = dict(zip(axis_names, combo_vals))
        est = raw_combo.get("estimator", estimator)
        combo_no_est = {k: v for k, v in raw_combo.items() if k != "estimator"}

        base_label = _format_spec_label(raw_combo) or getattr(est, "__name__", "spec")
        count = seen_labels.get(base_label, 0) + 1
        seen_labels[base_label] = count
        label = base_label if count == 1 else f"{base_label} #{count}"

        full_coords = {"estimator": est, **combo_no_est}
        is_base = bool(baseline_spec is not None and _matches_baseline(full_coords, baseline_spec))
        if is_base:
            baseline_raw_match_found = True

        try:
            target_kwargs, call_kwargs = _normalize_estimator_call(
                est, base_kwargs, combo_no_est, y, x
            )
            res = est(data, **target_kwargs, **call_kwargs)
        except Exception as exc:
            if on_error == "raise":
                raise
            failures[label] = f"{type(exc).__name__}: {exc}"
            continue

        results_by_label[label] = res
        resolved_by_label[label] = {
            "estimator": getattr(est, "__name__", str(est)),
            **call_kwargs,
        }
        if is_base:
            baseline_candidates.append(label)

    if len(results_by_label) < 2:
        raise RuntimeError(
            "lp_sensitivity_grid requires at least 2 successful specifications; "
            f"only {len(results_by_label)} succeeded. Failures: {failures}"
        )

    baseline_label: str | None = None
    warn_extra: list[str] = []
    if baseline_spec is not None:
        if baseline_candidates:
            baseline_label = baseline_candidates[0]
        elif not baseline_raw_match_found:
            raise ValueError(
                f"baseline_spec={dict(baseline_spec)!r} did not match any point in grid."
            )
        else:
            fallback_msg = (
                f"Baseline spec matching {dict(baseline_spec)!r} failed during grid execution; "
                f"falling back to {next(iter(results_by_label))!r} as baseline."
            )
            warn_extra.append(fallback_msg)
            warnings.warn(fallback_msg, UserWarning, stacklevel=2)

    cmp = compare_irfs(
        results_by_label,
        baseline=baseline_label,
        horizon_align=horizon_align,
        band_compat=band_compat,
    )

    metrics = cmp.metrics.copy()
    metrics["resolved_params"] = [
        ", ".join(f"{pk}={_format_axis_value(pv)}" for pk, pv in resolved_by_label[sp].items())
        if sp in resolved_by_label else ""
        for sp in metrics["spec"]
    ]

    merged_warnings = tuple(dict.fromkeys([*cmp.warnings, *warn_extra]))
    return replace(cmp, metrics=metrics, warnings=merged_warnings, errors=dict(failures))


__all__ = ["lp_sensitivity_grid"]
