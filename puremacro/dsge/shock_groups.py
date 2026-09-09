"""Grouped Historical Shock Decomposition for DSGE Models.

Groups structural shock contributions according to Dynare shock_groups blocks
or user-specified category mappings (e.g. supply, demand, monetary policy, others),
maintaining exact adding-up balance:
    sum_{g} y_t^(g) + y_t^(init) = y_t^(obs)
to machine precision (<= 1e-12).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._results import ShockDecompositionResult
from .decomposition import compute_shock_decomposition

__all__ = ["shock_groups_decomposition", "ShockDecompositionResult"]


def shock_groups_decomposition(
    model: Any,
    data: pd.DataFrame,
    groups: Mapping[str, Sequence[str]] | None = None,
    *,
    initial_state: np.ndarray | None = None,
    sigma: float | Mapping[str, float] | Sequence[float] | None = None,
) -> ShockDecompositionResult:
    """Decompose historical observed data into grouped shock contributions and initial condition.

    Parameters
    ----------
    model : LinearModel, DynareDR, or solved DSGE model
        Solved model instance with declared shocks and decision rules.
    data : pd.DataFrame
        Observed or simulated time-series data with column names matching model variables.
    groups : Mapping[str, Sequence[str]], optional
        Mapping from group names (e.g. 'demand', 'supply', 'monetary') to sequences
        of shock names. If None, uses `model.shock_groups` if defined; otherwise
        each shock is treated as its own group.
    initial_state : np.ndarray, optional
        Initial predetermined state vector at t=0 before innovations. If None,
        estimated via the Kalman smoother initialized from unconditional stationary distribution.
    sigma : float | Mapping[str, float] | Sequence[float], optional
        Shock standard deviations. Defaults to model's declared covariance.

    Returns
    -------
    ShockDecompositionResult
        Frozen presentation dataclass with `.components` mapping each group name,
        'Others' (if unassigned shocks exist), and 'initial' (initial condition + steady state)
        to DataFrames of shape (T, n_variables), satisfying:
            sum(components[g][var]) = data[var]
        to machine precision (<= 1e-12).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("data DataFrame is empty")

    model_shocks = list(getattr(model, "shocks", []))
    model_vars = list(getattr(model, "variables", []))

    # 1. Resolve shock groups
    resolved_groups: dict[str, list[str]] = {}
    if groups is not None:
        for g_name, s_list in groups.items():
            resolved_groups[str(g_name)] = [str(s) for s in s_list]
    elif hasattr(model, "shock_groups") and model.shock_groups:
        for g_name, s_list in model.shock_groups.items():
            resolved_groups[str(g_name)] = [str(s) for s in s_list]
    else:
        for s in model_shocks:
            resolved_groups[str(s)] = [str(s)]

    # Validate that grouped shocks exist in model
    for g_name, s_list in resolved_groups.items():
        for s in s_list:
            if model_shocks and s not in model_shocks:
                raise KeyError(f"Shock '{s}' in group '{g_name}' not found in model.shocks: {model_shocks}")

    # Gather unassigned shocks into "Others"
    assigned_shocks = {s for s_list in resolved_groups.values() for s in s_list}
    unassigned = [s for s in model_shocks if s not in assigned_shocks]
    if unassigned:
        resolved_groups["Others"] = unassigned

    # 2. Compute underlying per-shock decomposition via Kalman smoother
    single_res = compute_shock_decomposition(
        model=model,
        data=data,
        initial_state=initial_state,
        sigma=sigma,
    )

    all_vars = [v for v in model_vars if v in single_res.components] or list(single_res.components.keys())

    # 3. Aggregate shock contributions by group
    components: dict[str, pd.DataFrame] = {}
    for g_name, s_list in resolved_groups.items():
        df_g = pd.DataFrame(index=data.index, columns=all_vars, dtype=float)
        for v in all_vars:
            comp_v = single_res.components[v]
            active_cols = [s for s in s_list if s in comp_v.columns]
            if active_cols:
                df_g[v] = comp_v[active_cols].sum(axis=1)
            else:
                df_g[v] = 0.0
        components[g_name] = df_g

    # 4. Aggregate initial condition and steady state into 'initial'
    df_init = pd.DataFrame(index=data.index, columns=all_vars, dtype=float)
    for v in all_vars:
        comp_v = single_res.components[v]
        init_part = comp_v["initial_condition"].to_numpy(dtype=float) if "initial_condition" in comp_v.columns else 0.0
        ss_part = comp_v["steady_state"].to_numpy(dtype=float) if "steady_state" in comp_v.columns else 0.0
        df_init[v] = init_part + ss_part
    components["initial"] = df_init

    # 5. Check and enforce machine-precision adding-up invariant
    df_resid = pd.DataFrame(index=data.index, columns=all_vars, dtype=float)
    has_large_resid = False
    for v in all_vars:
        if v in data.columns:
            act = data[v].to_numpy(dtype=float)
            total = np.zeros(len(data), dtype=float)
            for g_name in resolved_groups:
                total += components[g_name][v].to_numpy(dtype=float)
            total += components["initial"][v].to_numpy(dtype=float)
            diff = act - total
            df_resid[v] = diff
            max_diff = float(np.max(np.abs(diff[np.isfinite(diff)]))) if np.isfinite(diff).any() else 0.0
            if max_diff > 1e-12:
                has_large_resid = True
            else:
                # Absorb machine-precision float rounding (< 1e-12) into initial condition
                # so that sum(components) == actual to strict floating-point equality
                components["initial"][v] += diff

    if has_large_resid:
        components["residual"] = df_resid

    actual_df = data[[v for v in all_vars if v in data.columns]].copy()

    return ShockDecompositionResult(
        groups={k: tuple(v) for k, v in resolved_groups.items()},
        components=components,
        decomposition=components,
        variables=tuple(all_vars),
        shock_names=tuple(model_shocks),
        actual=actual_df,
        initial_state=components["initial"],
    )
