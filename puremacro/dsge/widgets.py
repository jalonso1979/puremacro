"""Live interactive parameter slider widgets for DSGE models using pure Matplotlib.

Strictly operates under the 4-package Pyodide core (numpy, scipy, pandas, matplotlib)
with zero external UI dependencies (no ipywidgets, no NodeJS, no GUI server requirements).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd

from puremacro.dsge.build import LinearModel, ModelError
from puremacro.dsge.klein import BlanchardKahnError


def _is_interactive_backend() -> bool:
    """Return True if Matplotlib is currently running an interactive GUI backend."""
    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "pdf", "ps", "svg", "template", "cairo"}
    return backend not in non_interactive


class _FlexibleLineRegistry(dict):
    """Line dictionary supporting both (variable, shock) tuples and variable strings."""

    def __getitem__(self, key: Any) -> Line2D:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            matches = [v for k, v in self.items() if isinstance(k, tuple) and k[0] == key]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise KeyError(
                    f"Multiple lines match variable '{key}' across shocks; "
                    f"access using (variable, shock) tuple."
                )
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for k in self.keys():
                if isinstance(k, tuple) and k[0] == key:
                    return True
        return False


class _FlexibleAxesRegistry(dict):
    """Axes dictionary supporting (variable, shock) tuples and variable strings."""

    def __getitem__(self, key: Any) -> Axes:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            matches = [
                v for (var, sh), v in self.items()
                if key in (var, f"{var}_{sh}", f"{var} ({sh})")
            ]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                return matches[0]
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for (var, sh) in self.keys():
                if key in (var, f"{var}_{sh}", f"{var} ({sh})"):
                    return True
        return False


def _fast_resolve(
    model: LinearModel,
    p_dict: Mapping[str, float],
    *,
    strict: bool = False,
    qz_criterium: float = 1.0 + 1e-8,
) -> LinearModel:
    """Re-solve a LinearModel rapidly with updated parameter dictionary."""
    if hasattr(model, "first_order") and isinstance(getattr(model, "first_order"), LinearModel):
        model = model.first_order

    new_params = dict(getattr(model, "_params", {}) or {})
    new_params.update(p_dict)

    # 1. Lead-lag Dynare model (from build_dynare or load_mod)
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import build_dynare

        ss = getattr(model, "_steady_state_dict", None)
        if ss is None and hasattr(model, "steady_state"):
            ss = model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else dict(model.steady_state)

        return build_dynare(
            model._dynare_equations,
            variables=list(model.variables),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            states=list(model.states) if hasattr(model, "states") else None,
            shock_cov=getattr(model, "_shock_cov", None),
            method=getattr(model, "method", "complex"),
            check_steady_state=False,
            verify_derivatives=False,
            strict=strict,
            qz_criterium=qz_criterium,
        )

    # 2. Parsed Model DAG / AST equations
    elif hasattr(model, "compile_equations") and hasattr(model, "variables"):
        from puremacro.dsge.dynare import build_dynare

        eq_fn = model.compile_equations()
        ss = getattr(model, "steady_state", {v: 0.0 for v in model.variables})
        if hasattr(ss, "to_dict"):
            ss = ss.to_dict()
        return build_dynare(
            eq_fn,
            variables=list(model.variables),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            shock_cov=getattr(model, "_shock_cov", None),
            check_steady_state=False,
            verify_derivatives=False,
            strict=strict,
            qz_criterium=qz_criterium,
        )

    # 3. Klein timing model (from build)
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import build

        ss = getattr(model, "_steady_state_dict", None)
        if ss is None and hasattr(model, "steady_state"):
            ss = model.steady_state.to_dict() if hasattr(model.steady_state, "to_dict") else dict(model.steady_state)

        return build(
            model._equations,
            variables=list(model.variables),
            states=list(model.states),
            shocks=list(model.shocks),
            params=new_params,
            steady_state=ss,
            method=getattr(model, "method", "complex"),
            check_steady_state=False,
            verify_derivatives=False,
            strict=strict,
            qz_criterium=qz_criterium,
        )

    else:
        raise ValueError(
            f"Cannot re-solve model of type {type(model).__name__} without equation specifications."
        )


@dataclass
class InteractiveIRFResult:
    """Result object for interactive impulse response exploration with Matplotlib widgets.

    Supports tuple unpacking: ``fig, sliders = model.interactive_irf(...)``
    and standard puremacro presentation contracts (``.summary()``, ``.to_markdown()``,
    ``.to_latex()``, ``.to_typst()``, ``.plot()``).
    """

    fig: Figure
    axes: dict[Any, Axes]
    slider_axes: dict[str, Axes]
    sliders: dict[str, Slider]
    lines: dict[tuple[str, str], Line2D]
    baseline_lines: dict[tuple[str, str], Line2D]
    reset_button: Button | None
    model: LinearModel
    baseline_model: LinearModel
    parameters: list[str]
    shocks: list[str]
    variables: list[str]
    horizon: int
    size: float
    status_text: Any
    param_map: dict[str, str] = field(default_factory=dict)
    last_latency_ms: float = 0.0
    n_updates: int = 0
    strict: bool = False
    qz_criterium: float = 1.0 + 1e-8
    _is_interactive_backend: bool = False
    _cids: list[int] = field(default_factory=list)

    def __iter__(self):
        """Allow dual usage: fig, sliders = model.interactive_irf(...)"""
        yield self.fig
        yield self.sliders

    def set_value(self, parameter: str, value: float) -> None:
        """Programmatically adjust a parameter slider and trigger dynamic update."""
        target_p = parameter
        if target_p not in self.sliders:
            for k, mapped in self.param_map.items():
                if mapped == parameter and k in self.sliders:
                    target_p = k
                    break
        if target_p not in self.sliders:
            raise KeyError(
                f"No slider for parameter '{parameter}'; available: {list(self.sliders.keys())}"
            )
        self.sliders[target_p].set_val(value)

    def update(self, param_values: Mapping[str, float]) -> None:
        """Programmatically adjust multiple parameter sliders."""
        for p in param_values:
            target_p = p
            if target_p not in self.sliders:
                for k, mapped in self.param_map.items():
                    if mapped == p and k in self.sliders:
                        target_p = k
                        break
            if target_p not in self.sliders:
                raise KeyError(
                    f"No slider for parameter '{p}'; available: {list(self.sliders.keys())}"
                )
        for p, v in param_values.items():
            target_p = p if p in self.sliders else self.param_map.get(p, p)
            self.sliders[target_p].set_val(v)

    def reset(self) -> None:
        """Reset all parameter sliders to their initial baseline values."""
        for s in self.sliders.values():
            s.reset()

    def get_values(self) -> dict[str, float]:
        """Return dictionary of current parameter slider values."""
        return {p: float(s.val) for p, s in self.sliders.items()}

    def to_frame(self, shock: str | None = None) -> pd.DataFrame:
        """Return the current active IRF trajectory as a DataFrame."""
        sh = shock or self.shocks[0]
        if self.model.is_determinate:
            try:
                df = self.model.irf(sh, horizon=self.horizon, size=self.size)
                return df[self.variables]
            except Exception:
                pass
        return pd.DataFrame(
            np.nan,
            index=pd.RangeIndex(self.horizon + 1, name="h"),
            columns=list(self.variables),
        )

    def summary(self) -> str:
        """Render publication-grade text summary of interactive exploration."""
        status = "Determinate" if self.model.is_determinate else "Indeterminate / Unstable"
        lines = [
            "INTERACTIVE IMPULSE RESPONSE EXPLORATION",
            "=" * 72,
            f"Horizon             : {self.horizon} periods",
            f"Shocks              : {', '.join(self.shocks)}",
            f"Variables           : {', '.join(self.variables)}",
            f"Determinacy Status  : {status}",
            f"Updates Performed   : {self.n_updates}",
            f"Last Latency        : {self.last_latency_ms:.2f} ms",
            "-" * 72,
            "Parameters:",
        ]
        base_params = getattr(self.baseline_model, "_params", {}) or {}
        for p in self.parameters:
            curr = self.sliders[p].val if p in self.sliders else float("nan")
            actual_p = self.param_map.get(p, p)
            base = base_params.get(actual_p, base_params.get(p, float("nan")))
            lines.append(f"  {p:<18}: {curr:10.4f} (baseline: {base:10.4f})")
        lines.append("-" * 72)
        lines.append(f"Current Active IRF Trajectory ({self.shocks[0]}):")
        lines.append(self.to_frame().round(6).to_string())
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_markdown(self, shock: str | None = None, **kwargs) -> str:
        """Export active IRF table to Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(shock), **kwargs)

    def to_latex(self, shock: str | None = None, **kwargs) -> str:
        """Export active IRF table to LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(shock), **kwargs)

    def to_typst(self, shock: str | None = None, **kwargs) -> str:
        """Export active IRF table to Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(shock), **kwargs)

    def plot(self) -> Figure:
        """Return the interactive Matplotlib figure."""
        return self.fig

    def disconnect(self) -> None:
        """Disconnect all widget callbacks to release memory."""
        for s in self.sliders.values():
            try:
                s.disconnect_events()
            except Exception:
                pass
            s.eventson = False
        if self.reset_button is not None:
            try:
                self.reset_button.disconnect_events()
            except Exception:
                pass
            self.reset_button.eventson = False

    def _set_status(self, text: str) -> None:
        if self.status_text is not None:
            self.status_text.set_text(text)
            self.status_text.set_visible(bool(text))

    def _set_lines_nan(self) -> None:
        nan_data = np.full(self.horizon + 1, np.nan)
        for line in self.lines.values():
            line.set_ydata(nan_data)
            ax = line.axes
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

    def _on_slider_change(self, val=None) -> None:
        """Callback invoked instantaneously when any slider is moved."""
        t0 = time.perf_counter()
        new_params = {}
        for p in self.parameters:
            if p in self.sliders:
                actual_p = self.param_map.get(p, p)
                new_params[actual_p] = float(self.sliders[p].val)

        try:
            new_m = _fast_resolve(
                self.baseline_model,
                new_params,
                strict=self.strict,
                qz_criterium=self.qz_criterium,
            )
            if not new_m.is_determinate:
                self._set_status("[WARNING] Blanchard-Kahn violation: indeterminacy or explosive roots")
                self._set_lines_nan()
                self.model = new_m
            else:
                self._set_status("")
                self.model = new_m
                for shock in self.shocks:
                    df = new_m.irf(shock, horizon=self.horizon, size=self.size)
                    for var in self.variables:
                        key = (var, shock)
                        if key in self.lines:
                            line = self.lines[key]
                            line.set_ydata(df[var].to_numpy())
                            ax = line.axes
                            ax.relim()
                            ax.autoscale_view(scalex=False, scaley=True)
        except (BlanchardKahnError, ModelError) as err:
            self._set_status(f"[WARNING] Blanchard-Kahn violation: {err}")
            self._set_lines_nan()
        except Exception as err:
            self._set_status(f"[WARNING] Solve error: {err}")
            self._set_lines_nan()

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t0) * 1000.0
        self.n_updates += 1

        if self._is_interactive_backend:
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass


def interactive_irf(
    model: LinearModel,
    parameters: Sequence[str] | Mapping[str, tuple[float, ...]] | None = None,
    shocks: Sequence[str] | str | None = None,
    variables: Sequence[str] | None = None,
    horizon: int = 20,
    *,
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    param_steps: Mapping[str, float] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    title: str = "",
    size: float = 1.0,
    show_baseline: bool = True,
    show_reset: bool = True,
    strict: bool = False,
    qz_criterium: float = 1.0 + 1e-8,
) -> InteractiveIRFResult:
    """Spawn an interactive parameter exploration dashboard with Matplotlib Sliders.

    Parameters
    ----------
    model : LinearModel
        Solved first-order DSGE model.
    parameters : Sequence[str] | Mapping[str, tuple[float, ...]], optional
        Structural parameters to expose as interactive sliders. Can be:
        - None: automatically selects key structural parameters from model.
        - Sequence of names: ['sigma', 'kappa', 'phi_pi']. Bounds default
          to [0.2*val, 2.5*val] or can be overridden via ``param_bounds``.
        - Mapping: {'phi_pi': (1.01, 3.0), 'kappa': (0.01, 0.5, 0.1)} where tuples
          specify (min, max) or (min, max, valinit) or (min, max, valinit, valstep).
    shocks : Sequence[str] | str, optional
        Structural shocks to plot. Defaults to [model.shocks[0]].
    variables : Sequence[str], optional
        Endogenous variables to plot. Defaults to model._varobs or up to 6 key variables.
    horizon : int, default 20
        Periods after impact.
    param_bounds : Mapping[str, tuple[float, float]], optional
        Explicit override bounds for parameters.
    param_steps : Mapping[str, float], optional
        Step increment for each parameter slider.
    ncols : int, default 2
        Number of subplot columns for IRF panels.
    figsize : tuple[float, float], optional
        Figure size in inches (default auto-scaled to number of subplots and sliders).
    title : str, default ""
        Figure title.
    size : float, default 1.0
        Shock innovation scale.
    show_baseline : bool, default True
        If True, displays a static dashed line for the initial baseline calibration.
    show_reset : bool, default True
        If True, displays a Reset button to return all sliders to initial values.
    strict : bool, default False
        If False, Blanchard-Kahn failures display a visual warning without raising an exception.
    qz_criterium : float, default 1.0 + 1e-8
        Modulus threshold for explosive roots.

    Returns
    -------
    InteractiveIRFResult
        Interactive result container with .fig, .sliders, .lines, and presentation methods.
        Supports tuple unpacking: fig, sliders = model.interactive_irf(...)
    """
    if hasattr(model, "first_order") and hasattr(model.first_order, "variables"):
        model = model.first_order

    if not hasattr(model, "variables") or not hasattr(model, "shocks"):
        raise TypeError("model must be a LinearModel or DSGE solution object.")

    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    # Shocks validation
    if shocks is None:
        if not model.shocks:
            raise ValueError("Model has no structural shocks declared.")
        shocks_list = [model.shocks[0]]
    elif isinstance(shocks, str):
        shocks_list = [shocks]
    else:
        shocks_list = list(shocks)

    for sh in shocks_list:
        if sh not in model.shocks:
            raise ModelError(f"no shock named {sh!r}; declared: {list(model.shocks)}")

    # Variables validation
    if variables is None:
        if getattr(model, "_varobs", None):
            variables_list = [v for v in model._varobs if v in model.variables]
        else:
            variables_list = list(model.variables)[:6]
    elif isinstance(variables, str):
        variables_list = [variables]
    else:
        variables_list = list(variables)

    for v in variables_list:
        if v not in model.variables:
            raise ValueError(f"no variable named {v!r}; declared: {list(model.variables)}")

    # Parameters resolution
    model_params = dict(getattr(model, "_params", {}) or {})
    alias_map: dict[str, str] = {}
    if "crr" in model_params and "crhor" not in model_params:
        alias_map["crhor"] = "crr"

    slider_configs: dict[str, tuple[str, float, float, float, float | None]] = {}
    param_names: list[str] = []

    if parameters is None:
        candidate_params = [
            p for p, v in model_params.items()
            if isinstance(v, (int, float, np.number)) and not math.isnan(float(v))
        ]
        if not candidate_params:
            raise ValueError("Model has no structural parameters to adjust.")
        param_specs: Any = candidate_params[:4]
    else:
        param_specs = parameters

    if isinstance(param_specs, Mapping):
        for raw_p, spec in param_specs.items():
            actual_p = alias_map.get(raw_p, raw_p)
            if actual_p not in model_params:
                raise ValueError(f"Unknown parameter '{raw_p}'; declared: {sorted(model_params.keys())}")
            base_val = float(model_params[actual_p])
            if isinstance(spec, (int, float)):
                v = float(spec)
                if v > 0:
                    valmin, valmax = 0.2 * v, 2.5 * v
                elif v < 0:
                    valmin, valmax = 2.5 * v, 0.2 * v
                else:
                    valmin, valmax = -1.0, 1.0
                valinit = v
                valstep = None
            elif isinstance(spec, tuple):
                if len(spec) == 2:
                    valmin, valmax = float(spec[0]), float(spec[1])
                    valinit = base_val
                    valstep = None
                elif len(spec) == 3:
                    valmin, valmax, valinit = float(spec[0]), float(spec[1]), float(spec[2])
                    valstep = None
                elif len(spec) >= 4:
                    valmin, valmax, valinit, valstep = float(spec[0]), float(spec[1]), float(spec[2]), float(spec[3])
                else:
                    raise ValueError(f"Invalid parameter spec for '{raw_p}': {spec}")
            else:
                raise TypeError(f"Expected tuple or numeric spec for '{raw_p}', got {type(spec).__name__}")

            if param_bounds and raw_p in param_bounds:
                valmin, valmax = param_bounds[raw_p]
            elif param_bounds and actual_p in param_bounds:
                valmin, valmax = param_bounds[actual_p]

            if param_steps and raw_p in param_steps:
                valstep = param_steps[raw_p]
            elif param_steps and actual_p in param_steps:
                valstep = param_steps[actual_p]

            slider_configs[raw_p] = (actual_p, valmin, valmax, valinit, valstep)
            param_names.append(raw_p)

    else:
        for raw_p in param_specs:
            actual_p = alias_map.get(raw_p, raw_p)
            if actual_p not in model_params:
                raise ValueError(f"Unknown parameter '{raw_p}'; declared: {sorted(model_params.keys())}")
            base_val = float(model_params[actual_p])

            if param_bounds and raw_p in param_bounds:
                valmin, valmax = param_bounds[raw_p]
            elif param_bounds and actual_p in param_bounds:
                valmin, valmax = param_bounds[actual_p]
            else:
                if base_val == 0.0:
                    valmin, valmax = -1.0, 1.0
                elif base_val > 0:
                    valmin = 0.2 * base_val
                    valmax = 2.5 * base_val
                else:
                    valmin = 2.5 * base_val
                    valmax = 0.2 * base_val

            valinit = base_val
            valstep = None
            if param_steps and raw_p in param_steps:
                valstep = param_steps[raw_p]
            elif param_steps and actual_p in param_steps:
                valstep = param_steps[actual_p]

            slider_configs[raw_p] = (actual_p, valmin, valmax, valinit, valstep)
            param_names.append(raw_p)

    # Initial baseline IRFs
    df_base: dict[str, pd.DataFrame] = {}
    for sh in shocks_list:
        df_base[sh] = model.irf(sh, horizon=horizon, size=size)

    # Check if initial slider values differ from baseline
    needs_init_resolve = any(cfg[3] != float(model_params[cfg[0]]) for cfg in slider_configs.values())
    if needs_init_resolve:
        init_params = {cfg[0]: cfg[3] for cfg in slider_configs.values()}
        try:
            active_model = _fast_resolve(model, init_params, strict=strict, qz_criterium=qz_criterium)
            df_active: dict[str, pd.DataFrame] = {}
            for sh in shocks_list:
                df_active[sh] = active_model.irf(sh, horizon=horizon, size=size)
        except Exception:
            active_model = model
            df_active = df_base
    else:
        active_model = model
        df_active = df_base

    # Compute figure layout geometry
    n_panels = len(variables_list) * len(shocks_list)
    ncols = min(max(1, ncols), n_panels)
    nrows = int(math.ceil(n_panels / ncols))

    n_sliders = len(param_names)
    slider_cols = 2 if n_sliders > 4 else 1
    slider_rows = int(math.ceil(n_sliders / slider_cols))

    h_plots = max(3.2, 2.4 * nrows)
    h_sliders = max(1.6, 0.48 * slider_rows + (0.6 if show_reset else 0.2))
    fig_h = h_plots + h_sliders
    fig_w = max(8.5, 4.2 * ncols)
    if figsize is not None:
        fig_w, fig_h = figsize

    fig = plt.figure(figsize=(fig_w, fig_h))

    slider_frac = min(0.45, h_sliders / fig_h)
    plots_top = 0.93 if title else 0.95
    plots_bottom = slider_frac + 0.04

    gs_plots = GridSpec(
        nrows,
        ncols,
        figure=fig,
        top=plots_top,
        bottom=plots_bottom,
        left=0.10,
        right=0.92,
        hspace=0.38,
        wspace=0.28,
    )

    axes = _FlexibleAxesRegistry()
    lines = _FlexibleLineRegistry()
    baseline_lines = _FlexibleLineRegistry()

    panel_idx = 0
    for sh in shocks_list:
        df_a = df_active[sh]
        df_b = df_base[sh]
        for var in variables_list:
            r = panel_idx // ncols
            c = panel_idx % ncols
            ax = fig.add_subplot(gs_plots[r, c])

            ax.axhline(0.0, color="0.7", linestyle=":", linewidth=0.8)

            # Baseline line (static reference)
            if show_baseline:
                base_lbl = f"Baseline ({sh})" if len(shocks_list) > 1 else "Baseline"
                b_line, = ax.plot(
                    df_b.index,
                    df_b[var].to_numpy(),
                    linestyle="--",
                    color="0.55",
                    linewidth=1.2,
                    alpha=0.8,
                    label=base_lbl,
                )
                baseline_lines[(var, sh)] = b_line

            # Active line (dynamically updated)
            act_lbl = f"Active ({sh})" if len(shocks_list) > 1 else "Active"
            a_line, = ax.plot(
                df_a.index,
                df_a[var].to_numpy(),
                linestyle="-",
                color="#1f77b4",
                linewidth=1.8,
                label=act_lbl,
            )
            lines[(var, sh)] = a_line

            ax_title = f"{var} (to {sh})" if len(shocks_list) > 1 else var
            ax.set_title(ax_title, fontsize=11, weight="bold")
            ax.set_xlabel("Horizon (h)", fontsize=9)
            ax.set_ylabel("Response", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="best", fontsize=8)

            axes[(var, sh)] = ax

            panel_idx += 1

    # Sliders and Reset button layout in lower region
    slider_axes: dict[str, Axes] = {}
    sliders: dict[str, Slider] = {}

    slider_area_top = plots_bottom - 0.04
    slider_area_bottom = 0.04
    avail_h = max(0.10, slider_area_top - slider_area_bottom)
    reset_h = 0.038 if show_reset else 0.0
    usable_slider_h = avail_h - (reset_h + 0.02 if show_reset else 0.0)
    row_slot = usable_slider_h / max(1, slider_rows)
    bar_h = min(0.032, row_slot * 0.55)

    for i, raw_p in enumerate(param_names):
        actual_p, valmin, valmax, valinit, valstep = slider_configs[raw_p]
        if slider_cols == 1:
            y_pos = slider_area_top - (i + 1) * row_slot + (row_slot - bar_h) / 2
            ax_s = fig.add_axes([0.22, y_pos, 0.58, bar_h])
        else:
            r = i // 2
            c = i % 2
            y_pos = slider_area_top - (r + 1) * row_slot + (row_slot - bar_h) / 2
            x_pos = 0.12 if c == 0 else 0.58
            ax_s = fig.add_axes([x_pos, y_pos, 0.35, bar_h])

        s_kwargs: dict[str, Any] = {
            "ax": ax_s,
            "label": raw_p,
            "valmin": valmin,
            "valmax": valmax,
            "valinit": valinit,
        }
        if valstep is not None:
            s_kwargs["valstep"] = valstep

        slider = Slider(**s_kwargs)
        sliders[raw_p] = slider
        slider_axes[raw_p] = ax_s

    # Reset Button
    reset_btn = None
    if show_reset:
        btn_y = slider_area_bottom
        btn_x = 0.44 if slider_cols == 2 else 0.82
        btn_w = 0.12 if slider_cols == 2 else 0.10
        ax_btn = fig.add_axes([btn_x, btn_y, btn_w, max(0.028, reset_h)])
        reset_btn = Button(ax_btn, "Reset", color="0.95", hovercolor="0.85")

    # Status Banner and Figure Title
    status_text = fig.text(
        0.5,
        0.982,
        "",
        ha="center",
        va="top",
        color="crimson",
        fontsize=10,
        weight="bold",
    )
    if title:
        fig.suptitle(title, fontsize=13, weight="bold", y=0.96)

    param_map = {raw_p: slider_configs[raw_p][0] for raw_p in param_names}

    result = InteractiveIRFResult(
        fig=fig,
        axes=axes,
        slider_axes=slider_axes,
        sliders=sliders,
        lines=lines,
        baseline_lines=baseline_lines,
        reset_button=reset_btn,
        model=active_model,
        baseline_model=model,
        parameters=param_names,
        shocks=shocks_list,
        variables=variables_list,
        horizon=horizon,
        size=size,
        status_text=status_text,
        param_map=param_map,
        strict=strict,
        qz_criterium=qz_criterium,
        _is_interactive_backend=_is_interactive_backend(),
    )

    # Attach slider callbacks
    for p, s in sliders.items():
        cid = s.on_changed(result._on_slider_change)
        result._cids.append(cid)

    # Attach reset button callback
    if reset_btn is not None:
        cid = reset_btn.on_clicked(lambda event: result.reset())
        result._cids.append(cid)

    # Attach strong reference on figure to prevent GC in interactive backends
    fig._interactive_widget = result  # type: ignore[attr-defined]

    return result
