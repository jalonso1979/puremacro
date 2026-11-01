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
from puremacro.dsge._workbench_state import DSGEFailure, describe_failure


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
    """Re-solve a LinearModel rapidly with updated parameter dictionary.

    ``build_dynare`` / ``load_mod`` and ``build`` models go through the same
    re-solve as ``osr`` and estimation (:func:`puremacro.dsge.dynare.
    _resolve_dynare_model`, :func:`puremacro.dsge.build._resolve_build_model`):
    the steady state follows the parameters, the declared shock covariance is
    kept and a lag calibrated to 0 stays a state when a slider makes it live.
    A parsed model (``ParsedModelDAG``) goes through
    :func:`puremacro.dsge.dynare._resolve_parsed_dag_model`, with the same
    guarantees.
    """
    if hasattr(model, "first_order") and isinstance(getattr(model, "first_order"), LinearModel):
        model = model.first_order

    new_params = dict(getattr(model, "_params", {}) or {})
    new_params.update(p_dict)

    # 1. Lead-lag Dynare model (from build_dynare or load_mod)
    if getattr(model, "_dynare_equations", None) is not None:
        from puremacro.dsge.dynare import _resolve_dynare_model

        return _resolve_dynare_model(
            model, new_params, strict=strict, qz_criterium=qz_criterium,
        )

    # 2. Parsed Model DAG / AST equations: the sliders are merged with the
    # file's calibration (a DAG has no ``_params``, so they used to be the
    # only parameters passed), the steady_state_model block is re-evaluated
    # at the merged values and the shocks; block is kept.
    elif hasattr(model, "compile_equations") and hasattr(model, "variables"):
        from puremacro.dsge.dynare import _resolve_parsed_dag_model

        return _resolve_parsed_dag_model(
            model, dict(p_dict), strict=strict, qz_criterium=qz_criterium,
        )

    # 3. Klein timing model (from build)
    elif getattr(model, "_equations", None) is not None:
        from puremacro.dsge.build import _resolve_build_model

        # build() has no check_steady_state switch, so the old call raised
        # TypeError; the stored steady state is reused only where it still
        # solves the model.
        return _resolve_build_model(
            model, new_params, strict=strict, qz_criterium=qz_criterium,
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
    auto_solve: bool = True
    apply_button: Button | None = None
    state_text: Any = None
    run_state: str = "succeeded"
    failure: DSGEFailure | None = None
    last_success_model: LinearModel | None = None
    accepted_parameters: dict[str, float] = field(default_factory=dict)
    _current_frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __iter__(self):
        """Allow dual usage: fig, sliders = model.interactive_irf(...)"""
        yield self.fig
        yield self.sliders

    def set_value(self, parameter: str, value: float) -> None:
        """Edit one value; solve immediately only when ``auto_solve=True``."""
        self.update({parameter: value})

    def update(self, param_values: Mapping[str, float]) -> None:
        """Validate a batch before editing, then perform at most one solve."""
        edits = {}
        for p, value in param_values.items():
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
            slider = self.sliders[target_p]
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{p} must be finite")
            if target_p in edits:
                raise ValueError(f"Parameter {target_p} appears more than once through aliases")
            edits[target_p] = number
        if not edits:
            return
        for p, value in edits.items():
            slider = self.sliders[p]
            events, drawing = slider.eventson, slider.drawon
            try:
                slider.eventson = slider.drawon = False
                slider.set_val(value)
            finally:
                slider.eventson, slider.drawon = events, drawing
        self._on_edit()

    def reset(self) -> None:
        """Reset all parameter sliders to their initial baseline values."""
        self.update({name: slider.valinit for name, slider in self.sliders.items()})

    def get_values(self) -> dict[str, float]:
        """Return dictionary of current parameter slider values."""
        return {p: float(s.val) for p, s in self.sliders.items()}

    @property
    def is_current(self) -> bool:
        """Whether the displayed/exported run matches every current slider."""
        return self.run_state == "succeeded" and all(
            float(s.val) == self.accepted_parameters.get(self.param_map.get(p, p))
            for p, s in self.sliders.items()
        )

    def to_frame(self, shock: str | None = None) -> pd.DataFrame:
        """Return the current active IRF trajectory as a DataFrame."""
        sh = shock or self.shocks[0]
        if sh not in self.shocks:
            raise KeyError(f"Unknown plotted shock {sh!r}; available: {self.shocks}")
        if self.is_current and sh in self._current_frames:
            frame = self._current_frames[sh].loc[:, self.variables].copy(deep=True)
            frame.attrs.update(run_state=self.run_state, run_number=self.n_updates,
                               parameters=self.accepted_parameters.copy())
            return frame
        return pd.DataFrame(
            np.nan,
            index=pd.RangeIndex(self.horizon + 1, name="h"),
            columns=list(self.variables),
        )

    def summary(self) -> str:
        """Render publication-grade text summary of interactive exploration."""
        status = "Determinate" if self.is_current else "No current solved result"
        lines = [
            "INTERACTIVE IMPULSE RESPONSE EXPLORATION",
            "=" * 72,
            f"Horizon             : {self.horizon} periods",
            f"Shocks              : {', '.join(self.shocks)}",
            f"Variables           : {', '.join(self.variables)}",
            f"Determinacy Status  : {status}",
            f"Run State           : {self.run_state if self.is_current else ('edited' if self.run_state == 'succeeded' else self.run_state)}",
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

    def accuracy(self):
        """Inspect the current run, retaining untested accuracy dimensions."""
        from .workbench import solution_accuracy
        return solution_accuracy(self.model if self.is_current else None, failure=self.failure)

    def capture_run(self, *, label: str = "Interactive IRF"):
        """Freeze inputs, current tables and evidence; never capture stale curves."""
        from .workbench import DSGERun
        parameters = dict(getattr(self.baseline_model, "_params", {}) or {})
        parameters.update({self.param_map.get(p, p): float(s.val) for p, s in self.sliders.items()})
        status = "succeeded" if self.is_current else ("edited" if self.run_state == "succeeded" else self.run_state)
        return DSGERun.capture(
            self.model if self.is_current else None, label=label, status=status,
            parameters=parameters, failure=self.failure,
            configuration={"operation": "interactive_irf", "shocks": self.shocks,
                           "horizon": self.horizon, "size": self.size},
            tables={sh: self.to_frame(sh) for sh in self.shocks} if self.is_current else {},
            units=dict(self.model.units), timing=str(self.model.timing),
        )

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
        for button in (self.reset_button, self.apply_button):
            if button is None:
                continue
            try:
                button.disconnect_events()
            except Exception:
                pass
            button.eventson = False

    def _set_status(self, text: str) -> None:
        if self.status_text is not None:
            from textwrap import fill
            width = max(50, int(self.fig.get_figwidth() * 12))
            self.status_text.set_text(fill(text, width=width, max_lines=2, placeholder="…"))
            self.status_text.set_color("#8c2c28" if self.run_state == "failed" else "#73530b")
            self.status_text.set_visible(bool(text))

    def _set_lines_nan(self) -> None:
        nan_data = np.full(self.horizon + 1, np.nan)
        for line in self.lines.values():
            line.set_ydata(nan_data)

    def _on_slider_change(self, val=None) -> None:
        """One explicit solve of the current batch of slider values."""
        t0 = time.perf_counter()
        self.run_state = "running"
        self.failure = None
        self._current_frames.clear()
        self._set_lines_nan()
        self._update_state_text()
        self._set_status("Solving current parameters…")
        if self._is_interactive_backend:
            self.fig.canvas.draw_idle()
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
                self.model = new_m
                self.failure = DSGEFailure(
                    "determinacy", "BlanchardKahnError",
                    "Blanchard-Kahn violation: indeterminacy or explosive roots",
                    "Inspect variable timing, the eigenvalues and the policy-rule parameters.",
                )
                self.run_state = "failed"
                self._set_status(f"[determinacy] {self.failure.message}. {self.failure.next_step}")
            else:
                frames = {}
                for shock in self.shocks:
                    df = new_m.irf(shock, horizon=self.horizon, size=self.size)
                    if not np.isfinite(df.loc[:, self.variables].to_numpy()).all():
                        raise FloatingPointError("The solved IRF contains nonfinite values")
                    frames[shock] = df.copy(deep=True)
                self.model = self.last_success_model = new_m
                self.accepted_parameters = dict(getattr(new_m, "_params", {}) or {})
                self._current_frames = frames
                self.run_state = "succeeded"
                self._set_status("")
                for shock, df in frames.items():
                    for var in self.variables:
                        key = (var, shock)
                        if key in self.lines:
                            line = self.lines[key]
                            line.set_ydata(df[var].to_numpy())
                            ax = line.axes
                            if ax is not None:
                                ax.relim()
                                ax.autoscale_view(scalex=False, scaley=True)
        except Exception as err:
            self.failure = describe_failure(err)
            self.run_state = "failed"
            self._current_frames.clear()
            self._set_status(f"[{self.failure.category}] {self.failure.message}. {self.failure.next_step}")
            self._set_lines_nan()

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t0) * 1000.0
        self.n_updates += 1
        self._update_state_text()

        if self._is_interactive_backend:
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    def _update_state_text(self) -> None:
        if self.state_text is not None:
            self.state_text.set_text(f"Run {self.n_updates} · {self.run_state}")

    def _on_edit(self, val=None) -> None:
        if self.auto_solve:
            self.apply()
        else:
            self.run_state = "edited"
            self.failure = None
            self._current_frames.clear()
            self._set_lines_nan()
            self._set_status("Parameters edited — Apply to solve")
            self._update_state_text()
            if self._is_interactive_backend:
                self.fig.canvas.draw_idle()

    def apply(self) -> None:
        """Solve current values once; the previous successful model stays explicit."""
        self._on_slider_change()


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
    auto_solve: bool = True,
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
    auto_solve : bool, default True
        False batches edits until Apply is pressed. Edited or failed results
        export missing values, never the last successful model's curves.

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
        varobs = getattr(model, "_varobs", None)
        if varobs is not None:
            variables_list = [v for v in varobs if v in model.variables]
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
            actual_p: str = str(alias_map.get(raw_p, raw_p))
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
            actual_p = str(alias_map.get(raw_p, raw_p))
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

    for raw_p, (_, lo, hi, initial, step) in slider_configs.items():
        if not all(np.isfinite(value) for value in (lo, hi, initial)) or lo >= hi:
            raise ValueError(f"Invalid finite slider bounds for '{raw_p}'")
        if not lo <= initial <= hi:
            raise ValueError(f"Initial value for '{raw_p}' is outside its slider bounds")
        if step is not None and (not np.isfinite(step) or step <= 0):
            raise ValueError(f"Slider step for '{raw_p}' must be finite and positive")

    # Initial baseline IRFs
    df_base: dict[str, pd.DataFrame] = {}
    for sh in shocks_list:
        df_base[sh] = model.irf(sh, horizon=horizon, size=size)

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
    h_sliders = max(1.6, 0.48 * slider_rows + (0.6 if show_reset or not auto_solve else 0.2))
    fig_h = h_plots + h_sliders
    fig_w = max(8.5, 4.2 * ncols)
    if figsize is not None:
        fig_w, fig_h = figsize

    fig = plt.figure(figsize=(fig_w, fig_h))

    slider_frac = min(0.45, h_sliders / fig_h)
    plots_top = 0.85 if title else 0.88
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
            ax.set_ylabel(f"Response ({model.units.get(var, 'units unspecified')})", fontsize=9)
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
    reset_h = 0.038 if show_reset or not auto_solve else 0.0
    usable_slider_h = avail_h - (reset_h + 0.02 if show_reset or not auto_solve else 0.0)
    row_slot = usable_slider_h / max(1, slider_rows)
    bar_h = min(0.032, row_slot * 0.55)

    for i, raw_p in enumerate(param_names):
        actual_p, valmin, valmax, valinit, valstep = slider_configs[raw_p]
        if slider_cols == 1:
            y_pos = slider_area_top - (i + 1) * row_slot + (row_slot - bar_h) / 2
            ax_s = fig.add_axes((0.22, float(y_pos), 0.58, float(bar_h)))
        else:
            r = i // 2
            c = i % 2
            y_pos = slider_area_top - (r + 1) * row_slot + (row_slot - bar_h) / 2
            x_pos = 0.12 if c == 0 else 0.58
            ax_s = fig.add_axes((float(x_pos), float(y_pos), 0.35, float(bar_h)))

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
        ax_btn = fig.add_axes((float(btn_x), float(btn_y), float(btn_w), float(max(0.028, reset_h))))
        reset_btn = Button(ax_btn, "Reset", color="0.95", hovercolor="0.85")

    apply_btn = None
    if not auto_solve:
        ax_apply = fig.add_axes((0.68, float(slider_area_bottom), 0.10, float(max(0.028, reset_h))))
        apply_btn = Button(ax_apply, "Apply", color="0.90", hovercolor="0.80")

    # Status Banner and Figure Title
    status_text = fig.text(
        0.5,
        0.995,
        "",
        ha="center",
        va="top",
        color="crimson",
        fontsize=10,
        weight="bold",
    )
    if title:
        fig.suptitle(title, fontsize=13, weight="bold", y=0.935)

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
        auto_solve=auto_solve,
        apply_button=apply_btn,
        state_text=fig.text(0.10, 0.01, "Run 0 · succeeded", fontsize=9),
        last_success_model=model if model.is_determinate else None,
        accepted_parameters=model_params.copy(),
        _current_frames={sh: df.copy(deep=True) for sh, df in df_base.items()},
    )

    # Attach slider callbacks
    for p, s in sliders.items():
        cid = s.on_changed(result._on_edit)
        result._cids.append(cid)

    # Attach reset button callback
    if reset_btn is not None:
        cid = reset_btn.on_clicked(lambda event: result.reset())
        result._cids.append(cid)

    # Attach strong reference on figure to prevent GC in interactive backends
    if apply_btn is not None:
        result._cids.append(apply_btn.on_clicked(lambda event: result.apply()))
    # Matplotlib may snap valinit to valstep. Solve the values actually shown.
    needs_init_resolve = any(float(s.val) != float(model_params[param_map[p]]) for p, s in sliders.items())
    if needs_init_resolve or not model.is_determinate:
        result.apply()

    fig._interactive_widget = result  # type: ignore[attr-defined]

    return result
