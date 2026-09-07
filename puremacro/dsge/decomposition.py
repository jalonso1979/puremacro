"""Dynare-parity Forecast Error Variance Decomposition (FEVD) and Historical Shock Decomposition.

Companion state-space representation, in the timing the model *reports* its
variables (what ``irf()`` / ``simulate()`` / ``theoretical_moments()`` use):
    x_{t+1} = A @ x_t + B @ u_t
    y_t     = y_s + C @ x_t + D @ u_t

Under Dynare timing (``build_dynare`` / ``load_mod``) ``x_t`` is the lagged
state ``s_{t-1}`` and ``(C, D) = (ghx, ghu)``. Under Klein timing (``build``)
``x_t`` is the state itself and ``(C, D) = ([I; F], [0; L])``, so the state
rows are known one period ahead and carry no impact loading.

Vector Moving Average (VMA) representation:
    Psi_0 = D
    Psi_k = C @ (A^(k-1)) @ B   for k >= 1

Variance shares at finite horizon h (Dynare's conditional variance decomposition):
    V_{i, j}(h) = sum_{k=0}^{h-1} (Psi_k[i, j])^2 * sigma_j^2
    MSE_i(h)    = sum_j V_{i, j}(h)
    Share_{i, j}(h) = V_{i, j}(h) / MSE_i(h)

A row with MSE_i(h) = 0 has no defined shares: it is reported as NaN (with a
ZeroVarianceWarning), never padded with a uniform 1/n_shocks.

Asymptotic variance shares (horizon = None or np.inf) via discrete Lyapunov:
    Sigma_{s, j} = solve_discrete_lyapunov(A, sigma_j^2 * (B_{:, j} @ B_{:, j}^T))
    V_{i, j}(inf) = sigma_j^2 * (D[i, j]^2) + [C @ Sigma_{s, j} @ C^T]_{i, i}

Historical Shock Decomposition:
    Runs the Kalman smoother to extract smoothed states s_hat_t and shocks
    u_hat_t, then simulates the path of each individual shock, the initial
    state decay (C A^t x_0) and the steady state:
        recon_t = steady_state + C @ A^t @ x_0 + sum_{j} shock_j(t)
    ``recon_t`` is the model's own reconstruction. The column labelled
    'actual' always holds the caller's observed data, never the
    reconstruction, and a 'residual' column carries what the model cannot
    explain, so the adding-up identity

        steady_state + initial_condition + sum_j shock_j + residual == actual

    is checked against the data the user supplied. ``residual`` is the
    smoothed measurement error: it is at round-off level when the model can
    reproduce the data exactly (as many shocks as observables, no measurement
    error) and materially non-zero otherwise -- a warning is issued in that
    case rather than the mismatch being absorbed into 'actual'.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from matplotlib.figure import Figure

from puremacro.dsge._moments import conditional_fevd
from puremacro.plot import _new_ax, _palette
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.state_space import StateSpaceModel, kalman_smoother


@dataclass(frozen=True)
class FEVDResult:
    """Forecast Error Variance Decomposition (FEVD) result container.

    Attributes
    ----------
    table : pd.DataFrame
        Multi-index DataFrame with index ('Variable', 'Horizon') and columns
        corresponding to structural shock names. Rows record the fraction of
        forecast error variance attributed to each shock.
    horizons : list[int]
        Evaluation horizons specified for the decomposition.
    variable_names : list[str]
        Names of endogenous variables included in the decomposition.
    shock_names : list[str]
        Names of structural innovations.
    """

    table: pd.DataFrame
    horizons: list[int | None]
    variable_names: list[str]
    shock_names: list[str]

    def __post_init__(self) -> None:
        if self.table is not None and not self.table.empty:
            block = self.table[self.shock_names]
            # Rows whose forecast-error variance is zero carry NaN shares
            # (undefined), so the invariant is checked on the defined rows.
            defined = block.notna().all(axis=1)
            if not bool(defined.any()):
                return
            row_sums = block[defined].sum(axis=1)
            # Check invariant: row sums must equal 1.0 (or 100.0) within machine precision
            diff_one = np.abs(row_sums - 1.0)
            diff_hundred = np.abs(row_sums - 100.0)
            if np.min([np.max(diff_one), np.max(diff_hundred)]) > 1e-5:
                raise ValueError(
                    f"FEVD invariant violated: variance shares across shocks must sum to 1.0 (or 100%). "
                    f"Max deviation: {np.min([np.max(diff_one), np.max(diff_hundred)]):.3e}"
                )

    def to_frame(self) -> pd.DataFrame:
        """Return the multi-index FEVD table as a DataFrame."""
        return self.table.copy()

    def summary(self) -> str:
        """Render a Dynare-style text summary of the variance decomposition."""
        lines = [
            "FORECAST ERROR VARIANCE DECOMPOSITION (Dynare Format)",
            "=" * 72,
            self.table.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export FEVD table to Markdown format."""
        return _df_to_markdown(self.table, **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export FEVD table to LaTeX tabular format."""
        return _df_to_latex(self.table, **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export FEVD table to Typst table format."""
        return _df_to_typst(self.table, **kwargs)

    def plot(
        self,
        variables: Sequence[str] | str | None = None,
        style: str = "publication",
    ) -> Figure:
        """Generate stacked-bar / stacked-area plots of variance shares across horizons.

        Parameters
        ----------
        variables : Sequence[str] | str, optional
            Subset of variables to plot. Defaults to all variables (up to 9).
        style : str, default 'publication'
            Visual styling theme.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the FEVD plots.
        """
        if variables is None:
            vars_to_plot = list(self.variable_names[:9])
        elif isinstance(variables, str):
            vars_to_plot = [variables]
        else:
            vars_to_plot = list(variables)

        n_vars = len(vars_to_plot)
        if n_vars == 0:
            raise ValueError("No variables specified for plotting.")

        ncols = min(3, n_vars)
        nrows = int(np.ceil(n_vars / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False
        )

        n_shocks = len(self.shock_names)
        colors = _palette(n_shocks)

        for idx, var in enumerate(vars_to_plot):
            ax = axes.flatten()[idx]
            if var not in self.table.index.levels[0]:
                continue
            sub = self.table.loc[var]
            h_labels = [
                r"$\infty$" if str(h) in ("None", "inf", "Infinity") else str(h)
                for h in sub.index
            ]
            x_pos = np.arange(len(h_labels))

            bottom = np.zeros(len(h_labels))
            for s_idx, shk in enumerate(self.shock_names):
                shares = np.nan_to_num(sub[shk].to_numpy(dtype=float), nan=0.0)
                ax.bar(
                    x_pos,
                    shares,
                    bottom=bottom,
                    label=shk,
                    color=colors[s_idx % len(colors)],
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottom += shares

            ax.set_xticks(x_pos)
            ax.set_xticklabels(h_labels)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{var}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Variance share")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Hide any unused subplots
        for idx in range(n_vars, nrows * ncols):
            axes.flatten()[idx].set_visible(False)

        # Add single legend to top plot or figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(n_shocks, 7),
                frameon=False,
            )

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class ShockDecompResult:
    """Historical Shock Decomposition result container.

    Attributes
    ----------
    components : dict[str, pd.DataFrame]
        Dictionary mapping each endogenous variable name to a DataFrame with
        one column per structural shock plus 'initial_condition',
        'steady_state', 'residual' and 'actual'.

        'actual' is the caller's own observed series for every variable that
        appears in the data (and the model's reconstruction for the
        unobserved ones). 'residual' is what the smoothed model path cannot
        reproduce, so that

            steady_state + initial_condition + sum(shocks) + residual == actual

        holds period by period, to machine precision, against the *data*.
        A large 'residual' means the model cannot explain the observations
        with the shocks it has; :func:`compute_shock_decomposition` warns when
        that happens.
    variable_names : list[str]
        Names of endogenous variables decomposed.
    shock_names : list[str]
        Names of structural shocks.
    smoothed_shocks : pd.DataFrame
        DataFrame of recovered smoothed shocks u_hat_t (periods x shocks).

    Notes
    -----
    Periods whose 'actual' entry is missing (NaN) are excluded from the
    adding-up check -- there is no observation to check against. The Kalman
    smoother still interpolates the model path through them, and
    :meth:`summary` reports how many periods were excluded.
    """

    components: dict[str, pd.DataFrame]
    variable_names: list[str]
    shock_names: list[str]
    smoothed_shocks: pd.DataFrame

    def __post_init__(self) -> None:
        # Adding-up invariant, checked against the *observed* series:
        #   steady_state + initial_condition + sum(shocks) + residual == actual
        # The tolerance is relative to the level of the series, so it does not
        # become vacuous for data measured in large units.
        for var in self.variable_names:
            if var not in self.components:
                continue
            df = self.components[var]
            recon = (
                df["steady_state"]
                + df["initial_condition"]
                + df[self.shock_names].sum(axis=1)
            )
            if "residual" in df.columns:
                recon = recon + df["residual"]
            actual = df["actual"]
            gap = np.abs((recon - actual).to_numpy(dtype=float))
            finite = np.isfinite(gap)
            if not finite.any():
                continue
            err = float(np.max(gap[finite]))
            scale = float(np.max(np.abs(actual.to_numpy(dtype=float))[finite]))
            tol = 1e-10 * max(scale, 1.0)
            if err > tol:
                raise ValueError(
                    f"Historical shock decomposition invariant violated for '{var}': "
                    f"steady_state + initial_condition + sum(shocks) + residual "
                    f"differs from the observed series by max={err:.3e}, above the "
                    f"{tol:.3e} tolerance (1e-10 relative to a series level of {scale:.3e})."
                )

    def to_frame(self, variable: str) -> pd.DataFrame:
        """Return the decomposition table for a specific variable."""
        if variable not in self.components:
            raise KeyError(
                f"Variable {variable!r} not found in decomposition components. "
                f"Available: {self.variable_names}"
            )
        return self.components[variable].copy()

    def summary(self, variable: str | None = None) -> str:
        """Render text summary of the historical shock decomposition."""
        target_var = variable if variable is not None else self.variable_names[0]
        df = self.to_frame(target_var)
        lines = [
            f"HISTORICAL SHOCK DECOMPOSITION: {target_var}",
            "=" * 72,
            df.round(6).to_string(),
            "-" * 72,
        ]
        actual = df["actual"].to_numpy(dtype=float)
        n_missing = int(np.count_nonzero(~np.isfinite(actual)))
        if "residual" in df.columns:
            resid = np.abs(df["residual"].to_numpy(dtype=float))
            resid = resid[np.isfinite(resid)]
            max_resid = float(np.max(resid)) if resid.size else 0.0
            lines.append(
                f"unexplained residual (max |actual - reconstruction|): {max_resid:.3e}"
            )
        lines.append(
            f"periods excluded from the adding-up check (missing 'actual'): {n_missing}"
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_markdown(self, variable: str | None = None, **kwargs) -> str:
        """Export variable's decomposition table to Markdown."""
        target_var = variable if variable is not None else self.variable_names[0]
        return _df_to_markdown(self.to_frame(target_var), **kwargs)

    def to_latex(self, variable: str | None = None, **kwargs) -> str:
        """Export variable's decomposition table to LaTeX tabular."""
        target_var = variable if variable is not None else self.variable_names[0]
        return _df_to_latex(self.to_frame(target_var), **kwargs)

    def to_typst(self, variable: str | None = None, **kwargs) -> str:
        """Export variable's decomposition table to Typst table."""
        target_var = variable if variable is not None else self.variable_names[0]
        return _df_to_typst(self.to_frame(target_var), **kwargs)

    def plot(
        self,
        variable: str,
        style: str = "publication",
    ) -> Figure:
        """Stacked-bar chart of historical shock contributions overlaid with actual data.

        Parameters
        ----------
        variable : str
            Endogenous variable to plot.
        style : str, default 'publication'
            Visual styling theme.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with the stacked historical shock decomposition.
        """
        df = self.to_frame(variable)
        T = len(df)
        t = np.arange(T) if isinstance(df.index, pd.RangeIndex) else df.index

        # Bar width is in data units. On a date axis one data unit is one DAY,
        # so the matplotlib default of 0.8 draws a 0.8-day sliver for quarterly
        # or monthly data: scale it by the actual spacing of the index.
        width = 0.8
        if not isinstance(df.index, pd.RangeIndex) and T > 1:
            try:
                pos = mdates.date2num(pd.DatetimeIndex(df.index).to_pydatetime())
                spacing = float(np.median(np.diff(np.asarray(pos, dtype=float))))
                if np.isfinite(spacing) and spacing > 0.0:
                    width = 0.8 * spacing
            except (TypeError, ValueError):
                pass

        fig, ax = _new_ax(None, figsize=(8.0, 4.4))

        comp_names = list(self.shock_names) + ["initial_condition"]
        # The residual is only a component when it is visible; drawing it keeps
        # the stacked bars adding up to the black 'actual' line.
        if "residual" in df.columns:
            resid = np.abs(df["residual"].to_numpy(dtype=float))
            actual_scale = np.nanmax(np.abs(df["actual"].to_numpy(dtype=float)))
            ref = max(float(actual_scale) if np.isfinite(actual_scale) else 0.0, 1.0)
            if np.nanmax(resid) > 1e-8 * ref:
                comp_names.append("residual")
        colors = _palette(len(comp_names))

        pos_bottom = np.zeros(T)
        neg_bottom = np.zeros(T)

        for idx, comp in enumerate(comp_names):
            vals = df[comp].to_numpy()
            pos_vals = np.where(vals > 0, vals, 0.0)
            neg_vals = np.where(vals < 0, vals, 0.0)

            c = colors[idx % len(colors)]
            ax.bar(
                t,
                pos_vals,
                width=width,
                bottom=pos_bottom,
                label=comp,
                color=c,
                edgecolor="none",
                alpha=0.85,
            )
            pos_bottom += pos_vals

            ax.bar(
                t,
                neg_vals,
                width=width,
                bottom=neg_bottom,
                color=c,
                edgecolor="none",
                alpha=0.85,
            )
            neg_bottom += neg_vals

        # Actual deviation from steady state
        actual_dev = (df["actual"] - df["steady_state"]).to_numpy()
        ax.plot(
            t,
            actual_dev,
            color="black",
            linewidth=1.8,
            label=f"Actual ({variable} - SS)",
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(
            f"Historical Shock Decomposition: {variable}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Period")
        ax.set_ylabel("Deviation from Steady State")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout()
        return fig


def _extract_companion_matrices(
    model: Any,
    sigma: float | Mapping[str, float] | Sequence[float] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
    np.ndarray,
]:
    """Extract ``(A, B, C, D, ys, variables, shocks, states, sigma_u)``.

    ``(A, B, C, D)`` is the companion form of the reported vector *in the
    timing the model reports it*: ``x_{t+1} = A x_t + B u_t`` and
    ``v_t = ys + C x_t + D u_t``. For ``build_dynare`` / ``load_mod`` models
    that is Dynare's ``(ghx, ghu)``; for ``build`` (Klein timing) models it is
    ``([I; F], [0; L])``, matching ``irf()`` / ``simulate()`` /
    ``theoretical_moments()``. ``sigma_u`` is the full innovation covariance,
    correlations included.
    """
    # Case 1: SWResult (from puremacro.dsge.smets_wouters.solve_sw07)
    if hasattr(model, "Impact") and hasattr(model, "state_names") and hasattr(model, "control_names"):
        states = list(model.state_names)
        controls = list(model.control_names)
        variables = states + controls
        shocks = list(model.shock_names)
        n_s = len(states)
        A = np.asarray(model.G[:n_s, :n_s], dtype=float)
        B = np.asarray(model.Impact[:n_s, :], dtype=float)
        C = np.asarray(model.G[:, :n_s], dtype=float)
        D = np.asarray(model.Impact, dtype=float)
        ys = np.zeros(len(variables))

        from puremacro.dsge.smets_wouters import SW07_SHOCK_STDS

        if sigma is None:
            sigmas = np.array([SW07_SHOCK_STDS.get(s, 1.0) for s in shocks], dtype=float)
        elif isinstance(sigma, Mapping):
            sigmas = np.array([float(sigma.get(s, 1.0)) for s in shocks], dtype=float)
        elif isinstance(sigma, (int, float)):
            sigmas = np.full(len(shocks), float(sigma))
        else:
            sigmas = np.asarray(sigma, dtype=float)

        return A, B, C, D, ys, variables, shocks, states, np.diag(sigmas ** 2)

    # Case 2: LinearModel (from puremacro.dsge.build or load_mod) or DynareDR
    if hasattr(model, "decision_rules"):
        dr = model.decision_rules()
    elif hasattr(model, "dynare_dr") and model.dynare_dr is not None:
        dr = model.dynare_dr
    elif hasattr(model, "ghx") and hasattr(model, "ghu"):
        dr = model
    else:
        raise TypeError(
            f"Unsupported model type {type(model).__name__}: expected LinearModel, DynareDR, or SWResult."
        )

    states = list(dr.state_variables)
    variables = list(dr.variable_names)
    shocks = list(dr.shock_names)

    A = dr.ghx.loc[states, states].to_numpy(dtype=float)
    B = dr.ghu.loc[states, shocks].to_numpy(dtype=float)
    C = dr.ghx.loc[variables, states].to_numpy(dtype=float)
    D = dr.ghu.loc[variables, shocks].to_numpy(dtype=float)

    # ``decision_rules()`` is always Dynare-timed: its state rows are the
    # *end-of-period* state x_{t+1} = G x_t + N u_t. A model built with
    # ``build`` reports the state itself at t, so its loadings are
    # ([I; F], [0; L]) -- using ghx/ghu there would date every state row one
    # period later than the control rows and than the model's own moments.
    # For ``build_dynare`` / ``load_mod`` ``_reported_loadings()`` returns
    # exactly ([G; F], [N; L]) = (ghx, ghu), so those results are unchanged.
    if hasattr(model, "_reported_loadings") and hasattr(model, "controls"):
        M_x, M_u = model._reported_loadings()
        order_vars = list(model.states) + list(model.controls)
        try:
            idx = [order_vars.index(v) for v in variables]
        except ValueError:
            idx = None
        if idx is not None and np.shape(M_x)[0] == len(order_vars):
            C = np.asarray(M_x, dtype=float)[idx]
            D = np.asarray(M_u, dtype=float)[idx]

    if hasattr(dr, "ys") and dr.ys is not None:
        ys = dr.ys.loc[variables].to_numpy(dtype=float)
    else:
        ys = np.zeros(len(variables))

    # Innovation covariance. An explicit ``sigma=`` is always diagonal (it is
    # a set of standard deviations); a covariance declared with the model
    # (``shocks`` block / ``shock_cov=``) is carried through in full, including
    # the correlations that ``np.sqrt(np.diag(...))`` used to discard.
    n_u = len(shocks)
    if sigma is not None:
        if isinstance(sigma, Mapping):
            sigmas = np.array([float(sigma.get(s, 1.0)) for s in shocks], dtype=float)
        elif isinstance(sigma, (int, float)):
            sigmas = np.full(n_u, float(sigma))
        else:
            sigmas = np.asarray(sigma, dtype=float)
        return A, B, C, D, ys, variables, shocks, states, np.diag(sigmas ** 2)

    declared = None
    for attr in ("_shock_cov", "shock_cov"):
        cov = getattr(model, attr, None)
        if cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.shape == (n_u, n_u):
                declared = 0.5 * (cov + cov.T)
            break
    if declared is not None:
        return A, B, C, D, ys, variables, shocks, states, declared

    if hasattr(model, "_params") and isinstance(model._params, dict):
        p = model._params
        sig_list: list[float] = []
        for s in shocks:
            val = p.get(f"stderr_{s}", p.get(f"sigma_{s}", p.get(s, 1.0)))
            sig_list.append(float(val) if val is not None else 1.0)
        sigmas = np.array(sig_list, dtype=float)
    else:
        sigmas = np.ones(n_u, dtype=float)

    return A, B, C, D, ys, variables, shocks, states, np.diag(sigmas ** 2)


def compute_fevd(
    model: Any,
    horizons: Sequence[int | None] | None = None,
    *,
    sigma: float | Mapping[str, float] | Sequence[float] | None = None,
) -> FEVDResult:
    """Compute Forecast Error Variance Decomposition (FEVD) matching Dynare parity.

    Parameters
    ----------
    model : LinearModel, DynareDR, or SWResult
        Solved DSGE model.
    horizons : Sequence[int | None], optional
        Forecast horizons to evaluate. None or np.inf indicates asymptotic
        unconditional variance shares computed via the discrete Lyapunov equation.
        Defaults to (1, 4, 8, 16, 32, None).
    sigma : float | Mapping[str, float] | Sequence[float], optional
        Shock standard deviations. Defaults to the shock covariance declared
        with the model (``shocks`` block / ``shock_cov=``) or 1.0.

    Returns
    -------
    FEVDResult
        Frozen dataclass with multi-index DataFrame ('Variable', 'Horizon')
        and export / plotting methods. Shares are fractions summing to 1;
        a variable with zero forecast-error variance at a horizon has NaN
        shares there (and a ``ZeroVarianceWarning`` is issued).

    Raises
    ------
    ValueError
        If the model declares correlated innovations (a variance
        decomposition per shock is not defined without an orthogonalisation
        convention), if an asymptotic horizon is requested for a model with a
        unit or explosive root (the unconditional variance is infinite), or if
        a horizon is not a positive integer.

    Notes
    -----
    The decomposition is stated in the timing the model reports its variables
    in -- the same timing as :meth:`~puremacro.dsge.LinearModel.irf`,
    ``simulate`` and ``theoretical_moments``. For a Klein-timed model built
    with :func:`~puremacro.dsge.build`, the state rows are predetermined, so
    their one-step forecast error is exactly zero and their ``h = 1`` shares
    are undefined (NaN).
    """
    (
        A,
        B,
        C,
        D,
        ys,
        variables,
        shocks,
        states,
        sigma_u,
    ) = _extract_companion_matrices(model, sigma)

    if horizons is None:
        eval_horizons: list[int | None] = [1, 4, 8, 16, 32, None]
    else:
        eval_horizons = list(horizons)

    df_fevd = conditional_fevd(A, B, C, D, sigma_u, eval_horizons, variables, shocks)

    return FEVDResult(
        table=df_fevd,
        horizons=[h for h in eval_horizons],
        variable_names=list(variables),
        shock_names=list(shocks),
    )


def compute_shock_decomposition(
    model: Any,
    data: pd.DataFrame,
    initial_state: np.ndarray | None = None,
    *,
    sigma: float | Mapping[str, float] | Sequence[float] | None = None,
) -> ShockDecompResult:
    """Compute Historical Shock Decomposition using the Kalman smoother.

    Parameters
    ----------
    model : LinearModel, DynareDR, or SWResult
        Solved DSGE model.
    data : pd.DataFrame
        Observed time-series data with column names matching model variables.
    initial_state : np.ndarray, optional
        Initial state vector x_0 at t=0 (before the t=0 innovation). If None,
        estimated via the Kalman smoother initialized from the unconditional mean.
    sigma : float | Mapping[str, float] | Sequence[float], optional
        Shock standard deviations. Defaults to the covariance declared with
        the model (correlations included).

    Returns
    -------
    ShockDecompResult
        Frozen dataclass with per-variable decomposition components, smoothed
        shocks, and publication-ready export and plotting tools.

    Notes
    -----
    The shocks are the Kalman *smoother*'s estimates ``E[u_t | y_{1:T}]``,
    which weight each shock direction by its declared variance. There is no
    "exact projection" shortcut: solving ``D_obs u_t = y_t - ys - C_obs x_t``
    by least squares picks the minimum-norm solution whenever there are fewer
    observables than shocks, which is exact by construction (so the shortcut
    always fires) and splits every movement evenly across shock directions
    regardless of how large or small their declared standard deviations are.

    The decomposition is stated in the timing the model reports its variables
    in, so ``data`` must be in the same timing as :meth:`simulate` /
    :meth:`irf` output (plus the steady state).

    Because the model reconstruction need not reproduce the observations
    exactly -- more observables than shocks, a mis-specified model, genuine
    measurement error -- the 'actual' column always holds the caller's data
    and the leftover goes into an explicit 'residual' column. A residual that
    is large relative to the series triggers a ``UserWarning`` naming the
    variables concerned: it means the shock contributions on their own do not
    account for the data.
    """
    (
        A,
        B,
        C,
        D,
        ys,
        variables,
        shocks,
        states,
        sigma_u,
    ) = _extract_companion_matrices(model, sigma)

    obs_vars = [v for v in variables if v in data.columns]
    if len(obs_vars) == 0:
        raise ValueError(
            f"None of model variables {variables} are found in data columns {list(data.columns)}."
        )

    obs_idx = [variables.index(v) for v in obs_vars]
    n_s = len(states)
    n_u = len(shocks)
    n_obs = len(obs_vars)
    T_periods = len(data)

    C_obs = C[obs_idx, :]
    D_obs = D[obs_idx, :]
    ys_obs = ys[obs_idx]
    Y_obs = data[obs_vars].to_numpy(dtype=float)

    # Augmented state-space system where alpha_t = [x_t; u_t]
    # alpha_{t+1} = [A B; 0 0] alpha_t + [0; I] u_{t+1}
    # y_t         = ys_obs + [C_obs D_obs] alpha_t
    Tm = np.block([
        [A, B],
        [np.zeros((n_u, n_s)), np.zeros((n_u, n_u))],
    ])
    Rm = np.block([
        [np.zeros((n_s, n_u))],
        [np.eye(n_u)],
    ])
    Zm = np.hstack([C_obs, D_obs])
    dm = ys_obs
    # Full innovation covariance: correlations between structural shocks are
    # part of the model and belong in the smoother's Q.
    Qm = np.asarray(sigma_u, dtype=float)

    # Set up Kalman smoother initial conditions
    if initial_state is not None:
        s_0 = np.asarray(initial_state, dtype=float)
        a0 = np.hstack([s_0, np.zeros(n_u)])
        P0 = scipy.linalg.block_diag(1e-15 * np.eye(n_s), Qm)
    else:
        # Stationary covariance initialization
        try:
            Sigma_s = scipy.linalg.solve_discrete_lyapunov(A, B @ Qm @ B.T)
        except Exception:
            Sigma_s = 10.0 * np.eye(n_s)
        a0 = np.zeros(n_s + n_u)
        P0 = scipy.linalg.block_diag(Sigma_s, Qm)

    Hm = 1e-15 * np.eye(n_obs)
    ssm = StateSpaceModel(T=Tm, Z=Zm, R=Rm, Q=Qm, H=Hm, d=dm)

    try:
        out = kalman_smoother(Y_obs, ssm, a0=a0, P0=P0)
        a_smooth = out["a_smooth"]
        u_hat = a_smooth[:, n_s:]
        s_0_hat = s_0 if initial_state is not None else a_smooth[0, :n_s]
    except Exception:
        # Fallback with slightly higher measurement variance ridge for stability
        ssm.H = 1e-10 * np.eye(n_obs)
        out = kalman_smoother(Y_obs, ssm, a0=a0, P0=P0)
        a_smooth = out["a_smooth"]
        u_hat = a_smooth[:, n_s:]
        s_0_hat = s_0 if initial_state is not None else a_smooth[0, :n_s]

    # The smoothed shocks are used as they are. An earlier version overrode
    # them with a per-period least-squares solve of D_obs u_t = y_t - ys -
    # C_obs x_t, accepted whenever the residual was below 1e-10. With fewer
    # observables than shocks that system is underdetermined, `lstsq` returns
    # the minimum-norm solution, its residual is zero by construction, and the
    # override therefore fired at *every* period -- discarding the smoother,
    # ignoring the shock covariance entirely, and splitting each observed
    # movement equally across shock directions. The smoother already
    # reproduces exactly-identified, noiseless data to ~1e-13.
    u_effective = u_hat

    # 1. Initial condition decay: C @ A^t @ x_0
    y_init = np.zeros((T_periods, len(variables)))
    s_curr_init = s_0_hat.copy()
    for t in range(T_periods):
        y_init[t] = C @ s_curr_init
        s_curr_init = A @ s_curr_init

    # 2. Individual shock paths
    y_shk: dict[str, np.ndarray] = {s: np.zeros((T_periods, len(variables))) for s in shocks}
    for j, s in enumerate(shocks):
        s_curr_shk = np.zeros(n_s)
        for t in range(T_periods):
            u_vec = np.zeros(n_u)
            u_vec[j] = u_effective[t, j]
            y_shk[s][t] = C @ s_curr_shk + D @ u_vec
            s_curr_shk = A @ s_curr_shk + B @ u_vec

    # 3. Assemble decomposition components per variable
    components: dict[str, pd.DataFrame] = {}
    unexplained: list[tuple[str, float, float]] = []
    for i, var in enumerate(variables):
        df_var = pd.DataFrame(index=data.index)
        for s in shocks:
            df_var[s] = y_shk[s][:, i]
        df_var["initial_condition"] = y_init[:, i]
        df_var["steady_state"] = ys[i]

        recon = (
            df_var["steady_state"]
            + df_var["initial_condition"]
            + df_var[shocks].sum(axis=1)
        ).to_numpy(dtype=float)

        if var in data.columns:
            # The observed series goes into 'actual' unconditionally: replacing
            # it with the model's own reconstruction made the adding-up
            # invariant an algebraic identity that could never fail.
            actual = np.asarray(data[var], dtype=float)
        else:
            actual = recon

        residual = actual - recon
        df_var["residual"] = residual
        df_var["actual"] = actual

        if var in data.columns:
            finite = np.isfinite(residual)
            if finite.any():
                max_resid = float(np.max(np.abs(residual[finite])))
                scale = float(np.max(np.abs((actual - ys[i])[finite])))
                if max_resid > 1e-8 * max(scale, 1e-12):
                    unexplained.append((var, max_resid, scale))

        components[var] = df_var

    if unexplained:
        unexplained.sort(key=lambda item: -item[1])
        shown = ", ".join(
            f"{v} (max|residual|={r:.3e} vs series range {sc:.3e})"
            for v, r, sc in unexplained[:5]
        )
        more = "" if len(unexplained) <= 5 else f" (+{len(unexplained) - 5} more)"
        warnings.warn(
            "compute_shock_decomposition: the smoothed model path does not "
            f"reproduce the observed data for {shown}{more}. The shock "
            "contributions, the initial condition and the steady state "
            "therefore do not add up to 'actual' on their own; the gap is "
            "reported in the 'residual' column. This is expected when there "
            "are more observed series than structural shocks, and a sign of "
            "misspecification otherwise.",
            UserWarning,
            stacklevel=2,
        )

    smoothed_shocks_df = pd.DataFrame(
        np.asarray(u_effective, dtype=float), index=data.index, columns=shocks
    )

    return ShockDecompResult(
        components=components,
        variable_names=list(variables),
        shock_names=list(shocks),
        smoothed_shocks=smoothed_shocks_df,
    )


__all__ = [
    "FEVDResult",
    "ShockDecompResult",
    "compute_fevd",
    "compute_shock_decomposition",
]
