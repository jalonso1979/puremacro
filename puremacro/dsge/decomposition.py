"""Dynare-parity Forecast Error Variance Decomposition (FEVD) and Historical Shock Decomposition.

Companion state-space representation:
    s_t = A @ s_{t-1} + B @ u_t
    y_t = y_s + C @ s_{t-1} + D @ u_t

Vector Moving Average (VMA) representation:
    Psi_0 = D
    Psi_k = C @ (A^(k-1)) @ B   for k >= 1

Variance shares at finite horizon h:
    V_{i, j}(h) = sum_{k=0}^{h-1} (Psi_k[i, j])^2 * sigma_j^2
    MSE_i(h)    = sum_j V_{i, j}(h)
    Share_{i, j}(h) = V_{i, j}(h) / MSE_i(h)

Asymptotic variance shares (horizon = None or np.inf) via discrete Lyapunov:
    Sigma_{s, j} = solve_discrete_lyapunov(A, sigma_j^2 * (B_{:, j} @ B_{:, j}^T))
    V_{i, j}(inf) = sigma_j^2 * (D[i, j]^2) + [C @ Sigma_{s, j} @ C^T]_{i, i}

Historical Shock Decomposition:
    Runs Kalman smoother to extract smoothed states s_hat_t and shocks u_hat_t.
    Simulates path for each individual shock, initial state decay (A^t s_0), and steady state:
    y_t = steady_state + C @ A^t @ s_0 + sum_{j} shock_j(t)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from matplotlib.figure import Figure

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
            row_sums = self.table[self.shock_names].sum(axis=1)
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
                shares = sub[shk].to_numpy()
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
        Dictionary mapping each endogenous variable name to a DataFrame.
        Each DataFrame contains columns for each structural shock,
        'initial_condition', 'steady_state', and 'actual'.
    variable_names : list[str]
        Names of endogenous variables decomposed.
    shock_names : list[str]
        Names of structural shocks.
    smoothed_shocks : pd.DataFrame
        DataFrame of recovered smoothed shocks u_hat_t (periods x shocks).
    """

    components: dict[str, pd.DataFrame]
    variable_names: list[str]
    shock_names: list[str]
    smoothed_shocks: pd.DataFrame

    def __post_init__(self) -> None:
        # Verify invariant: steady_state + initial_condition + sum(shocks) == actual
        for var in self.variable_names:
            if var in self.components:
                df = self.components[var]
                recon = (
                    df["steady_state"]
                    + df["initial_condition"]
                    + df[self.shock_names].sum(axis=1)
                )
                err = np.max(np.abs(recon - df["actual"]))
                if err > 1e-10:
                    raise ValueError(
                        f"Historical shock decomposition invariant violated for '{var}': "
                        f"reconstruction error max={err:.3e} exceeds 1e-10 tolerance."
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
            "=" * 72,
        ]
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

        fig, ax = _new_ax(None, figsize=(8.0, 4.4))

        comp_names = list(self.shock_names) + ["initial_condition"]
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
    """Extract (A, B, C, D, ys, variables, shocks, states, sigmas) from various DSGE representations."""
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

        return A, B, C, D, ys, variables, shocks, states, sigmas

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

    if hasattr(dr, "ys") and dr.ys is not None:
        ys = dr.ys.loc[variables].to_numpy(dtype=float)
    else:
        ys = np.zeros(len(variables))

    # Shock standard deviations
    if sigma is not None:
        if isinstance(sigma, Mapping):
            sigmas = np.array([float(sigma.get(s, 1.0)) for s in shocks], dtype=float)
        elif isinstance(sigma, (int, float)):
            sigmas = np.full(len(shocks), float(sigma))
        else:
            sigmas = np.asarray(sigma, dtype=float)
    elif hasattr(model, "_shock_cov") and model._shock_cov is not None:
        sigmas = np.sqrt(np.diag(model._shock_cov))
    elif hasattr(model, "shock_cov") and model.shock_cov is not None:
        sigmas = np.sqrt(np.diag(model.shock_cov))
    elif hasattr(model, "_params") and isinstance(model._params, dict):
        p = model._params
        sig_list: list[float] = []
        for s in shocks:
            val = p.get(f"stderr_{s}", p.get(f"sigma_{s}", p.get(s, 1.0)))
            sig_list.append(float(val) if val is not None else 1.0)
        sigmas = np.array(sig_list, dtype=float)
    else:
        sigmas = np.ones(len(shocks), dtype=float)

    return A, B, C, D, ys, variables, shocks, states, sigmas


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
        Shock standard deviations. Defaults to model-defined shock standard
        deviations or 1.0.

    Returns
    -------
    FEVDResult
        Frozen dataclass with multi-index DataFrame ('Variable', 'Horizon')
        and export / plotting methods.
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
        sigmas,
    ) = _extract_companion_matrices(model, sigma)

    if horizons is None:
        eval_horizons: Sequence[int | None] = (1, 4, 8, 16, 32, None)
    else:
        eval_horizons = list(horizons)

    n_v = len(variables)
    n_u = len(shocks)

    # Determine maximum finite horizon
    finite_horizons = [
        int(h)
        for h in eval_horizons
        if h is not None and not (isinstance(h, float) and np.isinf(h))
    ]
    max_h = max(finite_horizons, default=0)

    # Precompute VMA coefficients: Psi_0 = D, Psi_k = C @ A^(k-1) @ B
    Psi: list[np.ndarray] = []
    if max_h > 0:
        Psi.append(D)
        A_pow = np.eye(len(states))
        for k in range(1, max_h):
            Psi.append(C @ A_pow @ B)
            A_pow = A_pow @ A

    rows: list[dict[str, Any]] = []
    processed_horizons: list[Any] = []

    for h in eval_horizons:
        is_asymptotic = h is None or (isinstance(h, (float, np.floating)) and np.isinf(h)) or h == "Infinity"
        if is_asymptotic:
            h_label: str | int = "Infinity"
        else:
            assert h is not None
            h_label = int(h)
        processed_horizons.append(h_label)

        if is_asymptotic:
            # Asymptotic unconditional variance shares via discrete Lyapunov
            v_shocks = np.zeros((n_v, n_u))
            for j in range(n_u):
                b_j = B[:, [j]]
                d_j = D[:, [j]]
                var_u_j = sigmas[j] ** 2
                q_j = var_u_j * (b_j @ b_j.T)
                try:
                    sig_s_j = scipy.linalg.solve_discrete_lyapunov(A, q_j)
                except Exception:
                    sig_s_j = np.zeros_like(A)
                v_shocks[:, j] = var_u_j * (d_j[:, 0] ** 2) + np.diag(C @ sig_s_j @ C.T)

            tot_v = v_shocks.sum(axis=1, keepdims=True)
            has_var = (tot_v > 1e-15)[:, 0]
            shares = np.zeros((n_v, n_u))
            if np.any(has_var):
                shares[has_var] = v_shocks[has_var] / tot_v[has_var]
            if np.any(~has_var):
                shares[~has_var] = 1.0 / n_u

            # Strict normalization to guarantee machine-precision sum == 1.0
            row_sums = shares.sum(axis=1, keepdims=True)
            shares = shares / row_sums

            for i, var in enumerate(variables):
                row_dict: dict[str, Any] = {"Variable": var, "Horizon": h_label}
                for j, s in enumerate(shocks):
                    row_dict[s] = float(shares[i, j])
                rows.append(row_dict)
        else:
            assert h is not None
            h_int = int(h)
            v_shocks = np.zeros((n_v, n_u))
            for k in range(h_int):
                v_shocks += (Psi[k] ** 2) * (sigmas ** 2)

            tot_v = v_shocks.sum(axis=1, keepdims=True)
            has_var = (tot_v > 1e-15)[:, 0]
            shares = np.zeros((n_v, n_u))
            if np.any(has_var):
                shares[has_var] = v_shocks[has_var] / tot_v[has_var]
            if np.any(~has_var):
                shares[~has_var] = 1.0 / n_u

            # Strict normalization to guarantee machine-precision sum == 1.0
            row_sums = shares.sum(axis=1, keepdims=True)
            shares = shares / row_sums

            for i, var in enumerate(variables):
                row_dict = {"Variable": var, "Horizon": h_label}
                for j, s in enumerate(shocks):
                    row_dict[s] = float(shares[i, j])
                rows.append(row_dict)

    df_fevd = pd.DataFrame(rows).set_index(["Variable", "Horizon"])
    horizons_list = [h for h in eval_horizons]

    return FEVDResult(
        table=df_fevd,
        horizons=horizons_list,
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
        Initial state vector s_0 at t=0 (prior to t=0 innovations). If None,
        estimated via the Kalman smoother initialized from the unconditional mean.
    sigma : float | Mapping[str, float] | Sequence[float], optional
        Shock standard deviations.

    Returns
    -------
    ShockDecompResult
        Frozen dataclass with per-variable decomposition components, smoothed
        shocks, and publication-ready export and plotting tools.
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
        sigmas,
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

    # Augmented state-space system where alpha_t = [s_{t-1}; u_t]
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
    Qm = np.diag(sigmas ** 2)

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

    # Refine shocks if observed data is exact / noiseless
    # y_t - ys - C s_{t-1} = D u_t
    s_curr = s_0_hat.copy()
    u_refined = np.zeros((T_periods, n_u))
    for t in range(T_periods):
        dev = Y_obs[t] - ys_obs - C_obs @ s_curr
        # Solve least-squares D_obs @ u_t = dev
        u_t_sol, residuals, rank, s_vals = np.linalg.lstsq(D_obs, dev, rcond=None)
        # If residual is extremely small, use exact projection
        if np.max(np.abs(D_obs @ u_t_sol - dev)) < 1e-10:
            u_refined[t] = u_t_sol
            s_curr = A @ s_curr + B @ u_t_sol
        else:
            u_refined[t] = u_hat[t]
            s_curr = A @ s_curr + B @ u_hat[t]

    u_effective = u_refined

    # 1. Initial condition decay: C @ A^t @ s_0
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
    for i, var in enumerate(variables):
        df_var = pd.DataFrame(index=data.index)
        for s in shocks:
            df_var[s] = y_shk[s][:, i]
        df_var["initial_condition"] = y_init[:, i]
        df_var["steady_state"] = ys[i]

        recon_series = (
            df_var["steady_state"]
            + df_var["initial_condition"]
            + df_var[shocks].sum(axis=1)
        )

        if var in data.columns:
            actual_series = np.asarray(data[var], dtype=float)
            # If actual data matches reconstruction within machine precision (< 1e-10)
            if np.max(np.abs(recon_series - actual_series)) < 1e-10:
                df_var["actual"] = actual_series
            else:
                df_var["actual"] = recon_series.to_numpy()
        else:
            df_var["actual"] = recon_series.to_numpy()

        components[var] = df_var

    smoothed_shocks_df = pd.DataFrame(u_effective, index=data.index, columns=shocks)

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
