"""News and anticipated shocks engine for DSGE models.

Motor de perturbaciones anticipadas y de noticias (news shocks) para modelos DSGE.

Implements companion state-space augmentation for forward-looking announcements:
    epsilon_t = eta_t^0 + sum_{k=1}^H eta_{t-k}^k
where eta_t^0 is the surprise innovation and eta_t^k is the news innovation
announced at date t about the shock that will realize at date t+k.

The auxiliary state vector V_t = [nu_{1,t}, ..., nu_{H,t}]' evolves according to:
    V_t = K_H V_{t-1} + eta_t
where K_H is an H x H strictly upper triangular matrix (superdiagonal of 1s).
All eigenvalues of K_H are identically 0, strictly preserving Blanchard-Kahn
determinacy.

References
----------
Beaudry, P. & Portier, F. (2006). "Stock Prices, News, and Economic Fluctuations."
    American Economic Review, 96(4), 1293-1307.
Schmitt-Grohé, S. & Uribe, M. (2012). "What's News in Business Cycles."
    Econometrica, 80(6), 2733-2764.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from puremacro.dsge.build import LinearModel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from puremacro.dsge.klein import BlanchardKahnError, KleinSolution, klein_solve
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


@dataclass(frozen=True)
class NewsIRFResult:
    """Impulse response container for surprise and news (anticipated) shocks.

    Contenedor de funciones de impulso-respuesta para perturbaciones sorpresa y
    anticipadas (noticias).

    Attributes
    ----------
    irf : pd.DataFrame
        Impulse responses of all model variables over horizon 0..horizon under
        the news shock announcement at t=0 for realization at t=lead.
    surprise_irf : pd.DataFrame
        Impulse responses under the contemporaneous surprise shock (lead=0).
    shock : str
        Name of the structural shock innovation.
    lead : int
        Anticipation lead in periods (k >= 0).
    horizon : int
        Simulation horizon in periods.
    size : float
        Magnitude of the structural shock.
    model : Any
        Underlying LinearModel.
    """

    irf: pd.DataFrame
    surprise_irf: pd.DataFrame
    shock: str
    lead: int
    horizon: int
    size: float
    model: Any

    def to_frame(self) -> pd.DataFrame:
        """Return the news impulse response DataFrame."""
        return self.irf.copy()

    def to_markdown(self, **kwargs: Any) -> str:
        """Export news impulse response table to Markdown format."""
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export news impulse response table to LaTeX tabular format."""
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export news impulse response table to Typst table format."""
        return _df_to_typst(self.to_frame(), **kwargs)

    def summary(self) -> str:
        """Generate formatted summary table comparing impact, realization, and peak."""
        lines = [
            "=" * 78,
            f"News Shock IRF Summary: shock='{self.shock}', lead={self.lead}, horizon={self.horizon}, size={self.size}",
            "-" * 78,
            f"{'Variable':<12} {'Impact (t=0)':<14} {'Pre-realiz':<14} {'Realiz (t=k)':<14} {'Peak Dev':<14} {'Peak t':<8}",
            "-" * 78,
        ]

        df = self.irf
        for var in df.columns:
            series = df[var]
            impact = series.iloc[0]
            if self.lead > 0 and self.lead < len(series):
                pre_val = series.iloc[self.lead - 1]
                realiz_val = series.iloc[self.lead]
            elif self.lead == 0:
                pre_val = np.nan
                realiz_val = impact
            else:
                pre_val = series.iloc[-1]
                realiz_val = series.iloc[-1]

            abs_series = np.abs(series.to_numpy())
            peak_idx = int(np.argmax(abs_series))
            peak_val = series.iloc[peak_idx]

            pre_str = f"{pre_val:12.4f}" if np.isfinite(pre_val) else f"{'N/A':>12}"
            lines.append(
                f"{var:<12} {impact:12.4f}  {pre_str}  {realiz_val:12.4f}  {peak_val:12.4f}  {peak_idx:<8}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        compare_surprise: bool = True,
        *,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Any]:
        """Plot publication-grade impulse response comparisons."""
        if variables is None:
            vars_to_plot = list(self.irf.columns)[:6]
        else:
            vars_to_plot = [v for v in variables if v in self.irf.columns]

        if not vars_to_plot:
            raise ValueError("No valid variables specified to plot.")

        n_vars = len(vars_to_plot)

        if n_vars == 1 and ax is not None:
            fig = ax.figure
            axes_array = np.array([ax])
            created_fig = False
        else:
            created_fig = True
            n_cols = 2 if n_vars > 1 else 1
            n_rows = int(np.ceil(n_vars / n_cols))
            calc_figsize = figsize or (5.5 * n_cols, 3.5 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=calc_figsize, squeeze=False)
            axes_array = axes.flatten()

        color_news = "#1f77b4" if style != "grayscale" else "0.2"
        color_surp = "#d62728" if style != "grayscale" else "0.5"

        for idx, var in enumerate(vars_to_plot):
            target_ax = axes_array[idx]
            h_grid = np.arange(len(self.irf))

            target_ax.plot(
                h_grid,
                self.irf[var],
                color=color_news,
                linewidth=2.0,
                label=f"News (lead={self.lead})",
            )
            if compare_surprise:
                target_ax.plot(
                    h_grid,
                    self.surprise_irf[var],
                    color=color_surp,
                    linestyle="--",
                    linewidth=1.6,
                    label="Surprise (lead=0)",
                )

            target_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
            target_ax.axvline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
            if self.lead > 0:
                target_ax.axvline(
                    self.lead,
                    color="#2ca02c" if style != "grayscale" else "0.4",
                    linestyle="-.",
                    linewidth=1.2,
                    alpha=0.8,
                    label="Realization (t=k)" if idx == 0 else None,
                )

            target_ax.set_title(f"{var}", fontsize=11, fontweight="bold")
            target_ax.set_xlabel("Horizon (h)")
            target_ax.set_ylabel("Deviation")
            target_ax.grid(True, linestyle=":", alpha=0.5)
            target_ax.spines["top"].set_visible(False)
            target_ax.spines["right"].set_visible(False)
            target_ax.legend(loc="best", frameon=False, fontsize=8)

        # Hide unused subplots
        for idx in range(n_vars, len(axes_array)):
            axes_array[idx].set_visible(False)

        if created_fig:
            fig.tight_layout()

        ret_ax = axes_array[0] if n_vars == 1 and not created_fig else axes_array
        return fig, ret_ax


@dataclass(frozen=True)
class NewsDecompositionResult:
    """Variance decomposition container separating surprise and news components.

    Contenedor de descomposición de varianza separando componentes sorpresa y
    noticias a distintos horizontes de anticipación.

    Attributes
    ----------
    variance_shares : pd.DataFrame
        Table of variance shares with index corresponding to variables and columns
        'surprise', 'news_1', ..., 'news_{max_lead}'. Rows sum to 1.0.
    historical : dict[str, pd.DataFrame] | None
        Optional historical shock decomposition trajectories.
    shock : str
        Name of the decomposed structural shock.
    horizon : int
        Evaluation horizon for the variance decomposition.
    max_lead : int
        Maximum anticipation lead included in the decomposition.
    model : Any
        Underlying LinearModel.
    dynamic_shares : dict[str, pd.DataFrame] | None, optional
        Per-variable variance shares indexed by horizon 0..horizon.
    """

    variance_shares: pd.DataFrame
    historical: dict[str, pd.DataFrame] | None
    shock: str
    horizon: int
    max_lead: int
    model: Any
    dynamic_shares: dict[str, pd.DataFrame] | None = None

    @property
    def total_news(self) -> pd.Series:
        """Sum of variance shares across all news leads (excluding surprise)."""
        news_cols = [c for c in self.variance_shares.columns if c.startswith("news_")]
        return self.variance_shares[news_cols].sum(axis=1)

    def to_frame(self) -> pd.DataFrame:
        """Return the variance shares DataFrame."""
        return self.variance_shares.copy()

    def to_markdown(self, **kwargs: Any) -> str:
        """Export variance shares to Markdown format."""
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export variance shares to LaTeX tabular format."""
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export variance shares to Typst table format."""
        return _df_to_typst(self.to_frame(), **kwargs)

    def summary(self) -> str:
        """Generate formatted summary table of forecast error variance shares."""
        df_display = self.variance_shares.copy()
        df_display["total_news"] = self.total_news

        lines = [
            "=" * 78,
            f"News Shock Variance Decomposition: shock='{self.shock}', horizon={self.horizon}, max_lead={self.max_lead}",
            "-" * 78,
            df_display.round(4).to_string(),
            "=" * 78,
        ]
        return "\n".join(lines)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Any]:
        """Plot stacked bar chart of variance decomposition shares."""
        df = self.variance_shares
        if variables is not None:
            vars_to_plot = [v for v in variables if v in df.index]
        else:
            vars_to_plot = list(df.index)[:10]

        if not vars_to_plot:
            raise ValueError("No valid variables specified to plot.")

        display_df = df.loc[vars_to_plot]

        if ax is None:
            calc_figsize = figsize or (8.5, max(4.0, 0.45 * len(vars_to_plot) + 1.5))
            fig, target_ax = plt.subplots(figsize=calc_figsize)
        else:
            fig = ax.figure
            target_ax = ax

        y_pos = np.arange(len(display_df))
        left = np.zeros(len(display_df))

        # Color palette for surprise and news components
        colors = ["#2b8cbe", "#7bccc4", "#bae4bc", "#f0f9e8", "#fc8d59", "#d7301f", "#fdcc8a", "#fef0d9"]
        if style == "grayscale":
            colors = [str(val) for val in np.linspace(0.2, 0.85, len(display_df.columns))]

        for idx, col in enumerate(display_df.columns):
            values = display_df[col].to_numpy(dtype=float)
            target_ax.barh(
                y_pos,
                values,
                left=left,
                color=colors[idx % len(colors)],
                label=col.replace("_", " ").title(),
                edgecolor="white",
                height=0.6,
            )
            left += values

        target_ax.set_yticks(y_pos)
        target_ax.set_yticklabels(display_df.index)
        target_ax.set_xlim(0.0, 1.0)
        target_ax.set_xlabel("Variance Share (Fraction)")
        target_ax.set_title(
            f"News Shock Variance Decomposition ({self.shock}, h={self.horizon})",
            fontsize=11,
            fontweight="bold",
        )
        target_ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)
        target_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=min(5, len(display_df.columns)), frameon=False)

        if ax is None:
            fig.tight_layout()
        return fig, target_ax


def augment_news_state_space(
    model: LinearModel,
    shock: str,
    max_lead: int = 8,
) -> LinearModel:
    """Construct companion state-space augmented model for anticipated news shocks.

    Constructs the augmented Klein state-space system introducing H auxiliary
    state variables V_t = [nu_{1,t}, ..., nu_{H,t}]' where:
        V_t = K_H V_{t-1} + eta_t
    and the structural shock entering the economic model at date t is:
        epsilon_{shock, t} = eta_t^0 + e_1' V_{t-1}
    Because K_H has all eigenvalues equal to 0, Blanchard-Kahn determinacy is
    unconditionally preserved.

    Parameters
    ----------
    model : LinearModel
        Base solved linear DSGE model.
    shock : str
        Name of structural shock innovation to augment.
    max_lead : int, default 8
        Maximum news anticipation lead H (H >= 1).

    Returns
    -------
    LinearModel
        Augmented LinearModel instance containing the auxiliary news states.
    """
    from puremacro.dsge.build import LinearModel, ModelError

    if shock not in model.shocks:
        raise ModelError(
            f"no shock named {shock!r}; declared shocks: {list(model.shocks)}"
        )
    if max_lead < 1:
        raise ValueError(f"max_lead must be at least 1, got {max_lead}")

    shock_idx = list(model.shocks).index(shock)
    H = int(max_lead)
    qz_criterium = getattr(model, "_qz_criterium", 1.0 + 1e-8)

    # Transition matrix K_H: 1s on superdiagonal, 0 elsewhere
    K_H = np.zeros((H, H))
    for i in range(H - 1):
        K_H[i, i + 1] = 1.0

    aux_states = tuple(f"nu_{shock}_{k}" for k in range(1, H + 1))
    aux_shocks = tuple(f"eta_{shock}_{k}" for k in range(1, H + 1))

    if model.timing == "dynare":
        n_s = len(model.states)
        n_vars = len(model.variables)
        n_u = len(model.shocks)
        n_pre = n_s + H
        n_fwd = n_vars
        n_tot = n_pre + n_fwd

        A_aug = np.zeros((n_tot, n_tot))
        B_aug = np.zeros((n_tot, n_tot))
        C_aug = np.zeros((n_tot, n_u + H))

        # 1. Model equations block (n_vars rows)
        A_aug[:n_vars, :n_s] = model.A[:n_vars, :n_s]
        A_aug[:n_vars, n_pre:] = model.A[:n_vars, n_s:]
        B_aug[:n_vars, :n_s] = model.B[:n_vars, :n_s]
        # Shock column C[:n_vars, shock_idx] loaded by e_1' V_{t-1} = nu_{1, t-1}
        B_aug[:n_vars, n_s] = model.C[:n_vars, shock_idx]
        B_aug[:n_vars, n_pre:] = model.B[:n_vars, n_s:]
        C_aug[:n_vars, :n_u] = model.C[:n_vars, :]

        # 2. Auxiliary news state block (H rows): I_H V_t - K_H V_{t-1} = I_H eta_t
        A_aug[n_vars : n_vars + H, n_s:n_pre] = np.eye(H)
        B_aug[n_vars : n_vars + H, n_s:n_pre] = K_H
        C_aug[n_vars : n_vars + H, n_u:] = np.eye(H)

        # 3. Identity rows (n_s rows): I_s s_t - P_s y_t = 0
        A_aug[n_vars + H :, :n_s] = np.eye(n_s)
        B_aug[n_vars + H :, n_pre:] = model.B[n_vars:, n_s:]

        sol_full = klein_solve(
            A_aug, B_aug, n_pre=n_pre, C=C_aug, strict=True, div=qz_criterium
        )
        F_full = np.asarray(sol_full.F, dtype=float)
        L_full = np.asarray(sol_full.L, dtype=float)

        state_idx = [model.variables.index(v) for v in model.states]
        ctrl_idx = [model.variables.index(v) for v in model.controls]

        # In Dynare timing, G is transition of s_t, but here we keep full augmented state
        G_s = F_full[state_idx]
        G_aug = np.zeros((n_pre, n_pre))
        G_aug[:n_s, :] = G_s
        G_aug[n_s:, n_s:] = K_H
        N_aug = np.zeros((n_pre, n_u + H))
        N_aug[:n_s, :] = L_full[state_idx]
        N_aug[n_s:, n_u:] = np.eye(H)

        F_aug = F_full[ctrl_idx]
        L_aug = L_full[ctrl_idx]

        aug_solution = KleinSolution(
            G=G_aug,
            F=F_aug,
            N=N_aug,
            L=L_aug,
            eu=tuple(sol_full.eu),
            eigenvalues=sol_full.eigenvalues,
        )

        aug_states = tuple(model.states) + aux_states
        aug_shocks_tuple = tuple(model.shocks) + aux_shocks

        aug_ss = pd.concat([model.steady_state, pd.Series(0.0, index=list(aux_states))])
        aug_units = dict(model.units)
        for s in aux_states:
            aug_units[s] = "level"

        m_aug = LinearModel(
            variables=model.variables,
            states=aug_states,
            controls=model.controls,
            shocks=aug_shocks_tuple,
            steady_state=aug_ss,
            units=aug_units,
            solution=aug_solution,
            A=A_aug,
            B=B_aug,
            C=C_aug,
            method=model.method,
            residual_norm=model.residual_norm,
            timing="dynare",
        )
        object.__setattr__(m_aug, "_qz_criterium", qz_criterium)
        object.__setattr__(m_aug, "_sol_full", sol_full)
        return m_aug

    else:
        # Klein timing: model.A has shape (n_s + n_c, n_s + n_c)
        n_s = model.n_states
        n_c = model.n_controls
        n_u = len(model.shocks)
        n_pre = n_s + H
        n_tot = n_pre + n_c

        A_econ = np.zeros((n_s + n_c, n_tot))
        B_econ = np.zeros((n_s + n_c, n_tot))
        C_econ = np.zeros((n_s + n_c, n_u + H))

        A_econ[:, :n_s] = model.A[:, :n_s]
        A_econ[:, n_pre:] = model.A[:, n_s:]
        B_econ[:, :n_s] = model.B[:, :n_s]
        B_econ[:, n_s] = model.C[:, shock_idx]
        B_econ[:, n_pre:] = model.B[:, n_s:]
        C_econ[:, :n_u] = model.C[:, :n_u]

        A_V = np.zeros((H, n_tot))
        B_V = np.zeros((H, n_tot))
        C_V = np.zeros((H, n_u + H))
        A_V[:, n_s:n_pre] = np.eye(H)
        B_V[:, n_s:n_pre] = K_H
        C_V[:, n_u:] = np.eye(H)

        A_full = np.vstack([A_econ, A_V])
        B_full = np.vstack([B_econ, B_V])
        C_full = np.vstack([C_econ, C_V])

        sol_full = klein_solve(
            A_full, B_full, n_pre=n_pre, C=C_full, strict=True, div=qz_criterium
        )

        aug_states = tuple(model.states) + aux_states
        aug_shocks_tuple = tuple(model.shocks) + aux_shocks
        aug_ss = pd.concat([model.steady_state, pd.Series(0.0, index=list(aux_states))])
        aug_units = dict(model.units)
        for s in aux_states:
            aug_units[s] = "level"

        m_aug = LinearModel(
            variables=model.variables,
            states=aug_states,
            controls=model.controls,
            shocks=aug_shocks_tuple,
            steady_state=aug_ss,
            units=aug_units,
            solution=sol_full,
            A=A_full,
            B=B_full,
            C=C_full,
            method=model.method,
            residual_norm=model.residual_norm,
            timing="klein",
        )
        object.__setattr__(m_aug, "_qz_criterium", qz_criterium)
        return m_aug


def news_irf(
    model: LinearModel,
    shock: str,
    lead: int = 0,
    horizon: int = 40,
    size: float = 1.0,
) -> NewsIRFResult:
    """Compute impulse response function for surprise or news (anticipated) shock.

    Parameters
    ----------
    model : LinearModel
        Solved first-order DSGE model.
    shock : str
        Name of the structural shock innovation.
    lead : int, default 0
        Anticipation lead (0 for contemporaneous surprise shock).
        When lead = k > 0, an announcement arrives at t=0 about an innovation
        of magnitude `size` that realizes at date t=k.
    horizon : int, default 40
        Simulation horizon in periods after impact.
    size : float, default 1.0
        Magnitude of the structural shock.

    Returns
    -------
    NewsIRFResult
        Object containing news and surprise IRFs, metadata, and presentation methods.
    """
    from puremacro.dsge.build import ModelError

    if shock not in model.shocks:
        raise ModelError(
            f"no shock named {shock!r}; declared shocks: {list(model.shocks)}"
        )
    if lead < 0:
        raise ValueError(f"lead must be non-negative, got {lead}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    # Standard surprise IRF
    surprise_df = model.irf(shock, horizon=horizon, size=size)

    if lead == 0:
        return NewsIRFResult(
            irf=surprise_df.copy(),
            surprise_irf=surprise_df,
            shock=shock,
            lead=0,
            horizon=horizon,
            size=size,
            model=model,
        )

    # Lead k > 0: simulate augmented state space
    k = int(lead)
    m_aug = augment_news_state_space(model, shock=shock, max_lead=k)
    n_u = len(model.shocks)

    # Impulse on news innovation eta_{shock}^k at index n_u + k - 1
    total_shocks = len(m_aug.shocks)
    impulse = np.zeros(total_shocks)
    news_shock_col = n_u + k - 1
    impulse[news_shock_col] = size

    if model.timing == "dynare":
        sol_full = getattr(m_aug, "_sol_full", None)
        if sol_full is None:
            sol_full = m_aug.solution

        F_full = np.asarray(sol_full.F, dtype=float)
        L_full = np.asarray(sol_full.L, dtype=float)
        G_aug = np.asarray(m_aug.solution.G, dtype=float)
        N_aug = np.asarray(m_aug.solution.N, dtype=float)

        n_pre = len(m_aug.states)
        n_vars = len(model.variables)
        out = np.zeros((horizon + 1, n_vars))

        # At t=0: initial state is 0. Controls jump to L_full @ impulse
        x_t = np.zeros(n_pre)
        out[0] = F_full @ x_t + L_full @ impulse
        x_t = G_aug @ x_t + N_aug @ impulse

        for h in range(1, horizon + 1):
            out[h] = F_full @ x_t
            x_t = G_aug @ x_t

        irf_df = pd.DataFrame(out, columns=list(model.variables))
        irf_df.index.name = "h"

    else:
        # Klein timing: reported variables are [x_t[:n_s]; y_t]
        n_s = model.n_states
        n_c = model.n_controls
        n_pre = len(m_aug.states)
        G_aug, F_aug = m_aug.solution.G, m_aug.solution.F
        N_aug, L_aug = m_aug.solution.N, m_aug.solution.L

        out = np.zeros((horizon + 1, n_s + n_c))
        x_t = np.zeros(n_pre)
        y_0 = L_aug @ impulse
        out[0, :n_s] = x_t[:n_s]
        out[0, n_s:] = y_0
        x_t = G_aug @ x_t + N_aug @ impulse

        for h in range(1, horizon + 1):
            out[h, :n_s] = x_t[:n_s]
            out[h, n_s:] = F_aug @ x_t
            x_t = G_aug @ x_t

        df_raw = pd.DataFrame(out, columns=list(model.states) + list(model.controls))
        irf_df = df_raw[list(model.variables)].copy()
        irf_df.index.name = "h"

    return NewsIRFResult(
        irf=irf_df,
        surprise_irf=surprise_df,
        shock=shock,
        lead=k,
        horizon=horizon,
        size=size,
        model=model,
    )


def decompose_news(
    model: LinearModel,
    shock: str | None = None,
    horizon: int = 40,
    max_lead: int = 8,
) -> NewsDecompositionResult:
    """Compute forecast error variance decomposition separating surprise from news shocks.

    Decomposes the forecast error variance of all variables into contemporaneous
    surprise innovation eta^0 and news innovations eta^1, ..., eta^H.
    Variance shares strictly sum to 1.0 across surprise and all news leads.

    Parameters
    ----------
    model : LinearModel
        Solved first-order DSGE model.
    shock : str, optional
        Name of the structural shock process to decompose. Defaults to the first
        declared shock if None.
    horizon : int, default 40
        Forecast horizon for the variance decomposition.
    max_lead : int, default 8
        Maximum news lead H to decompose (H >= 1).

    Returns
    -------
    NewsDecompositionResult
        Object containing variance share tables, metadata, and presentation methods.
    """
    from puremacro.dsge.build import ModelError

    if shock is None:
        if not model.shocks:
            raise ModelError("Model has no structural shocks declared.")
        shock_name = model.shocks[0]
    else:
        shock_name = shock

    if shock_name not in model.shocks:
        raise ModelError(
            f"no shock named {shock_name!r}; declared shocks: {list(model.shocks)}"
        )
    if max_lead < 1:
        raise ValueError(f"max_lead must be at least 1, got {max_lead}")
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    H = int(max_lead)
    n_vars = len(model.variables)

    # 1. Compute unit IRF for surprise shock
    irf_surp = news_irf(model, shock=shock_name, lead=0, horizon=horizon, size=1.0).irf

    # 2. Compute unit IRFs for news leads 1..H
    irf_news: list[pd.DataFrame] = []
    for k in range(1, H + 1):
        irf_k = news_irf(model, shock=shock_name, lead=k, horizon=horizon, size=1.0).irf
        irf_news.append(irf_k)

    # 3. Dynamic cumulative variance profiles
    # V_{v, k}(h) = sum_{j=0}^h (Y_{v, k}(j))^2
    cum_surp = (irf_surp.to_numpy() ** 2).cumsum(axis=0)  # shape (horizon+1, n_vars)
    cum_news = [
        (irf.to_numpy() ** 2).cumsum(axis=0) for irf in irf_news
    ]  # H arrays of shape (horizon+1, n_vars)

    cols = ["surprise"] + [f"news_{k}" for k in range(1, H + 1)]
    var_shares_dict: dict[str, list[float]] = {c: [] for c in cols}

    dynamic_shares: dict[str, pd.DataFrame] = {}

    for v_idx, var in enumerate(model.variables):
        v_surp_prof = cum_surp[:, v_idx]
        v_news_profs = [cn[:, v_idx] for cn in cum_news]
        total_prof = v_surp_prof + sum(v_news_profs)

        # Dynamic shares over horizon
        share_matrix = np.zeros((horizon + 1, 1 + H))
        nonzero_mask = total_prof > 1e-16

        share_matrix[nonzero_mask, 0] = v_surp_prof[nonzero_mask] / total_prof[nonzero_mask]
        for k_idx in range(H):
            share_matrix[nonzero_mask, k_idx + 1] = (
                v_news_profs[k_idx][nonzero_mask] / total_prof[nonzero_mask]
            )

        # Inactive rows default to surprise = 1.0
        share_matrix[~nonzero_mask, 0] = 1.0

        # Normalise to guarantee exact sum = 1.0 to machine precision
        row_sums = share_matrix.sum(axis=1, keepdims=True)
        share_matrix = share_matrix / row_sums

        df_dyn = pd.DataFrame(share_matrix, columns=cols)
        df_dyn.index.name = "h"
        dynamic_shares[var] = df_dyn

        # Final horizon shares
        final_row = share_matrix[horizon]
        var_shares_dict["surprise"].append(float(final_row[0]))
        for k_idx in range(H):
            var_shares_dict[f"news_{k_idx + 1}"].append(float(final_row[k_idx + 1]))

    variance_shares = pd.DataFrame(var_shares_dict, index=list(model.variables))

    # Guarantee exact unit row sums
    row_sums = variance_shares.sum(axis=1)
    for col in variance_shares.columns:
        variance_shares[col] = variance_shares[col] / row_sums

    return NewsDecompositionResult(
        variance_shares=variance_shares,
        historical=None,
        shock=shock_name,
        horizon=horizon,
        max_lead=H,
        model=model,
        dynamic_shares=dynamic_shares,
    )


def plot_news_vs_surprise(
    model: LinearModel,
    shock: str,
    leads: tuple[int, ...] = (0, 2, 4, 8),
    variables: list[str] | None = None,
    horizon: int = 40,
    size: float = 1.0,
    *,
    figsize: tuple[float, float] | None = None,
    style: str = "default",
) -> tuple[Figure, Any]:
    """Plot multi-lead news responses compared to surprise shock.

    Overlays impulse response trajectories across different news leads against
    the contemporaneous surprise shock (lead=0). Headless Matplotlib Agg compatible.

    Parameters
    ----------
    model : LinearModel
        Solved DSGE model.
    shock : str
        Name of structural shock to plot.
    leads : tuple[int, ...], default (0, 2, 4, 8)
        Tuple of announcement leads to compare.
    variables : list[str], optional
        Variables to plot. Defaults to first 4-6 variables of the model.
    horizon : int, default 40
        Simulation horizon in periods.
    size : float, default 1.0
        Shock innovation magnitude.
    figsize : tuple[float, float], optional
        Figure dimensions.
    style : str, default 'default'
        Plot styling ('default', 'publication', 'grayscale').

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if variables is None:
        vars_to_plot = list(model.variables)[:6]
    else:
        vars_to_plot = [v for v in variables if v in model.variables]

    if not vars_to_plot:
        raise ValueError("No valid variables specified to plot.")

    # Compute IRFs across leads
    irfs_by_lead: dict[int, pd.DataFrame] = {}
    for lead in leads:
        res = news_irf(model, shock=shock, lead=lead, horizon=horizon, size=size)
        irfs_by_lead[lead] = res.irf

    n_vars = len(vars_to_plot)
    n_cols = 2 if n_vars > 1 else 1
    n_rows = int(np.ceil(n_vars / n_cols))
    calc_figsize = figsize or (5.5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=calc_figsize, squeeze=False)
    axes_array = axes.flatten()

    # Distinct line styles and colors
    lead_palette = ["#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    lead_styles = ["-", "--", "-.", ":", "-", "--"]

    for idx, var in enumerate(vars_to_plot):
        target_ax = axes_array[idx]
        h_grid = np.arange(horizon + 1)

        for l_idx, lead in enumerate(sorted(leads)):
            df = irfs_by_lead[lead]
            label = "Surprise (lead=0)" if lead == 0 else f"News (lead={lead})"
            color = lead_palette[l_idx % len(lead_palette)] if style != "grayscale" else str(0.15 + 0.15 * l_idx)
            ls = lead_styles[l_idx % len(lead_styles)]
            lw = 2.0 if lead == 0 else 1.5

            target_ax.plot(h_grid, df[var], label=label, color=color, linestyle=ls, linewidth=lw)

        target_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        target_ax.set_title(f"{var}", fontsize=11, fontweight="bold")
        target_ax.set_xlabel("Horizon (h)")
        target_ax.set_ylabel("Deviation")
        target_ax.grid(True, linestyle=":", alpha=0.5)
        target_ax.spines["top"].set_visible(False)
        target_ax.spines["right"].set_visible(False)
        target_ax.legend(loc="best", frameon=False, fontsize=8)

    for idx in range(n_vars, len(axes_array)):
        axes_array[idx].set_visible(False)

    fig.tight_layout()
    return fig, axes_array
