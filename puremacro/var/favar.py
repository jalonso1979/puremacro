"""Factor-Augmented Vector Autoregression (FAVAR).

Implements the two-step FAVAR framework of Bernanke, Boivin & Eliasz (2005, *QJE*):
- Static Principal Component factor extraction from high-dimensional informational panels X_t.
- Orthogonalization of latent factors F_t against observed policy instruments Y_t.
- Joint VAR estimation on state vector Z_t = [Y_t', F_t']'.
- Full cross-sectional impulse response mapping: IRF_{X_i}(h) = Lambda_i * IRF_Z(h).
- Residual bootstrap confidence intervals for all N macroeconomic series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf
from puremacro.var.identify.cholesky import safe_cholesky


@dataclass(frozen=True)
class FAVARResult:
    """Results from Factor-Augmented VAR estimation.

    Attributes
    ----------
    irf_panel : pd.DataFrame
        Impulse responses for all N variables across horizons (H+1, N).
    irf_lower_panel : pd.DataFrame
        Lower confidence bounds for panel IRFs (H+1, N).
    irf_upper_panel : pd.DataFrame
        Upper confidence bounds for panel IRFs (H+1, N).
    irf_policy : np.ndarray
        Response of the observed policy instrument (H+1,).
    irf_factors : np.ndarray
        Response of the K latent factors (H+1, K).
    factors : np.ndarray
        Estimated latent factors F_t (T, K).
    loadings : np.ndarray
        Factor loadings matrix Lambda (N, M + K).
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by each principal component (K,).
    variable_names : list of str
        Names of the N informational panel variables.
    horizon : int
        IRF horizon H.
    """
    irf_panel: pd.DataFrame
    irf_lower_panel: pd.DataFrame
    irf_upper_panel: pd.DataFrame
    irf_policy: np.ndarray
    irf_factors: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    explained_variance_ratio: np.ndarray
    variable_names: list[str]
    horizon: int

    def summary(self) -> str:
        T, K = self.factors.shape
        N = len(self.variable_names)
        tot_var = float(np.sum(self.explained_variance_ratio)) * 100.0
        lines = [
            "Factor-Augmented VAR (FAVAR) — Bernanke, Boivin & Eliasz (2005)",
            "=" * 72,
            f"Panel Dimensions                : T = {T} periods, N = {N} series",
            f"Latent Factors Extracted (K)    : {K}",
            f"Total Panel Variance Explained  : {tot_var:.2f}%",
            f"IRF Horizon                     : {self.horizon} periods",
            "-" * 72,
            "Variance Explained per Factor:",
        ]
        for k_i, var_share in enumerate(self.explained_variance_ratio, 1):
            lines.append(f"  Factor {k_i:<2d}                       : {var_share*100:.2f}%")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return full panel impulse responses as a DataFrame."""
        return self.irf_panel.copy()

    def to_markdown(self, **kwargs) -> str:
        """Export panel IRFs to Markdown table."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export panel IRFs to LaTeX table."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export panel IRFs to Typst table."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        *,
        figsize: tuple[float, float] | None = None,
        ax: Any = None,
    ):
        """Plot impulse responses for selected panel variables with confidence bands."""
        import matplotlib.pyplot as plt

        vars_to_plot = list(variables) if variables is not None else self.variable_names[:min(4, len(self.variable_names))]
        n_vars = len(vars_to_plot)

        if ax is None:
            n_cols = min(2, n_vars) if n_vars > 0 else 1
            n_rows = int(np.ceil(n_vars / n_cols)) if n_vars > 0 else 1
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=figsize or (5.5 * n_cols, 3.5 * n_rows),
                squeeze=False,
            )
            axes_flat = axes.ravel()
        else:
            axes_flat = [ax] if not hasattr(ax, "__len__") else ax
            fig = axes_flat[0].get_figure()

        h_idx = range(self.horizon + 1)
        for i, var in enumerate(vars_to_plot):
            if i >= len(axes_flat):
                break
            a = axes_flat[i]
            if var in self.irf_panel.columns:
                pt = self.irf_panel[var].values
                lo = self.irf_lower_panel[var].values
                hi = self.irf_upper_panel[var].values

                a.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                a.plot(h_idx, pt, color="#1f77b4", linewidth=1.5, label="IRF")
                a.fill_between(h_idx, lo, hi, color="#1f77b4", alpha=0.2, label="CI")
                a.set_title(var)
                a.set_xlabel("Horizon")
                a.set_ylabel("Response")
                a.grid(True, linestyle=":", alpha=0.5)

        if ax is None:
            for j in range(n_vars, len(axes_flat)):
                axes_flat[j].set_visible(False)
            fig.tight_layout()
            return fig
        return ax


def favar(
    panel_data: pd.DataFrame | np.ndarray,
    policy_var: pd.Series | np.ndarray,
    *,
    n_factors: int = 3,
    p: int = 2,
    horizon: int = 20,
    n_boot: int = 200,
    ci: float = 0.90,
    seed: int = 42,
) -> FAVARResult:
    """Estimate Factor-Augmented VAR and compute impulse responses for the full panel.

    Parameters
    ----------
    panel_data : DataFrame or ndarray of shape (T, N)
        High-dimensional informational panel of macroeconomic time series.
    policy_var : Series or ndarray of shape (T,)
        Observable policy instrument series (e.g. Federal Funds Rate or Policy Rate).
    n_factors : int, default 3
        Number of latent unobserved factors K to extract.
    p : int, default 2
        Lag order for the joint VAR on [Y_t, F_t].
    horizon : int, default 20
        Impulse response horizon H.
    n_boot : int, default 200
        Number of residual bootstrap replications for confidence bands.
    ci : float, default 0.90
        Coverage level for pointwise confidence intervals.
    seed : int, default 42
        Random seed for bootstrap replications.

    Returns
    -------
    FAVARResult
    """
    rng = np.random.default_rng(seed)
    
    # 1. Format Inputs
    if isinstance(panel_data, pd.DataFrame):
        var_names = list(panel_data.columns)
        X = panel_data.to_numpy(dtype=float)
    else:
        X = np.asarray(panel_data, dtype=float)
        var_names = [f"Var_{i+1}" for i in range(X.shape[1])]

    T, N = X.shape
    if isinstance(policy_var, (pd.Series, pd.DataFrame)):
        Y = policy_var.to_numpy(dtype=float).ravel()
    else:
        Y = np.asarray(policy_var, dtype=float).ravel()

    if len(Y) != T:
        raise ValueError(f"favar: length of policy_var ({len(Y)}) does not match panel_data T ({T})")

    # 2. Standardize panel
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0.0] = 1.0
    X_norm = (X - X_mean) / X_std

    # 3. Factor Extraction via SVD / PCA
    # X_norm = U @ S @ V.T
    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
    var_explained = (S ** 2) / np.sum(S ** 2)
    
    # Extract K principal components
    F_hat = U[:, :n_factors] * np.sqrt(T)
    var_shares = var_explained[:n_factors]

    # Orthogonalize factors against observed policy variable Y:
    # F_tilde = F_hat - Y * (Y'Y)^(-1) Y' F_hat
    Y_col = Y[:, None]
    Y_proj = Y_col @ (np.linalg.inv(Y_col.T @ Y_col) @ (Y_col.T @ F_hat))
    F_ortho = F_hat - Y_proj

    # 4. Joint State Vector Z_t = [Y_t, F_t] of dimension M + K = 1 + n_factors
    Z = np.column_stack([Y, F_ortho])
    m_k = Z.shape[1]

    # Regress X_norm on Z to get factor loadings: X_norm = Z @ Lambda.T + e
    Lambda = np.linalg.lstsq(Z, X_norm, rcond=None)[0].T  # (N, 1 + K)

    # 5. Estimate VAR on Z_t
    A_list, c, Sigma, resid, X_design = estimate_var(Z, p=p)
    P = safe_cholesky(Sigma, name="favar")
    
    # Policy shock is shock 0 in recursive Cholesky
    B_impact = P
    irf_Z = irf(A_list, B_impact, horizon=horizon)  # (H+1, m_k, m_k)
    
    # Impulse response to policy shock (shock 0):
    irf_policy_shock = irf_Z[:, :, 0]  # (H+1, m_k)
    irf_Y = irf_policy_shock[:, 0]
    irf_F = irf_policy_shock[:, 1:]

    # 6. Map IRFs to full panel in original units:
    # IRF_X = (irf_policy_shock @ Lambda.T) * X_std
    irf_panel_norm = irf_policy_shock @ Lambda.T  # (H+1, N)
    irf_panel_orig = irf_panel_norm * X_std

    df_irf = pd.DataFrame(irf_panel_orig, index=range(horizon + 1), columns=var_names)

    # 7. Residual Bootstrap Inference
    boot_irfs = np.zeros((n_boot, horizon + 1, N))
    T_eff = resid.shape[0]

    for b in range(n_boot):
        # Resample residuals
        resample_idx = rng.choice(T_eff, size=T_eff, replace=True)
        e_b = resid[resample_idx]
        
        # Reconstruct bootstrap series Z_b
        Z_b = np.zeros_like(Z)
        Z_b[:p] = Z[:p]
        for t in range(p, T):
            lag_terms = sum(A_list[l] @ Z_b[t - 1 - l] for l in range(p))
            Z_b[t] = c + lag_terms + e_b[t - p]

        try:
            A_b, c_b, Sig_b, _, _ = estimate_var(Z_b, p=p)
            P_b = safe_cholesky(Sig_b, name="favar_boot")
            irf_Zb = irf(A_b, P_b, horizon=horizon)
            irf_Xb = (irf_Zb[:, :, 0] @ Lambda.T) * X_std
            boot_irfs[b] = irf_Xb
        except Exception:
            boot_irfs[b] = irf_panel_orig

    alpha_lo = (1.0 - ci) / 2.0
    alpha_hi = 1.0 - alpha_lo
    lo_vals = np.quantile(boot_irfs, alpha_lo, axis=0)
    hi_vals = np.quantile(boot_irfs, alpha_hi, axis=0)

    df_lo = pd.DataFrame(lo_vals, index=range(horizon + 1), columns=var_names)
    df_hi = pd.DataFrame(hi_vals, index=range(horizon + 1), columns=var_names)

    return FAVARResult(
        irf_panel=df_irf,
        irf_lower_panel=df_lo,
        irf_upper_panel=df_hi,
        irf_policy=irf_Y,
        irf_factors=irf_F,
        factors=F_ortho,
        loadings=Lambda,
        explained_variance_ratio=var_shares,
        variable_names=var_names,
        horizon=horizon,
    )


__all__ = [
    "FAVARResult",
    "favar",
]
