"""Granular Instrumental Variables (Gabaix & Koijen 2024, Econometrica).

Constructs granular instruments from entity-level shares and idiosyncratic
shocks after stripping unobserved common factors via Principal Component
Analysis (PCA) and applying inverse-variance precision weighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "GIVResult",
    "estimate_giv",
]


def _newey_west_var(z: np.ndarray, lags: int) -> float:
    """Compute Newey-West HAC variance of the sample sum/mean of 1D series z."""
    T = len(z)
    if T <= 1:
        return 0.0
    z_c = z - np.mean(z)
    gamma0 = float(np.dot(z_c, z_c) / T)
    v = gamma0
    max_l = min(max(0, int(lags)), T - 1)
    for l_idx in range(1, max_l + 1):
        w = 1.0 - l_idx / (max_l + 1.0)
        gamma_l = float(np.dot(z_c[l_idx:], z_c[:-l_idx]) / T)
        v += 2.0 * w * gamma_l
    return max(v, 1e-16)


@dataclass(frozen=True)
class GIVResult:
    """Result container for Granular Instrumental Variables (Gabaix & Koijen 2024).

    Implements the 7-method Result-Object Presentation Standard 2.0+.
    """

    multiplier: float
    se: float
    tstat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    ols_estimate: float
    first_stage_coef: float
    first_stage_f: float
    herfindahl: float
    effective_n_entities: float
    granular_instrument: np.ndarray
    idiosyncratic_shocks: np.ndarray
    precision_weights: np.ndarray
    entity_names: tuple[str, ...]
    n_periods: int
    n_entities: int
    n_factors: int
    overid_j_stat: float
    overid_pvalue: float
    leave_one_out: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ols_multiplier(self) -> float:
        """Alias for :attr:`ols_estimate`."""
        return float(self.ols_estimate)

    @property
    def overid_j_pvalue(self) -> float:
        """Alias for :attr:`overid_pvalue`."""
        return float(self.overid_pvalue)

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary table as a pandas DataFrame."""
        rows = [
            ("giv_multiplier", float(self.multiplier)),
            ("giv_se_hac", float(self.se)),
            ("giv_tstat", float(self.tstat)),
            ("giv_pvalue", float(self.pvalue)),
            ("ols_naive_estimate", float(self.ols_estimate)),
            ("first_stage_effective_f", float(self.first_stage_f)),
            ("herfindahl_index", float(self.herfindahl)),
            ("effective_n_entities", float(self.effective_n_entities)),
            ("overid_j_pvalue", float(self.overid_pvalue)),
            ("n_periods", float(self.n_periods)),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> str:
        """Return a formatted ASCII summary of GIV estimation metrics and diagnostics."""
        lines = [
            "==========================================================================",
            "Granular Instrumental Variables (Gabaix & Koijen 2024)",
            "==========================================================================",
            self.to_dataframe().to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "==========================================================================",
        ]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export summary table to GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe())

    def to_latex(self) -> str:
        """Export summary table to LaTeX tabular format."""
        return df_to_latex(self.to_dataframe())

    def to_typst(self) -> str:
        """Export summary table to Typst table format."""
        return df_to_typst(self.to_dataframe())

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        show: bool = False,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | matplotlib.axes.Axes:
        """Plot the leave-one-out giant sensitivity and granular instrument."""
        import matplotlib.pyplot as plt

        created_fig = False
        if ax is None:
            fig, ax_obj = plt.subplots(figsize=kwargs.get("figsize", (7.5, 4.2)))
            created_fig = True
        else:
            ax_obj = ax
            fig = ax_obj.figure  # type: ignore[assignment]

        loo = self.leave_one_out
        x_pos = np.arange(len(loo))
        ax_obj.bar(
            x_pos,
            loo["multiplier_loo"].to_numpy(dtype=float),
            color="#1f77b4",
            alpha=0.75,
            label="Leave-One-Out GIV",
        )
        ax_obj.axhline(
            self.multiplier,
            color="#d62728",
            lw=1.8,
            ls="--",
            label=f"Full-Sample GIV ({self.multiplier:.3f})",
        )
        ax_obj.set_xticks(x_pos)
        ax_obj.set_xticklabels(loo["dropped_entity"].astype(str).tolist(), rotation=30, ha="right")
        ax_obj.set_ylabel("Estimated Multiplier / Elasticity")
        ax_obj.set_title("Granular IV: Leave-One-Out Top-Entity Sensitivity")
        ax_obj.legend(loc="best", frameon=True)
        ax_obj.grid(True, alpha=0.25)

        if show:
            plt.show()
        return fig if created_fig else ax_obj


def estimate_giv(
    entity_outcomes: np.ndarray | pd.DataFrame,
    shares: np.ndarray | pd.DataFrame | Sequence[float],
    agg_endogenous: Sequence[float] | np.ndarray | pd.Series,
    agg_outcome: Sequence[float] | np.ndarray | pd.Series | None = None,
    *,
    endog_macro: Sequence[float] | np.ndarray | pd.Series | None = None,
    n_factors: int = 1,
    precision_weighted: bool = True,
    precision_weights: bool | None = None,
    hac_lags: int = 4,
    alpha: float = 0.05,
    top_k_loo: int = 5,
    leave_one_out_top_k: int | None = None,
    entity_names: Sequence[str] | None = None,
) -> GIVResult:
    """Estimate a structural multiplier or elasticity via Granular Instrumental Variables.

    Implements Gabaix & Koijen (2024, *Econometrica*):
    1. Demeans entity outcomes ``y_{i,t}`` across time and extracts ``n_factors``
       unobserved macro/common factors via PCA on entity innovations.
    2. Estimates idiosyncratic entity variances ``sigma_i^2`` and forms inverse-variance
       precision weights ``E_i = (1/sigma_i^2) / sum_j (1/sigma_j^2)`` (or equal
       weights ``1/N`` when ``precision_weighted=False``).
    3. Forms the granular instrument:
           z_t = sum_{i=1}^N (S_{i, t-1} - E_i) * u_tilde_{i, t}
       which is orthogonal to common shocks affecting all entities equally.
    4. Estimates the structural effect of ``agg_endogenous`` on ``agg_outcome``
       (defaulting to the share-weighted aggregate ``y_{S, t} = sum_i S_{i,t-1} y_{i,t}``
       when ``agg_outcome`` is None) via 2SLS using ``z_t`` as instrument, with
       Newey-West HAC standard errors, first-stage effective F, Sargan/Hansen
       over-identification test (using top-entity sub-instruments), and
       leave-one-out top-entity sensitivity analysis.

    Parameters
    ----------
    entity_outcomes : np.ndarray | pd.DataFrame
        Array or DataFrame of shape ``(T, N)`` with entity-level growth rates,
        flows, or shocks ``y_{i, t}``.
    shares : np.ndarray | pd.DataFrame | Sequence[float]
        Size shares ``S_i`` of shape ``(N,)`` (time-invariant) or ``(T, N)``
        (pre-period lagged shares ``S_{i, t-1}``). Rows are normalized to sum to 1.
    agg_endogenous : Sequence[float] | np.ndarray | pd.Series
        1D array of length ``T`` representing the endogenous aggregate regressor
        ``X_t`` (e.g., aggregate price, interest rate, or share-weighted shock).
    agg_outcome : Sequence[float] | np.ndarray | pd.Series, optional
        1D array of length ``T`` representing the aggregate outcome ``Y_t`` to
        explain. If ``None``, uses the equal-weighted mean ``y_{E, t}`` when
        ``agg_endogenous`` is the share-weighted aggregate, or ``sum_i S_{i,t} y_{i,t}``.
    n_factors : int, default 1
        Number of latent common factors stripped via PCA from entity residuals.
    precision_weighted : bool, default True
        If True, weights the benchmark subtraction by inverse idiosyncratic variance
        ``1 / sigma_i^2`` (optimal GIV under heteroskedastic entities).
    hac_lags : int, default 4
        Newey-West truncation lag for HAC standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    top_k_loo : int, default 5
        Number of largest entities to drop one-by-one in the leave-one-out check.
    entity_names : Sequence[str], optional
        Labels for the ``N`` entities.

    Returns
    -------
    GIVResult
        Container with GIV estimate, HAC standard error, first-stage F,
        Herfindahl index, over-identification test, and leave-one-out table.
    """
    if isinstance(entity_outcomes, pd.DataFrame):
        if entity_names is None:
            entity_names = [str(c) for c in entity_outcomes.columns]
        Y_mat = entity_outcomes.to_numpy(dtype=float)
    else:
        Y_mat = np.asarray(entity_outcomes, dtype=float)

    if Y_mat.ndim != 2:
        raise ValueError(f"entity_outcomes must be 2D (T, N), got shape {Y_mat.shape}")
    T, N = Y_mat.shape
    if T < 6 or N < 3:
        raise ValueError(f"estimate_giv requires T >= 6 and N >= 3, got T={T}, N={N}")
    if not np.all(np.isfinite(Y_mat)):
        raise ValueError("entity_outcomes must contain finite values")

    if entity_names is None:
        names = tuple(f"entity_{i}" for i in range(N))
    else:
        if len(entity_names) != N:
            raise ValueError(f"entity_names length ({len(entity_names)}) must match N ({N})")
        names = tuple(str(x) for x in entity_names)

    S_raw = np.asarray(shares, dtype=float)
    if S_raw.ndim == 1:
        if S_raw.shape[0] != N:
            raise ValueError(f"1D shares must have length N={N}, got {S_raw.shape}")
        if np.any(S_raw < 0) or np.sum(S_raw) <= 0:
            raise ValueError("shares must be non-negative with positive sum")
        S_vec = S_raw / np.sum(S_raw)
        S_mat = np.tile(S_vec, (T, 1))
    elif S_raw.ndim == 2:
        if S_raw.shape != (T, N):
            raise ValueError(f"2D shares must have shape (T={T}, N={N}), got {S_raw.shape}")
        row_sums = np.sum(S_raw, axis=1, keepdims=True)
        if np.any(S_raw < 0) or np.any(row_sums <= 0):
            raise ValueError("shares must be non-negative with positive row sums")
        S_mat = S_raw / row_sums
        S_vec = np.mean(S_mat, axis=0)
    else:
        raise ValueError("shares must be 1D (N,) or 2D (T, N)")

    if precision_weights is not None:
        precision_weighted = bool(precision_weights)
    if leave_one_out_top_k is not None:
        top_k_loo = int(leave_one_out_top_k)

    if endog_macro is not None:
        y_agg = np.asarray(agg_endogenous, dtype=float).reshape(-1)
        x_agg = np.asarray(endog_macro, dtype=float).reshape(-1)
    elif agg_outcome is not None:
        x_agg = np.asarray(agg_endogenous, dtype=float).reshape(-1)
        y_agg = np.asarray(agg_outcome, dtype=float).reshape(-1)
    else:
        y_agg = np.asarray(agg_endogenous, dtype=float).reshape(-1)
        x_agg = np.sum(S_mat * Y_mat, axis=1)

    if x_agg.shape[0] != T or not np.all(np.isfinite(x_agg)):
        raise ValueError(f"endogenous macro series must be a finite 1D array of length T={T}")
    if y_agg.shape[0] != T or not np.all(np.isfinite(y_agg)):
        raise ValueError(f"aggregate outcome series must be a finite 1D array of length T={T}")

    if not isinstance(n_factors, (int, np.integer)) or n_factors < 0 or n_factors >= min(T - 1, N - 1):
        raise ValueError(f"n_factors must be in [0, min(T-2, N-2)], got {n_factors}")

    def _extract_idiosyncratic(
        Y_sub: np.ndarray, S_sub: np.ndarray, k_fac: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        # Demean each entity across time; extract top k_fac common factors BEFORE cross-sectional demeaning
        # so that common factors with positive mean loadings are captured by the leading singular vectors.
        Y_c = Y_sub - np.mean(Y_sub, axis=0, keepdims=True)
        F_hat: np.ndarray | None = None
        if k_fac > 0:
            u_svd, s_svd, vt_svd = np.linalg.svd(Y_c, full_matrices=False)
            F_hat = u_svd[:, :k_fac]
            recon = (F_hat * s_svd[:k_fac]) @ vt_svd[:k_fac, :]
            u_hat = Y_c - recon
        else:
            u_hat = Y_c - np.mean(Y_c, axis=1, keepdims=True)

        if precision_weighted:
            var_i = np.var(u_hat, axis=0, ddof=1)
            inv_var = 1.0 / np.maximum(var_i, 1e-10)
            E_w = inv_var / np.sum(inv_var)
        else:
            E_w = np.full(Y_sub.shape[1], 1.0 / Y_sub.shape[1], dtype=float)

        # Ensure exact cross-sectional demeaning under precision weights E_w
        u_tilde = u_hat - (u_hat @ E_w)[:, None]
        z_vec = np.sum((S_sub - E_w[None, :]) * u_tilde, axis=1)
        return z_vec, u_tilde, E_w, F_hat

    z_t, u_tilde, E_weights, F_hat = _extract_idiosyncratic(Y_mat, S_mat, int(n_factors))

    # Demean aggregate outcome, endogenous regressor, and granular instrument
    x_raw_c = x_agg - np.mean(x_agg)
    y_raw_c = y_agg - np.mean(y_agg)
    z_c = z_t - np.mean(z_t)

    # Frisch-Waugh-Lovell projection orthogonal to extracted common factors F_hat (Gabaix & Koijen 2024)
    if F_hat is not None and F_hat.shape[1] > 0:
        F_c = F_hat - np.mean(F_hat, axis=0, keepdims=True)
        Q_f, _ = np.linalg.qr(F_c, mode="reduced")
        x_c = x_raw_c - Q_f @ (Q_f.T @ x_raw_c)
        y_c = y_raw_c - Q_f @ (Q_f.T @ y_raw_c)
        z_c = z_c - Q_f @ (Q_f.T @ z_c)
    else:
        x_c = x_raw_c
        y_c = y_raw_c

    denom_iv = float(np.dot(z_c, x_c))
    if abs(denom_iv) < 1e-14:
        raise ValueError("Granular instrument has zero covariance with agg_endogenous (weak/collinear instrument)")

    beta_giv = float(np.dot(z_c, y_c) / denom_iv)
    resid_iv = y_c - beta_giv * x_c

    # HAC standard error of 2SLS / IV estimator: Var(beta_hat) = Var(sum_t z_t * e_t / T) / (Cov(z, x))^2 / T
    ze_series = z_c * resid_iv
    omega_ze = _newey_west_var(ze_series, hac_lags)
    cov_zx = denom_iv / T
    se_giv = float(np.sqrt(omega_ze / (T * (cov_zx ** 2))))
    tstat = float(beta_giv / se_giv) if se_giv > 0 else float("nan")
    pvalue = float(2.0 * norm.sf(abs(tstat))) if np.isfinite(tstat) else float("nan")
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower = float(beta_giv - z_crit * se_giv)
    ci_upper = float(beta_giv + z_crit * se_giv)

    # Naive OLS benchmark (without common-factor stripping or GIV)
    denom_ols = float(np.dot(x_raw_c, x_raw_c))
    beta_ols = float(np.dot(x_raw_c, y_raw_c) / denom_ols) if denom_ols > 1e-14 else float("nan")

    # First-stage effective F-statistic (Olea-Pflueger / HAC F)
    denom_zz = float(np.dot(z_c, z_c))
    pi_fs = float(np.dot(z_c, x_c) / denom_zz)
    v_fs = x_c - pi_fs * z_c
    omega_zv = _newey_west_var(z_c * v_fs, hac_lags)
    se_pi = float(np.sqrt(omega_zv / (T * ((denom_zz / T) ** 2))))
    first_stage_f = float((pi_fs / se_pi) ** 2) if se_pi > 0 else 0.0

    # Herfindahl index and effective number of entities
    h_index = float(np.sqrt(np.sum(S_vec ** 2)))
    eff_n = float(1.0 / np.sum(S_vec ** 2))

    # Over-identification Sargan/Hansen J-test splitting granular instrument into top entity vs rest
    top_idx = int(np.argmax(S_vec))
    z1 = (S_mat[:, top_idx] - E_weights[top_idx]) * u_tilde[:, top_idx]
    z2 = z_t - z1
    Z_two = np.column_stack([z1 - np.mean(z1), z2 - np.mean(z2)])
    try:
        Pz_x = Z_two @ np.linalg.lstsq(Z_two, x_c, rcond=None)[0]
        beta_2sls_overid = float(np.dot(Pz_x, y_c) / max(float(np.dot(Pz_x, x_c)), 1e-14))
        u_overid = y_c - beta_2sls_overid * x_c
        g_bar = (Z_two.T @ u_overid) / np.sqrt(T)
        S_mom = (Z_two.T * (u_overid ** 2)) @ Z_two / T
        j_stat = float(g_bar.T @ np.linalg.pinv(S_mom) @ g_bar)
        j_pval = float(chi2.sf(j_stat, df=1))
    except np.linalg.LinAlgError:
        j_stat, j_pval = float("nan"), float("nan")

    # Leave-one-out sensitivity dropping each of the top_k largest entities
    order = np.argsort(S_vec)[::-1]
    k_loo = min(max(1, int(top_k_loo)), N - 2)
    loo_rows: list[dict[str, Any]] = []
    for idx_drop in order[:k_loo]:
        keep = [j for j in range(N) if j != int(idx_drop)]
        S_keep = S_mat[:, keep]
        S_keep = S_keep / np.sum(S_keep, axis=1, keepdims=True)
        z_loo, _, _, _ = _extract_idiosyncratic(Y_mat[:, keep], S_keep, min(int(n_factors), len(keep) - 2))
        z_loo_c = z_loo - np.mean(z_loo)
        d_loo = float(np.dot(z_loo_c, x_c))
        b_loo = float(np.dot(z_loo_c, y_c) / d_loo) if abs(d_loo) > 1e-14 else float("nan")
        loo_rows.append(
            {
                "dropped_entity": names[int(idx_drop)],
                "entity_share": float(S_vec[int(idx_drop)]),
                "multiplier_loo": b_loo,
                "diff_from_full": float(b_loo - beta_giv) if np.isfinite(b_loo) else float("nan"),
            }
        )
    loo_df = pd.DataFrame(loo_rows)

    return GIVResult(
        multiplier=beta_giv,
        se=se_giv,
        tstat=tstat,
        pvalue=pvalue,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ols_estimate=beta_ols,
        first_stage_coef=pi_fs,
        first_stage_f=first_stage_f,
        herfindahl=h_index,
        effective_n_entities=eff_n,
        granular_instrument=z_t,
        idiosyncratic_shocks=u_tilde,
        precision_weights=E_weights,
        entity_names=names,
        n_periods=T,
        n_entities=N,
        n_factors=int(n_factors),
        overid_j_stat=j_stat,
        overid_pvalue=j_pval,
        leave_one_out=loo_df,
        metadata={
            "precision_weighted": precision_weighted,
            "hac_lags": hac_lags,
            "alpha": alpha,
        },
    )
