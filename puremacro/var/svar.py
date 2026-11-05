"""First-Class Structural VAR (SVAR) Identification Suite.

Provides a unified, Pyodide-compatible structural VAR estimation and inference
interface returning :class:`SVARResult` containers conforming to the PureMacro
Result-Object Presentation Standard 2.0+:

- :func:`svar_heteroskedastic`: Rigobon (2003) / Lanne & Lütkepohl (2008)
  identification through volatility regimes via simultaneous diagonalization
  with pairwise Wald tests for distinct relative variances.
- :func:`svar_non_gaussian`: Lanne, Meitz & Saikkonen (2017) / Keweloh (2021)
  identification via higher-moment co-skewness and co-kurtosis GMM or FastICA.
- :func:`svar_proxy`: Mertens & Ravn (2013) / Stock & Watson (2018) external
  instrument (Proxy-SVAR) identification with Olea-Pflueger effective F,
  wild bootstrap bands, and weak-IV Anderson-Rubin impact confidence sets.
- :func:`svar_sign_narrative`: Rubio-Ramírez, Waggoner & Zha (2010) Haar-QR
  sign restrictions combined with Antolín-Díaz & Rubio-Ramírez (2018) narrative
  restrictions and Giacomini & Kitagawa (2021, *Econometrica*) prior-robust
  identified-set bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import chi2

from puremacro._linalg import safe_cholesky
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.var.estimate import estimate_var
from puremacro.var.irf import fevd as compute_fevd
from puremacro.var.irf import irf as compute_irf

__all__ = [
    "SVARResult",
    "svar_heteroskedastic",
    "svar_non_gaussian",
    "svar_proxy",
    "svar_sign_narrative",
]


@dataclass(frozen=True)
class SVARResult:
    """Unified Structural VAR result container (Presentation Standard 2.0+).

    Attributes
    ----------
    B0 : np.ndarray
        Structural impact matrix of shape ``(n, n)``, where column ``j`` is the
        contemporaneous impact of structural shock ``j``.
    A_list : list of np.ndarray
        Reduced-form autoregressive coefficient matrices ``[A_1, ..., A_p]``,
        each of shape ``(n, n)``.
    Sigma : np.ndarray
        Reduced-form residual covariance matrix of shape ``(n, n)``.
    irf : np.ndarray
        Structural impulse response functions of shape ``(horizon + 1, n, n)``,
        where ``irf[h, i, j]`` is the response of variable ``i`` at horizon ``h``
        to structural shock ``j``.
    fevd : np.ndarray
        Forecast-error variance decomposition of shape ``(horizon + 1, n, n)``.
    structural_shocks : np.ndarray
        Identified structural shocks ``eps_t = (B0^{-1} u_t')'`` of shape ``(T_eff, n)``.
    method : str
        Identification scheme identifier (``"heteroskedastic"``, ``"non_gaussian"``,
        ``"proxy"``, or ``"sign_narrative"``).
    horizon : int
        Maximum impulse response horizon ``H``.
    var_names : tuple of str
        Names of the ``n`` endogenous variables.
    ci_lower : np.ndarray or None
        Lower pointwise confidence/credible band of shape ``(horizon + 1, n, n)``.
    ci_upper : np.ndarray or None
        Upper pointwise confidence/credible band of shape ``(horizon + 1, n, n)``.
    gk_lower : np.ndarray or None
        Giacomini & Kitagawa (2021) prior-robust identified-set lower bound of
        shape ``(horizon + 1, n, n)``.
    gk_upper : np.ndarray or None
        Giacomini & Kitagawa (2021) prior-robust identified-set upper bound of
        shape ``(horizon + 1, n, n)``.
    diagnostics : dict
        Identification-specific test statistics and diagnostics.
    metadata : dict
        Estimation metadata (sample size, lag order, settings).
    """

    B0: np.ndarray
    A_list: list[np.ndarray]
    Sigma: np.ndarray
    irf: np.ndarray
    fevd: np.ndarray
    structural_shocks: np.ndarray
    method: str
    horizon: int
    var_names: tuple[str, ...]
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    gk_lower: np.ndarray | None = None
    gk_upper: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of structural impact and horizon metrics."""
        n = len(self.var_names)
        rows: list[dict[str, Any]] = []
        for j in range(n):
            shock_name = f"shock_{self.var_names[j]}"
            for i in range(n):
                resp_name = self.var_names[i]
                irf_path = self.irf[:, i, j]
                peak_idx = int(np.argmax(np.abs(irf_path)))
                row: dict[str, Any] = {
                    "method": self.method,
                    "response": resp_name,
                    "shock": shock_name,
                    "impact_B0": float(self.B0[i, j]),
                    "peak_response": float(irf_path[peak_idx]),
                    "peak_horizon": peak_idx,
                    "terminal_irf": float(irf_path[-1]),
                    "terminal_fevd": float(self.fevd[-1, i, j]),
                }
                if self.ci_lower is not None and self.ci_upper is not None:
                    row["impact_ci_lower"] = float(self.ci_lower[0, i, j])
                    row["impact_ci_upper"] = float(self.ci_upper[0, i, j])
                if self.gk_lower is not None and self.gk_upper is not None:
                    row["impact_gk_lower"] = float(self.gk_lower[0, i, j])
                    row["impact_gk_upper"] = float(self.gk_upper[0, i, j])
                rows.append(row)
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame of impulse responses across horizons."""
        n = len(self.var_names)
        H_plus = self.irf.shape[0]
        data: dict[str, Any] = {"horizon": np.arange(H_plus, dtype=int)}
        for j in range(n):
            s_name = self.var_names[j]
            for i in range(n):
                r_name = self.var_names[i]
                col = f"{r_name}<-{s_name}"
                data[col] = self.irf[:, i, j]
                if self.ci_lower is not None and self.ci_upper is not None:
                    data[f"{col}_ci_lower"] = self.ci_lower[:, i, j]
                    data[f"{col}_ci_upper"] = self.ci_upper[:, i, j]
                if self.gk_lower is not None and self.gk_upper is not None:
                    data[f"{col}_gk_lower"] = self.gk_lower[:, i, j]
                    data[f"{col}_gk_upper"] = self.gk_upper[:, i, j]
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the impulse response table as GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the impulse response table as a LaTeX tabular environment."""
        return df_to_latex(self.to_dataframe(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the impulse response table as a Typst ``#table``."""
        return df_to_typst(self.to_dataframe(), index=False, **kwargs)

    def plot(self, ax: Any = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot structural impulse responses with confidence and identified-set bands."""
        import matplotlib.pyplot as plt

        n = len(self.var_names)
        horizons = np.arange(self.horizon + 1)
        figsize = kwargs.get("figsize", (3.6 * n, 2.8 * n))

        if ax is None:
            fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
        else:
            if isinstance(ax, np.ndarray):
                axes = ax.reshape(n, n) if ax.size == n * n else np.atleast_2d(ax)
                fig = axes.flat[0].figure
            elif isinstance(ax, (list, tuple)):
                axes = np.asarray(ax, dtype=object).reshape(n, n)
                fig = axes.flat[0].figure
            else:
                axes = np.array([[ax]], dtype=object)
                fig = ax.figure

        for i in range(min(n, axes.shape[0])):
            for j in range(min(n, axes.shape[1])):
                panel = axes[i, j]
                panel.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
                if self.gk_lower is not None and self.gk_upper is not None:
                    panel.fill_between(
                        horizons,
                        self.gk_lower[:, i, j],
                        self.gk_upper[:, i, j],
                        color="#ff7f0e",
                        alpha=0.20,
                        label="GK bounds",
                    )
                if self.ci_lower is not None and self.ci_upper is not None:
                    panel.fill_between(
                        horizons,
                        self.ci_lower[:, i, j],
                        self.ci_upper[:, i, j],
                        color="#1f77b4",
                        alpha=0.25,
                        label="CI",
                    )
                panel.plot(
                    horizons,
                    self.irf[:, i, j],
                    color="#1f77b4",
                    linewidth=1.8,
                    label="IRF",
                )
                panel.set_title(f"{self.var_names[i]} <- {self.var_names[j]}", fontsize=10)
                if i == n - 1:
                    panel.set_xlabel("Horizon")
                panel.grid(True, linestyle=":", alpha=0.4)

        if ax is None:
            fig.tight_layout()
        if show:
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_var_inputs(
    Y: np.ndarray | pd.DataFrame,
    p: int,
    var_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Validate Y, extract variable names, and run reduced-form VAR(p)."""
    if p < 1:
        raise ValueError(f"Lag order p must be >= 1; got {p}.")
    if isinstance(Y, pd.DataFrame):
        inferred_names = tuple(str(c) for c in Y.columns)
        Y_arr = Y.to_numpy(dtype=float)
    else:
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim != 2:
            raise ValueError(f"Y must be a 2D array of shape (T, n); got shape {Y_arr.shape}.")
        inferred_names = tuple(f"y{i}" for i in range(Y_arr.shape[1]))

    T, n = Y_arr.shape
    if var_names is not None:
        names = tuple(str(v) for v in var_names)
        if len(names) != n:
            raise ValueError(f"len(var_names)={len(names)} does not match n={n} columns of Y.")
    else:
        names = inferred_names

    est = estimate_var(Y_arr, p=p)
    A_list = list(est.A_list)
    c = np.asarray(est.c, dtype=float)
    Sigma = np.asarray(est.Sigma, dtype=float)
    resid = np.asarray(est.resid, dtype=float)
    return Y_arr, names, A_list, c, Sigma, resid


def _canonicalize_b0_columns(
    B0: np.ndarray,
    aux_arrays: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Permute and sign-normalize columns of B0 for diagonal dominance and B0[j, j] > 0.

    Uses the Hungarian algorithm on column-normalized ``|B0|`` so each column ``j``
    is matched to variable ``j``, and flips column signs so ``B0[j, j] >= 0``.
    """
    B = np.asarray(B0, dtype=float).copy()
    col_norms = np.linalg.norm(B, axis=0, keepdims=True)
    col_norms = np.where(col_norms > 1e-15, col_norms, 1.0)
    score = np.abs(B) / col_norms
    _, col_ind = linear_sum_assignment(-score)
    B = B[:, col_ind]

    signs = np.ones(B.shape[1], dtype=float)
    for j in range(B.shape[1]):
        if B[j, j] < 0.0:
            signs[j] = -1.0
    B = B * signs[None, :]

    out_aux: list[np.ndarray] = []
    if aux_arrays is not None:
        for arr in aux_arrays:
            a = np.asarray(arr).copy()
            if a.ndim == 1:
                out_aux.append(a[col_ind])
            elif a.ndim == 2:
                out_aux.append((a[:, col_ind]) * signs[None, :])
            else:
                out_aux.append(a)
    return B, col_ind, out_aux


def _compute_structural_shocks(resid: np.ndarray, B0: np.ndarray) -> np.ndarray:
    """Compute structural shocks eps_t = (B0^{-1} u_t')' of shape (T_eff, n)."""
    try:
        return np.linalg.solve(B0, resid.T).T
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(B0) @ resid.T).T


# ---------------------------------------------------------------------------
# 1. Identification via Heteroskedasticity (Rigobon 2003 / Lanne-Lütkepohl 2008)
# ---------------------------------------------------------------------------


def svar_heteroskedastic(
    Y: np.ndarray | pd.DataFrame,
    p: int = 4,
    *,
    regime_indicator: Sequence[int | bool] | np.ndarray | int,
    horizon: int = 20,
    var_names: Sequence[str] | None = None,
) -> SVARResult:
    """Identify an SVAR via volatility regimes (Rigobon 2003; Lanne & Lütkepohl 2008).

    Simultaneously diagonalizes the regime-conditional reduced-form residual
    covariance matrices :math:`\\Sigma_1 = B_0 B_0'` and
    :math:`\\Sigma_2 = B_0 \\Lambda B_0'` via generalized eigenvalue decomposition
    ``scipy.linalg.eigh(Sigma_2, Sigma_1)``, canonicalizes :math:`B_0` for
    positive diagonal dominance, and computes pairwise Wald tests for distinct
    relative shock variances :math:`\\lambda_i \\ne \\lambda_j`.

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame of shape (T, n)
        Endogenous time-series data.
    p : int, default 4
        VAR lag order.
    regime_indicator : int or Sequence[int | bool] or np.ndarray
        Either an integer break index splitting the sample into regime 1
        (:math:`t < t_{\\text{break}}`) and regime 2 (:math:`t \\ge t_{\\text{break}}`),
        or a 1D array of length ``T`` or ``T - p`` containing binary regime labels.
    horizon : int, default 20
        Maximum IRF and FEVD horizon.
    var_names : Sequence[str], optional
        Variable names.

    Returns
    -------
    SVARResult
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0; got {horizon}.")
    Y_arr, names, A_list, _, Sigma, resid = _extract_var_inputs(Y, p, var_names)
    T, n = Y_arr.shape
    T_eff = resid.shape[0]

    if isinstance(regime_indicator, (int, np.integer)) and not isinstance(
        regime_indicator, (bool, np.bool_)
    ):
        t_break = int(regime_indicator)
        # If t_break is specified in full-sample coordinates, convert to residual index
        split_idx = t_break - p if (t_break - p >= n + 1 and T_eff - (t_break - p) >= n + 1) else t_break
        if split_idx < n + 1 or (T_eff - split_idx) < n + 1:
            raise ValueError(
                f"svar_heteroskedastic: regime break index {t_break} leaves fewer than "
                f"n+1={n + 1} residual observations in one of the regimes (T_eff={T_eff})."
            )
        mask1 = np.zeros(T_eff, dtype=bool)
        mask1[:split_idx] = True
        mask2 = ~mask1
    else:
        ri = np.asarray(regime_indicator)
        if ri.ndim != 1:
            ri = ri.ravel()
        if len(ri) == T:
            ri = ri[p:]
        elif len(ri) != T_eff:
            raise ValueError(
                f"svar_heteroskedastic: regime_indicator length ({len(ri)}) must equal "
                f"T ({T}) or T - p ({T_eff})."
            )
        unique_regimes = np.unique(ri)
        if len(unique_regimes) != 2:
            raise ValueError(
                "svar_heteroskedastic: regime_indicator must contain exactly 2 distinct "
                f"regime labels; got {unique_regimes.tolist()}."
            )
        mask1 = ri == unique_regimes[0]
        mask2 = ri == unique_regimes[1]

    T1 = int(np.sum(mask1))
    T2 = int(np.sum(mask2))
    if T1 <= n or T2 <= n:
        raise ValueError(
            f"svar_heteroskedastic: each regime requires at least n+1={n + 1} observations; "
            f"got T1={T1}, T2={T2}."
        )

    u1 = resid[mask1] - np.mean(resid[mask1], axis=0, keepdims=True)
    u2 = resid[mask2] - np.mean(resid[mask2], axis=0, keepdims=True)
    Sigma_1 = (u1.T @ u1) / T1
    Sigma_2 = (u2.T @ u2) / T2

    # Ensure Sigma_1 and Sigma_2 are positive definite
    _ = safe_cholesky(Sigma_1, name="svar_heteroskedastic Sigma_1")
    _ = safe_cholesky(Sigma_2, name="svar_heteroskedastic Sigma_2")

    # Simultaneous diagonalization: Sigma_2 V = Sigma_1 V Lambda with V' Sigma_1 V = I_n
    # Hence B0 = V^{-T} = Sigma_1 @ V satisfies B0 B0' = Sigma_1 and B0 Lambda B0' = Sigma_2.
    evals, V = scipy.linalg.eigh(Sigma_2, Sigma_1)
    B0_raw = Sigma_1 @ V
    B0, perm, aux = _canonicalize_b0_columns(B0_raw, [evals])
    lambda_ratios = np.asarray(aux[0], dtype=float)

    # Pairwise Wald tests for H0: lambda_i = lambda_j
    # Asymptotic variance of lambda_j under regime sample sizes T1, T2:
    # Var(lambda_j) = 2 * lambda_j^2 * (1/T1 + 1/T2)
    var_lambda = 2.0 * (lambda_ratios ** 2) * (1.0 / T1 + 1.0 / T2)
    wald_tests: dict[str, dict[str, float]] = {}
    wald_stats: list[float] = []
    wald_pvals: list[float] = []
    gaps: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            diff = float(lambda_ratios[i] - lambda_ratios[j])
            gaps.append(abs(diff))
            denom = float(var_lambda[i] + var_lambda[j])
            w_stat = (diff ** 2) / max(denom, 1e-18)
            p_val = float(chi2.sf(w_stat, df=1))
            wald_tests[f"shock_{i}_vs_{j}"] = {
                "lambda_i": float(lambda_ratios[i]),
                "lambda_j": float(lambda_ratios[j]),
                "stat": float(w_stat),
                "p_value": p_val,
            }
            wald_stats.append(float(w_stat))
            wald_pvals.append(p_val)

    min_gap = float(min(gaps)) if gaps else float("inf")
    min_wald = float(min(wald_stats)) if wald_stats else float("inf")
    max_pval = float(max(wald_pvals)) if wald_pvals else 0.0

    irf_arr = compute_irf(A_list, B0, horizon=horizon)
    fevd_arr = compute_fevd(A_list, B0, horizon=horizon)
    shocks = _compute_structural_shocks(resid, B0)

    diagnostics: dict[str, Any] = {
        "lambda_ratios": lambda_ratios,
        "min_lambda_gap": min_gap,
        "wald_tests": wald_tests,
        "min_wald_stat": min_wald,
        "max_wald_pvalue": max_pval,
        "identified": bool(max_pval < 0.05),
        "Sigma_1": Sigma_1,
        "Sigma_2": Sigma_2,
        "T_regime_1": T1,
        "T_regime_2": T2,
        "column_permutation": perm,
    }
    metadata: dict[str, Any] = {
        "T": T,
        "T_eff": T_eff,
        "p": p,
        "n": n,
    }
    return SVARResult(
        B0=B0,
        A_list=A_list,
        Sigma=Sigma,
        irf=irf_arr,
        fevd=fevd_arr,
        structural_shocks=shocks,
        method="heteroskedastic",
        horizon=horizon,
        var_names=names,
        diagnostics=diagnostics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 2. Identification via Non-Gaussianity (LMS 2017 / Keweloh 2021)
# ---------------------------------------------------------------------------


def _givens_rotation_matrix(angles: np.ndarray, n: int) -> np.ndarray:
    """Construct an SO(n) rotation matrix from n(n-1)/2 Givens angles."""
    Q = np.eye(n, dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            theta = float(angles[idx])
            c = np.cos(theta)
            s = np.sin(theta)
            col_i = c * Q[:, i] - s * Q[:, j]
            col_j = s * Q[:, i] + c * Q[:, j]
            Q[:, i] = col_i
            Q[:, j] = col_j
            idx += 1
    return Q


def _higher_moment_gmm_loss(angles: np.ndarray, e: np.ndarray) -> float:
    """Cross-shock co-skewness and co-kurtosis GMM objective (Keweloh 2021)."""
    T_eff, n = e.shape
    Q = _givens_rotation_matrix(angles, n)
    eps = e @ Q
    eps2 = eps ** 2
    eps3 = eps2 * eps

    # Co-skewness E[eps_i^2 eps_j] for i != j
    coskew = (eps2.T @ eps) / T_eff
    np.fill_diagonal(coskew, 0.0)

    # Co-kurtosis (3, 1): E[eps_i^3 eps_j] for i != j
    cokurt_31 = (eps3.T @ eps) / T_eff
    np.fill_diagonal(cokurt_31, 0.0)

    # Co-kurtosis (2, 2): E[eps_i^2 eps_j^2] - 1 for i < j
    cokurt_22 = (eps2.T @ eps2) / T_eff - 1.0
    iu = np.triu_indices(n, k=1)

    loss = (
        float(np.sum(coskew ** 2))
        + float(np.sum(cokurt_31 ** 2))
        + float(np.sum(cokurt_22[iu] ** 2))
    )
    return loss


def _fastica_orthogonal(e: np.ndarray, rng: np.random.Generator, n_restarts: int = 8) -> np.ndarray:
    """Symmetric FastICA on whitened residuals e of shape (T_eff, n). Returns Q in O(n)."""
    T_eff, n = e.shape
    X = e.T  # (n, T_eff)
    best_Q = np.eye(n)
    best_contrast = -np.inf

    for r in range(n_restarts):
        if r == 0:
            W = np.eye(n)
        else:
            A = rng.standard_normal((n, n))
            W, R_mat = np.linalg.qr(A)
            W = W @ np.diag(np.where(np.diag(R_mat) >= 0, 1.0, -1.0))

        for _ in range(250):
            WX = W @ X
            # Combine tanh and cubic contrast for both super- and sub-Gaussian shocks
            gWX = np.tanh(WX)
            g_prime = 1.0 - gWX ** 2
            W_new = (gWX @ X.T) / T_eff - np.diag(g_prime.mean(axis=1)) @ W
            U, _, Vt = np.linalg.svd(W_new, full_matrices=False)
            W_new = U @ Vt
            if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)) < 1e-7:
                W = W_new
                break
            W = W_new

        Q_cand = W.T
        eps = e @ Q_cand
        # Evaluate marginal non-Gaussianity (kurtosis^2 + skewness^2)
        skew = np.mean(eps ** 3, axis=0)
        kurt = np.mean(eps ** 4, axis=0) - 3.0
        contrast = float(np.sum(skew ** 2) + 0.25 * np.sum(kurt ** 2))
        if contrast > best_contrast:
            best_contrast = contrast
            best_Q = Q_cand

    return best_Q


def svar_non_gaussian(
    Y: np.ndarray | pd.DataFrame,
    p: int = 4,
    *,
    method: str = "gmm",
    horizon: int = 20,
    seed: int = 42,
    var_names: Sequence[str] | None = None,
) -> SVARResult:
    """Identify an SVAR via non-Gaussian higher moments (LMS 2017; Keweloh 2021).

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame of shape (T, n)
        Endogenous time-series data.
    p : int, default 4
        VAR lag order.
    method : {"gmm", "fastica"}, default "gmm"
        Identification estimator:
        - ``"gmm"``: minimizes cross-shock co-skewness and excess co-kurtosis
          over Givens rotations :math:`Q \\in SO(n)`.
        - ``"fastica"``: symmetric FastICA contrast maximization on whitened
          reduced-form innovations.
    horizon : int, default 20
        Maximum IRF and FEVD horizon.
    seed : int, default 42
        Random seed for multi-start optimization.
    var_names : Sequence[str], optional
        Variable names.

    Returns
    -------
    SVARResult
    """
    method_clean = method.lower().strip()
    if method_clean not in {"gmm", "fastica"}:
        raise ValueError(f"svar_non_gaussian: method must be 'gmm' or 'fastica'; got {method!r}.")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0; got {horizon}.")

    Y_arr, names, A_list, _, Sigma, resid = _extract_var_inputs(Y, p, var_names)
    T, n = Y_arr.shape
    T_eff = resid.shape[0]
    rng = np.random.default_rng(seed)

    u_centered = resid - np.mean(resid, axis=0, keepdims=True)
    Sigma_u = (u_centered.T @ u_centered) / T_eff
    L = safe_cholesky(Sigma_u, name="svar_non_gaussian Sigma_u")
    e = np.linalg.solve(L, u_centered.T).T  # (T_eff, n), exact sample covariance I_n

    n_angles = n * (n - 1) // 2
    Q_ica = _fastica_orthogonal(e, rng=rng, n_restarts=10)

    if method_clean == "fastica" or n_angles == 0:
        Q_opt = Q_ica
        angles_zero = np.zeros(n_angles, dtype=float)
        gmm_loss = _higher_moment_gmm_loss(angles_zero, e @ Q_opt)
    else:
        # Multi-start Givens optimization initialized from identity, FastICA base, and random angles
        best_loss = _higher_moment_gmm_loss(np.zeros(n_angles), e @ Q_ica)
        best_Q = Q_ica.copy()

        # Optimize in the frame of Q_ica first, plus random starts
        bases = [Q_ica, np.eye(n)]
        for base_Q in bases:
            e_base = e @ base_Q
            starts = [np.zeros(n_angles, dtype=float)] + [
                rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=n_angles) for _ in range(5)
            ]
            for theta0 in starts:
                res = minimize(
                    _higher_moment_gmm_loss,
                    theta0,
                    args=(e_base,),
                    method="BFGS",
                    options={"gtol": 1e-7, "maxiter": 300},
                )
                if float(res.fun) < best_loss:
                    best_loss = float(res.fun)
                    best_Q = base_Q @ _givens_rotation_matrix(res.x, n)

        Q_opt = best_Q
        gmm_loss = best_loss

    B0_raw = L @ Q_opt
    B0, perm, aux = _canonicalize_b0_columns(B0_raw, [Q_opt])
    Q_final = aux[0]

    shocks = e @ Q_final
    skewness = np.mean(shocks ** 3, axis=0)
    excess_kurtosis = np.mean(shocks ** 4, axis=0) - 3.0
    jb_stats = (T_eff / 6.0) * (skewness ** 2 + 0.25 * (excess_kurtosis ** 2))
    jb_pvalues = chi2.sf(jb_stats, df=2)

    irf_arr = compute_irf(A_list, B0, horizon=horizon)
    fevd_arr = compute_fevd(A_list, B0, horizon=horizon)

    diagnostics: dict[str, Any] = {
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_stat": jb_stats,
        "jarque_bera_pvalue": jb_pvalues,
        "joint_jb_stat": float(np.sum(jb_stats)),
        "joint_jb_pvalue": float(chi2.sf(np.sum(jb_stats), df=2 * n)),
        "gmm_objective": float(gmm_loss),
        "Q": Q_final,
        "column_permutation": perm,
    }
    metadata: dict[str, Any] = {
        "T": T,
        "T_eff": T_eff,
        "p": p,
        "n": n,
        "estimator": method_clean,
        "seed": seed,
    }
    return SVARResult(
        B0=B0,
        A_list=A_list,
        Sigma=Sigma,
        irf=irf_arr,
        fevd=fevd_arr,
        structural_shocks=shocks,
        method="non_gaussian",
        horizon=horizon,
        var_names=names,
        diagnostics=diagnostics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 3. Identification via External Instrument / Proxy-SVAR (Mertens & Ravn 2013)
# ---------------------------------------------------------------------------


def _estimate_proxy_column(
    u: np.ndarray,
    z: np.ndarray,
    Sigma_u: np.ndarray,
    shocked_var: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute structural impact matrix B0, 1-SD impact vector, and unit-shock vector."""
    n = u.shape[1]
    z_c = z - np.mean(z)
    u_c = u - np.mean(u, axis=0, keepdims=True)
    denom = float(z_c @ u_c[:, shocked_var])
    if abs(denom) < 1e-14:
        raise np.linalg.LinAlgError(
            "svar_proxy: instrument z has near-zero covariance with the shocked variable residual."
        )
    # Relative impact vector gamma = b_1 / b_{1, shocked_var}
    gamma = (u_c.T @ z_c) / denom  # gamma[shocked_var] == 1.0

    # Covariance partition scale: b_1' Sigma_u^{-1} b_1 = 1
    L = safe_cholesky(Sigma_u, name="svar_proxy Sigma_u")
    v = np.linalg.solve(L, gamma)
    b_shocked_1sd = 1.0 / float(np.sqrt(max(float(v @ v), 1e-18)))
    b_col_1sd = gamma * b_shocked_1sd
    b_col_scaled = b_col_1sd * float(scale)

    # Complete B0 so that column `shocked_var` is `b_col_scaled` and remaining
    # columns span the orthogonal complement in Sigma_u metric.
    B0 = np.zeros((n, n), dtype=float)
    B0[:, shocked_var] = b_col_scaled
    rem_cov = Sigma_u - np.outer(b_col_1sd, b_col_1sd)
    eigvals, eigvecs = np.linalg.eigh((rem_cov + rem_cov.T) * 0.5)
    order = np.argsort(eigvals)[::-1]
    other_cols = [c for c in range(n) if c != shocked_var]
    for k, col_idx in enumerate(other_cols):
        lam = max(float(eigvals[order[k]]), 0.0)
        vec = eigvecs[:, order[k]] * np.sqrt(lam)
        if vec[col_idx] < 0.0:
            vec = -vec
        B0[:, col_idx] = vec

    return B0, b_col_scaled, gamma * float(scale)


def svar_proxy(
    Y: np.ndarray | pd.DataFrame,
    z: Sequence[float] | np.ndarray,
    p: int = 4,
    *,
    shocked_var: int = 0,
    scale: float = 1.0,
    horizon: int = 20,
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 42,
    var_names: Sequence[str] | None = None,
) -> SVARResult:
    """Identify an SVAR column using an external instrument (Mertens & Ravn 2013; Stock & Watson 2018).

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame of shape (T, n)
        Endogenous time-series data.
    z : Sequence[float] or np.ndarray
        External proxy / instrument series of length ``T`` or ``T - p``. May contain
        ``NaN`` values for periods outside the proxy availability window.
    p : int, default 4
        VAR lag order.
    shocked_var : int, default 0
        Index of the instrumented endogenous variable.
    scale : float, default 1.0
        Scale factor applied to the identified structural shock.
    horizon : int, default 20
        Maximum IRF and FEVD horizon.
    n_boot : int, default 200
        Number of Rademacher wild bootstrap replications.
    alpha : float, default 0.10
        Significance level for ``1 - alpha`` bootstrap bands and Anderson-Rubin sets.
    seed : int, default 42
        Random seed for bootstrap inference.
    var_names : Sequence[str], optional
        Variable names.

    Returns
    -------
    SVARResult
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0; got {horizon}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    Y_arr, names, A_list, c_vec, Sigma, resid = _extract_var_inputs(Y, p, var_names)
    T, n = Y_arr.shape
    T_eff = resid.shape[0]
    if not (0 <= shocked_var < n):
        raise ValueError(f"shocked_var must be in [0, {n - 1}]; got {shocked_var}.")

    z_arr = np.asarray(z, dtype=float).ravel()
    if len(z_arr) == T:
        z_aligned = z_arr[p:]
    elif len(z_arr) == T_eff:
        z_aligned = z_arr
    else:
        raise ValueError(
            f"svar_proxy: instrument z length ({len(z_arr)}) must match T ({T}) or T - p ({T_eff})."
        )

    valid_mask = np.isfinite(z_aligned)
    T_z = int(np.sum(valid_mask))
    if T_z <= n + 2:
        raise ValueError(f"svar_proxy: only {T_z} non-NaN instrument observations available.")

    u_valid = resid[valid_mask]
    z_valid = z_aligned[valid_mask]
    Sigma_u = (resid.T @ resid) / T_eff

    B0, b_col_scaled, unit_shock_vec = _estimate_proxy_column(
        u_valid, z_valid, Sigma_u, shocked_var=shocked_var, scale=scale
    )

    # First-stage heteroskedasticity-robust Olea-Pflueger / EHW effective F-statistic
    z_tilde = z_valid - np.mean(z_valid)
    u_tilde = u_valid - np.mean(u_valid, axis=0, keepdims=True)
    u1 = u_tilde[:, shocked_var]
    zz = float(z_tilde @ z_tilde)
    pi_hat = float(z_tilde @ u1) / max(zz, 1e-18)
    v_first = u1 - pi_hat * z_tilde
    var_pi_ehw = (T_z / max(T_z - 2, 1)) * float(np.sum((z_tilde ** 2) * (v_first ** 2))) / max(zz ** 2, 1e-18)
    mop_f = (pi_hat ** 2) / max(var_pi_ehw, 1e-18)

    # Weak-instrument-robust Anderson-Rubin (1949) confidence set at impact
    crit_val = float(chi2.ppf(1.0 - alpha, df=1))
    s_zu1 = float(z_tilde @ u1)
    v_11 = (T_z / max(T_z - 2, 1)) * float(np.sum((z_tilde ** 2) * (u1 ** 2)))
    A_coef = (s_zu1 ** 2) - crit_val * v_11
    ar_bounded = bool(A_coef > 0.0)

    ar_rel_lower = np.zeros(n, dtype=float)
    ar_rel_upper = np.zeros(n, dtype=float)
    for j in range(n):
        if j == shocked_var:
            ar_rel_lower[j] = 1.0
            ar_rel_upper[j] = 1.0
            continue
        uj = u_tilde[:, j]
        s_zuj = float(z_tilde @ uj)
        v_1j = (T_z / max(T_z - 2, 1)) * float(np.sum((z_tilde ** 2) * u1 * uj))
        v_jj = (T_z / max(T_z - 2, 1)) * float(np.sum((z_tilde ** 2) * (uj ** 2)))
        B_coef = s_zu1 * s_zuj - crit_val * v_1j
        C_coef = (s_zuj ** 2) - crit_val * v_jj
        disc = max(B_coef ** 2 - A_coef * C_coef, 0.0)
        if A_coef > 0.0:
            ar_rel_lower[j] = (B_coef - np.sqrt(disc)) / A_coef
            ar_rel_upper[j] = (B_coef + np.sqrt(disc)) / A_coef
        else:
            ar_rel_lower[j] = -np.inf
            ar_rel_upper[j] = np.inf

    shock_scale = float(b_col_scaled[shocked_var])
    ar_ci_lower = ar_rel_lower * shock_scale
    ar_ci_upper = ar_rel_upper * shock_scale

    irf_arr = compute_irf(A_list, B0, horizon=horizon)
    fevd_arr = compute_fevd(A_list, B0, horizon=horizon)
    shocks = _compute_structural_shocks(resid, B0)

    # Wild bootstrap confidence bands (Mertens & Ravn 2013 Rademacher multiplier)
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boot_irfs: list[np.ndarray] = []
        for _ in range(n_boot):
            w = rng.choice(np.array([-1.0, 1.0]), size=T_eff)
            u_star = resid * w[:, None]
            z_star = z_aligned * w

            Y_star = np.zeros_like(Y_arr)
            Y_star[:p] = Y_arr[:p]
            for t in range(p, T):
                y_t = c_vec.copy()
                for l_idx in range(p):
                    y_t = y_t + A_list[l_idx] @ Y_star[t - 1 - l_idx]
                Y_star[t] = y_t + u_star[t - p]

            try:
                est_b = estimate_var(Y_star, p=p)
                u_b = np.asarray(est_b.resid, dtype=float)
                Sigma_ub = (u_b.T @ u_b) / T_eff
                B0_b, _, _ = _estimate_proxy_column(
                    u_b[valid_mask],
                    z_star[valid_mask],
                    Sigma_ub,
                    shocked_var=shocked_var,
                    scale=scale,
                )
                if np.sign(B0_b[shocked_var, shocked_var]) != np.sign(B0[shocked_var, shocked_var]):
                    B0_b[:, shocked_var] *= -1.0
                boot_irfs.append(compute_irf(list(est_b.A_list), B0_b, horizon=horizon))
            except Exception:
                continue

        if boot_irfs:
            stack = np.stack(boot_irfs, axis=0)
            ci_lower = np.quantile(stack, alpha / 2.0, axis=0)
            ci_upper = np.quantile(stack, 1.0 - alpha / 2.0, axis=0)

    diagnostics: dict[str, Any] = {
        "mop_f": float(mop_f),
        "first_stage_f": float(mop_f),
        "weak_iv": bool(mop_f < 10.0),
        "ar_ci_lower": ar_ci_lower,
        "ar_ci_upper": ar_ci_upper,
        "ar_relative_lower": ar_rel_lower,
        "ar_relative_upper": ar_rel_upper,
        "ar_bounded": ar_bounded,
        "impact_vector": b_col_scaled,
        "unit_shock_impact": unit_shock_vec,
        "n_valid_instrument": T_z,
    }
    metadata: dict[str, Any] = {
        "T": T,
        "T_eff": T_eff,
        "p": p,
        "n": n,
        "shocked_var": shocked_var,
        "scale": scale,
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
    }
    return SVARResult(
        B0=B0,
        A_list=A_list,
        Sigma=Sigma,
        irf=irf_arr,
        fevd=fevd_arr,
        structural_shocks=shocks,
        method="proxy",
        horizon=horizon,
        var_names=names,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        diagnostics=diagnostics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 4. Sign + Narrative Restrictions with Giacomini-Kitagawa (2021) Bounds
# ---------------------------------------------------------------------------


def _parse_sign_restrictions(
    sign_restrictions: Mapping[tuple[int, int, int], int] | Sequence[tuple[int, int, int, int]],
    n: int,
    horizon: int,
) -> list[tuple[int, int, int, int]]:
    """Normalize sign restrictions into a list of (var_idx, shock_idx, h, sign)."""
    parsed: list[tuple[int, int, int, int]] = []
    if isinstance(sign_restrictions, Mapping):
        for key, s_val in sign_restrictions.items():
            if len(key) != 3:
                raise ValueError(
                    f"svar_sign_narrative: mapping key must be (var_idx, shock_idx, horizon); got {key!r}."
                )
            v_idx, s_idx, h = int(key[0]), int(key[1]), int(key[2])
            s = int(s_val)
            if s == 0:
                continue
            if s not in (-1, 1):
                raise ValueError(f"Sign restriction must be +1 or -1; got {s}.")
            if not (0 <= v_idx < n and 0 <= s_idx < n and 0 <= h <= horizon):
                raise ValueError(
                    f"Sign restriction index ({v_idx}, {s_idx}, {h}) out of bounds for n={n}, horizon={horizon}."
                )
            parsed.append((v_idx, s_idx, h, s))
    else:
        for item in sign_restrictions:
            if len(item) != 4:
                raise ValueError(
                    "svar_sign_narrative: sequence items must be 4-tuples "
                    f"(var_idx, shock_idx, horizon, sign); got {item!r}."
                )
            v_idx, s_idx, h, s = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            if s == 0:
                continue
            if s not in (-1, 1):
                raise ValueError(f"Sign restriction must be +1 or -1; got {s}.")
            if not (0 <= v_idx < n and 0 <= s_idx < n and 0 <= h <= horizon):
                raise ValueError(
                    f"Sign restriction index ({v_idx}, {s_idx}, {h}) out of bounds for n={n}, horizon={horizon}."
                )
            parsed.append((v_idx, s_idx, h, s))
    return parsed


def _check_narrative_restrictions(
    narrative_restrictions: Sequence[tuple[str, int, int, int]],
    eps: np.ndarray,
    B0: np.ndarray,
    n: int,
    p: int,
) -> bool:
    """Verify whether all narrative restrictions hold for structural shocks eps and impact B0."""
    T_eff = eps.shape[0]
    for item in narrative_restrictions:
        if len(item) != 4:
            raise ValueError(
                "svar_sign_narrative: narrative restriction must be a 4-tuple "
                f"(kind, shock_idx, t_idx, val); got {item!r}."
            )
        kind = str(item[0]).lower().strip()
        a1, a2, a3 = int(item[1]), int(item[2]), int(item[3])
        # Automatically disambiguate (shock_idx, t_idx) if user swapped order and t_idx >= n
        if a1 >= n and 0 <= a2 < n:
            shock_idx, t_raw = a2, a1
        else:
            shock_idx, t_raw = a1, a2

        t_idx = t_raw - p if (t_raw >= T_eff and 0 <= t_raw - p < T_eff) else t_raw
        if not (0 <= shock_idx < n and 0 <= t_idx < T_eff):
            raise ValueError(
                f"svar_sign_narrative: narrative index (shock={shock_idx}, t={t_raw}) out of bounds."
            )

        if kind in {"shock_sign", "sign"}:
            if a3 not in (-1, 1):
                raise ValueError(f"Narrative shock_sign must be +1 or -1; got {a3}.")
            if np.sign(eps[t_idx, shock_idx]) != a3:
                return False
        elif kind in {"hd_dominance", "dominance", "hd_max"}:
            var_idx = a3
            if not (0 <= var_idx < n):
                raise ValueError(f"Narrative hd_dominance var_idx={var_idx} out of bounds.")
            contribs = np.abs(B0[var_idx, :] * eps[t_idx, :])
            own = contribs[shock_idx]
            others = np.delete(contribs, shock_idx)
            if own < float(np.max(others)):
                return False
        else:
            raise ValueError(f"Unknown narrative restriction kind {kind!r}.")
    return True


def svar_sign_narrative(
    Y: np.ndarray | pd.DataFrame,
    p: int = 4,
    *,
    sign_restrictions: Mapping[tuple[int, int, int], int] | Sequence[tuple[int, int, int, int]],
    narrative_restrictions: Sequence[tuple[str, int, int, int]] | None = None,
    horizon: int = 20,
    n_draws: int = 500,
    max_trials: int = 20000,
    alpha: float = 0.16,
    seed: int = 42,
    var_names: Sequence[str] | None = None,
) -> SVARResult:
    """Identify an SVAR via Haar-QR sign & narrative restrictions with GK (2021) bounds.

    Combines Rubio-Ramírez, Waggoner & Zha (2010) sign restrictions and
    Antolín-Díaz & Rubio-Ramírez (2018) narrative sign / historical decomposition
    restrictions, and reports Giacomini & Kitagawa (2021, *Econometrica*)
    prior-robust identified-set bounds ``gk_lower`` and ``gk_upper``.

    Parameters
    ----------
    Y : np.ndarray or pd.DataFrame of shape (T, n)
        Endogenous time-series data.
    p : int, default 4
        VAR lag order.
    sign_restrictions : Mapping[(var_idx, shock_idx, horizon), sign] or Sequence[(var_idx, shock_idx, horizon, sign)]
        Traditional impulse-response sign restrictions where ``sign`` is ``+1`` or ``-1``.
    narrative_restrictions : Sequence[(kind, shock_idx, t_idx, val)], optional
        Narrative restrictions at residual time index ``t_idx``:
        - ``("shock_sign", shock_idx, t_idx, +1/-1)``: structural shock sign at ``t_idx``.
        - ``("hd_dominance", shock_idx, t_idx, var_idx)``: shock ``shock_idx`` is the
          largest contemporaneous contributor to variable ``var_idx`` at ``t_idx``.
    horizon : int, default 20
        Maximum IRF and FEVD horizon.
    n_draws : int, default 500
        Target number of accepted orthogonal rotations ``Q``.
    max_trials : int, default 20000
        Maximum number of Haar-QR rotation proposals.
    alpha : float, default 0.16
        Tail probability for pointwise percentile bands ``ci_lower`` and ``ci_upper``
        (default 0.16 gives 68% bands).
    seed : int, default 42
        Random seed for Haar rotation sampling.
    var_names : Sequence[str], optional
        Variable names.

    Returns
    -------
    SVARResult
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0; got {horizon}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if n_draws < 1 or max_trials < 1:
        raise ValueError("n_draws and max_trials must be >= 1.")

    Y_arr, names, A_list, _, Sigma, resid = _extract_var_inputs(Y, p, var_names)
    T, n = Y_arr.shape
    T_eff = resid.shape[0]

    parsed_signs = _parse_sign_restrictions(sign_restrictions, n=n, horizon=horizon)
    narr_list = list(narrative_restrictions) if narrative_restrictions is not None else []

    Sigma_u = (resid.T @ resid) / T_eff
    L = safe_cholesky(Sigma_u, name="svar_sign_narrative Sigma_u")
    e_whitened = np.linalg.solve(L, resid.T).T  # (T_eff, n)

    # Precompute Cholesky-orthogonalized IRF basis Phi_L of shape (horizon + 1, n, n)
    Phi_L = compute_irf(A_list, L, horizon=horizon)

    # Group sign restrictions by shock column j so we can flip individual column signs
    restr_by_shock: dict[int, list[tuple[int, int, int]]] = {j: [] for j in range(n)}
    for v_idx, s_idx, h, s in parsed_signs:
        restr_by_shock[s_idx].append((v_idx, h, s))

    rng = np.random.default_rng(seed)
    accepted_irfs: list[np.ndarray] = []
    accepted_B0s: list[np.ndarray] = []
    trials = 0

    while len(accepted_irfs) < n_draws and trials < max_trials:
        trials += 1
        G = rng.standard_normal((n, n))
        Q, R_mat = np.linalg.qr(G)
        d = np.diag(R_mat)
        Q = Q * np.where(d >= 0.0, 1.0, -1.0)[None, :]

        # Try flipping individual column signs using the first restriction on each shock
        for j in range(n):
            if restr_by_shock[j]:
                v0, h0, s0 = restr_by_shock[j][0]
                resp0 = float(Phi_L[h0, v0, :] @ Q[:, j])
                if np.sign(resp0) != s0:
                    Q[:, j] = -Q[:, j]
            else:
                if float(L[j, :] @ Q[:, j]) < 0.0:
                    Q[:, j] = -Q[:, j]

        irf_cand = np.einsum("hij,jk->hik", Phi_L, Q)

        signs_ok = True
        for v_idx, s_idx, h, s in parsed_signs:
            if np.sign(irf_cand[h, v_idx, s_idx]) != s:
                signs_ok = False
                break
        if not signs_ok:
            continue

        B0_cand = L @ Q
        if narr_list:
            eps_cand = e_whitened @ Q
            if not _check_narrative_restrictions(narr_list, eps_cand, B0_cand, n=n, p=p):
                continue

        accepted_irfs.append(irf_cand)
        accepted_B0s.append(B0_cand)

    if not accepted_irfs:
        raise RuntimeError(
            f"svar_sign_narrative: 0 draws accepted out of {trials} trials. "
            "Relax restrictions or increase max_trials."
        )

    irf_stack = np.stack(accepted_irfs, axis=0)  # (M, H+1, n, n)
    B0_stack = np.stack(accepted_B0s, axis=0)    # (M, n, n)

    # Pointwise median and Fry-Pagan (2011) median-target draw
    irf_med = np.median(irf_stack, axis=0)
    scale_std = np.std(irf_stack, axis=0)
    scale_std = np.where(scale_std > 1e-12, scale_std, 1.0)
    dists = np.sum(((irf_stack - irf_med[None, ...]) / scale_std[None, ...]) ** 2, axis=(1, 2, 3))
    best_idx = int(np.argmin(dists))

    B0_best = B0_stack[best_idx]
    fevd_arr = compute_fevd(A_list, B0_best, horizon=horizon)
    shocks = e_whitened @ np.linalg.solve(L, B0_best)

    ci_lower = np.quantile(irf_stack, alpha / 2.0, axis=0)
    ci_upper = np.quantile(irf_stack, 1.0 - alpha / 2.0, axis=0)
    gk_lower = np.min(irf_stack, axis=0)
    gk_upper = np.max(irf_stack, axis=0)

    diagnostics: dict[str, Any] = {
        "n_accepted": len(accepted_irfs),
        "n_trials": trials,
        "acceptance_rate": float(len(accepted_irfs)) / float(trials),
        "identified_set_mean_width": float(np.mean(gk_upper - gk_lower)),
        "identified_set_impact_width": gk_upper[0] - gk_lower[0],
        "accepted_B0_draws": B0_stack,
        "accepted_irf_draws": irf_stack,
        "median_target_index": best_idx,
    }
    metadata: dict[str, Any] = {
        "T": T,
        "T_eff": T_eff,
        "p": p,
        "n": n,
        "n_draws": n_draws,
        "max_trials": max_trials,
        "alpha": alpha,
        "seed": seed,
    }
    return SVARResult(
        B0=B0_best,
        A_list=A_list,
        Sigma=Sigma,
        irf=irf_med,
        fevd=fevd_arr,
        structural_shocks=shocks,
        method="sign_narrative",
        horizon=horizon,
        var_names=names,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        gk_lower=gk_lower,
        gk_upper=gk_upper,
        diagnostics=diagnostics,
        metadata=metadata,
    )
