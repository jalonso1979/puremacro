"""Dynamic Factor Model with Kalman smoothing — Doz-Giannone-Reichlin
(2011) two-step estimator.

Standard nowcasting pipeline:
  1. PCA on the standardised complete-cases prefix gives an initial
     factor estimate F̂_0.
  2. Estimate a VAR(p) on F̂_0 → state transition matrix.
  3. Estimate Λ by OLS of X on F̂_0 → observation loading.
  4. Cast as a Gaussian state-space model and run the Kalman smoother.
     NaN handling in :mod:`puremacro.state_space` lets the *current*
     month/quarter carry only the variables already published; the
     smoothed factors at t = T are the nowcast.

For static factor analysis (no Kalman, no missing data) use
:func:`puremacro.factor.static_dfm_fit` directly.

References
----------
Doz, C., Giannone, D., Reichlin, L. (2011). A two-step estimator for
    large approximate dynamic factor models based on Kalman filtering.
    Journal of Econometrics 164(1), 188-205.
Giannone, D., Reichlin, L., Small, D. (2008). Nowcasting: the real-time
    informational content of macroeconomic data. JME 55(4), 665-676.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any, Optional

from ..factor import pca_factors
from ..state_space import StateSpaceModel, kalman_smoother


class KalmanDFMResult(dict):
    """Return type of :func:`kalman_dfm`: a ``dict`` with the documented
    keys plus ``summary()``, ``to_frame()``, ``to_markdown()``,
    ``to_latex()``, ``to_typst()`` and ``plot()``.

    ``out["factors"]``, ``"factors" in out`` and every other dict
    operation work unchanged; the presentation methods are additive.
    """

    columns: Optional[list[str]] = None

    def _names(self) -> list[str]:
        n = self["loadings"].shape[0]
        return list(self.columns) if self.columns is not None else [f"x{i}" for i in range(n)]

    def summary(self) -> str:
        F = self["factors"]
        T, k = F.shape
        n = self["loadings"].shape[0]
        A = self["A"]
        p = A.shape[0] // k
        eig = np.abs(np.linalg.eigvals(A)).max() if A.size else float("nan")
        n_filled = int(np.isnan(self["X_filled"]).sum()) if "X_filled" in self else 0
        lines = [
            "Two-step DFM with Kalman smoother (Doz-Giannone-Reichlin 2011)",
            "=" * 66,
            f"Panel                          : T = {T}, n = {n}",
            f"Factors / VAR order            : k = {k}, p = {p}",
            f"Max |eigenvalue| of transition : {eig:.4f}",
            f"Log-likelihood                 : {self.get('loglik', float('nan')):.3f}",
            f"Missing entries filled         : {int(self.get('n_missing', 0))}"
            + (f" (still NaN: {n_filled})" if n_filled else ""),
            "-" * 66,
            "Loadings (first factor):",
        ]
        for name, lam in zip(self._names(), self["loadings"][:, 0]):
            lines.append(f"  {name:<24s}: {lam:+8.4f}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Loadings table (n series x k factors)."""
        k = self["loadings"].shape[1]
        return pd.DataFrame(
            self["loadings"], index=pd.Index(self._names(), name="series"),
            columns=[f"F{i + 1}" for i in range(k)],
        ).round(4)

    def to_markdown(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, *, ax: Any = None, title: str = "Smoothed factors") -> Any:
        """Line plot of the smoothed factor paths. Returns the Figure."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
        else:
            fig = ax.figure
        F = self["factors_df"] if "factors_df" in self else pd.DataFrame(
            self["factors"], columns=[f"F{i + 1}" for i in range(self["factors"].shape[1])])
        for col in F.columns:
            ax.plot(F.index, F[col], lw=1.6, label=str(col))
        ax.axhline(0.0, color="grey", lw=0.6)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        return fig


def _var_ols(F: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Companion-form VAR(p) on a (T, k) factor matrix.

    Returns
    -------
    A : (k·p, k·p) state-transition (companion form).
    Q : (k·p, k·p) state-shock covariance (block-diagonal: top-left is
        the residual covariance of the VAR, rest is zero).
    """
    T, k = F.shape
    if T <= p + 1:
        raise ValueError(f"need T > p + 1 = {p + 1}; got T = {T}")
    Y = F[p:]
    X = np.column_stack([F[p - lag - 1: T - lag - 1] for lag in range(p)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)         # (k·p, k)
    resid = Y - X @ beta
    Sigma = (resid.T @ resid) / max(1, T - p)

    A = np.zeros((k * p, k * p))
    A[:k] = beta.T                                        # top block
    if p > 1:
        A[k:, :-k] = np.eye(k * (p - 1))                  # shift
    Q = np.zeros((k * p, k * p))
    Q[:k, :k] = Sigma
    return A, Q


def kalman_dfm(
    X: np.ndarray | pd.DataFrame,
    *,
    n_factors: int,
    p: int = 1,
    standardize: bool = True,
    diffuse_scale: float = 1e6,
) -> KalmanDFMResult:
    """Two-step DFM with Kalman smoothing.

    Parameters
    ----------
    X : (T, n) ndarray or DataFrame
        Observation panel. NaN entries (e.g. ragged-edge missing
        observations at the latest periods) are handled by the Kalman
        filter; the smoothed factors fill them in.
    n_factors : int
        Number of latent factors.
    p : int, default 1
        VAR order on the factor process.
    standardize : bool, default True
        Z-score columns before fitting (standard practice for DFMs).
    diffuse_scale : float
        Initial state-covariance scale (approximately diffuse prior).

    Returns
    -------
    KalmanDFMResult
        A ``dict`` subclass (all dict operations work) with
        ``factors``      (T, n_factors) — smoothed factor path,
        ``loadings``     (n, n_factors) — Λ,
        ``A``            (k·p, k·p)   — companion-form transition,
        ``Q``            (k·p, k·p)   — state-shock covariance,
        ``H``            (n, n)       — diagonal idiosyncratic-noise covariance,
        ``X_filled``     (T, n)       — observations with the smoothed
                                          model-implied values substituted
                                          for NaN entries (the *nowcast*),
        ``means``, ``stds`` — used for back-transforming if standardize=True,
        ``loglik``       float        — Gaussian log-likelihood of the panel,
        ``n_missing``    int          — number of NaN entries that were filled,
        ``factors_df``, ``X_filled_df`` — DataFrame views, only when ``X``
                                          was a DataFrame,
        plus ``summary()``, ``to_frame()`` (loadings), ``to_markdown()``,
        ``to_latex()``, ``to_typst()`` and ``plot()``.
    """
    if isinstance(X, pd.DataFrame):
        idx = X.index
        cols = list(X.columns)
        X_arr = X.values.astype(float)
    else:
        idx = None; cols = None
        X_arr = np.asarray(X, dtype=float)
    T, n = X_arr.shape

    # 1) Standardisation (column-wise, ignoring NaNs).
    means = np.nanmean(X_arr, axis=0) if standardize else np.zeros(n)
    stds = np.nanstd(X_arr, axis=0, ddof=0) if standardize else np.ones(n)
    stds = np.where(stds < 1e-12, 1.0, stds)
    Xs = (X_arr - means) / stds if standardize else X_arr.copy()

    # 2) PCA on complete-cases prefix to initialise factors and loadings.
    complete = ~np.isnan(Xs).any(axis=1)
    if complete.sum() <= max(n_factors, p) + 2:
        raise ValueError(
            f"too few complete-cases rows ({int(complete.sum())}) for "
            f"n_factors={n_factors}, p={p}"
        )
    pca = pca_factors(Xs[complete], k=n_factors,
                       demean=False, standardize=False)
    F_init = pca["factors"]               # (T_complete, n_factors)
    Lambda = pca["loadings"]              # (n, n_factors)

    # Idiosyncratic-noise covariance (diagonal H, fitted on complete cases).
    e = Xs[complete] - F_init @ Lambda.T
    H_diag = np.maximum(np.var(e, axis=0, ddof=0), 1e-6)
    H = np.diag(H_diag)

    # 3) VAR(p) on the initial factors.
    A, Q = _var_ols(F_init, p)
    k = n_factors
    # Observation loading in companion form: only the contemporaneous
    # block matters; pad with zeros for the lagged states.
    Z = np.zeros((n, k * p))
    Z[:, :k] = Lambda

    # 4) State-space model + Kalman smoother on the *full* (possibly
    # ragged) standardised panel.
    ssm = StateSpaceModel(T=A, Z=Z, Q=Q, H=H)
    sm = kalman_smoother(Xs, ssm, diffuse_scale=diffuse_scale)
    a_smooth = sm["a_smooth"]                # (T, k·p)
    factors = a_smooth[:, :k]                # (T, n_factors)

    # 5) Nowcast: replace NaNs in Xs with the model-implied predictions.
    X_implied = factors @ Lambda.T
    X_filled = np.where(np.isnan(Xs), X_implied, Xs)
    if standardize:
        X_filled = X_filled * stds + means

    out = KalmanDFMResult(
        factors=factors,
        loadings=Lambda,
        A=A,
        Q=Q,
        H=H,
        X_filled=X_filled,
        means=means,
        stds=stds,
        loglik=float(sm["loglik"]),
        n_missing=int(np.isnan(Xs).sum()),
    )
    out.columns = [str(c) for c in cols] if cols is not None else None
    if idx is not None:
        out["factors_df"] = pd.DataFrame(
            factors, index=idx,
            columns=[f"F{i + 1}" for i in range(n_factors)],
        )
        out["X_filled_df"] = pd.DataFrame(X_filled, index=idx, columns=cols)
    return out


__all__ = ["KalmanDFMResult", "kalman_dfm"]
