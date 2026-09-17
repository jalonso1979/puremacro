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

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class DynamicFactorModelResult:
    """Immutable result object from :class:`DynamicFactorModel` estimation.

    Parameters
    ----------
    factors : np.ndarray
        Smoothed latent factors, shape (T, n_factors).
    loadings : np.ndarray
        Observation loadings matrix Lambda, shape (n, n_factors).
    A : np.ndarray
        Companion-form transition matrix, shape (n_factors*p, n_factors*p).
    Q : np.ndarray
        State shock covariance matrix, shape (n_factors*p, n_factors*p).
    H : np.ndarray
        Diagonal idiosyncratic covariance matrix, shape (n, n).
    X_filled : np.ndarray
        Full panel with Kalman-smoothed implied predictions filling missing entries.
    means : np.ndarray
        Sample means used for centering.
    stds : np.ndarray
        Sample standard deviations used for scaling.
    loglik : float
        Gaussian log-likelihood of the state-space model.
    n_missing : int
        Count of NaN values filled in the panel.
    n_factors : int
        Number of common latent factors.
    p : int
        VAR lag order of the factor transition process.
    columns : tuple[str, ...]
        Names of the panel series.
    index : Optional[tuple[Any, ...]]
        Time or observation index of the panel.
    factors_df : pd.DataFrame
        DataFrame view of smoothed factors.
    X_filled_df : pd.DataFrame
        DataFrame view of filled panel.
    ssm : Optional[StateSpaceModel]
        Fitted underlying StateSpaceModel instance.
    smoother_out : dict[str, Any]
        Raw output dictionary from kalman_smoother.
    """

    factors: np.ndarray
    loadings: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    X_filled: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    loglik: float
    n_missing: int
    n_factors: int
    p: int
    columns: tuple[str, ...] = field(default_factory=tuple)
    index: Optional[tuple[Any, ...]] = None
    factors_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    X_filled_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ssm: Optional[StateSpaceModel] = None
    smoother_out: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style attribute access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.smoother_out:
            return self.smoother_out[key]
        raise KeyError(f"DynamicFactorModelResult has no key {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute or smoother_out key with default."""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        if key in self.smoother_out:
            return self.smoother_out[key]
        return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.smoother_out

    def _names(self) -> list[str]:
        if self.columns:
            return list(self.columns)
        return [f"x{i}" for i in range(self.loadings.shape[0])]

    def summary(self) -> str:
        """Text summary of Dynamic Factor Model estimation."""
        T, k = self.factors.shape
        n = self.loadings.shape[0]
        eig = np.abs(np.linalg.eigvals(self.A)).max() if self.A.size else float("nan")
        n_filled = int(np.isnan(self.X_filled).sum())
        lines = [
            "=" * 70,
            "Dynamic Factor Model with Kalman Smoothing (Doz-Giannone-Reichlin 2011)",
            "=" * 70,
            f"Observations / Time periods   : T = {T}",
            f"Series in panel               : n = {n}",
            f"Common factors (k)            : {k}",
            f"Factor VAR lag order (p)      : {self.p}",
            f"Max transition eigenvalue     : {eig:.4f}",
            f"Gaussian log-likelihood       : {self.loglik:.3f}",
            f"Missing entries filled (NaN)  : {self.n_missing}"
            + (f" (unfilled: {n_filled})" if n_filled else ""),
            "-" * 70,
            f"{'Series':<24} {'Lambda (F1)':>12} {'Mean':>12} {'Std':>12}",
            "-" * 70,
        ]
        names = self._names()
        for idx_col, name in enumerate(names):
            lam1 = self.loadings[idx_col, 0] if k > 0 else 0.0
            m = self.means[idx_col] if idx_col < len(self.means) else 0.0
            s = self.stds[idx_col] if idx_col < len(self.stds) else 1.0
            lines.append(f"{name:<24s} {lam1:+12.4f} {m:>12.4f} {s:>12.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Loadings matrix table (n series x k factors)."""
        k = self.loadings.shape[1]
        return pd.DataFrame(
            self.loadings,
            index=pd.Index(self._names(), name="series"),
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

    def plot(self, *, ax: Any = None, title: str = "Smoothed Dynamic Factors") -> Any:
        """Plot smoothed factor trajectories."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        F_df = self.factors_df if not self.factors_df.empty else pd.DataFrame(
            self.factors, columns=[f"Factor {i + 1}" for i in range(self.factors.shape[1])]
        )
        for col in F_df.columns:
            ax.plot(F_df.index, F_df[col], lw=1.8, label=str(col))

        ax.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Period")
        ax.set_ylabel("Factor Value")
        ax.legend(loc="best", frameon=True, fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        return fig

    def nowcast(self, variable: str | int = 0, period: Any = -1) -> float:
        """Extract the model-implied nowcast for a specific variable at a given period.

        Parameters
        ----------
        variable : str | int, default 0
            Variable name or column index.
        period : Any, default -1
            Period index label or integer row offset.

        Returns
        -------
        float
            The filled / nowcasted value.
        """
        names = self._names()
        if isinstance(variable, str):
            if variable not in names:
                raise KeyError(f"Variable {variable!r} not found in model columns: {names}")
            col_idx = names.index(variable)
        else:
            col_idx = int(variable)

        if isinstance(period, int) and (self.index is None or period < 0 or period >= len(self.factors)):
            row_idx = period if period >= 0 else len(self.factors) + period
        elif self.index is not None and period in self.index:
            row_idx = list(self.index).index(period)
        else:
            row_idx = -1

        return float(self.X_filled[row_idx, col_idx])

    def predict(self, steps: int = 1) -> pd.DataFrame:
        """Forecast dynamic factors and observations forward using companion VAR.

        Parameters
        ----------
        steps : int, default 1
            Forecast horizon (number of periods ahead).

        Returns
        -------
        pd.DataFrame
            Predicted values for all observable series over the forecast horizon.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        k = self.n_factors
        p = self.p
        last_state = self.smoother_out.get("a_smooth", np.zeros((len(self.factors), k * p)))[-1].copy()

        state_forecasts = []
        curr_state = last_state
        for _ in range(steps):
            curr_state = self.A @ curr_state
            state_forecasts.append(curr_state[:k])

        factors_pred = np.vstack(state_forecasts)  # (steps, k)
        Xs_pred = factors_pred @ self.loadings.T   # (steps, n)
        X_pred = Xs_pred * self.stds + self.means

        if self.index is not None and len(self.index) > 0 and hasattr(self.index[0], "year"):
            # Try date continuation
            try:
                dt_idx = pd.to_datetime(self.index)
                freq = pd.infer_freq(dt_idx) or "MS"
                future_dates = pd.date_range(dt_idx[-1], periods=steps + 1, freq=freq)[1:]
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                future_dates = [f"t+{h}" for h in range(1, steps + 1)]
        else:
            future_dates = [f"t+{h}" for h in range(1, steps + 1)]

        return pd.DataFrame(X_pred, index=future_dates, columns=self._names())


class DynamicFactorModel:
    """Object-oriented Dynamic Factor Model with Kalman smoothing (Doz-Giannone-Reichlin 2011).

    Parameters
    ----------
    n_factors : int, default 1
        Number of common latent dynamic factors.
    p : int, default 1
        Autoregressive lag length for factor VAR transition.
    standardize : bool, default True
        Whether to standardize (Z-score) variables prior to factor estimation.
    diffuse_scale : float, default 1e6
        Scale of diagonal state prior covariance matrix (approximate diffuse prior).
    """

    def __init__(
        self,
        n_factors: int = 1,
        p: int = 1,
        standardize: bool = True,
        diffuse_scale: float = 1e6,
    ) -> None:
        if n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {n_factors}")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        self.n_factors = n_factors
        self.p = p
        self.standardize = standardize
        self.diffuse_scale = diffuse_scale
        self.result_: Optional[DynamicFactorModelResult] = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> DynamicFactorModel:
        """Fit the Dynamic Factor Model on a panel with potentially ragged edges.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Observed macroeconomic panel (T, n). Missing entries (NaNs) are handled
            natively via the Kalman smoother.

        Returns
        -------
        DynamicFactorModel
            Self instance with fitted ``result_`` attribute.
        """
        if isinstance(X, pd.DataFrame):
            idx = tuple(X.index)
            cols = tuple(str(c) for c in X.columns)
            X_arr = X.values.astype(float)
        else:
            idx = None
            cols = tuple(f"x{i}" for i in range(X.shape[1]))
            X_arr = np.asarray(X, dtype=float)

        T, n = X_arr.shape

        # 1) Standardization
        means = np.nanmean(X_arr, axis=0) if self.standardize else np.zeros(n)
        stds = np.nanstd(X_arr, axis=0, ddof=0) if self.standardize else np.ones(n)
        stds = np.where(stds < 1e-12, 1.0, stds)
        Xs = (X_arr - means) / stds if self.standardize else X_arr.copy()

        # 2) PCA on complete cases
        complete = ~np.isnan(Xs).any(axis=1)
        if complete.sum() <= max(self.n_factors, self.p) + 2:
            raise ValueError(
                f"too few complete-cases rows ({int(complete.sum())}) for "
                f"n_factors={self.n_factors}, p={self.p}"
            )
        pca = pca_factors(Xs[complete], k=self.n_factors, demean=False, standardize=False)
        F_init = pca["factors"]
        Lambda = pca["loadings"]

        # Idiosyncratic covariance
        e = Xs[complete] - F_init @ Lambda.T
        H_diag = np.maximum(np.var(e, axis=0, ddof=0), 1e-6)
        H = np.diag(H_diag)

        # 3) VAR(p) on factors
        A, Q = _var_ols(F_init, self.p)
        k = self.n_factors
        Z = np.zeros((n, k * self.p))
        Z[:, :k] = Lambda

        # 4) State-space model + RTS smoother
        ssm = StateSpaceModel(T=A, Z=Z, Q=Q, H=H)
        sm = kalman_smoother(Xs, ssm, diffuse_scale=self.diffuse_scale)
        a_smooth = sm["a_smooth"]
        factors = a_smooth[:, :k]

        # 5) Model-implied predictions
        X_implied = factors @ Lambda.T
        X_filled = np.where(np.isnan(Xs), X_implied, Xs)
        if self.standardize:
            X_filled = X_filled * stds + means

        idx_pd = pd.Index(idx) if idx is not None else None
        factors_df = pd.DataFrame(
            factors,
            index=idx_pd,
            columns=[f"F{i + 1}" for i in range(k)],
        )
        X_filled_df = pd.DataFrame(X_filled, index=idx_pd, columns=list(cols))

        self.result_ = DynamicFactorModelResult(
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
            n_factors=self.n_factors,
            p=self.p,
            columns=cols,
            index=idx,
            factors_df=factors_df,
            X_filled_df=X_filled_df,
            ssm=ssm,
            smoother_out=sm,
        )
        return self

    @property
    def factors(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before accessing factors.")
        return self.result_.factors

    @property
    def loadings(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before accessing loadings.")
        return self.result_.loadings

    @property
    def loglik(self) -> float:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before accessing loglik.")
        return self.result_.loglik

    def summary(self) -> str:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling summary().")
        return self.result_.summary()

    def to_frame(self) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling to_frame().")
        return self.result_.to_frame()

    def to_markdown(self, **kwargs: Any) -> str:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling to_markdown().")
        return self.result_.to_markdown(**kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling to_latex().")
        return self.result_.to_latex(**kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling to_typst().")
        return self.result_.to_typst(**kwargs)

    def plot(self, *, ax: Any = None, title: str = "Smoothed Dynamic Factors") -> Any:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling plot().")
        return self.result_.plot(ax=ax, title=title)

    def nowcast(self, variable: str | int = 0, period: Any = -1) -> float:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling nowcast().")
        return self.result_.nowcast(variable=variable, period=period)

    def predict(self, steps: int = 1) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.result_.predict(steps=steps)

    def news(
        self,
        old_vintage: Any,
        new_vintage: Any,
        target_series: str | int = 0,
        target_period: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Compute Bańbura & Modugno (2014) news decomposition using this fitted model."""
        from .news import banbura_modugno_news
        return banbura_modugno_news(
            self,
            old_vintage,
            new_vintage,
            target_series=target_series,
            target_period=target_period,
            **kwargs,
        )


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
    """Two-step DFM with Kalman smoothing (Doz, Giannone, Reichlin 2011).

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
        A ``dict`` subclass (all dict operations work) with factors, loadings,
        transition matrices, filled panel, log-likelihood, and presentation methods.
    """
    if isinstance(X, pd.DataFrame):
        idx = X.index
        cols = list(X.columns)
        X_arr = X.values.astype(float)
    else:
        idx = None; cols = None
        X_arr = np.asarray(X, dtype=float)
    T, n = X_arr.shape

    means = np.nanmean(X_arr, axis=0) if standardize else np.zeros(n)
    stds = np.nanstd(X_arr, axis=0, ddof=0) if standardize else np.ones(n)
    stds = np.where(stds < 1e-12, 1.0, stds)
    Xs = (X_arr - means) / stds if standardize else X_arr.copy()

    complete = ~np.isnan(Xs).any(axis=1)
    if complete.sum() <= max(n_factors, p) + 2:
        raise ValueError(
            f"too few complete-cases rows ({int(complete.sum())}) for "
            f"n_factors={n_factors}, p={p}"
        )
    pca = pca_factors(Xs[complete], k=n_factors,
                       demean=False, standardize=False)
    F_init = pca["factors"]
    Lambda = pca["loadings"]

    e = Xs[complete] - F_init @ Lambda.T
    H_diag = np.maximum(np.var(e, axis=0, ddof=0), 1e-6)
    H = np.diag(H_diag)

    A, Q = _var_ols(F_init, p)
    k = n_factors
    Z = np.zeros((n, k * p))
    Z[:, :k] = Lambda

    ssm = StateSpaceModel(T=A, Z=Z, Q=Q, H=H)
    sm = kalman_smoother(Xs, ssm, diffuse_scale=diffuse_scale)
    a_smooth = sm["a_smooth"]
    factors = a_smooth[:, :k]

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


__all__ = [
    "DynamicFactorModel",
    "DynamicFactorModelResult",
    "KalmanDFMResult",
    "kalman_dfm",
]
