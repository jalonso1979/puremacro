"""Double / Debiased Machine Learning (DML) for Partially Linear Models.

Implements Chernozhukov et al. (2018) "Double/debiased machine learning for treatment
and structural parameters", The Econometrics Journal 21(1), C1-C68.

Partially Linear Regression (PLR) model:
    Y = D * theta_0 + g_0(X) + U,   E[U | X, D] = 0
    D = m_0(X) + V,                 E[V | X] = 0

Orthogonal (Neyman / Robinson 1988) score:
    psi(W; theta, eta) = (Y - l(X) - (D - m(X)) * theta) * (D - m(X))
                       = (Y_tilde - D_tilde * theta) * D_tilde
where l_0(X) = E[Y | X] and m_0(X) = E[D | X].

Features:
- Pure-NumPy regularized learners (LassoCoordinateDescent, RidgeGCV).
- K-fold cross-fitting with out-of-fold residualization.
- Root-N asymptotic standard errors and plug-in sandwich variance.
- Frozen dataclass `DMLResult` with full presentation suite
  (.summary, .plot, .to_markdown, .to_latex, .to_typst).
- Zero non-Pyodide runtime dependencies (numpy, scipy, pandas, matplotlib only).
"""
from __future__ import annotations

import copy
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


# ===========================================================================
# Pure-NumPy Regularized Learners (Pyodide Core Compliant)
# ===========================================================================


def _as_design(X: Any) -> np.ndarray:
    """Coerce ``X`` to a 2-D float array; a 1-D ``X`` is read as a single column."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X_arr.shape}.")
    return X_arr


def _validate_xy(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite ``(X, y)`` arrays of matching length or raise a clear ``ValueError``."""
    X_arr = _as_design(X)
    y_arr = np.asarray(y, dtype=float).ravel()
    n, p = X_arr.shape
    if n == 0 or p == 0:
        raise ValueError("X must have at least one sample and one feature.")
    if len(y_arr) != n:
        raise ValueError(f"X has {n} rows but y has {len(y_arr)} entries.")
    if not np.isfinite(X_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("X and y must not contain NaN or inf.")
    return X_arr, y_arr


def _check_penalties(alphas: Any, name: str) -> np.ndarray:
    """Return ``alphas`` as a non-empty 1-D array of finite, non-negative penalties.

    An empty grid used to fail with ``IndexError``, a NaN penalty produced NaN
    coefficients that only surfaced as a ``LinAlgError`` deep inside DML, and a
    negative penalty was accepted silently.
    """
    arr = np.atleast_1d(np.asarray(alphas, dtype=float)).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one penalty value.")
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{name} must be finite and non-negative, got {alphas!r}.")
    return arr


def _warn_stacklevel() -> int:
    """``stacklevel`` for :func:`warnings.warn` that points at the first frame
    outside this module, so a warning raised inside :meth:`DoubleMLPLR.fit` is
    attributed to the user's call whether it went through :func:`dml_plr` or not."""
    level = 1  # stacklevel=1 is the frame that calls ``warnings.warn``
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


class LassoCoordinateDescent:
    """Pure-NumPy Lasso (L1-regularized linear regression) via cyclical coordinate descent.

    Solves:
        min_beta  (1 / (2*N)) * ||y - X*beta||_2^2 + alpha * ||beta||_1

    With ``fit_intercept=True`` (default) the columns of ``X`` are centred and
    scaled to unit variance before the penalty is applied (the reported
    ``coef_`` is on the original scale); with ``fit_intercept=False`` the
    penalty acts on the raw columns, whatever their scale.

    Supports warm restarts along a decreasing regularization path and model
    selection via BIC or AIC (``criterion``).
    """

    def __init__(
        self,
        alpha: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        n_alphas: int = 50,
        eps: float = 1e-3,
        criterion: str = "bic",
        fit_intercept: bool = True,
    ) -> None:
        if alpha is not None:
            if np.ndim(alpha) != 0:
                raise ValueError(f"alpha must be a scalar penalty or None, got {alpha!r}.")
            _check_penalties(alpha, "alpha")
        if int(n_alphas) < 1:
            raise ValueError(f"n_alphas must be a positive integer, got {n_alphas!r}.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_alphas = int(n_alphas)
        self.eps = eps
        crit = str(criterion).lower()
        if crit not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}.")
        self.criterion = crit
        self.fit_intercept = fit_intercept

        # Fitted attributes
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.alpha_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoCoordinateDescent":
        X_arr, y_arr = _validate_xy(X, y)
        n, p = X_arr.shape

        # Center and standardize
        if self.fit_intercept:
            x_mean = np.mean(X_arr, axis=0)
            x_std = np.std(X_arr, axis=0)
            constant = x_std < 1e-12
            x_std[constant] = 1.0
            y_mean = float(np.mean(y_arr))
            Xs = (X_arr - x_mean) / x_std
            # A (near-)constant column carries no information once centred. Zero it
            # so its Gram entry below is exactly 0 and the coordinate stays at 0;
            # otherwise the exact update divides by a ~1e-28 Gram entry and, at
            # alpha = 0, blows the coefficient up to ~1e12.
            Xs[:, constant] = 0.0
            yc = y_arr - y_mean
        else:
            x_mean = np.zeros(p)
            x_std = np.ones(p)
            y_mean = 0.0
            Xs = X_arr.copy()
            yc = y_arr.copy()

        # Regularization path
        if self.alpha is not None:
            alphas = [float(_check_penalties(self.alpha, "alpha")[0])]
        else:
            # Maximum alpha where all coefficients are zero: max_j |X_j' y| / n
            alpha_max = float(np.max(np.abs(Xs.T @ yc)) / n)
            if alpha_max < 1e-12:
                alpha_max = 1e-3
            alpha_min = max(alpha_max * self.eps, 1e-7)
            alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), self.n_alphas)

        best_score = float("inf")
        best_beta_s = np.zeros(p)
        best_alpha = alphas[0]

        beta_s = np.zeros(p)

        # Per-column Gram entries Xs[:, j]' Xs[:, j] / n. These equal 1.0 for the
        # standardized design (fit_intercept=True) but are arbitrary for raw
        # columns (fit_intercept=False); the exact coordinate minimizer is
        # beta_j = S(rho_j, a) / c_j, so treating c_j as 1 turns the update into
        # a gradient step of size c_j that diverges for c_j > 2.
        col_norm2 = np.einsum("ij,ij->j", Xs, Xs) / n

        for a in alphas:
            # Cyclical coordinate descent with warm start
            r = yc - Xs @ beta_s
            for _ in range(self.max_iter):
                max_change = 0.0
                for j in range(p):
                    c_j = col_norm2[j]
                    if c_j <= 0.0:
                        # All-zero column: the coefficient is unidentified, keep it at 0.
                        continue
                    old_bj = beta_s[j]
                    # Gradient of the smooth part at beta_j = 0 with the other
                    # coordinates fixed: Xs_j' (r + Xs_j * old_bj) / n
                    rho_j = float(Xs[:, j] @ r / n) + c_j * old_bj
                    # Soft thresholding
                    if rho_j > a:
                        new_bj = (rho_j - a) / c_j
                    elif rho_j < -a:
                        new_bj = (rho_j + a) / c_j
                    else:
                        new_bj = 0.0

                    delta = new_bj - old_bj
                    if delta != 0.0:
                        beta_s[j] = new_bj
                        r -= delta * Xs[:, j]
                        change = abs(delta)
                        if change > max_change:
                            max_change = change

                if max_change < self.tol:
                    break

            # Model evaluation (BIC / AIC)
            mse = float(np.mean(r ** 2))
            df = int(np.count_nonzero(beta_s))
            if self.criterion == "aic":
                score = n * np.log(mse + 1e-14) + 2.0 * df
            else:  # default BIC
                score = n * np.log(mse + 1e-14) + df * np.log(n)

            if score < best_score:
                best_score = score
                best_beta_s = beta_s.copy()
                best_alpha = a

        self.alpha_ = best_alpha
        # Unstandardize coefficients
        self.coef_ = best_beta_s / x_std
        if not np.isfinite(self.coef_).all():
            raise RuntimeError(
                "LassoCoordinateDescent produced non-finite coefficients; "
                "the design is numerically degenerate."
            )
        if self.fit_intercept:
            self.intercept_ = y_mean - float(x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LassoCoordinateDescent is not fitted yet.")
        return _as_design(X) @ self.coef_ + self.intercept_


class RidgeGCV:
    """Pure-NumPy Ridge Regression with Generalized Cross-Validation (GCV).

    Solves, on the centred and unit-variance-scaled columns of ``X`` when
    ``fit_intercept=True`` (the default) and on the raw columns otherwise:
        min_beta  ||y - X*beta||_2^2 + alpha * ||beta||_2^2

    i.e. ``beta = (X'X + alpha I)^{-1} X'y`` with the penalty in the same
    absolute units as ``sklearn.linear_model.Ridge``; it is *not* divided by
    ``N``. ``alpha_`` and any user-supplied ``alphas`` grid are therefore
    absolute, and the default grid ``np.logspace(-4, 6, 100)`` is chosen for
    standardized columns. The reported ``coef_`` is on the original scale.

    Uses SVD decomposition of X for O(p) evaluation of the GCV criterion
    across a geometric grid of penalty parameters.
    """

    def __init__(
        self,
        alphas: Sequence[float] | np.ndarray | None = None,
        fit_intercept: bool = True,
    ) -> None:
        if alphas is not None:
            _check_penalties(alphas, "alphas")
        self.alphas = alphas
        self.fit_intercept = fit_intercept

        # Fitted attributes
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.alpha_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeGCV":
        X_arr, y_arr = _validate_xy(X, y)
        n, p = X_arr.shape

        # Center and standardize
        if self.fit_intercept:
            x_mean = np.mean(X_arr, axis=0)
            x_std = np.std(X_arr, axis=0)
            x_std[x_std < 1e-12] = 1.0
            y_mean = float(np.mean(y_arr))
            Xs = (X_arr - x_mean) / x_std
            yc = y_arr - y_mean
        else:
            x_mean = np.zeros(p)
            x_std = np.ones(p)
            y_mean = 0.0
            Xs = X_arr.copy()
            yc = y_arr.copy()

        # Economy SVD: Xs = U S V^T
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        d = U.T @ yc
        k = len(S)

        alphas = (
            _check_penalties(self.alphas, "alphas")
            if self.alphas is not None
            else np.logspace(-4, 6, 100)
        )

        best_gcv = float("inf")
        best_alpha = float(alphas[0])

        # Sum of squared residuals outside projection space: ||yc - U U' yc||^2
        rss_base = float(np.sum((yc - U @ d) ** 2))

        for a in alphas:
            # Under ridge penalty a (with scaling matching ||y - Xb||^2 + a ||b||^2):
            denom = S ** 2 + a
            fitted_in_U = (S ** 2 / denom) * d
            tr_H = float(np.sum(S ** 2 / denom))
            denom_gcv = max(1.0 - tr_H / n, 1e-6) ** 2

            rss_in_U = float(np.sum((d - fitted_in_U) ** 2))
            rss = rss_base + rss_in_U
            gcv = (rss / n) / denom_gcv

            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(a)

        self.alpha_ = best_alpha
        # Compute optimal beta in standardized space: V (S / (S^2 + alpha)) d
        beta_s = Vt.T @ ((S / (S ** 2 + best_alpha)) * d)

        # Unstandardize
        self.coef_ = beta_s / x_std
        if self.fit_intercept:
            self.intercept_ = y_mean - float(x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeGCV is not fitted yet.")
        return _as_design(X) @ self.coef_ + self.intercept_


class LogisticCoordinateDescent:
    """Pure-NumPy regularized logistic regression via coordinate descent with surrogate upper bounds.

    Solves:
        min_{beta_0, beta}  - (1/N) * sum_{i=1}^N [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
                            + alpha * ( l1_ratio * ||beta||_1 + (1 - l1_ratio) / 2 * ||beta||_2^2 )

    where p_i = 1 / (1 + exp(-(beta_0 + x_i' beta))).

    Uses the majorization-minimization quadratic surrogate upper bound (Böhning & Lindsay 1988),
    exploiting p*(1-p) <= 1/4 to achieve monotonic descent without step-size backtracking.

    Supports warm starts along a geometric penalty path and model selection via AIC or BIC.
    """

    def __init__(
        self,
        penalty: str = "l1",
        C: float | None = None,
        alpha: float | None = None,
        l1_ratio: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        n_alphas: int = 50,
        eps: float = 1e-3,
        criterion: str = "bic",
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        pen = str(penalty).lower()
        if pen in ("l1", "lasso"):
            ratio = 1.0 if l1_ratio is None else float(l1_ratio)
        elif pen in ("l2", "ridge"):
            ratio = 0.0 if l1_ratio is None else float(l1_ratio)
        elif pen in ("elasticnet", "elastic-net"):
            ratio = 0.5 if l1_ratio is None else float(l1_ratio)
        elif pen in ("none", "unpenalized"):
            ratio = 0.0
            alpha = 0.0
        else:
            raise ValueError(f"Unknown penalty {penalty!r}. Supported: 'l1', 'l2', 'elasticnet', 'none'.")

        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"l1_ratio must lie in [0, 1], got {ratio!r}.")

        if alpha is not None:
            if np.ndim(alpha) != 0:
                raise ValueError(f"alpha must be a scalar penalty or None, got {alpha!r}.")
            _check_penalties(alpha, "alpha")
        if C is not None:
            if float(C) <= 0.0:
                raise ValueError(f"C must be strictly positive, got {C!r}.")
        if int(n_alphas) < 1:
            raise ValueError(f"n_alphas must be a positive integer, got {n_alphas!r}.")
        crit = str(criterion).lower()
        if crit not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}.")

        self.penalty = pen
        self.C = C
        self.alpha = alpha
        self.l1_ratio = ratio
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_alphas = int(n_alphas)
        self.eps = float(eps)
        self.criterion = crit
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Fitted attributes
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.alpha_: float | None = None
        self.alphas_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None
        self.classes_: np.ndarray = np.array([0, 1])

    def fit(self, X: Any, y: Any) -> "LogisticCoordinateDescent":
        X_arr, y_arr = _validate_xy(X, y)
        n, p = X_arr.shape

        unique_y = np.unique(y_arr)
        if not np.isin(unique_y, [0.0, 1.0]).all():
            raise ValueError(
                f"LogisticCoordinateDescent requires binary targets in {{0, 1}}, got {unique_y.tolist()}."
            )

        mean_y = float(np.mean(y_arr))
        if mean_y == 0.0 or mean_y == 1.0:
            self.coef_ = np.zeros(p)
            self.intercept_ = -18.0 if mean_y == 0.0 else 18.0
            self.alpha_ = 0.0
            self.alphas_ = np.array([0.0])
            self.scores_ = np.array([0.0])
            return self

        if self.fit_intercept:
            x_mean = np.mean(X_arr, axis=0)
            x_std = np.std(X_arr, axis=0)
            constant = x_std < 1e-12
            x_std[constant] = 1.0
            Xs = (X_arr - x_mean) / x_std
            Xs[:, constant] = 0.0
        else:
            x_mean = np.zeros(p)
            x_std = np.ones(p)
            Xs = X_arr.copy()

        col_norm2 = np.einsum("ij,ij->j", Xs, Xs) / n
        c_j = col_norm2 / 4.0
        c_0 = 0.25

        if self.alpha is not None:
            alphas = np.array([float(self.alpha)])
        elif self.C is not None:
            alphas = np.array([1.0 / float(self.C)])
        else:
            y_bar = np.clip(mean_y, 1e-4, 1.0 - 1e-4)
            g_init = Xs.T @ (np.full(n, y_bar) - y_arr) / n
            ratio_eff = max(self.l1_ratio, 1e-3)
            alpha_max = float(np.max(np.abs(g_init))) / ratio_eff
            if alpha_max < 1e-6:
                alpha_max = 1e-3
            alpha_min = max(alpha_max * self.eps, 1e-7)
            alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), self.n_alphas)

        y_bar = np.clip(mean_y, 1e-4, 1.0 - 1e-4)
        b0 = float(np.log(y_bar / (1.0 - y_bar))) if self.fit_intercept else 0.0
        beta_s = np.zeros(p)
        eta = np.full(n, b0) if self.fit_intercept else np.zeros(n)
        p_cur = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))

        best_score = float("inf")
        best_beta_s = beta_s.copy()
        best_b0 = b0
        best_alpha = alphas[0]
        scores_list = []

        for a in alphas:
            l1_pen = a * self.l1_ratio
            l2_pen = a * (1.0 - self.l1_ratio)

            for _ in range(self.max_iter):
                max_change = 0.0

                if self.fit_intercept:
                    g0 = float(np.mean(p_cur - y_arr))
                    d0 = - g0 / c_0
                    if abs(d0) > 1e-12:
                        b0 += d0
                        eta += d0
                        p_cur = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
                        if abs(d0) > max_change:
                            max_change = abs(d0)

                for j in range(p):
                    cj = c_j[j]
                    if cj <= 0.0:
                        continue
                    gj = float(np.dot(Xs[:, j], p_cur - y_arr)) / n
                    old_bj = beta_s[j]
                    rho_j = cj * old_bj - gj
                    denom = cj + l2_pen

                    if rho_j > l1_pen:
                        new_bj = (rho_j - l1_pen) / denom
                    elif rho_j < -l1_pen:
                        new_bj = (rho_j + l1_pen) / denom
                    else:
                        new_bj = 0.0

                    diff = new_bj - old_bj
                    if abs(diff) > 1e-12:
                        beta_s[j] = new_bj
                        eta += diff * Xs[:, j]
                        p_cur = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
                        change = abs(diff)
                        if change > max_change:
                            max_change = change

                if max_change < self.tol:
                    break

            p_safe = np.clip(p_cur, 1e-15, 1.0 - 1e-15)
            dev = -2.0 * float(np.sum(y_arr * np.log(p_safe) + (1.0 - y_arr) * np.log(1.0 - p_safe)))
            df = (1 if self.fit_intercept else 0) + int(np.count_nonzero(np.abs(beta_s) > 1e-6))
            if self.criterion == "aic":
                score = dev + 2.0 * df
            else:
                score = dev + np.log(n) * df

            scores_list.append(score)
            if score < best_score:
                best_score = score
                best_beta_s = beta_s.copy()
                best_b0 = b0
                best_alpha = a

        self.alpha_ = best_alpha
        self.alphas_ = alphas
        self.scores_ = np.array(scores_list)
        self.coef_ = best_beta_s / x_std
        if not np.isfinite(self.coef_).all():
            raise RuntimeError("LogisticCoordinateDescent produced non-finite coefficients.")
        if self.fit_intercept:
            self.intercept_ = best_b0 - float(x_mean @ self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticCoordinateDescent is not fitted yet.")
        X_arr = _as_design(X)
        eta = X_arr @ self.coef_ + self.intercept_
        p1 = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)), np.exp(eta) / (1.0 + np.exp(eta)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticCoordinateDescent is not fitted yet.")
        return _as_design(X) @ self.coef_ + self.intercept_


def _is_learner_instance(learner: Any) -> bool:
    """True for an already-constructed learner object (has ``fit``/``predict``, is not a class)."""
    return (
        not isinstance(learner, (str, type))
        and hasattr(learner, "fit")
        and (hasattr(learner, "predict") or hasattr(learner, "predict_proba"))
    )


def _learner_name(learner: str | Any) -> str:
    """Human-readable learner label for ``DMLResult.learner``."""
    if isinstance(learner, str):
        return learner
    if isinstance(learner, type):
        return learner.__name__
    if _is_learner_instance(learner):
        return type(learner).__name__
    return getattr(learner, "__name__", type(learner).__name__)


def _get_learner(learner: str | Any, **kwargs: Any) -> Any:
    """Return a fresh, unfitted learner for one nuisance regression.

    ``learner`` may be a string alias, a class or factory callable (called with
    ``kwargs``), or an already-constructed instance. An instance is deep-copied
    so that each nuisance model (l(X), every m_j(X), every fold) gets its own
    object and the caller's instance is never fitted in place.
    """
    if isinstance(learner, str):
        key = learner.lower()
        if key in ("lasso", "l1"):
            return LassoCoordinateDescent(**kwargs)
        elif key in ("ridge", "l2"):
            return RidgeGCV(**kwargs)
        elif key in ("logistic", "logit", "logistic_cd", "logistic_coordinatedescent"):
            return LogisticCoordinateDescent(**kwargs)
        else:
            raise ValueError(f"Unknown learner {learner!r}. Supported: 'lasso', 'ridge', 'logistic'.")
    if _is_learner_instance(learner):
        if kwargs:
            raise ValueError(
                "learner_kwargs cannot be combined with a learner instance "
                f"({type(learner).__name__}); configure the instance directly or "
                "pass the learner class / string alias instead."
            )
        return copy.deepcopy(learner)
    if callable(learner):
        obj = learner(**kwargs)
        if not (hasattr(obj, "fit") and (hasattr(obj, "predict") or hasattr(obj, "predict_proba"))):
            raise TypeError(
                f"learner {learner!r} returned {type(obj).__name__}, which has no fit/predict methods."
            )
        return obj
    raise TypeError(
        f"learner must be a string alias, a class, or an object with fit/predict; got {type(learner).__name__}."
    )


# ===========================================================================
# DML Result Container (Frozen Dataclass with Full Presentation Suite)
# ===========================================================================


@dataclass(frozen=True)
class DMLResult:
    """Results container for Double Machine Learning Partially Linear Regression (DML-PLR).

    Attributes
    ----------
    theta : float | np.ndarray
        Debiased causal parameter estimate(s). Float for single treatment, array for multiple.
    se : float | np.ndarray
        Root-N asymptotic standard error(s).
    t_stat : float | np.ndarray
        Asymptotic z / t test statistic(s).
    p_value : float | np.ndarray
        Two-sided p-value(s).
    ci_lower : float | np.ndarray
        Lower bound(s) of asymptotic confidence interval.
    ci_upper : float | np.ndarray
        Upper bound(s) of asymptotic confidence interval.
    n_obs : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    learner : str
        Learner used for nuisance function estimation.
    residuals_y : np.ndarray
        Out-of-fold outcome residuals Y - l_hat(X).
    residuals_d : np.ndarray
        Out-of-fold treatment residuals D - m_hat(X).
    ci_level : float
        Confidence level, e.g. 0.95.
    treatment_names : tuple[str, ...]
        Name(s) of treatment variable(s).
    outcome_name : str
        Name of outcome variable.
    """

    theta: float | np.ndarray
    se: float | np.ndarray
    t_stat: float | np.ndarray
    p_value: float | np.ndarray
    ci_lower: float | np.ndarray
    ci_upper: float | np.ndarray
    n_obs: int
    n_folds: int
    learner: str
    residuals_y: np.ndarray
    residuals_d: np.ndarray
    ci_level: float = 0.95
    treatment_names: tuple[str, ...] = ("D",)
    outcome_name: str = "Y"

    def summary(self) -> str:
        """Text summary of Double Machine Learning estimation results."""
        pct = int(round(self.ci_level * 100))
        lines = [
            "=" * 78,
            "Double / Debiased Machine Learning (DML-PLR)",
            "Model: Partially Linear Regression (Robinson 1988 / Chernozhukov et al. 2018)",
            "=" * 78,
            f"Outcome: {self.outcome_name:<18} Observations: {self.n_obs:<12} Folds: {self.n_folds}",
            f"Learner: {self.learner:<18} Score: Robinson Orthogonal (DML2)",
            "-" * 78,
            f"{'Variable':<16} {'Coef.':>10} {'Std.Err.':>10} {'z':>8} {'P>|z|':>8} "
            f"[{pct}% Conf. Interval]",
            "-" * 78,
        ]

        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
            lines.append(
                f"{name:<16} {th:>10.4f} {s:>10.4f} {t_val:>8.3f} {p_str:>8} "
                f"{lo:>10.4f} {hi:>10.4f}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(self, kind: str = "forest", ax: Any = None, **kwargs: Any) -> Any:
        """Plot DML estimation results.

        Parameters
        ----------
        kind : {'forest', 'residuals'}, default 'forest'
            - 'forest': Forest plot of treatment coefficient estimates and confidence intervals.
            - 'residuals': Scatter plot of out-of-fold residuals D_tilde vs Y_tilde with estimated slope.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))
        else:
            fig = ax.figure

        if kind == "residuals":
            res_y = self.residuals_y
            res_d = self.residuals_d
            if res_d.ndim > 1 and res_d.shape[1] > 1:
                res_d = res_d[:, 0]
            th = float(np.atleast_1d(self.theta)[0])
            name = self.treatment_names[0]

            ax.scatter(res_d, res_y, alpha=kwargs.get("alpha", 0.4), color="steelblue", s=18, label="Out-of-fold Residuals")
            grid_d = np.linspace(res_d.min(), res_d.max(), 100)
            ax.plot(grid_d, th * grid_d, color="firebrick", lw=2, label=f"DML Slope: {th:.4f}")
            ax.set_xlabel(f"Residualized Treatment: $\\tilde{{{name}}}$")
            ax.set_ylabel(f"Residualized Outcome: $\\tilde{{{self.outcome_name}}}$")
            ax.set_title("DML-PLR Orthogonal Residuals & Estimated Effect")
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3, ls=":")
        else:  # 'forest'
            thetas = np.atleast_1d(self.theta)
            los = np.atleast_1d(self.ci_lower)
            his = np.atleast_1d(self.ci_upper)
            k = len(thetas)
            y_pos = np.arange(k)

            ax.errorbar(
                thetas,
                y_pos,
                xerr=[thetas - los, his - thetas],
                fmt="o",
                color="navy",
                ecolor="steelblue",
                elinewidth=2,
                capsize=4,
                capthick=1.5,
                markersize=6,
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.treatment_names)
            pct = int(round(self.ci_level * 100))
            ax.set_xlabel(f"Treatment Effect $\\theta$ ({pct}% CI)")
            ax.set_title(f"Double Machine Learning Point Estimates ({self.learner})")
            ax.grid(True, alpha=0.3, ls=":")

        return ax

    def to_markdown(self) -> str:
        """Export results to a GitHub-flavoured Markdown table.

        Pipes inside cells (the ``P>|z|`` header, any ``|`` in a variable
        name) are escaped as ``\\|`` so the header and delimiter rows have the
        same cell count and renderers recognise the table.
        """
        pct = int(round(self.ci_level * 100))
        lines = [
            f"### DML-PLR: {self.outcome_name}",
            f"*Learner: {self.learner} | N: {self.n_obs} | Folds: {self.n_folds}*",
            "",
            f"| Variable | Coef. | Std.Err. | z | P>\\|z\\| | [{pct}% Conf. Interval] |",
            "|:---|---:|---:|---:|---:|:---:|",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
            cell_name = str(name).replace("|", "\\|")
            lines.append(
                f"| {cell_name} | {th:.4f} | {s:.4f} | {t_val:.3f} | {p_str} | [{lo:.4f}, {hi:.4f}] |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Export results to a LaTeX ``table`` that needs only the ``booktabs`` package.

        The p-value cell is set in math mode (``$<0.001$``) because a bare
        ``<`` in OT1 text mode prints as an inverted exclamation mark, and the
        sample-size note is a ``\\multicolumn`` row inside the tabular rather
        than a ``\\subcaption`` (which needs the ``subcaption`` package).
        """
        from puremacro.reports import latex_escape

        pct = int(round(self.ci_level * 100))
        n_cols = 6
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Double Machine Learning Estimates for {latex_escape(self.outcome_name)}}}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            f"Variable & Coef. & Std. Err. & $z$ & $P>|z|$ & [{pct}\\% CI] \\\\",
            r"\midrule",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "$<0.001$" if p_val < 0.001 else f"{p_val:.4f}"
            clean_name = latex_escape(name)
            lines.append(
                f"{clean_name} & {th:.4f} & {s:.4f} & {t_val:.3f} & {p_str} & [{lo:.4f}, {hi:.4f}] \\\\"
            )
        note = (
            f"Observations: {self.n_obs}; Folds: {self.n_folds}; "
            f"Learner: {latex_escape(self.learner)}."
        )
        lines.extend([
            r"\bottomrule",
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\footnotesize {note}}} \\\\",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_typst(self) -> str:
        """Export results to a Typst ``#figure(table(...))``.

        Variable and learner names and the ``<0.001`` p-value cell are passed
        through :func:`puremacro.reports.typst_escape`, like every other
        result object's Typst table, so characters Typst reads as markup
        inside a content block (``_``, ``$``, ``#``, ``[``, ``<`` ...) are
        rendered literally.
        """
        from puremacro.reports import typst_escape

        pct = int(round(self.ci_level * 100))
        lines = [
            f"#figure(",
            f"  table(",
            f"    columns: (2fr, 1.2fr, 1.2fr, 1fr, 1fr, 2fr),",
            f"    align: (left, right, right, right, right, center),",
            f"    stroke: none,",
            f"    table.hline(),",
            f"    [*Variable*], [*Coef.*], [*Std.Err.*], [*z*], [*P>|z|*], [*{pct}% CI*],",
            f"    table.hline(stroke: 0.5pt),",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = typst_escape("<0.001") if p_val < 0.001 else f"{p_val:.4f}"
            lines.append(
                f"    [{typst_escape(name)}], [{th:.4f}], [{s:.4f}], [{t_val:.3f}], [{p_str}], [[{lo:.4f}, {hi:.4f}]],"
            )
        lines.extend([
            f"    table.hline(),",
            f"  ),",
            f"  caption: [Double Machine Learning Estimates ({typst_escape(self.learner)}, N={self.n_obs})],",
            f")",
        ])
        return "\n".join(lines)

    def plot_coefficients(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot treatment effect coefficient estimate(s) with confidence intervals."""
        return self.plot(kind="forest", ax=ax, **kwargs)


# ===========================================================================
# DML IRM & IV Result Containers (Frozen Dataclasses)
# ===========================================================================


@dataclass(frozen=True)
class DMLIRMResult:
    """Results container for Double Machine Learning Interactive Regression Model (DML-IRM).

    Attributes
    ----------
    theta : float
        Debiased treatment effect estimate (ATE or ATT).
    se : float
        Root-N asymptotic standard error.
    t_stat : float
        Asymptotic z / t test statistic.
    p_value : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of asymptotic confidence interval.
    ci_upper : float
        Upper bound of asymptotic confidence interval.
    n_obs : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    learner : str
        Learner description used for nuisance function estimation.
    score_type : str
        Target parameter score: 'ATE' or 'ATT'.
    propensity_scores : np.ndarray
        Out-of-fold estimated propensity scores m_hat(X).
    g0_pred : np.ndarray
        Out-of-fold estimated conditional outcome under control g_0(0, X).
    g1_pred : np.ndarray
        Out-of-fold estimated conditional outcome under treatment g_0(1, X).
    treatment_values : np.ndarray
        Treatment vector D.
    outcomes : np.ndarray
        Outcome vector Y.
    trimming_rule : str
        Rule used for overlap trimming: 'clip' or 'drop'.
    trimming_threshold : float
        Trimming boundary epsilon.
    n_trimmed : int
        Number of observations trimmed.
    ci_level : float
        Confidence level, e.g. 0.95.
    treatment_name : str
        Name of treatment variable.
    outcome_name : str
        Name of outcome variable.
    models_g0 : tuple[Any, ...]
        Fitted models for g_0 across folds.
    models_g1 : tuple[Any, ...]
        Fitted models for g_1 across folds.
    models_m : tuple[Any, ...]
        Fitted models for m across folds.
    feature_names : tuple[str, ...]
        Names of control features X.
    """

    theta: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    n_folds: int
    learner: str
    score_type: str
    propensity_scores: np.ndarray
    g0_pred: np.ndarray
    g1_pred: np.ndarray
    treatment_values: np.ndarray
    outcomes: np.ndarray
    trimming_rule: str
    trimming_threshold: float
    n_trimmed: int
    ci_level: float = 0.95
    treatment_name: str = "D"
    outcome_name: str = "Y"
    models_g0: tuple[Any, ...] = ()
    models_g1: tuple[Any, ...] = ()
    models_m: tuple[Any, ...] = ()
    feature_names: tuple[str, ...] = ()

    def summary(self) -> str:
        """Text summary of DML-IRM estimation results."""
        pct = int(round(self.ci_level * 100))
        lines = [
            "=" * 78,
            "Double / Debiased Machine Learning (DML-IRM)",
            f"Model: Interactive Regression Model ({self.score_type})",
            "=" * 78,
            f"Outcome: {self.outcome_name:<18} Observations: {self.n_obs:<12} Folds: {self.n_folds}",
            f"Treatment: {self.treatment_name:<16} Trimmed: {self.n_trimmed} ({self.trimming_rule})  Threshold: {self.trimming_threshold}",
            f"Learner: {self.learner:<18} Score: Doubly Robust {self.score_type}",
            "-" * 78,
            f"{'Target':<16} {'Coef.':>10} {'Std.Err.':>10} {'z':>8} {'P>|z|':>8} "
            f"[{pct}% Conf. Interval]",
            "-" * 78,
        ]
        p_str = "<0.001" if self.p_value < 0.001 else f"{self.p_value:.4f}"
        lines.append(
            f"{self.score_type:<16} {self.theta:>10.4f} {self.se:>10.4f} {self.t_stat:>8.3f} {p_str:>8} "
            f"{self.ci_lower:>10.4f} {self.ci_upper:>10.4f}"
        )
        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(self, kind: str = "overlap", ax: Any = None, **kwargs: Any) -> Any:
        """Plot DML-IRM estimation results.

        Parameters
        ----------
        kind : {'overlap', 'forest', 'coefficients'}, default 'overlap'
            - 'overlap': Propensity score overlap histogram by treatment group.
            - 'forest': Forest plot of the estimated treatment effect and confidence interval.
            - 'coefficients': Regularized coefficients of nuisance models.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        """
        if kind == "forest":
            import matplotlib.pyplot as plt

            if ax is None:
                fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 3)))

            ax.errorbar(
                [self.theta],
                [0],
                xerr=[[self.theta - self.ci_lower], [self.ci_upper - self.theta]],
                fmt="o",
                color="navy",
                ecolor="steelblue",
                elinewidth=2,
                capsize=5,
                capthick=1.5,
                markersize=6,
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks([0])
            ax.set_yticklabels([self.score_type])
            pct = int(round(self.ci_level * 100))
            ax.set_xlabel(f"Treatment Effect ({pct}% CI)")
            ax.set_title(f"DML-IRM {self.score_type} Point Estimate ({self.learner})")
            ax.grid(True, alpha=0.3, ls=":")
            return ax
        elif kind == "coefficients":
            return self.plot_coefficients(ax=ax, **kwargs)
        else:  # 'overlap'
            return self.plot_overlap(ax=ax, **kwargs)

    def plot_overlap(self, ax: Any = None, bins: int = 30, **kwargs: Any) -> Any:
        """Plot propensity score distribution by treatment group with trimming bounds."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        d = np.asarray(self.treatment_values).ravel()
        ps = np.asarray(self.propensity_scores).ravel()
        p_ctrl = ps[d == 0]
        p_treat = ps[d == 1]

        ax.hist(
            p_ctrl,
            bins=bins,
            density=True,
            alpha=kwargs.get("alpha_ctrl", 0.5),
            color=kwargs.get("color_ctrl", "steelblue"),
            label=f"Control ({self.treatment_name}=0, n={len(p_ctrl)})",
        )
        ax.hist(
            p_treat,
            bins=bins,
            density=True,
            alpha=kwargs.get("alpha_treat", 0.5),
            color=kwargs.get("color_treat", "firebrick"),
            label=f"Treated ({self.treatment_name}=1, n={len(p_treat)})",
        )

        eps = float(self.trimming_threshold)
        if eps > 0.0:
            ax.axvline(
                eps,
                color="black",
                linestyle="--",
                lw=1.5,
                label=f"Trimming bounds ({eps:g}, {1.0-eps:g})",
            )
            ax.axvline(1.0 - eps, color="black", linestyle="--", lw=1.5)
            ax.axvspan(0.0, eps, alpha=0.12, color="gray")
            ax.axvspan(1.0 - eps, 1.0, alpha=0.12, color="gray")

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Propensity Score $\hat{m}(X)$")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Propensity Score Overlap ({self.score_type})\n"
            f"Trimmed: {self.n_trimmed}/{self.n_obs} observations ({self.trimming_rule})"
        )
        ax.legend(loc=kwargs.get("legend_loc", "upper center"), frameon=True)
        ax.grid(True, alpha=0.3, ls=":")
        return ax

    def plot_coefficients(
        self, model: str = "treatment", top_k: int = 20, ax: Any = None, **kwargs: Any
    ) -> Any:
        """Plot coefficient estimates with confidence intervals or regularized nuisance coefficients.

        Parameters
        ----------
        model : {'treatment', 'all', 'm', 'g0', 'g1'}, default 'treatment'
            - 'treatment': Treatment effect estimate with asymptotic confidence interval.
            - 'all', 'm', 'g0', 'g1': Regularized nuisance feature coefficients across controls.
        top_k : int, default 20
            Maximum number of top coefficients to display when inspecting nuisance models.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        """
        if int(top_k) < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k!r}")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        if model == "treatment":
            pct = int(round(self.ci_level * 100))
            ax.errorbar(
                [self.theta],
                [0],
                xerr=[[self.theta - self.ci_lower], [self.ci_upper - self.theta]],
                fmt="o",
                color="navy",
                ecolor="steelblue",
                elinewidth=2,
                capsize=5,
                capthick=1.5,
                markersize=7,
                label=f"{self.score_type}: {self.theta:.4f} (SE: {self.se:.4f})",
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks([0])
            ax.set_yticklabels([self.score_type])
            ax.set_xlabel(f"Causal Effect $\\theta$ ({pct}% CI)")
            ax.set_title(f"DML-IRM {self.score_type} Estimate with {pct}% CI")
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3, ls=":")
            return ax

        # Nuisance model coefficient inspection
        m_key = str(model).lower()
        coef_dict: dict[str, np.ndarray] = {}
        if m_key in ("all", "m"):
            coefs = [m.coef_ for m in self.models_m if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["m(X)"] = np.mean(coefs, axis=0)
        if m_key in ("all", "g0"):
            coefs = [m.coef_ for m in self.models_g0 if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["g0(X)"] = np.mean(coefs, axis=0)
        if m_key in ("all", "g1"):
            coefs = [m.coef_ for m in self.models_g1 if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["g1(X)"] = np.mean(coefs, axis=0)

        if not coef_dict:
            ax.text(
                0.5, 0.5, "No nuisance model coefficients available",
                ha="center", va="center", transform=ax.transAxes
            )
            return ax

        p = len(next(iter(coef_dict.values())))
        feat_names = (
            list(self.feature_names)
            if self.feature_names and len(self.feature_names) == p
            else [f"X_{i}" for i in range(p)]
        )

        max_abs = np.zeros(p)
        for c in coef_dict.values():
            max_abs = np.maximum(max_abs, np.abs(c))
        k = min(top_k, p)
        top_idx = np.argsort(max_abs)[-k:]

        y_positions = np.arange(k)
        n_models = len(coef_dict)
        bar_height = 0.8 / max(n_models, 1)
        colors = ["steelblue", "firebrick", "seagreen", "goldenrod"]

        for i, (name, c) in enumerate(coef_dict.items()):
            offsets = y_positions + (i - (n_models - 1) / 2.0) * bar_height
            ax.barh(
                offsets,
                c[top_idx],
                height=bar_height * 0.9,
                color=colors[i % len(colors)],
                label=name,
                alpha=0.85,
            )

        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([feat_names[j] for j in top_idx])
        ax.set_xlabel("Average Regularized Coefficient (across folds)")
        ax.set_title(f"Top {k} Nuisance Feature Coefficients")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3, ls=":")
        return ax

    def plot_tuning(self, model: str = "m", ax: Any = None, **kwargs: Any) -> Any:
        """Plot regularization tuning curve across penalty parameters."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        m_key = str(model).lower()
        if m_key not in ("m", "g0", "g1"):
            raise ValueError(f"model must be one of ('m', 'g0', 'g1'), got {model!r}")
        target_models = self.models_m if m_key == "m" else (self.models_g0 if m_key == "g0" else self.models_g1)

        chosen = None
        for m in target_models:
            if getattr(m, "alphas_", None) is not None and getattr(m, "scores_", None) is not None:
                chosen = m
                break

        if chosen is None or len(getattr(chosen, "alphas_", [])) <= 1:
            ax.text(
                0.5, 0.5, f"No regularization path tuning available for model {model!r}",
                ha="center", va="center", transform=ax.transAxes
            )
            return ax

        crit_label = getattr(chosen, "criterion", "Tuning").upper()
        log_alphas = np.log10(chosen.alphas_)
        ax.plot(
            log_alphas,
            chosen.scores_,
            marker="o",
            markersize=4,
            lw=1.8,
            color="navy",
            label=f"{crit_label} Criterion",
        )
        if chosen.alpha_ is not None and chosen.alpha_ > 0:
            ax.axvline(
                np.log10(chosen.alpha_),
                color="firebrick",
                linestyle="--",
                lw=1.5,
                label=f"Selected $\\alpha^*$ = {chosen.alpha_:.4g}",
            )

        ax.set_xlabel(r"$\log_{10}(\alpha)$")
        ax.set_ylabel(f"{crit_label} Score")
        ax.set_title(f"Regularization Path Tuning ({chosen.__class__.__name__}, model={model})")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3, ls=":")
        return ax

    def to_markdown(self) -> str:
        """Export results to a GitHub-flavoured Markdown table."""
        pct = int(round(self.ci_level * 100))
        lines = [
            f"### DML-IRM ({self.score_type}): {self.outcome_name}",
            f"*Learner: {self.learner} | N: {self.n_obs} | Folds: {self.n_folds} | Trimmed: {self.n_trimmed} ({self.trimming_rule})*",
            "",
            f"| Target | Coef. | Std.Err. | z | P>\\|z\\| | [{pct}% Conf. Interval] |",
            "|:---|---:|---:|---:|---:|:---:|",
        ]
        p_str = "<0.001" if self.p_value < 0.001 else f"{self.p_value:.4f}"
        lines.append(
            f"| {self.score_type} | {self.theta:.4f} | {self.se:.4f} | {self.t_stat:.3f} | {p_str} | [{self.ci_lower:.4f}, {self.ci_upper:.4f}] |"
        )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Export results to a LaTeX table with booktabs."""
        from puremacro.reports import latex_escape

        pct = int(round(self.ci_level * 100))
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Double Machine Learning IRM Estimates for {latex_escape(self.outcome_name)}}}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            f"Target & Coef. & Std. Err. & $z$ & $P>|z|$ & [{pct}\\% CI] \\\\",
            r"\midrule",
        ]
        p_str = "$<0.001$" if self.p_value < 0.001 else f"{self.p_value:.4f}"
        lines.append(
            f"{self.score_type} & {self.theta:.4f} & {self.se:.4f} & {self.t_stat:.3f} & {p_str} & [{self.ci_lower:.4f}, {self.ci_upper:.4f}] \\\\"
        )
        note = (
            f"Observations: {self.n_obs}; Folds: {self.n_folds}; "
            f"Score: {self.score_type}; Trimmed: {self.n_trimmed} ({self.trimming_rule}); "
            f"Learner: {latex_escape(self.learner)}."
        )
        lines.extend([
            r"\bottomrule",
            f"\\multicolumn{{6}}{{l}}{{\\footnotesize {note}}} \\\\",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_typst(self) -> str:
        """Export results to a Typst table."""
        from puremacro.reports import typst_escape

        pct = int(round(self.ci_level * 100))
        p_str = typst_escape("<0.001") if self.p_value < 0.001 else f"{self.p_value:.4f}"
        lines = [
            "#figure(",
            "  table(",
            "    columns: (2fr, 1.2fr, 1.2fr, 1fr, 1fr, 2fr),",
            "    align: (left, right, right, right, right, center),",
            "    stroke: none,",
            "    table.hline(),",
            f"    [*Target*], [*Coef.*], [*Std.Err.*], [*z*], [*P>|z|*], [*{pct}% CI*],",
            "    table.hline(stroke: 0.5pt),",
            f"    [{typst_escape(self.score_type)}], [{self.theta:.4f}], [{self.se:.4f}], [{self.t_stat:.3f}], [{p_str}], [[{self.ci_lower:.4f}, {self.ci_upper:.4f}]],",
            "    table.hline(),",
            "  ),",
            f"  caption: [Double Machine Learning IRM Estimates ({typst_escape(self.learner)}, {self.score_type}, N={self.n_obs})],",
            ")",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class DMLIVResult:
    """Results container for Double Machine Learning Instrumental Variables (DML-IV).

    Attributes
    ----------
    theta : float | np.ndarray
        Debiased causal parameter estimate(s).
    se : float | np.ndarray
        Root-N asymptotic standard error(s).
    t_stat : float | np.ndarray
        Asymptotic z / t test statistic(s).
    p_value : float | np.ndarray
        Two-sided p-value(s).
    ci_lower : float | np.ndarray
        Lower bound(s) of asymptotic confidence interval.
    ci_upper : float | np.ndarray
        Upper bound(s) of asymptotic confidence interval.
    n_obs : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    learner : str
        Learner description used for nuisance estimation.
    residuals_y : np.ndarray
        Out-of-fold outcome residuals Y_tilde.
    residuals_d : np.ndarray
        Out-of-fold treatment residuals D_tilde.
    residuals_z : np.ndarray
        Out-of-fold instrument residuals Z_tilde.
    first_stage_f : float
        Conventional first-stage F-statistic.
    first_stage_effective_f : float
        Montiel Olea & Pflueger (2013) effective F-statistic.
    weak_instrument : bool
        True if first_stage_effective_f < 10.
    ci_level : float
        Confidence level, e.g. 0.95.
    treatment_names : tuple[str, ...]
        Names of treatment variables.
    instrument_names : tuple[str, ...]
        Names of instrumental variables.
    outcome_name : str
        Name of outcome variable.
    models_l : tuple[Any, ...]
        Fitted models for l across folds.
    models_m : tuple[Any, ...]
        Fitted models for m across folds.
    models_r : tuple[Any, ...]
        Fitted models for r across folds.
    feature_names : tuple[str, ...]
        Names of control features X.
    """

    theta: float | np.ndarray
    se: float | np.ndarray
    t_stat: float | np.ndarray
    p_value: float | np.ndarray
    ci_lower: float | np.ndarray
    ci_upper: float | np.ndarray
    n_obs: int
    n_folds: int
    learner: str
    residuals_y: np.ndarray
    residuals_d: np.ndarray
    residuals_z: np.ndarray
    first_stage_f: float
    first_stage_effective_f: float
    weak_instrument: bool
    ci_level: float = 0.95
    treatment_names: tuple[str, ...] = ("D",)
    instrument_names: tuple[str, ...] = ("Z",)
    outcome_name: str = "Y"
    models_l: tuple[Any, ...] = ()
    models_m: tuple[Any, ...] = ()
    models_r: tuple[Any, ...] = ()
    feature_names: tuple[str, ...] = ()

    def summary(self) -> str:
        """Text summary of DML-IV estimation results."""
        pct = int(round(self.ci_level * 100))
        lines = [
            "=" * 78,
            "Double / Debiased Machine Learning (DML-IV)",
            "Model: Partially Linear Instrumental Variables (Chernozhukov et al. 2018)",
            "=" * 78,
            f"Outcome: {self.outcome_name:<18} Observations: {self.n_obs:<12} Folds: {self.n_folds}",
            f"Learner: {self.learner:<18} Instruments: {', '.join(self.instrument_names)}",
            f"First-Stage F: {self.first_stage_f:<12.2f} Effective F (MOP): {self.first_stage_effective_f:<10.2f} Weak IV: {'Yes' if self.weak_instrument else 'No'}",
            "-" * 78,
            f"{'Variable':<16} {'Coef.':>10} {'Std.Err.':>10} {'z':>8} {'P>|z|':>8} "
            f"[{pct}% Conf. Interval]",
            "-" * 78,
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
            lines.append(
                f"{name:<16} {th:>10.4f} {s:>10.4f} {t_val:>8.3f} {p_str:>8} "
                f"{lo:>10.4f} {hi:>10.4f}"
            )
        lines.append("=" * 78)
        if self.weak_instrument:
            lines.append(
                f"Warning: Weak instruments detected (MOP F_eff = {self.first_stage_effective_f:.2f} < 10). "
                "IV estimates and standard errors may be subject to substantial bias."
            )
            lines.append("=" * 78)
        return "\n".join(lines)

    def plot(self, kind: str = "forest", ax: Any = None, **kwargs: Any) -> Any:
        """Plot DML-IV estimation results.

        Parameters
        ----------
        kind : {'forest', 'residuals', 'first_stage', 'coefficients'}, default 'forest'
            - 'forest': Forest plot of treatment coefficient estimate(s) and CI.
            - 'residuals': Scatter of D_tilde vs Y_tilde with 2SLS slope.
            - 'first_stage': Scatter of Z_tilde vs D_tilde with first-stage line.
            - 'coefficients': Regularized coefficients of nuisance models.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        if kind == "residuals":
            res_y = self.residuals_y
            res_d = self.residuals_d[:, 0] if self.residuals_d.ndim > 1 else self.residuals_d
            th = float(np.atleast_1d(self.theta)[0])
            name = self.treatment_names[0]

            ax.scatter(res_d, res_y, alpha=kwargs.get("alpha", 0.4), color="steelblue", s=18, label="Residuals")
            grid_d = np.linspace(res_d.min(), res_d.max(), 100)
            ax.plot(grid_d, th * grid_d, color="firebrick", lw=2, label=f"2SLS Slope: {th:.4f}")
            ax.set_xlabel(f"Residualized Treatment: $\\tilde{{{name}}}$")
            ax.set_ylabel(f"Residualized Outcome: $\\tilde{{{self.outcome_name}}}$")
            ax.set_title("DML-IV Structural Residuals & Causal Effect")
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3, ls=":")
        elif kind == "first_stage":
            res_z = self.residuals_z[:, 0] if self.residuals_z.ndim > 1 else self.residuals_z
            res_d = self.residuals_d[:, 0] if self.residuals_d.ndim > 1 else self.residuals_d
            t_name = self.treatment_names[0]
            z_name = self.instrument_names[0]

            denom = float(np.dot(res_z, res_z))
            gamma = float(np.dot(res_z, res_d) / denom) if denom > 0 else 0.0
            ax.scatter(res_z, res_d, alpha=kwargs.get("alpha", 0.4), color="purple", s=18, label="Orthogonal Residuals")
            grid_z = np.linspace(res_z.min(), res_z.max(), 100)
            ax.plot(grid_z, gamma * grid_z, color="darkorange", lw=2, label=f"First-Stage Slope: {gamma:.4f}")
            ax.set_xlabel(f"Residualized Instrument: $\\tilde{{{z_name}}}$")
            ax.set_ylabel(f"Residualized Treatment: $\\tilde{{{t_name}}}$")
            ax.set_title(f"DML-IV First Stage ($F_{{eff}} = {self.first_stage_effective_f:.2f}$)")
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3, ls=":")
        elif kind == "coefficients":
            return self.plot_coefficients(ax=ax, **kwargs)
        else:  # 'forest'
            thetas = np.atleast_1d(self.theta)
            los = np.atleast_1d(self.ci_lower)
            his = np.atleast_1d(self.ci_upper)
            k = len(thetas)
            y_pos = np.arange(k)

            ax.errorbar(
                thetas,
                y_pos,
                xerr=[thetas - los, his - thetas],
                fmt="o",
                color="navy",
                ecolor="steelblue",
                elinewidth=2,
                capsize=5,
                capthick=1.5,
                markersize=6,
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.treatment_names)
            pct = int(round(self.ci_level * 100))
            ax.set_xlabel(f"Treatment Effect $\\theta$ ({pct}% CI)")
            ax.set_title(f"Double ML Instrumental Variables ({self.learner}, $F_{{eff}}={self.first_stage_effective_f:.2f}$)")
            ax.grid(True, alpha=0.3, ls=":")

        return ax

    def plot_coefficients(
        self, model: str = "treatment", top_k: int = 20, ax: Any = None, **kwargs: Any
    ) -> Any:
        """Plot coefficient estimates with confidence intervals or regularized nuisance coefficients.

        Parameters
        ----------
        model : {'treatment', 'all', 'l', 'm', 'r'}, default 'treatment'
            - 'treatment': 2SLS causal parameter estimate(s) with asymptotic confidence interval(s).
            - 'all', 'l', 'm', 'r': Regularized nuisance feature coefficients across controls.
        top_k : int, default 20
            Maximum number of top coefficients to display when inspecting nuisance models.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.
        """
        if int(top_k) < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k!r}")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        if model == "treatment":
            thetas = np.atleast_1d(self.theta)
            los = np.atleast_1d(self.ci_lower)
            his = np.atleast_1d(self.ci_upper)
            k = len(thetas)
            y_pos = np.arange(k)
            pct = int(round(self.ci_level * 100))

            ax.errorbar(
                thetas,
                y_pos,
                xerr=[thetas - los, his - thetas],
                fmt="o",
                color="navy",
                ecolor="steelblue",
                elinewidth=2,
                capsize=5,
                capthick=1.5,
                markersize=7,
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.treatment_names)
            ax.set_xlabel(f"Causal Effect $\\theta$ ({pct}% CI)")
            ax.set_title(f"DML-IV Point Estimates with {pct}% CI ($F_{{eff}}={self.first_stage_effective_f:.2f}$)")
            ax.grid(True, alpha=0.3, ls=":")
            return ax

        m_key = str(model).lower()
        coef_dict: dict[str, np.ndarray] = {}
        if m_key in ("all", "l"):
            coefs = [m.coef_ for m in self.models_l if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["l(X)"] = np.mean(coefs, axis=0)
        if m_key in ("all", "m"):
            coefs = [m.coef_ for m in self.models_m if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["m(X)"] = np.mean(coefs, axis=0)
        if m_key in ("all", "r"):
            coefs = [m.coef_ for m in self.models_r if getattr(m, "coef_", None) is not None]
            if coefs:
                coef_dict["r(X)"] = np.mean(coefs, axis=0)

        if not coef_dict:
            ax.text(
                0.5, 0.5, "No nuisance model coefficients available",
                ha="center", va="center", transform=ax.transAxes
            )
            return ax

        p = len(next(iter(coef_dict.values())))
        feat_names = (
            list(self.feature_names)
            if self.feature_names and len(self.feature_names) == p
            else [f"X_{i}" for i in range(p)]
        )

        max_abs = np.zeros(p)
        for c in coef_dict.values():
            max_abs = np.maximum(max_abs, np.abs(c))
        k = min(top_k, p)
        top_idx = np.argsort(max_abs)[-k:]

        y_positions = np.arange(k)
        n_models = len(coef_dict)
        bar_height = 0.8 / max(n_models, 1)
        colors = ["steelblue", "darkorange", "seagreen", "goldenrod"]

        for i, (name, c) in enumerate(coef_dict.items()):
            offsets = y_positions + (i - (n_models - 1) / 2.0) * bar_height
            ax.barh(
                offsets,
                c[top_idx],
                height=bar_height * 0.9,
                color=colors[i % len(colors)],
                label=name,
                alpha=0.85,
            )

        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([feat_names[j] for j in top_idx])
        ax.set_xlabel("Average Regularized Coefficient (across folds)")
        ax.set_title(f"Top {k} Nuisance Feature Coefficients")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3, ls=":")
        return ax

    def plot_tuning(self, model: str = "l", ax: Any = None, **kwargs: Any) -> Any:
        """Plot regularization tuning curve across penalty parameters."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 4)))

        m_key = str(model).lower()
        if m_key not in ("l", "m", "r"):
            raise ValueError(f"model must be one of ('l', 'm', 'r'), got {model!r}")
        target_models = self.models_l if m_key == "l" else (self.models_m if m_key == "m" else self.models_r)

        chosen = None
        for m in target_models:
            if getattr(m, "alphas_", None) is not None and getattr(m, "scores_", None) is not None:
                chosen = m
                break

        if chosen is None or len(getattr(chosen, "alphas_", [])) <= 1:
            ax.text(
                0.5, 0.5, f"No regularization path tuning available for model {model!r}",
                ha="center", va="center", transform=ax.transAxes
            )
            return ax

        crit_label = getattr(chosen, "criterion", "Tuning").upper()
        log_alphas = np.log10(chosen.alphas_)
        ax.plot(
            log_alphas,
            chosen.scores_,
            marker="o",
            markersize=4,
            lw=1.8,
            color="navy",
            label=f"{crit_label} Criterion",
        )
        if chosen.alpha_ is not None and chosen.alpha_ > 0:
            ax.axvline(
                np.log10(chosen.alpha_),
                color="firebrick",
                linestyle="--",
                lw=1.5,
                label=f"Selected $\\alpha^*$ = {chosen.alpha_:.4g}",
            )

        ax.set_xlabel(r"$\log_{10}(\alpha)$")
        ax.set_ylabel(f"{crit_label} Score")
        ax.set_title(f"Regularization Path Tuning ({chosen.__class__.__name__}, model={model})")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3, ls=":")
        return ax

    def to_markdown(self) -> str:
        """Export results to a GitHub-flavoured Markdown table."""
        pct = int(round(self.ci_level * 100))
        lines = [
            f"### DML-IV: {self.outcome_name}",
            f"*Learner: {self.learner} | N: {self.n_obs} | Folds: {self.n_folds} | F_eff: {self.first_stage_effective_f:.2f}*",
            "",
            f"| Variable | Coef. | Std.Err. | z | P>\\|z\\| | [{pct}% Conf. Interval] |",
            "|:---|---:|---:|---:|---:|:---:|",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
            cell_name = str(name).replace("|", "\\|")
            lines.append(
                f"| {cell_name} | {th:.4f} | {s:.4f} | {t_val:.3f} | {p_str} | [{lo:.4f}, {hi:.4f}] |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Export results to a LaTeX table with booktabs."""
        from puremacro.reports import latex_escape

        pct = int(round(self.ci_level * 100))
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Double Machine Learning IV Estimates for {latex_escape(self.outcome_name)}}}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            f"Variable & Coef. & Std. Err. & $z$ & $P>|z|$ & [{pct}\\% CI] \\\\",
            r"\midrule",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = "$<0.001$" if p_val < 0.001 else f"{p_val:.4f}"
            clean_name = latex_escape(name)
            lines.append(
                f"{clean_name} & {th:.4f} & {s:.4f} & {t_val:.3f} & {p_str} & [{lo:.4f}, {hi:.4f}] \\\\"
            )
        note = (
            f"Observations: {self.n_obs}; Folds: {self.n_folds}; "
            f"$F_{{eff}}$: {self.first_stage_effective_f:.2f}; "
            f"Learner: {latex_escape(self.learner)}."
        )
        lines.extend([
            r"\bottomrule",
            f"\\multicolumn{{6}}{{l}}{{\\footnotesize {note}}} \\\\",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_typst(self) -> str:
        """Export results to a Typst table."""
        from puremacro.reports import typst_escape

        pct = int(round(self.ci_level * 100))
        lines = [
            "#figure(",
            "  table(",
            "    columns: (2fr, 1.2fr, 1.2fr, 1fr, 1fr, 2fr),",
            "    align: (left, right, right, right, right, center),",
            "    stroke: none,",
            "    table.hline(),",
            f"    [*Variable*], [*Coef.*], [*Std.Err.*], [*z*], [*P>|z|*], [*{pct}% CI*],",
            "    table.hline(stroke: 0.5pt),",
        ]
        thetas = np.atleast_1d(self.theta)
        ses = np.atleast_1d(self.se)
        ts = np.atleast_1d(self.t_stat)
        ps = np.atleast_1d(self.p_value)
        los = np.atleast_1d(self.ci_lower)
        his = np.atleast_1d(self.ci_upper)

        for name, th, s, t_val, p_val, lo, hi in zip(
            self.treatment_names, thetas, ses, ts, ps, los, his
        ):
            p_str = typst_escape("<0.001") if p_val < 0.001 else f"{p_val:.4f}"
            lines.append(
                f"    [{typst_escape(name)}], [{th:.4f}], [{s:.4f}], [{t_val:.3f}], [{p_str}], [[{lo:.4f}, {hi:.4f}]],"
            )
        lines.extend([
            "    table.hline(),",
            "  ),",
            f"  caption: [Double Machine Learning IV Estimates ({typst_escape(self.learner)}, N={self.n_obs}, F_eff={self.first_stage_effective_f:.2f})],",
            ")",
        ])
        return "\n".join(lines)


# ===========================================================================
# DoubleMLPLR Estimator & Functional Interface
# ===========================================================================


class DoubleMLPLR:
    """Double / Debiased Machine Learning for Partially Linear Models (Chernozhukov et al. 2018).
    Estimates the structural causal parameter theta_0 in:
        Y = D * theta_0 + g_0(X) + U,   E[U | X, D] = 0
        D = m_0(X) + V,                 E[V | X] = 0

    using Robinson orthogonal score and K-fold cross-fitting.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-fitting folds (K >= 2, and at most N).
    learner : str, learner class, or learner instance, default 'lasso'
        Base learner for estimating nuisance functions l_0(X) = E[Y|X] and m_0(X) = E[D|X].
        Options: 'lasso' (L1 coordinate descent) or 'ridge' (L2 GCV closed-form), a
        class / factory with ``fit``/``predict`` (constructed with ``learner_kwargs``),
        or a configured instance (deep-copied for every nuisance model, so the
        caller's object is never fitted in place; ``learner_kwargs`` must then be empty).
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 -> 95% CI); must lie
        strictly between 0 and 1.
    random_state : int or None, default 42
        Seed for reproducible random fold splitting.
    learner_kwargs : dict, optional
        Additional keyword arguments passed to learner constructors.

    Notes
    -----
    Valid root-N inference requires each training fold, of size roughly
    ``N * (1 - 1/K)``, to be comfortably larger than the number of controls
    ``p`` for the ridge learner, and a sparse nuisance structure for the
    lasso learner. When a training fold has ``n_train <= p + 1`` observations
    ``fit`` emits a ``UserWarning``: in that regime ``RidgeGCV`` interpolates
    the training data and both built-in learners give biased estimates whose
    nominal 95% intervals under-cover.
    """

    def __init__(
        self,
        n_folds: int = 5,
        learner: str | Any = "lasso",
        alpha: float = 0.05,
        random_state: int | None = 42,
        learner_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {n_folds}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha (significance level) must lie strictly between 0 and 1, got {alpha!r}")
        self.n_folds = int(n_folds)
        self.learner = learner
        self.alpha = float(alpha)
        self.random_state = random_state
        self.learner_kwargs = learner_kwargs or {}
        if self.learner_kwargs and _is_learner_instance(learner):
            raise ValueError(
                "learner_kwargs cannot be combined with a learner instance "
                f"({type(learner).__name__}); configure the instance directly or "
                "pass the learner class / string alias instead."
            )

    def fit(
        self,
        Y: np.ndarray | pd.Series,
        D: np.ndarray | pd.Series | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
    ) -> DMLResult:
        """Estimate partially linear model with K-fold cross-fitting.

        Parameters
        ----------
        Y : array-like of shape (N,)
            Outcome variable.
        D : array-like of shape (N,) or (N, k_d)
            Policy / treatment variable(s).
        X : array-like of shape (N, p)
            High-dimensional controls / covariates.

        Returns
        -------
        DMLResult
            Frozen dataclass container with estimates, standard errors, and diagnostics.
        """
        # Extract names and arrays
        outcome_name = getattr(Y, "name", None) or "Y"
        if isinstance(D, pd.DataFrame):
            treatment_names = tuple(str(c) for c in D.columns)
            D_arr = D.to_numpy(dtype=float)
        elif isinstance(D, pd.Series):
            treatment_names = (str(D.name) if D.name else "D",)
            D_arr = D.to_numpy(dtype=float)[:, None]
        else:
            D_arr = np.asarray(D, dtype=float)
            if D_arr.ndim == 1:
                treatment_names = ("D",)
                D_arr = D_arr[:, None]
            elif D_arr.ndim == 2:
                treatment_names = tuple(f"D_{j+1}" for j in range(D_arr.shape[1]))
            else:
                raise ValueError(f"D must be 1-D or 2-D (N, k_d), got shape {D_arr.shape}.")

        Y_arr = np.asarray(Y, dtype=float).ravel()
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        X_arr = _as_design(X_arr)
        if D_arr.shape[1] == 0:
            raise ValueError("D must have at least one treatment column.")

        n = len(Y_arr)
        if len(D_arr) != n or len(X_arr) != n:
            raise ValueError(
                f"Sample size mismatch: Y has {n}, D has {len(D_arr)}, X has {len(X_arr)} rows."
            )
        if n == 0 or X_arr.shape[1] == 0:
            raise ValueError("Y, D and X must have at least one observation and X at least one column.")
        for label, arr in (("Y", Y_arr), ("D", D_arr), ("X", X_arr)):
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{label} contains NaN or inf; drop or impute missing observations before calling fit()."
                )
        if self.n_folds > n:
            raise ValueError(f"n_folds must not exceed the number of observations, got n_folds={self.n_folds} > N={n}.")
        k_d = D_arr.shape[1]
        p = X_arr.shape[1]

        # Generate K-fold partition
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        n_train_min = n - max(len(f) for f in folds)
        if n_train_min <= p + 1:
            warnings.warn(
                f"DML nuisance learners are trained on as few as {n_train_min} observations per "
                f"fold with p = {p} controls (n_train <= p + 1). In this high-dimensional regime "
                "RidgeGCV interpolates the training fold and the root-N inference for theta is "
                "unreliable for either built-in learner (biased estimates, under-covering "
                "intervals). Increase N, reduce n_folds or the number of controls, or supply a "
                "learner suited to p >= n.",
                UserWarning,
                stacklevel=_warn_stacklevel(),
            )

        res_y = np.zeros(n)
        res_d = np.zeros((n, k_d))

        learner_name = _learner_name(self.learner)

        # Cross-fitting loop
        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(indices, test_idx)
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]
            D_train, D_test = D_arr[train_idx], D_arr[test_idx]

            # 1. Nuisance model for Y: l(X) = E[Y | X]
            model_y = _get_learner(self.learner, **self.learner_kwargs)
            model_y.fit(X_train, Y_train)
            pred_y = model_y.predict(X_test)
            res_y[test_idx] = Y_test - pred_y

            # 2. Nuisance models for D: m(X) = E[D | X]
            for j in range(k_d):
                model_d = _get_learner(self.learner, **self.learner_kwargs)
                model_d.fit(X_train, D_train[:, j])
                pred_d_j = model_d.predict(X_test)
                res_d[test_idx, j] = D_test[:, j] - pred_d_j

        # Robinson orthogonal score solution:
        # Solve: (D_tilde' D_tilde) theta = D_tilde' Y_tilde
        DtD = res_d.T @ res_d
        DtY = res_d.T @ res_y

        try:
            theta_hat = np.linalg.solve(DtD, DtY)
        except np.linalg.LinAlgError:
            theta_hat = np.linalg.pinv(DtD) @ DtY

        # Residuals U_hat = Y_tilde - D_tilde * theta_hat
        u_hat = res_y - res_d @ theta_hat

        # Asymptotic variance via plug-in sandwich:
        # J = (1 / n) * D_tilde' D_tilde
        # Omega = (1 / n) * sum_i u_hat_i^2 * (D_tilde_i' D_tilde_i)
        # V = (1 / n) * J^{-1} Omega J^{-1}
        psi = res_d * u_hat[:, None]  # shape (n, k_d)
        Omega = (psi.T @ psi) / n
        J_inv = np.linalg.pinv(DtD / n)
        vcov = (J_inv @ Omega @ J_inv) / n

        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
        z_crit = norm.ppf(1.0 - self.alpha / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, theta_hat / se, np.nan)
        p_val = 2.0 * norm.sf(np.abs(t_stat))
        ci_lo = theta_hat - z_crit * se
        ci_hi = theta_hat + z_crit * se

        # Unpack scalars for 1D treatment
        if k_d == 1:
            theta_out: float | np.ndarray = float(theta_hat[0])
            se_out: float | np.ndarray = float(se[0])
            t_out: float | np.ndarray = float(t_stat[0])
            p_out: float | np.ndarray = float(p_val[0])
            lo_out: float | np.ndarray = float(ci_lo[0])
            hi_out: float | np.ndarray = float(ci_hi[0])
            res_d_out = res_d.ravel()
        else:
            theta_out = theta_hat
            se_out = se
            t_out = t_stat
            p_out = p_val
            lo_out = ci_lo
            hi_out = ci_hi
            res_d_out = res_d

        return DMLResult(
            theta=theta_out,
            se=se_out,
            t_stat=t_out,
            p_value=p_out,
            ci_lower=lo_out,
            ci_upper=hi_out,
            n_obs=n,
            n_folds=self.n_folds,
            learner=learner_name,
            residuals_y=res_y,
            residuals_d=res_d_out,
            ci_level=1.0 - self.alpha,
            treatment_names=treatment_names,
            outcome_name=str(outcome_name),
        )


def dml_plr(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    learner: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    **learner_kwargs: Any,
) -> DMLResult:
    """Convenience functional interface for Double Machine Learning Partially Linear Regression.

    Parameters
    ----------
    Y : array-like of shape (N,)
        Outcome variable.
    D : array-like of shape (N,) or (N, k_d)
        Treatment variable(s).
    X : array-like of shape (N, p)
        Covariates / control variables.
    n_folds : int, default 5
        Number of cross-fitting folds (2 <= n_folds <= N).
    learner : str or learner class/instance, default 'lasso'
        Base learner ('lasso' or 'ridge'), a learner class, or a configured
        instance (copied per nuisance model; incompatible with ``learner_kwargs``).
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 -> 95% CI), strictly
        between 0 and 1. Note this is the DML significance level, not the lasso
        penalty: set that through ``learner_kwargs={"alpha": ...}`` on
        ``DoubleMLPLR`` or a configured ``LassoCoordinateDescent`` instance.
    random_state : int, default 42
        Seed for reproducible fold splitting.
    **learner_kwargs : Any
        Additional keyword arguments passed to the learner constructor.

    Returns
    -------
    DMLResult
        Frozen dataclass container with estimates and inference methods.
    """
    est = DoubleMLPLR(
        n_folds=n_folds,
        learner=learner,
        alpha=alpha,
        random_state=random_state,
        learner_kwargs=learner_kwargs or None,
    )
    return est.fit(Y, D, X)


# ===========================================================================
# DoubleMLIRM Estimator & Functional Interface
# ===========================================================================


class DoubleMLIRM:
    """Double / Debiased Machine Learning for Interactive Regression Models (Chernozhukov et al. 2018).

    Estimates Average Treatment Effect (ATE) or Average Treatment Effect on the Treated (ATT)
    with heterogeneous treatment effects:
        Y = g_0(D, X) + U,   E[U | D, X] = 0
        D in {0, 1},         m_0(X) = P(D = 1 | X)

    using Neyman-orthogonal doubly robust score and K-fold cross-fitting.
    Supports automatic overlap trimming (clip or drop) to handle limited common support.

    Parameters
    ----------
    n_folds_or_data : int or Any, default None
        Number of cross-fitting folds (K >= 2) or dataset container if passed positionally.
    ml_g : str, learner class, or learner instance, default 'lasso'
        Learner for outcome regression models g_0(d, X) = E[Y | D=d, X].
    ml_m : str, learner class, or learner instance, default 'logistic'
        Learner for propensity score model m_0(X) = P(D=1 | X).
    n_folds : int, default 5
        Number of cross-fitting folds (K >= 2, and at most N).
    score : {'ATE', 'ATT'}, default 'ATE'
        Causal target parameter: Average Treatment Effect or Treatment on Treated.
    trimming_threshold : float, default 0.01
        Boundary epsilon for overlap trimming (0.0 < epsilon < 0.5).
    trimming_rule : {'clip', 'drop'}, default 'clip'
        How to handle extreme propensity scores outside [epsilon, 1 - epsilon].
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 -> 95% CI).
    random_state : int or None, default 42
        Seed for reproducible random fold splitting.
    learner_kwargs : dict, optional
        Additional keyword arguments passed to learner constructors.
    obj_dml_data : Any, optional
        Optional dataset container (tuple of (Y, D, X), dict, or object with attributes).
    """

    def __init__(
        self,
        n_folds_or_data: Any = None,
        ml_g: str | Any = "lasso",
        ml_m: str | Any = "logistic",
        n_folds: int = 5,
        score: str = "ATE",
        trimming_threshold: float = 0.01,
        trimming_rule: str = "clip",
        alpha: float = 0.05,
        random_state: int | None = 42,
        learner_kwargs: dict[str, Any] | None = None,
        obj_dml_data: Any = None,
    ) -> None:
        if n_folds_or_data is None:
            actual_n_folds = int(n_folds)
            self.data = obj_dml_data
        elif isinstance(n_folds_or_data, (int, np.integer)):
            actual_n_folds = int(n_folds_or_data)
            self.data = obj_dml_data
        else:
            self.data = n_folds_or_data if obj_dml_data is None else obj_dml_data
            actual_n_folds = int(n_folds)

        if actual_n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {actual_n_folds}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha (significance level) must lie strictly between 0 and 1, got {alpha!r}")

        sc = str(score).upper()
        if sc not in ("ATE", "ATT"):
            raise ValueError(f"score must be 'ATE' or 'ATT', got {score!r}")

        t_rule = str(trimming_rule).lower()
        if t_rule not in ("clip", "drop"):
            raise ValueError(f"trimming_rule must be 'clip' or 'drop', got {trimming_rule!r}")

        eps = float(trimming_threshold)
        if not (0.0 < eps < 0.5):
            raise ValueError(f"trimming_threshold must lie strictly between 0 and 0.5, got {trimming_threshold!r}")

        self.n_folds = actual_n_folds
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = sc
        self.trimming_threshold = eps
        self.trimming_rule = t_rule
        self.alpha = float(alpha)
        self.random_state = random_state
        self.learner_kwargs = learner_kwargs or {}

        if self.learner_kwargs and (_is_learner_instance(ml_g) or _is_learner_instance(ml_m)):
            raise ValueError(
                "learner_kwargs cannot be combined with a learner instance; "
                "configure the instance directly or pass the learner class / string alias instead."
            )

    def fit(
        self,
        Y: Any = None,
        D: Any = None,
        X: Any = None,
    ) -> DMLIRMResult:
        """Estimate interactive regression model with K-fold cross-fitting."""
        if Y is None:
            if self.data is None:
                raise ValueError("No data provided to fit(); pass (Y, D, X) or supply data in constructor.")
            if isinstance(self.data, tuple) and len(self.data) == 3:
                Y, D, X = self.data
            elif isinstance(self.data, dict) and "Y" in self.data and "D" in self.data and "X" in self.data:
                Y, D, X = self.data["Y"], self.data["D"], self.data["X"]
            elif hasattr(self.data, "y") and hasattr(self.data, "d") and hasattr(self.data, "x"):
                Y, D, X = self.data.y, self.data.d, self.data.x
            else:
                raise ValueError("Could not extract (Y, D, X) from obj_dml_data.")

        outcome_name = getattr(Y, "name", None) or "Y"
        treatment_name = getattr(D, "name", None) or "D"

        if isinstance(X, pd.DataFrame):
            feature_names = tuple(str(c) for c in X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = ()

        X_arr = _as_design(X_arr)
        Y_arr = np.asarray(Y, dtype=float).ravel()
        D_arr = np.asarray(D, dtype=float).ravel()

        n = len(Y_arr)
        if len(D_arr) != n or len(X_arr) != n:
            raise ValueError(
                f"Sample size mismatch: Y has {n}, D has {len(D_arr)}, X has {len(X_arr)} rows."
            )
        if n == 0 or X_arr.shape[1] == 0:
            raise ValueError("Y, D and X must have at least one observation and X at least one column.")
        for label, arr in (("Y", Y_arr), ("D", D_arr), ("X", X_arr)):
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{label} contains NaN or inf; drop or impute missing observations before calling fit()."
                )
        if self.n_folds > n:
            raise ValueError(f"n_folds must not exceed the number of observations, got n_folds={self.n_folds} > N={n}.")

        unique_d = np.unique(D_arr)
        if not np.isin(unique_d, [0.0, 1.0]).all():
            raise ValueError(f"Treatment D in DoubleMLIRM must be binary with values in {{0, 1}}, got {unique_d.tolist()}.")

        p = X_arr.shape[1]

        # Stratified K-fold partition by treatment status
        rng = np.random.default_rng(self.random_state)
        idx_0 = np.where(D_arr == 0.0)[0]
        idx_1 = np.where(D_arr == 1.0)[0]
        if len(idx_0) < self.n_folds or len(idx_1) < self.n_folds:
            raise ValueError(
                f"Each treatment class must have at least n_folds={self.n_folds} observations; "
                f"got {len(idx_0)} control and {len(idx_1)} treated units."
            )
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)
        splits_0 = np.array_split(idx_0, self.n_folds)
        splits_1 = np.array_split(idx_1, self.n_folds)
        folds = [np.concatenate([splits_0[k], splits_1[k]]) for k in range(self.n_folds)]

        m_hat = np.zeros(n)
        g0_hat = np.zeros(n)
        g1_hat = np.zeros(n)

        models_m: list[Any] = []
        models_g0: list[Any] = []
        models_g1: list[Any] = []

        all_indices = np.arange(n)
        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(all_indices, test_idx)
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]
            D_train, D_test = D_arr[train_idx], D_arr[test_idx]

            # 1. Propensity model m(X) = P(D = 1 | X)
            model_m = _get_learner(self.ml_m, **self.learner_kwargs)
            model_m.fit(X_train, D_train)
            if hasattr(model_m, "predict_proba"):
                prob = model_m.predict_proba(X_test)
                if prob.ndim == 2 and prob.shape[1] >= 2:
                    pred_m = prob[:, 1]
                else:
                    pred_m = prob.ravel()
            else:
                pred_m = np.asarray(model_m.predict(X_test), dtype=float).ravel()
            m_hat[test_idx] = pred_m
            models_m.append(model_m)

            # 2. Conditional outcome under control g_0(0, X) = E[Y | D = 0, X]
            mask_0 = D_train == 0.0
            model_g0 = _get_learner(self.ml_g, **self.learner_kwargs)
            model_g0.fit(X_train[mask_0], Y_train[mask_0])
            g0_hat[test_idx] = model_g0.predict(X_test)
            models_g0.append(model_g0)

            # 3. Conditional outcome under treatment g_0(1, X) = E[Y | D = 1, X]
            mask_1 = D_train == 1.0
            model_g1 = _get_learner(self.ml_g, **self.learner_kwargs)
            model_g1.fit(X_train[mask_1], Y_train[mask_1])
            g1_hat[test_idx] = model_g1.predict(X_test)
            models_g1.append(model_g1)

        # Overlap trimming
        eps = self.trimming_threshold
        trimmed_mask = (m_hat < eps) | (m_hat > 1.0 - eps)
        n_trimmed = int(np.count_nonzero(trimmed_mask))

        if self.trimming_rule == "clip":
            m_eval = np.clip(m_hat, eps, 1.0 - eps)
            eval_idx = all_indices
        else:  # "drop"
            eval_idx = np.where(~trimmed_mask)[0]
            if len(eval_idx) == 0:
                raise ValueError("All observations were trimmed by overlap rule. Check propensity scores.")
            m_eval = m_hat[eval_idx]

        Y_eval = Y_arr[eval_idx]
        D_eval = D_arr[eval_idx]
        g0_eval = g0_hat[eval_idx]
        g1_eval = g1_hat[eval_idx]
        n_eval = len(eval_idx)

        # Doubly robust scores
        if self.score == "ATE":
            gamma = (
                g1_eval
                - g0_eval
                + (D_eval * (Y_eval - g1_eval)) / m_eval
                - ((1.0 - D_eval) * (Y_eval - g0_eval)) / (1.0 - m_eval)
            )
            theta_hat = float(np.mean(gamma))
            psi = gamma - theta_hat
            sigma2 = float(np.mean(psi ** 2))
            se = float(np.sqrt(max(sigma2 / n_eval, 0.0)))
        else:  # "ATT"
            p_bar = float(np.mean(D_eval))
            if p_bar == 0.0:
                raise ValueError("No treated units remaining in evaluation sample for ATT estimation.")
            num = D_eval * (Y_eval - g0_eval) - (
                m_eval * (1.0 - D_eval) * (Y_eval - g0_eval)
            ) / (1.0 - m_eval)
            theta_hat = float(np.sum(num) / np.sum(D_eval))
            psi = (num - D_eval * theta_hat) / p_bar
            sigma2 = float(np.mean(psi ** 2))
            se = float(np.sqrt(max(sigma2 / n_eval, 0.0)))

        z_crit = norm.ppf(1.0 - self.alpha / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = float(theta_hat / se) if se > 0 else np.nan
        p_val = float(2.0 * norm.sf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        ci_lo = float(theta_hat - z_crit * se)
        ci_hi = float(theta_hat + z_crit * se)

        learner_label = f"{_learner_name(self.ml_g)} / {_learner_name(self.ml_m)}"

        return DMLIRMResult(
            theta=theta_hat,
            se=se,
            t_stat=t_stat,
            p_value=p_val,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_obs=n,
            n_folds=self.n_folds,
            learner=learner_label,
            score_type=self.score,
            propensity_scores=m_hat,
            g0_pred=g0_hat,
            g1_pred=g1_hat,
            treatment_values=D_arr,
            outcomes=Y_arr,
            trimming_rule=self.trimming_rule,
            trimming_threshold=self.trimming_threshold,
            n_trimmed=n_trimmed,
            ci_level=1.0 - self.alpha,
            treatment_name=str(treatment_name),
            outcome_name=str(outcome_name),
            models_g0=tuple(models_g0),
            models_g1=tuple(models_g1),
            models_m=tuple(models_m),
            feature_names=feature_names,
        )


def dml_irm(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    ml_g: str | Any = "lasso",
    ml_m: str | Any = "logistic",
    score: str = "ATE",
    trimming_threshold: float = 0.01,
    trimming_rule: str = "clip",
    alpha: float = 0.05,
    random_state: int | None = 42,
    **learner_kwargs: Any,
) -> DMLIRMResult:
    """Convenience functional interface for Double Machine Learning Interactive Regression Model (IRM).

    Parameters
    ----------
    Y : array-like of shape (N,)
        Outcome variable.
    D : array-like of shape (N,)
        Binary treatment indicator (values in {0, 1}).
    X : array-like of shape (N, p)
        Covariates / control variables.
    n_folds : int, default 5
        Number of cross-fitting folds (2 <= n_folds <= N).
    ml_g : str or learner class/instance, default 'lasso'
        Learner for outcome regression models g(0, X) and g(1, X).
    ml_m : str or learner class/instance, default 'logistic'
        Learner for propensity score model m(X).
    score : {'ATE', 'ATT'}, default 'ATE'
        Target treatment effect parameter.
    trimming_threshold : float, default 0.01
        Propensity trimming threshold (0 < eps < 0.5).
    trimming_rule : {'clip', 'drop'}, default 'clip'
        How to handle propensity scores outside [eps, 1 - eps].
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 -> 95% CI).
    random_state : int, default 42
        Seed for reproducible fold splitting.
    **learner_kwargs : Any
        Additional keyword arguments passed to learner constructors.

    Returns
    -------
    DMLIRMResult
        Frozen dataclass container with estimates, diagnostics, and presentation methods.
    """
    est = DoubleMLIRM(
        n_folds=n_folds,
        ml_g=ml_g,
        ml_m=ml_m,
        score=score,
        trimming_threshold=trimming_threshold,
        trimming_rule=trimming_rule,
        alpha=alpha,
        random_state=random_state,
        learner_kwargs=learner_kwargs or None,
    )
    return est.fit(Y, D, X)


# ===========================================================================
# DoubleMLIV Estimator & Functional Interface
# ===========================================================================


class DoubleMLIV:
    """Double / Debiased Machine Learning for Instrumental Variable Models (Chernozhukov et al. 2018).

    Estimates causal parameter theta_0 in:
        Y = D' theta_0 + g_0(X) + U,   E[U | X, Z] = 0
        D = m_0(X) + Pi_0(X, Z) + V,   E[V | X] = 0

    using 2SLS on cross-fitted orthogonal residuals (Y_tilde, D_tilde, Z_tilde).
    Computes heteroskedasticity-robust Montiel Olea & Pflueger (2013) effective F-statistic.
    """

    def __init__(
        self,
        n_folds_or_data: Any = None,
        ml_l: str | Any = "lasso",
        ml_m: str | Any = "lasso",
        ml_r: str | Any = "lasso",
        n_folds: int = 5,
        alpha: float = 0.05,
        random_state: int | None = 42,
        learner_kwargs: dict[str, Any] | None = None,
        obj_dml_data: Any = None,
    ) -> None:
        if n_folds_or_data is None:
            actual_n_folds = int(n_folds)
            self.data = obj_dml_data
        elif isinstance(n_folds_or_data, (int, np.integer)):
            actual_n_folds = int(n_folds_or_data)
            self.data = obj_dml_data
        else:
            self.data = n_folds_or_data if obj_dml_data is None else obj_dml_data
            actual_n_folds = int(n_folds)

        if actual_n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {actual_n_folds}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha (significance level) must lie strictly between 0 and 1, got {alpha!r}")

        self.n_folds = actual_n_folds
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.alpha = float(alpha)
        self.random_state = random_state
        self.learner_kwargs = learner_kwargs or {}

        if self.learner_kwargs and (
            _is_learner_instance(ml_l) or _is_learner_instance(ml_m) or _is_learner_instance(ml_r)
        ):
            raise ValueError(
                "learner_kwargs cannot be combined with a learner instance; "
                "configure the instance directly or pass the learner class / string alias instead."
            )

    def fit(
        self,
        Y: Any = None,
        D: Any = None,
        Z: Any = None,
        X: Any = None,
    ) -> DMLIVResult:
        """Estimate partially linear IV model with K-fold cross-fitting."""
        if Y is None:
            if self.data is None:
                raise ValueError("No data provided to fit(); pass (Y, D, Z, X) or supply data in constructor.")
            if isinstance(self.data, tuple) and len(self.data) == 4:
                Y, D, Z, X = self.data
            elif isinstance(self.data, dict) and all(k in self.data for k in ("Y", "D", "Z", "X")):
                Y, D, Z, X = self.data["Y"], self.data["D"], self.data["Z"], self.data["X"]
            elif hasattr(self.data, "y") and hasattr(self.data, "d") and hasattr(self.data, "z") and hasattr(self.data, "x"):
                Y, D, Z, X = self.data.y, self.data.d, self.data.z, self.data.x
            else:
                raise ValueError("Could not extract (Y, D, Z, X) from obj_dml_data.")

        outcome_name = getattr(Y, "name", None) or "Y"
        if isinstance(D, pd.DataFrame):
            treatment_names = tuple(str(c) for c in D.columns)
            D_arr = D.to_numpy(dtype=float)
        elif isinstance(D, pd.Series):
            treatment_names = (str(D.name) if D.name else "D",)
            D_arr = D.to_numpy(dtype=float)[:, None]
        else:
            D_arr = np.asarray(D, dtype=float)
            if D_arr.ndim == 1:
                treatment_names = ("D",)
                D_arr = D_arr[:, None]
            else:
                treatment_names = tuple(f"D_{j+1}" for j in range(D_arr.shape[1]))

        if isinstance(Z, pd.DataFrame):
            instrument_names = tuple(str(c) for c in Z.columns)
            Z_arr = Z.to_numpy(dtype=float)
        elif isinstance(Z, pd.Series):
            instrument_names = (str(Z.name) if Z.name else "Z",)
            Z_arr = Z.to_numpy(dtype=float)[:, None]
        else:
            Z_arr = np.asarray(Z, dtype=float)
            if Z_arr.ndim == 1:
                instrument_names = ("Z",)
                Z_arr = Z_arr[:, None]
            else:
                instrument_names = tuple(f"Z_{j+1}" for j in range(Z_arr.shape[1]))

        if isinstance(X, pd.DataFrame):
            feature_names = tuple(str(c) for c in X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = ()

        X_arr = _as_design(X_arr)
        Y_arr = np.asarray(Y, dtype=float).ravel()

        n = len(Y_arr)
        k_d = D_arr.shape[1]
        k_z = Z_arr.shape[1]

        if len(D_arr) != n or len(Z_arr) != n or len(X_arr) != n:
            raise ValueError(
                f"Sample size mismatch: Y has {n}, D has {len(D_arr)}, Z has {len(Z_arr)}, X has {len(X_arr)} rows."
            )
        if n == 0 or X_arr.shape[1] == 0:
            raise ValueError("Y, D, Z and X must have at least one observation and X at least one column.")
        if k_z < k_d:
            raise ValueError(
                f"Model is under-identified: number of instruments ({k_z}) must be >= number of treatments ({k_d})."
            )

        for label, arr in (("Y", Y_arr), ("D", D_arr), ("Z", Z_arr), ("X", X_arr)):
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{label} contains NaN or inf; drop or impute missing observations before calling fit()."
                )
        if self.n_folds > n:
            raise ValueError(f"n_folds must not exceed the number of observations, got n_folds={self.n_folds} > N={n}.")

        # Generate K-fold partition
        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        res_y = np.zeros(n)
        res_d = np.zeros((n, k_d))
        res_z = np.zeros((n, k_z))

        models_l: list[Any] = []
        models_m: list[Any] = []
        models_r: list[Any] = []

        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(indices, test_idx)
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]
            D_train, D_test = D_arr[train_idx], D_arr[test_idx]
            Z_train, Z_test = Z_arr[train_idx], Z_arr[test_idx]

            # 1. Nuisance model for Y: l(X) = E[Y | X]
            model_l = _get_learner(self.ml_l, **self.learner_kwargs)
            model_l.fit(X_train, Y_train)
            pred_y = model_l.predict(X_test)
            res_y[test_idx] = Y_test - pred_y
            models_l.append(model_l)

            # 2. Nuisance models for D: m_j(X) = E[D_j | X]
            for j in range(k_d):
                model_m = _get_learner(self.ml_m, **self.learner_kwargs)
                model_m.fit(X_train, D_train[:, j])
                pred_d_j = model_m.predict(X_test)
                res_d[test_idx, j] = D_test[:, j] - pred_d_j
                models_m.append(model_m)

            # 3. Nuisance models for Z: r_l(X) = E[Z_l | X]
            for l in range(k_z):
                model_r = _get_learner(self.ml_r, **self.learner_kwargs)
                model_r.fit(X_train, Z_train[:, l])
                pred_z_l = model_r.predict(X_test)
                res_z[test_idx, l] = Z_test[:, l] - pred_z_l
                models_r.append(model_r)

        # 2SLS on orthogonal residuals
        # First stage: project res_d on res_z
        ZtZ = res_z.T @ res_z
        ZtD = res_z.T @ res_d
        try:
            gamma_hat = np.linalg.solve(ZtZ, ZtD)
        except np.linalg.LinAlgError:
            gamma_hat = np.linalg.pinv(ZtZ) @ ZtD

        D_hat = res_z @ gamma_hat  # shape (n, k_d)

        # Second stage: project res_y on D_hat
        DhD = D_hat.T @ res_d
        DhY = D_hat.T @ res_y
        try:
            theta_hat = np.linalg.solve(DhD, DhY)
        except np.linalg.LinAlgError:
            theta_hat = np.linalg.pinv(DhD) @ DhY

        # Structural residuals
        u_hat = res_y - res_d @ theta_hat

        # Asymptotic sandwich covariance
        psi = D_hat * u_hat[:, None]  # shape (n, k_d)
        Omega = (psi.T @ psi) / n
        J_inv = np.linalg.pinv(DhD / n)
        vcov = (J_inv @ Omega @ J_inv) / n

        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
        z_crit = norm.ppf(1.0 - self.alpha / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, theta_hat / se, np.nan)
        p_val = 2.0 * norm.sf(np.abs(t_stat))
        ci_lo = theta_hat - z_crit * se
        ci_hi = theta_hat + z_crit * se

        # Weak instrument diagnostics (Montiel Olea & Pflueger 2013)
        v_hat = res_d - D_hat
        v_first = v_hat[:, 0]
        M = res_z * v_first[:, None]
        M_cov = M.T @ M
        try:
            denom = float(np.trace(np.linalg.solve(ZtZ, M_cov)))
        except np.linalg.LinAlgError:
            denom = float(np.trace(np.linalg.pinv(ZtZ) @ M_cov))
        g_first = gamma_hat[:, 0]
        num = float(g_first.T @ ZtZ @ g_first)
        F_eff = float(num / denom) if denom > 0.0 else 0.0

        sigma_v2 = float(np.sum(v_first ** 2)) / max(n - k_z, 1)
        F_conv = float(num / (k_z * sigma_v2)) if sigma_v2 > 0.0 else 0.0

        weak_iv = bool(F_eff < 10.0)
        if weak_iv:
            warnings.warn(
                f"Weak instruments detected: Montiel Olea & Pflueger effective F-statistic is {F_eff:.2f} < 10. "
                "IV estimates and standard errors may be subject to substantial bias and size distortion.",
                UserWarning,
                stacklevel=_warn_stacklevel(),
            )

        learner_label = f"{_learner_name(self.ml_l)} / {_learner_name(self.ml_m)}"

        if k_d == 1:
            theta_out: float | np.ndarray = float(theta_hat[0])
            se_out: float | np.ndarray = float(se[0])
            t_out: float | np.ndarray = float(t_stat[0])
            p_out: float | np.ndarray = float(p_val[0])
            lo_out: float | np.ndarray = float(ci_lo[0])
            hi_out: float | np.ndarray = float(ci_hi[0])
            res_d_out = res_d.ravel()
        else:
            theta_out = theta_hat
            se_out = se
            t_out = t_stat
            p_out = p_val
            lo_out = ci_lo
            hi_out = ci_hi
            res_d_out = res_d

        res_z_out = res_z.ravel() if k_z == 1 else res_z

        return DMLIVResult(
            theta=theta_out,
            se=se_out,
            t_stat=t_out,
            p_value=p_out,
            ci_lower=lo_out,
            ci_upper=hi_out,
            n_obs=n,
            n_folds=self.n_folds,
            learner=learner_label,
            residuals_y=res_y,
            residuals_d=res_d_out,
            residuals_z=res_z_out,
            first_stage_f=F_conv,
            first_stage_effective_f=F_eff,
            weak_instrument=weak_iv,
            ci_level=1.0 - self.alpha,
            treatment_names=treatment_names,
            instrument_names=instrument_names,
            outcome_name=str(outcome_name),
            models_l=tuple(models_l),
            models_m=tuple(models_m),
            models_r=tuple(models_r),
            feature_names=feature_names,
        )


def dml_iv(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    Z: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    ml_l: str | Any = "lasso",
    ml_m: str | Any = "lasso",
    ml_r: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    **learner_kwargs: Any,
) -> DMLIVResult:
    """Convenience functional interface for Double Machine Learning Instrumental Variables (DML-IV).

    Parameters
    ----------
    Y : array-like of shape (N,)
        Outcome variable.
    D : array-like of shape (N,) or (N, k_d)
        Endogenous treatment variable(s).
    Z : array-like of shape (N,) or (N, k_z)
        Excluded instrumental variable(s) (k_z >= k_d).
    X : array-like of shape (N, p)
        Covariates / control variables.
    n_folds : int, default 5
        Number of cross-fitting folds (2 <= n_folds <= N).
    ml_l : str or learner class/instance, default 'lasso'
        Learner for outcome nuisance model l(X) = E[Y | X].
    ml_m : str or learner class/instance, default 'lasso'
        Learner for treatment nuisance model m(X) = E[D | X].
    ml_r : str or learner class/instance, default 'lasso'
        Learner for instrument nuisance model r(X) = E[Z | X].
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 -> 95% CI).
    random_state : int, default 42
        Seed for reproducible fold splitting.
    **learner_kwargs : Any
        Additional keyword arguments passed to learner constructors.

    Returns
    -------
    DMLIVResult
        Frozen dataclass container with 2SLS estimates, weak IV diagnostics, and presentation methods.
    """
    est = DoubleMLIV(
        n_folds=n_folds,
        ml_l=ml_l,
        ml_m=ml_m,
        ml_r=ml_r,
        alpha=alpha,
        random_state=random_state,
        learner_kwargs=learner_kwargs or None,
    )
    return est.fit(Y, D, Z, X)


__all__ = [
    "DoubleMLPLR",
    "DoubleMLIRM",
    "DoubleMLIV",
    "DMLResult",
    "DMLIRMResult",
    "DMLIVResult",
    "LassoCoordinateDescent",
    "RidgeGCV",
    "LogisticCoordinateDescent",
    "dml_plr",
    "dml_irm",
    "dml_iv",
]

