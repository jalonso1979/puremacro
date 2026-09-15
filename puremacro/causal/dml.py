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


def _is_learner_instance(learner: Any) -> bool:
    """True for an already-constructed learner object (has ``fit``/``predict``, is not a class)."""
    return (
        not isinstance(learner, (str, type))
        and hasattr(learner, "fit")
        and hasattr(learner, "predict")
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
        else:
            raise ValueError(f"Unknown learner {learner!r}. Supported: 'lasso', 'ridge'.")
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
        if not (hasattr(obj, "fit") and hasattr(obj, "predict")):
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


__all__ = [
    "DoubleMLPLR",
    "DMLResult",
    "LassoCoordinateDescent",
    "RidgeGCV",
    "dml_plr",
]
