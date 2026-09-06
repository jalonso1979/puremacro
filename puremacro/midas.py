"""Mixed-frequency regression (MIDAS).

Two flavours:
- ``u_midas``  : unrestricted MIDAS — the K high-frequency lags enter
                  with separate coefficients (Foroni-Marcellino-Schumacher 2015).
                  OLS, easy, but parameter count grows with K.
- ``beta_midas`` : Beta-polynomial MIDAS (Ghysels-Santa-Clara-Valkanov 2007).
                  Imposes a two-parameter lag-weighting kernel evaluated on
                  the midpoint grid u_k = (k - 0.5) / K, k = 1..K,
                      w(k) ∝ u_k^(theta1 - 1) (1 - u_k)^(theta2 - 1),
                  normalised to sum to one (the Beta density kernel on the
                  open interval, so the end lags never receive an exact
                  zero weight), so the low-frequency slope is identified
                  from a single scalar ``beta`` regardless of K.

Both functions assume the high-frequency series ``x_hf`` has been
pre-aligned so that ``x_hf[i*K + j]`` is the j-th sub-period of
low-frequency period i. ``y_lf[i]`` is the contemporaneous (or lagged)
low-frequency observation.

References
----------
Ghysels, E., Santa-Clara, P. and Valkanov, R. (2007). MIDAS regressions:
    further results and new directions. Econometric Reviews 26.
Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted
    mixed data sampling (MIDAS): MIDAS regressions with unrestricted
    lag polynomials. JRSS-A 178(1), 57-82.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist


def _bar_figure(labels: list[str], values: np.ndarray, *, ax: Any, title: str, ylabel: str) -> Any:
    """Bar chart helper shared by the MIDAS result objects; returns the Figure."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure
    ax.bar(range(len(values)), values, color="#1f77b4")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    return fig


def _frame_exports(cls):  # type: ignore[no-untyped-def]
    """Attach ``to_markdown`` / ``to_latex`` / ``to_typst`` built on ``to_frame``."""
    def to_markdown(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), **kwargs)

    to_markdown.__doc__ = "Export ``to_frame()`` as a Markdown table."
    to_latex.__doc__ = "Export ``to_frame()`` as a LaTeX ``tabular``."
    to_typst.__doc__ = "Export ``to_frame()`` as a Typst ``#table``."
    cls.to_markdown = to_markdown
    cls.to_latex = to_latex
    cls.to_typst = to_typst
    return cls


@_frame_exports
@dataclass(frozen=True)
class UMidasResult:
    """Result of :func:`u_midas` (unrestricted MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : np.ndarray
        Per-lag MIDAS coefficients, length ``K * (1 + n_low_lags)``.
    fitted : np.ndarray
        In-sample fitted values, length ``n_obs``.
    residuals : np.ndarray
        In-sample residuals, length ``n_obs``.
    R2 : float
        In-sample R².
    n_obs : int
        Effective sample size used in the regression.

    References
    ----------
    Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted
        mixed data sampling (MIDAS): MIDAS regressions with unrestricted
        lag polynomials. JRSS-A 178(1), 57-82.
    """

    intercept: float
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    n_obs: int

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.3f}" for b in self.beta)
        return (
            f"U-MIDAS (Foroni-Marcellino-Schumacher)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {coefs}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  n_obs             : {self.n_obs}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Per-lag coefficient table (``lag`` 1 = most recent sub-period)."""
        return pd.DataFrame(
            {"coef": np.round(self.beta, 4)},
            index=pd.Index(range(1, len(self.beta) + 1), name="lag"),
        )

    def plot(self, *, ax: Any = None, title: str = "U-MIDAS lag coefficients") -> Any:
        """Bar chart of the per-lag coefficients. Returns the Figure."""
        return _bar_figure([str(i) for i in range(1, len(self.beta) + 1)],
                           np.asarray(self.beta, dtype=float), ax=ax, title=title, ylabel="coefficient")


@_frame_exports
@dataclass(frozen=True)
class BetaMidasResult:
    """Result of :func:`beta_midas` (Beta-polynomial MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : float
        Single low-frequency slope on the kernel-weighted high-frequency sum.
    theta1, theta2 : float
        Beta-pdf shape parameters (positive).
    weights : np.ndarray
        Kernel weights ``w(k; theta1, theta2)``, length ``K``, sum to 1.
    fitted : np.ndarray
        In-sample fitted values, length ``n_low``.
    residuals : np.ndarray
        In-sample residuals, length ``n_low``.
    R2 : float
        In-sample R².
    converged : bool
        Whether scipy.optimize.minimize converged.

    References
    ----------
    Ghysels, E., Santa-Clara, P. and Valkanov, R. (2007). MIDAS regressions:
        further results and new directions. Econometric Reviews 26.
    """

    intercept: float
    beta: float
    theta1: float
    theta2: float
    weights: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    converged: bool

    def summary(self) -> str:
        w = ", ".join(f"{x:.3f}" for x in self.weights)
        return (
            f"Beta-MIDAS (Ghysels-Santa-Clara-Valkanov)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {self.beta:+.4f}\n"
            f"  theta1, theta2    : {self.theta1:.3f}, {self.theta2:.3f}\n"
            f"  weights           : {w}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  converged         : {self.converged}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Kernel weights and implied per-lag effect ``beta * w(k)``."""
        w = np.asarray(self.weights, dtype=float)
        return pd.DataFrame(
            {"weight": np.round(w, 4), "beta_x_weight": np.round(self.beta * w, 4)},
            index=pd.Index(range(1, len(w) + 1), name="lag"),
        )

    def plot(self, *, ax: Any = None, title: str = "Beta-MIDAS kernel weights") -> Any:
        """Bar chart of the estimated kernel weights. Returns the Figure."""
        w = np.asarray(self.weights, dtype=float)
        return _bar_figure([str(i) for i in range(1, len(w) + 1)], w, ax=ax,
                           title=f"{title} (theta = {self.theta1:.2f}, {self.theta2:.2f})",
                           ylabel="weight")


def _stack_hf_lags(x_hf: np.ndarray, K: int, n_periods: int,
                    n_low_lags: int = 0) -> np.ndarray | None:
    """Build the design matrix of high-frequency lags.

    For each low-frequency period i in [n_low_lags, n_periods), the
    matrix row holds ``x_hf`` at sub-periods (i*K - 1, i*K - 2, ..., i*K - K)
    — the K most-recent observations *strictly before* the sub-period
    just begun.  When ``n_low_lags > 0`` we extend to cover n_low_lags
    full prior low-frequency periods.
    """
    rows = []
    total_lags = K * (1 + n_low_lags)
    for i in range(n_low_lags + 1, n_periods + 1):
        end = i * K
        start = end - total_lags
        if start < 0:
            return None
        rows.append(x_hf[start:end][::-1])  # most-recent first
    return np.array(rows)


def u_midas(
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    K: int,
    n_low_lags: int = 0,
) -> UMidasResult:
    """Unrestricted MIDAS (Foroni-Marcellino-Schumacher).

    Parameters
    ----------
    y_lf : (n_low,) ndarray of low-frequency observations.
    x_hf : (n_low * K,) ndarray of aligned high-frequency observations.
    K : int — sub-periods per low-frequency period (e.g. 3 for QM).
    n_low_lags : int — additional low-frequency lags of x_hf to include.
        Must be smaller than ``n_low``; the first ``n_low_lags`` low-frequency
        observations are dropped from the regression.

    Returns
    -------
    UMidasResult
        Frozen dataclass with intercept, beta, fitted, residuals, R2, n_obs.

    Raises
    ------
    ValueError
        If ``len(x_hf) != n_low * K``, or ``n_low_lags >= n_low`` (no
        observations would be left).
    """
    y_lf = np.asarray(y_lf, dtype=float).ravel()
    x_hf = np.asarray(x_hf, dtype=float).ravel()
    n_low = len(y_lf)
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")
    if n_low_lags < 0:
        raise ValueError(f"n_low_lags must be >= 0; got {n_low_lags}")
    if len(x_hf) != n_low * K:
        raise ValueError(f"x_hf length must be n_low*K = {n_low * K}; "
                         f"got {len(x_hf)}")
    if n_low_lags >= n_low:
        raise ValueError(
            f"n_low_lags ({n_low_lags}) must be smaller than the number of "
            f"low-frequency observations ({n_low}): no rows would be left "
            "for the regression."
        )
    X = _stack_hf_lags(x_hf, K, n_low, n_low_lags=n_low_lags)
    if X is None:
        raise ValueError("Not enough high-frequency observations for the "
                         "requested n_low_lags.")
    y_aligned = y_lf[n_low_lags:]
    n_obs = len(y_aligned)
    X1 = np.column_stack([np.ones(n_obs), X])
    beta = np.linalg.lstsq(X1, y_aligned, rcond=None)[0]
    fitted = X1 @ beta
    resid = y_aligned - fitted
    rss = float(resid @ resid)
    tss = float(((y_aligned - y_aligned.mean()) ** 2).sum())
    return UMidasResult(
        intercept=float(beta[0]),
        beta=beta[1:],
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        n_obs=n_obs,
    )


def _beta_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """Beta-pdf weights on K nodes (k=1..K), normalised to sum to 1."""
    if theta1 <= 0 or theta2 <= 0:
        return np.full(K, 1.0 / K)
    grid = (np.arange(1, K + 1) - 0.5) / K  # midpoints in (0,1)
    w = grid ** (theta1 - 1) * (1.0 - grid) ** (theta2 - 1)
    return w / w.sum()


def beta_midas(
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    K: int,
    *,
    theta1_init: float = 1.0,
    theta2_init: float = 5.0,
) -> BetaMidasResult:
    """Beta-polynomial MIDAS by NLS.

    Model
    -----
        y_i = alpha + beta * sum_{k=1..K} w(k; theta1, theta2) * x_hf[(i-1)*K + (K-k)]
            + eps_i

    with w(k) the Beta-pdf-style kernel.  alpha, beta are linear; theta1,
    theta2 enter the kernel non-linearly and are estimated by minimising
    the SSR.

    Returns
    -------
    BetaMidasResult
        Frozen dataclass with intercept, beta, theta1, theta2, weights,
        fitted, residuals, R2, converged.
    """
    y_lf = np.asarray(y_lf, dtype=float).ravel()
    x_hf = np.asarray(x_hf, dtype=float).ravel()
    n_low = len(y_lf)
    X_hf = _stack_hf_lags(x_hf, K, n_low, n_low_lags=0)
    if X_hf is None:
        raise ValueError("x_hf too short.")

    def _loss(params):
        t1, t2 = params
        if t1 <= 0 or t2 <= 0:
            return 1e10
        w = _beta_weights(K, t1, t2)
        z = X_hf @ w
        Z1 = np.column_stack([np.ones(len(z)), z])
        try:
            beta_lin, *_ = np.linalg.lstsq(Z1, y_lf, rcond=None)
        except np.linalg.LinAlgError:
            return 1e10
        resid = y_lf - Z1 @ beta_lin
        return float(resid @ resid)

    res = minimize(
        _loss, x0=np.array([theta1_init, theta2_init]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 1000},
    )
    t1, t2 = res.x
    w_hat = _beta_weights(K, t1, t2)
    z = X_hf @ w_hat
    Z1 = np.column_stack([np.ones(len(z)), z])
    beta_lin, *_ = np.linalg.lstsq(Z1, y_lf, rcond=None)
    fitted = Z1 @ beta_lin
    resid = y_lf - fitted
    rss = float(resid @ resid); tss = float(((y_lf - y_lf.mean()) ** 2).sum())
    return BetaMidasResult(
        intercept=float(beta_lin[0]),
        beta=float(beta_lin[1]),
        theta1=float(t1),
        theta2=float(t2),
        weights=w_hat,
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        converged=bool(res.success),
    )


@_frame_exports
@dataclass(frozen=True)
class GarchMidasResult:
    """Result of :func:`garch_midas` (Engle, Ghysels & Sohn 2013).

    Attributes
    ----------
    mu : float
        Estimated unconditional mean return.
    m : float
        Long-run log-tau intercept.
    theta : float
        MIDAS slope parameter on the low-frequency driver.
    w2 : float
        Beta polynomial decay parameter (with w1=1.0 fixed).
    alpha : float
        Short-run ARCH parameter in ``g_{i,t}``.
    beta : float
        Short-run GARCH parameter in ``g_{i,t}``.
    weights : np.ndarray
        Beta polynomial weights across the L low-frequency lags, sum to 1.
    tau : pd.Series | np.ndarray
        Low-frequency long-run variance component ``tau_t`` (expanded to HF length).
    g : pd.Series | np.ndarray
        High-frequency short-run variance component ``g_{i,t}``.
    sigma : pd.Series | np.ndarray
        Total conditional standard deviation ``sqrt(tau * g)``.
    loglik : float
        Maximized Gaussian log-likelihood.
    converged : bool
        Whether optimizer converged.
    n_obs : int
        Number of high-frequency observations.
    n_low : int
        Number of low-frequency periods.
    """

    mu: float
    m: float
    theta: float
    w2: float
    alpha: float
    beta: float
    weights: np.ndarray
    tau: pd.Series | np.ndarray
    g: pd.Series | np.ndarray
    sigma: pd.Series | np.ndarray
    loglik: float
    converged: bool
    n_obs: int
    n_low: int

    def summary(self) -> str:
        return (
            f"GARCH-MIDAS (Engle-Ghysels-Sohn 2013)\n"
            f"  mu (mean)         : {self.mu:+.4f}\n"
            f"  m (log-tau const) : {self.m:+.4f}\n"
            f"  theta (macro)     : {self.theta:+.4f}\n"
            f"  w2 (decay)        : {self.w2:.3f}\n"
            f"  alpha (ARCH)      : {self.alpha:.4f}\n"
            f"  beta (GARCH)      : {self.beta:.4f}\n"
            f"  persistence       : {self.alpha + self.beta:.4f}\n"
            f"  log-likelihood    : {self.loglik:.2f}\n"
            f"  converged         : {self.converged}\n"
            f"  n_obs (HF)        : {self.n_obs}\n"
        )

    def to_frame(self) -> pd.DataFrame:
        """Parameter table (one row per estimated parameter)."""
        rows = [
            ("mu", self.mu), ("m", self.m), ("theta", self.theta), ("w2", self.w2),
            ("alpha", self.alpha), ("beta", self.beta),
            ("persistence", self.alpha + self.beta), ("loglik", self.loglik),
        ]
        return pd.DataFrame(
            {"value": [round(float(v), 4) for _, v in rows]},
            index=pd.Index([k for k, _ in rows], name="parameter"),
        )

    def plot(self, *, ax: Any = None, title: str = "GARCH-MIDAS conditional volatility") -> Any:
        """Total conditional sd ``sigma`` (solid) and long-run ``sqrt(tau)``
        (dashed). Returns the Figure."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 3.4))
        else:
            fig = ax.figure
        sig = pd.Series(self.sigma) if not isinstance(self.sigma, pd.Series) else self.sigma
        tau = pd.Series(self.tau) if not isinstance(self.tau, pd.Series) else self.tau
        ax.plot(sig.index, sig.to_numpy(), lw=1.0, label="sigma (total)")
        ax.plot(tau.index, np.sqrt(tau.to_numpy()), lw=1.6, ls="--", label="sqrt(tau) (long-run)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        return fig


def _beta_weights_decay(L: int, w2: float) -> np.ndarray:
    """One-parameter Beta polynomial weights with w1=1.0 fixed (monotonic decay)."""
    j = np.arange(1, L + 1)
    raw = (1.0 - j / (L + 1.0)) ** (w2 - 1.0)
    s = np.sum(raw)
    return raw / s if s > 0 else np.full(L, 1.0 / L)


def garch_midas(
    returns_hf: np.ndarray | pd.Series,
    x_lf: np.ndarray | pd.Series | None = None,
    K: int = 22,
    L: int = 12,
    *,
    w2_init: float = 3.0,
) -> GarchMidasResult:
    """Fit GARCH-MIDAS model for mixed-frequency volatility (Engle, Ghysels & Sohn 2013).

    Decomposes conditional variance into a short-run daily GARCH component ``g_{i,t}``
    and a low-frequency macroeconomic/realized volatility trend ``tau_t``:
    ``r_{i,t} = mu + sqrt(tau_t * g_{i,t}) * eps_{i,t}``
    where ``log(tau_t) = m + theta * sum_{l=1}^L w(l; w2) * X_{t-l}``.

    Parameters
    ----------
    returns_hf : np.ndarray or pd.Series
        High-frequency (e.g. daily) return series.
    x_lf : np.ndarray or pd.Series, optional
        Low-frequency (e.g. monthly/quarterly) macro variable or driver.
        If None, uses low-frequency realized volatility (sum of squared returns per period).
    K : int, default 22
        Sub-periods per low-frequency period (e.g. 22 trading days per month).
    L : int, default 12
        Number of low-frequency lags included in the MIDAS polynomial.
    w2_init : float, default 3.0
        Initial value for the Beta polynomial decay parameter (>= 1.0).

    Returns
    -------
    GarchMidasResult
        Frozen dataclass with parameters, filtered components (tau, g, sigma),
        log-likelihood, and convergence status.

    References
    ----------
    Engle, R.F., Ghysels, E., and Sohn, B. (2013). Stock Market Volatility and
        Macroeconomic Fundamentals. The Review of Economics and Statistics 95(3), 776-798.
    """
    if isinstance(returns_hf, pd.Series):
        hf_idx = returns_hf.index
        r_arr = returns_hf.to_numpy(dtype=float)
    else:
        hf_idx = None
        r_arr = np.asarray(returns_hf, dtype=float).ravel()

    N = len(r_arr)
    T = N // K
    if T <= L:
        raise ValueError(
            f"GARCH-MIDAS requires T > L low-frequency periods (got T={T}, L={L}; N={N}, K={K})."
        )

    # Trim to exact multiple of K
    N_usable = T * K
    r_arr = r_arr[:N_usable]
    if hf_idx is not None:
        hf_idx = hf_idx[:N_usable]

    if x_lf is None:
        reshaped = r_arr.reshape(T, K)
        x_arr = np.sum(reshaped ** 2, axis=1)
    else:
        if isinstance(x_lf, pd.Series):
            x_arr = x_lf.to_numpy(dtype=float)
        else:
            x_arr = np.asarray(x_lf, dtype=float).ravel()
        if len(x_arr) < T:
            raise ValueError(f"x_lf length ({len(x_arr)}) must be >= T ({T}).")
        x_arr = x_arr[:T]

    mu_init = float(np.mean(r_arr))
    var_init = float(np.var(r_arr))
    m_init = float(np.log(max(var_init, 1e-6)))

    x0 = [mu_init, m_init, 0.0, float(w2_init), 0.05, 0.90]

    def _neg_loglik(params):
        mu, m, theta, w2, alpha, beta = params
        if alpha <= 0 or beta < 0 or alpha + beta >= 0.999 or w2 < 1.0:
            return 1e10

        weights = _beta_weights_decay(L, w2)
        tau = np.empty(T)
        for t in range(L, T):
            window = x_arr[t - L : t][::-1]
            tau[t] = np.exp(m + theta * np.dot(weights, window))

        ll = 0.0
        g_prev = 1.0
        r_prev = r_arr[L * K - 1]
        tau_prev = tau[L]

        for t in range(L, T):
            tau_t = tau[t]
            for i in range(K):
                idx = t * K + i
                r_curr = r_arr[idx]
                eps_prev2 = (r_prev - mu) ** 2 / tau_prev
                g_curr = (1.0 - alpha - beta) + alpha * eps_prev2 + beta * g_prev
                g_curr = max(g_curr, 1e-8)

                var_curr = tau_t * g_curr
                ll += 0.5 * (np.log(2 * np.pi) + np.log(var_curr) + (r_curr - mu) ** 2 / var_curr)

                g_prev = g_curr
                r_prev = r_curr
                tau_prev = tau_t

        return float(ll)

    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (1.0, 50.0),
        (1e-6, 0.5),
        (1e-6, 0.999),
    ]

    opt = minimize(_neg_loglik, x0, bounds=bounds, method="L-BFGS-B")
    mu, m, theta, w2, alpha, beta = opt.x
    weights = _beta_weights_decay(L, w2)

    tau_full = np.empty(T)
    for t in range(min(L, T)):
        tau_full[t] = np.exp(m)
    for t in range(L, T):
        window = x_arr[t - L : t][::-1]
        tau_full[t] = np.exp(m + theta * np.dot(weights, window))

    g_full = np.empty(N_usable)
    g_prev = 1.0
    r_prev = r_arr[0]
    tau_prev = tau_full[0]

    for t in range(T):
        tau_t = tau_full[t]
        for i in range(K):
            idx = t * K + i
            r_curr = r_arr[idx]
            eps_prev2 = (r_prev - mu) ** 2 / tau_prev
            g_curr = (1.0 - alpha - beta) + alpha * eps_prev2 + beta * g_prev
            g_curr = max(g_curr, 1e-8)
            g_full[idx] = g_curr
            g_prev = g_curr
            r_prev = r_curr
            tau_prev = tau_t

    tau_hf = np.repeat(tau_full, K)
    sigma = np.sqrt(tau_hf * g_full)

    if hf_idx is not None:
        tau_out = pd.Series(tau_hf, index=hf_idx)
        g_out = pd.Series(g_full, index=hf_idx)
        sigma_out = pd.Series(sigma, index=hf_idx)
    else:
        tau_out = tau_hf
        g_out = g_full
        sigma_out = sigma

    return GarchMidasResult(
        mu=float(mu),
        m=float(m),
        theta=float(theta),
        w2=float(w2),
        alpha=float(alpha),
        beta=float(beta),
        weights=weights,
        tau=tau_out,
        g=g_out,
        sigma=sigma_out,
        loglik=float(-opt.fun),
        converged=bool(opt.success),
        n_obs=N_usable,
        n_low=T,
    )


__all__ = [
    "u_midas",
    "beta_midas",
    "garch_midas",
    "UMidasResult",
    "BetaMidasResult",
    "GarchMidasResult",
]
