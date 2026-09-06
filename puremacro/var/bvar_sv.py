"""Bayesian VAR with Stochastic Volatility (BVAR-SV).

Estimates a Vector Autoregression with time-varying residual covariance
via MCMC Gibbs sampling:

    y_t = c + sum_{l=1}^p A_l y_{t-l} + u_t,  t = 1, ..., T
    u_t = A^{-1} D_t^{1/2} ε_t,               ε_t ~ N(0, I_n)

where:
    - A is a lower-triangular contemporaneous impact matrix with unit diagonal:
          A = [[1,       0, ..., 0],
               [a_{2,1}, 1, ..., 0],
               [...,     ..., 1, 0],
               [a_{n,1}, ..., ..., 1]]
    - D_t = diag(exp(h_{1,t}), ..., exp(h_{n,t})) contains time-varying variances.
    - Reduced-form residual covariance is Σ_t = A^{-1} D_t A^{-T}.
    - Log-volatilities follow stationary AR(1) state dynamics for i = 1, ..., n:
          h_{i,t} - μ_i = φ_i (h_{i,t-1} - μ_i) + σ_{h,i} η_{i,t},  |φ_i| < 1,
      with stationary initial condition h_{i,1} - μ_i ~ N(0, σ_{h,i}² / (1 - φ_i²)).

The MCMC algorithm uses:
    1. Minnesota prior shrinkage on VAR coefficients β, drawn *jointly* from
       the exact conditional posterior N(β̄, V̄) with the (nk × nk)
       precision-weighted GLS precision V̄^{-1} = V0^{-1} + Σ_t Σ_t^{-1} ⊗ x_t x_t'.
       This is exact but O((nk)³) per sweep; it is not the equation-by-equation
       triangular algorithm of Carriero, Clark & Marcellino (2019).
       Draws are accepted only if the companion matrix is stable; after 50
       consecutive rejections the previous B is kept and the event is counted
       (``BVAR_SVResult.n_stuck_iterations``) and warned about.
    2. Equation-by-equation sampling of the contemporaneous relations A
       (Cogley & Sargent 2005 / Primiceri 2005 triangular regressions).
    3. Kim, Shephard & Chib (1998) 7-component Gaussian mixture approximation
       with Carter-Kohn (1994) forward-filtering backward-sampling (FFBS) for
       the log-volatility paths h_{i,t}.
    4. Sampling of the AR(1) volatility parameters (μ_i, φ_i, σ_{h,i}²):
       conjugate Normal for μ_i, independence Metropolis-Hastings for φ_i
       (KSC 1998, §3.3) and conjugate Inverse-Gamma for σ_{h,i}².
       Hyper-priors are fixed: μ_i ~ N(0, 10), φ_i ~ N(0.85, 0.1) truncated to
       |φ_i| < 0.999, σ_{h,i}² ~ IG(2, 0.05); the KSC offset is c = 1e-6.
    5. Multi-chain split-chain Gelman-Rubin convergence diagnostics (R̂ < 1.1),
       with jittered starting values for chains 2..n_chains.
    6. Volatility-conditioned impulse responses, posterior-predictive forecasts
       with fan charts, an in-sample log pointwise predictive density (lppd)
       and an out-of-sample predictive log score on a hold-out sample.

References
----------
Carriero, A., Clark, T. E., & Marcellino, M. (2016). Common drifting volatility in
    large Bayesian VARs. Journal of Business & Economic Statistics, 34(3), 375-390.
Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian vector
    autoregressions with stochastic volatility and non-conjugate priors.
    Journal of Econometrics, 212(1), 137-154.
Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: likelihood inference
    and comparison with ARCH models. The Review of Economic Studies, 65(3), 361-393.
Carter, C. K., & Kohn, R. (1994). On Gibbs sampling for state space models.
    Biometrika, 81(3), 541-553.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .._linalg import safe_cholesky
from ..mcmc import gelman_rubin
from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst
from .bvar import _univariate_sigma

# ===========================================================================
# Kim, Shephard & Chib (1998) 7-component Gaussian mixture approximation to
# log(χ²_1). Weights, means, and variances from Table 4 (shifted by -1.2704).
# ===========================================================================
_KSC_WEIGHTS = np.array([0.00730, 0.10556, 0.00002, 0.04395,
                         0.34001, 0.24566, 0.25750])
_KSC_MEANS   = np.array([-11.40039, -5.24321, -9.83726, 1.50746,
                          -0.65098,  0.52478, -2.35859])
_KSC_VARS    = np.array([5.79596, 2.61369, 5.17950, 0.16735,
                          0.64009, 0.34023, 1.26261])

# Maximum number of candidate β draws per sweep before the previous B is kept.
_MAX_STABILITY_RETRIES = 50


def _sample_mixture_indicators(y_star: np.ndarray, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample mixture component indicators s_t in {0..6} for KSC mixture."""
    T = len(y_star)
    diff = (y_star - h)[:, None] - _KSC_MEANS[None, :]
    log_phi = -0.5 * np.log(2.0 * np.pi * _KSC_VARS)[None, :] - 0.5 * diff ** 2 / _KSC_VARS[None, :]
    log_post = np.log(_KSC_WEIGHTS)[None, :] + log_phi
    log_post = log_post - log_post.max(axis=1, keepdims=True)
    p = np.exp(log_post)
    p = p / p.sum(axis=1, keepdims=True)
    cum = np.cumsum(p, axis=1)
    u = rng.uniform(size=T)[:, None]
    return (u > cum).sum(axis=1)


def _ffbs_sv_ar1(
    y_adj: np.ndarray,
    v_t: np.ndarray,
    phi: float,
    sigma_h2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Carter-Kohn forward-filtering backward-sampling for stationary AR(1) state.

    State model:
        h~_t = φ h~_{t-1} + η_t,   η_t ~ N(0, σ_h²)
        Initial: h~_0 ~ N(0, σ_h² / (1 - φ²))
    Observation:
        y_adj_t = h~_t + e_t,      e_t ~ N(0, v_t)
    """
    T = len(y_adj)
    p0 = sigma_h2 / max(1.0 - phi ** 2, 1e-6)
    a_pred = np.empty(T + 1)
    P_pred = np.empty(T + 1)
    a_filt = np.empty(T)
    P_filt = np.empty(T)

    a_pred[0] = 0.0
    P_pred[0] = p0

    for t in range(T):
        v_err = y_adj[t] - a_pred[t]
        F_k = P_pred[t] + v_t[t]
        K_k = P_pred[t] / max(F_k, 1e-12)
        a_filt[t] = a_pred[t] + K_k * v_err
        P_filt[t] = P_pred[t] * (1.0 - K_k)
        a_pred[t + 1] = phi * a_filt[t]
        P_pred[t + 1] = (phi ** 2) * P_filt[t] + sigma_h2

    # Backward sampling
    h_tilde = np.empty(T)
    h_tilde[-1] = a_filt[-1] + np.sqrt(max(P_filt[-1], 1e-12)) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        var_pred = (phi ** 2) * P_filt[t] + sigma_h2
        K_b = (phi * P_filt[t]) / max(var_pred, 1e-12)
        m_b = a_filt[t] + K_b * (h_tilde[t + 1] - phi * a_filt[t])
        v_b = max(P_filt[t] * sigma_h2 / max(var_pred, 1e-12), 1e-12)
        h_tilde[t] = m_b + np.sqrt(v_b) * rng.standard_normal()

    return h_tilde


# Hyper-prior of the AR(1) persistence: φ_i ~ N(_PHI_PRIOR_MEAN, _PHI_PRIOR_VAR), |φ_i| < 0.999.
_PHI_PRIOR_MEAN = 0.85
_PHI_PRIOR_VAR = 0.1


def _sample_phi_mh(h_dev: np.ndarray, phi_i: float, sig2_i: float, rng: np.random.Generator) -> float:
    """One independence Metropolis-Hastings update of the AR(1) persistence φ_i.

    ``h_dev`` is the demeaned log-volatility path h_{i,1:T} - μ_i. The proposal is
    the Gaussian conditional posterior of the regression h~_t = φ h~_{t-1} + η_t
    (t = 2..T) under the N(0.85, 0.1) prior, i.e. the target without the
    stationary initial-state density. The MH correction is therefore the ratio
    of initial-state densities (Kim, Shephard & Chib 1998, §3.3)

        p(h~_1 | φ) ∝ sqrt(1 - φ²) exp(-(1 - φ²) h~_1² / (2 σ_h²)),

        log[p(h~_1|φ') / p(h~_1|φ)] = ½ log(1-φ'²) - ½ log(1-φ²) + ½ (φ'² - φ²) h~_1² / σ_h².

    Proposals outside |φ| < 0.999 are rejected (stationarity truncation).
    """
    x_phi = h_dev[:-1]
    y_phi = h_dev[1:]
    v_phi_inv = (1.0 / _PHI_PRIOR_VAR) + float(np.sum(x_phi ** 2)) / sig2_i
    v_phi = 1.0 / max(v_phi_inv, 1e-12)
    m_phi = v_phi * ((_PHI_PRIOR_MEAN / _PHI_PRIOR_VAR) + float(np.sum(x_phi * y_phi)) / sig2_i)
    prop_phi = m_phi + np.sqrt(v_phi) * rng.standard_normal()

    if abs(prop_phi) < 0.999:
        log_acc = (
            0.5 * np.log(max(1.0 - prop_phi ** 2, 1e-8))
            - 0.5 * np.log(max(1.0 - phi_i ** 2, 1e-8))
            + 0.5 * (prop_phi ** 2 - phi_i ** 2) * (h_dev[0] ** 2) / sig2_i
        )
        if np.log(max(rng.uniform(), 1e-12)) < log_acc:
            return float(prop_phi)
    return float(phi_i)


def _is_stable_companion(B: np.ndarray, n: int, p: int) -> bool:
    """Check whether the companion matrix of VAR coefficients is stable (< 1)."""
    if p == 0:
        return True
    comp = np.zeros((n * p, n * p))
    for l in range(p):
        comp[:n, l * n : (l + 1) * n] = B[1 + l * n : 1 + (l + 1) * n, :].T
    if p > 1:
        comp[n:, :-n] = np.eye(n * (p - 1))
    eigs = np.linalg.eigvals(comp)
    return float(np.max(np.abs(eigs))) < 1.0


def _lag_vector(Y_lags: np.ndarray, p: int) -> np.ndarray:
    """Regressor vector [1, y_{t-1}', ..., y_{t-p}'] from the (p, n) most recent rows.

    ``Y_lags`` is ordered oldest-first, so the last row is y_{t-1}.
    """
    parts = [np.ones(1)]
    for l in range(1, p + 1):
        parts.append(Y_lags[-l])
    return np.concatenate(parts)


def _extend_index(index: pd.Index, horizon: int) -> pd.Index:
    """Extend a data index by ``horizon`` future periods (dates if possible)."""
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        freq = index.freq or pd.infer_freq(index)
        if freq is not None:
            return pd.date_range(start=index[-1], periods=horizon + 1, freq=freq)[1:]
        step = index[-1] - index[-2]
        return pd.DatetimeIndex([index[-1] + step * (j + 1) for j in range(horizon)])
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(start=index[-1] + 1, periods=horizon, freq=index.freq)
    T = len(index)
    return pd.RangeIndex(T, T + horizon)


class BVAR_SV_IRF(np.ndarray):
    """Impulse response array subclass representing posterior median and bands.

    Behaves as an ndarray of shape (H+1, n, n) (the median IRF), while
    exposing `.median`, `.lower`, `.upper`, and `.draws`.
    """

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    draws: np.ndarray | None
    horizon: int
    ci: float
    t_idx: int

    def __new__(
        cls,
        median: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        draws: np.ndarray | None = None,
        horizon: int = 20,
        ci: float = 0.9,
        t_idx: int = -1,
    ):
        obj = np.asarray(median).view(cls)
        obj.median = np.asarray(median, dtype=float)
        obj.lower = np.asarray(lower, dtype=float)
        obj.upper = np.asarray(upper, dtype=float)
        obj.draws = np.asarray(draws, dtype=float) if draws is not None else None
        obj.horizon = int(horizon)
        obj.ci = float(ci)
        obj.t_idx = int(t_idx)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        fallback = np.asarray(self, dtype=float)
        self.median = getattr(obj, "median", fallback)
        self.lower = getattr(obj, "lower", fallback)
        self.upper = getattr(obj, "upper", fallback)
        self.draws = getattr(obj, "draws", None)
        self.horizon = getattr(obj, "horizon", 20)
        self.ci = getattr(obj, "ci", 0.9)
        self.t_idx = getattr(obj, "t_idx", -1)

    def to_frame(
        self,
        target_idx: int | None = None,
        shock_idx: int | None = None,
        names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return tidy DataFrame with horizon, median response, and bands.

        ``target_idx`` and ``shock_idx`` each act as an independent filter:
        pass one of them to keep every shock for a given responding variable
        (or every response to a given shock), both to keep a single pair, or
        neither for all (target, shock) combinations.
        """
        H_plus, n_resp, n_shock = self.median.shape
        targets = range(n_resp) if target_idx is None else [int(target_idx)]
        shocks = range(n_shock) if shock_idx is None else [int(shock_idx)]
        rows = []
        for h in range(H_plus):
            for r in targets:
                for s in shocks:
                    resp_name = names[r] if names else str(r)
                    shock_name = names[s] if names else str(s)
                    rows.append({
                        "horizon": h,
                        "target": resp_name,
                        "shock": shock_name,
                        "median": float(self.median[h, r, s]),
                        "lower": float(self.lower[h, r, s]),
                        "upper": float(self.upper[h, r, s]),
                    })
        return pd.DataFrame(rows)


@dataclass
class BVAR_SVForecast:
    """Posterior-predictive forecast paths from a BVAR-SV model.

    Attributes
    ----------
    paths : np.ndarray
        Simulated future observations of shape (D, H, n): one path per
        retained posterior draw, with parameter, volatility and shock
        uncertainty all integrated out.
    h_paths : np.ndarray
        Simulated future log-volatilities of shape (D, H, n).
    index : pd.Index
        Forecast-period index (dates when the estimation data carried a
        regular DatetimeIndex, otherwise integer positions T, ..., T+H-1).
    names : list[str]
        Variable names.
    history : pd.DataFrame
        The estimation sample (used to draw the history in fan charts).
    ci : float
        Credible-interval width used for ``lower`` / ``upper``.
    """

    paths: np.ndarray
    h_paths: np.ndarray
    index: pd.Index
    names: list[str]
    history: pd.DataFrame
    ci: float = 0.9

    @property
    def horizon(self) -> int:
        """Forecast horizon H."""
        return int(self.paths.shape[1])

    @property
    def median(self) -> np.ndarray:
        """Posterior-predictive median path (H, n)."""
        return np.median(self.paths, axis=0)

    @property
    def mean(self) -> np.ndarray:
        """Posterior-predictive mean path (H, n)."""
        return np.mean(self.paths, axis=0)

    def quantile(self, q: float) -> np.ndarray:
        """Posterior-predictive quantile ``q`` in (0, 1) of shape (H, n)."""
        return np.quantile(self.paths, float(q), axis=0)

    @property
    def lower(self) -> np.ndarray:
        """Lower bound of the central ``ci`` predictive band (H, n)."""
        return self.quantile((1.0 - self.ci) / 2.0)

    @property
    def upper(self) -> np.ndarray:
        """Upper bound of the central ``ci`` predictive band (H, n)."""
        return self.quantile((1.0 + self.ci) / 2.0)

    def to_frame(self) -> pd.DataFrame:
        """Tidy table with one row per (period, variable): median, mean, lower, upper."""
        med, mean, lo, hi = self.median, self.mean, self.lower, self.upper
        rows = []
        for j in range(self.horizon):
            for i, name in enumerate(self.names):
                rows.append({
                    "horizon": j + 1,
                    "period": self.index[j],
                    "variable": name,
                    "median": float(med[j, i]),
                    "mean": float(mean[j, i]),
                    "lower": float(lo[j, i]),
                    "upper": float(hi[j, i]),
                })
        return pd.DataFrame(rows)

    def plot(
        self,
        *,
        var_idx: int | None = None,
        levels: Sequence[float] = (0.5, 0.8, 0.95),
        n_history: int = 40,
        figsize: tuple[float, float] | None = None,
        ax: Any = None,
    ):
        """Fan chart(s) of the posterior-predictive distribution.

        One panel per variable (or a single panel for ``var_idx``), showing the
        last ``n_history`` observations, the predictive median and nested
        central bands at the requested ``levels``. Returns the Figure, or the
        supplied axes when ``ax`` is given.
        """
        import matplotlib.pyplot as plt

        var_list = list(range(len(self.names))) if var_idx is None else [int(var_idx)]
        if ax is not None:
            axes = np.atleast_1d(np.asarray(ax, dtype=object)).ravel()
            if len(axes) < len(var_list):
                raise ValueError(
                    f"`ax` must contain at least {len(var_list)} subplot(s), got {len(axes)}"
                )
            fig = None
        else:
            fig, axes_raw = plt.subplots(
                1, len(var_list), figsize=figsize or (5.0 * len(var_list), 3.6), squeeze=False
            )
            axes = axes_raw.ravel()

        hist = self.history.iloc[-int(n_history):] if n_history > 0 else self.history.iloc[0:0]
        hist_x = np.asarray(hist.index)
        fc_x = np.asarray(self.index)
        sorted_levels = sorted(float(l) for l in levels)
        sorted_levels.reverse()
        shades = np.linspace(0.85, 0.55, max(len(sorted_levels), 1))
        for panel, i in enumerate(var_list):
            a = axes[panel]
            if len(hist) > 0:
                a.plot(hist_x, hist.iloc[:, i].to_numpy(dtype=float), color="navy", linewidth=1.2, label="history")
            for shade, level in zip(shades, sorted_levels):
                lo = self.quantile((1.0 - level) / 2.0)[:, i]
                hi = self.quantile((1.0 + level) / 2.0)[:, i]
                a.fill_between(fc_x, lo, hi, color=f"{shade:.3f}", linewidth=0, label=f"{int(round(level * 100))}%")
            a.plot(fc_x, self.median[:, i], color="0.0", linewidth=1.4, label="median")
            a.set_title(f"Forecast fan chart: {self.names[i]}")
            a.grid(True, linestyle="--", alpha=0.5)
            a.legend(loc="best", frameon=False)

        if fig is not None:
            fig.tight_layout()
            return fig
        return ax


@dataclass
class BVAR_SVResult:
    """Posterior estimation results for BVAR with Stochastic Volatility.

    All ``*_draws`` arrays pool the retained draws of every chain, so their
    leading dimension is ``D = n_chains * n_draws`` (``n_total_draws``).

    Attributes
    ----------
    beta_draws : np.ndarray
        MCMC draws of VAR coefficient matrix B of shape (D, 1 + n*p, n).
    h_draws : np.ndarray
        MCMC draws of log-volatilities of shape (D, T_eff, n).
    a_draws : np.ndarray
        MCMC draws of contemporaneous impact matrix A of shape (D, n, n).
    r_hat : dict[str, float]
        Gelman-Rubin split-Rhat convergence diagnostics (also ``rhat``);
        ``r_hat['max']`` (also ``max_rhat``) is the largest finite value.
    data : pd.DataFrame
        Original endogenous variables panel.
    lags : int
        VAR lag order p.
    mu_draws : np.ndarray
        AR(1) volatility mean draws (D, n).
    phi_draws : np.ndarray
        AR(1) persistence draws (D, n).
    sigma_h_draws : np.ndarray
        AR(1) innovation volatility draws (D, n).
    log_scores : np.ndarray
        In-sample log pointwise predictive density (lppd) of each estimation
        observation, shape (T_eff,): log of the posterior average of
        p(y_t | β, A, h_t) with the *smoothed* volatility draws h_t. This is a
        fit measure, not an out-of-sample score; see ``log_score(holdout)``.
    n_draws : int
        Retained draws per chain (as requested).
    n_burn : int
        Number of burned initial draws per chain.
    n_chains : int
        Number of chains.
    n_unstable_rejections : int
        Total number of candidate β draws rejected by the companion-stability
        check (all chains, burn-in included).
    n_stuck_iterations : int
        Number of sweeps in which no stable β candidate was found within the
        retry budget and the previous B was kept (all chains, burn-in included).
    """

    beta_draws: np.ndarray
    h_draws: np.ndarray
    a_draws: np.ndarray
    r_hat: dict[str, float]
    data: pd.DataFrame
    lags: int
    mu_draws: np.ndarray
    phi_draws: np.ndarray
    sigma_h_draws: np.ndarray
    log_scores: np.ndarray
    n_draws: int
    n_burn: int
    n_chains: int = 1
    n_unstable_rejections: int = 0
    n_stuck_iterations: int = 0
    irf_median: np.ndarray | None = None
    irf_lower: np.ndarray | None = None
    irf_upper: np.ndarray | None = None

    @property
    def p(self) -> int:
        """Alias for lag order."""
        return self.lags

    @property
    def n(self) -> int:
        """Number of endogenous variables."""
        return self.h_draws.shape[2]

    @property
    def T_eff(self) -> int:
        """Effective sample size (T - p)."""
        return self.h_draws.shape[1]

    @property
    def n_total_draws(self) -> int:
        """Pooled number of retained draws D = n_chains * n_draws."""
        return int(self.beta_draws.shape[0])

    @property
    def names(self) -> list[str]:
        """Names of endogenous variables."""
        return list(self.data.columns.astype(str))

    @property
    def rhat(self) -> dict[str, float]:
        """Alias of ``r_hat``: split-chain Gelman-Rubin diagnostics by parameter group."""
        return self.r_hat

    @property
    def max_rhat(self) -> float:
        """Largest finite split-chain R-hat across all monitored parameters."""
        return float(self.r_hat.get("max", float("nan")))

    @property
    def A_draws(self) -> np.ndarray:
        """Draws of VAR lag coefficient matrices (D, p, n, n)."""
        D = self.beta_draws.shape[0]
        n = self.n
        p = self.lags
        out = np.empty((D, p, n, n))
        for l in range(p):
            out[:, l] = self.beta_draws[:, 1 + l * n : 1 + (l + 1) * n, :].transpose(0, 2, 1)
        return out

    @property
    def intercept_draws(self) -> np.ndarray:
        """Draws of VAR intercepts (D, n)."""
        return self.beta_draws[:, 0, :]

    def gelman_rubin(self) -> dict[str, float]:
        """Return Gelman-Rubin convergence diagnostics dictionary."""
        return self.r_hat

    def predictive_log_score(self, point_by_point: bool = False) -> float | np.ndarray:
        """In-sample log pointwise predictive density (lppd).

        Returns ``sum_t log_scores[t]`` (or the pointwise array when
        ``point_by_point=True``). The volatilities entering each term are the
        smoothed posterior draws h_t, which condition on the whole estimation
        sample including y_t itself, so this is an in-sample fit measure. For
        an out-of-sample predictive score use ``log_score(holdout)``.
        """
        if point_by_point:
            return self.log_scores.copy()
        return float(np.sum(self.log_scores))

    def _log_density_draws(self, y_t: np.ndarray, x_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
        """log p(y_t | β_d, A_d, h_t^d) for every draw d, given regressors x_t and (D, n) log-vols."""
        n = self.n
        u_draws = y_t[None, :] - (self.beta_draws.transpose(0, 2, 1) @ x_t)  # (D, n)
        nu_draws = np.einsum("dij,dj->di", self.a_draws, u_draws)            # (D, n)
        return (
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * np.sum(h_t, axis=1)
            - 0.5 * np.sum(np.exp(-h_t) * (nu_draws ** 2), axis=1)
        )

    def log_score(
        self,
        holdout: pd.DataFrame | np.ndarray | None = None,
        *,
        point_by_point: bool = False,
        seed: int | None = None,
    ) -> float | np.ndarray:
        """Log predictive score.

        Parameters
        ----------
        holdout : DataFrame or ndarray of shape (H, n), optional
            Observations that follow the estimation sample. When given, the
            method returns the **out-of-sample** log predictive score
            ``sum_j log p(y_{T+j} | y_{1:T+j-1}, posterior)``, where for every
            retained draw the conditional mean uses the realised lags of
            y_{T+j} and the volatility h_{T+j} is projected forward from the
            end-of-sample state through the AR(1) law of motion
            h_{T+j} = μ + φ (h_{T+j-1} - μ) + σ_h η (no re-estimation and no
            re-filtering of the volatility on hold-out data). When omitted the
            in-sample lppd of ``predictive_log_score`` is returned.
        point_by_point : bool, default False
            Return the per-observation array instead of the total.
        seed : int, optional
            Seed for the volatility projection (hold-out case only).
        """
        if holdout is None:
            return self.predictive_log_score(point_by_point=point_by_point)

        Y_new = np.asarray(holdout.values if isinstance(holdout, pd.DataFrame) else holdout, dtype=float)
        if Y_new.ndim == 1 and self.n == 1:
            Y_new = Y_new[:, None]
        if Y_new.ndim != 2 or Y_new.shape[1] != self.n:
            raise ValueError(f"holdout must have shape (H, {self.n}); got {Y_new.shape}")
        if Y_new.shape[0] == 0:
            raise ValueError("holdout must contain at least one observation")
        if not np.all(np.isfinite(Y_new)):
            raise ValueError("holdout contains NaN or infinite values")

        p = self.lags
        n = self.n
        D = self.n_total_draws
        rng = np.random.default_rng(seed)
        Y_hist = np.vstack([np.asarray(self.data.values, dtype=float)[-p:], Y_new])
        h_prev = self.h_draws[:, -1, :].copy()
        scores = np.empty(Y_new.shape[0])
        for j in range(Y_new.shape[0]):
            x_t = _lag_vector(Y_hist[j : j + p], p)
            h_t = self.mu_draws + self.phi_draws * (h_prev - self.mu_draws) \
                + self.sigma_h_draws * rng.standard_normal((D, n))
            log_p_d = self._log_density_draws(Y_new[j], x_t, h_t)
            scores[j] = float(logsumexp(log_p_d) - np.log(D))
            h_prev = h_t
        if point_by_point:
            return scores
        return float(np.sum(scores))

    def forecast(
        self,
        horizon: int = 8,
        *,
        ci: float = 0.9,
        seed: int | None = None,
    ) -> BVAR_SVForecast:
        """Simulate posterior-predictive paths H periods beyond the sample.

        For every retained draw d the log-volatilities are propagated with
        h_{T+j} = μ + φ (h_{T+j-1} - μ) + σ_h η, the structural shocks are drawn
        ε ~ N(0, I), u = A^{-1} D_{T+j}^{1/2} ε, and y_{T+j} = c + Σ_l A_l y_{T+j-l} + u
        is iterated forward on the simulated path. The result integrates over
        parameter, volatility and shock uncertainty and offers fan charts.
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not (0.0 < ci < 1.0):
            raise ValueError(f"ci must be in (0, 1), got {ci}")
        p = self.lags
        n = self.n
        D = self.n_total_draws
        rng = np.random.default_rng(seed)

        Y_arr = np.asarray(self.data.values, dtype=float)
        lag_buf = np.repeat(Y_arr[-p:][None, :, :], D, axis=0)  # (D, p, n), oldest first
        h_prev = self.h_draws[:, -1, :].copy()
        paths = np.empty((D, horizon, n))
        h_paths = np.empty((D, horizon, n))
        for j in range(horizon):
            h_t = self.mu_draws + self.phi_draws * (h_prev - self.mu_draws) \
                + self.sigma_h_draws * rng.standard_normal((D, n))
            nu = np.exp(h_t / 2.0) * rng.standard_normal((D, n))
            u = np.linalg.solve(self.a_draws, nu[:, :, None])[:, :, 0]
            x = np.concatenate([np.ones((D, 1))] + [lag_buf[:, -l, :] for l in range(1, p + 1)], axis=1)
            y = np.einsum("dk,dkn->dn", x, self.beta_draws) + u
            paths[:, j] = y
            h_paths[:, j] = h_t
            lag_buf = np.concatenate([lag_buf[:, 1:, :], y[:, None, :]], axis=1)
            h_prev = h_t

        return BVAR_SVForecast(
            paths=paths,
            h_paths=h_paths,
            index=_extend_index(self.data.index, horizon),
            names=self.names,
            history=self.data.copy(),
            ci=float(ci),
        )

    def irf(
        self,
        horizon: int = 20,
        t_idx: int = -1,
        ci: float = 0.9,
    ) -> BVAR_SV_IRF:
        """Compute posterior impulse responses conditioned on the volatility state at date t*.

        Parameters
        ----------
        horizon : int, default 20
            Impulse response horizon H.
        t_idx : int, default -1
            Date index in [0, T_eff-1] (or negative index) to condition the
            volatility matrix on.
        ci : float, default 0.9
            Credible interval width (e.g. 0.9 for 90% bands).

        Returns
        -------
        BVAR_SV_IRF
            Array of shape (H+1, n, n) containing the median response, with
            `.lower`, `.upper`, and `.draws` attributes.
        """
        D = self.beta_draws.shape[0]
        n = self.n
        p = self.lags
        T_eff = self.T_eff

        if t_idx < 0:
            t_idx = T_eff + t_idx
        if not (0 <= t_idx < T_eff):
            raise ValueError(f"t_idx={t_idx} is out of bounds for T_eff={T_eff}")

        irf_draws = np.empty((D, horizon + 1, n, n))

        for d in range(D):
            # Extract lag matrices A_1..A_p: A_l has shape (n, n)
            A_list = [
                self.beta_draws[d, 1 + l * n : 1 + (l + 1) * n, :].T
                for l in range(p)
            ]

            # Structural impact matrix at date t_idx:
            # B0 = A^{-1} diag(exp(h_{t_idx} / 2))
            A_mat = self.a_draws[d]
            scale_t = np.exp(self.h_draws[d, t_idx, :] / 2.0)
            D_half = np.diag(scale_t)
            try:
                B0 = np.linalg.solve(A_mat, D_half)
            except np.linalg.LinAlgError:
                B0 = np.linalg.pinv(A_mat) @ D_half

            # MA recursion
            Phi = [np.eye(n)]
            for h in range(1, horizon + 1):
                Ph = np.zeros((n, n))
                for j in range(1, min(h, p) + 1):
                    Ph += Phi[h - j] @ A_list[j - 1]
                Phi.append(Ph)

            for h in range(horizon + 1):
                irf_draws[d, h] = Phi[h] @ B0

        alpha_low = 100.0 * (1.0 - ci) / 2.0
        alpha_high = 100.0 * (1.0 + ci) / 2.0

        median = np.median(irf_draws, axis=0)
        lower = np.percentile(irf_draws, alpha_low, axis=0)
        upper = np.percentile(irf_draws, alpha_high, axis=0)

        self.irf_median = median
        self.irf_lower = lower
        self.irf_upper = upper

        return BVAR_SV_IRF(
            median=median,
            lower=lower,
            upper=upper,
            draws=irf_draws,
            horizon=horizon,
            ci=ci,
            t_idx=t_idx,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return summary DataFrame of posterior parameters and diagnostics."""
        rows = []
        for i, name in enumerate(self.names):
            mu_m = float(np.mean(self.mu_draws[:, i]))
            mu_s = float(np.std(self.mu_draws[:, i], ddof=1))
            phi_m = float(np.mean(self.phi_draws[:, i]))
            phi_s = float(np.std(self.phi_draws[:, i], ddof=1))
            sig_m = float(np.mean(self.sigma_h_draws[:, i]))
            sig_s = float(np.std(self.sigma_h_draws[:, i], ddof=1))
            h_m = float(np.mean(self.h_draws[:, :, i]))
            rhat_val = float(self.r_hat.get(f"h_{name}_mean", self.r_hat.get("max", 1.0)))

            rows.append({
                "variable": name,
                "mu_mean": round(mu_m, 4),
                "mu_sd": round(mu_s, 4),
                "phi_mean": round(phi_m, 4),
                "phi_sd": round(phi_s, 4),
                "sigma_h_mean": round(sig_m, 4),
                "sigma_h_sd": round(sig_s, 4),
                "log_vol_mean": round(h_m, 4),
                "R_hat": round(rhat_val, 4),
            })
        return pd.DataFrame(rows)

    def summary(self, ci: float = 0.9) -> str:
        """Render a text summary: sampler settings, convergence, fit, and the
        posterior median with a central ``ci`` credible interval and the
        split-chain R-hat of every log-volatility AR(1) parameter."""
        max_rhat = self.max_rhat
        if not np.isfinite(max_rhat):
            conv_flag = "UNDIAGNOSED (too few draws for a split R̂)"
        else:
            conv_flag = "CONVERGED (R̂ < 1.1)" if max_rhat < 1.1 else "WARNING (R̂ >= 1.1)"
        total_ls = float(np.sum(self.log_scores))
        mean_ls = float(np.mean(self.log_scores))
        q_lo, q_hi = 50.0 * (1.0 - ci), 50.0 * (1.0 + ci)

        lines = [
            "==================================================================",
            "  Bayesian VAR with Stochastic Volatility (BVAR-SV)",
            "==================================================================",
            f"  Variables (n)       : {self.n} ({', '.join(self.names)})",
            f"  Sample size (T)     : {len(self.data)} (Effective T = {self.T_eff})",
            f"  Lag order (p)       : {self.lags}",
            f"  Retained draws (D)  : {self.n_total_draws} "
            f"({self.n_chains} chain(s) x {self.n_draws}, burn-in = {self.n_burn} per chain)",
            f"  Convergence status  : {conv_flag} [Max R̂ = {max_rhat:.4f}]",
            f"  In-sample lppd      : Total = {total_ls:.2f}, Mean = {mean_ls:.4f}"
            " (smoothed volatilities; not out-of-sample)",
        ]
        if self.n_stuck_iterations > 0:
            lines.append(
                f"  Stability rejections: {self.n_unstable_rejections} candidate draws rejected; "
                f"previous B kept in {self.n_stuck_iterations} sweep(s)"
            )
        lines += [
            "------------------------------------------------------------------",
            f"  Log-Volatility AR(1) Parameters (posterior median [{int(round(ci * 100))}% CI]; split R̂):",
        ]
        for i, name in enumerate(self.names):
            for label, arr, key in (
                ("μ", self.mu_draws[:, i], f"mu_{name}"),
                ("φ", self.phi_draws[:, i], f"phi_{name}"),
                ("σ_h", self.sigma_h_draws[:, i], f"sigma_h_{name}"),
            ):
                med = float(np.median(arr))
                lo, hi = np.percentile(arr, [q_lo, q_hi])
                rh = self.r_hat.get(key, float("nan"))
                lines.append(
                    f"    {name:<12} {label:<3}= {med:+.3f} [{lo:+.3f}, {hi:+.3f}]  R̂ = {rh:.3f}"
                )
        lines.append("==================================================================")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format summary table as Markdown."""
        return _df_to_markdown(self.to_frame(), index=False)

    def to_latex(self) -> str:
        """Format summary table as LaTeX tabular."""
        return _df_to_latex(self.to_frame(), index=False)

    def to_typst(self) -> str:
        """Format summary table as Typst table."""
        return _df_to_typst(self.to_frame(), index=False)

    def plot(
        self,
        *,
        t_idx: int = -1,
        horizon: int = 20,
        ci: float = 0.9,
        shock_idx: int = 0,
        target_idx: int | None = None,
        figsize: tuple[float, float] | None = None,
        ax: Any = None,
    ):
        """Create a 3-panel plot of BVAR-SV results.

        Panel 1: Log-volatilities h_{i,t} with credible bands, one line per
                 variable (or only ``target_idx`` when given).
        Panel 2: Conditional standard deviations exp(h_{i,t} / 2), same layout.
        Panel 3: Volatility-conditioned impulse response of variable
                 ``target_idx`` (first variable when None) to shock ``shock_idx``
                 at date ``t_idx``, with credible bands.
        """
        import matplotlib.pyplot as plt

        alpha_low = 100.0 * (1.0 - ci) / 2.0
        alpha_high = 100.0 * (1.0 + ci) / 2.0

        h_med = np.median(self.h_draws, axis=0)
        h_low = np.percentile(self.h_draws, alpha_low, axis=0)
        h_high = np.percentile(self.h_draws, alpha_high, axis=0)

        sd_draws = np.exp(self.h_draws / 2.0)
        sd_med = np.median(sd_draws, axis=0)
        sd_low = np.percentile(sd_draws, alpha_low, axis=0)
        sd_high = np.percentile(sd_draws, alpha_high, axis=0)

        irf_res = self.irf(horizon=horizon, t_idx=t_idx, ci=ci)

        if ax is not None:
            if isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 3:
                fig = None
                axes = ax
            else:
                raise ValueError("If `ax` is supplied, it must contain 3 subplots.")
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 4))

        vol_vars = list(range(self.n)) if target_idx is None else [int(target_idx)]
        irf_target = 0 if target_idx is None else int(target_idx)
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["navy"])

        t_axis = np.arange(self.T_eff)
        # 1. Log-volatilities
        for j, i in enumerate(vol_vars):
            col = colors[j % len(colors)]
            axes[0].plot(t_axis, h_med[:, i], label=f"{self.names[i]} (median)", color=col)
            axes[0].fill_between(t_axis, h_low[:, i], h_high[:, i], color=col, alpha=0.2)
        axes[0].set_title(f"Log-Volatility $h_{{t}}$ ({int(ci*100)}% CI)")
        axes[0].set_xlabel("Time (effective)")
        axes[0].set_ylabel("Log-Variance")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="best")

        # 2. Conditional SD
        for j, i in enumerate(vol_vars):
            col = colors[j % len(colors)]
            axes[1].plot(t_axis, sd_med[:, i], label=f"{self.names[i]} (median)", color=col)
            axes[1].fill_between(t_axis, sd_low[:, i], sd_high[:, i], color=col, alpha=0.2)
        axes[1].set_title(f"Cond. Std. Dev. $\\exp(h_{{t}}/2)$ ({int(ci*100)}% CI)")
        axes[1].set_xlabel("Time (effective)")
        axes[1].set_ylabel("Std Dev")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="best")

        # 3. Volatility-conditioned IRF
        h_axis = np.arange(horizon + 1)
        axes[2].plot(h_axis, irf_res.median[:, irf_target, shock_idx], label="Median IRF", color="darkgreen")
        axes[2].fill_between(
            h_axis,
            irf_res.lower[:, irf_target, shock_idx],
            irf_res.upper[:, irf_target, shock_idx],
            color="darkgreen",
            alpha=0.25,
            label=f"{int(ci*100)}% CI",
        )
        axes[2].axhline(0, color="black", linestyle=":", linewidth=1)
        axes[2].set_title(
            f"IRF: {self.names[shock_idx]} $\\to$ {self.names[irf_target]} (date {irf_res.t_idx})"
        )
        axes[2].set_xlabel("Horizon")
        axes[2].set_ylabel("Response")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="best")

        if fig is not None:
            fig.tight_layout()
            return fig
        return axes


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer-valued argument with a lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {int(value)}")
    return int(value)


def bvar_sv(
    data: pd.DataFrame | np.ndarray,
    lags: int = 4,
    n_draws: int = 2000,
    n_burn: int = 1000,
    minnesota_prior: bool = True,
    seed: int | None = None,
    *,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    thin: int = 1,
    n_chains: int = 2,
    p: int | None = None,
) -> BVAR_SVResult:
    """Fit Bayesian VAR with Stochastic Volatility via MCMC Gibbs sampler.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray, shape (T, n)
        Endogenous variables panel. Must be finite (no NaN / inf).
    lags : int, default 4
        VAR lag order. The effective sample T - lags must exceed the number
        of regressors per equation k = 1 + n * lags.
    n_draws : int, default 2000
        Number of post-burn-in MCMC draws to retain **per chain** (>= 2). The
        result pools ``n_chains * n_draws`` draws.
    n_burn : int, default 1000
        Number of burn-in iterations per chain (>= 0).
    minnesota_prior : bool, default True
        Whether to impose Minnesota shrinkage on VAR lag coefficients. When
        False the lag coefficients get an independent diffuse N(0, 100²)
        prior; the intercept prior is ``intercept_prior_std`` in both cases.
    seed : int, optional
        RNG seed for exact reproducibility.
    lambda1 : float, default 0.2
        Overall Minnesota prior shrinkage.
    lambda2 : float, default 0.5
        Cross-variable Minnesota prior shrinkage.
    lambda3 : float, default 1.0
        Lag-decay exponent for Minnesota prior.
    intercept_prior_std : float, default 1e3
        Prior standard deviation of the intercepts (diffuse by default).
    thin : int, default 1
        Thinning interval for the MCMC chain (>= 1): one draw is retained
        every ``thin`` post-burn-in sweeps until ``n_draws`` are collected.
    n_chains : int, default 2
        Number of independent MCMC chains (>= 1). Chains after the first start
        from jittered initial values so split-R-hat reflects genuine mixing.
    p : int, optional
        Positional / keyword alias for ``lags``.

    Returns
    -------
    BVAR_SVResult
        Rich dataclass containing draws, Gelman-Rubin convergence diagnostics,
        IRFs, forecasts, log predictive scores, and presentation methods.

    Raises
    ------
    ValueError
        On non-finite data, a 1-D array, ``lags < 1``, ``n_draws < 2``,
        ``n_burn < 0``, ``thin < 1``, ``n_chains < 1`` or too few observations.

    Warns
    -----
    UserWarning
        When at least one sweep exhausted the stability retry budget and kept
        the previous B (see ``BVAR_SVResult.n_stuck_iterations``).
    """
    if p is not None:
        lags = p
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or int(lags) <= 0:
        raise ValueError(f"lags must be a positive integer, got {lags}")
    lags = int(lags)
    if isinstance(n_draws, (int, np.integer)) and not isinstance(n_draws, bool) and int(n_draws) <= 0:
        raise ValueError("n_draws must be > 0 and n_burn must be >= 0")
    n_draws = _check_int(n_draws, "n_draws", 2)
    n_burn = _check_int(n_burn, "n_burn", 0)
    thin = _check_int(thin, "thin", 1)
    n_chains = _check_int(n_chains, "n_chains", 1)

    if isinstance(data, pd.DataFrame):
        df_data = data.copy()
        var_names = list(data.columns.astype(str))
        Y_arr = np.asarray(data.values, dtype=float)
    else:
        Y_arr = np.asarray(data, dtype=float)
        if Y_arr.ndim != 2:
            raise ValueError("data must be a 2D array or DataFrame")
        var_names = [f"y{i+1}" for i in range(Y_arr.shape[1])]
        df_data = pd.DataFrame(Y_arr, columns=var_names)
    if Y_arr.ndim != 2:
        raise ValueError("data must be a 2D array or DataFrame")
    if not np.all(np.isfinite(Y_arr)):
        raise ValueError("data contains NaN or infinite values; drop or impute them before calling bvar_sv")

    T, n = Y_arr.shape
    if n < 1:
        raise ValueError("data must contain at least one variable")
    if T <= lags + 2:
        raise ValueError(f"Insufficient observations: T={T} must exceed lags+2={lags+2}")

    T_eff = T - lags
    k = 1 + n * lags
    if T_eff <= k:
        raise ValueError(
            f"Insufficient observations: effective sample T_eff = T - lags = {T_eff} must exceed the "
            f"number of regressors per equation k = 1 + n*lags = {k}; reduce lags (n={n}, T={T})"
        )

    # Build design matrix X and dependent variable Y_dep
    Y_dep = Y_arr[lags:]  # (T_eff, n)
    X_rows = []
    for t in range(lags, T):
        row = [1.0]
        for lag_idx in range(1, lags + 1):
            row.extend(Y_arr[t - lag_idx])
        X_rows.append(row)
    X = np.asarray(X_rows, dtype=float)  # (T_eff, k)

    # Prior on vec(B') (equation-major ordering: index i*k + j is coefficient j of equation i)
    beta0 = np.zeros(n * k)
    if minnesota_prior:
        sigmas = np.array([_univariate_sigma(Y_arr[:, i], lags) for i in range(n)])
        v0 = np.ones(n * k) * (intercept_prior_std ** 2)

        for i in range(n):
            # Intercept
            v0[i * k] = intercept_prior_std ** 2
            # Own lag 1 prior mean is 1.0
            beta0[i * k + 1 + i] = 1.0

            for lag in range(1, lags + 1):
                decay = lag ** lambda3
                for j in range(n):
                    idx = i * k + 1 + (lag - 1) * n + j
                    if i == j:
                        # Own lag
                        std = lambda1 / decay
                    else:
                        # Cross lag
                        std = (lambda1 * lambda2 * sigmas[i]) / max(decay * sigmas[j], 1e-12)
                    v0[idx] = max(std ** 2, 1e-12)
    else:
        # Diffuse N(0, 100^2) on lag coefficients; intercept_prior_std on the intercepts.
        v0 = np.ones(n * k) * 1e4
        for i in range(n):
            v0[i * k] = intercept_prior_std ** 2

    V0_inv = np.diag(1.0 / v0)
    V0_inv_beta0 = V0_inv @ beta0

    # Outer product of regressors for vectorized GLS
    X_outer = X[:, :, None] * X[:, None, :]  # (T_eff, k, k)

    base_rng = np.random.default_rng(seed)

    draws_per_chain = n_draws
    n_unstable_rejections = 0
    n_stuck_iterations = 0

    all_chains_beta = []
    all_chains_h = []
    all_chains_a = []
    all_chains_mu = []
    all_chains_phi = []
    all_chains_sigma_h = []

    # OLS starting point for B (shared by all chains)
    try:
        B_init, *_ = np.linalg.lstsq(X, Y_dep, rcond=None)
    except Exception:
        B_init = np.zeros((k, n))
    # The sampler truncates the coefficient posterior to the stationary region,
    # so the chains must start inside it: shrink the OLS lag coefficients (rows
    # 1..k-1; row 0 is the intercept) until the companion matrix is stable.
    # Otherwise an explosive OLS start could be retained as a "draw" whenever
    # the stability retries are exhausted.
    _shrink = 0
    while not _is_stable_companion(B_init, n, lags) and _shrink < 200:
        B_init[1:] *= 0.97
        _shrink += 1

    for chain_id in range(n_chains):
        # Chain-specific RNG
        chain_seed = base_rng.integers(0, 2**31 - 1)
        rng = np.random.default_rng(chain_seed)

        # Initial parameter states
        B_cur = B_init.copy()
        A_cur = np.eye(n)
        U_init = Y_dep - X @ B_cur
        h_cur = np.zeros((T_eff, n))
        for i in range(n):
            resid_var = float(np.var(U_init[:, i], ddof=1)) if len(U_init) > 1 else 1.0
            h_cur[:, i] = np.log(max(resid_var, 1e-4))

        phi_cur = np.full(n, 0.85)
        sigma_h2_cur = np.full(n, 0.05)
        if chain_id > 0:
            # Over-dispersed starts so that split-R-hat measures genuine mixing
            h_cur = h_cur + rng.normal(0.0, 0.5, size=(1, n))
            phi_cur = np.asarray(rng.uniform(0.5, 0.95, size=n), dtype=float)
            sigma_h2_cur = np.exp(np.asarray(rng.uniform(np.log(0.02), np.log(0.2), size=n), dtype=float))
        mu_cur = np.mean(h_cur, axis=0)

        chain_beta = []
        chain_h = []
        chain_a = []
        chain_mu = []
        chain_phi = []
        chain_sigma_h = []

        total_iters = n_burn + draws_per_chain * thin

        for it in range(total_iters):
            # -------------------------------------------------------------
            # 1. Sample VAR coefficients β conditional on A and h_{1:T}
            # -------------------------------------------------------------
            exp_neg_h = np.exp(-h_cur)  # (T_eff, n)
            # Precision matrix of residuals: Σ_t^{-1} = A' diag(exp(-h_t)) A
            Sigma_inv = np.einsum("ji,tj,jk->tik", A_cur, exp_neg_h, A_cur)
            Sigma_inv = 0.5 * (Sigma_inv + np.swapaxes(Sigma_inv, 1, 2))

            # Joint precision: V_post^{-1} = V0^{-1} + sum_t Σ_t^{-1} ⊗ (x_t x_t')
            prec = V0_inv + np.einsum("tij,tkl->ikjl", Sigma_inv, X_outer).reshape(n * k, n * k)
            prec = 0.5 * (prec + prec.T)
            rhs = V0_inv_beta0 + np.einsum("tij,tj,tk->ik", Sigma_inv, Y_dep, X).ravel()

            diag_p = np.diag(prec)
            jitter_p = 1e-11 * float(np.max(diag_p)) if diag_p.size else 1e-11
            L_prec = safe_cholesky(prec, name="bvar_sv_beta_precision", jitter=jitter_p)

            beta_mean = np.linalg.solve(prec, rhs)

            # Sample β with companion stability rejection (bounded retries)
            stable_draw = False
            for _ in range(_MAX_STABILITY_RETRIES):
                z_b = rng.standard_normal(n * k)
                beta_cand = beta_mean + np.linalg.solve(L_prec.T, z_b)
                B_cand = beta_cand.reshape(n, k).T
                if _is_stable_companion(B_cand, n, lags):
                    B_cur = B_cand
                    stable_draw = True
                    break
                n_unstable_rejections += 1
            if not stable_draw:
                # Retry budget exhausted: keep the previous B and record the event.
                n_stuck_iterations += 1

            # -------------------------------------------------------------
            # 2. Sample contemporaneous coefficients A equation-by-equation
            # -------------------------------------------------------------
            U = Y_dep - X @ B_cur
            for i in range(1, n):
                # Equation i: u_{i,t} = - sum_{j<i} a_{i,j} u_{j,t} + ν_{i,t}
                w_t = -U[:, :i]  # (T_eff, i)
                u_t = U[:, i]    # (T_eff,)
                weight_i = np.exp(-h_cur[:, i] / 2.0)
                w_star = w_t * weight_i[:, None]
                u_star = u_t * weight_i

                # Normal prior N(0, 100 * I)
                prec_a = 0.01 * np.eye(i) + w_star.T @ w_star
                prec_a = 0.5 * (prec_a + prec_a.T)
                rhs_a = w_star.T @ u_star

                diag_a = np.diag(prec_a)
                jitter_a = 1e-11 * float(np.max(diag_a)) if diag_a.size else 1e-11
                L_a = safe_cholesky(prec_a, name=f"bvar_sv_A_row_{i}", jitter=jitter_a)

                a_mean = np.linalg.solve(prec_a, rhs_a)
                z_a = rng.standard_normal(i)
                a_draw = a_mean + np.linalg.solve(L_a.T, z_a)
                A_cur[i, :i] = a_draw

            # -------------------------------------------------------------
            # 3. Sample log-volatilities h_{i,t} via KSC (1998) + Carter-Kohn FFBS
            # -------------------------------------------------------------
            nu = U @ A_cur.T  # (T_eff, n)
            offset = 1e-6
            y_star = np.log(nu ** 2 + offset)

            for i in range(n):
                s_i = _sample_mixture_indicators(y_star[:, i], h_cur[:, i], rng)
                m_t = _KSC_MEANS[s_i]
                v_t = _KSC_VARS[s_i]

                # FFBS for demeaned AR(1) state h~_{i,t} = h_{i,t} - μ_i
                y_adj = y_star[:, i] - mu_cur[i] - m_t
                h_tilde = _ffbs_sv_ar1(y_adj, v_t, phi_cur[i], sigma_h2_cur[i], rng)
                h_cur[:, i] = mu_cur[i] + h_tilde

                # ---------------------------------------------------------
                # 4. Sample AR(1) parameters (μ_i, φ_i, σ_{h,i}²)
                # ---------------------------------------------------------
                phi_i = phi_cur[i]
                sig2_i = sigma_h2_cur[i]

                # Sample μ_i ~ N(m_μ, V_μ)  (prior N(0, 10))
                v_mu_inv = (
                    (1.0 / 10.0)
                    + (1.0 - phi_i ** 2) / sig2_i
                    + ((1.0 - phi_i) ** 2) * (T_eff - 1) / sig2_i
                )
                v_mu = 1.0 / max(v_mu_inv, 1e-12)
                m_mu = v_mu * (
                    ((1.0 - phi_i ** 2) / sig2_i) * h_cur[0, i]
                    + ((1.0 - phi_i) / sig2_i) * float(np.sum(h_cur[1:, i] - phi_i * h_cur[:-1, i]))
                )
                mu_cur[i] = m_mu + np.sqrt(v_mu) * rng.standard_normal()

                # Sample φ_i: independence MH (KSC 1998, §3.3), see _sample_phi_mh.
                h_dev = h_cur[:, i] - mu_cur[i]
                phi_cur[i] = _sample_phi_mh(h_dev, phi_i, sig2_i, rng)

                # Sample σ_{h,i}² ~ IG(a_post, b_post)  (prior IG(2, 0.05))
                res_h = h_dev[1:] - phi_cur[i] * h_dev[:-1]
                a_post = 2.0 + T_eff / 2.0
                b_post = 0.05 + 0.5 * ((1.0 - phi_cur[i] ** 2) * (h_dev[0] ** 2) + float(np.sum(res_h ** 2)))
                sigma_h2_cur[i] = 1.0 / max(rng.gamma(a_post, 1.0 / b_post), 1e-12)

            # Store retained draws
            if it >= n_burn and (it - n_burn + 1) % thin == 0:
                chain_beta.append(B_cur.copy())
                chain_h.append(h_cur.copy())
                chain_a.append(A_cur.copy())
                chain_mu.append(mu_cur.copy())
                chain_phi.append(phi_cur.copy())
                chain_sigma_h.append(np.sqrt(sigma_h2_cur).copy())

        all_chains_beta.append(np.array(chain_beta))
        all_chains_h.append(np.array(chain_h))
        all_chains_a.append(np.array(chain_a))
        all_chains_mu.append(np.array(chain_mu))
        all_chains_phi.append(np.array(chain_phi))
        all_chains_sigma_h.append(np.array(chain_sigma_h))

    if n_stuck_iterations > 0:
        warnings.warn(
            f"bvar_sv: no stable VAR coefficient draw was found within {_MAX_STABILITY_RETRIES} "
            f"attempts in {n_stuck_iterations} sweep(s) ({n_unstable_rejections} candidates rejected "
            "in total); the previous B was kept in those sweeps. The posterior is truncated to the "
            "stationary region and may be unreliable if the data are near-explosive.",
            UserWarning,
            stacklevel=2,
        )

    # Gelman-Rubin split-chain convergence diagnostics
    # Standard split-chain: split each chain into 2 halves
    split_chains_for_gr = []
    for c in range(n_chains):
        half_pt = draws_per_chain // 2
        split_chains_for_gr.append((c, 0, half_pt))
        split_chains_for_gr.append((c, half_pt, 2 * half_pt))

    r_hat_dict: dict[str, float] = {}
    r_hat_all: list[float] = []

    # Diagnostics on VAR coefficients
    beta_rhats = []
    n_params_beta = n * k
    for param_idx in range(n_params_beta):
        chains_stacked = []
        for c, start, end in split_chains_for_gr:
            c_vals = all_chains_beta[c][start:end].reshape(end - start, -1)[:, param_idx]
            chains_stacked.append(c_vals)
        gr_res = gelman_rubin(np.array(chains_stacked))
        val = float(gr_res.get("R_hat", 1.0))
        if np.isfinite(val):
            beta_rhats.append(val)
            r_hat_all.append(val)

    if beta_rhats:
        r_hat_dict["beta_max"] = float(np.max(beta_rhats))
        r_hat_dict["beta_mean"] = float(np.mean(beta_rhats))

    # Diagnostics on Contemporaneous matrix A
    a_rhats = []
    for r in range(1, n):
        for c_idx in range(r):
            chains_stacked = []
            for ch, start, end in split_chains_for_gr:
                c_vals = all_chains_a[ch][start:end, r, c_idx]
                chains_stacked.append(c_vals)
            gr_res = gelman_rubin(np.array(chains_stacked))
            val = float(gr_res.get("R_hat", 1.0))
            if np.isfinite(val):
                a_rhats.append(val)
                r_hat_all.append(val)

    if a_rhats:
        r_hat_dict["a_max"] = float(np.max(a_rhats))

    # Diagnostics on volatility parameters and paths
    h_rhats = []
    for i, name in enumerate(var_names):
        # Mean log volatility
        chains_stacked = []
        for ch, start, end in split_chains_for_gr:
            c_vals = np.mean(all_chains_h[ch][start:end, :, i], axis=1)
            chains_stacked.append(c_vals)
        gr_res = gelman_rubin(np.array(chains_stacked))
        val = float(gr_res.get("R_hat", 1.0))
        r_hat_dict[f"h_{name}_mean"] = val
        if np.isfinite(val):
            h_rhats.append(val)
            r_hat_all.append(val)

        # AR(1) mu
        chains_mu = []
        for ch, start, end in split_chains_for_gr:
            chains_mu.append(all_chains_mu[ch][start:end, i])
        gr_mu = gelman_rubin(np.array(chains_mu))
        val_mu = float(gr_mu.get("R_hat", 1.0))
        r_hat_dict[f"mu_{name}"] = val_mu
        if np.isfinite(val_mu):
            r_hat_all.append(val_mu)

        # AR(1) phi
        chains_phi = []
        for ch, start, end in split_chains_for_gr:
            chains_phi.append(all_chains_phi[ch][start:end, i])
        gr_phi = gelman_rubin(np.array(chains_phi))
        val_phi = float(gr_phi.get("R_hat", 1.0))
        r_hat_dict[f"phi_{name}"] = val_phi
        if np.isfinite(val_phi):
            r_hat_all.append(val_phi)

        # AR(1) sigma_h
        chains_sig = []
        for ch, start, end in split_chains_for_gr:
            chains_sig.append(all_chains_sigma_h[ch][start:end, i])
        gr_sig = gelman_rubin(np.array(chains_sig))
        val_sig = float(gr_sig.get("R_hat", 1.0))
        r_hat_dict[f"sigma_h_{name}"] = val_sig
        if np.isfinite(val_sig):
            r_hat_all.append(val_sig)

    if h_rhats:
        r_hat_dict["h_max"] = float(np.max(h_rhats))
        r_hat_dict["h_mean"] = float(np.mean(h_rhats))

    # NaN (not a reassuring 1.0) when no split R-hat could be computed, e.g.
    # fewer than two draws per half-chain; summary() reports it as undiagnosed.
    r_hat_dict["max"] = float(np.max(r_hat_all)) if r_hat_all else float("nan")

    # Pool draws across chains
    pooled_beta = np.concatenate(all_chains_beta, axis=0)
    pooled_h = np.concatenate(all_chains_h, axis=0)
    pooled_a = np.concatenate(all_chains_a, axis=0)
    pooled_mu = np.concatenate(all_chains_mu, axis=0)
    pooled_phi = np.concatenate(all_chains_phi, axis=0)
    pooled_sigma_h = np.concatenate(all_chains_sigma_h, axis=0)
    total_draws = pooled_beta.shape[0]

    # -----------------------------------------------------------------
    # In-sample log pointwise predictive density (lppd) across MCMC draws
    # log p(y_t | β, A, h_t) = -n/2 log(2π) - 1/2 sum_i h_{i,t} - 1/2 sum_i exp(-h_{i,t}) ν_{i,t}²
    # evaluated with the smoothed volatility draws h_t (in-sample fit measure).
    # -----------------------------------------------------------------
    log_scores = np.empty(T_eff)
    for t in range(T_eff):
        y_t = Y_dep[t]  # (n,)
        x_t = X[t]      # (k,)
        # For all d:
        # u_t(d) = y_t - B(d).T @ x_t
        u_draws = y_t[None, :] - (pooled_beta.transpose(0, 2, 1) @ x_t)  # (D, n)
        nu_draws = np.einsum("dij,dj->di", pooled_a, u_draws)           # (D, n)
        h_t_draws = pooled_h[:, t, :]                                    # (D, n)

        log_p_d = (
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * np.sum(h_t_draws, axis=1)
            - 0.5 * np.sum(np.exp(-h_t_draws) * (nu_draws ** 2), axis=1)
        )
        log_scores[t] = float(logsumexp(log_p_d) - np.log(total_draws))

    return BVAR_SVResult(
        beta_draws=pooled_beta,
        h_draws=pooled_h,
        a_draws=pooled_a,
        r_hat=r_hat_dict,
        data=df_data,
        lags=lags,
        mu_draws=pooled_mu,
        phi_draws=pooled_phi,
        sigma_h_draws=pooled_sigma_h,
        log_scores=log_scores,
        n_draws=draws_per_chain,
        n_burn=n_burn,
        n_chains=n_chains,
        n_unstable_rejections=n_unstable_rejections,
        n_stuck_iterations=n_stuck_iterations,
    )


__all__ = [
    "bvar_sv",
    "BVAR_SVResult",
    "BVAR_SVForecast",
    "BVAR_SV_IRF",
]
