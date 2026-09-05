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
          h_{i,t} - μ_i = φ_i (h_{i,t-1} - μ_i) + σ_{h,i} η_{i,t},  |φ_i| < 1.

The MCMC algorithm uses:
    1. Minnesota prior shrinkage on VAR coefficients β with precision-weighted GLS.
    2. Equation-by-equation sampling of contemporaneous relations A.
    3. Kim, Shephard & Chib (1998) 7-component Gaussian mixture approximation
       with Carter-Kohn (1994) forward-filtering backward-sampling (FFBS) for
       the log-volatility paths h_{i,t}.
    4. Conjugate sampling of AR(1) volatility parameters (μ_i, φ_i, σ_{h,i}^2).
    5. Multi-chain / split-chain Gelman-Rubin convergence diagnostics (R̂ < 1.1).
    6. Volatility-conditioned impulse responses and predictive density log-scores.

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

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from .._linalg import inv_xtx, safe_cholesky
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
        self.median = getattr(obj, "median", None)
        self.lower = getattr(obj, "lower", None)
        self.upper = getattr(obj, "upper", None)
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
        """Return tidy DataFrame with horizon, median response, and bands."""
        H_plus, n_resp, n_shock = self.median.shape
        rows = []
        for h in range(H_plus):
            if target_idx is not None and shock_idx is not None:
                resp_name = names[target_idx] if names else str(target_idx)
                shock_name = names[shock_idx] if names else str(shock_idx)
                rows.append({
                    "horizon": h,
                    "target": resp_name,
                    "shock": shock_name,
                    "median": float(self.median[h, target_idx, shock_idx]),
                    "lower": float(self.lower[h, target_idx, shock_idx]),
                    "upper": float(self.upper[h, target_idx, shock_idx]),
                })
            else:
                for r in range(n_resp):
                    for s in range(n_shock):
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
class BVAR_SVResult:
    """Posterior estimation results for BVAR with Stochastic Volatility.

    Attributes
    ----------
    beta_draws : np.ndarray
        MCMC draws of VAR coefficient matrix B of shape (D, 1 + n*p, n).
    h_draws : np.ndarray
        MCMC draws of log-volatilities of shape (D, T_eff, n).
    a_draws : np.ndarray
        MCMC draws of contemporaneous impact matrix A of shape (D, n, n).
    r_hat : dict[str, float]
        Gelman-Rubin split-Rhat convergence diagnostics.
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
        Pointwise predictive density log-scores (T_eff,).
    n_draws : int
        Number of retained draws.
    n_burn : int
        Number of burned initial draws.
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
    def names(self) -> list[str]:
        """Names of endogenous variables."""
        return list(self.data.columns.astype(str))

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
        """Return total predictive log score or pointwise array."""
        if point_by_point:
            return self.log_scores.copy()
        return float(np.sum(self.log_scores))

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

    def summary(self) -> str:
        """Render a formatted text summary of model estimation and diagnostics."""
        max_rhat = self.r_hat.get("max", 1.0)
        conv_flag = "CONVERGED (R̂ < 1.1)" if max_rhat < 1.1 else "WARNING (R̂ >= 1.1)"
        total_ls = float(np.sum(self.log_scores))
        mean_ls = float(np.mean(self.log_scores))

        lines = [
            "==================================================================",
            "  Bayesian VAR with Stochastic Volatility (BVAR-SV)",
            "==================================================================",
            f"  Variables (n)       : {self.n} ({', '.join(self.names)})",
            f"  Sample size (T)     : {len(self.data)} (Effective T = {self.T_eff})",
            f"  Lag order (p)       : {self.lags}",
            f"  Retained draws (D)  : {self.n_draws} (burn-in = {self.n_burn})",
            f"  Convergence status  : {conv_flag} [Max R̂ = {max_rhat:.4f}]",
            f"  Predictive Log-Score: Total = {total_ls:.2f}, Mean = {mean_ls:.4f}",
            "------------------------------------------------------------------",
            "  Log-Volatility AR(1) Parameters (Posterior Mean ± SD):",
        ]
        for i, name in enumerate(self.names):
            mu_m = float(np.mean(self.mu_draws[:, i]))
            mu_s = float(np.std(self.mu_draws[:, i], ddof=1))
            phi_m = float(np.mean(self.phi_draws[:, i]))
            phi_s = float(np.std(self.phi_draws[:, i], ddof=1))
            sig_m = float(np.mean(self.sigma_h_draws[:, i]))
            sig_s = float(np.std(self.sigma_h_draws[:, i], ddof=1))
            lines.append(
                f"    {name:<12}: μ = {mu_m:+.3f} ± {mu_s:.3f}, "
                f"φ = {phi_m:.3f} ± {phi_s:.3f}, "
                f"σ_h = {sig_m:.3f} ± {sig_s:.3f}"
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
        target_idx: int = 0,
        figsize: tuple[float, float] | None = None,
        ax: Any = None,
    ):
        """Create a 3-panel plot of BVAR-SV results.

        Panel 1: Log-volatilities h_{i,t} with credible bands.
        Panel 2: Conditional standard deviations exp(h_{i,t} / 2) with credible bands.
        Panel 3: Volatility-conditioned impulse response with credible bands.
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

        t_axis = np.arange(self.T_eff)
        # 1. Log-volatilities
        axes[0].plot(t_axis, h_med[:, target_idx], label=f"{self.names[target_idx]} (median)", color="navy")
        axes[0].fill_between(t_axis, h_low[:, target_idx], h_high[:, target_idx], color="navy", alpha=0.25, label=f"{int(ci*100)}% CI")
        axes[0].set_title(f"Log-Volatility $h_{{t}}$ ({self.names[target_idx]})")
        axes[0].set_xlabel("Time (effective)")
        axes[0].set_ylabel("Log-Variance")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="best")

        # 2. Conditional SD
        axes[1].plot(t_axis, sd_med[:, target_idx], label=f"{self.names[target_idx]} (median)", color="crimson")
        axes[1].fill_between(t_axis, sd_low[:, target_idx], sd_high[:, target_idx], color="crimson", alpha=0.25, label=f"{int(ci*100)}% CI")
        axes[1].set_title(f"Cond. Std. Dev. $\\exp(h_{{t}}/2)$ ({self.names[target_idx]})")
        axes[1].set_xlabel("Time (effective)")
        axes[1].set_ylabel("Std Dev")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="best")

        # 3. Volatility-conditioned IRF
        h_axis = np.arange(horizon + 1)
        axes[2].plot(h_axis, irf_res.median[:, target_idx, shock_idx], label="Median IRF", color="darkgreen")
        axes[2].fill_between(
            h_axis,
            irf_res.lower[:, target_idx, shock_idx],
            irf_res.upper[:, target_idx, shock_idx],
            color="darkgreen",
            alpha=0.25,
            label=f"{int(ci*100)}% CI",
        )
        axes[2].axhline(0, color="black", linestyle=":", linewidth=1)
        axes[2].set_title(f"IRF: {self.names[shock_idx]} $\\to$ {self.names[target_idx]} (date {t_idx})")
        axes[2].set_xlabel("Horizon")
        axes[2].set_ylabel("Response")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="best")

        if fig is not None:
            fig.tight_layout()
            return fig
        return axes


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
        Endogenous variables panel.
    lags : int, default 4
        VAR lag order.
    n_draws : int, default 2000
        Number of post-burn-in MCMC draws to retain per chain.
    n_burn : int, default 1000
        Number of burn-in iterations.
    minnesota_prior : bool, default True
        Whether to impose Minnesota shrinkage on VAR lag coefficients.
    seed : int, optional
        RNG seed for exact reproducibility.
    lambda1 : float, default 0.2
        Overall Minnesota prior shrinkage.
    lambda2 : float, default 0.5
        Cross-variable Minnesota prior shrinkage.
    lambda3 : float, default 1.0
        Lag-decay exponent for Minnesota prior.
    intercept_prior_std : float, default 1e3
        Prior standard deviation for the intercept (diffuse).
    thin : int, default 1
        Thinning interval for MCMC chain.
    n_chains : int, default 1
        Number of independent MCMC chains to run.
    p : int, optional
        Positional / keyword alias for ``lags``.

    Returns
    -------
    BVAR_SVResult
        Rich dataclass containing draws, Gelman-Rubin convergence diagnostics,
        IRFs, predictive log scores, and presentation methods.
    """
    if p is not None:
        lags = p
    if lags <= 0:
        raise ValueError(f"lags must be a positive integer, got {lags}")
    if n_draws <= 0 or n_burn < 0:
        raise ValueError("n_draws must be > 0 and n_burn must be >= 0")

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

    T, n = Y_arr.shape
    if T <= lags + 2:
        raise ValueError(f"Insufficient observations: T={T} must exceed lags+2={lags+2}")

    T_eff = T - lags
    k = 1 + n * lags

    # Build design matrix X and dependent variable Y_dep
    Y_dep = Y_arr[lags:]  # (T_eff, n)
    X_rows = []
    for t in range(lags, T):
        row = [1.0]
        for lag_idx in range(1, lags + 1):
            row.extend(Y_arr[t - lag_idx])
        X_rows.append(row)
    X = np.asarray(X_rows, dtype=float)  # (T_eff, k)

    # Minnesota prior setup
    if minnesota_prior:
        sigmas = np.array([_univariate_sigma(Y_arr[:, i], lags) for i in range(n)])
        beta0 = np.zeros(n * k)
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
        beta0 = np.zeros(n * k)
        v0 = np.ones(n * k) * 1e4

    V0_inv = np.diag(1.0 / v0)
    V0_inv_beta0 = V0_inv @ beta0

    # Outer product of regressors for vectorized GLS
    X_outer = X[:, :, None] * X[:, None, :]  # (T_eff, k, k)

    base_rng = np.random.default_rng(seed)

    draws_per_chain = max(1, n_draws // n_chains if n_chains > 1 else n_draws)

    all_chains_beta = []
    all_chains_h = []
    all_chains_a = []
    all_chains_mu = []
    all_chains_phi = []
    all_chains_sigma_h = []

    for chain_id in range(n_chains):
        # Chain-specific RNG
        chain_seed = base_rng.integers(0, 2**31 - 1)
        rng = np.random.default_rng(chain_seed)

        # Initial parameter states
        # OLS starting point for B
        try:
            B_init, *_ = np.linalg.lstsq(X, Y_dep, rcond=None)
        except Exception:
            B_init = np.zeros((k, n))

        B_cur = B_init.copy()
        A_cur = np.eye(n)
        U_init = Y_dep - X @ B_cur
        h_cur = np.zeros((T_eff, n))
        for i in range(n):
            resid_var = float(np.var(U_init[:, i], ddof=1)) if len(U_init) > 1 else 1.0
            h_cur[:, i] = np.log(max(resid_var, 1e-4))

        mu_cur = np.mean(h_cur, axis=0)
        phi_cur = np.full(n, 0.85)
        sigma_h2_cur = np.full(n, 0.05)

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

            # Sample β with companion stability rejection (max 50 retries)
            stable_draw = False
            for _ in range(50):
                z_b = rng.standard_normal(n * k)
                beta_cand = beta_mean + np.linalg.solve(L_prec.T, z_b)
                B_cand = beta_cand.reshape(n, k).T
                if _is_stable_companion(B_cand, n, lags):
                    B_cur = B_cand
                    stable_draw = True
                    break
            if not stable_draw:
                # If retries exceeded, retain current B or shrink slightly
                pass

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

                # Sample μ_i ~ N(m_μ, V_μ)
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

                # Sample φ_i with stationarity truncation
                h_dev = h_cur[:, i] - mu_cur[i]
                x_phi = h_dev[:-1]
                y_phi = h_dev[1:]
                v_phi_inv = (1.0 / 0.1) + float(np.sum(x_phi ** 2)) / sig2_i
                v_phi = 1.0 / max(v_phi_inv, 1e-12)
                m_phi = v_phi * ((0.85 / 0.1) + float(np.sum(x_phi * y_phi)) / sig2_i)
                prop_phi = m_phi + np.sqrt(v_phi) * rng.standard_normal()

                if abs(prop_phi) < 0.999:
                    log_acc = (
                        0.5 * np.log(max(1.0 - prop_phi ** 2, 1e-8))
                        - 0.5 * np.log(max(1.0 - phi_i ** 2, 1e-8))
                        - 0.5 * (prop_phi ** 2 - phi_i ** 2) * (h_dev[0] ** 2) / sig2_i
                    )
                    if np.log(max(rng.uniform(), 1e-12)) < log_acc:
                        phi_cur[i] = prop_phi

                # Sample σ_{h,i}² ~ IG(a_post, b_post)
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

        all_chains_beta.append(np.array(chain_beta[:draws_per_chain]))
        all_chains_h.append(np.array(chain_h[:draws_per_chain]))
        all_chains_a.append(np.array(chain_a[:draws_per_chain]))
        all_chains_mu.append(np.array(chain_mu[:draws_per_chain]))
        all_chains_phi.append(np.array(chain_phi[:draws_per_chain]))
        all_chains_sigma_h.append(np.array(chain_sigma_h[:draws_per_chain]))

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

    r_hat_dict["max"] = float(np.max(r_hat_all)) if r_hat_all else 1.0

    # Pool draws across chains
    pooled_beta = np.concatenate(all_chains_beta, axis=0)
    pooled_h = np.concatenate(all_chains_h, axis=0)
    pooled_a = np.concatenate(all_chains_a, axis=0)
    pooled_mu = np.concatenate(all_chains_mu, axis=0)
    pooled_phi = np.concatenate(all_chains_phi, axis=0)
    pooled_sigma_h = np.concatenate(all_chains_sigma_h, axis=0)
    total_draws = pooled_beta.shape[0]

    # -----------------------------------------------------------------
    # Compute predictive density log-scores across MCMC draws
    # log p(y_t | β, A, h_t) = -n/2 log(2π) - 1/2 sum_i h_{i,t} - 1/2 sum_i exp(-h_{i,t}) ν_{i,t}²
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
        n_draws=total_draws,
        n_burn=n_burn,
    )


__all__ = [
    "bvar_sv",
    "BVAR_SVResult",
    "BVAR_SV_IRF",
]
