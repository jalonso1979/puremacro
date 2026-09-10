"""Del Negro & Schorfheide (2004) DSGE-VAR Hybrid Modeling.

Implements the DSGE-VAR(lambda) estimator linking theoretical DSGE
cross-equation autocovariances Gamma_k(theta) to an informative
Normal-Inverted-Wishart conjugate prior on an unrestricted VAR(p).

Reference:
----------
Del Negro, M., & Schorfheide, F. (2004). Priors from General Equilibrium Models
for VARs. International Economic Review, 45(2), 643-673.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.special
import scipy.stats

from puremacro.dsge._moments import first_order_moments
from puremacro.var.irf import fevd as _var_fevd
from puremacro.var.irf import irf as _var_irf

__all__ = ["DSGEVARResult", "estimate_dsge_var"]


def _log_mvgamma(a: float, d: int) -> float:
    """Log multivariate gamma function ln Gamma_d(a).

    ln Gamma_d(a) = d*(d-1)/4 * ln(pi) + sum_{j=1}^d gammaln(a + (1-j)/2)
    """
    j = np.arange(1, d + 1, dtype=float)
    return float(
        0.25 * d * (d - 1) * np.log(np.pi)
        + np.sum(scipy.special.gammaln(a + 0.5 * (1.0 - j)))
    )


@dataclass
class DSGEVARResult:
    """Estimation result container for Del Negro & Schorfheide (2004) DSGE-VAR.

    Attributes
    ----------
    lamb : float
        Prior weight parameter lambda used in posterior estimation.
    hat_lambda : float or None
        Optimal hyperparameter hat{lambda} maximizing marginal data density,
        or None if fixed lambda was evaluated without optimization.
    lambda_min : float
        Admissibility bound (k + n) / T for a proper Inverted-Wishart prior.
    log_mdd : float
        Log marginal data density ln p(Y | lambda, theta).
    log_mdd_grid : pd.DataFrame or None
        Grid evaluation of log marginal data density across candidate lambda values.
    A_list : list of np.ndarray
        List of length p holding (n, n) VAR autoregressive coefficient matrices.
    intercept : np.ndarray
        (n,) intercept vector (zeros if intercept=False).
    Sigma : np.ndarray
        (n, n) posterior residual covariance matrix tilde{Sigma}(lambda).
    B0 : np.ndarray
        (n, n) structural impact matrix tilde{A}_0 under selected identification.
    Phi_star : np.ndarray
        (k, n) theoretical DSGE prior mean VAR coefficients.
    Sigma_star : np.ndarray
        (n, n) theoretical DSGE prior residual covariance matrix.
    Phi_ols : np.ndarray
        (k, n) sample OLS VAR coefficients.
    Sigma_ols : np.ndarray
        (n, n) sample OLS residual covariance matrix.
    resid : np.ndarray
        (T, n) posterior in-sample residuals Y - X * tilde{Phi}.
    data : pd.DataFrame
        In-sample data used for estimation.
    names : tuple of str
        Names of observable variables.
    shock_names : tuple of str
        Names of structural shocks.
    p : int
        VAR lag order.
    T : int
        Effective sample size (T_raw - p).
    model : Any
        Underlying solved DSGE LinearModel.
    identification : str
        Identification scheme used ('dsge' or 'cholesky').
    """

    lamb: float
    hat_lambda: float | None
    lambda_min: float
    log_mdd: float
    log_mdd_grid: pd.DataFrame | None
    A_list: list[np.ndarray]
    intercept: np.ndarray
    Sigma: np.ndarray
    B0: np.ndarray
    Phi_star: np.ndarray
    Sigma_star: np.ndarray
    Phi_ols: np.ndarray
    Sigma_ols: np.ndarray
    resid: np.ndarray
    data: pd.DataFrame
    names: tuple[str, ...]
    shock_names: tuple[str, ...]
    p: int
    T: int
    model: Any
    identification: str

    def irf(
        self,
        horizon: int = 20,
        shock: str | int | None = None,
    ) -> pd.DataFrame:
        """Compute structural impulse response functions up to horizon.

        Parameters
        ----------
        horizon : int, default 20
            Number of periods after impact (0..horizon).
        shock : str, int, or None, default None
            Specific shock by name or 0-based integer index. If None,
            returns responses to all shocks in a MultiIndex DataFrame.

        Returns
        -------
        pd.DataFrame
            Impulse response paths indexed by horizon h = 0..horizon.
        """
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        irf_3d = _var_irf(self.A_list, self.B0, horizon)  # (H+1, n, n)
        h_idx = pd.RangeIndex(horizon + 1, name="h")
        n = len(self.names)

        if shock is not None:
            if isinstance(shock, str):
                if shock not in self.shock_names:
                    raise ValueError(
                        f"Unknown shock {shock!r}; declared: {list(self.shock_names)}"
                    )
                shock_idx = self.shock_names.index(shock)
            else:
                shock_idx = int(shock)
                if not (0 <= shock_idx < len(self.shock_names)):
                    raise IndexError(
                        f"Shock index {shock_idx} out of range [0, {len(self.shock_names)})"
                    )
            return pd.DataFrame(
                irf_3d[:, :, shock_idx], index=h_idx, columns=list(self.names)
            )

        cols = pd.MultiIndex.from_product(
            [self.shock_names, self.names], names=["shock", "variable"]
        )
        data_2d = np.zeros((horizon + 1, len(self.shock_names) * n))
        col_pos = 0
        for s_idx in range(len(self.shock_names)):
            data_2d[:, col_pos : col_pos + n] = irf_3d[:, :, s_idx]
            col_pos += n

        return pd.DataFrame(data_2d, index=h_idx, columns=cols)

    def fevd(self, horizon: int = 20) -> np.ndarray:
        """Forecast error variance decomposition.

        Parameters
        ----------
        horizon : int, default 20
            Decomposition horizon.

        Returns
        -------
        np.ndarray
            Array of shape (horizon + 1, n, n) where [h, i, j] is the share
            of forecast error variance of variable i explained by shock j.
        """
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")
        return _var_fevd(self.A_list, self.B0, horizon)

    def forecast(
        self,
        horizon: int = 8,
        ci: float = 0.90,
        seed: int | None = None,
    ) -> DSGEForecastResult:
        """Compute out-of-sample forecasts and analytic confidence bands.

        Parameters
        ----------
        horizon : int, default 8
            Forecast horizon periods ahead.
        ci : float, default 0.90
            Confidence interval coverage in (0, 1).
        seed : int or None, default None
            Random seed (reserved for simulation compatibility).

        Returns
        -------
        DSGEForecastResult
            Forecast container with .mean, .lower, .upper DataFrames.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if not (0.0 < ci < 1.0):
            raise ValueError(f"ci must be in (0, 1), got {ci}")

        n = len(self.names)
        p = self.p
        y_hist = self.data[list(self.names)].to_numpy()
        T_tot = len(y_hist)

        # MA lag coefficients Phi_l for forecast covariance
        Phi_list = [np.eye(n)]
        for h in range(1, horizon):
            Ph = np.zeros((n, n))
            for j in range(1, min(h, p) + 1):
                Ph += Phi_list[h - j] @ self.A_list[j - 1]
            Phi_list.append(Ph)

        # Point forecast recursion
        y_fc = np.zeros((horizon, n))
        history_buffer = list(y_hist[max(0, T_tot - p) :])
        while len(history_buffer) < p:
            history_buffer.insert(0, np.zeros(n))

        for h in range(horizon):
            pred = np.array(self.intercept, copy=True)
            for j in range(1, p + 1):
                pred += self.A_list[j - 1] @ history_buffer[-j]
            y_fc[h] = pred
            history_buffer.append(pred)

        # Variance recursion
        z_crit = float(scipy.stats.norm.ppf(0.5 * (1.0 + ci)))
        cov_cum = np.zeros((n, n))
        se_arr = np.zeros((horizon, n))
        for h in range(horizon):
            cov_cum += Phi_list[h] @ self.Sigma @ Phi_list[h].T
            diag_var = np.maximum(np.diagonal(cov_cum), 0.0)
            se_arr[h] = np.sqrt(diag_var)

        lower_arr = y_fc - z_crit * se_arr
        upper_arr = y_fc + z_crit * se_arr

        idx = pd.RangeIndex(1, horizon + 1, name="h")
        mean_df = pd.DataFrame(y_fc, index=idx, columns=list(self.names))
        lower_df = pd.DataFrame(lower_arr, index=idx, columns=list(self.names))
        upper_df = pd.DataFrame(upper_arr, index=idx, columns=list(self.names))

        from puremacro.dsge._results import DSGEForecastResult

        return DSGEForecastResult(
            mean=mean_df,
            lower=lower_df,
            upper=upper_df,
            ci=ci,
            horizon=horizon,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return estimated VAR coefficients as a DataFrame."""
        n = len(self.names)
        p = self.p
        has_intercept = not np.allclose(self.intercept, 0.0) or len(self.A_list) * n != len(self.Phi_star)

        row_names: list[str] = []
        if has_intercept:
            row_names.append("const")
        for l in range(1, p + 1):
            for v in self.names:
                row_names.append(f"L{l}.{v}")

        k = len(row_names)
        phi_mat = np.zeros((k, n))
        start_row = 0
        if has_intercept:
            phi_mat[0, :] = self.intercept
            start_row = 1

        for l in range(p):
            phi_mat[start_row + l * n : start_row + (l + 1) * n, :] = self.A_list[l].T

        return pd.DataFrame(phi_mat, index=row_names, columns=list(self.names))

    def summary(self) -> str:
        """Publication-grade summary table of DSGE-VAR estimation."""
        n = len(self.names)
        hat_str = f"{self.hat_lambda:.4f}" if self.hat_lambda is not None else "None (fixed)"
        lines = [
            "=" * 78,
            "DSGE-VAR Estimation (Del Negro & Schorfheide 2004)",
            "=" * 78,
            f"Sample size (T):           {self.T} (Effective after {self.p} lags)",
            f"Observables (n):           {n} ({', '.join(self.names)})",
            f"Structural shocks:         {len(self.shock_names)} ({', '.join(self.shock_names)})",
            f"VAR lag order (p):         {self.p}",
            f"Identification:            {self.identification}",
            "-" * 78,
            f"Prior weight (lambda):     {self.lamb:.4f}",
            f"Admissibility bound (min): {self.lambda_min:.4f} = (k + n) / T",
            f"Optimal lambda (hat):      {hat_str}",
            f"Log Marginal Data Density: {self.log_mdd:.4f}",
            "=" * 78,
            "Estimated VAR Coefficients (tilde{Phi}):",
            self.to_frame().round(4).to_string(),
            "-" * 78,
            "Posterior Innovation Covariance (tilde{Sigma}):",
            pd.DataFrame(self.Sigma, index=list(self.names), columns=list(self.names))
            .round(6)
            .to_string(),
            "-" * 78,
            "Structural Impact Matrix (B0 = tilde{A}_0):",
            pd.DataFrame(self.B0, index=list(self.names), columns=list(self.shock_names))
            .round(6)
            .to_string(),
            "=" * 78,
        ]
        return "\n".join(lines)

    def plot(
        self,
        kind: str = "irf",
        target: str | int = 0,
        shock: str | int = 0,
        horizon: int = 20,
        ax: Any | None = None,
        **kwargs,
    ) -> tuple[Any, Any]:
        """Plot DSGE-VAR results (IRF, MDD surface, or Forecasts).

        Parameters
        ----------
        kind : str, default 'irf'
            Type of plot: 'irf', 'mdd', or 'forecast'.
        target : str or int, default 0
            Target response variable for 'irf' plots.
        shock : str or int, default 0
            Impulse shock for 'irf' plots.
        horizon : int, default 20
            Horizon for IRF or forecast plots.
        ax : matplotlib.axes.Axes or None
            Target axes. If None, a new figure and axes are created.
        **kwargs
            Passed to matplotlib plotting methods.

        Returns
        -------
        tuple of (Figure, Axes)
        """
        if ax is None:
            fig, target_ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 4.8)))
        else:
            fig = ax.figure
            target_ax = ax

        if kind.lower() == "irf":
            shock_name = self.shock_names[shock] if isinstance(shock, int) else shock
            target_name = self.names[target] if isinstance(target, int) else target
            irf_df = self.irf(horizon=horizon, shock=shock_name)
            target_ax.plot(
                irf_df.index,
                irf_df[target_name],
                color=kwargs.pop("color", "#1f77b4"),
                linewidth=kwargs.pop("linewidth", 1.8),
                label=f"DSGE-VAR (λ={self.lamb:.2f})",
                **kwargs,
            )
            target_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            target_ax.set_title(
                f"DSGE-VAR IRF: {target_name} to {shock_name}",
                fontsize=11,
                fontweight="bold",
            )
            target_ax.set_xlabel("Horizon")
            target_ax.set_ylabel("Response")
            target_ax.grid(True, linestyle=":", alpha=0.5)
            target_ax.legend(frameon=False)

        elif kind.lower() == "mdd":
            if self.log_mdd_grid is None:
                raise ValueError(
                    "log_mdd_grid is None; evaluate with lambda_grid or optimize to plot MDD."
                )
            grid_df = self.log_mdd_grid
            target_ax.plot(
                grid_df["lambda"],
                grid_df["log_mdd"],
                marker="o",
                color=kwargs.pop("color", "#2ca02c"),
                linewidth=kwargs.pop("linewidth", 1.8),
                label=r"$\ln p(Y|\lambda, \theta)$",
                **kwargs,
            )
            if self.hat_lambda is not None:
                target_ax.axvline(
                    self.hat_lambda,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Optimal $\\hat{{\\lambda}} = {self.hat_lambda:.2f}$",
                )
            target_ax.set_title(
                r"DSGE-VAR Marginal Data Density $\ln p(Y|\lambda, \theta)$",
                fontsize=11,
                fontweight="bold",
            )
            target_ax.set_xlabel(r"Prior Weight $\lambda$")
            target_ax.set_ylabel("Log MDD")
            target_ax.grid(True, linestyle=":", alpha=0.5)
            target_ax.legend(frameon=False)

        elif kind.lower() == "forecast":
            fc = self.forecast(horizon=horizon)
            target_var = self.names[target] if isinstance(target, int) else target
            target_ax.plot(
                fc.mean.index,
                fc.mean[target_var],
                label=f"Forecast {target_var}",
                color=kwargs.pop("color", "#1f77b4"),
                linewidth=kwargs.pop("linewidth", 1.8),
                **kwargs,
            )
            target_ax.fill_between(
                fc.mean.index,
                fc.lower[target_var],
                fc.upper[target_var],
                alpha=0.25,
                color="#1f77b4",
                label=f"{int(round(100 * fc.ci))}% CI",
            )
            target_ax.set_title(
                f"DSGE-VAR Forecast: {target_var} ({int(round(100 * fc.ci))}% band)",
                fontsize=11,
                fontweight="bold",
            )
            target_ax.set_xlabel("Horizon")
            target_ax.set_ylabel(target_var)
            target_ax.grid(True, linestyle=":", alpha=0.5)
            target_ax.legend(frameon=False)
        else:
            raise ValueError(f"Unknown kind {kind!r}; choose 'irf', 'mdd', or 'forecast'.")

        return fig, target_ax

    def to_markdown(self, **kwargs) -> str:
        """Format coefficients as Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Format coefficients as LaTeX table."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Format coefficients as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


def estimate_dsge_var(
    model: Any,
    data: pd.DataFrame | np.ndarray,
    p: int = 4,
    lamb: float | str | None = None,
    *,
    varobs: Sequence[str] | None = None,
    shocks: Sequence[str] | None = None,
    intercept: bool = True,
    identification: str = "dsge",
    lambda_grid: Sequence[float] | None = None,
    check_bounds: bool = True,
) -> DSGEVARResult:
    """Estimate Del Negro & Schorfheide (2004) DSGE-VAR(lambda).

    Parameters
    ----------
    model : Any
        Solved LinearModel instance with .solution, .states, .controls, .shocks.
    data : pd.DataFrame or np.ndarray
        Observable time series data matrix.
    p : int, default 4
        VAR lag order.
    lamb : float, 'optimal', or None, default None
        Prior weight parameter lambda. If None or 'optimal', maximizes the
        marginal data density over lambda in [0.2, 5.0].
    varobs : Sequence[str] or None, default None
        Names of observable variables. If None, inferred from DataFrame columns
        or model variables.
    shocks : Sequence[str] or None, default None
        Names of structural shocks matching varobs for identification.
    intercept : bool, default True
        Whether to include a constant term in the VAR regressor vector.
    identification : str, default 'dsge'
        Structural identification scheme: 'dsge' (DSGE rotation Q*) or 'cholesky'.
    lambda_grid : Sequence[float] or None, default None
        Grid of lambda candidate values to evaluate log marginal data density.
    check_bounds : bool, default True
        If True, enforces proper prior bound lambda >= lambda_min = (k + n) / T.

    Returns
    -------
    DSGEVARResult
        Estimated hybrid model result object.
    """
    if p < 1:
        raise ValueError(f"Lag order p must be positive, got {p}")
    if identification not in ("dsge", "cholesky"):
        raise ValueError(
            f"Unknown identification {identification!r}; expected 'dsge' or 'cholesky'"
        )

    # 1. Parse observable variables and data
    if isinstance(data, pd.DataFrame):
        if varobs is None:
            # Match columns that exist in model.variables
            matched = [c for c in data.columns if c in model.variables]
            if not matched:
                obs_names = tuple(str(c) for c in data.columns)
            else:
                obs_names = tuple(matched)
        else:
            obs_names = tuple(varobs)
            for v in obs_names:
                if v not in model.variables:
                    raise ValueError(
                        f"Observable variable {v!r} is not declared in model.variables: {list(model.variables)}"
                    )
                if v not in data.columns:
                    raise ValueError(f"Observable {v!r} not found in data columns")
        df_obs = data[list(obs_names)].copy()
    elif isinstance(data, np.ndarray):
        if varobs is None:
            obs_names = tuple(model.variables[: data.shape[1]])
        else:
            obs_names = tuple(varobs)
        if len(obs_names) != data.shape[1]:
            raise ValueError(
                f"Number of varobs ({len(obs_names)}) does not match data columns ({data.shape[1]})"
            )
        df_obs = pd.DataFrame(data, columns=list(obs_names))
    else:
        raise TypeError(f"data must be pd.DataFrame or np.ndarray, got {type(data).__name__}")

    for v in obs_names:
        if v not in model.variables:
            raise ValueError(
                f"Observable variable {v!r} is not declared in model.variables: {list(model.variables)}"
            )

    n = len(obs_names)
    T_raw = len(df_obs)
    T = T_raw - p
    if T <= 0:
        raise ValueError(
            f"Sample size T_raw={T_raw} is too short for lag order p={p}; effective T={T} <= 0"
        )

    # 2. Parse structural shocks
    if shocks is None:
        shock_names = tuple(model.shocks[:n])
    else:
        shock_names = tuple(shocks)
        for s in shock_names:
            if s not in model.shocks:
                raise ValueError(
                    f"Shock {s!r} not declared in model.shocks: {list(model.shocks)}"
                )

    if identification == "dsge" and len(shock_names) != n:
        raise ValueError(
            f"Structural DSGE identification requires equal number of shocks ({len(shock_names)}) "
            f"and observables ({n})."
        )

    # 3. Assemble OLS sample data moments
    Y_arr = df_obs.to_numpy(dtype=float)
    Y = Y_arr[p:]  # shape (T, n)

    if intercept:
        X = np.column_stack(
            [np.ones(T)] + [Y_arr[p - l - 1 : T_raw - l - 1] for l in range(p)]
        )
        k = 1 + n * p
    else:
        X = np.column_stack([Y_arr[p - l - 1 : T_raw - l - 1] for l in range(p)])
        k = n * p

    XtX = X.T @ X
    XtY = X.T @ Y
    YtY = Y.T @ Y

    Phi_ols = np.linalg.solve(XtX, XtY)
    resid_ols = Y - X @ Phi_ols
    S_ols = resid_ols.T @ resid_ols
    Sigma_ols = S_ols / T
    Sigma_ols = 0.5 * (Sigma_ols + Sigma_ols.T)

    # 4. Check model stationarity and compute theoretical DSGE autocovariances
    g_eigs = np.abs(scipy.linalg.eigvals(model.solution.G))
    if np.any(g_eigs >= 1.0 - 1e-7):
        bad = g_eigs[g_eigs >= 1.0 - 1e-7]
        raise ValueError(
            f"State transition matrix G has non-stationary eigenvalues (|λ| >= 1.0: {bad}); "
            "unconditional stationary moments do not exist."
        )

    G = model.solution.G
    N = model.solution.N
    M_x, M_u = model._reported_loadings()
    sig_u = model._shock_covariance(None)
    order_vars = list(model.states) + list(model.controls)
    sigma_x, gamma_0_full, gammas_full = first_order_moments(G, N, M_x, M_u, sig_u, lags=p)

    cov_0 = (
        pd.DataFrame(gamma_0_full, index=order_vars, columns=order_vars)
        .loc[list(obs_names), list(obs_names)]
        .to_numpy(dtype=float)
    )
    gammas = [
        pd.DataFrame(gammas_full[l], index=order_vars, columns=order_vars)
        .loc[list(obs_names), list(obs_names)]
        .to_numpy(dtype=float)
        for l in range(p)
    ]

    mu_y = model.steady_state.loc[list(obs_names)].to_numpy(dtype=float).reshape(n, 1)

    # Assemble theoretical moments Gamma_XX, Gamma_XY, Gamma_YY
    if intercept:
        Gamma_YY = cov_0 + mu_y @ mu_y.T

        Gamma_XX = np.zeros((k, k), dtype=float)
        Gamma_XX[0, 0] = 1.0
        for j in range(1, p + 1):
            col_slice = slice(1 + (j - 1) * n, 1 + j * n)
            Gamma_XX[0, col_slice] = mu_y.flatten()
            Gamma_XX[col_slice, 0] = mu_y.flatten()

        for i in range(1, p + 1):
            row_slice = slice(1 + (i - 1) * n, 1 + i * n)
            for j in range(1, p + 1):
                col_slice = slice(1 + (j - 1) * n, 1 + j * n)
                if i == j:
                    block = cov_0 + mu_y @ mu_y.T
                elif i < j:
                    block = gammas[j - i - 1] + mu_y @ mu_y.T
                else:
                    block = gammas[i - j - 1].T + mu_y @ mu_y.T
                Gamma_XX[row_slice, col_slice] = block

        Gamma_XY = np.zeros((k, n), dtype=float)
        Gamma_XY[0, :] = mu_y.flatten()
        for i in range(1, p + 1):
            row_slice = slice(1 + (i - 1) * n, 1 + i * n)
            Gamma_XY[row_slice, :] = gammas[i - 1].T + mu_y @ mu_y.T
    else:
        Gamma_YY = cov_0

        Gamma_XX = np.zeros((k, k), dtype=float)
        for i in range(1, p + 1):
            row_slice = slice((i - 1) * n, i * n)
            for j in range(1, p + 1):
                col_slice = slice((j - 1) * n, j * n)
                if i == j:
                    block = cov_0
                elif i < j:
                    block = gammas[j - i - 1]
                else:
                    block = gammas[i - j - 1].T
                Gamma_XX[row_slice, col_slice] = block

        Gamma_XY = np.zeros((k, n), dtype=float)
        for i in range(1, p + 1):
            row_slice = slice((i - 1) * n, i * n)
            Gamma_XY[row_slice, :] = gammas[i - 1].T

    Gamma_XX = 0.5 * (Gamma_XX + Gamma_XX.T)
    Gamma_YY = 0.5 * (Gamma_YY + Gamma_YY.T)
    Gamma_YX = Gamma_XY.T

    # Theoretical VAR prior moments
    Phi_star = np.linalg.solve(Gamma_XX, Gamma_XY)
    Sigma_star = Gamma_YY - Gamma_YX @ Phi_star
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)

    # 5. Admissibility bound lambda_min
    lambda_min = float((k + n) / T)

    # 6. Log Marginal Data Density evaluator ln p(Y | lambda, theta)
    def compute_log_mdd(candidate_lambda: float) -> float:
        l_val = float(candidate_lambda)
        nu_post = (1.0 + l_val) * T - k
        nu_pri = l_val * T - k
        if nu_pri <= n:
            return -np.inf

        tilde_GXX = l_val * T * Gamma_XX + XtX
        tilde_GXY = l_val * T * Gamma_XY + XtY
        tilde_GYY = l_val * T * Gamma_YY + YtY

        try:
            tilde_Phi = np.linalg.solve(tilde_GXX, tilde_GXY)
            tilde_S = tilde_GYY - tilde_GXY.T @ tilde_Phi
            tilde_S = 0.5 * (tilde_S + tilde_S.T)

            sign_gxx, logdet_tilde_gxx = np.linalg.slogdet(tilde_GXX)
            sign_pri_gxx, logdet_pri_gxx = np.linalg.slogdet(l_val * T * Gamma_XX)
            sign_s, logdet_tilde_s = np.linalg.slogdet(tilde_S)
            sign_pri_s, logdet_pri_s = np.linalg.slogdet(l_val * T * Sigma_star)

            if sign_gxx <= 0 or sign_pri_gxx <= 0 or sign_s <= 0 or sign_pri_s <= 0:
                return -np.inf

            val = (
                _log_mvgamma(0.5 * nu_post, n)
                - _log_mvgamma(0.5 * nu_pri, n)
                - 0.5 * n * (logdet_tilde_gxx - logdet_pri_gxx)
                - 0.5 * nu_post * logdet_tilde_s
                + 0.5 * nu_pri * logdet_pri_s
                - 0.5 * n * T * np.log(np.pi)
            )
            return float(val)
        except np.linalg.LinAlgError:
            return -np.inf

    # 7. Determine lambda (grid, optimize, or fixed)
    hat_lambda: float | None = None
    log_mdd_grid: pd.DataFrame | None = None

    if lambda_grid is not None:
        grid_vals = [float(lv) for lv in lambda_grid]
        grid_mdds = [compute_log_mdd(lv) for lv in grid_vals]
        log_mdd_grid = pd.DataFrame({"lambda": grid_vals, "log_mdd": grid_mdds})

    should_optimize = lamb is None or (isinstance(lamb, str) and lamb.lower() == "optimal")

    if should_optimize:
        if log_mdd_grid is None:
            default_grid = np.unique(
                np.clip(
                    np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
                    lambda_min + 1e-3,
                    None,
                )
            )
            grid_mdds = [compute_log_mdd(lv) for lv in default_grid]
            log_mdd_grid = pd.DataFrame({"lambda": default_grid, "log_mdd": grid_mdds})

        # Bounded scalar optimization over [max(lambda_min + 1e-4, 0.2), 5.0]
        lower_bound = max(lambda_min + 1e-4, 0.2)
        upper_bound = max(5.0, lower_bound + 1.0)
        opt_res = scipy.optimize.minimize_scalar(
            lambda l_val: -compute_log_mdd(l_val),
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": 1e-4},
        )
        hat_lambda = float(opt_res.x)
        chosen_lambda = hat_lambda
    else:
        chosen_lambda = float(lamb)  # type: ignore[arg-type]
        if check_bounds and chosen_lambda < lambda_min:
            raise ValueError(
                f"lambda={chosen_lambda} is below admissibility bound lambda_min={lambda_min:.4f} "
                f"= (k + n) / T. Set lamb >= {lambda_min:.4f} for a proper prior, or check_bounds=False."
            )

    # 8. Posterior moments at chosen lambda
    if chosen_lambda == 0.0:
        tilde_Phi = Phi_ols
        tilde_Sigma = Sigma_ols
    else:
        tilde_GXX = chosen_lambda * T * Gamma_XX + XtX
        tilde_GXY = chosen_lambda * T * Gamma_XY + XtY
        tilde_GYY = chosen_lambda * T * Gamma_YY + YtY

        tilde_Phi = np.linalg.solve(tilde_GXX, tilde_GXY)
        tilde_S = tilde_GYY - tilde_GXY.T @ tilde_Phi
        tilde_Sigma = tilde_S / ((1.0 + chosen_lambda) * T)
        tilde_Sigma = 0.5 * (tilde_Sigma + tilde_Sigma.T)

    log_mdd_final = compute_log_mdd(chosen_lambda)

    # Unpack intercept and A_list
    if intercept:
        intercept_vec = tilde_Phi[0, :]
        A_list = [
            tilde_Phi[1 + l * n : 1 + (l + 1) * n, :].T
            for l in range(p)
        ]
    else:
        intercept_vec = np.zeros(n, dtype=float)
        A_list = [
            tilde_Phi[l * n : (l + 1) * n, :].T
            for l in range(p)
        ]

    resid = Y - X @ tilde_Phi

    # 9. Structural identification
    if identification == "dsge":
        dr = model.decision_rules()
        sd_shocks = model._shock_sd(None)
        shock_indices = [model.shocks.index(s) for s in shock_names]
        sd_diag = np.diag(sd_shocks[shock_indices])

        # DSGE structural impact matrix A0_dsge
        A0_dsge = dr.ghu.loc[list(obs_names), list(shock_names)].to_numpy(dtype=float) @ sd_diag

        Sigma_star_chol = np.linalg.cholesky(Sigma_star)
        Q_raw = np.linalg.solve(Sigma_star_chol, A0_dsge)
        Q_star, R = np.linalg.qr(Q_raw)
        d = np.diagonal(R)
        ph = np.where(d >= 0.0, 1.0, -1.0)
        Q_star = Q_star * ph

        tilde_Sigma_chol = np.linalg.cholesky(tilde_Sigma)
        B0 = tilde_Sigma_chol @ Q_star
    else:
        # Cholesky identification
        B0 = np.linalg.cholesky(tilde_Sigma)

    return DSGEVARResult(
        lamb=chosen_lambda,
        hat_lambda=hat_lambda,
        lambda_min=lambda_min,
        log_mdd=log_mdd_final,
        log_mdd_grid=log_mdd_grid,
        A_list=A_list,
        intercept=intercept_vec,
        Sigma=tilde_Sigma,
        B0=B0,
        Phi_star=Phi_star,
        Sigma_star=Sigma_star,
        Phi_ols=Phi_ols,
        Sigma_ols=Sigma_ols,
        resid=resid,
        data=df_obs,
        names=obs_names,
        shock_names=shock_names,
        p=p,
        T=T,
        model=model,
        identification=identification,
    )
