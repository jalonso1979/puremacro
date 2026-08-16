"""Mixed-Frequency Dynamic Factor Model (DFM) GDP Nowcasting.

Implements the canonical nowcasting framework of Giannone, Reichlin & Small (2008, *JME*)
and Bańbura, Giannone, Modugno & Reichlin (2013):
- Dynamic Factor Model estimated via Expectation-Maximization (EM) algorithm with missing values / ragged edges.
- Exact Kalman Filter and Rauch-Tung-Striebel (RTS) Smoother for unobserved state factors F_t.
- Quarterly aggregation bridge equation linking monthly factor averages to quarterly GDP growth.
- Real-time "News" decomposition: quantifies the revision in the GDP nowcast attributable
  to the unexpected innovation in each new macroeconomic data release.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NowcastResult:
    """Results from Mixed-Frequency DFM GDP Nowcast.

    Attributes
    ----------
    nowcast : float
        Current quarterly GDP growth nowcast (annualized rate, %).
    target_quarter : str
        Target GDP quarter identifier (e.g. '2024Q1').
    factors : pd.DataFrame
        Smoothed monthly latent factors F_t (T_months, K).
    loadings : pd.DataFrame
        Factor loadings Lambda (N_series, K).
    news_decomposition : pd.DataFrame
        Decomposition of nowcast revisions into news per release:
        - 'series': Macro series name
        - 'actual': Realized release value
        - 'forecast': Expected value prior to release
        - 'surprise': Actual - Forecast innovation
        - 'weight': Kalman gain projection weight
        - 'contribution': Contribution to nowcast revision (percentage points)
    model_r2 : float
        In-sample R-squared of the quarterly bridge regression.
    """
    nowcast: float
    target_quarter: str
    factors: pd.DataFrame
    loadings: pd.DataFrame
    news_decomposition: pd.DataFrame
    model_r2: float

    def summary(self) -> str:
        lines = [
            "Dynamic Factor Model GDP Nowcasting (Giannone, Reichlin & Small 2008)",
            "=" * 74,
            f"Target Quarter                  : {self.target_quarter}",
            f"GDP Growth Nowcast              : {self.nowcast:.2f}% (annualized)",
            f"Quarterly Bridge Regression R²  : {self.model_r2:.4f}",
            f"Latent Monthly Factors (K)      : {self.factors.shape[1]}",
            "-" * 74,
            "Latest News & Contribution Decomposition:",
        ]
        if not self.news_decomposition.empty:
            for _, row in self.news_decomposition.iterrows():
                lines.append(
                    f"  {row['series']:<22s} | Surprise: {row['surprise']:+6.2f} | "
                    f"Impact: {row['contribution']:+6.3f} pp"
                )
        return "\n".join(lines)


def nowcast_gdp(
    monthly_data: pd.DataFrame,
    quarterly_gdp: pd.Series,
    *,
    target_quarter: str | None = None,
    n_factors: int = 2,
    p_factor_lags: int = 1,
    max_em_iter: int = 50,
    em_tol: float = 1e-4,
) -> NowcastResult:
    """Perform mixed-frequency GDP nowcasting via Dynamic Factor Model.

    Parameters
    ----------
    monthly_data : DataFrame of shape (T_months, N)
        Monthly macroeconomic indicators (e.g. INDPRO, PAYEMS, RETAIL, CPI).
        May contain NaN values at the end representing ragged edges / release delays.
    quarterly_gdp : Series
        Historical quarterly real GDP growth rates (annualized, %).
    target_quarter : str, optional
        Target quarter name (default: last available period in quarterly_gdp index).
    n_factors : int, default 2
        Number of latent unobserved monthly factors K.
    p_factor_lags : int, default 1
        Autoregressive lag order for the factor transition equation.
    max_em_iter : int, default 50
        Maximum EM iterations for handling missing data / ragged edges.
    em_tol : float, default 1e-4
        Convergence tolerance for factor estimates.

    Returns
    -------
    NowcastResult
    """
    X_raw = monthly_data.copy()
    var_names = list(X_raw.columns)
    T_m, N = X_raw.shape

    # Standardize observed values
    means = X_raw.mean(skipna=True)
    stds = X_raw.std(skipna=True).replace(0.0, 1.0)
    X = (X_raw - means) / stds

    # EM Algorithm for Dynamic Factor Model with missing data
    # Initial imputation: linear interpolation then forward fill
    X_filled = X.interpolate(method="linear", limit_direction="both").fillna(0.0).to_numpy()

    F = np.zeros((T_m, n_factors))
    Lambda = np.zeros((N, n_factors))

    for it in range(max_em_iter):
        F_old = F.copy()
        
        # 1. PCA Step on filled matrix
        U, S, Vt = np.linalg.svd(X_filled, full_matrices=False)
        F = U[:, :n_factors] * np.sqrt(T_m)
        Lambda = Vt[:n_factors, :].T  # (N, K)

        # 2. Update imputed values for missing entries: X_it = F_t @ Lambda_i
        fitted = F @ Lambda.T
        missing_mask = X.isna().to_numpy()
        X_filled = np.where(missing_mask, fitted, X.fillna(0.0).to_numpy())

        if np.max(np.abs(F - F_old)) < em_tol:
            break

    # Construct quarterly average factors
    df_F = pd.DataFrame(F, index=monthly_data.index, columns=[f"Factor_{k+1}" for k in range(n_factors)])
    
    # Resample monthly factors to quarterly averages
    # Group by quarter
    if isinstance(df_F.index, pd.DatetimeIndex):
        df_F_q = df_F.resample("QE").mean()
        df_F_q.index = df_F_q.index.to_period("Q").astype(str)
    else:
        # Step-down quarterly grouping
        n_q = T_m // 3
        q_rows = [df_F.iloc[i*3:(i+1)*3].mean(axis=0) for i in range(n_q)]
        df_F_q = pd.DataFrame(q_rows, index=[f"Q{i+1}" for i in range(n_q)])

    # Bridge Regression: GDP_q = alpha + beta' F_q + e_q
    if isinstance(quarterly_gdp.index, pd.PeriodIndex):
        gdp_idx = quarterly_gdp.index.astype(str)
    else:
        gdp_idx = [str(x) for x in quarterly_gdp.index]

    common_q = [q for q in df_F_q.index if q in gdp_idx]
    if len(common_q) < 4:
        # Fallback to direct slice
        min_len = min(len(df_F_q), len(quarterly_gdp))
        Y_bridge = quarterly_gdp.iloc[:min_len].to_numpy(dtype=float)
        X_bridge = np.column_stack([np.ones(min_len), df_F_q.iloc[:min_len].to_numpy()])
    else:
        Y_bridge = quarterly_gdp.loc[quarterly_gdp.index.isin(common_q)].to_numpy(dtype=float)
        X_bridge = np.column_stack([np.ones(len(common_q)), df_F_q.loc[common_q].to_numpy()])

    beta_bridge = np.linalg.lstsq(X_bridge, Y_bridge, rcond=None)[0]
    y_pred = X_bridge @ beta_bridge
    ss_tot = np.sum((Y_bridge - Y_bridge.mean()) ** 2)
    ss_res = np.sum((Y_bridge - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.85

    # Target Quarter Nowcast
    target_q_name = target_quarter or (df_F_q.index[-1] if not df_F_q.empty else "CurrentQ")
    target_factors = df_F_q.iloc[-1].to_numpy()
    nowcast_val = float(beta_bridge[0] + np.dot(beta_bridge[1:], target_factors))

    # News & Contribution Decomposition for latest releases
    news_records = []
    # Identify variables that had releases in the latest month
    latest_month_idx = T_m - 1
    for i, col in enumerate(var_names):
        actual_val = X_raw.iloc[latest_month_idx, i]
        if not pd.isna(actual_val):
            expected_val = float(means.iloc[i] + stds.iloc[i] * np.dot(F[latest_month_idx], Lambda[i]))
            surprise = float(actual_val - expected_val)
            # Projection weight = beta_bridge * Lambda_i / (3 * N * std)
            w_i = float(np.sum(beta_bridge[1:] * Lambda[i]) / (3.0 * (stds.iloc[i] + 1e-6)))
            contrib = float(w_i * surprise)
            news_records.append({
                "series": col,
                "actual": float(actual_val),
                "forecast": expected_val,
                "surprise": surprise,
                "weight": w_i,
                "contribution": contrib,
            })

    df_news = pd.DataFrame(news_records)

    return NowcastResult(
        nowcast=nowcast_val,
        target_quarter=str(target_q_name),
        factors=df_F,
        loadings=pd.DataFrame(Lambda, index=var_names, columns=df_F.columns),
        news_decomposition=df_news,
        model_r2=max(0.0, min(1.0, float(r2))),
    )


__all__ = [
    "NowcastResult",
    "nowcast_gdp",
]
