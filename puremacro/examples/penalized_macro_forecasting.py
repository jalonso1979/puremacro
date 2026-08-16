"""High-Dimensional Penalized Macroeconomic Forecasting.

Demonstrates sparse predictor selection and multi-horizon forecasting with Elastic Net
and Adaptive Lasso following Zou (2006, *JASA*) and Hastie et al. (2015):
- Pure NumPy Coordinate Descent.
- BIC regularisation tuning path across 40 lambda points.
- Sparse coefficient recovery and point forecasting.

Outputs:
  puremacro/examples/output/penalized_macro_forecasting.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.forecast import forecast_penalized


def main():
    print("--- 1. Simulating High-Dimensional Macroeconomic Panel ---")
    rng = np.random.default_rng(123)
    T = 160
    P = 30
    dates = pd.date_range("2010-01-01", periods=T, freq="MS")
    
    # 30 macro predictors with AR(1) dynamics
    X = np.zeros((T, P))
    for j in range(P):
        rho = rng.uniform(0.3, 0.8)
        for t in range(1, T):
            X[t, j] = rho * X[t-1, j] + rng.normal(scale=0.8)

    # Target series (e.g. CPI Inflation) driven by 4 key predictors + noise
    y = np.zeros(T)
    active_indices = [1, 5, 12, 22]
    weights = [1.8, -1.4, 1.2, -0.9]
    for t in range(1, T):
        signal = sum(w * X[t-1, idx] for w, idx in zip(weights, active_indices))
        y[t] = 2.0 + signal + rng.normal(scale=0.5)
    y[0] = 2.0

    df_X = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
    s_y = pd.Series(y, index=dates, name="CPI Inflation")

    print("\n--- 2. Estimating Elastic Net & Adaptive Lasso Forecast ---")
    res_enet = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=False)
    print("Elastic Net Result:")
    print(res_enet.summary())

    res_alasso = forecast_penalized(df_X, s_y, horizon=1, alpha=1.0, adaptive=True)
    print("\nAdaptive Lasso Result:")
    print(res_alasso.summary())

    print("\n--- 3. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "penalized_macro_forecasting.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Actual vs In-Sample Fit
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    fitted_vals = res_alasso.intercept + df_X.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
    ax_a.plot(dates[1:], s_y.iloc[1:], color="black", lw=1.5, label="Actual Inflation")
    ax_a.plot(dates[1:], fitted_vals, color="#1f77b4", lw=2, linestyle="--", label=f"Adaptive Lasso Fit (R²={res_alasso.in_sample_r2:.2f})")
    ax_a.set_title("(a) Actual vs. Penalized Model Fitted Values", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Inflation Rate (%)")
    ax_a.legend()
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: BIC Regularisation Path
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    ax_b.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, color="#d62728", lw=2, marker="o", markersize=3)
    ax_b.axvline(np.log10(res_alasso.optimal_lambda), color="black", linestyle="--", label=f"Optimal λ* = {res_alasso.optimal_lambda:.4f}")
    ax_b.set_title("(b) Bayesian Information Criterion (BIC) Path", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("log10(λ)")
    ax_b.set_ylabel("BIC Score")
    ax_b.legend()
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Sparse Estimated Coefficients
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    nonzeros = res_alasso.coefficients[res_alasso.coefficients.abs() > 1e-5]
    if not nonzeros.empty:
        nonzeros.plot(kind="barh", ax=ax_c, color="#2ca02c", edgecolor="#333", alpha=0.85)
    ax_c.set_title("(c) Selected Predictor Coefficients (Adaptive Lasso)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Estimated Coefficient β")
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Elastic Net vs. Adaptive Lasso Comparison
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    df_comp = pd.DataFrame({
        "Elastic Net (α=0.5)": res_enet.coefficients,
        "Adaptive Lasso (α=1.0)": res_alasso.coefficients,
    })
    top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
    if not top_feats.empty:
        top_feats.plot(kind="bar", ax=ax_d, edgecolor="#333", alpha=0.85)
    ax_d.set_title("(d) Coefficient Shrinkage: Elastic Net vs. Adaptive Lasso", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Coefficient Magnitude")
    ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax_d.legend()
    ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
