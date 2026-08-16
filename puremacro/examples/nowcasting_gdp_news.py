"""Mixed-Frequency GDP Nowcasting & News Decomposition.

Demonstrates real-time GDP tracking from monthly indicators with ragged release delays
following Giannone, Reichlin & Small (2008, *JME*):
- Dynamic Factor Model state space with EM missing value imputation.
- Quarterly aggregation bridge equation.
- News decomposition quantifying the surprise and impact of each data release.

Outputs:
  puremacro/examples/output/nowcasting_gdp_news.png
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from puremacro.nowcast import nowcast_gdp


def main():
    print("--- 1. Simulating Mixed-Frequency Panel & GDP ---")
    rng = np.random.default_rng(42)
    n_months = 72
    dates_m = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    
    # 2 true underlying common business cycle factors
    F_true = np.zeros((n_months, 2))
    for t in range(1, n_months):
        F_true[t, 0] = 0.85 * F_true[t-1, 0] + rng.normal(scale=0.8)
        F_true[t, 1] = 0.60 * F_true[t-1, 1] + rng.normal(scale=0.5)

    var_names = [
        "Industrial Production", "Payroll Employment", "Retail Sales",
        "Housing Starts", "Capacity Utilization", "PMI Manufacturing",
        "Core CPI Inflation", "Real Personal Income", "Export Orders", "Consumer Sentiment"
    ]
    N = len(var_names)
    X = np.zeros((n_months, N))
    for i in range(N):
        load = rng.uniform(0.4, 1.6, size=2)
        X[:, i] = F_true @ load + rng.normal(scale=0.4, size=n_months)

    df_X = pd.DataFrame(X, index=dates_m, columns=var_names)
    
    # Ragged edges: simulate staggered release delays for the current quarter
    df_X.iloc[-1, [3, 4, 8, 9]] = np.nan # Later releases not yet reported

    # Quarterly GDP series
    dates_q = pd.date_range("2018-01-01", periods=n_months // 3, freq="QS")
    F_q = df_X.resample("QE").mean().to_numpy().mean(axis=1)[:len(dates_q)]
    gdp = 2.2 + 1.4 * F_q + rng.normal(scale=0.3, size=len(dates_q))
    s_gdp = pd.Series(gdp, index=dates_q.to_period("Q").astype(str), name="GDP Growth")

    print("\n--- 2. Estimating Dynamic Factor Nowcast ---")
    res = nowcast_gdp(df_X, s_gdp, n_factors=2)
    print(res.summary())

    print("\n--- 3. Generating Publication Figures ---")
    out_dir = pathlib.Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nowcasting_gdp_news.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#fafbfc")

    # Panel A: Latent Monthly Common Factors
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(df_X.index, res.factors["Factor_1"], color="#1f77b4", lw=2, label="Factor 1 (Real Activity)")
    ax_a.plot(df_X.index, res.factors["Factor_2"], color="#ff7f0e", lw=2, linestyle="--", label="Factor 2 (Sentiment / Demand)")
    ax_a.set_title("(a) Smoothed Monthly Latent Factors F_t", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Date")
    ax_a.set_ylabel("Factor Level")
    ax_a.legend()
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Factor Loadings Heatmap
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#ffffff")
    im = ax_b.imshow(res.loadings.to_numpy(), cmap="coolwarm", aspect="auto")
    ax_b.set_yticks(range(N))
    ax_b.set_yticklabels(var_names, fontsize=8)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Factor 1", "Factor 2"])
    ax_b.set_title("(b) Estimated Factor Loadings Λ", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_b, label="Loading Weight")

    # Panel C: Release Surprise Innovations
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#ffffff")
    if not res.news_decomposition.empty:
        bars = ax_c.barh(res.news_decomposition["series"], res.news_decomposition["surprise"], color="#2ca02c", alpha=0.85, edgecolor="#333")
        for b in bars:
            if b.get_width() < 0:
                b.set_color("#d62728")
        ax_c.set_title("(c) Data Release Surprise (Actual - Forecast)", fontsize=11, fontweight="bold")
        ax_c.set_xlabel("Surprise Innovation (σ units)")
        ax_c.grid(True, linestyle=":", alpha=0.6)

    # Panel D: News Contribution to GDP Nowcast Revision
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#ffffff")
    if not res.news_decomposition.empty:
        bars_d = ax_d.barh(res.news_decomposition["series"], res.news_decomposition["contribution"], color="#1f77b4", alpha=0.85, edgecolor="#333")
        for b in bars_d:
            if b.get_width() < 0:
                b.set_color("#d62728")
        ax_d.set_title("(d) Impact on GDP Nowcast (Percentage Points)", fontsize=11, fontweight="bold")
        ax_d.set_xlabel("Contribution to Nowcast Revision (pp)")
        ax_d.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()
