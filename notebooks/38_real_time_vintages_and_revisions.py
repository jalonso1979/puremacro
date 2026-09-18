# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Real-Time Data, Historical Vintages & The Mankiw-Shapiro Test
#
# **How do data revisions affect macroeconomic analysis, and how can we determine whether initial statistical releases reflect rational forecasts ("News") or measurement error ("Noise")?**
#
# Macroeconomic aggregates like Gross Domestic Product (GDP), consumption, and investment are continuously revised as statistical agencies incorporate higher-quality administrative data, benchmark surveys, and methodological updates:
#
# 1. **Real-Time Information & Policy Errors**:
#    Athanasios Orphanides (2001, *American Economic Review*) showed that the monetary policy mistakes of the 1970s arose from estimating the output gap on real-time data that subsequently underwent massive revisions. Croushore & Stark (2001, *Journal of Economic Literature*) formalized the real-time data architecture.
#
# 2. **The Mankiw & Shapiro (1986) News vs. Noise Econometric Test**:
#    Let $y_{0,t}$ be the initial statistical release of GDP growth at quarter $t$, and $y_{T,t}$ be the final revised benchmark. The revision is:
#    $$ r_t = y_{T,t} - y_{0,t} $$
#    Mankiw & Shapiro (1986, *Journal of Business & Economic Statistics*) formulate the OLS regression:
#    $$ r_t = \alpha + \beta \, y_{0,t} + \varepsilon_t $$
#
#    - **News Hypothesis ($\beta = 0$, $R^2 \approx 0$)**: The initial release is an optimal conditional forecast given information at time $t$ ($y_{0,t} = \mathbb{E}[y_{T,t} \mid \Omega_t]$). The revision $r_t$ represents genuinely new statistical information orthogonal to $y_{0,t}$.
#    - **Noise Hypothesis ($\beta = -1$, $\alpha = 0$)**: The initial release is equal to the true value corrupted by classical measurement error ($y_{0,t} = y_{T,t} + v_t$, where $\text{Cov}(y_{T,t}, v_t) = 0$). Hence, $r_t = -v_t$ is negatively correlated with $y_{0,t}$.
#
# In this notebook, we explore multi-country Quarterly National Accounts (QNA) vintages, construct $(T \times V)$ revision triangles, slice point-in-time real-time panels, and execute the Mankiw-Shapiro test using `puremacro.fetch` and `puremacro.vintages`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.fetch import (
    get_qna_vintage_catalog,
    QNAVintagePanel,
)

# %% [markdown]
# ## 1. Inspecting the 45+ Country QNA Vintage Catalog
#
# Real-time macroeconomic research requires access to historical snapshots as they appeared to policymakers and market participants at the time decisions were made. In archival databases such as the Federal Reserve Bank of St. Louis ALFRED (ArchivaL Federal Reserve Economic Data), each macroeconomic indicator is recorded along with its publication date (vintage).
#
# `puremacro` provides a standardized cross-country catalog mapping 45+ economies (all 38 OECD members, G7, G20, and key emerging markets) to real-time historical publication vintages on ALFRED. Each entry captures the country code, geographic region, standardized variable definition (such as real GDP, real gross fixed capital formation, or household consumption), and the ALFRED native series identifier.

# %%
catalog = get_qna_vintage_catalog()
print(f"Total catalog series available: {len(catalog)}")
print("\nSample of Catalog Entries:")
print(catalog[["country", "country_name", "region", "variable", "series_id"]].head(10))

# Summary by variable
var_counts = catalog.groupby("variable")["country"].count().reset_index()
var_counts.columns = ["Variable", "Country Count"]
print("\nVariable Coverage across Countries:")
print(var_counts)

# %% [markdown]
# ## 2. Constructing a Real-Time Vintage Panel
#
# Let us simulate a realistic multi-country historical revision structure spanning 40 quarters ($T=40$) and 50 publication vintages ($V=50$) to demonstrate the real-time analytics pipeline.
#
# Let $y_t^*$ denote the true underlying economic activity in quarter $t$. The advance statistical release $y_{0, t}$ is published with a one-quarter lag ($t+1$) and contains both latent state information and initial survey measurement error $v_t \sim \mathcal{N}(0, \sigma_v^2)$:
#
# $$ y_{0, t} = y_t^* + v_t $$
#
# Over subsequent publication vintages $j \ge 1$, statistical agencies incorporate comprehensive administrative tax records, annual business censuses, and revised seasonal adjustment factors. The vintage estimate $y_{j, t}$ converges gradually toward the benchmark value with geometric decay:
#
# $$ y_{j, t} = y_t^* + e^{-j / \kappa} \, v_t, \qquad \kappa > 0 $$

# %%
rng = np.random.default_rng(1986)
n_quarters = 40
obs_dates = pd.date_range("2010-01-01", periods=n_quarters, freq="QS")
vintage_dates = pd.date_range("2010-04-01", periods=50, freq="QS")

countries = ["USA", "DEU", "GBR", "MEX"]
variables = ["gdp_real", "gfcf_real"]

records = []
for c in countries:
    for var in variables:
        # True underlying latent growth rate
        true_growth = rng.normal(loc=2.2, scale=1.5, size=n_quarters)
        
        # Initial release (advance estimate) with noise + news components
        noise = rng.normal(loc=0.0, scale=0.6, size=n_quarters)
        y_0 = true_growth + noise
        
        for i, obs in enumerate(obs_dates):
            # First publication occurs 1 quarter after observation
            first_pub_idx = i + 1
            for j, vint in enumerate(vintage_dates):
                if j >= first_pub_idx:
                    # Gradual revision toward true final benchmark over 4 vintages
                    lag = j - first_pub_idx
                    decay = np.exp(-lag / 3.0)
                    val = true_growth[i] + decay * noise[i]
                    records.append({
                        "country": c,
                        "variable": var,
                        "date": obs,
                        "vintage": vint,
                        "value": float(val),
                    })

df_raw = pd.DataFrame(records)
panel = QNAVintagePanel(df=df_raw)
print(f"Built QNAVintagePanel with {len(df_raw):,} records across {len(countries)} countries.")

# %% [markdown]
# ## 3. Visualizing the $(T \times V)$ Revision Triangle
#
# The canonical representation of vintage data is the lower-triangular revision matrix $\mathbf{R} \in \mathbb{R}^{T \times V}$:
#
# $$ \mathbf{R} = \begin{bmatrix} y_{1, v_1} & y_{1, v_2} & y_{1, v_3} & \dots & y_{1, v_V} \\ \text{NaN} & y_{2, v_2} & y_{2, v_3} & \dots & y_{2, v_V} \\ \text{NaN} & \text{NaN} & y_{3, v_3} & \dots & y_{3, v_V} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \text{NaN} & \text{NaN} & \text{NaN} & \dots & y_{T, v_V} \end{bmatrix} $$
#
# Each row index corresponds to an observation period $t$ (quarter of reference), while each column corresponds to a publication vintage date $v$. The leading diagonal contains the first release (advance estimate) for each quarter. Reading across any single row traces the historical lifecycle of revisions for that specific quarter as statistical agencies refine their estimates.

# %%
tri_usa = panel.revision_matrix("USA", "gdp_real")
print("USA Real GDP Growth Revision Matrix (first 6 quarters × 6 vintages):")
print(tri_usa.iloc[:6, :6])

# %%
fig, ax = _nbstyle.figura(figsize=(9.2, 5.2))

# Plot heatmap of available vintages
im = ax.imshow(tri_usa.iloc[:24, :24].to_numpy(), cmap=_nbstyle.CMAP_SEQ, aspect="auto")
ax.set_title("USA Real GDP Growth: Historical Revision Triangle (T × V)", fontsize=11, fontweight="bold")
ax.set_xlabel("Publication Vintage Date Index", color=_nbstyle.TEXTO)
ax.set_ylabel("Observation Quarter Index", color=_nbstyle.TEXTO)

# Formatting tick labels
ax.set_xticks(range(0, 24, 4))
ax.set_xticklabels([d.strftime("%YQ%q") for d in tri_usa.columns[:24:4]], rotation=45)
ax.set_yticks(range(0, 24, 4))
ax.set_yticklabels([d.strftime("%YQ%q") for d in tri_usa.index[:24:4]])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Annualized Real GDP Growth (%)", color=_nbstyle.TEXTO)

# %% [markdown]
# ## 4. First Release vs. Latest Benchmark Revisions
#
# Comparing the advance initial estimate against the latest available benchmark reveals the magnitude, direction, and cyclical persistence of macroeconomic revisions:
#
# $$ r_t = y_{T, t} - y_{0, t} $$
#
# If revisions are systematically non-zero on average ($\bar{r} \neq 0$), the initial release suffers from statistical bias. Furthermore, if revisions correlate with macroeconomic expansions or contractions, policy decisions based on unrevised indicators may inadvertently amplify the business cycle (Orphanides, 2001).

# %%
s_first = panel.first_release("USA", "gdp_real")
s_latest = panel.latest_release("USA", "gdp_real")

fig, (ax1, ax2) = _nbstyle.figura(2, 1, figsize=(9.5, 6.0), sharex=True)

# Panel 1: Series Levels
ax1.plot(s_first.index, s_first.values, **_nbstyle.S1, label="First Release (Advance Estimate)")
ax1.plot(s_latest.index, s_latest.values, **_nbstyle.S2, label="Latest Revised Benchmark")
ax1.set_title("USA Real GDP Growth: Initial vs. Final Revised Series", fontsize=11, fontweight="bold")
ax1.set_ylabel("Growth Rate (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# Panel 2: Total Revision (Final - First)
revision = s_latest - s_first
ax2.bar(revision.index, revision.values, color=_nbstyle.S3["color"], width=60, edgecolor=_nbstyle.SPINE, alpha=0.8, label="Revision ($y_T - y_0$)")
ax2.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="-")
ax2.set_title("Total Historical Revision Series", fontsize=11, fontweight="bold")
ax2.set_xlabel("Observation Date", color=_nbstyle.TEXTO)
ax2.set_ylabel("Revision (% pts)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 5. Executing the Mankiw-Shapiro (1986) News vs. Noise Test
#
# Gregory Mankiw and Matthew Shapiro (1986) developed the foundational econometric framework for evaluating the rationality of preliminary statistical data.
#
# We estimate the OLS regression of the revision $r_t = y_{T, t} - y_{0, t}$ on the initial estimate $y_{0, t}$:
#
# $$ r_t = \alpha + \beta \, y_{0, t} + \varepsilon_t $$
#
# ### Theoretical Hypotheses
# 1. **News Hypothesis ($H_{\text{news}}$)**:
#    Statistical agencies form rational forecasts of the final benchmark based on all currently available information set $\Omega_t$:
#    $$ y_{0, t} = \mathbb{E}[y_{T, t} \mid \Omega_t] \implies y_{T, t} = y_{0, t} + \nu_t, \quad \text{with } \mathbb{E}[\nu_t \mid \Omega_t] = 0 $$
#    Under rational expectations, the revision $r_t = \nu_t$ represents pure *news* that is entirely unpredictable from the initial release:
#    $$ \alpha = 0, \qquad \beta = 0, \qquad R^2 \approx 0 $$
#
# 2. **Noise Hypothesis ($H_{\text{noise}}$)**:
#    The statistical agency observes the true benchmark corrupted by classical measurement error $u_t$:
#    $$ y_{0, t} = y_{T, t} + u_t, \quad \text{with } \text{Cov}(y_{T, t}, u_t) = 0, \quad u_t \sim \text{i.i.d.}(0, \sigma_u^2) $$
#    In this case, the revision is simply the negative of the measurement noise ($r_t = -u_t$). Regressing $r_t$ on $y_{0, t}$ yields:
#    $$ \beta = \frac{\text{Cov}(-u_t, y_{T, t} + u_t)}{\text{Var}(y_{0, t})} = \frac{-\sigma_u^2}{\sigma_y^2 + \sigma_u^2} < 0 $$
#    In the limit where noise dominates, $\beta \to -1$.

# %%
stats_usa = panel.revision_stats("USA", "gdp_real")
print("==================================================================")
print("  MANKIW & SHAPIRO (1986) NEWS VS NOISE ECONOMETRIC TEST")
print("==================================================================")
print(f"Country:               USA")
print(f"Variable:              Real GDP (gdp_real)")
print(f"Sample Size (N):       {stats_usa['n_obs']}")
print(f"Mean Revision (alpha): {stats_usa['mean_revision']:.4f}")
print(f"Abs Mean Revision:     {stats_usa['abs_mean_revision']:.4f}")
print(f"Revision Std Dev:      {stats_usa['std_revision']:.4f}")
print("------------------------------------------------------------------")
print(f"Regression Alpha:        {stats_usa['mankiw_shapiro_alpha']:.4f}")
print(f"Regression Slope (beta): {stats_usa['mankiw_shapiro_beta']:.4f}")
print(f"Beta Std Error:          {stats_usa['mankiw_shapiro_se']:.4f}")
print(f"t-Statistic (H0: beta=0):{stats_usa['mankiw_shapiro_tstat']:.4f}")
print(f"p-Value:                 {stats_usa['mankiw_shapiro_pvalue']:.4f}")
print(f"Conclusion:              {stats_usa['hypothesis']}")
print("==================================================================")

# %%
# Cross-country test comparison
results_all = []
for c in countries:
    for v in variables:
        st = panel.revision_stats(c, v)
        results_all.append({
            "Country": c,
            "Variable": v,
            "N": st["n_obs"],
            "Mean Revision": st["mean_revision"],
            "Std Revision": st["std_revision"],
            "Beta": st["mankiw_shapiro_beta"],
            "t-stat": st["mankiw_shapiro_tstat"],
            "p-value": st["mankiw_shapiro_pvalue"],
            "Hypothesis": st["hypothesis"],
        })

df_test_summary = pd.DataFrame(results_all)
print("\nCross-Country Mankiw-Shapiro Test Summary Table:")
print(df_test_summary.to_string(index=False))

# %% [markdown]
# ## 6. Visualizing the Mankiw-Shapiro Regression Scatter
#
# Plotting the initial release $y_{0, t}$ on the horizontal axis against the revision $r_t = y_{T, t} - y_{0, t}$ on the vertical axis provides an intuitive geometric diagnostic:
#
# - A horizontal line ($\beta = 0$) corresponds to the **Pure News benchmark**, indicating efficient forecasts.
# - A downward-sloping line with slope $\beta = -1$ corresponds to the **Pure Noise benchmark**, indicating unadjusted survey error.
# - The estimated OLS regression slope reveals whether statistical agencies under- or over-adjust preliminary indicators.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.8))

# Scatter plot: Initial release vs Total Revision
x_vals = s_first.values
y_vals = (s_latest - s_first).values
ax.scatter(x_vals, y_vals, color=_nbstyle.S1["color"], edgecolors=_nbstyle.SPINE, s=50, alpha=0.85, label="Observations ($y_{0,t}, r_t$)")

# Fitted OLS regression line
x_grid = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 100)
y_fit = stats_usa["mean_revision"] + stats_usa["mankiw_shapiro_beta"] * x_grid
ax.plot(x_grid, y_fit, **_nbstyle.S2, label=f"OLS Fit ($\\beta={stats_usa['mankiw_shapiro_beta']:.2f}$, p={stats_usa['mankiw_shapiro_pvalue']:.3f})")

# Theoretical Noise line (slope = -1)
y_noise = -1.0 * x_grid
ax.plot(x_grid, y_noise, color=_nbstyle.NOTA, lw=1.5, linestyle=":", label="Pure Noise Benchmark ($\\beta=-1$)")

# Theoretical News line (slope = 0)
ax.axhline(0, color=_nbstyle.SPINE, lw=1.2, linestyle="--", label="Pure News Benchmark ($\\beta=0$)")

ax.set_title("Mankiw & Shapiro (1986) News vs. Noise Diagnostic Plot", fontsize=11, fontweight="bold")
ax.set_xlabel("Initial Release $y_{0,t}$ (%)", color=_nbstyle.TEXTO)
ax.set_ylabel("Total Revision $y_{T,t} - y_{0,t}$ (% pts)", color=_nbstyle.TEXTO)
ax.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 7. Real-Time Point-in-Time Dataset Slicing (`.as_of()`)
#
# When conducting pseudo-out-of-sample forecasting experiments or structural VAR historical decompositions, utilizing revised data introduces lookahead bias (endogeneity through future revisions).
#
# The method `panel.as_of(date)` reconstructs the exact cross-sectional and time-series information available to an econometrician as of a specified publication date. For any historical vintage $V^*$, it filters all series to satisfy:
#
# $$ \mathcal{I}_{V^*} = \left\{ y_{v, t} \;\Big|\; v \le V^* \text{ and } v = \max_{u \le V^*} u \right\} $$

# %%
# Slicing the exact state of knowledge as of 2018-04-01
df_2018 = panel.as_of("2018-04-01")
print("Historical Snapshot Panel as of 2018-04-01:")
print(df_2018.head(10))

# Slicing as of 2022-01-01
df_2022 = panel.as_of("2022-01-01")
print(f"\nObservations available in 2018 snapshot: {len(df_2018)}")
print(f"Observations available in 2022 snapshot: {len(df_2022)}")

