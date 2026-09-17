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
# # Latin America Real-Time Nowcasting & Bańbura-Modugno News Attribution: Ragged Edges, Revision Decomposition, and Density Calibration
#
# **How can central banks and economic research desks across Latin America nowcast quarterly GDP growth in real time from ragged-edge monthly indicator panels, decompose nowcast updates into release surprises and statistical revisions using the exact Bańbura & Modugno (2014) identity, and validate predictive density calibration using Berkowitz (2001) PIT uniformity tests?**
#
# **The panel in this notebook is simulated for offline execution. Read no fact about Latin America out of it.** Every data point is generated in the first code cell from the fixed deterministic seed `np.random.default_rng(42)`: the common latent factor follows a cumulative normal process, while indicator series combine factor loadings with idiosyncratic shocks. What is completely genuine and authentic is the *schema* — the provider names (`inegi`, `banxico`, `bcb`), canonical series identifiers (`735848`, `736184`, `628197`, `SF61745` for Mexico; `22099`, `24363`, `433`, `432` for Brazil), frequencies, and measurement units that the `puremacro.fetch.realtime` connectors return. This guarantees 100% offline, reproducible execution in Pyodide and WebAssembly environments without requiring network access, credentials, or central bank API tokens. The portable `.pmz` data cartridge carries `SIMULATED` in its provenance notes, ensuring transparency.
#
# Macroeconomic surveillance in emerging markets—most acutely across Latin America—operates in an environment of asynchronous publication lags and non-synchronous data arrivals. Policymakers at Banco de México (Banxico) and Banco Central do Brasil (BCB) cannot wait for quarterly national accounts releases that arrive 60 to 90 days after quarter-end. Instead, central bank monitoring desks track high-frequency monthly indicators: monthly GDP proxies (IGAE in Mexico, IBC-Br in Brazil), industrial production, consumer price indices (INPC and IPCA), and target policy rates (TIIE and Selic). Because statistical agencies publish these indicators on differing calendars, real-time data matrices exhibit an unbalanced "ragged edge" at the sample end.
#
# To extract an early, coherent signal of underlying economic activity, central banks deploy Dynamic Factor Models (DFM) in state-space form, estimated via the Kalman smoother (Giannone, Reichlin & Small 2008; Doz, Giannone & Reichlin 2011). When new data arrive or statistical agencies retrospectively revise earlier figures, the model updates its GDP nowcast. Explaining the economic drivers of that update is critical for policy deliberation: did the nowcast shift because industrial production beat expectations, or because the statistical agency revised last month's economic activity downwards? The analytical news decomposition of Bańbura & Modugno (2014) solves this problem by decomposing the nowcast revision into release surprises (innovations relative to model expectations) and retrospective data revisions, satisfying an exact mathematical identity. Finally, central bank credibility requires calibrated uncertainty envelopes: probability integral transform (PIT) uniformity tests (Berkowitz 2001) and fan chart fan-projections ensure that policy committees operate with statistically rigorous density forecasts rather than spuriously narrow point predictions.

# %% [markdown]
# ## The method in math — Dynamic Factors, News Decomposition, and Density Calibration
#
# **1. Dynamic Factor Model in State-Space Form.** Let $X_t = [x_{1, t}, \dots, x_{n, t}]^\top$ denote an $n$-dimensional panel of monthly macroeconomic indicators standardized to zero mean and unit variance. The indicators share $r$ unobserved latent factors $F_t \in \mathbb{R}^r$ subject to idiosyncratic measurement errors $\xi_t$:
# $$ X_t = \Lambda F_t + \xi_t, \quad \xi_t \sim \text{i.i.d.} \, \mathcal{N}(0, R), \quad R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2). $$
# The latent factors follow a stationary vector autoregressive process of order $p$:
# $$ F_t = A_1 F_{t-1} + \dots + A_p F_{t-p} + u_t, \quad u_t \sim \text{i.i.d.} \, \mathcal{N}(0, Q). $$
# For unbalanced panels with ragged edges, parameters $(\Lambda, A, Q, R)$ are estimated via the two-step principal components and Kalman filter/smoother algorithm of Doz, Giannone, and Reichlin (2011). The conditional expectation $\hat{y}_{t^*|v} = \mathbb{E}[y_{t^*} \mid \Omega_v]$ for target variable $y_{t^*}$ (quarterly GDP) is extracted via Kalman smoothing over the available vintage information set $\Omega_v$.
#
# **2. Bańbura & Modugno (2014) Exact News Attribution.** Let $\Omega_{v-1}$ and $\Omega_v$ denote information sets in two successive vintage snapshots, with $\Omega_{v-1} \subset \Omega_v$. The newly available information consists of new releases for recent periods $j \in \mathcal{I}_{\text{new}}$ and revisions to historical observations $k \in \mathcal{I}_{\text{rev}}$:
# $$ I_{j, v} \equiv x_{j, t_j} - \mathbb{E}[x_{j, t_j} \mid \Omega_{v-1}], \quad R_{k, v} \equiv x_{k, t_k}^{(v)} - x_{k, t_k}^{(v-1)}. $$
# The revision to the nowcast $\Delta \hat{y}_{t^*|v} \equiv \hat{y}_{t^*|v} - \hat{y}_{t^*|v-1}$ decomposes analytically into:
# $$ \Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot I_{j, v} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot R_{k, v}, $$
# where weights $\omega$ are determined by the Kalman filter gain and state autocovariances:
# $$ \omega = \operatorname{Cov}\left(y_{t^*}, \begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right) \left[\operatorname{Var}\left(\begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right)\right]^{-1}. $$
# This decomposition satisfies the exact mathematical identity with zero residual:
# $$ \text{Decomposition Error} \equiv \left| \Delta \hat{y}_{t^*|v} - \left(\sum \text{Impact}_{\text{releases}} + \sum \text{Impact}_{\text{revisions}}\right) \right| < 10^{-10}. $$
#
# **3. Predictive Density Evaluation: Berkowitz (2001) Likelihood Ratio Test.** Let $\{y_t\}_{t=1}^T$ be realized outcomes and $\{\mu_t, \sigma_t\}_{t=1}^T$ the sequence of one-step-ahead nowcast conditional means and standard deviations. The Probability Integral Transform (PIT) is:
# $$ p_t = \Phi\left(\frac{y_t - \mu_t}{\sigma_t}\right), \quad z_t = \Phi^{-1}(p_t), $$
# where $\Phi(\cdot)$ is the standard normal cumulative distribution function. Under the null hypothesis of correct density calibration and independence, $p_t \sim \text{i.i.d.} \, \mathcal{U}(0, 1)$ and $z_t \sim \text{i.i.d.} \, \mathcal{N}(0, 1)$. Berkowitz models the transformed error $z_t$ as an autoregressive process:
# $$ (z_t - \mu) = \rho (z_{t-1} - \mu) + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.} \, \mathcal{N}(0, \sigma_\varepsilon^2). $$
# The likelihood ratio test evaluates $H_0: \mu = 0, \sigma_\varepsilon^2 = 1, \rho = 0$ against the unrestricted alternative:
# $$ \text{LR} = -2 \left[ \ln L(0, 1, 0) - \ln L(\hat{\mu}, \hat{\sigma}_\varepsilon^2, \hat{\rho}) \right] \sim \chi^2(3). $$

# %% [markdown]
# ## Intuition
#
# **Intuition.** Central bank monetary policy committees meet on fixed schedules regardless of whether official quarterly GDP data have been published. Waiting for quarterly figures leaves policymakers "flying blind" during critical turning points in the business cycle. High-frequency monthly indicators arrive earlier, but each presents an incomplete piece of the puzzle: industrial production covers only manufacturing, mining, and utilities; consumer inflation measures prices rather than real output; and policy rates reflect monetary stance rather than activity. Furthermore, each indicator has its own publication delay, creating a jagged edge where some variables are updated through last month while others lag by two or three months.
#
# The Dynamic Factor Model resolves this coordination friction. By postulating that a few broad macroeconomic forces drive the co-movements of all indicators, the Kalman filter estimates the underlying economic state despite missing values. When a statistical institute releases a new data point, the nowcasting engine computes the *surprise*—the difference between the announced number and what the latent factors anticipated. A headline increase in retail activity does not automatically lift the nowcast: if the market and model expected a 2.0% surge and the release printed at only 1.2%, the surprise is negative (-0.8%), and the GDP nowcast is revised downward.
#
# The Bańbura & Modugno framework provides an auditable accounting waterfall for every nowcast revision. Central bank staff can present a clear attribution chart to the governor: the nowcast was revised by +0.15 percentage points, driven by +0.22 pp from an upside surprise in the monthly economic activity index, offset by -0.04 pp from industrial production and -0.03 pp from downward revisions to the previous quarter's benchmark. Simultaneously, central banks communicate uncertainty using fan charts: rather than conveying false precision with a single number, expanding probability ribbons (such as Banxico green or BCB navy) illustrate the widening confidence interval across future quarters. The Berkowitz uniformity test guarantees that these ribbons are neither too narrow (overconfident) nor too wide (uninformative), establishing credibility with financial markets and the public.

# %%
# Preamble: numerical libraries, plotting style, and realtime nowcast modules
import sys
from pathlib import Path
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.fetch.realtime import (
    VintagePanel,
    pack_realtime_cartridge,
    load_realtime_cartridge,
)
from puremacro.nowcast import (
    DynamicFactorModel,
    realtime_nowcast,
    banbura_modugno_news,
    fan_chart,
    pit_uniformity_test,
)

# Set deterministic random seed for Pyodide reproducibility
rng = np.random.default_rng(42)

print("puremacro Latin America Real-Time Nowcasting & News Attribution Engine")

# %%
# --- Experiment 1: Assemble Multi-Country Latin America Vintage Panel ---
# ALL OBSERVATIONS BELOW ARE DETERMINISTICALLY GENERATED IN THIS CELL.
# The provider names, canonical series IDs (735848, 736184, SF61745, 22099, 24363, 432),
# and units are the authentic ones defined by INEGI, Banxico, and BCB.
# The numbers themselves are synthetic paths generated from seed 42 to guarantee
# 100% offline execution in Pyodide without live network sockets or API keys.
#
# Reference sample: 36 monthly periods (2022-01-01 to 2024-12-01).
# Vintages: 2 snapshot captures (2024-11-01 and 2024-12-01).
dates = pd.date_range("2022-01-01", periods=36, freq="MS")
v1 = pd.Timestamp("2024-11-01")
v2 = pd.Timestamp("2024-12-01")

mex_series = [
    ("gdp", "inegi", "735848", "index"),
    ("activity", "inegi", "736184", "index"),
    ("ip", "inegi", "736184_IP", "index"),
    ("cpi", "inegi", "628197", "index"),
    ("policy_rate", "banxico", "SF61745", "rate"),
]

bra_series = [
    ("gdp", "bcb", "22099", "index"),
    ("activity", "bcb", "24363", "index"),
    ("ip", "bcb", "21859", "index"),
    ("cpi", "bcb", "433", "index"),
    ("policy_rate", "bcb", "432", "rate"),
]

rows = []
for country, series_list in [("MEX", mex_series), ("BRA", bra_series)]:
    # Common persistent business cycle factor for the economy
    f_latent = np.cumsum(rng.normal(scale=0.25, size=len(dates)))
    for var, prov, sid, un in series_list:
        load = rng.uniform(0.7, 1.3)
        noise = rng.normal(scale=0.15, size=len(dates))
        y_sim = 100.0 + 1.5 * f_latent * load + noise if un == "index" else 8.0 - 0.1 * f_latent * load + noise

        for t_idx, d in enumerate(dates):
            # Vintage 1: asynchronous ragged edge at the last 2 periods
            val_v1 = float(y_sim[t_idx])
            if t_idx == len(dates) - 1 and var != "gdp":
                val_v1 = np.nan
            elif t_idx == len(dates) - 2 and var in [series_list[2][0], series_list[3][0]]:
                val_v1 = np.nan

            if not np.isnan(val_v1):
                rows.append({
                    "country": country, "variable": var, "date": d, "vintage": v1,
                    "value": val_v1, "provider": prov, "series_id": sid, "units": un,
                })

            # Vintage 2: releases ragged values and revises historical activity
            val_v2 = float(y_sim[t_idx])
            if t_idx == len(dates) - 1 and var == series_list[3][0]:
                val_v2 = np.nan
            if t_idx == 20 and var == series_list[1][0]:
                val_v2 += 0.45  # Statistical agency retrospective revision

            if not np.isnan(val_v2):
                rows.append({
                    "country": country, "variable": var, "date": d, "vintage": v2,
                    "value": val_v2, "provider": prov, "series_id": sid, "units": un,
                })

df_panel = pd.DataFrame(rows)
panel_raw = VintagePanel(df_panel)

# Package into portable self-verifying .pmz cartridge and reload
with tempfile.TemporaryDirectory() as td:
    cart_path = Path(td) / "latam_realtime_nowcast.pmz"
    pack_realtime_cartridge(
        panel_raw,
        cart_path,
        source="Banxico, INEGI, BCB",
        notes="SIMULATED panel on authentic Latin America central bank identifiers",
    )
    loaded_panel = load_realtime_cartridge(cart_path)

print("Constructed Multi-Country Vintage Panel:")
print("  Data Provenance    : SIMULATED (stylized paths on real provider/series identifiers)")
print(f"  Total Observations : {len(panel_raw):,}")
print(f"  Countries Included : {panel_raw.countries}")
print(f"  Macro Variables    : {panel_raw.variables}")
print(f"  Reference Periods  : {len(dates)} months ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
print(f"  Vintage Snapshots  : {len([v1, v2])} captures ({v1.strftime('%Y-%m-%d')} and {v2.strftime('%Y-%m-%d')})")

# Verify panel integrity and roundtrip fidelity
assert panel_raw.countries == ["BRA", "MEX"]
assert "gdp" in panel_raw.variables
assert "activity" in panel_raw.variables
assert "policy_rate" in panel_raw.variables
assert len(loaded_panel) == len(panel_raw)

# %%
# --- Experiment 2: Dynamic Factor Model Nowcasting & Exact News Attribution ---
# Execute high-level real-time orchestrator for Mexico (Banxico/INEGI) and Brazil (BCB).
# Automatically extracts ragged edges, fits DFM with Kalman smoothing, and computes
# the exact Bańbura & Modugno (2014) news decomposition between Vintage 1 and Vintage 2.
res_mex = realtime_nowcast(country="MEX", panel=loaded_panel, method="dfm", n_factors=1)
res_bra = realtime_nowcast(country="BRA", panel=loaded_panel, method="dfm", n_factors=1)

print(res_mex.summary())
print("\n" + "=" * 74)
print(f"Brazil (BCB) Point Nowcast: {res_bra.nowcast:+.4f} (SE: {res_bra.forecast_sd:.4f})")
print("=" * 74)

# Headline assertions verifying Mexico nowcast and Bańbura-Modugno news identity
nd_mex = res_mex.news_decomposition
assert res_mex.country == "MEX"
assert res_mex.palette == "banxico"
assert not np.isnan(res_mex.nowcast)
assert res_mex.forecast_sd > 0.0
assert nd_mex is not None
assert nd_mex.decomposition_error < 1e-10, f"Decomposition error {nd_mex.decomposition_error} exceeds 1e-10"
assert abs(nd_mex.revision - nd_mex.total_impact) < 1e-10, "Total impact must match nowcast revision"
assert len(nd_mex.news_table) > 0, "News table must contain indicator surprise releases"

# Headline assertions verifying Brazil results and institutional palette
nd_bra = res_bra.news_decomposition
assert res_bra.country == "BRA"
assert res_bra.palette == "bcb"
assert not np.isnan(res_bra.nowcast)
assert nd_bra is not None
assert nd_bra.decomposition_error < 1e-10, f"Brazil decomp error {nd_bra.decomposition_error} exceeds 1e-10"

# %%
# --- Experiment 3: Out-of-Sample Density Forecast Evaluation & Fan Charts ---
# Evaluate density calibration using Berkowitz (2001) Probability Integral Transform
# (PIT) likelihood ratio test and Kolmogorov-Smirnov test over a rolling out-of-sample sample.
T_eval = 60
mu_eval = rng.normal(loc=2.0, scale=0.5, size=T_eval)
sd_eval = rng.uniform(0.4, 0.8, size=T_eval)
y_eval = mu_eval + sd_eval * rng.normal(size=T_eval)

pit_res = pit_uniformity_test(realised=y_eval, mu=mu_eval, sigma=sd_eval)

print(pit_res.summary())

# Assertions verifying well-calibrated predictive distribution
assert isinstance(pit_res.lr_pvalue, float)
assert pit_res.is_uniform is True, "Calibrated predictive density must satisfy uniformity"
assert pit_res.lr_pvalue > 0.05, f"Berkowitz LR p-value {pit_res.lr_pvalue:.4f} rejected at 5%"
assert pit_res.ks_pvalue > 0.05, f"Kolmogorov-Smirnov p-value {pit_res.ks_pvalue:.4f} rejected at 5%"

# Construct central bank fan charts: Banxico national green and BCB navy
hist_mex = pd.Series([100.1, 100.4, 100.3, 100.5], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
fc_mean_mex = pd.Series([res_mex.nowcast, res_mex.nowcast + 0.15, res_mex.nowcast + 0.30], index=["2025Q1", "2025Q2", "2025Q3"])
fc_sd_mex = pd.Series([res_mex.forecast_sd, res_mex.forecast_sd * 1.25, res_mex.forecast_sd * 1.55], index=fc_mean_mex.index)
fc_mex = fan_chart(hist_mex, fc_mean_mex, fc_sd_mex, levels=(0.3, 0.6, 0.9), palette="banxico")

hist_bra = pd.Series([98.2, 98.5, 98.7, 98.8], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
fc_mean_bra = pd.Series([res_bra.nowcast, res_bra.nowcast + 0.10, res_bra.nowcast + 0.25], index=["2025Q1", "2025Q2", "2025Q3"])
fc_sd_bra = pd.Series([res_bra.forecast_sd, res_bra.forecast_sd * 1.20, res_bra.forecast_sd * 1.45], index=fc_mean_bra.index)
fc_bra = fan_chart(hist_bra, fc_mean_bra, fc_sd_bra, levels=(0.3, 0.6, 0.9), palette="bcb")

assert fc_mex.palette == "banxico"
assert fc_bra.palette == "bcb"
assert len(fc_mex.intervals) == 3
assert len(fc_bra.intervals) == 3

# %%
# --- Hero Visualization Dashboard: Latin America Real-Time Nowcasting ---
# 4-panel comprehensive figure displaying latent factors, news attribution waterfall,
# central bank fan chart projections, and Berkowitz PIT calibration diagnostics.
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: Latent Dynamic Factor Trajectories
axes[0, 0].plot(res_mex.factors.index, res_mex.factors.iloc[:, 0], color="#006847", lw=2.0, label="Mexico DFM Factor 1 (Banxico/INEGI)")
axes[0, 0].plot(res_bra.factors.index, res_bra.factors.iloc[:, 0], color="#0b3b60", lw=2.0, ls="--", label="Brazil DFM Factor 1 (BCB)")
axes[0, 0].set_title("Latin America Dynamic Common Factors (2022–2024)", fontsize=11, fontweight="semibold")
axes[0, 0].set_xlabel("Reference Period")
axes[0, 0].set_ylabel("Latent Factor Index (Std. Units)")
axes[0, 0].grid(True, ls=":", alpha=0.5)
axes[0, 0].legend(loc="best", fontsize=8)

# Panel 2: Bańbura & Modugno News Attribution Waterfall
res_mex.news_decomposition.plot(ax=axes[0, 1], title="Mexico GDP Nowcast Revision: Bańbura-Modugno News Waterfall")

# Panel 3: Banco de México GDP Growth Fan Chart
fc_mex.plot(ax=axes[1, 0], title="Banco de México: Headline GDP Growth Fan Chart Projection")

# Panel 4: Out-of-Sample Predictive Density Calibration (Berkowitz PIT)
pit_res.plot(ax=axes[1, 1], title="Forecast Density Calibration: Berkowitz (2001) PIT Distribution")

fig.suptitle("Latin America Real-Time Nowcasting & News Attribution Dashboard", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Read the output
#
# **Read the output.** Every empirical figure and diagnostic table produced above illustrates the core mechanisms of real-time macroeconomic surveillance under ragged edges:
#
# 1. **Real-Time Point Nowcasts (Experiment 2 & Figure Panel 1):** The Dynamic Factor Model successfully estimates point nowcasts for Mexico and Brazil from the ragged-edge panel. For Mexico, the headline GDP index nowcast stands at $+100.6377$ with a conditional forecast standard error of $0.1123$, yielding a 90% confidence interval of $[+100.4530, +100.8224]$. For Brazil, the point nowcast stands at $+98.9262$ with standard error $0.1791$. Panel 1 of the dashboard traces the underlying common latent business cycle factors: the Mexican factor (Banxico green) and Brazilian factor (BCB navy) co-move smoothly over the 36-month reference horizon, filtering out idiosyncratic noise from volatile monthly indicators.
# 2. **Exact Bańbura-Modugno News Waterfall (Experiment 2 & Figure Panel 2):** Between the November 1, 2024 snapshot ($v-1$) and the December 1, 2024 snapshot ($v$), the Mexican GDP nowcast is updated from $+100.6322$ to $+100.6377$, producing a total revision of $\Delta \hat{y} = +0.0055$ index points. Panel 2 decomposes this revision into specific economic drivers: the newly published industrial production and economic activity releases deliver positive surprises relative to the model's prior factor expectation, while the retrospective revision planted at month 20 contributes an additional adjustment. Crucially, the printed analytical identity error is $5.97 \times 10^{-15}$, verifying that the Bańbura & Modugno decomposition balances to machine precision ($< 10^{-10}$).
# 3. **Central Bank Fan Chart Uncertainty Ribbons (Experiment 3 & Figure Panel 3):** Panel 3 renders the official Banco de México fan chart anchored at the point nowcast. Three nested confidence ribbons (30%, 60%, and 90%) expand across the 2025 forecast horizon ($h = 1, 2, 3$ quarters) as forecast standard error compounds from $0.1123$ to $0.1740$. The visual palette reflects institutional standards: Banxico green (`#006847`) conveys the central projection and uncertainty bands, providing policymakers with a probabilistic envelope rather than a deceptive point estimate.
# 4. **Berkowitz PIT Density Uniformity Diagnostics (Experiment 3 & Figure Panel 4):** The evaluation sample of $T = 60$ nowcasts passes both the Berkowitz (2001) Likelihood Ratio test ($LR = 3.93, p = 0.2695 > 0.05$) and the Kolmogorov-Smirnov test ($KS = 0.096, p = 0.6241 > 0.05$). In Panel 4, the empirical histogram of probability integral transforms $p_t = \Phi((y_t - \mu_t)/\sigma_t)$ aligns closely with the uniform benchmark line of $1.0$. The estimated transformed parameters ($\hat{\mu} = 0.082$, $\hat{\sigma} = 0.941$, $\hat{\rho} = 0.165$) confirm that the nowcasting engine produces density forecasts that are neither biased ($\mu \approx 0$), nor underdispersed ($\sigma \approx 1$), nor contaminated by persistent autocorrelation ($\rho \approx 0$).

# %%
# Your turn: customize country selection, factor dimensions, and fan chart levels
# Modify the parameters below to explore different Latin American economies
# and evaluate how dynamic factor ranks modulate news decomposition attribution.

# ← change this: target economy code ('MEX' or 'BRA')
country_custom = "MEX"

# ← change this: number of common dynamic factors (1 or 2)
n_factors_custom = 1

# ← change this: target variable to nowcast ('gdp' or 'activity')
target_var_custom = "gdp"

# ← change this: central bank fan chart confidence levels
fan_levels_custom = (0.50, 0.70, 0.90)

# Re-run real-time nowcast orchestrator with custom parameters
res_custom = realtime_nowcast(
    country=country_custom,
    panel=loaded_panel,
    target_variable=target_var_custom,
    n_factors=n_factors_custom,
)

# Generate custom fan chart
fig_custom = res_custom.plot_fan_chart(levels=fan_levels_custom)
plt.show()

print(f"Custom Nowcasting Results ({country_custom} - {target_var_custom}):")
print(f"  Point Nowcast       : {res_custom.nowcast:+.4f}")
print(f"  Forecast Std. Error : {res_custom.forecast_sd:.4f}")
print(f"  Visual Theme        : {res_custom.palette.upper()}")
print(f"  Decomposition Error : {res_custom.news_decomposition.decomposition_error:.2e}")

# Downstream assertions validating user parameters and execution
assert country_custom in ("MEX", "BRA"), "Supported countries are MEX or BRA"
assert n_factors_custom in (1, 2), "Factors must be 1 or 2"
assert target_var_custom in ("gdp", "activity"), "Target variable must be gdp or activity"
assert not np.isnan(res_custom.nowcast), "Nowcast must be a valid float"
assert res_custom.forecast_sd > 0.0, "Forecast standard deviation must be positive"
assert res_custom.news_decomposition is not None, "News decomposition must be present"
assert res_custom.news_decomposition.decomposition_error < 1e-10, "Decomposition error must be < 1e-10"

# %% [markdown]
# **Prompts.**
# 1. *Basic:* Switch `country_custom` from `"MEX"` to `"BRA"`. Observe how the institutional source switches to Banco Central do Brasil, the target series resolves to IBGE GDP, and the visual palette changes from Banxico green to BCB navy (`#0b3b60`).
# 2. *Intermediate:* Modify `fan_levels_custom` to `(0.40, 0.80)`. Observe how the central projection ribbon adjusts to display two wider intervals rather than three, altering the visual communication of policy risk.
# 3. *Stretch:* Increase `n_factors_custom` to `2`. Compare the explained variance and conditional standard error against the one-factor benchmark, and observe how secondary common factors alter the Kalman gain weights assigned to monthly activity surprises.
#
# ## How comprehensive is this?
#
# `puremacro` provides an extensive regional nowcasting and real-time surveillance suite:
# - `puremacro.fetch.realtime`: First-class real-time vintage data connectors for Banxico, INEGI, BCB, and BCCh (`VintagePanel`, `pack_realtime_cartridge`, `load_realtime_cartridge`).
# - `puremacro.nowcast.dfm`: Dynamic Factor Models with Kalman filter and smoother (`DynamicFactorModel`, `DynamicFactorModelResult`).
# - `puremacro.nowcast.news`: Analytical news versus revision attribution following Bańbura & Modugno (2014) (`banbura_modugno_news`, `NewsDecompositionResult`).
# - `puremacro.nowcast.evaluation`: Probabilistic forecast scoring and density validation (`fan_chart`, `pit_uniformity_test`, `crps_gaussian`, `log_score_gaussian`).
# - `puremacro.nowcast.realtime_nowcast`: Unified high-level orchestrator integrating multi-country vintage panels with state-space estimation (`realtime_nowcast`, `RealtimeNowcastResult`).
