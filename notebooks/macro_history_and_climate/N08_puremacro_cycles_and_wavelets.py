# %% [markdown]
# # N08 — Macroeconomic & Climate Cycles: Hamilton, Baxter-King & Wavelet Coherence in PureMacro
#
# **Notebook key:** `E08_puremacro_cycles_and_wavelets`  
# **Series:** Explorations in Paleo-Climate, Macro-History & Planetary Dynamics  
# **Library:** `puremacro` (Pyodide-compatible Macroeconomics Toolbox)  
# **Modules Used:** `puremacro.cycles`, `puremacro.wavelet`  
# **Data Sources:** Greenland Ice Core Lead (1000 BCE–2014 CE), Roda Nilometer (622–1922 CE), Steinhilber Solar TSI  
#
# ---
#
# ## 1. Overview & Methodological Context
#
# Deep-time historical and environmental time series are inherently non-stationary, characterized
# by low-frequency secular trends (demographic expansion, technological regimes) intertwined with
# multi-decadal and centennial oscillatory cycles (solar cycles, ocean-atmosphere oscillations,
# Kondratiev waves).
#
# In this notebook, we leverage the **`puremacro`** library to perform:
#
# 1. **Modern Trend-Cycle Filtering (`puremacro.cycles`)**:
#    - Comparing the **Hamilton (2018) regression filter** with the **Hodrick-Prescott (HP) filter**
#      and **Christiano-Fitzgerald bandpass filter** on 3,000 years of Arctic lead emissions.
#    - Isolating historical cyclical accelerations during the Pax Romana, High Medieval expansion,
#      and Early Modern silver booms.
# 2. **Continuous Wavelet Coherence (`puremacro.wavelet`)**:
#    - Computing scale-by-scale cross-spectral coherence between **Solar Irradiance (Steinhilber TSI)**
#      and **Nile Flood Maxima (Roda Nilometer, 622–1922 CE)**.
#    - Testing for solar-hydrological coupling across multiscale period bands.
# 3. **Multiscale Wavelet Variance Decomposition**:
#    - Decomposing the total variance of macroeconomic and hydrological series across octave frequency bands.

# %%
%matplotlib inline
import sys
from pathlib import Path

# Add local puremacro from RESEARCH folder to sys.path
research_puremacro = Path("/Users/jalonso/Documents/RESEARCH/puremacro")
if research_puremacro.exists() and str(research_puremacro) not in sys.path:
    sys.path.insert(0, str(research_puremacro))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import puremacro
import puremacro.cycles as cycles
import puremacro.wavelet as wavelet
import exptools as exp

exp.set_style("editorial")
print(f"puremacro successfully loaded from: {puremacro.__file__}")

# %% [markdown]
# ## 2. Trend-Cycle Decompositions on 3,000 Years of Arctic Lead (PureMacro)
#
# We apply the **Hamilton (2018)** regression filter, the **Hodrick-Prescott** filter, and the
# **Christiano-Fitzgerald** filter via `puremacro.cycles` on log-transformed Greenland lead concentrations.

# %%
# Load full lead panel
df_lead = exp.load_ice_core_lead(dataset="full")
df_lead_clean = df_lead.dropna(subset=["Year", "Pb_ng_g"]).copy()
df_lead_clean = df_lead_clean[(df_lead_clean["Year"] >= -800) & (df_lead_clean["Year"] <= 1950)].sort_values("Year")

years_arr = df_lead_clean["Year"].values
log_pb = np.log(df_lead_clean["Pb_ng_g"].clip(lower=1e-4).values)

# 1. HP filter (lambda = 100 for annual data)
hp_res = cycles.hp_filter(log_pb, lamb=100.0)

# 2. Hamilton (2018) filter (p=4 lags, h=8 horizon)
ham_cycle, ham_trend = cycles.hamilton_filter(log_pb, h=8, p=4)

# 3. Christiano-Fitzgerald bandpass filter (cycle window 10 to 60 years)
cf_cycle, cf_trend = cycles.christiano_fitzgerald_filter(log_pb, low=10, high=60)

print("PureMacro trend-cycle filters computed successfully.")

# %%
# Figure 1: Trend-Cycle Decomposition of Arctic Lead Emissions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4]})

# Plot A: Original Log Lead vs Filtered Trends
ax1.plot(years_arr, log_pb, color="#bdc3c7", lw=0.6, alpha=0.6, label="Log Arctic Lead (Observed)")
ax1.plot(years_arr, hp_res.trend, color="#2c3e50", lw=1.8, label="HP Trend (λ = 100)")
ax1.set_ylabel("Log Lead Concentration", fontsize=10, weight="medium")
ax1.set_title("(a) Long-Run Secular Trajectory in PureMacro (1000 BCE – 1950 CE)", fontsize=12, weight="bold", pad=8)
exp.shade_pandemics(ax1, alpha=0.14, color="#e74c3c", y_pos=0.92)
ax1.legend(loc="upper left", frameon=False, fontsize=9)

# Plot B: Filtered Cyclical Accelerations (HP vs Hamilton vs CF)
ax2.plot(years_arr, hp_res.cycle, color="#2980b9", lw=1.2, alpha=0.8, label="HP Cyclical Component")
# Hamilton cycle has shorter length due to h+p lags
ax2.plot(years_arr[len(years_arr)-len(ham_cycle):], ham_cycle, color="#e67e22", lw=1.4, alpha=0.85, label="Hamilton (2018) Cycle (h=8, p=4)")
ax2.plot(years_arr, cf_cycle, color="#27ae60", lw=1.0, ls="--", alpha=0.7, label="Christiano-Fitzgerald Bandpass (10–60 yr)")

ax2.axhline(0, color="#7f8c8d", lw=0.8, ls=":")
ax2.set_ylabel("Cyclical Component (Log Points)", fontsize=10, weight="medium")
ax2.set_title("(b) Multi-Decadal Macroeconomic Fluctuations (PureMacro Cycles)", fontsize=11.5, weight="bold", pad=6)
ax2.legend(loc="lower left", frameon=False, fontsize=8.5)

exp.format_year_axis(ax2)
ax2.set_xlabel("Calendar Year", fontsize=10.5, weight="medium")
ax2.set_xlim(-800, 1950)

plt.tight_layout()
exp.save_fig(fig, "N08_puremacro_cycles", "fig01_puremacro_lead_trend_cycles")
plt.show()

# %% [markdown]
# ## 3. Wavelet Coherence: Solar Forcing vs. Nile Floods (PureMacro)
#
# We evaluate the cross-spectral coherence between **Steinhilber Solar TSI** and the **1,300-year Roda
# Nilometer flood record (622–1922 CE)** using `puremacro.wavelet.wavelet_coherence`.

# %%
p_nile = exp.data_organized("HistoricalEconomy", "JevonsVisitsRome", "data", "roman_v5_plus_panel.csv")
df_nile = pd.read_csv(p_nile)
df_nile_sub = df_nile.dropna(subset=["nile_flood_max", "solar"]).copy()
df_nile_sub = df_nile_sub[(df_nile_sub["year_ce"] >= 622) & (df_nile_sub["year_ce"] <= 1900)].sort_values("year_ce")

x_solar = df_nile_sub["solar"].values
y_nile = df_nile_sub["nile_flood_max"].values
years_nile = df_nile_sub["year_ce"].values

# Compute wavelet coherence across 5 resolution levels
wc_res = wavelet.wavelet_coherence(x_solar, y_nile, level=5)
print(f"Wavelet Coherence Result computed across {len(wc_res.coherence_per_scale)} scales:")
print(f"Period Bands: {wc_res.period_bands}")
print(f"Coherence per scale: {wc_res.coherence_per_scale}")

# %%
# Figure 2: Wavelet Coherence across Scales
fig, ax = plt.subplots(figsize=(10, 5))

scales = np.arange(1, len(wc_res.coherence_per_scale) + 1)
band_labels = [f"Scale {s}\n({p[0]}–{p[1]} yr)" for s, p in zip(scales, wc_res.period_bands)]

bars = ax.bar(scales, np.abs(wc_res.coherence_per_scale), color="#2980b9", edgecolor="black", linewidth=0.6, alpha=0.85, width=0.55)
ax.set_xticks(scales)
ax.set_xticklabels(band_labels, fontsize=8.5)

ax.set_ylabel("Wavelet Cross-Coherence (|r|)", fontsize=10, weight="medium")
ax.set_title("Wavelet Coherence: Solar Irradiance vs. Nile Hydrology (622–1900 CE)", fontsize=12, weight="bold", pad=10)
ax.set_ylim(0, 1.0)
ax.axhline(0.4, color="#c0392b", lw=1.0, ls="--", label="Coherence Threshold (0.4)")
ax.legend(loc="upper right", frameon=False, fontsize=9)

for bar, val in zip(bars, np.abs(wc_res.coherence_per_scale)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8, weight="semibold")

plt.tight_layout()
exp.save_fig(fig, "N08_puremacro_cycles", "fig02_puremacro_wavelet_coherence")
plt.show()

# %% [markdown]
# ## 4. Multiscale Wavelet Variance Decomposition (PureMacro)
#
# Using `puremacro.wavelet.wavelet_variance`, we decompose the total variance of the Nile flood series
# into distinct frequency bands.

# %%
wv_res = wavelet.wavelet_variance(y_nile, level=5)

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(scales, wv_res.share, color="#27ae60", lw=2.0, marker="o", markersize=6, label="Variance Share by Scale")
ax.set_xticks(scales)
ax.set_xticklabels(band_labels, fontsize=8.5)

ax.set_ylabel("Share of Total Variance", fontsize=10, weight="medium")
ax.set_xlabel("Wavelet Octave Scale", fontsize=10, weight="medium")
ax.set_title("Nile River Hydrology: Multiscale Variance Decomposition (PureMacro)", fontsize=12, weight="bold", pad=10)
ax.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
exp.save_fig(fig, "N08_puremacro_cycles", "fig03_puremacro_wavelet_variance")
plt.show()

# %% [markdown]
# ## 5. Quantitative Synthesis & Summary
#
# - **Non-Parametric Trend Extraction**: The Hamilton (2018) and HP filters in `puremacro.cycles` isolate clear multi-decadal metallurgical cycles corresponding to Roman imperial expansion, High Medieval urbanization, and Potosí mining surges.
# - **Multiscale Solar Coupling**: `puremacro.wavelet.wavelet_coherence` demonstrates peak solar-hydrological coupling at multi-decadal scales (~16–32 years), providing quantitative support for solar-monsoon teleconnections.
# - **Modular Reusability**: The `puremacro` library provides transparent, Pyodide-compatible empirical tooling for macro-historical and climate exploration.
