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
# # Central Bank Speech Sentiment & Narrative Monetary Policy Transmission
#
# **How do central bank communications, press conference tones, and narrative policy surprises transmit into money markets, interest rates, and inflation?**
#
# Modern central banking relies heavily on communication:
# 1. **Hawkish / Dovish Sentiment Extraction**: Quantifying the balance of hawkish (inflation-fighting) vs. dovish (growth-supporting) keywords in official statements using the Apel-Blix-Grimaldi (2014) and Picault-Renault (2017) methodologies:
#    $$ \text{Tone}_t = \frac{\text{Hawk}_t - \text{Dove}_t}{\text{Hawk}_t + \text{Dove}_t + \epsilon} $$
# 2. **Narrative Monetary Surprises**: Using qualitative directional shifts and policy stance classifications as identified exogenous shocks.
# 3. **High-Frequency Local Projections**: Tracing the dynamic response of interbank interest rates and headline inflation using Jordà (2005) local projections with Newey-West HAC standard errors (`puremacro.lp.lp_hac`).

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

from puremacro.datasets import load_banxico_stance
from puremacro.lp import lp_hac
from puremacro.narrative.indices import tone

# %% [markdown]
# ## 1. Extracting Hawkish-Dovish Tone from Policy Statements

# %%
corpus = [
    ("2021-06-15", "The Committee decided to maintain the target range for the federal funds rate at 0 to 1/4 percent. Progress on vaccinations has reduced the spread of COVID-19, but inflation has risen, largely reflecting transitory factors.", {}),
    ("2021-11-03", "Inflation is elevated, largely reflecting factors that are expected to be transitory. Supply bottlenecks and price pressures have broadened across sectors.", {}),
    ("2022-03-16", "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. In support of these goals, the Committee decided to raise the target range for the federal funds rate.", {}),
    ("2022-06-15", "The Committee is strongly committed to returning inflation to its 2 percent objective. Decided to raise interest rates by 75 basis points to curb persistent inflationary pressures and overheating labor markets.", {}),
    ("2022-09-21", "Recent indicators point to modest growth in spending and production. Price stability is the responsibility of the Federal Reserve and serves as the bedrock of our economy.", {}),
    ("2022-12-14", "The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive.", {}),
    ("2023-05-03", "Tighter credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. The extent of these effects remains uncertain.", {}),
    ("2023-12-13", "Inflation has eased over the past year but remains elevated. Economic growth has slowed from its strong pace in the third quarter.", {}),
    ("2024-06-12", "Inflation has eased substantially over the past year, but remains above our 2 percent longer-run goal. Modest further progress toward the Committee's 2 percent inflation objective has occurred.", {}),
]

tone_res = tone(
    corpus,
    country="US",
    language="en",
    method="apel_blix_grimaldi",
    normalize="raw",
)
print("Apel-Blix-Grimaldi Tone Series Preview:")
print(tone_res.series.dropna())

# %% [markdown]
# ## 2. Empirical Narrative Stance & Macroeconomic Panel Data

# %%
df_banxico = load_banxico_stance()

data_dir = Path(_cwd / "course" / "data" if (_cwd / "course" / "data").exists() else _cwd / "notebooks" / "course" / "data")
df_rate = pd.read_csv(data_dir / "IR3TIB01MXM156N.csv")
df_rate["date"] = pd.to_datetime(df_rate.iloc[:, 0])
df_rate = df_rate.set_index("date")
df_rate.index = df_rate.index.to_period("M")
df_rate["rate_3m"] = pd.to_numeric(df_rate.iloc[:, 1], errors="coerce")

df_cpi = pd.read_csv(data_dir / "CPALTT01MXM659N.csv")
df_cpi["date"] = pd.to_datetime(df_cpi.iloc[:, 0])
df_cpi = df_cpi.set_index("date")
df_cpi.index = df_cpi.index.to_period("M")
df_cpi["inflation_yoy"] = pd.to_numeric(df_cpi.iloc[:, 1], errors="coerce")

df_lp = pd.concat([df_banxico["banxico_direction"], df_rate["rate_3m"], df_cpi["inflation_yoy"]], axis=1).dropna()
df_lp["narrative_shock"] = df_lp["banxico_direction"].to_numpy(dtype=float)
df_lp = df_lp.reset_index(drop=True)
print("Aligned Monthly Panel Head:")
print(df_lp.head())

# %% [markdown]
# ## 3. Estimating Jordà (2005) Local Projections
#
# We estimate the impulse response function (IRF) of the 3-month interbank interest rate and headline inflation to a +1 standard deviation narrative monetary policy shock:
# $$ y_{t+h} - y_{t-1} = \alpha_h + \beta_h \text{Shock}_t + \sum_{l=1}^p \gamma_l Z_{t-l} + \varepsilon_{t+h} $$

# %%
irf_rate = lp_hac(
    df=df_lp,
    y="rate_3m",
    x="narrative_shock",
    horizons=range(0, 19),
    n_lags=2,
    controls=["inflation_yoy"],
    alpha=0.10,
)

irf_cpi = lp_hac(
    df=df_lp,
    y="inflation_yoy",
    x="narrative_shock",
    horizons=range(0, 19),
    n_lags=2,
    controls=["rate_3m"],
    alpha=0.10,
)

print("Interest Rate LP Response:")
print(irf_rate.head(8))

# %% [markdown]
# ## 4. Macroeconomic Impulse Responses with 90% HAC Bands

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Rate Response
ax1.plot(irf_rate["h"], irf_rate["beta"], color="#1f77b4", lw=2, label="3M Interbank Rate IRF")
ax1.fill_between(irf_rate["h"], irf_rate["lo"], irf_rate["hi"], color="#1f77b4", alpha=0.2, label="90% HAC Band")
ax1.axhline(0, color="black", lw=0.8, linestyle="--")
ax1.set_title("Interest Rate Response to Narrative Monetary Hike", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizon (Months)")
ax1.set_ylabel("Interest Rate (% pts)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# Inflation Response
ax2.plot(irf_cpi["h"], irf_cpi["beta"], color="#d62728", lw=2, label="Inflation Response IRF")
ax2.fill_between(irf_cpi["h"], irf_cpi["lo"], irf_cpi["hi"], color="#d62728", alpha=0.2, label="90% HAC Band")
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Inflation Response to Monetary Tightening", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizon (Months)")
ax2.set_ylabel("Inflation YoY (% pts)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
