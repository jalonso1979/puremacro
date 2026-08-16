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
# # Narrative Macroeconomics — Multi-Speaker Transcripts, Dynamic Topics, and Bayesian SVARs
#
# **How can central bank communications and public discussions be systematically transformed into structural macroeconomic shocks?**
#
# Textual data contains forward-looking information about policy intentions, inflation perceptions, and emerging financial risks.
# In this interactive showcase, we walk through the complete **puremacro narrative econometrics pipeline**:
# 1. **Multi-Speaker Transcript Parsing**: Separating prepared policy guidance from spontaneous journalist questioning (FOMC, ECB, Banxico).
# 2. **Pure-Python Dynamic Topic Modeling**: Tracking the evolution of macroeconomic themes over time using Non-Negative Matrix Factorization (NMF).
# 3. **Narrative Burst Anomaly Detection**: Identifying statistical keyword surges ($z$-scores) before official macro data releases.
# 4. **Bayesian Narrative SVAR Identification**: Constraining monetary impulse responses using Ludvigson–Ma–Ng (2021) shock magnitude bounds and conjugate Normal-Inverse-Wishart posterior sampling.

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

from puremacro.narrative import (
    DynamicTopicModel,
    detect_narrative_bursts,
    score_spanish_macro_sentiment,
)
from puremacro.narrative.sources import parse_transcript
from puremacro.var.identify import NarrativeRestriction, narrative_sign_svar

# %% [markdown]
# ## 1. Multi-Speaker Transcript Parsing & Speaker Asymmetry
#
# In central bank press conferences, the **prepared opening remarks** reflect a committee-vetted consensus, while **Q&A exchanges** reveal the Chair's candid assessment under questioning.
# `puremacro.narrative.sources.parse_transcript` splits dialogue turns by speaker role and section.

# %%
raw_transcript = """
FEDERAL RESERVE PRESS CONFERENCE
WASHINGTON, D.C.

CHAIR POWELL: Good afternoon. My colleagues and I remain firmly committed to bringing inflation back down to our 2 percent goal. We decided today to raise the target range for the federal funds rate by 25 basis points and continue reducing our securities holdings. The labor market remains extremely tight, with wage growth elevated.

QUESTION AND ANSWER PERIOD

REPORTER NICK: Nick Timiraos, Wall Street Journal. Chair Powell, how do recent banking sector stresses affect your trajectory for additional rate hikes?

CHAIR POWELL: Thanks Nick. We will closely monitor incoming credit conditions. Inflation pressures remain persistent, and restoring price stability is essential.

REPORTER STEVE: Steve Liesman, CNBC. Do you see a path to a soft landing without significant unemployment spikes?

CHAIR POWELL: A soft landing remains plausible, but softening labor market conditions and anchored inflation expectations will be critical.
"""

doc = parse_transcript(raw_transcript, title="FOMC Press Conference", institution="FED")
print(f"Parsed {len(doc.turns)} dialogue turns across {doc.institution} press conference.")
print(f"Prepared opening remarks word count: {len(doc.opening_statement().split())}")
print(f"Chair Q&A spoken text word count:    {len(doc.chair_qa_text().split())}")
print(f"Press questions spoken word count:   {len(doc.press_questions_text().split())}")

# %% [markdown]
# ## 2. Pure-Python Dynamic Topic Modeling (NMF)
#
# Unlike heavy external machine learning libraries, `puremacro.narrative.DynamicTopicModel` is 100% pure Python and NumPy, running seamlessly inside Pyodide in the browser.
# It factors the document-term matrix $X \approx W H$ using multiplicative updates with Frobenius loss.

# %%
rng = np.random.default_rng(42)
dates = pd.date_range("2022-01-01", periods=24, freq="MS")

corpus_templates = [
    "Inflation surge driven by energy prices and supply chain bottlenecks.",
    "Central bank hikes policy rate to anchor long term inflation expectations.",
    "Labor market conditions remain tight with low unemployment and wage pressure.",
    "Financial market stress and credit standards tightening across regional banks.",
    "Economic growth slowing down as restrictive monetary policy cools demand.",
    "Consumer spending resilient but household credit balances expanding rapidly.",
]

dated_corpus = []
for d in dates:
    for _ in range(4):
        idx = rng.choice(len(corpus_templates))
        text = corpus_templates[idx]
        if pd.Timestamp("2023-03-01") <= d <= pd.Timestamp("2023-06-01"):
            text += " Bank run liquidity pressures and emergency discount window borrowing."
        dated_corpus.append((d, text))

texts_only = [t[1] for t in dated_corpus]
dates_only = [t[0] for t in dated_corpus]

dtm = DynamicTopicModel(n_topics=3, random_state=42)
dtm_res = dtm.fit_transform_corpus(texts_only, dates_only, freq="MS")

fig, ax = plt.subplots(figsize=(10, 4.5))
dtm_res.topic_shares.plot(ax=ax, lw=2)
ax.set_title("Evolution of Latent Macroeconomic Themes (NMF Topic Shares)", fontsize=12, fontweight="bold")
ax.set_ylabel("Monthly Topic Share")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Narrative Burst Anomaly Detection
#
# Nascent economic shocks (such as supply chain bottlenecks or banking runs) appear in narrative streams before registering in quarterly national accounts.
# `detect_narrative_bursts` computes the rolling baseline mean and standard deviation of term frequencies, flagging terms with $z$-score surges above threshold.

# %%
target_date = "2023-03-01"
bursts = detect_narrative_bursts(
    dated_corpus,
    target_date=target_date,
    window_periods=12,
    freq="MS",
    min_count=2,
)

print(f"Top narrative bursts detected for {target_date}:")
for b in bursts[:5]:
    print(f"  • {b.term:15s} | Z-Score: {b.z_score:+6.2f} | Burst Magnitude: {b.burst_magnitude:5.1f}x")

# %% [markdown]
# ## 4. Bayesian Narrative SVAR with Ludvigson–Ma–Ng Shock Bounds
#
# Following Ludvigson, Ma & Ng (2021, *JME*) and Antolín-Díaz & Rubio-Ramírez (2018, *AER*), we identify monetary policy shocks by constraining both the sign and magnitude of structural shocks on critical historical dates:
# $$ |\varepsilon_{\text{monetary}, t^*}| \ge \underline{c} $$
# Setting `bayes_draws=True` samples full reduced-form VAR posterior parameters via the Normal-Inverse-Wishart Bartlett decomposition.

# %%
# Synthetic 2-variable macro VAR (Interest Rate, Inflation)
T = 120
e = rng.standard_normal((T, 2))
Y = np.zeros((T, 2))
for t in range(1, T):
    Y[t, 0] = 0.7 * Y[t-1, 0] + 0.1 * Y[t-1, 1] + e[t, 0]
    Y[t, 1] = 0.2 * Y[t-1, 0] + 0.6 * Y[t-1, 1] + 0.5 * e[t, 0] + e[t, 1]

# Restriction: On date 21, the contractionary shock was positive and >= 0.5 standard deviations
restr = [
    NarrativeRestriction(
        kind="shock_bound",
        date=21,
        shock=0,
        min_magnitude=0.5,
        sign=+1,
    )
]
sign_matrix = {0: np.array([[1, 0], [1, 1]])}

svar_res = narrative_sign_svar(
    Y,
    p=1,
    horizon=10,
    sign_matrix=sign_matrix,
    restrictions=restr,
    bayes_draws=True,
    n_draws=500,
    seed=42,
)

fig, ax = plt.subplots(figsize=(8, 4.5))
h = np.arange(svar_res.irf_median.shape[0])
ax.plot(h, svar_res.irf_median[:, 1, 0], color="#1f77b4", lw=2, label="Bayesian Median IRF")
ax.fill_between(h, svar_res.irf_lower[:, 1, 0], svar_res.irf_upper[:, 1, 0], color="#1f77b4", alpha=0.25, label="90% Posterior Credible Band")
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Response of Inflation to a Contractionary Policy Shock (Ludvigson-Ma-Ng Bound)", fontsize=11, fontweight="bold")
ax.set_xlabel("Horizon (Months)")
ax.set_ylabel("Impulse Response")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
