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
# **How can central bank communications, press conference dialogues, and public narrative discussions be systematically transformed into structural macroeconomic shocks and identified policy impulse responses?**
#
# Qualitative textual records contain rich, high-dimensional, forward-looking information about policy intentions, inflation perceptions, supply bottleneck developments, and emerging banking fragilities. However, utilizing textual sources in macroeconometrics requires overcoming two distinct challenges:
# 1. **Structured Discourse Extraction**: In central bank press conferences, communications are not monolithic. Prepared opening statements reflect the formal institutional consensus of the monetary policy committee, whereas the spontaneous Question-and-Answer (Q&A) session reveals the Chair's candid assessments under journalist scrutiny.
# 2. **Structural Identification via Narrative Constraints**: Traditional sign restrictions identify shocks by restricting the contemporaneous response of endogenous variables ($Y_{t} = B_0 \varepsilon_t$). However, sign restrictions often produce overly wide identification sets. As demonstrated by Juan Antolín-Díaz and Juan Rubio-Ramírez (2018, *AER*) and Sydney Ludvigson, Sai Ma, and Serena Ng (2021, *JME*), conditioning structural identification on historical narrative episodes (e.g. Volcker's 1979 tightening or the 2023 banking stress) drastically narrows credible sets and eliminates price puzzles.
#
# In this interactive showcase, we walk through the complete **puremacro narrative econometrics pipeline**:
# 1. Parsing multi-speaker central bank transcripts by speaker role (Fed, ECB, Banxico).
# 2. Estimating a 100% pure-Python/NumPy Dynamic Topic Model via Non-Negative Matrix Factorization (NMF).
# 3. Detecting statistical narrative burst anomalies ($z$-scores) prior to macro data releases.
# 4. Estimating a Bayesian Narrative SVAR with Ludvigson–Ma–Ng shock magnitude bounds.

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
# Central bank press conferences exhibit a systematic informational division of labor:
# - **Prepared Opening Remarks**: Carefully drafted and negotiated among committee members; serves as the official monetary policy announcement.
# - **Journalist Q&A Session**: Spontaneous dialogue where reporters probe downside risks, policy conditionalities, and financial stress, often extracting forward-looking signals not present in the prepared text.
#
# `puremacro.narrative.sources.parse_transcript` parses raw transcripts using regular expressions and speaker identification heuristics, segmenting the dialogue into structured turns and extracting speaker-specific corpuses.

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
# Rather than relying on heavy external NLP frameworks (such as Gensim or PyTorch), `puremacro.narrative.DynamicTopicModel` is written in 100% pure Python and NumPy, running deterministically inside standard Python and Pyodide environments.
#
# Given a document-term frequency matrix $X \in \mathbb{R}_{+}^{D \times V}$ over $D$ documents and vocabulary size $V$, Non-Negative Matrix Factorization (NMF) decomposes $X$ into non-negative factor matrices:
#
# $$ \min_{W \ge 0, H \ge 0} \frac{1}{2} \| X - W H \|_F^2 = \frac{1}{2} \sum_{d=1}^D \sum_{v=1}^V \left( X_{d, v} - [W H]_{d, v} \right)^2 $$
#
# where $W \in \mathbb{R}_{+}^{D \times K}$ denotes document topic weights, $H \in \mathbb{R}_{+}^{K \times V}$ denotes topic-term distributions, and updates follow Lee & Seung's (2001) multiplicative rules:
#
# $$ H \leftarrow H \odot \frac{W^\top X}{W^\top W H + \varepsilon}, \qquad W \leftarrow W \odot \frac{X H^\top}{W H H^\top + \varepsilon} $$
#
# Slicing and normalizing $W$ by observation date produces the dynamic evolution of macroeconomic topic shares $\theta_t \in \Delta^{K-1}$ over time.

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

fig, ax = _nbstyle.figura(figsize=(9.0, 4.4))
dtm_res.topic_shares.plot(ax=ax, color=_nbstyle.palette(3), lw=2)
ax.set_title("Evolution of Latent Macroeconomic Themes (NMF Topic Shares)", fontsize=11, fontweight="bold")
ax.set_xlabel("Date", color=_nbstyle.TEXTO)
ax.set_ylabel("Monthly Topic Share", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 3. Narrative Burst Anomaly Detection
#
# Nascent macroeconomic shocks—such as sudden supply chain disruptions or banking liquidity panics—frequently emerge in textual communications before manifesting in quarterly national account releases.
#
# `detect_narrative_bursts` identifies these episodes using a rolling baseline statistical filter. For each candidate term $w$, let $f_{w, t}$ be its frequency at period $t$, and let $\mu_{w, t}$ and $\sigma_{w, t}$ be the rolling sample mean and standard deviation over a historical window of length $W$:
#
# $$ Z_{w, t} = \frac{f_{w, t} - \mu_{w, t}}{\sigma_{w, t} + \epsilon}, \qquad \text{Burst Magnitude} = \frac{f_{w, t}}{\mu_{w, t} + \epsilon} $$
#
# An anomalous narrative burst is triggered when $Z_{w, t} \ge Z_{\text{crit}}$ and the absolute count satisfies $f_{w, t} \ge \text{min\_count}$.
#
# In the output below, the algorithm pinpoints the banking liquidity shock in March 2023, detecting dramatic surges in terms like *bank*, *run*, *pressures*, and *liquidity*.

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
# Consider a structural vector autoregression $Y_t = \sum_{l=1}^p A_l Y_{t-l} + u_t$, where reduced-form residuals $u_t$ are related to orthonormal structural shocks $\varepsilon_t$ by $u_t = B_0 \varepsilon_t = P Q \varepsilon_t$, with $P$ being the lower Cholesky factor of $\Sigma_u$ and $Q \in \mathcal{O}(n)$ an orthogonal rotation matrix ($Q Q^\top = I$).
#
# Traditional sign restrictions impose sign conditions on the impulse responses: $\mathcal{S}_{h} = \text{sign}\left( C_h B_0 \right) \odot S \ge 0$.
#
# Following Antolín-Díaz & Rubio-Ramírez (2018) and Ludvigson, Ma & Ng (2021), narrative restrictions condition the draw of $Q$ directly on historical dates $t^*$:
# 1. **Shock Sign Restriction**: $\text{sign}(\varepsilon_{i, t^*}) = s_{i, t^*}$.
# 2. **Shock Magnitude Bound**: $|\varepsilon_{i, t^*}| \ge \underline{c}$.
# 3. **Dominant Contribution**: The identified shock explains the majority of the historical residual in target variable $j$: $|\varepsilon_{i, t^*} B_{0, j, i}| > \sum_{k \neq i} |\varepsilon_{k, t^*} B_{0, j, k}|$.
#
# Setting `bayes_draws=True` computes the full posterior distribution over $(A, \Sigma_u, Q)$ using conjugate Normal-Inverse-Wishart sampling with Haar-measure rotation candidates.

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

fig, ax = _nbstyle.figura(figsize=(8.5, 4.4))
h = np.arange(svar_res.irf_median.shape[0])
ax.plot(h, svar_res.irf_median[:, 1, 0], **_nbstyle.S1, label="Bayesian Median IRF")
ax.fill_between(h, svar_res.irf_lower[:, 1, 0], svar_res.irf_upper[:, 1, 0], color=_nbstyle.TINTA, alpha=0.15, label="90% Posterior Credible Band")
ax.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax.set_title("Response of Inflation to a Contractionary Policy Shock (Ludvigson-Ma-Ng Bound)", fontsize=11, fontweight="bold")
ax.set_xlabel("Horizon (Months)", color=_nbstyle.TEXTO)
ax.set_ylabel("Impulse Response", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
