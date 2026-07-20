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
# # Build an uncertainty index from text — no API key, no LLM
#
# Measuring economic-policy uncertainty usually means a Bloomberg terminal or a
# paid news-sentiment feed. `puremacro.narrative` builds research-grade
# **text-as-data uncertainty indices** from any corpus you supply, using only
# pure-numpy dictionary scoring — no network, no API key, no language model. Here
# we plant a known uncertainty shock in a synthetic news corpus and recover it
# with the Baker-Bloom-Davis **EPU** and the monetary-policy **MPU** indices.

# %% [markdown]
# ## The index in one equation
#
# Baker-Bloom-Davis EPU flags a document $d$ as *uncertain* only when it hits all
# **three** term groups at once:
#
# $$ \mathrm{flag}(d) = \mathbb{1}\!\left[(\exists\,t\in E:\,t\in d)\,\wedge\,(\exists\,t\in P:\,t\in d)\,\wedge\,(\exists\,t\in U:\,t\in d)\right], $$
#
# where $E,P,U$ are the **economy**, **policy**, and **uncertainty** lexicons. The raw
# index in period $\tau$ is the share of flagged documents,
# $\mathrm{EPU}^{\mathrm{raw}}_\tau = \frac{1}{N_\tau}\sum_{d\in\tau}\mathrm{flag}(d)$,
# and `normalize="bbd_100"` rescales it to the published units (sample mean 100, sd 50):
# $\mathrm{EPU}_\tau = 100 + 50\cdot(\mathrm{EPU}^{\mathrm{raw}}_\tau - \overline{\mathrm{EPU}^{\mathrm{raw}}})/\mathrm{sd}(\mathrm{EPU}^{\mathrm{raw}})$.
#
# **Intuition.** The *co-occurrence* of all three groups is what makes the index specific.
# A piece about "economic growth" (no policy, no uncertainty) or a sports story that
# happens to say "uncertain" never fires; only documents simultaneously about the economy,
# about policy, and about uncertainty count. MPU drops the co-occurrence requirement and
# just counts monetary-policy keywords, then z-scores — looser, but enough when the
# vocabulary is already narrow.

# %% [markdown]
# ## Setup — imports and style
#
# Everything below uses only the **pure-numpy dictionary path** inside
# `puremacro.narrative.indices` — no embedding model, no LLM, no API key.
# The full module imports cleanly in Pyodide/browser environments because
# the dictionary-scoring kernels depend only on `re`, `numpy`, and `pandas`.

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

# Dictionary-scoring indices (EPU, MPU) and their lexicons.
from puremacro.narrative.indices import epu, mpu, LEXICONS
from puremacro.narrative.indices._kernels import (
    cooccurrence_kernel,
    keyword_count_kernel,
)

# Pyodide-clean guard: none of these heavy/paid modules should have leaked.
_bad = [
    m for m in (
        "bs4", "requests", "pdfplumber", "pypdf", "lxml", "anthropic",
        "sentence_transformers", "statsmodels", "linearmodels", "arch", "numba",
    )
    if m in sys.modules
]
assert _bad == [], f"Unexpected module leak: {_bad}"

# %% [markdown]
# ## 1. Build a fully synthetic corpus with a controlled uncertainty shock
#
# We create **288 mock "news" documents** (4 per month, Jan 2018 – Dec 2023)
# using plain string literals — no files, no network, no LLM.
#
# The key design decision: we **inject** EPU and MPU signal phrases during
# 2020Q1–Q4 with high probability (0.75 / 0.70) and keep them rare outside
# that window (0.08 / 0.06). This lets us verify the index responds to exactly
# the signal we planted.
#
# ### EPU phrase anatomy
# Baker-Bloom-Davis EPU counts documents that contain **at least one term from
# each of three groups**: Economy (`economic`, `economy`, …), Policy (`policy`,
# `regulation`, `federal reserve`, …), Uncertainty (`uncertain`, `uncertainty`,
# …). Our injected high-uncertainty phrases are crafted to hit all three groups.

# %%
RNG = np.random.default_rng(42)

HIGH_START = pd.Timestamp("2020-01-01")   # injected shock window
HIGH_END   = pd.Timestamp("2020-12-31")

# ------ neutral background corpus -------------------------------------------
NEUTRAL = [
    "The weather forecast shows mild temperatures for the region.",
    "Consumer electronics sales remained robust in the holiday season.",
    "Agricultural output exceeded prior-year levels by a small margin.",
    "Tourism activity picked up as travel restrictions eased.",
    "Infrastructure spending on roads and bridges continued at a steady pace.",
    "Retail sales figures came in broadly in line with expectations.",
    "Housing starts edged up modestly from the previous month.",
    "Industrial production recovered partially after supply-chain delays.",
    "Corporate earnings reports were mixed across sectors.",
    "Trade flows normalized as port congestion eased over the quarter.",
]

# ------ EPU-signal phrases (Economy ∩ Policy ∩ Uncertainty) -----------------
EPU_HIGH = [
    "Economic policy uncertainty has surged amid congressional gridlock over fiscal legislation.",
    "Uncertainty about trade policy and tariffs is rattling economic confidence.",
    "Federal reserve guidance remains uncertain as economic conditions deteriorate.",
    "Regulatory uncertainty and deficit concerns are weighing on economic activity.",
    "Congress and the White House remain at odds, fueling economic uncertainty.",
    "Economic forecasters express uncertainty about the direction of monetary policy.",
    "Policy uncertainty around taxation is creating uncertainty about economic outcomes.",
    "Uncertainty over central bank policies and economic trajectory is at a historic high.",
    "Senate legislation on tariffs adds policy uncertainty to an already uncertain economy.",
    "Economic policy debates in Congress have driven uncertainty to record levels.",
]

# ------ MPU-signal phrases (Monetary ∩ Uncertainty) -------------------------
MPU_HIGH = [
    "Monetary policy uncertainty is elevated as the central bank debates interest rate paths.",
    "The Federal Reserve's policy rate outlook remains highly uncertain.",
    "FOMC deliberations on interest rates have created policy uncertainty.",
    "Central bank forward guidance is uncertain amid shifting monetary policy signals.",
    "Monetary policy ambiguity at the ECB and Fed is weighing on markets.",
    "Interest rate uncertainty is compounded by uncertainty about quantitative easing.",
    "The Bank of England faces uncertainty over its monetary policy stance.",
    "Policy uncertainty at the Federal Reserve is at its highest in years.",
    "Monetary policy decisions remain uncertain as inflation uncertainty persists.",
    "Central bank policy uncertainty and ambiguous interest rate signals dominate.",
]


def _make_doc(date, in_window):
    """Return a 4-tuple (date, text, url, meta) with probabilistic signal."""
    base = NEUTRAL[int(RNG.integers(0, len(NEUTRAL)))]
    p_epu, p_mpu = (0.75, 0.70) if in_window else (0.08, 0.06)
    epu_txt = EPU_HIGH[int(RNG.integers(0, len(EPU_HIGH)))] if RNG.random() < p_epu else ""
    mpu_txt = MPU_HIGH[int(RNG.integers(0, len(MPU_HIGH)))] if RNG.random() < p_mpu else ""
    text = " ".join(p for p in [base, epu_txt, mpu_txt] if p)
    url  = f"https://synthetic-news.example/{date.date()}"
    meta = {"language": "en"}
    return (date, text, url, meta)


records = []
for month in pd.date_range("2018-01-01", "2023-12-01", freq="MS"):
    for k in range(4):
        day  = int(RNG.integers(1, 28))
        date = month + pd.Timedelta(days=day - 1)
        in_w = HIGH_START <= date <= HIGH_END
        records.append(_make_doc(pd.Timestamp(date), in_w))

# Shuffle so quarter assignment is random within the quarter
shuffle_idx = RNG.permutation(len(records))
records = [records[i] for i in shuffle_idx]

print(f"Corpus: {len(records)} documents, "
      f"{pd.Timestamp(records[0][0]).date()} → {pd.Timestamp(records[-1][0]).date()}")
assert len(records) == 288

# %% [markdown]
# ## 2. Compute the EPU and MPU indices
#
# Both functions accept an **iterable of `(date, text, source_url, metadata)`
# 4-tuples** — the same schema used by all live data connectors.
#
# `epu()` — Baker-Bloom-Davis three-group co-occurrence, normalized to
# BBD's 100/50 scale (mean 100, std 50 over the full sample).
#
# `mpu()` — Husted-Rogers-Sun flat-keyword count, z-score normalized.

# %%
ri_epu = epu(records, country="SYN", language="en", normalize="bbd_100")
ri_mpu = mpu(records, country="SYN", language="en", normalize="zscore")

epu_s = ri_epu.series.dropna()
mpu_s = ri_mpu.series.dropna()

print(f"EPU series:  {len(epu_s)} quarters | "
      f"mean = {epu_s.mean():.1f} | std = {epu_s.std():.1f}")
print(f"MPU series:  {len(mpu_s)} quarters | "
      f"mean = {mpu_s.mean():.2f} | std = {mpu_s.std():.2f}")

# Sanity: bbd_100 normalization targets mean=100, std=50 across the sample.
assert abs(epu_s.mean() - 100.0) < 5.0,  f"EPU mean = {epu_s.mean():.1f}"
assert abs(mpu_s.mean())         < 0.10,  f"MPU mean = {mpu_s.mean():.2f}"

# %% [markdown]
# ## 3. Validate — the index responds to the injected signal
#
# If the dictionary scoring works, the EPU series should be **materially
# higher** during 2020Q1–Q4 (where we injected signal) than in adjacent
# calm periods.

# %%
window = (epu_s.index >= HIGH_START) & (epu_s.index <= HIGH_END)

epu_in  = epu_s[window].mean()
epu_out = epu_s[~window].mean()
mpu_in  = mpu_s[(mpu_s.index >= HIGH_START) & (mpu_s.index <= HIGH_END)].mean()
mpu_out = mpu_s[~((mpu_s.index >= HIGH_START) & (mpu_s.index <= HIGH_END))].mean()

print(f"EPU  | in-window = {epu_in:.1f}  | out-window = {epu_out:.1f}  | "
      f"gap = {epu_in - epu_out:.1f} pts")
print(f"MPU  | in-window = {mpu_in:.2f}  | out-window = {mpu_out:.2f}  | "
      f"gap = {mpu_in - mpu_out:.2f} z-units")

# Hard assertion: in-window index must exceed out-window by a robust margin.
assert epu_in > epu_out + 30, (
    f"EPU: in-window ({epu_in:.1f}) should exceed out-window ({epu_out:.1f}) by >30 pts"
)
assert mpu_in > mpu_out + 0.40, (
    f"MPU: in-window z ({mpu_in:.2f}) should exceed out-window z ({mpu_out:.2f}) by >0.4"
)

# %% [markdown]
# ### Hero figure — EPU time series with injected uncertainty window shaded

# %%
cols = _nbstyle.palette(2)

fig, ax = plt.subplots()

ax.plot(epu_s.index, epu_s.values, color=cols[0], linewidth=1.8,
        label="Synthetic EPU (bbd_100, mean=100)")
ax.axhline(100, color="0.6", linewidth=0.8, linestyle=":")

# Shade the 2020 injection window (extend a quarter past Dec-2020 for visibility)
shade_end = HIGH_END + pd.Timedelta(days=92)
ax.axvspan(HIGH_START, shade_end, color="0.85", alpha=0.9, zorder=0)
ax.text(
    HIGH_START + pd.Timedelta(days=18),
    epu_s.max() * 0.89,
    "Injected\nhigh-uncertainty\nwindow (2020)",
    fontsize=8, color="0.30", va="top",
)

ax.set_xlabel("Quarter")
ax.set_ylabel("EPU (mean 100, std 50)")
ax.set_title(
    "Text-based EPU from a synthetic corpus\n"
    "(pure-numpy dictionary scoring — no API key, no network)"
)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Supporting figure — keyword hit rates confirm the signal anatomy
#
# The *EPU hit rate* (fraction of documents per quarter that co-occur all
# three groups) and the *MPU keyword density* (average raw keyword hits per
# doc) both jump sharply in 2020, tracking the injected probability exactly.

# %%
# Compute raw (pre-normalization) per-document scores.
epu_groups = [
    LEXICONS["epu"]["en"]["economy"],
    LEXICONS["epu"]["en"]["policy"],
    LEXICONS["epu"]["en"]["uncertainty"],
]
mpu_terms = LEXICONS["mpu"]["en"]

raw_epu = pd.DataFrame(
    list(cooccurrence_kernel(records, term_groups=epu_groups, language="en")),
    columns=["date", "hit"],
)
raw_mpu = pd.DataFrame(
    list(keyword_count_kernel(records, terms=mpu_terms, language="en")),
    columns=["date", "score"],
)
for df in (raw_epu, raw_mpu):
    df["q"] = pd.to_datetime(df["date"]).dt.to_period("Q").dt.to_timestamp()

hits_per_q = raw_epu.groupby("q")["hit"].mean()
mpu_per_q  = raw_mpu.groupby("q")["score"].mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.5), sharex=True)

ax1.bar(hits_per_q.index, hits_per_q.values, width=60,
        color=cols[0], alpha=0.85)
ax1.axvspan(HIGH_START, shade_end, color="0.85", alpha=0.9, zorder=0)
ax1.set_ylabel("EPU hit rate (fraction of docs)")
ax1.set_title("Fraction of docs with EPU three-group co-occurrence")

ax2.bar(mpu_per_q.index, mpu_per_q.values, width=60,
        color=cols[1], alpha=0.85)
ax2.axvspan(HIGH_START, shade_end, color="0.85", alpha=0.9, zorder=0)
ax2.set_ylabel("MPU keyword hits per doc (raw)")
ax2.set_xlabel("Quarter")
ax2.set_title("Average MPU keyword density per document")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Your turn — build a *climate*-policy uncertainty index
#
# `epu()` takes a `lexicon=` override of the form
# `{"economy": frozenset(...), "policy": frozenset(...), "uncertainty": frozenset(...)}`.
# Swap the "economy" group for a **climate** vocabulary and you have a climate-policy
# uncertainty index — with no new library code. Change the three frozensets (and the
# injected signal phrases) below to your own domain and re-run.

# %%
# ← Replace these three groups with your own domain vocabulary.
my_lexicon = {
    "economy": frozenset({"climate", "emissions", "carbon", "warming", "greenhouse"}),
    "policy": frozenset({"policy", "regulation", "treaty", "tax", "subsidy", "mandate"}),
    "uncertainty": frozenset({"uncertain", "uncertainty", "unclear", "ambiguous"}),
}
# ← And the phrases injected into the 2020 window (must hit all three groups).
CLIMATE_HIGH = [
    "Climate policy uncertainty is high as the carbon tax treaty remains unclear.",
    "Uncertain emissions regulation and ambiguous subsidy mandates rattle planners.",
    "Greenhouse policy is uncertain amid an ambiguous warming-target treaty.",
]

clim_records = []
for (date, text, url, meta) in records:
    in_window = HIGH_START <= date <= HIGH_END
    add = CLIMATE_HIGH[int(RNG.integers(0, len(CLIMATE_HIGH)))] if (in_window and RNG.random() < 0.70) else ""
    clim_records.append((date, (text + " " + add).strip(), url, meta))

clim_s = epu(clim_records, country="SYN", language="en",
             lexicon=my_lexicon, normalize="bbd_100").series.dropna()
in_w = (clim_s.index >= HIGH_START) & (clim_s.index <= HIGH_END)
clim_gap = clim_s[in_w].mean() - clim_s[~in_w].mean()
print(f"climate-EPU  in-window = {clim_s[in_w].mean():.1f}  "
      f"out-window = {clim_s[~in_w].mean():.1f}  gap = {clim_gap:.1f} pts")
assert clim_gap > 20, "your climate lexicon should fire in the injected window"

# %% [markdown]
# **Prompts.** (1) Broaden the `economy` (climate) group with more domain terms (e.g.
# `renewable`, `fossil`, `sustainability`) — does the index flag more of your climate
# documents? (2) Set `normalize="zscore"` and compare
# the in/out gap in z-units. (3) Build the same index in another language by translating the
# three frozensets and passing `language="es"`.
#
# **How comprehensive is this?** The same `(date, text, url, metadata)` records feed every
# narrative index in puremacro: `mpu` (monetary-policy uncertainty), `lui`/`lwui`
# (labor-market and wage uncertainty), and `tone` (hawkish/dovish central-bank tone). The
# library ships ~70 free text connectors (central-bank speeches, Beige Book, EUR-Lex,
# Bluesky, …) and multilingual lexicons. **Notebook 13** generalizes the *idea* of an index
# beyond text — the same "define a kernel, apply it to data, normalize" recipe builds macro,
# financial, and cross-sectional uncertainty indices.
