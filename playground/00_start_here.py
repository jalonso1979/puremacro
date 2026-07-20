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
# # puremacro in your browser
#
# This is a live, in-browser Python environment (JupyterLite + Pyodide) — no
# install, no account, no server. Run the cell below once to install
# `puremacro`, then open any of the showcase notebooks in the file browser.

# %%
# %pip install puremacro
import puremacro
print("puremacro", puremacro.__version__, "— ready in the browser")

# %% [markdown]
# ## Showcase notebooks
# **Heterogeneous-agent macro:** `01_wealth_inequality` · `02_aggregate_shocks`
# · `03_life_cycle_and_demographics` · `04_firm_dynamics` ·
# `05_portfolios_and_preferences`
#
# **Empirical econometrics:** `06_svar_identification` · `07_local_projections`
# · `08_garch_volatility` · `09_growth_at_risk` · `10_staggered_did` ·
# `14_tax_multiplier_three_ways` — the same tax multiplier via proxy-SVAR,
# LP-IV and narrative event study, on frozen US fiscal data
#
# **Text-as-data:** `11_narrative_uncertainty` — build an EPU/MPU uncertainty
# index from a text corpus with no API key and no language model — and
# `13_build_your_own_index` — assemble your own narrative uncertainty index
# step by step.
#
# **Trust, but verify:** `12_validation_gallery` — run the built-in validation
# scorecard right here in the browser and see every estimator checked against
# an independent reference.
#
# **En español:** every showcase notebook has a Spanish twin with the `_es`
# suffix (e.g. `06_svar_identification_es`) — same code, Spanish narrative.
# Course companion lessons (`00_syllabus`, `01_business_cycle_facts`, EN/ES)
# are included when present.
#
# Each notebook is fully synthetic and offline. Edit a cell and re-run — it all
# executes locally in your browser tab.
#
# *(One extra lives outside the playground: the `local_llm_uncertainty`
# notebook in the repo's `notebooks/` folder scores narrative text with a free
# local language model — desktop-only, since local inference cannot run inside
# the browser.)*
