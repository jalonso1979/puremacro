# notebooks/local_llm_uncertainty.py
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
# # Free local-LLM narrative analysis (no API key, $0)
#
# puremacro's LLM features can run on a model on **your own machine** — Apple
# MLX, llama.cpp, or a local Ollama/LM Studio server — instead of a paid API.
#
# **Desktop only:** local inference needs a real engine, so this notebook does
# not run a model inside the browser playground. With no engine installed it
# falls back to an offline Mock so the notebook still executes; install one with
# `pip install "puremacro[local-llm]"` (or run Ollama) to see real inference.

# %%
import _nbstyle  # noqa: F401  (grayscale figure style; see notebooks/_nbstyle.py)

from puremacro.narrative.scoring import get_default_backend, score_llm
from puremacro.narrative.indices import get_default_provider, llm_prob_kernel

CORPUS = [
    ("2020-03-15",
     "The government announced a $500 billion infrastructure investment package.",
     "http://example.test/a"),
    ("2020-04-01",
     "Officials warned the outlook is highly uncertain and could shift abruptly.",
     "http://example.test/b"),
]

# %% [markdown]
# ## 1. Pick the best available local engine
# `get_default_backend` / `get_default_provider` auto-select MLX -> llama.cpp ->
# Ollama, falling back to a Mock if none is installed (which is what happens in
# CI / the browser).

# %%
backend = get_default_backend("qwen2.5-3b-instruct")
provider = get_default_provider("qwen2.5-3b-instruct")

# %% [markdown]
# ## 2. Extract narrative fiscal events (free)

# %%
events = score_llm(CORPUS, backend=backend, kind="fiscal")
print(f"extracted {len(events)} event(s)")
for ev in events:
    print(ev.date.date(), ev.sign, ev.magnitude, ev.magnitude_unit)

# %% [markdown]
# ## 3. Build a per-document uncertainty index (free)

# %%
series = list(llm_prob_kernel(CORPUS, provider=provider,
                              category="economic uncertainty"))
for date, p in series:
    print(date.date(), round(p, 3))

# %% [markdown]
# With a real engine installed, the April "uncertain" document scores higher
# than the March "investment" document. Swap models via the `model=` argument
# (e.g. `"gemma2-2b"` for Google's Gemma, `"llama3.2-3b"` for Meta's Llama).
