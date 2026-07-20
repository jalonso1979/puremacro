# notebooks/local_llm_uncertainty_es.py
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
# # Análisis narrativo con LLM local (sin clave de API, sin costo)
#
# Las funcionalidades LLM de puremacro pueden ejecutarse sobre un modelo en
# **tu propia máquina** — Apple MLX, llama.cpp u un servidor Ollama/LM Studio
# local — en lugar de una API de pago.
#
# **Solo en escritorio:** la inferencia local requiere un motor real, por lo que
# este cuaderno no ejecuta un modelo en el entorno de navegador. Sin un motor
# instalado, recurre a un Mock sin conexión para que el cuaderno se ejecute de
# todos modos; instala uno con `pip install "puremacro[local-llm]"` (o ejecuta
# Ollama) para ver inferencia real.

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
# ## 1. Seleccionar el mejor motor local disponible
# `get_default_backend` / `get_default_provider` seleccionan automáticamente
# MLX -> llama.cpp -> Ollama, recurriendo a un Mock si ninguno está instalado
# (que es lo que ocurre en CI / el navegador).

# %%
backend = get_default_backend("qwen2.5-3b-instruct")
provider = get_default_provider("qwen2.5-3b-instruct")

# %% [markdown]
# ## 2. Extraer eventos fiscales narrativos (sin costo)

# %%
events = score_llm(CORPUS, backend=backend, kind="fiscal")
print(f"extracted {len(events)} event(s)")
for ev in events:
    print(ev.date.date(), ev.sign, ev.magnitude, ev.magnitude_unit)

# %% [markdown]
# ## 3. Construir un índice de incertidumbre por documento (sin costo)

# %%
series = list(llm_prob_kernel(CORPUS, provider=provider,
                              category="economic uncertainty"))
for date, p in series:
    print(date.date(), round(p, 3))

# %% [markdown]
# Con un motor real instalado, el documento de abril ("uncertain") obtiene una
# puntuación más alta que el documento de marzo ("investment"). El modelo puede
# cambiarse mediante el argumento `model=` (p. ej., `"gemma2-2b"` para Gemma de
# Google o `"llama3.2-3b"` para Llama de Meta).
