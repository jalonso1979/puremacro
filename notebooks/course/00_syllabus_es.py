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

# %% [markdown] slideshow={"slide_type": "slide"}
# # Curso complementario de puremacro — programa
#
# Un curso bilingüe (EN/ES) de macroeconomía de posgrado construido sobre **puremacro**:
# Python puro, ejecutable en el navegador, $0. Cada lección es *una sola fuente* que
# puedes **leer como diapositivas**, **ejecutar como cuaderno** y **explorar con un tutor
# de IA sin conexión**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Cómo usar una lección
# - **Ejecútala:** abre el cuaderno de la lección (aquí o en el playground del navegador).
# - **Preséntala:** las celdas están etiquetadas como diapositivas —
#   `python tools/build_course_slides.py` genera una presentación reveal.js desde el
#   mismo cuaderno (las figuras provienen del código en vivo).
# - **Pregunta al tutor:** `from _tutor import tutor; tutor("…")` responde sin conexión
#   con un LLM local si hay uno instalado (`pip install puremacro[local-llm]`); si no, te
#   remite a las indicaciones de exploración con IA.
#
# **Plantilla de la lección:** objetivos de aprendizaje → teoría concisa → un ejemplo
# trabajado con puremacro → un "laboratorio" práctico (un cuaderno de funcionalidades) →
# ejercicios con soluciones → exploración con IA.

# %% slideshow={"slide_type": "slide"}
import pandas as pd

modules = pd.DataFrame(
    [
        ("1", "Business-cycle facts & filtering", "cycles, spectral, data", "—", "available"),
        ("2", "Neoclassical growth & VFI", "vfi", "02", "planned"),
        ("3", "Heterogeneous agents (Aiyagari/Huggett)", "vfi", "01", "planned"),
        ("4", "VAR & SVAR identification", "var", "06", "planned"),
        ("5", "Local projections & shocks", "lp", "07", "planned"),
        ("6", "Volatility & growth-at-risk", "garch, gar", "08, 09", "planned"),
        ("7", "Causal designs (DiD, synthetic control)", "did, synthetic_control", "10", "planned"),
        ("8", "Uncertainty, narrative & AI-in-macro", "narrative", "11", "planned"),
    ],
    columns=["module", "topic", "puremacro", "lab", "status"],
).set_index("module")
modules

# %% [markdown] slideshow={"slide_type": "slide"}
# El Módulo 1 se publica ahora como plantilla trabajada; los módulos 2–8 siguen. Cada uno
# vincula un tema del curso con un subsistema de puremacro y un "laboratorio" práctico.
# **Empieza con el Módulo 1 → `01_business_cycle_facts`.**
