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
# # puremacro course companion — syllabus
#
# A bilingual (EN/ES) graduate-macro course built on **puremacro**: pure-Python,
# browser-runnable, $0. Each lesson is *one source* you can **read as slides**, **run as
# a notebook**, and **explore with an offline AI tutor**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## How to use a lesson
# - **Run it:** open the lesson notebook (here, or in the browser playground) and execute.
# - **Present it:** the cells are slide-tagged — `python tools/build_course_slides.py`
#   renders a reveal.js deck from the same notebook (figures come from live code).
# - **Ask the tutor:** `from _tutor import tutor; tutor("…")` answers offline via a local
#   LLM if one is installed (`pip install puremacro[local-llm]`), else points you to the
#   AI-exploration prompts.
#
# **Lesson template:** learning objectives → concise theory → a worked puremacro example
# → a hands-on "lab" (a feature-showcase notebook) → exercises with solutions → AI
# exploration.

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
# Module 1 ships now as the worked template; modules 2–8 follow. Each maps a course
# topic to a puremacro subsystem and a hands-on feature-showcase "lab".
# **Start with Module 1 → `01_business_cycle_facts`.**
