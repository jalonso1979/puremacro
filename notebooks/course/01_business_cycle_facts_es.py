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
# # Módulo 1 — Hechos del ciclo económico y filtrado
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Separar tendencia y ciclo** de una serie macroeconómica.
# 2. Contrastar el filtro de **Hodrick–Prescott** con el de **Hamilton (2018)**.
# 3. Calcular los **hechos del ciclo económico**: volatilidad relativa, comovimiento, persistencia.
# 4. Leer la vista **espectral**: cuánta varianza cae en la banda de 6–32 trimestres.
#
# Todo corre en Python puro (navegador/iPad, $0).

# %% slideshow={"slide_type": "skip"}
import sys
import numpy as np
import matplotlib.pyplot as plt

_cwd = __import__("pathlib").Path.cwd()
_nb = _cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"
sys.path.insert(0, str(_nb)); sys.path.insert(0, str(_nb / "course"))
import _nbstyle; _nbstyle.apply_style()
from _tutor import tutor

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. Tendencia y ciclo
#
# Una serie macro $y_t$ (digamos el PIB en logaritmos) mezcla una **tendencia** lenta
# $\tau_t$ y un componente **cíclico** $c_t = y_t - \tau_t$. El análisis del ciclo
# económico estudia el ciclo: su volatilidad, su comovimiento entre variables y su
# persistencia. Lo difícil es *definir la tendencia*.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Dos filtros
# - **Hodrick–Prescott (HP):** elige $\tau_t$ equilibrando ajuste y suavidad, mediante
#   $\lambda$ (1600 para datos trimestrales).
# - **Hamilton (2018):** regresa $y_{t+h}$ sobre sus propios rezagos recientes; el
#   residuo es el ciclo. Hamilton sostiene que el filtro HP induce dinámicas *espurias*
#   ("Why You Should Never Use the Hodrick–Prescott Filter") y propone esta alternativa.
#
# `puremacro` incluye ambos — `data.hp_filter` y `cycles.hamilton_filter` — en NumPy puro.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Un ejemplo trabajado
# Simulamos una serie de PIB (log): tendencia de crecimiento determinista + un ciclo de
# ~24 trimestres + ruido, junto con un "consumo" que comueve, y extraemos el ciclo de
# dos maneras.

# %% slideshow={"slide_type": "fragment"}
from puremacro.data import hp_filter
from puremacro.cycles import hamilton_filter
from puremacro.spectral import business_cycle_band_power

rng = np.random.default_rng(20260531)
T = 240                                   # 60 years, quarterly
t = np.arange(T)
trend = 100.0 + 0.4 * t                   # deterministic growth trend (log points)
bc = np.zeros(T)                          # an AR(2) business-cycle component
for i in range(2, T):
    bc[i] = 1.55 * bc[i - 1] - 0.70 * bc[i - 2] + rng.standard_normal()
bc = 2.0 * bc / bc.std()                  # scale to ~2 log points
gdp = trend + bc + 0.3 * rng.standard_normal(T)
cons = 0.95 * trend + 0.6 * bc + 0.4 * rng.standard_normal(T)  # comoves, less volatile

hp_cycle, hp_trend = hp_filter(gdp)               # (cycle, trend), cycle has mean ~0
ham_cycle, ham_trend = hamilton_filter(gdp)       # (cycle, trend); first h+p obs are NaN

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Los dos componentes cíclicos
# Ambos aíslan las oscilaciones de ~24 trimestres; el filtro de Hamilton se define solo
# tras su calentamiento (se descartan los primeros trimestres) y tiende a dejar un ciclo
# de mayor amplitud.

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(t, np.asarray(hp_cycle), color=cols[0], lw=1.4, label="HP cycle ($\\lambda=1600$)")
ax.plot(t, np.asarray(ham_cycle), color="0.55", lw=1.4, ls=(0, (4, 2)), label="Hamilton (2018) cycle")
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel("quarter"); ax.set_ylabel("log points")
ax.set_title("Cyclical component of (log) GDP: HP vs Hamilton")
ax.legend(loc="upper right", ncol=2)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Hechos del ciclo económico
# Sobre el ciclo HP: volatilidad relativa del consumo, persistencia (autocorrelación de
# primer orden), comovimiento (correlación de los ciclos de PIB y consumo) y la cuota
# espectral de varianza en la banda de 6–32 trimestres.

# %% slideshow={"slide_type": "fragment"}
cons_cycle, _ = hp_filter(cons)
cons_cycle = np.asarray(cons_cycle); g = np.asarray(hp_cycle)

rel_vol = cons_cycle.std() / g.std()
persistence = np.corrcoef(g[1:], g[:-1])[0, 1]
comovement = np.corrcoef(g, cons_cycle)[0, 1]
band_share = business_cycle_band_power(g)

print(f"relative volatility  sd(cons)/sd(gdp) = {rel_vol:.2f}")
print(f"persistence  corr(c_t, c_t-1)          = {persistence:.2f}")
print(f"comovement   corr(gdp, cons) cycles     = {comovement:.2f}")
print(f"spectral share in 6-32q band            = {band_share:.2f}")

assert 0.0 < band_share <= 1.0
assert comovement > 0.5            # consumption comoves with output

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Laboratorio práctico
# Profundiza con los cuadernos de funcionalidades: **`notebooks/09_growth_at_risk`** (la
# vista distribucional del ciclo) y los casos de validación de **`spectral`**
# (`puremacro.validation`, subsistema `spectral`) para la maquinaria de Welch usada aquí.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Ejercicios
# 1. Repite con $\lambda = 6.25$ (anual) y $\lambda = 129{,}600$ (mensual). ¿Cómo cambia
#    el ciclo HP?
# 2. Varía el horizonte `h` de Hamilton (p. ej. 4, 8, 12). ¿Qué ocurre con la pérdida de
#    calentamiento y la amplitud del ciclo?
# 3. Agrega una segunda serie *menos* comovida y recalcula el comovimiento.
# 4. Calcula `business_cycle_band_power` sobre el `gdp` **crudo** vs el ciclo HP — ¿por
#    qué difieren tanto?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Menor $\lambda$ → la tendencia sigue más de cerca a los datos → ciclo más pequeño y
#    de mayor frecuencia; mayor $\lambda$ → tendencia más suave → ciclo más grande y de
#    menor frecuencia.
# 2. Un `h` mayor descarta más observaciones iniciales y suele dar un ciclo de mayor
#    amplitud y menor frecuencia.
# 3. Una serie con más peso de ruido propio frente al término compartido `bc` muestra
#    menor comovimiento.
# 4. La serie cruda está dominada por la tendencia (la mayor parte de la varianza en las
#    frecuencias más bajas), así que su cuota en 6–32t es pequeña; el ciclo HP está
#    detrended, por lo que su masa cae en la banda.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 5. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué argumentó Hamilton (2018) en contra del filtro HP? Da la intuición en una frase."
# - "Si el consumo es *menos* volátil que el producto pero comueve fuertemente, ¿qué dice
#   eso sobre el suavizamiento del consumo?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("In one sentence, why did Hamilton (2018) argue against the HP filter?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Descompusimos una serie en tendencia y ciclo de dos maneras (HP vs
# Hamilton), leímos los hechos canónicos del ciclo económico (volatilidad relativa,
# persistencia, comovimiento) y cuantificamos la masa espectral del ciclo — todo en
# `puremacro` (Python puro), ejecutable en el navegador. **Siguiente módulo:** el modelo
# neoclásico de crecimiento por iteración de la función de valor.
