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
# # Proyecciones locales: efectos causales dinámicos
#
# ¿Cómo se propaga un choque puntual a lo largo de la economía en los años
# siguientes, y se propaga de forma distinta en una recesión que en una expansión?
# Las proyecciones locales de Jordà (2005) responden a esto con una regresión por
# horizonte —robustas ante una especificación dinámica errónea y fáciles de volver
# *dependientes del estado del ciclo*. Plantamos un choque sintético cuyo efecto es
# más intenso en recesiones y recuperamos tanto las respuestas lineales como las
# específicas por régimen con `puremacro.lp`.

# %% [markdown]
# ## El método en ecuaciones
#
# Para trazar el efecto causal dinámico de un choque $s_t$ sobre una variable $y$, Jordà
# (2005) corre **una regresión por horizonte** $h$, proyectando el cambio acumulado
# de la variable desde $t-1$ sobre el choque de hoy:
# $$ y_{t+h} - y_{t-1} = \alpha_h + \beta_h\, s_t + \gamma_h' x_t + e_{t+h}, \qquad h = 0, 1, \dots, H. $$
# La función de impulso-respuesta *es* la sucesión de coeficientes de pendiente
# $\{\beta_h\}_{h=0}^{H}$, leída horizonte a horizonte —sin un VAR que invertir e iterar
# hacia adelante. Los controles $x_t$ (aquí rezagos de $y$ y de $s$) absorben la dinámica
# predecible para que $\beta_h$ aísle el efecto del choque.
#
# **Inferencia.** Apilar horizontes implica que el residuo $e_{t+h}$ se solapa con su propio
# futuro: $e_{t+h}$ y $e_{t+1+h}$ comparten $h$ períodos de innovaciones, por lo que están
# correlacionados serialmente *por construcción*. Las pendientes de MCO siguen siendo
# consistentes, pero sus errores estándar de manual son incorrectos. Empleamos por ello
# errores estándar **HAC (Newey–West)** con un núcleo de Bartlett y ancho de banda $h+1$
# (Plagborg-Møller–Wolf 2021), y reportamos bandas
# $\beta_h \pm z_{1-\alpha/2}\, \mathrm{se}(\beta_h)$.
#
# **Dependencia del estado.** Como cada horizonte es su propia regresión, la no linealidad
# encaja sin costo. Interactúe el choque con un indicador de estado $I_t$ (p. ej. $1$ en
# recesiones) para obtener respuestas específicas por régimen en una sola ecuación:
# $$ y_{t+h} = I_t\big(\alpha_h^{R} + \beta_h^{R} s_t\big) + (1-I_t)\big(\alpha_h^{E} + \beta_h^{E} s_t\big) + \gamma_h' x_t + e_{t+h}, $$
# que entrega una FIR de recesión $\{\beta_h^{R}\}$ y una FIR de expansión $\{\beta_h^{E}\}$,
# cada una con su propia banda HAC.
#
# **Intuición.** Las proyecciones locales estiman cada horizonte *por separado*, de modo que
# —a diferencia del SVAR del Cuaderno 6, que impone una única ley de movimiento paramétrica y
# la itera hacia adelante— una dinámica de corto plazo mal especificada no contamina la
# respuesta de horizonte largo, y una partición por estado es apenas un término de
# interacción en lugar de un segundo modelo completo. El precio es la *eficiencia*: descartar
# las restricciones entre horizontes vuelve las estimaciones más ruidosas, sobre todo a lo
# lejos, donde las ventanas solapadas reducen la muestra efectiva y las bandas HAC se abren
# en abanico. Las PL compran robustez y flexibilidad con varianza; el SVAR compra precisión
# con estructura.

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

from puremacro.lp.jorda import lp_hac
from puremacro.lp.state_dep import lp_state_dep

# Pyodide contract: after importing the functions we use, no forbidden module
# should have been pulled in. Both LP paths go through the pure-numpy ols_hac.
_bad = [m for m in ("statsmodels", "linearmodels", "arch", "bs4", "requests", "numba")
        if m in sys.modules]
assert _bad == [], f"forbidden modules imported: {_bad}"

# %% [markdown]
# ## 1. Un choque sintético con propagación dependiente del estado
# Un choque i.i.d. unitario incide sobre una variable cuyo impacto es *mayor en
# recesiones*. El indicador dicotómico de recesión se construye a partir de un
# factor latente persistente del ciclo económico.

# %%
rng = np.random.default_rng(20240529)
T = 360

x = rng.standard_normal(T)                       # identified structural shock
g = np.zeros(T)                                  # persistent business-cycle factor
for t in range(1, T):
    g[t] = 0.85 * g[t - 1] + rng.standard_normal() * 0.5
recession = (g < np.quantile(g, 0.35)).astype(float)   # binary state dummy

beta_exp, beta_rec = -0.25, -0.90                # impact in expansion vs recession
y = np.zeros(T)
for t in range(1, T):
    impact = beta_rec if recession[t] else beta_exp
    y[t] = (0.55 * y[t - 1]
            + impact * x[t - 1]
            + 0.30 * impact * x[t]
            + rng.standard_normal() * 0.35)

df = pd.DataFrame(
    {"y": y, "shock": x, "recession": recession},
    index=pd.date_range("1985-01-01", periods=T, freq="MS"),
)
print(f"recession sample share = {df['recession'].mean():.2f}")

H = 16
horizons = range(0, H + 1)

# %% [markdown]
# ## 2. Proyección local lineal con corrección HAC
# `lp_hac` regresa `y_{t+h} - y_{t-1}` sobre el choque más sus rezagos, con
# errores estándar HAC de Newey-West (ancho de banda h+1) y bandas de confianza
# al 90%.

# %%
lin = lp_hac(df, y="y", x="shock", horizons=horizons, n_lags=2, alpha=0.10)
print(lin.head().to_string(index=False))

# Bands present, finite, and correctly ordered.
assert set(lin.columns) >= {"h", "beta", "se", "t", "lo", "hi"}
assert np.all(np.isfinite(lin[["beta", "se", "lo", "hi"]].values))
assert np.all(lin["se"].values > 0)
assert np.all(lin["lo"].values <= lin["beta"].values + 1e-9)
assert np.all(lin["beta"].values <= lin["hi"].values + 1e-9)
z95 = 1.6448536269514722                          # z_{0.95} for the 90% band
assert np.allclose(lin["hi"] - lin["lo"], 2 * z95 * lin["se"], atol=1e-6)

# %% [markdown]
# **Lee la salida.** Cada fila es la regresión de un horizonte: `beta` es $\beta_h$, la
# respuesta de $y$ a un choque unitario $h$ meses adelante, y `[lo, hi]` es su banda HAC al
# 90%. La columna `beta` traza una joroba —negativa en el impacto, profundizándose durante
# unos meses a medida que los términos rezagados del choque del proceso generador se
# transmiten, y luego decayendo hacia cero conforme el AR(1) de $y$ se extingue. Dos rasgos
# son diagnósticos de las PL. Primero, las bandas se *abren en abanico* con $h$: la columna
# `se` aumenta porque el ancho de banda HAC crece ($h+1$) y la muestra de ventanas solapadas
# se reduce —la PL de horizonte largo es genuinamente más ruidosa, justamente el costo de
# eficiencia de estimar cada horizonte por separado. Segundo, el último `assert` es la
# *definición* de la banda: ancho $=2 z_{0.95}\,\mathrm{se}$, de modo que lo único que
# cambia el nivel de confianza es ese multiplicador, nunca la estimación puntual (la palanca
# de *Tu turno* más abajo).

# %% [markdown]
# ## 3. Funciones de impulso-respuesta dependientes del estado: recesión vs. expansión
# `lp_state_dep` con `transition="threshold"` descompone el choque en una
# interacción con el estado alto (recesión) y el estado bajo (expansión), y
# devuelve coeficientes y bandas HAC separadas para cada régimen.

# %%
sd = lp_state_dep(df, y="y", x="shock", state="recession",
                  horizons=horizons, n_lags=2,
                  transition="threshold", alpha=0.10)
print(sd[["h", "beta_H", "beta_L"]].head().to_string(index=False))

assert {"beta_H", "se_H", "lo_H", "hi_H",
        "beta_L", "se_L", "lo_L", "hi_L"}.issubset(sd.columns)
assert np.all(np.isfinite(sd[["beta_H", "beta_L", "se_H", "se_L"]].values))
assert np.all(sd["lo_H"].values <= sd["hi_H"].values)
assert np.all(sd["lo_L"].values <= sd["hi_L"].values)

peak_H = sd["beta_H"].min()        # recession regime (state dummy = 1)
peak_L = sd["beta_L"].min()        # expansion regime
print(f"peak recession beta_H = {peak_H:.3f}   peak expansion beta_L = {peak_L:.3f}")
assert peak_H < peak_L - 0.10, (peak_H, peak_L)        # recession bites harder
gap = sd["hi_H"].values < sd["lo_L"].values            # H-band entirely below L-band
print(f"horizons with non-overlapping regime bands: {int(gap.sum())}")
assert gap.any(), "expected at least one horizon where the regimes separate"

# %% [markdown]
# **Lee la salida.** Ahora hay *dos* funciones de impulso-respuesta a partir de una sola
# regresión. `beta_H` es la respuesta en recesión (estado alto, $I_t=1$) y `beta_L` la
# respuesta en expansión. Ambas son negativas, pero el mínimo en recesión es mucho más
# profundo —el proceso generador plantó $\beta^{R}=-0.90$ frente a $\beta^{E}=-0.25$, y la PL
# recupera un mínimo de recesión aproximadamente el doble del de expansión. El objeto
# económicamente interesante es la *brecha*: en los horizontes contados arriba, la banda al
# 90% de recesión queda *enteramente por debajo* de la banda de expansión (`hi_H < lo_L`), de
# modo que la diferencia no es solo una estimación puntual —sobrevive a la inferencia HAC.
# Este es el rédito de las proyecciones locales: una FIR dependiente del estado, con bandas de
# incertidumbre específicas por régimen, a partir de una regresión interactuada en lugar de
# dos modelos separados. (Las bandas son *por régimen*; una prueba formal de
# $\beta_h^{R}=\beta_h^{E}$ usaría su covarianza conjunta —el no solapamiento es una señal
# suficiente y conservadora.)

# %% [markdown]
# ### Figura principal — proyección local lineal con corrección HAC y bandas al 90%

# %%
cols = _nbstyle.palette(3)
fig, ax = plt.subplots()
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.fill_between(lin["h"], lin["lo"], lin["hi"], color="0.80", label="90% HAC band")
ax.plot(lin["h"], lin["beta"], color=cols[0], marker="o", markersize=3,
        label=r"$\beta_h$ (linear LP)")
ax.set_xlabel("Horizon h (months)")
ax.set_ylabel("Response of y to a unit shock")
ax.set_title("Local-projection IRF (Jordà, HAC bands)")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Figura complementaria — funciones de impulso-respuesta por régimen
# La respuesta en recesión (línea continua) es marcadamente más profunda que la
# respuesta en expansión (línea discontinua); las regiones sombreadas corresponden
# a las bandas HAC al 90% de cada régimen.

# %%
fig, ax = plt.subplots()
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
ax.fill_between(sd["h"], sd["lo_H"], sd["hi_H"], color=cols[1], alpha=0.25)
ax.fill_between(sd["h"], sd["lo_L"], sd["hi_L"], color=cols[2], alpha=0.25)
ax.plot(sd["h"], sd["beta_H"], color=cols[1], marker="o", markersize=3,
        label="Recession")
ax.plot(sd["h"], sd["beta_L"], color=cols[2], marker="s", markersize=3,
        linestyle="--", label="Expansion")
ax.set_xlabel("Horizon h (months)")
ax.set_ylabel("Response of y to a unit shock")
ax.set_title("State-dependent IRFs: shocks bite harder in recessions")
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Figura complementaria — las series sintéticas y el estado del ciclo

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 3.4))
t_idx = np.arange(T)
rec = df["recession"].values
axL.plot(t_idx, df["y"].values, color="0.25", linewidth=0.8)
axL.fill_between(t_idx, df["y"].min(), df["y"].max(), where=rec > 0,
                 color="0.85", step="mid", label="recession")
axL.set_xlabel("t"); axL.set_ylabel("y")
axL.set_title("Outcome with recession shading"); axL.legend(loc="upper right")
axR.bar(["expansion", "recession"], [(1 - rec).mean(), rec.mean()],
        color=[cols[2], cols[1]])
axR.set_ylabel("sample share"); axR.set_title("State frequencies")
plt.show()

# %% [markdown]
# ## Tu turno — el nivel de confianza mueve la banda, no la FIR
#
# Las bandas de arriba son al 90% (`alpha=0.10`). Una confusión habitual es pensar que una
# banda más ancha significa una estimación "distinta" o "más incierta" —pero la FIR puntual
# $\{\beta_h\}$ es simplemente el ajuste de MCO y no depende en absoluto del nivel de
# confianza; lo único que cambia es el multiplicador $z_{1-\alpha/2}$ sobre el error estándar
# HAC (que es fijo). Vuelve a estimar la PL lineal a un nivel más estricto y confirma ambos
# hechos: la FIR queda inalterada hasta el último byte, y la banda de cada horizonte se
# vuelve *más ancha*. Cambia `your_alpha` a continuación.

# %%
# ← change this confidence level (alpha = two-sided tail mass; 0.05 → 95% bands).
your_alpha = 0.05
wide = lp_hac(df, y="y", x="shock", horizons=horizons, n_lags=2, alpha=your_alpha)

base_w = (lin["hi"] - lin["lo"]).values          # 90% band width per horizon
your_w = (wide["hi"] - wide["lo"]).values        # your-level band width
print(f"alpha={your_alpha}: mean band width {your_w.mean():.3f} vs 90% width {base_w.mean():.3f}")
print(f"point IRF unchanged: max |Δβ| = {np.max(np.abs(wide['beta'].values - lin['beta'].values)):.2e}")

# The point IRF is the same OLS fit — only the band multiplier z_{1-α/2} changes.
assert np.allclose(wide["beta"].values, lin["beta"].values, atol=1e-12)
# A tighter tail (smaller alpha) widens the band at every horizon, since every se>0.
assert np.all(your_w > base_w + 1e-9), "expected wider bands at a higher confidence level"

# %% [markdown]
# **Ejercicios.** (1) Fija `your_alpha = 0.32` (≈68%, bandas de una sigma) y confirma que las
# bandas ahora se vuelven *más estrechas* que la base al 90% —el segundo `assert` saltará, y
# ese es justamente el punto: cámbialo a `your_w < base_w` para que pase y habrás demostrado
# que el multiplicador es monótono en el nivel de confianza. (2) Alarga los controles: vuelve
# a correr el `lp_hac` de la sección 2 con `n_lags=4` y verifica que el coeficiente de impacto
# `lin["beta"][0]` siga coincidiendo con un MCO simple de `y_t − y_{t-1}` sobre `shock_t` más
# esos mismos rezagos —en $h=0$ la proyección local *es* esa regresión contemporánea. (3) Lleva
# el horizonte a `H=24` y observa cómo las bandas HAC del extremo largo se abren aún más en
# abanico conforme se reduce la muestra de ventanas solapadas.
#
# **¿Qué tan completa es la biblioteca?** `puremacro.lp` es un conjunto de herramientas
# completo de proyecciones locales más allá de estos dos estimadores: PL con aumento de
# rezagos para cobertura uniforme en horizontes largos (`la_lp`, Plagborg-Møller–Wolf 2021),
# PL-VI / identificación por proxy (`lp_iv`, Stock–Watson 2018), dependencia del estado de
# transición suave y asimétrica (por signo del choque) (`lp_smooth`, `lp_asymmetric`), PL por
# cuantiles (`lp_quantile`), y variantes de **panel** con efectos fijos en dos vías y errores
# estándar de Driscoll–Kraay, de grupo medio y CCE (`panel_lp`, `panel_lp_dk`, `panel_lp_iv`,
# `mean_group_panel_lp`, `cce_panel_lp`). La alternativa estructural —imponer todo el sistema
# dinámico en lugar de una regresión por horizonte— es el SVAR del **Cuaderno 6**
# (`puremacro.var`). Las demostraciones de principio a fin viven en `puremacro/examples/`
# (`la_lp_pmw_demo`, `lp_smooth_demo`, `lp_asymmetric_tenreyro`, `lp_panel_dk`,
# `narrative_panel_lp`). Cada estimador aquí circula por la ruta `ols_hac` de numpy puro, por
# lo que el cuaderno permanece compatible con Pyodide (sin statsmodels).
