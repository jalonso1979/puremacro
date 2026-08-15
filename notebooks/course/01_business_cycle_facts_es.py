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
# **Curso complementario · puremacro · mazo Slides01 — medición del ciclo (semanas 1–2)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Separar tendencia y ciclo** de una serie macroeconómica.
# 2. Contrastar el filtro de **Hodrick–Prescott** con el de **Hamilton (2018)**.
# 3. Calcular los **hechos del ciclo económico**: volatilidad relativa, comovimiento, persistencia.
# 4. Leer la vista **espectral**: cuánta varianza cae en la banda de 6–32 trimestres.
# 5. **Producir** el hecho central del mazo A1: el **sesgo de punto final** del filtro HP,
#    con el PIB real de EE. UU. congelado en el propio *bundle* del curso.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`): sin conexión y sin costo.

# %% slideshow={"slide_type": "skip"}
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = __import__("pathlib").Path.cwd()
_nb = _cwd if (_cwd / "_nbstyle.py").exists() else _cwd.parent
sys.path.insert(0, str(_nb)); sys.path.insert(0, str(_nb / "course"))
import _nbstyle; _nbstyle.apply_style()
from _tutor import tutor

DATA = (_nb / "course" / "data")          # CSV congelados del bundle: todo corre sin red

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
# - **Hamilton (2018):** regresa $y_{t+h}$ sobre sus propios rezagos recientes ($h=8$,
#   $p=4$ en datos trimestrales); el residuo es el ciclo. Hamilton sostiene que el filtro
#   HP induce dinámicas *espurias* ("Why You Should Never Use the Hodrick–Prescott
#   Filter") y propone esta alternativa. Al ser **unilateral** (usa solo información
#   pasada) no sufre el sesgo de punto final del HP — lo mediremos en la sección 4.
#
# **Salvedad — no hay filtro gratis.** El filtro de Hamilton *también* distorsiona:
# amplifica los ciclos de frecuencia media, y $(h,p)$ son parámetros de sintonía tan
# discrecionales como $\lambda$ (Schüler 2018; Hodrick 2020). La elección depende de la
# pregunta, no de que un filtro sea "el correcto".
#
# `puremacro` incluye ambos — `data.hp_filter` y `cycles.hamilton_filter` — en NumPy puro.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Un ejemplo trabajado (datos **simulados**)
# Simulamos una serie de PIB (log ×100, es decir en puntos porcentuales): tendencia de
# crecimiento determinista + un ciclo AR(2) + ruido, junto con un "consumo" que comueve,
# y extraemos el ciclo de dos maneras.
#
# El AR(2) $c_t = \phi_1 c_{t-1} + \phi_2 c_{t-2} + \varepsilon_t$ con
# $(\phi_1,\phi_2)=(1.55,-0.70)$ tiene raíces complejas, así que su periodo implícito es
# $2\pi/\theta$ con $\cos\theta = \phi_1/(2\sqrt{-\phi_2})$. El código lo imprime abajo:
# cae **dentro** de la banda de 6–32 trimestres, que es justamente lo que queremos.

# %% slideshow={"slide_type": "fragment"}
from puremacro.data import hp_filter
from puremacro.cycles import hamilton_filter
from puremacro.spectral import business_cycle_band_power

rng = np.random.default_rng(20260531)
T = 240                                   # 60 años, datos trimestrales
t = np.arange(T)
trend = 100.0 + 0.4 * t                   # tendencia determinista (log×100; 0.4 pp/trim ≈ 1.6% anual)
PHI1, PHI2 = 1.55, -0.70                  # AR(2) con raíces complejas
bc = np.zeros(T)                          # componente cíclico AR(2)
for i in range(2, T):
    bc[i] = PHI1 * bc[i - 1] + PHI2 * bc[i - 2] + rng.standard_normal()
bc = 2.0 * bc / bc.std()                  # escala a sd = 2 puntos porcentuales
gdp = trend + bc + 0.3 * rng.standard_normal(T)
cons = 0.95 * trend + 0.6 * bc + 0.4 * rng.standard_normal(T)  # comueve, menos volátil

# Periodo implícito del AR(2): 2*pi/theta con cos(theta) = phi1 / (2*sqrt(-phi2)).
theta = np.arccos(PHI1 / (2.0 * np.sqrt(-PHI2)))
print(f"periodo implícito del ciclo simulado = {2 * np.pi / theta:.1f} trimestres "
      f"({2 * np.pi / theta / 4:.1f} años) — dentro de la banda 6–32t")

hp_cycle, hp_trend = hp_filter(gdp)               # (ciclo, tendencia); el ciclo tiene media ~0
ham_cycle, ham_trend = hamilton_filter(gdp)       # (ciclo, tendencia)
# El ciclo de Hamilton no existe antes de t = h+p-1: las primeras h+p-1 = 11 obs son NaN.
print(f"observaciones NaN al inicio del ciclo de Hamilton = {np.isnan(ham_cycle).sum()}")

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Los dos componentes cíclicos
# Ambos aíslan la misma oscilación de ~16 trimestres; el de Hamilton se define solo tras
# su calentamiento (las primeras $h+p-1=11$ observaciones son `NaN`) y deja un ciclo de
# **mayor amplitud**. La celda siguiente al gráfico imprime las dos desviaciones estándar
# y su correlación, para que no haya que creerlo: se comprueba.

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(t, np.asarray(hp_cycle), color=cols[0], lw=1.4, label="ciclo HP ($\\lambda=1600$)")
ax.plot(t, np.asarray(ham_cycle), color="0.55", lw=1.4, ls=(0, (4, 2)), label="ciclo de Hamilton (2018)")
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel("trimestre"); ax.set_ylabel("puntos porcentuales (log × 100)")
ax.set_title("Componente cíclico del PIB (log): HP vs Hamilton")
ax.legend(loc="upper right", ncol=2)
plt.show()

# %% slideshow={"slide_type": "fragment"}
_hp = np.asarray(hp_cycle); _ham = np.asarray(ham_cycle)
_ok = ~np.isnan(_ham)
print(f"sd(ciclo HP)       = {_hp.std():.2f} pp")
print(f"sd(ciclo Hamilton) = {np.nanstd(_ham):.2f} pp   -> Hamilton amplifica")
print(f"corr(HP, Hamilton) = {np.corrcoef(_hp[_ok], _ham[_ok])[0, 1]:.2f}")

assert np.nanstd(_ham) > _hp.std()        # el ciclo de Hamilton es el de mayor amplitud

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

print(f"volatilidad relativa  sd(cons)/sd(PIB)      = {rel_vol:.2f}")
print(f"persistencia          corr(c_t, c_t-1)      = {persistence:.2f}")
print(f"comovimiento          corr(PIB, cons) ciclos = {comovement:.2f}")
print(f"cuota espectral en la banda 6-32t           = {band_share:.2f}")

assert 0.0 < band_share <= 1.0
assert comovement > 0.5            # el consumo comueve con el producto

# %% [markdown] slideshow={"slide_type": "subslide"}
# > **Estos cuatro números no son un hecho estilizado.** Salen de una serie *simulada* por
# > nosotros: la volatilidad relativa que ves es la que programamos en el coeficiente
# > `0.6` de `cons`, no una medición de ninguna economía. La regla del curso es que
# > **ningún segundo momento se publica sin su ficha de medición de seis campos** (fuente
# > y serie, muestra, filtro, base de precios, orden recortar–filtrar, edición del dato).
# > Las cifras canónicas del curso para $\sigma_c/\sigma_y$ —México y EE. UU.— y su ficha
# > completa se construyen en la **lección 04 (momentos del RBC)** y se estresan en la
# > **10b (economía abierta)**. Aquí solo aprendemos la maquinaria.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Datos reales: el sesgo de punto final del HP
#
# Hasta aquí, simulación. Ahora el hecho central del mazo A1, con el **PIB real de
# EE. UU.** (`GDPC1`, FRED) congelado en `data/` del propio *bundle*: no hace falta red.
#
# La pregunta del mazo A1 es de política, no de econometría: *en diciembre de 2019, ¿cuál
# de los dos ciclos habrían querido conocer?* El filtro HP es **bilateral** —para estimar
# la tendencia en 2019Q4 usa también lo que pasó después—, así que el ciclo que reporta
# hoy para 2019Q4 **no es** el que habría reportado en diciembre de 2019. El de Hamilton
# es **unilateral** y apenas se mueve.

# %% slideshow={"slide_type": "fragment"}
gdp_us = pd.read_csv(DATA / "GDPC1.csv")
gdp_us["q"] = pd.PeriodIndex(pd.to_datetime(gdp_us["observation_date"]), freq="Q")
y_us = 100.0 * np.log(gdp_us.set_index("q")["GDPC1"])      # log-PIB en puntos porcentuales
print(f"GDPC1: {y_us.index[0]}–{y_us.index[-1]}, {len(y_us)} trimestres (CSV congelado)")

hp_full = np.asarray(hp_filter(y_us)[0])                   # muestra completa
ham_full = np.asarray(hamilton_filter(y_us)[0])
_ok_us = ~np.isnan(ham_full)
print(f"sd(HP) = {hp_full.std():.2f} pp   sd(Hamilton) = {np.nanstd(ham_full):.2f} pp   "
      f"corr = {np.corrcoef(hp_full[_ok_us], ham_full[_ok_us])[0, 1]:.2f}")

# %% slideshow={"slide_type": "fragment"}
i19 = y_us.index.get_loc(pd.Period("2019Q4", freq="Q"))
y_trunc = y_us.iloc[:i19 + 1]                              # lo que se sabía en 2019Q4
hp_trunc = np.asarray(hp_filter(y_trunc)[0])
ham_trunc = np.asarray(hamilton_filter(y_trunc)[0])

rev_hp = hp_full[i19] - hp_trunc[-1]
rev_ham = ham_full[i19] - ham_trunc[-1]
print(f"ciclo HP en 2019Q4:       en tiempo real {hp_trunc[-1]:+.2f} pp  ->  hoy {hp_full[i19]:+.2f} pp   (revisión {rev_hp:+.2f} pp)")
print(f"ciclo Hamilton en 2019Q4: en tiempo real {ham_trunc[-1]:+.2f} pp  ->  hoy {ham_full[i19]:+.2f} pp   (revisión {rev_ham:+.2f} pp)")
print(f"el HP se revisa {abs(rev_hp) / abs(rev_ham):.0f} veces más que el de Hamilton")

assert abs(rev_hp) > 10 * abs(rev_ham)     # el sesgo de punto final es de otro orden

# ¿Es de *punto final* o afecta a toda la serie? Un trimestre interior (2015Q4, a 16
# trimestres del corte) casi no se revisa; y la revisión de 2019Q4 ya está completa en
# 2021Q4, es decir la aportan los propios trimestres de la pandemia.
def hp_en(periodo: str, fin: str) -> float:
    i = y_us.index.get_loc(pd.Period(periodo, freq="Q"))
    j = y_us.index.get_loc(pd.Period(fin, freq="Q"))
    return float(np.asarray(hp_filter(y_us.iloc[:j + 1])[0])[i])

print(f"ciclo de 2015Q4:  tiempo real {hp_en('2015Q4', '2015Q4'):+.2f}  "
      f"corte 2019Q4 {hp_en('2015Q4', '2019Q4'):+.2f}  hoy {hp_en('2015Q4', '2026Q1'):+.2f}")
print(f"ciclo de 2019Q4:  tiempo real {hp_en('2019Q4', '2019Q4'):+.2f}  "
      f"corte 2021Q4 {hp_en('2019Q4', '2021Q4'):+.2f}  hoy {hp_en('2019Q4', '2026Q1'):+.2f}")

# %% slideshow={"slide_type": "slide"}
_w = slice(i19 - 40, i19 + 1)
_qs = np.arange(len(y_us))[_w]
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.plot(_qs, hp_full[_w], color=cols[0], lw=1.5, label="ciclo HP, muestra completa (hoy)")
ax.plot(_qs, hp_trunc[i19 - 40:], color="0.55", lw=1.5, ls=(0, (4, 2)),
        label="ciclo HP recalculado con datos hasta 2019Q4")
ax.axhline(0, color="0.85", lw=0.6)
ax.annotate(f"revisión {rev_hp:+.2f} pp", xy=(_qs[-1], hp_full[i19]),
            xytext=(_qs[-1] - 22, hp_full[i19] + 0.55), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
ax.set_ylim(top=hp_full[i19] + 1.1)
ax.set_xticks(_qs[::8]); ax.set_xticklabels([str(p) for p in y_us.index[_w][::8]], fontsize=8)
ax.set_ylabel("puntos porcentuales"); ax.set_title("Sesgo de punto final del filtro HP (GDPC1)")
ax.legend(loc="lower left")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** El mismo trimestre, el mismo dato y el mismo filtro dan dos respuestas
# distintas según cuánta muestra futura exista. Es exactamente el mecanismo de las
# *revisiones* del mazo A1 (ALFRED, vintages), pero producido por el **filtro** y no por
# la agencia estadística: aquí el dato subyacente no cambió, cambió la ventana.
#
# Los dos diagnósticos impresos arriba precisan el resultado, y conviene leerlos antes de
# sobrevenderlo:
# 1. Es un sesgo **de punto final**, no una deformación de toda la serie: el ciclo de
#    2015Q4 —a 16 trimestres del corte— vale $+0.19$ en tiempo real y $+0.18$ hoy. Basta
#    alejarse del extremo para que la revisión desaparezca.
# 2. La revisión de 2019Q4 ya está completa en **2021Q4** ($+1.85$, contra $+1.73$ hoy):
#    la aportan los trimestres de la **pandemia**, que arrastran hacia abajo la tendencia
#    estimada para 2019 y por tanto elevan el ciclo. La magnitud concreta de $1.34$ pp es
#    de este episodio; el sesgo de punto final, en cambio, es estructural (Hamilton 2018).
# 3. Que Hamilton se revise poco **no** lo hace mejor filtro (ver la salvedad de la
#    sección 1): es unilateral por construcción, y ese es todo el mérito aquí — como
#    verás en el ejercicio 5, con otro corte también se revisa.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Laboratorio práctico
# Profundiza con la lección **`23_growth_at_risk_es`** de este mismo paquete de cuadernos
# (la vista distribucional del ciclo) y, si quieres auditar la maquinaria de Welch usada
# aquí, corre los casos de validación del subsistema `spectral`:
#
# ```python
# from puremacro import validation
# print(validation.scorecard(validation.run_all("spectral")))
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Ejercicios
# 1. Repite con $\lambda = 6.25$ (anual) y $\lambda = 129{,}600$ (mensual). ¿Cómo cambia
#    el ciclo HP?
# 2. Varía el horizonte `h` de Hamilton (p. ej. 4, 8, 12). ¿Qué ocurre con la pérdida de
#    calentamiento y la amplitud del ciclo?
# 3. Agrega una segunda serie *menos* comovida y recalcula el comovimiento.
# 4. Calcula `business_cycle_band_power` sobre el `gdp` **crudo** vs el ciclo HP — ¿por
#    qué difieren tanto?
# 5. Repite la sección 3 truncando en **2007Q4** en vez de 2019Q4 (víspera de la Gran
#    Recesión). ¿Sigue siendo el HP el que más se revisa?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Menor $\lambda$ → la tendencia sigue más de cerca a los datos → ciclo más pequeño y
#    de mayor frecuencia; mayor $\lambda$ → tendencia más suave → ciclo más grande y de
#    menor frecuencia. Sobre la simulación: $sd$ del ciclo $=0.57$, $1.73$ y $1.96$ para
#    $\lambda = 6.25$, $1600$ y $129{,}600$. (Ojo: $6.25$ y $129{,}600$ son las reglas
#    para datos **anuales** y **mensuales**; aplicarlas a datos trimestrales es
#    justamente el error de sintonía que el ejercicio quiere hacer visible.)
# 2. El calentamiento perdido es exactamente $h+p-1$, así que crece uno a uno con `h`
#    ($7$, $11$, $15$ observaciones para $h=4,8,12$ con $p=4$). La amplitud **no** es
#    monótona: sobre esta simulación $sd$ pasa de $2.40$ a $2.94$ y baja a $2.73$
#    ($h=4,8,12$). Al proyectar más allá de medio ciclo ($\approx 8$ trimestres aquí) el
#    residuo deja de crecer. Es la razón por la que $(h,p)$ son parámetros de sintonía tan
#    discrecionales como $\lambda$: no hay un valor "correcto" que la teoría entregue.
# 3. Una serie con más peso de ruido propio frente al término compartido `bc` muestra
#    menor comovimiento.
# 4. La serie cruda está dominada por la tendencia (la mayor parte de la varianza en las
#    frecuencias más bajas), así que su cuota en 6–32t es pequeña; el ciclo HP está
#    *detrended*, por lo que su masa cae en la banda. Con la simulación de la sección 2:
#    $0.27$ sobre `gdp` crudo contra $0.93$ sobre `hp_cycle`. (No es $0$ sobre el crudo
#    porque `welch_psd` resta la media de cada segmento de 64 obs., lo que ya elimina
#    parte de la tendencia.)
# 5. Sí, pero por mucho menos margen: la revisión del HP en 2007Q4 es $\approx +2.8$ pp y
#    la de Hamilton $\approx +0.7$ pp (razón $\approx 4$, contra $\approx 55$ en 2019Q4).
#    El ciclo de Hamilton usa solo datos pasados, pero las betas de la regresión **sí** se
#    reestiman al alargar la muestra: unilateral no significa inmune a la revisión.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 6. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué argumentó Hamilton (2018) en contra del filtro HP? Da la intuición en una frase."
# - "Si el consumo es *menos* volátil que el producto pero comueve fuertemente, ¿qué dice
#   eso sobre el suavizamiento del consumo?"
# - "El ciclo HP de 2019Q4 cambió 1.3 pp al añadir los datos de 2020-2026. ¿Por qué un
#   filtro bilateral reescribe el pasado, y qué tiene que ver con las revisiones de datos?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase, ¿por qué argumentó Hamilton (2018) en contra del filtro HP?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Descompusimos una serie en tendencia y ciclo de dos maneras (HP vs
# Hamilton), leímos los hechos canónicos del ciclo económico (volatilidad relativa,
# persistencia, comovimiento) sobre datos **simulados**, cuantificamos la masa espectral
# del ciclo y, con el PIB real de EE. UU., **medimos el sesgo de punto final del HP**:
# 1.3 pp en 2019Q4 contra 0.02 pp del filtro de Hamilton. La lección que queda es que el
# filtro no es un detalle técnico —decide el ciclo—, y que ningún segundo momento vale
# sin su ficha de medición. Todo en `puremacro` (Python puro), sobre tu instalación
# local y sin red. **Siguiente módulo:** la lección
# **01b — Kaldor, capital y participación del trabajo**, que cierra el bloque de medición
# (semanas 1–2) antes de pasar al modelo neoclásico de crecimiento (lección 02).
