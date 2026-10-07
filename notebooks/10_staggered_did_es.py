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
# # Diferencias en diferencias escalonadas
#
# **¿Cómo pueden los investigadores identificar el efecto causal dinámico de intervenciones de política cuando distintas unidades adoptan el tratamiento en diferentes momentos sin caer en la trampa de comparaciones prohibidas de los efectos fijos bidireccionales de libro de texto?**
# ¿Funcionó una política, cuando distintas unidades la adoptaron en años
# diferentes? Cuando el momento del tratamiento es *escalonado*, la regresión de
# libro de texto con efectos fijos en dos vías (TWFE) utiliza las unidades ya tratadas
# como controles, lo que puede contaminar la estimación cuando los efectos varían por
# cohorte o duración de la exposición. Plantamos un efecto conocido en un panel escalonado
# **simulado** y lo recuperamos con los estimadores robustos a la heterogeneidad de
# `puremacro.did`: los ATT por grupo-tiempo de **Callaway-Sant'Anna**, agregados con pesos
# por tamaño de cohorte, y el estudio de eventos ponderado por interacciones de
# **Sun-Abraham**. No se usan datos reales; todo se ejecuta en el navegador.

# %% [markdown]
# ## El método en matemáticas
#
# **Planteamiento.** La unidad $i$ recibe el tratamiento por primera vez en el tiempo calendario
# $G_i = g$ (su *cohorte*); las unidades nunca tratadas tienen $G_i=\infty$. Los resultados
# potenciales son $Y_{it}(0)$ (trayectoria sin tratar) y $Y_{it}(g)$ (trayectoria tratada-en-$g$);
# solo observamos uno de ellos.
#
# **El problema del TWFE.** La especificación aplicada por defecto regresa
# $$ Y_{it} = \alpha_i + \lambda_t + \beta\,D_{it} + \varepsilon_{it}, \qquad
# D_{it}=\mathbb{1}\{t \ge G_i\}, $$
# y lee $\hat\beta$ como "el" efecto del tratamiento. **Goodman-Bacon (2021)** demuestra que
# $\hat\beta$ es un promedio ponderado de *todas* las comparaciones $2\times2$ de DiD posibles —
# incluidas las **prohibidas**, que usan unidades *ya tratadas* como grupo de control. Los pesos
# de esas comparaciones son no negativos, pero las comparaciones que restan un efecto de
# tratamiento que cambia pueden estar sesgadas. Escritos como pesos sobre los efectos de
# tratamiento subyacentes, algunos pesos pueden ser negativos: $\hat\beta$ puede tener el signo
# equivocado aun cuando el efecto verdadero de cada unidad sea positivo.
#
# **Callaway-Sant'Anna (2021).** Esquivan la regresión. Definen un **ATT por grupo-tiempo** limpio
# para la cohorte $g$ en el tiempo $t$, diferenciado respecto del último período previo al
# tratamiento de la cohorte $g-1$ y de un conjunto de control $C$ *nunca tratado* (o aún no tratado):
# $$ \text{ATT}(g,t)=\mathbb{E}\!\left[Y_t-Y_{g-1}\mid G=g\right]
# -\mathbb{E}\!\left[Y_t-Y_{g-1}\mid C\right]. $$
# El primer término es el cambio propio de la cohorte desde el período base $g-1$ hasta $t$; el
# segundo descuenta el cambio del grupo de control en la misma ventana, dejando el efecto causal.
#
# **Pesos de agregación.** CS ponderan cada resumen por el **tamaño de la cohorte**. Con $n_g$
# unidades en la cohorte $g$ y $\mathcal{G}_e$ las cohortes observadas $e$ períodos después de la
# adopción, el perfil del estudio de eventos (su ec. 3.4) y el resumen global recomendado (ecs. 3.7
# y 3.11) son
# $$ \theta_{es}(e)=\sum_{g\in\mathcal{G}_e}\frac{n_g}{\sum_{g'\in\mathcal{G}_e} n_{g'}}\,\text{ATT}(g,g+e),
# \qquad \theta^O_{sel}=\sum_g \frac{n_g}{\sum_{g'} n_{g'}}\,\theta_{sel}(g),\quad
# \theta_{sel}(g)=\frac{1}{T-g+1}\sum_{t\ge g}\text{ATT}(g,t). $$
# $\theta^O_{sel}$ primero promedia cada cohorte sobre sus propios períodos post-tratamiento y
# luego pondera las cohortes por tamaño. La alternativa $\theta^O_W$ (ec. 3.10) promedia todas las
# celdas post-tratamiento con pesos por tamaño de cohorte, así que se inclina hacia las cohortes
# tempranas, que aportan más celdas. `callaway_santanna` reporta $\theta^O_{sel}$ como
# `att_overall` y todos los demás resúmenes en `overall_aggregations`.
#
# **Sun-Abraham (2021).** Reparan en cambio el estudio de eventos *dinámico* del TWFE: estiman
# efectos por cohorte$\times$tiempo relativo $\text{CATT}(g,e)$ en una regresión totalmente
# interactuada en cohorte y tiempo relativo, y luego los promedian en cada $e$ con las
# participaciones muestrales de las cohortes observadas ahí (su ec. 27). Sin covariables y con
# controles nunca tratados, este estimador ponderado por interacciones **coincide con
# Callaway-Sant'Anna** (SA, p. 24): las mismas celdas $\text{ATT}(g,t)$ con los mismos pesos por
# tamaño de cohorte, de modo que $\text{ATT}_{\text{SA}}(e)=\theta_{es}(e)$. Por eso los dos
# estudios de eventos de abajo son idénticos. En puremacro las cifras principales difieren solo
# porque difieren los resúmenes globales por defecto: `sun_abraham` reporta $\theta^O_W$ y
# `callaway_santanna` reporta $\theta^O_{sel}$.
#
# ### Parametrización base
#
# | Símbolo | Rol económico / econométrico | Calibración base | Unidades |
# |---|---|---|---|
# | $G_i$ | Cohorte de adopción del tratamiento (tiempo calendario) | $\{5, 8, 11, \infty\}$, cada una con probabilidad 1/4 | Índice temporal ($t$) |
# | $D_{it}$ | Indicador binario de tratamiento ($\mathbb{1}\{t \ge G_i\}$) | Despliegue escalonado | Binario $\{0, 1\}$ |
# | $\text{ATT}$ | Efecto medio del tratamiento en los tratados plantado | Constante $+1.0$ | Unidades del resultado |
# | $N$ | Número de unidades de sección cruzada en el panel | $90$ (tamaños de cohorte impresos abajo) | Unidades transversales |
# | $T$ | Número de períodos de tiempo del panel | $14$ | Períodos temporales |
# | $e$ | Tiempo relativo del evento ($t - G_i$) | $-10, \dots, +9$ | Períodos relativos |
# | $\sigma$ | Desviación estándar del ruido idiosincrático | $0.3$ | Unidades del resultado |
# | $n_{\text{boot}}$ | Réplicas del bootstrap de panel (se remuestrean unidades completas) | $400$ | Extracciones bootstrap |
#
# **Intuición.** El sesgo proviene de las *comparaciones prohibidas*. Una cohorte de adopción tardía,
# diferenciada contra una cohorte de adopción temprana que *ya* está tratada, atribuye a la cohorte
# tardía la respuesta dinámica en curso de la cohorte temprana — con un signo que puede invertir el
# resultado principal. CS y SA restringen cada comparación a un control *limpio* (nunca tratado o aún
# no tratado) y a un período previo *fijo* $g-1$, y luego agregan con pesos que se pueden leer. El
# perfil del estudio de eventos que reportan es el ATT dinámico por horizonte $e$ *desde* el
# tratamiento: cómo se acumula (o se desvanece) el efecto tras el encendido. Los pesos responden a
# "¿el efecto de quién?": $\theta^O_{sel}$ es el efecto promedio que experimentan las unidades que
# alguna vez recibieron el tratamiento, y así lo motivan CS (p. 18), como el análogo del ATT del
# caso $2\times2$.
#
# ### Referencias bibliográficas seminales
#
# - Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230. Los números de ecuación y de página siguen arXiv:1803.09015v4.
# - de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964–2996.
# - Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254–277.
# - Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175–199. Los números de ecuación y de página siguen arXiv:1804.05785.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.did import (
    callaway_santanna, sun_abraham,
    CallawaySantannaResult, SunAbrahamResult,
)

# %% [markdown]
# ## 1. Panel escalonado simulado con un ATT conocido
# Tres cohortes son tratadas por primera vez en t = 5, 8, 11; un cuarto grupo
# nunca recibe el tratamiento. El DGP corresponde a efectos fijos en dos vías con
# tendencias previas paralelas y un impulso de tratamiento *constante* `tau = +1.0`,
# por lo que el perfil verdadero del estudio de eventos es plano en +1 a partir del
# tiempo de evento 0 en adelante.

# %%
rng = np.random.default_rng(20240529)

N_UNITS, T = 90, 14
COHORTS = (5, 8, 11, np.nan)          # three treated cohorts + never-treated
TRUE_ATT, SIGMA = 1.0, 0.3

assign = rng.choice(len(COHORTS), size=N_UNITS, replace=True)
unit_fe = rng.standard_normal(N_UNITS)
time_fe = rng.standard_normal(T) * 0.3

rows = []
for i in range(N_UNITS):
    g = COHORTS[assign[i]]
    for t in range(1, T + 1):
        treated = 1.0 if (not pd.isna(g)) and t >= g else 0.0
        y = unit_fe[i] + time_fe[t - 1] + TRUE_ATT * treated + SIGMA * rng.standard_normal()
        rows.append({"unit": i, "time": t, "y": y, "treat_time": g})
panel = pd.DataFrame(rows)

first = panel.groupby("unit")["treat_time"].first()
n_g = first.value_counts()                       # cohort sizes n_g (treated cohorts only)
n_never = int(first.isna().sum())
print(f"Panel: {N_UNITS} units x {T} periods; planted ATT = {TRUE_ATT:+.1f}")
print("cohort sizes: " + ", ".join(f"g={int(g)}: {n}" for g, n in n_g.sort_index().items())
      + f", never treated: {n_never}")

# %% [markdown]
# **Intuición.** Este es el mundo más favorable posible para los estimadores: tendencias previas
# paralelas (lo único que separa a las unidades antes del tratamiento es el efecto fijo `unit_fe`),
# un grupo de control genuinamente nunca tratado y un efecto homogéneo. El ruido muestral todavía
# separa la estimación de `+1.0`; la recuperación es aproximada. El grupo nunca tratado es lo que
# permite a Callaway-Sant'Anna evitar por completo las comparaciones prohibidas con unidades ya
# tratadas. Los tamaños de cohorte son aleatorios, así que los pesos por tamaño de cohorte no son
# exactamente iguales.

# %% [markdown]
# ## 2. Dos estimadores robustos a la heterogeneidad sobre el mismo panel
# `callaway_santanna` construye la matriz de ATT(g,t) por grupo-tiempo a partir de
# controles nunca tratados, la agrega con pesos por tamaño de cohorte y remuestrea unidades
# completas por bootstrap; `sun_abraham` reutiliza *las mismas* celdas y extracciones bootstrap.
# El efecto constante y homogéneo de este ejemplo también permite estimar TWFE de manera válida;
# esta sección verifica la recuperación, y el ejercicio final de efectos crecientes con la exposición muestra al TWFE fallar.

# %%
cs = callaway_santanna(panel, unit="unit", time="time", outcome="y",
                       treat_time="treat_time", n_boot=400, alpha=0.10, seed=0)
sa = sun_abraham(panel, unit="unit", time="time", outcome="y",
                 treat_time="treat_time", n_boot=400, alpha=0.10, seed=0)

assert isinstance(cs, CallawaySantannaResult)
assert isinstance(sa, SunAbrahamResult)

print(cs.summary())
print(sa.summary())

# %% [markdown]
# ## 3. Verificaciones de recuperación y de agregación (aserciones sobre los resultados principales)
# El efecto plantado es la verificación independiente: el ATT agregado debe quedar cerca de él.
# Después, las fórmulas de agregación se recalculan a mano a partir de las propias celdas ATT(g, t)
# de la biblioteca y de los tamaños de cohorte. Es una verificación interna: prueba los pesos, no
# las celdas. El período base (tiempo de evento -1) es el punto de normalización (exactamente 0);
# y cada horizonte post-tratamiento sigue la constante +1.0.

# %%
es_cs = cs.att_event_study.sort_values("event_time").reset_index(drop=True)
es_sa = sa.att_event_study.sort_values("event_time").reset_index(drop=True)

assert abs(cs.att_overall - TRUE_ATT) < 0.25, cs.att_overall
assert abs(sa.att_overall - TRUE_ATT) < 0.25, sa.att_overall

# The CS summaries by hand, from the ATT(g, t) cells and the cohort sizes n_g.
gt_post = cs.att_gt[cs.att_gt["event_time"] >= 0].assign(n=lambda d: d["g"].map(n_g))
theta_sel_g = gt_post.groupby("g")["att"].mean()                                  # eq. 3.7
theta_O_sel = np.average(theta_sel_g, weights=n_g.reindex(theta_sel_g.index))    # eq. 3.11
theta_O_W = np.average(gt_post["att"], weights=gt_post["n"])                     # eq. 3.10
assert np.isclose(cs.att_overall, theta_O_sel)   # CS default: theta^O_sel
assert np.isclose(sa.att_overall, theta_O_W)     # SA default: theta^O_W
# SA's interaction-weighted event study is CS eq. 3.4 (SA p. 24): same cells, weights and draws.
assert np.array_equal(es_cs["event_time"], es_sa["event_time"])
assert np.allclose(es_cs[["att", "se"]], es_sa[["att", "se"]], rtol=0, atol=1e-12)

# The overall interval is the bootstrap distribution of the aggregate itself. Averaging
# the event-study CI endpoints would ignore their dependence and use different weights.
lo_overall, hi_overall = cs.att_overall_lo, cs.att_overall_hi
assert lo_overall < cs.att_overall < hi_overall

# Base period (event_time == -1) is the normalization point: exactly 0.
base = es_cs[es_cs["event_time"] == -1]
assert len(base) == 1 and abs(float(base["att"].iloc[0])) < 1e-12
leads = es_cs[es_cs["event_time"] < -1]
assert not leads.empty and np.isfinite(leads[["att", "se", "lo", "hi"]]).all().all()
assert (leads["se"] > 0).all()  # genuine estimated leads, unlike the normalized base

# Every post-treatment horizon tracks the constant true effect.
post = es_cs[es_cs["event_time"] >= 0]
for _, r in post.iterrows():
    assert abs(r["att"] - TRUE_ATT) < 0.40, (r["event_time"], r["att"])

w_sel = n_g.sort_index() / n_g.sum()
print("theta^O_sel cohort weights: " + ", ".join(f"g={int(g)}: {w:.3f}" for g, w in w_sel.items()))
print(f"CS overall ATT (theta^O_sel, eq. 3.11) = {cs.att_overall:+.3f}   se {cs.att_overall_se:.3f}"
      f"   90% CI [{lo_overall:+.3f}, {hi_overall:+.3f}]")
print(f"SA overall ATT (theta^O_W,   eq. 3.10) = {sa.att_overall:+.3f}   se {sa.att_overall_se:.3f}"
      f"   90% CI [{sa.att_overall_lo:+.3f}, {sa.att_overall_hi:+.3f}]")
print(f"true ATT = {TRUE_ATT:+.3f}; does the CS interval contain it? {lo_overall <= TRUE_ATT <= hi_overall}")
oa = cs.overall_aggregations.set_index("aggregation").loc[["group", "simple", "dynamic", "calendar"]]
print("all CS overall summaries:\n" + oa[["att", "se", "lo", "hi"]].round(3).to_string())
print(f"max |CS - SA| over the event study: att {np.abs(es_cs['att'] - es_sa['att']).max():.1e}, "
      f"se {np.abs(es_cs['se'] - es_sa['se']).max():.1e}")

# %% [markdown]
# **Lectura de los resultados.** Ambos estimadores se acercan al `+1.0` plantado. Los pesos de
# cohorte son 0.325, 0.351 y 0.325 porque los tamaños aleatorios de las cohortes son 25, 27 y 25. El
# ATT global de CS ($\theta^O_{sel}$) es `+1.138`, con un intervalo bootstrap al 90% de
# `[+1.005, +1.258]`. Su extremo inferior queda justo por encima de la verdad, así que esta
# realización no la contiene. La cobertura nominal se refiere al muestreo repetido, no a una
# garantía para cada conjunto de datos, y en muestras finitas la cobertura bootstrap es
# aproximada. No debemos escoger una semilla para forzar cobertura ni promediar extremos de
# intervalos puntuales para obtener un intervalo agregado.
#
# SA reporta `+1.129` porque su resumen por defecto es $\theta^O_W$, que pondera celdas en lugar de
# cohortes. No es otra ponderación de las cohortes: los dos estudios de eventos son idénticos
# (diferencia `0.0e+00` tanto en las estimaciones como en los errores estándar), como SA afirman
# que deben serlo. Sus intervalos se construyen de forma algo distinta: CS reporta intervalos de
# percentiles y SA intervalos normales, $\pm z\,\text{se}$. Por eso el `[+1.007, +1.251]` de SA no
# coincide con la fila `simple` de la tabla de CS, `[1.002, 1.249]`.
#
# El **perfil post-tratamiento** es plano salvo ruido: cada estimación con `event_time >= 0` queda
# a menos de 0.40 de `+1.0` (la aserción de arriba), sin acumulación ni desvanecimiento, como se
# plantó. El **coeficiente en `event_time = -1` es exactamente 0** *por construcción*, no porque haya
# pasado una prueba de tendencias previas. Los períodos anteriores son adelantos estimados genuinos
# respecto de $g-1$ y permiten examinar posibles desviaciones de tendencias paralelas. Sus bandas
# puntuales no constituyen una prueba simultánea de tendencias previas, y no rechazar tampoco
# demuestra tendencias paralelas.

# %% [markdown]
# ### Figura principal — el estudio de eventos
# Las estimaciones saltan desde el 0 normalizado en el período base hasta la línea
# verdadera de +1.0 y permanecen cerca; los adelantos anteriores fluctúan alrededor de cero.
# El sombreado muestra intervalos bootstrap de panel puntuales al 90%, no cobertura simultánea
# de toda la curva.

# %%
es = es_cs
fig, ax = _nbstyle.figura()
ax.axhline(0.0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax.axvline(-0.5, color=_nbstyle.SPINE, linewidth=0.8, linestyle="--")
ax.fill_between(es["event_time"], es["lo"], es["hi"], color=_nbstyle.NOTA, alpha=0.35,
                step="mid", label="90% pointwise CI")
ax.plot(es["event_time"], es["att"], **_nbstyle.S1, marker="o", markersize=4,
        label="CS = Sun-Abraham (eq. 3.4)")
post_e = es[es["event_time"] >= 0]
ax.plot(post_e["event_time"], [TRUE_ATT] * len(post_e), **_nbstyle.S2,
        label=f"true ATT = {TRUE_ATT:+.1f}")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Event time e (periods since treatment)")
ax.set_ylabel("ATT (outcome units)")
ax.set_title("Staggered DiD event study (Callaway-Sant'Anna)")
ax.legend(loc="upper left", fontsize=8)

# %% [markdown]
# ### Figura complementaria — ¿qué ATT global?
# Las mismas celdas ATT(g, t) dan cuatro resúmenes globales (ecs. 3.10-3.12 de CS), cada uno con su
# propio intervalo bootstrap. $\theta^O_{sel}$ es el valor por defecto de `callaway_santanna` y
# $\theta^O_W$ el de `sun_abraham`. Con un efecto homogéneo, los cuatro apuntan al mismo +1.0.

# %%
names = {"group": "$\\theta^O_{sel}$ (eq. 3.11)\nCS default",
         "simple": "$\\theta^O_W$ (eq. 3.10)\nSA default",
         "dynamic": "$\\theta^O_{es}$ (eq. 3.12)", "calendar": "$\\theta^O_c$ (eq. 3.12)"}
ypos = np.arange(len(oa))[::-1]
fig, ax = _nbstyle.figura()
ax.axvline(TRUE_ATT, color=_nbstyle.SPINE, linewidth=1.0, linestyle="--")
ax.text(TRUE_ATT, len(oa) - 0.45, f" true ATT = {TRUE_ATT:+.1f}", ha="left", va="center",
        color=_nbstyle.TEXTO, fontsize=8)
ax.errorbar(oa["att"], ypos, xerr=[oa["att"] - oa["lo"], oa["hi"] - oa["att"]],
            fmt="o", color=_nbstyle.TINTA, ecolor=_nbstyle.TEXTO, capsize=3)
ax.set_yticks(ypos, [names[a] for a in oa.index])
ax.set_ylim(-0.5, len(oa) - 0.2)
ax.set_xlabel("Overall ATT, estimate and 90% bootstrap CI (outcome units)")
ax.set_title("Four overall summaries of the same ATT(g, t) cells")

# %% [markdown]
# ### Figura complementaria — efectos grupo-tiempo subyacentes
# El ATT(g, t) posterior al tratamiento de cada cohorte orbita alrededor de +1.0
# (línea discontinua); los adelantos fluctúan alrededor de cero. En cada tiempo de evento, el
# estudio de eventos (ec. 3.4) es el promedio ponderado por tamaño de las cohortes observadas ahí;
# la leyenda da cada $n_g$.

# %%
fig, ax = _nbstyle.figura()
gt = cs.att_gt
cohorts = sorted(gt["g"].unique())
stys = _nbstyle.styles(len(cohorts))
ccols = _nbstyle.palette(len(cohorts))
for c, color, sty in zip(cohorts, ccols, stys):
    sub = gt[gt["g"] == c].sort_values("event_time")
    ax.plot(sub["event_time"], sub["att"], marker="o", markersize=3,
            color=color, linestyle=sty, label=f"cohort g={int(c)} (n={n_g[c]})")
ax.axhline(0.0, color=_nbstyle.SPINE, linewidth=0.8, linestyle=":")
ax.axhline(TRUE_ATT, color=_nbstyle.SPINE, linewidth=0.8, linestyle="--")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Event time e (periods since treatment)")
ax.set_ylabel("ATT(g, t) (outcome units)")
ax.set_title("Group-time effects by cohort")
ax.legend(loc="upper left", fontsize=8)

# %% [markdown]
# **Lectura de los resultados.** La figura de resúmenes muestra los cuatro ATT globales entre
# `1.117` (calendar) y `1.138` (group), con intervalos que se traslapan: con un efecto homogéneo
# estiman el mismo número y solo difieren en cómo ponderan las mismas celdas ruidosas. En esta
# extracción solo el intervalo de $\theta^O_{es}$, `[0.991, 1.263]`, contiene la verdad; los cuatro
# intervalos comparten las mismas extracciones bootstrap, así que tienden a fallar juntos. La
# figura de efectos grupo-tiempo expone la maquinaria. La curva $\text{ATT}(g,t)$ de cada cohorte orbita la línea
# discontinua en `+1.0` después del tratamiento, y el punto del estudio de eventos en cada horizonte
# es su promedio ponderado por tamaño de cohorte a ese tiempo transcurrido. Solo la cohorte 5 se
# observa más allá de $e = 6$, así que la cola derecha del estudio de eventos es la curva de una sola
# cohorte, con bandas más anchas. Los pesos empiezan a importar cuando los efectos difieren entre
# cohortes o crecen con la exposición: entonces los resúmenes responden preguntas distintas, y el
# TWFE no responde ninguna (véase *Tu turno*).

# %% [markdown]
# ## Tu turno — ¿cuándo falla el TWFE? Efectos que crecen con la exposición
#
# La sección 1 plantó un impulso constante, el único caso en el que el TWFE estático funciona. Ahora
# deje que el efecto crezca con la exposición: $\tau(e) = 1 + \text{slope}\cdot e$ para
# $e = t - g \ge 0$. `simulate_dynamic` vuelve a generar el panel de la sección 1 con la misma
# semilla y las mismas extracciones, y además guarda `D` y el `tau` plantado. El objetivo plantado
# es $\theta^O_{sel}$ calculado a partir de `tau` y de los tamaños de cohorte realizados. Cada
# estimador se corre dos veces, sobre el panel con ruido y sobre el resultado sin ruido $y = \tau$;
# la segunda corrida separa lo que un estimador *persigue* del ruido muestral.
#
# **Prediga primero:** al aumentar la pendiente, ¿sigue el TWFE estático al objetivo plantado? ¿En
# qué dirección falla cuando el efecto se *desvanece* (pendiente negativa)?

# %%
def simulate_dynamic(slope, cohorts=COHORTS):
    """Section 1's panel with tau(e) = TRUE_ATT + slope*e for e = t - g >= 0.

    Same seed and draws as section 1, so slope = 0 reproduces `panel`.
    """
    rng = np.random.default_rng(20240529)
    assign = rng.choice(len(cohorts), size=N_UNITS, replace=True)
    unit_fe = rng.standard_normal(N_UNITS)
    time_fe = rng.standard_normal(T) * 0.3
    rows = []
    for i in range(N_UNITS):
        g = cohorts[assign[i]]
        for t in range(1, T + 1):
            on = (not pd.isna(g)) and t >= g
            tau = TRUE_ATT + slope * (t - g) if on else 0.0
            y = unit_fe[i] + time_fe[t - 1] + tau + SIGMA * rng.standard_normal()
            rows.append({"unit": i, "time": t, "y": y, "treat_time": g,
                         "D": float(on), "tau": tau})
    return pd.DataFrame(rows)


def twfe(df, col="y"):
    """Static TWFE coefficient on D (exact two-way demeaning; balanced panel)."""
    dm = {c: (df[c] - df.groupby("unit")[c].transform("mean")
              - df.groupby("time")[c].transform("mean") + df[c].mean()).to_numpy()
          for c in (col, "D")}
    return float(dm["D"] @ dm[col] / (dm["D"] @ dm["D"]))


# ← Change this: growth of the effect per period of exposure (try 0, 0.2, 0.4 or -0.2; range -0.3 to 1.0).
SLOPE_YOU = 0.4

p_you = simulate_dynamic(SLOPE_YOU)
assert np.allclose(simulate_dynamic(0.0)["y"], panel["y"])   # slope 0 is section 1's panel

# Planted target: theta^O_sel from the planted tau(g, t) cells and the cohort sizes n_g.
cells = p_you[p_you["D"] == 1].groupby(["treat_time", "time"])["tau"].first()
theta_g = cells.groupby(level="treat_time").mean()
truth = float(np.average(theta_g, weights=n_g.reindex(theta_g.index)))

cs_you = callaway_santanna(p_you, n_boot=0)
cs_nf = callaway_santanna(p_you.assign(y=p_you["tau"]), n_boot=0)   # noise-free: y = tau
tw, tw_nf = twfe(p_you), twfe(p_you, "tau")

print(f"slope {SLOPE_YOU:+.2f}: planted theta^O_sel = {truth:+.3f}")
print(f"  noise-free y = tau : CS {cs_nf.att_overall:+.3f}   static TWFE {tw_nf:+.3f}")
print(f"  noisy panel        : CS {cs_you.att_overall:+.3f}   static TWFE {tw:+.3f}")
other = cs_nf.overall_aggregations.set_index("aggregation")["att"]
print("  other CS summaries, noise-free: "
      + ", ".join(f"{a} {other[a]:+.3f}" for a in ("simple", "dynamic", "calendar")))

# 1. On noise-free data CS reproduces the planted theta^O_sel exactly: eq. 3.11 weights.
assert np.isclose(cs_nf.att_overall, truth), (cs_nf.att_overall, truth)
# 2. With noise, the CS error is section 1's error: the estimate is linear in y and its
#    weights do not depend on y, so the slope moves estimate and target one for one.
assert np.isclose(cs_you.att_overall - truth, cs.att_overall - TRUE_ATT)
# 3. Static TWFE misses in the direction of the slope: it understates effects that grow
#    and overstates effects that fade, and is exact only when the effect is constant.
assert np.sign(round(truth - tw_nf, 9)) == np.sign(SLOPE_YOU), (truth, tw_nf)

# %% [markdown]
# **Ejercicios.**
#
# 1. *Básico.* Fije `SLOPE_YOU = 0`. Antes de ejecutar, prediga si el TWFE estático acierta el
#    efecto plantado en el panel sin ruido. Debería: con un efecto constante, las comparaciones
#    prohibidas restan una cohorte ya tratada cuyo efecto ya no cambia.
# 2. *Intermedio.* Corra 0.2 y luego 0.4. ¿Crece la brecha del TWFE en proporción a la pendiente?
#    Use las comparaciones prohibidas de Goodman-Bacon para explicar su dirección: en el $2\times2$
#    que compara la cohorte 11 con la cohorte 5 ya tratada, el efecto de la cohorte 5 sigue
#    creciendo, y ese crecimiento se resta como si fuera una tendencia de control. Luego lea los
#    otros resúmenes impresos. ¿Por qué $\theta^O_W$ (`simple`) queda por encima de
#    $\theta^O_{sel}$ cuando los efectos crecen, y por qué $\theta^O_{es}$ (`dynamic`) queda aún más
#    arriba? (¿Qué cohorte identifica por sí sola los tiempos de evento largos?) Los tres son
#    estimandos válidos que responden preguntas distintas; el TWFE no es ninguno de ellos.
# 3. *Avanzado.* Elimine el grupo nunca tratado con
#    `simulate_dynamic(slope, cohorts=(5, 8, 11))`. (a) Llame a `callaway_santanna` con el control
#    por defecto, lea el error y cambie a `control="not_yet_treated"`. (b) Calcule los pesos
#    implícitos del TWFE sobre las observaciones tratadas por Frisch-Waugh-Lovell: quite a `D` las
#    medias por unidad y por tiempo para obtener $\tilde D$ y defina
#    $w_{it} = \tilde D_{it} D_{it} / \sum \tilde D D$. Verifique con aserciones que los pesos suman 1,
#    que `(w * tau).sum()` es igual a `twfe(p, "tau")` y que algunas celdas tratadas reciben peso
#    negativo. ¿Qué cohorte y qué tiempos de evento son, y por qué se penalizan las celdas de
#    exposición larga de la cohorte más temprana? (c) Prediga y luego recorra la pendiente en 0,
#    0.4 y 2.5; en 2.5 verifique que el TWFE sin ruido es negativo aunque ningún efecto plantado
#    sea menor que 1. (d) ¿Qué celdas post-tratamiento puede identificar CS con controles aún no tratados (la
#    cohorte 5 hasta $e=5$, la cohorte 8 hasta $e=2$, la cohorte 11 ninguna)? Compruebe que, con
#    $y=\tau$, su `att_overall` es igual a la media ponderada por tamaño de cohorte de los promedios
#    por cohorte de las celdas identificadas, y compárelo con el objetivo del panel completo. Esa
#    brecha es lo que cuesta no tener un grupo de control limpio.
#
# **¿Qué tan exhaustivo es esto?** `puremacro.did` es un conjunto completo de herramientas modernas
# de DiD escalonado: junto a `callaway_santanna` (con `aggregation=` "group", "simple", "dynamic" o
# "calendar") y `sun_abraham` incluye el estimador por imputación `borusyak_jaravel_spiess` (2024),
# el estimador de conmutadores `cdh_did` (de Chaisemartin-D'Haultfoeuille 2020), las cotas de
# sensibilidad `honest_did` (pase `sigma=cs.event_study_vcov` para usar la covarianza bootstrap
# completa) y `synthetic_did` / `sdid_multi_cohort` (Arkhangelsky et al. 2021) — todos sobre el mismo
# panel largo `(unit, time, outcome, treat_time)`. El cuaderno 15 llega al mismo estudio de eventos
# con proyecciones locales (LP-DiD), y el cuaderno 29 cubre DiD sintético con una sola unidad tratada.
