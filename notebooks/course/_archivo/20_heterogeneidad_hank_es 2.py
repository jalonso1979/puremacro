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
# # Módulo 12 (semana 11) — Heterogeneidad y HANK: por qué el agente representativo pierde el mecanismo
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Resolver el modelo de Aiyagari (1994)** de mercados incompletos en equilibrio
#    estacionario con `puremacro.vfi` y leer la **distribución de riqueza** que produce.
# 2. Calcular la **propensión marginal a consumir (MPC)** hogar por hogar y ver que buena
#    parte de la población vive **al día ("mano a boca")** con MPC cercana a 1.
# 3. **Confrontar** un hecho fiscal empírico —el multiplicador fiscal de los choques
#    narrativos de Romer–Romer— con la predicción del **agente representativo**, que con MPC ≈ 0
#    casi no lo genera.
# 4. Entender cómo **HANK** (Kaplan–Moll–Violante 2018; Auclert 2019) racionaliza el hecho:
#    las MPC altas de los hogares restringidos reactivan el **mecanismo** que el agente
#    representativo promedia y pierde.
#
# Todo corre en Python puro (navegador/iPad, $0), leyendo datos locales del bundle.

# %% slideshow={"slide_type": "skip"}
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
_cwd = pathlib.Path.cwd()
_nb = _cwd if (_cwd / "_nbstyle.py").exists() else _cwd.parent
sys.path.insert(0, str(_nb)); sys.path.insert(0, str(_nb / "course"))
import _nbstyle; _nbstyle.apply_style()
from _tutor import tutor
DATA = (_nb / "course" / "data")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. El gancho: un hecho fiscal que el agente representativo no puede generar
#
# El caballito de batalla del RBC/DSGE es el **agente representativo (RA)**: un solo hogar
# de vida infinita, con mercados completos, que **suaviza el consumo**. En ese mundo una
# transferencia de suma fija transitoria casi no mueve el consumo agregado: por la renta
# permanente (o la equivalencia ricardiana), la MPC de un peso extra es apenas la anualidad
# $\;r/(1+r)\approx 0$ — **casi todo se ahorra**. En consecuencia, los multiplicadores
# fiscales del RA rara vez superan 1.
#
# Los **datos** dicen otra cosa. Retomemos los choques fiscales **narrativos** de
# Romer–Romer y su efecto sobre el PIB real, con una
# **proyección local** de Jordà (`puremacro.lp.lp_hac`).

# %% slideshow={"slide_type": "fragment"}
from puremacro.lp import lp_hac

tax = pd.read_csv(DATA / "tax14_narrative_tax_shocks.csv").rename(columns={"date": "q"})
gdp = pd.read_csv(DATA / "GDPC1.csv")
gdp["q"] = pd.PeriodIndex(pd.to_datetime(gdp["observation_date"]), freq="Q").astype(str)
gdp = gdp.rename(columns={"GDPC1": "gdp"})[["q", "gdp"]]

fis = tax.merge(gdp, on="q", how="inner").sort_values("q").reset_index(drop=True)
fis["y"] = 100.0 * np.log(fis["gdp"])                     # log PIB real, en %
irf = lp_hac(fis, y="y", x="rr_exog", horizons=range(0, 17), n_lags=4)  # respuesta a +1 pp de impuesto

peak = irf.loc[irf["beta"].idxmin()]
print(f"muestra: {fis['q'].iloc[0]}–{fis['q'].iloc[-1]}  ({len(fis)} trimestres)")
print(f"efecto máximo sobre el PIB: {peak['beta']:.2f}%  en h={int(peak['h'])}  (t={peak['t']:.1f})")
assert peak["beta"] < -1.0        # el PIB cae >1% tras un alza de impuestos: multiplicador grande

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El hecho, en una figura
# Un aumento exógeno de impuestos de 1 punto **contrae el PIB varios puntos** y de forma
# **persistente** (efecto significativo alrededor del horizonte 9–10). Un multiplicador
# fiscal de esta magnitud es difícil de reconciliar con un hogar que ahorra casi toda su
# renta transitoria. **Ese es el mecanismo que buscamos.**

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
fig, ax = plt.subplots(figsize=(7.2, 3.7))
ax.fill_between(irf["h"], irf["lo"], irf["hi"], color="0.85", label="IC 90%")
ax.plot(irf["h"], irf["beta"], color=cols[0], lw=1.8, marker="o", ms=3, label="Respuesta del PIB")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("trimestres tras el choque fiscal")
ax.set_ylabel("PIB real (%)")
ax.set_title("Dato: respuesta del PIB a un alza de impuestos (Romer–Romer, LP)")
ax.legend(loc="lower left")
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Teoría breve: el modelo de Aiyagari (1994)
#
# Aiyagari sustituye al agente representativo por un **continuo de hogares** que enfrentan:
# - **Riesgo idiosincrático** de productividad laboral $z$, un AR(1) (aquí discretizado con
#   Tauchen): a veces te va bien, a veces mal, y **no puedes asegurarte** contra ello.
# - Un **límite de endeudamiento** $a \ge 0$: no puedes pedir prestado indefinidamente.
#
# El hogar resuelve $\;\max\; \mathbb{E}\sum_t \beta^t u(c_t)\;$ sujeto a
# $\;c_t + a_{t+1} = w\,e^{z_t} + (1+r)\,a_t,\; a_{t+1}\ge 0$. Con mercados incompletos
# aparece el **ahorro precautorio**: los hogares acumulan un colchón de activos para
# amortiguar los golpes de ingreso. En equilibrio, la tasa $r$ iguala la oferta de capital
# de los hogares con la demanda de una empresa Cobb–Douglas.
#
# La consecuencia clave: el modelo genera una **distribución de riqueza no degenerada** y,
# con ella, una **distribución de MPC**. Los hogares pobres, pegados a la restricción,
# tienen MPC ≈ 1; los ricos, con colchón, tienen MPC ≈ 0. El agente representativo colapsa
# todo eso a un **promedio** — y ahí pierde el mecanismo.
#
# `puremacro.vfi` trae este modelo listo en `aiyagari_steady_state` (iteración de la función
# de valor + distribución estacionaria + equilibrio general, en NumPy puro).

# %% slideshow={"slide_type": "fragment"}
from puremacro.vfi.examples import aiyagari_steady_state
from puremacro.vfi import evaluate_on_grid, weighted_quantile

res = aiyagari_steady_state(n_z=5, n_a=150, a_max=80.0)   # GE estacionario de Aiyagari
eq, sol, mu, prob = res["equilibrium"], res["equilibrium"].solution, res["equilibrium"].distribution, res["equilibrium"].problem
a_grid = np.asarray(prob.a_grid); z_grid = np.asarray(prob.z_grid)
r, w = res["r"], res["w"]

print(f"tasa de equilibrio r = {r:.3f}   salario w = {w:.3f}")
print(f"capital K = {res['K']:.2f}   producto Y = {res['Y']:.2f}   Gini de riqueza = {res['wealth_gini']:.3f}")
assert res["wealth_gini"] > 0.4        # la riqueza está claramente dispersa

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Riqueza y MPC hogar por hogar
# Con la política óptima ya resuelta, reconstruimos el **consumo** en cada estado con
# `evaluate_on_grid` (integrador de `puremacro.vfi`). La **MPC** la medimos como la fracción
# de una transferencia de suma fija $\tau$ (10% del ingreso medio) que el hogar gasta:
# $\;\text{MPC}(a,z) = [\,c(a+\tau/(1+r),z) - c(a,z)\,]/\tau$. La transferencia entra al
# efectivo disponible; interpolamos la política de consumo sobre la malla de activos con
# `np.interp` (el único paso fuera de `puremacro`: la VFI discreta no expone un interpolante
# continuo — `puremacro.vfi.egm` sí lo daría, pero este ejemplo usa VFI discreta).

# %% slideshow={"slide_type": "fragment"}
def cons_fn(ap, a, z, r_, w_, xp=np):            # consumo realizado: c = w e^z + (1+r)a - a'
    return w_ * xp.exp(z) + (1.0 + r_) * a - ap

C = evaluate_on_grid(cons_fn, sol.policy_aprime, a_grid, z_grid, params=prob.params)  # (n_a, n_z)

tau = 0.10 * w * res["L"]                          # transferencia = 10% del ingreso laboral medio
mpc = np.empty_like(C)
for j in range(z_grid.size):
    c_up = np.interp(a_grid + tau / (1.0 + r), a_grid, C[:, j])   # consumo con el efectivo extra
    mpc[:, j] = (c_up - C[:, j]) / tau

avg_mpc = float((mu * mpc).sum())                 # MPC promedio ponderada por la población
ra_mpc = r / (1.0 + r)                            # MPC del agente representativo (anualidad)
share_hi = float(mu[mpc > 0.3].sum())             # fracción con MPC alta (mano a boca)

print(f"MPC promedio (heterogénea) = {avg_mpc:.2f}")
print(f"MPC del agente representativo r/(1+r) = {ra_mpc:.3f}")
print(f"fracción de hogares con MPC > 0.3      = {share_hi:.2f}")
assert avg_mpc > 5 * ra_mpc        # el promedio heterogéneo es MUCHÍSIMO mayor que el del RA

# %% slideshow={"slide_type": "slide"}
mu_a = mu.sum(axis=1)                              # distribución marginal de riqueza
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))

ax[0].fill_between(a_grid, mu_a, step="mid", color="0.75", lw=0)
ax[0].plot(a_grid, mu_a, drawstyle="steps-mid", color="0.15", lw=1.2)
ax[0].axvline(res["K"], color=cols[0], lw=1.2, ls=(0, (4, 2)), label=f"media (K={res['K']:.1f})")
ax[0].set_xlim(0, 40); ax[0].set_xlabel("riqueza  $a$"); ax[0].set_ylabel("masa de hogares")
ax[0].set_title("Distribución de riqueza"); ax[0].legend(loc="upper right")

wl = mpc.reshape(-1); ww = mu.reshape(-1)
ax[1].hist(wl, bins=np.linspace(0, 1, 26), weights=ww, color="0.75", edgecolor="0.3")
ax[1].axvline(ra_mpc, color=cols[0], lw=1.6, ls=(0, (4, 2)), label=f"MPC del RA ≈ {ra_mpc:.02f}")
ax[1].axvline(avg_mpc, color="0.0", lw=1.6, label=f"MPC media HA = {avg_mpc:.2f}")
ax[1].set_xlabel("MPC del hogar"); ax[1].set_ylabel("masa de hogares")
ax[1].set_title("Distribución de la MPC"); ax[1].legend(loc="upper center")
fig.suptitle("Aiyagari: mucha gente pobre, con MPC alta (mano a boca)", y=1.02)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lectura
# La izquierda muestra una montaña de masa **pegada a la restricción** $a\approx 0$: hogares
# sin colchón. La derecha traduce eso a comportamiento: una parte grande de la población
# tiene **MPC alta**, muy lejos del punto casi cero del agente representativo (línea
# discontinua). El promedio poblacional (línea negra) es del orden de $0.2$–$0.3$: un peso
# de transferencia mueve el consumo agregado **diez veces más** que en el RA.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. El experimento de transferencia: RA vs. heterogeneidad
#
# Repartamos una transferencia de suma fija idéntica a todos y midamos cuánto **consumo
# agregado** genera (el efecto directo de primera ronda, à la Auclert 2019). En el RA se
# ahorra casi todo; con heterogeneidad, **los hogares de mano a boca lo gastan** — y son
# tantos que la respuesta agregada se dispara. Además, veamos **quién** hace ese gasto.

# %% slideshow={"slide_type": "fragment"}
mpc_a = (mu * mpc).sum(axis=1) / np.where(mu_a > 0, mu_a, 1.0)   # E[MPC | a]: MPC media por riqueza
contrib_a = mu_a * mpc_a                                          # aporte de cada nivel de riqueza (suma = avg_mpc)

med = float(weighted_quantile(mu.reshape(-1),
                              np.broadcast_to(a_grid[:, None], mu.shape).reshape(-1), [0.5])[0])
share_bottom = float(contrib_a[a_grid <= med].sum() / contrib_a.sum())   # aporte de la mitad más pobre

print(f"riqueza mediana = {med:.2f}")
print(f"respuesta del consumo agregado a la transferencia:  RA = {ra_mpc:.02f}   HA = {avg_mpc:.2f}")
print(f"de esa respuesta HA, la mitad MÁS POBRE aporta el {100*share_bottom:.0f}%")

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))

ax[0].plot(a_grid, mpc_a, color="0.15", lw=1.8)
ax[0].axvspan(0, med, color="0.9", label="mitad más pobre")
ax[0].axhline(ra_mpc, color=cols[0], lw=1.4, ls=(0, (4, 2)), label=f"MPC del RA ≈ {ra_mpc:.02f}")
ax[0].set_xlim(0, 40); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("riqueza  $a$"); ax[0].set_ylabel("MPC media")
ax[0].set_title("La MPC cae con la riqueza"); ax[0].legend(loc="upper right")

bars = ax[1].bar(["Agente\nrepresentativo", "Heterogéneo\n(Aiyagari)"],
                 [ra_mpc, avg_mpc], color=["0.7", "0.2"], width=0.6)
ax[1].bar_label(bars, fmt="%.2f", padding=3)
ax[1].set_ylabel("Δ consumo por peso transferido")
ax[1].set_ylim(0, max(avg_mpc * 1.25, 0.35))
ax[1].set_title("Respuesta a una transferencia de suma fija")
fig.suptitle("En RA casi todo se ahorra; con heterogeneidad, los pobres lo gastan", y=1.02)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La moraleja HANK
# Kaplan, Moll y Violante (2018) muestran que este es **el** canal que le falta al DSGE de
# agente representativo: al reintroducir hogares heterogéneos con MPC altas, la política
# fiscal (y la monetaria, vía ingreso) recupera fuerza. Auclert (2019) formaliza que el
# efecto agregado de una transferencia es esencialmente la **MPC promedio** ponderada por
# quién la recibe. El Aiyagari básico *subestima* aún la MPC y la cola rica; la versión de
# dos activos de KMV (riqueza ilíquida → "ricos de mano a boca") la acerca a los datos.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Preguntas para pensar
# 1. **¿Quién absorbe una recesión?** En este modelo, ¿qué hogares recortan más su consumo
#    cuando cae el ingreso: los de la montaña en $a\approx 0$ o los de la cola rica? ¿Por
#    qué la desigualdad de riqueza convierte a una recesión agregada en un golpe muy
#    desigual?
# 2. El agente representativo *no está equivocado sobre el promedio* de la riqueza: acierta
#    en $K$. ¿Por qué entonces **falla** en el multiplicador fiscal? ¿Qué momento de la
#    distribución —no la media— es el que importa para el mecanismo?
# 3. Una transferencia **más grande** (digamos 50% del ingreso) tendría una MPC media
#    **menor** que la de una pequeña. ¿Por qué? (Pista: piensa en el hogar que, con el
#    dinero extra, logra despegarse de la restricción $a\ge 0$.)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 5. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "En una frase, ¿por qué la MPC promedio importa para el multiplicador fiscal, y por qué
#   el agente representativo la subestima?"
# - "¿Qué agrega el modelo de dos activos de Kaplan–Moll–Violante frente al Aiyagari de un
#   activo para explicar las MPC altas?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase, ¿por qué el modelo de agente representativo subestima el "
            "multiplicador fiscal frente a un modelo con hogares heterogéneos (HANK)?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Partimos de un **hecho** fiscal en datos reales (el PIB cae varios puntos tras
# un alza de impuestos narrativa — multiplicador grande) que el **agente representativo**,
# con MPC ≈ 0, no puede generar. Resolvimos el **Aiyagari** con `puremacro.vfi`, vimos su
# **distribución de riqueza** y su **distribución de MPC** —muchos hogares de mano a boca— y
# mostramos, con un experimento de **transferencia**, que la respuesta del consumo agregado
# es ~10× la del RA porque los pobres gastan. Ese es el **mecanismo de HANK**
# (Aiyagari 1994; Kaplan–Moll–Violante 2018; Auclert 2019): la heterogeneidad no es un
# detalle contable, es *el* canal.
#
# ### Referencias
# - Aiyagari, S. R. (1994). *Uninsured Idiosyncratic Risk and Aggregate Saving.* QJE.
# - Kaplan, G., Moll, B. y Violante, G. (2018). *Monetary Policy According to HANK.* AER.
# - Auclert, A. (2019). *Monetary Policy and the Redistribution Channel.* AER.
