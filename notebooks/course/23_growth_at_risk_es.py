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
# # Módulo (electivo) — Growth-at-Risk: la distribución del PIB, no solo su media, y la humildad del dato en tiempo real
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Confrontar** el modelo estándar de agente representativo —que reduce el pronóstico a
#    una **media** con varianza constante y simétrica— con un hecho que no explica: la
#    **distribución condicional** del crecimiento del PIB es **asimétrica y variable**.
# 2. Estimar **Growth-at-Risk** con `puremacro.gar`: una **autorregresión cuantílica** (`qar`)
#    del crecimiento del PIB con las **condiciones financieras** como control.
# 3. Ajustar una **skew-t** a los cuantiles predichos (`fit_skewt_to_quantiles`, `skewt_pdf`)
#    y leer la **densidad condicional**: la cola izquierda **engorda** cuando aprietan las
#    condiciones financieras (Adrian, Boyarchenko y Giannone, 2019).
# 4. Interiorizar la **humildad del dato en tiempo real** (Orphanides): el riesgo se evalúa
#    hoy con el dato equivocado — el PIB de 2008Q4 se vio **−3.8 %** y hoy es **−8.5 %**.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), leyendo datos locales del bundle: sin conexión y sin costo.

# %% slideshow={"slide_type": "skip"}
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
try:  # bajo Jupyter/ipykernel: conserva el backend inline (captura figuras)
    get_ipython()
except NameError:
    matplotlib.use("Agg")  # script plano / CLI: backend no interactivo
import matplotlib.pyplot as plt
_cwd = pathlib.Path.cwd()
_nb = _cwd if (_cwd / "_nbstyle.py").exists() else _cwd.parent
sys.path.insert(0, str(_nb)); sys.path.insert(0, str(_nb / "course"))
import _nbstyle; _nbstyle.apply_style()
from _tutor import tutor
DATA = (_nb / "course" / "data")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. La confrontación: la media no es el riesgo
#
# El modelo neoclásico/RBC con **agente representativo** escribe el producto como una
# tendencia más un choque:
#
# $$\log y_{t+1} = \mu_t + \varepsilon_{t+1}, \qquad \varepsilon_{t+1}\sim\mathcal{N}(0,\sigma^2).$$
#
# Con innovaciones **gaussianas** y varianza **constante**, la densidad predictiva es
# **simétrica** y su forma no cambia con el estado de la economía: el **pronóstico puntual**
# $\mu_t$ (la media) es un estadístico suficiente. Bajo esa vista, "riesgo a la baja" y
# "riesgo al alza" son lo mismo.
#
# **El hecho que no explica** (Adrian, Boyarchenko y Giannone, 2019, *Vulnerable Growth*):
# la distribución condicional del crecimiento del PIB **no es simétrica ni fija**. Cuando las
# **condiciones financieras se aprietan**, la **cola izquierda se engorda** —el escenario malo
# empeora mucho— mientras la **cola derecha casi no se mueve**. El centro de la distribución
# se desplaza, sí, pero **mucho menos que la cola**: abajo estimaremos que un barrido completo
# de la NFCI hunde el cuantil 5 % unos **10 puntos** y la mediana solo **2.5**. El **riesgo a
# la baja** se dispara sin que el pronóstico puntual dé la alarma en la misma medida: el
# resumen por la media pierde justo lo que le importa a la política.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Growth-at-Risk con `puremacro.gar`
#
# **Ingredientes** (Python puro):
# - `A191RL1Q225SBEA.csv` — crecimiento del PIB real de EE. UU., trimestral **anualizado (%)**.
# - `NFCI.csv` — índice de condiciones financieras nacionales de la Fed de Chicago (semanal;
#   **más alto = más apretado**), que agregamos a trimestral.
#
# **Ficha de medición** (el curso no publica un momento condicional sin ella):
#
# | campo | valor |
# |---|---|
# | fuente/serie | FRED `A191RL1Q225SBEA` (PIB real, tasa trimestral **anualizada**, %) y FRED `NFCI` |
# | muestra de estimación | **1971Q1–2019Q4** (n = 196); el PIB arranca en 1947 y la NFCI en 1971 — hay que **alinear antes de regresar** |
# | agregación de la NFCI | semanal → media trimestral (`resample("QS")`), fechada al inicio del trimestre |
# | especificación | QAR con p = 4 rezagos de $g$, NFCI$_t$ como control, horizonte h = 1 |
# | corte muestral | **excluye 2020 en adelante**: 2020Q2 (−28 %) y 2020Q3 (+35 %) son dos observaciones que, en una regresión cuantílica, *dominan* las colas (§2.3 lo muestra) |
# | edición/vintage | serie **revisada** de hoy (CSV congelado del bundle); §3 la confronta con las primeras publicaciones |
#
# **Receta ABG-2019** en dos pasos, ambos en `puremacro.gar`:
# 1. `qar` — autorregresión cuantílica: para cada cuantil $\tau$ estima
#    $Q_\tau[g_{t+1}\mid g_t,\dots,g_{t-3},\ \text{NFCI}_t]=\alpha_\tau+\sum_l\beta_{\tau,l}\,g_{t-l}+\gamma_\tau\,\text{NFCI}_t$.
# 2. `fit_skewt_to_quantiles` — ajusta una **skew-t** de Azzalini $(\mu,\sigma,\alpha,\nu)$ a
#    los cuantiles predichos; `skewt_pdf` da la **densidad condicional** suave.

# %% slideshow={"slide_type": "fragment"}
from puremacro.gar import qar, fit_skewt_to_quantiles, skewt_pdf

# Crecimiento del PIB (anualizado, %), trimestral.
g = (pd.read_csv(DATA / "A191RL1Q225SBEA.csv", parse_dates=["observation_date"])
       .rename(columns={"A191RL1Q225SBEA": "gdp"})
       .set_index("observation_date")["gdp"])

# NFCI semanal -> media trimestral (inicio de trimestre, alineado con el PIB).
nfci_q = (pd.read_csv(DATA / "NFCI.csv", parse_dates=["observation_date"])
            .set_index("observation_date")["NFCI"]
            .resample("QS").mean())

# Alinear PRIMERO (el PIB arranca en 1947; la NFCI en 1971): `qar` apila por
# posición, así que sin esta intersección el control iría desfasado 24 años.
panel_full = pd.concat([g, nfci_q.rename("nfci")], axis=1).dropna()
panel = panel_full.loc[:"2019-10-01"]        # corte pre-COVID (ver ficha)
def _qlabel(ts):
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
print(f"Datos alineados:      {_qlabel(panel_full.index.min())}"
      f" -> {_qlabel(panel_full.index.max())}  (n = {len(panel_full)})")
print(f"Muestra de estimación: {_qlabel(panel.index.min())}"
      f" -> {_qlabel(panel.index.max())}  (n = {len(panel)})")
print(f"NFCI  2005Q2 (tranquilo) = {panel.loc['2005-04-01','nfci']:+.2f}   "
      f"2008Q4 (crisis) = {panel.loc['2008-10-01','nfci']:+.2f}")

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El motor: `qar` con NFCI como control
#
# Estimamos los cuantiles de $g_{t+1}$ (horizonte $h=1$) con 4 rezagos y la NFCI como control.
# La firma `qar(...)` devuelve, por cada $(h,\tau)$, los coeficientes `beta0…beta5`
# (intercepto, 4 rezagos, y el coeficiente de la NFCI en `beta5`).
#
# **Convención de fechado** (importante para reconstruir a mano el predictor): el bloque
# autorregresivo de `qar` empieza en el valor **contemporáneo**, así que
# `beta1`$=g_t$, `beta2`$=g_{t-1}$, `beta3`$=g_{t-2}$, `beta4`$=g_{t-3}$ y `beta5`$=$NFCI$_t$.

# %% slideshow={"slide_type": "fragment"}
taus = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
fit = qar(panel["gdp"].values, quantiles=taus, horizons=(1,), p=4,
          controls=panel["nfci"].values, n_boot=100, seed=1)
Q = fit[fit["h"] == 1].set_index("tau")
betacols = ["beta0", "beta1", "beta2", "beta3", "beta4", "beta5"]

# El corazón del hecho ABG: el coeficiente de la NFCI (beta5) por cuantil.
print("  tau   coef. NFCI (beta5)")
for t in taus:
    print(f"  {t:.2f}      {Q.loc[t, 'beta5']:+.2f}")
gamma_lo, gamma_hi = Q.loc[0.05, "beta5"], Q.loc[0.95, "beta5"]
print(f"\nApretar condiciones golpea la cola IZQUIERDA (tau=.05: {gamma_lo:+.2f}) "
      f"mucho más que la derecha (tau=.95: {gamma_hi:+.2f}).")
assert gamma_lo < gamma_hi                    # la asimetría de ABG (2019)
assert abs(gamma_lo) > 5 * abs(gamma_hi)      # y es una asimetría GRANDE

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 2.3 Por qué la muestra se corta en 2019Q4 (y por qué hay que decirlo)
#
# Una regresión **cuantílica** de cola estima con muy pocos puntos: el 5 % de 196 trimestres
# son ~10 observaciones. Meter 2020 basta para desfigurarlas: en 2020Q2 el PIB cae **−28 %**
# con la NFCI en −0.08 —su lectura más alta desde 2012, por encima del 64 % de la muestra
# 1971–2019— y al trimestre siguiente **rebota +35 %**. Esa pareja (condiciones relativamente
# apretadas hoy, crecimiento enorme mañana) es exactamente lo contrario del hecho de ABG, y
# con dos observaciones de tanta palanca **cuadruplica** el coeficiente de la NFCI en el
# cuantil 95 %: apretar condiciones parecería *subir* la cola buena. No es un hecho económico;
# es el cierre y la reapertura de la economía. Lo mostramos en vez de esconderlo: la elección
# de muestra **es** un resultado que hay que reportar.

# %% slideshow={"slide_type": "fragment"}
fit_covid = qar(panel_full["gdp"].values, quantiles=taus, horizons=(1,), p=4,
                controls=panel_full["nfci"].values, n_boot=20, seed=1)
Qc = fit_covid[fit_covid["h"] == 1].set_index("tau")
print("  tau    beta5 (hasta 2019Q4)   beta5 (muestra completa, con 2020)")
for t in taus:
    print(f"  {t:.2f}        {Q.loc[t,'beta5']:+.2f}                  {Qc.loc[t,'beta5']:+.2f}")
print(f"\nCon 2020 dentro, beta5 del cuantil 95 % pasa de {Q.loc[0.95,'beta5']:+.2f} a "
      f"{Qc.loc[0.95,'beta5']:+.2f}: apretar condiciones 'mejoraría' el mejor escenario.\n"
      "Eso es el rebote mecánico de 2020Q3, no vulnerable growth.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### De cuantiles a densidad: dos estados de la economía
#
# Evaluamos la recta cuantílica ajustada, $\hat Q_\tau(x)=x^\top\hat\beta_\tau$, en dos
# **estados históricos reales** —el predictor $x=[1,\,g_t,\dots,g_{t-3},\,\text{NFCI}_t]$— y
# ajustamos una skew-t a cada colección de cuantiles:
# - **2005Q2**: trimestre tranquilo previo a la crisis (condiciones **flojas**, NFCI −0.61).
# - **2008Q4**: pánico financiero (condiciones **apretadas**, NFCI +2.77).
#
# **Cuidado con la lectura:** entre estos dos estados cambian **dos cosas a la vez**, la NFCI
# y los rezagos del PIB. La comparación mezcla el canal financiero con la inercia
# autorregresiva; la Figura 2 aislará el canal financiero fijando los rezagos.

# %% slideshow={"slide_type": "fragment"}
def predictor_row(date):
    """x = [1, g_t, g_{t-1}, g_{t-2}, g_{t-3}, NFCI_t] en la fecha dada."""
    i = panel.index.get_loc(pd.Timestamp(date))
    lags = [panel["gdp"].values[i - k] for k in range(4)]
    return np.array([1.0, *lags, panel["nfci"].values[i]])

def cond_quantiles(x):
    return {float(t): float(x @ Q.loc[t, betacols].values.astype(float)) for t in taus}

x_calm,  x_tight  = predictor_row("2005-04-01"), predictor_row("2008-10-01")
qc, qt = cond_quantiles(x_calm), cond_quantiles(x_tight)
F_calm,  F_tight  = fit_skewt_to_quantiles(qc), fit_skewt_to_quantiles(qt)

for lab, F in (("2005Q2 tranquilo", F_calm), ("2008Q4 crisis", F_tight)):
    print(f"{lab:>16}:  GaR 5% = {F.ppf(0.05):+6.1f}%   "
          f"shortfall esperado = {F.expected_shortfall(0.05):+6.1f}%   "
          f"mediana = {F.ppf(0.50):+.1f}%")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 1 — La densidad condicional del crecimiento
# La misma economía, dos estados. La densidad "tranquila" es **angosta y casi simétrica**
# ($\hat\alpha\approx-0.1$, $\hat\sigma\approx1.9$); la de **2008Q4** se **ensancha y se
# tuerce a la izquierda** ($\hat\alpha\approx-1.9$, $\hat\sigma\approx7.0$). La mediana cae
# unos **6 puntos**, pero el **5 % GaR cae el doble**: de ≈ +0.2 % a ≈ −11.9 %. Eso es lo que
# la **media** —el punto— no ve.

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
xg = np.linspace(-22, 16, 500)
fig, ax = plt.subplots(figsize=(7.4, 3.9))
for (lab, F, c, ls) in (("2005Q2 (condiciones flojas)", F_calm, cols[0], "-"),
                        ("2008Q4 (condiciones apretadas)", F_tight, "0.45", (0, (4, 2)))):
    pdf = skewt_pdf(xg, F.mu, F.sigma, F.alpha, F.nu)
    ax.plot(xg, pdf, color=c, lw=1.8, ls=ls, label=lab)
    gar = F.ppf(0.05)
    ax.axvline(gar, color=c, lw=1.0, ls=":")
    ax.fill_between(xg, 0, pdf, where=(xg <= gar), color=c, alpha=0.18)
ax.axvline(0, color="0.85", lw=0.6)
ax.set_xlabel("crecimiento del PIB al trimestre siguiente (anualizado, %)")
ax.set_ylabel("densidad")
ax.set_title("Densidad predictiva condicional del PIB: tranquilo vs. 2008Q4\n"
             "(sombra = cola de 5 % Growth-at-Risk)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Figura 2 — El abanico de "vulnerable growth"
# Aislamos el canal financiero: fijamos los rezagos en su media muestral y **barremos la
# NFCI**. Los cuantiles 5 %, 50 % y 95 % de $g_{t+1}$ dibujan un **abanico que se abre hacia
# abajo**. De un extremo al otro del barrido (NFCI de −0.9 a +3.2, los percentiles 2 y 99
# de la muestra):
#
# | cuantil | NFCI = −0.9 | NFCI = +3.2 | cambio |
# |---|---|---|---|
# | 5 % (GaR) | +1.2 % | −8.5 % | **−9.7 pp** |
# | 50 % (mediana) | +3.4 % | +0.8 % | −2.5 pp |
# | 95 % | +7.2 % | +8.0 % | +0.8 pp |
#
# El 5 % se **hunde** casi diez puntos; la mediana cae **cuatro veces menos**; el 95 %
# prácticamente no se mueve. La asimetría de ABG (2019), en una imagen — y ojo: es la
# **mediana**, no la media, la que cae poco (con colas tan gruesas la media condicional no es
# un buen resumen).

# %% slideshow={"slide_type": "slide"}
lag_mean = panel["gdp"].mean()
nfci_grid = np.linspace(panel["nfci"].quantile(0.02), panel["nfci"].quantile(0.99), 40)
fan = {t: np.array([np.array([1.0, lag_mean, lag_mean, lag_mean, lag_mean, z])
                    @ Q.loc[t, betacols].values.astype(float) for z in nfci_grid])
       for t in (0.05, 0.50, 0.95)}

fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.fill_between(nfci_grid, fan[0.05], fan[0.95], color="0.88", label="banda 5–95 %")
ax.plot(nfci_grid, fan[0.95], color="0.55", lw=1.4, ls=(0, (1, 1)), label="cuantil 95 %")
ax.plot(nfci_grid, fan[0.50], color="0.25", lw=1.6, label="mediana")
ax.plot(nfci_grid, fan[0.05], color="0.00", lw=1.8, label="cuantil 5 % (GaR)")
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel("condiciones financieras  NFCI  (→ más apretadas)")
ax.set_ylabel("crecimiento $g_{t+1}$ (anualizado, %)")
ax.set_title("Vulnerable growth: la cola izquierda se hunde cuando aprietan\n"
             "las condiciones financieras; la derecha casi no se mueve")
ax.legend(loc="lower left", ncol=2)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. La humildad del dato en tiempo real (Orphanides)
#
# Toda esa evaluación de riesgo se hace **en tiempo real, con el dato de hoy** — y el dato de
# hoy es **provisional**. Orphanides (2001) mostró que reglas y diagnósticos calculados con
# datos revisados **no son los que se pudieron calcular en el momento**. El caso emblemático
# es 2008Q4: leemos las **primeras publicaciones** del BEA y las comparamos con el valor
# **actual** (vintage revisado que cargamos arriba).

# %% slideshow={"slide_type": "fragment"}
vint = pd.read_csv(DATA / "alfred_pib_primeras_publicaciones.csv",
                   parse_dates=["release_date"])
v08 = vint[vint["quarter"] == "2008Q4"].sort_values("release_date")
actual_08 = float(g.loc["2008-10-01"])       # vintage actual, del mismo archivo del PIB

print("2008Q4 — cómo se vio el PIB con el paso del tiempo:")
for _, r in v08.iterrows():
    print(f"  {r['release_date']:%Y-%m-%d}  ({r['vintage']:>7}):  {r['growth_annualized']:+.1f}%")
print(f"  vintage actual (revisado):        {actual_08:+.1f}%")
primera = float(v08.iloc[0]["growth_annualized"])
print(f"\nEn tiempo real se vio {primera:+.1f}%; hoy sabemos que fue {actual_08:+.1f}% "
      f"— más del doble de profundo.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Las dos humildades se juntan: el GaR con el dato de enero de 2009
#
# El predictor de 2008Q4 que usamos en §2 lleva dentro $g_t=-8.5$ — un número que **nadie
# tenía** cuando había que decidir. Rehagamos el mismo GaR sustituyendo ese rezago por el
# dato **de tiempo real** (−3.8 %) y dejando todo lo demás igual.

# %% slideshow={"slide_type": "fragment"}
for lab, g_t in (("con el dato revisado  (-8.5)", actual_08),
                 ("con el dato de enero 2009 (-3.8)", primera)):
    x = x_tight.copy(); x[1] = g_t
    F = fit_skewt_to_quantiles(cond_quantiles(x))
    print(f"{lab:>34}:  GaR 5% = {F.ppf(0.05):+6.1f}%   mediana = {F.ppf(0.50):+5.1f}%")
print("\nCon el dato disponible en el momento, el riesgo se veía ~1 punto menos negro y la\n"
      "mediana ~1.5 puntos mejor. La revisión del PIB no solo corrige la historia: corrige,\n"
      "hacia atrás, el riesgo que creíamos estar corriendo.")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 3 — El mismo trimestre, distintas verdades
# La caída de 2008Q4 se "profundizó" **sin que pasara nada nuevo**: solo llegaron las
# revisiones. Quien evaluó el riesgo en enero de 2009 lo hizo con **−3.8 %**, no con **−8.5 %**.

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.plot(v08["release_date"], v08["growth_annualized"], color="0.00",
        marker="o", lw=1.6, label="publicaciones en tiempo real (BEA)")
ax.axhline(actual_08, color="0.55", lw=1.4, ls=(0, (4, 2)),
           label=f"vintage actual ({actual_08:+.1f}%)")
for _, r in v08.iterrows():
    ax.annotate(f"{r['growth_annualized']:+.1f}%",
                (r["release_date"], r["growth_annualized"]),
                textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")
ax.set_ylabel("PIB 2008Q4 (crec. anualizado, %)")
ax.set_xlabel("fecha de publicación")
ax.set_title("La humildad del dato: el PIB de 2008Q4 en tiempo real vs. hoy")
ax.legend(loc="upper right")
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Síntesis — por qué la media no basta
#
# - El **agente representativo gaussiano** entrega una densidad simétrica y fija: el pronóstico
#   puntual (la **media**) lo resume todo. No hay lugar para riesgo a la baja *asimétrico*.
# - Los datos dicen otra cosa: al apretar las condiciones financieras, la **cola izquierda**
#   del crecimiento se **engorda** (Figs. 1–2). Entre 2005Q2 y 2008Q4 el **5 % GaR** cae de
#   ≈ +0.2 % a ≈ −11.9 %, mientras la mediana cae la mitad (de +3.3 % a −2.6 %); aislando el
#   canal financiero (Fig. 2) la brecha es aún más nítida: −9.7 pp contra −2.5 pp.
#   **La media pierde el riesgo.**
# - Y encima, ese riesgo se evalúa **en tiempo real con el dato equivocado** (Fig. 3):
#   la humildad del dato (Orphanides) se **suma** a la humildad del modelo.
#
# La macro que solo mira la media pierde, por partida doble, lo que de verdad importa: la
# **cola**, y la **incertidumbre sobre el propio dato**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Preguntas para pensar
# 1. **¿De qué sirve un pronóstico puntual si la cola es lo que importa?** Si dos trimestres
#    tienen la **misma mediana** pronosticada pero uno tiene 5 % GaR = 0 % y el otro −12 %,
#    ¿son igual de "buenos" para un banco central? ¿Qué objeto —media o cuantil— debería
#    entrar en la función de pérdida?
# 2. El coeficiente de la NFCI es **muy negativo en el cuantil 5 %** (−2.34) y **casi nulo en
#    el 95 %** (+0.20). ¿Qué mecanismo económico haría que las condiciones financieras muevan
#    la cola mala pero no la buena? (Pista: frenos financieros, ventas forzadas, no linealidad.)
# 4. En §2.3 vimos que dos trimestres de 2020 dan vuelta al coeficiente del cuantil 95 %.
#    ¿Es legítimo excluirlos? ¿Qué regla te impide excluir, además, lo que no te guste?
#    (Pista: la regla se fija **antes** de ver el resultado y se **reporta**, como aquí.)
# 3. Si el PIB de 2008Q4 se vio **−3.8 %** y hoy es **−8.5 %**, ¿cómo cambia eso tu lectura de
#    las decisiones de política de finales de 2008? ¿Es justo evaluarlas con el dato de hoy?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 6. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "En una frase, ¿por qué la distribución del crecimiento del PIB es asimétrica según Adrian, Boyarchenko y Giannone (2019)?"
# - "¿Por qué el problema del dato en tiempo real (Orphanides) empeora la evaluación de riesgo a la baja?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase, ¿por qué la cola izquierda del crecimiento del PIB "
            "se engorda cuando aprietan las condiciones financieras (ABG 2019)?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Estimamos **Growth-at-Risk** con `puremacro.gar` —autorregresión cuantílica
# (`qar`) del PIB de EE. UU. con condiciones financieras, **1971Q1–2019Q4**, y una **skew-t**
# ajustada a los cuantiles (`fit_skewt_to_quantiles`, `skewt_pdf`)— y vimos que la **cola
# izquierda engorda** al apretar las condiciones (5 % GaR de +0.2 % a −11.9 % entre 2005Q2 y
# 2008Q4; Adrian, Boyarchenko y Giannone, 2019): el hecho que el agente
# representativo gaussiano, centrado en la **media**, no explica. Y añadimos la **humildad del
# dato en tiempo real** (Orphanides, 2001): el PIB de 2008Q4 se vio −3.8 % y hoy es −8.5 %.
# La macro que solo mira la media pierde el **riesgo a la baja**.
#
# **Referencias.** Adrian, T., Boyarchenko, N. y Giannone, D. (2019), "Vulnerable Growth",
# *American Economic Review* 109(4), 1263–1289. · Orphanides, A. (2001), "Monetary Policy
# Rules Based on Real-Time Data", *American Economic Review* 91(4), 964–985.
