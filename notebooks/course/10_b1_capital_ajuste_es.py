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
# # Módulo 10 — Utilización variable, costos de ajuste del capital y q de Tobin
#
# **Curso complementario · puremacro · mazo Slides07 (bloque B1) — mecanismos y economía abierta**
#
# El mazo B1 cubre las semanas 13–14; **esta lección es la mitad de la semana 14**
# (capital y choques: utilización, costos de ajuste, $q$ de Tobin, choques específicos a la
# inversión). La otra mitad —economía pequeña y abierta— va en la lección `10b`.
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Explicar la **utilización variable del capital**, por qué es **procíclica** y cómo
#    sesga al residuo de Solow.
# 2. Motivar los **costos de ajuste** de la inversión y derivar la intuición de la
#    **q de Tobin** (marginal vs. promedio, Hayashi 1982).
# 3. Estimar con **proyecciones locales (LP-HAC)** la respuesta de la inversión ante un
#    choque a $\Delta\log q$, y ver por qué la q agregada explica **poco**.
# 4. Interpretar los **choques específicos a la inversión** vía el precio relativo de los
#    bienes de inversión.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), con los datos congelados del *bundle*: sin conexión y sin costo.
#
# > **Dónde se evalúa esto.** La extensión del RBC con **utilización variable** $u_t$ y
# > **costos de ajuste** $\phi$ entra en los **dos** finales: la **derivación analítica** en el
# > **Examen Final Teórico** —Sección 1 (CPO de utilización, Euler con $q$ de Tobin) y
# > Sección 3 (lectura de la salida empírica de $q$), que juntas son el **60%** de ese
# > examen— y su **implementación** en la **Parte D del Examen Final Computacional**. Lo que
# > **no** es es material de la **Tarea 2**, que se queda con el RBC canónico del mazo A4, sin
# > $u_t$ y sin $\phi$. Esta lección prepara para ambas cosas.

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


# %% slideshow={"slide_type": "skip"}
# Helper: leer un CSV estilo FRED del bundle (columna de fechas + columna de la serie).
# Siempre local — nunca por red.
def fred(name: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{name}.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    return df.set_index("observation_date")[name].astype(float)


START = pd.Timestamp("1955-01-01")   # ventana común, trimestral y sin huecos

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. Utilización variable del capital
#
# En el modelo neoclásico básico el capital $K_t$ es **fijo dentro del periodo**, así que
# los *servicios de capital* son proporcionales a $K_t$. En la realidad la empresa decide
# también **cuán intensamente** usa esa maquinaria: la tasa de utilización $u_t$. Los
# servicios efectivos son $u_t\,K_t$, no $K_t$.
#
# - Correr la planta más fuerte (más turnos, menos tiempo muerto) **eleva la producción**
#   sin instalar capital nuevo.
# - El costo: la utilización acelera la **depreciación**, $\delta(u_t)$ con $\delta'>0$.
#   Ese margen relaciona $u_t$ con las condiciones de demanda.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### ¿Por qué importa? El residuo de Solow
# La utilización es **procíclica**: en auges las empresas usan el capital instalado más
# intensamente. Si medimos la PTF (residuo de Solow) usando $K_t$ en vez de $u_t K_t$,
# atribuimos a "tecnología" lo que en realidad es **mayor utilización**. Resultado: el
# residuo de Solow medido **exagera** la ciclicidad de la PTF (Burnside–Eichenbaum–Rebelo;
# Basu–Kimball).
#
# El mazo documenta este hecho con **datos observados**: la utilización de la capacidad de
# EUA (`TCU`, Reserva Federal) es fuertemente procíclica frente a la producción industrial
# (`INDPRO`). Esa serie **no viaja en el *bundle* offline** y el curso corre con la red
# cerrada, así que aquí la **simulamos** anclada al ciclo real del PIB (`GDPC1`) y la
# marcamos como simulada.
#
# > **Qué prueba y qué no prueba lo que sigue.** La correlación que vas a ver es **positiva
# > por construcción**: la fabricamos así. No es evidencia de nada — es una **ilustración**
# > del mecanismo. La evidencia está en `TCU` (mazo B1) y en la PTF ajustada por utilización
# > de Fernald. Tampoco calibramos la volatilidad relativa contra `TCU`: el número que se
# > imprime abajo es el de la simulación, no un momento del dato.

# %% slideshow={"slide_type": "fragment"}
from puremacro.data import hp_filter

gdp = fred("GDPC1").dropna()
gdp = gdp[gdp.index >= START]
log_gdp = 100.0 * np.log(gdp.values)             # PIB real en log-puntos
gdp_cycle, _ = hp_filter(log_gdp)                 # (ciclo, tendencia)
gdp_cycle = np.asarray(gdp_cycle)

# SERIE SIMULADA (declarada): utilización procíclica alrededor de 80%, anclada al ciclo
# real del PIB más ruido idiosincrático. No es un dato observado.
rng = np.random.default_rng(20260721)
util_sim = 80.0 + 0.9 * gdp_cycle + 0.5 * rng.standard_normal(gdp_cycle.size)
util_cycle, _ = hp_filter(util_sim)
util_cycle = np.asarray(util_cycle)

corr_util = np.corrcoef(util_cycle, gdp_cycle)[0, 1]
print(f"corr(ciclo utilización SIMULADA, ciclo PIB) = {corr_util:.2f}  [por construcción]")
print(f"volatilidad relativa  sd(u)/sd(PIB)         = {util_cycle.std()/gdp_cycle.std():.2f}"
      "  [de la simulación, NO calibrada contra TCU]")
assert corr_util > 0.5                            # procíclica por construcción

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
q_idx = gdp.index
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(q_idx, gdp_cycle, color=cols[0], lw=1.4, label="ciclo del PIB real (HP)")
ax.plot(q_idx, util_cycle, color="0.55", lw=1.4, ls=(0, (4, 2)),
        label="ciclo de utilización (SIMULADA)")
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel("trimestre"); ax.set_ylabel("log-puntos / desviación")
ax.set_title("Utilización SIMULADA vs. ciclo del PIB — "
             f"corr $\\approx$ {corr_util:.2f} (por construcción)")
ax.legend(loc="upper left", ncol=1)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Costos de ajuste y la q de Tobin
#
# Sin costos de ajuste, la inversión es un problema *bang-bang*: la empresa saltaría su
# capital al punto donde el producto marginal iguala el costo de uso — una regla de
# esquina, sin dinámica interesante. La **q de Tobin** necesita **costos de ajuste
# convexos** $\Phi(I_t/K_t)$: instalar rápido es costoso, así que la inversión se suaviza.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### q marginal y q promedio
# Con costos de ajuste convexos, la condición de primer orden de la empresa dice que la
# inversión es **creciente en la q marginal**:
# $$ q^{m}_t = \frac{\text{valor sombra de una unidad de capital instalada}}{\text{precio del capital}}. $$
# El problema: $q^m$ **no se observa**. Hayashi (1982) mostró que, bajo **rendimientos
# constantes a escala** y competencia perfecta, la q marginal **iguala** a la q promedio:
# $$ q^{m}_t = q^{a}_t = \frac{\text{valor de mercado de la empresa}}{\text{costo de reposición del capital}}, $$
# y esta última **sí** se observa. Es la q de Tobin.
#
# Proxy agregado (sector corporativo no financiero, *Flow of Funds*):
# $$ q_t = \frac{\text{valor de mercado del capital accionario (NCBEILQ027S)}}{\text{patrimonio neto a valor de mercado (TNWMVBSNNCB)}}. $$
# Es la conocida "Q ratio" de Smithers–Wright: $q>1$ = el mercado valora a las empresas por
# encima del costo de reponer su capital → incentivo a invertir.
#
# > **Ficha de medición (y el control de unidades del mazo).**
# > *Fuente/serie:* Fed Z.1 vía FRED — `NCBEILQ027S` (acciones corporativas, pasivo) sobre
# > `TNWMVBSNNCB` (patrimonio neto a valor de mercado), empresas **no financieras**;
# > inversión `GPDIC1` (real, encadenada). *Muestra:* 1955Q1 en adelante — antes de **1952**
# > la Z.1 sólo publica el cuarto trimestre, así que la serie no es trimestral contigua.
# > *Unidades:* las **dos** series vienen en **millones de dólares**, de modo que el cociente
# > es **adimensional y del orden de la unidad**. Si te sale $q\approx 0.001$ dividiste sólo
# > el numerador entre 1000: comprueba la ficha de FRED, no el nombre de la serie.
# > *Transformación:* $q$ en nivel para la gráfica, $100\,\Delta\log q$ como impulso.
# > *Advertencia conceptual:* el denominador es el patrimonio neto **a valor de mercado**,
# > que el curso usa como aproximación al costo de reposición; omite el capital **intangible**
# > (de ahí la *total q* de Peters–Taylor que discute el mazo).

# %% slideshow={"slide_type": "fragment"}
equity = fred("NCBEILQ027S")     # valor de mercado del capital accionario (pasivo)
networth = fred("TNWMVBSNNCB")   # patrimonio neto a valor de mercado (Net Worth Market Value)
inv = fred("GPDIC1")             # inversión privada bruta real

# Ambas en MILLONES de dólares: el cociente sale adimensional. NO dividir una sola.
q_ratio = (equity / networth).rename("q")

df = pd.concat([q_ratio, inv.rename("inv")], axis=1).dropna()
df = df[df.index >= START]
df["log_inv"] = 100.0 * np.log(df["inv"])        # inversión en log-puntos (respuesta en %)
df["dlog_q"] = 100.0 * np.log(df["q"]).diff()    # choque: variación % de la q
print(f"q de Tobin: min = {df['q'].min():.2f}, max = {df['q'].max():.2f}, "
      f"último = {df['q'].iloc[-1]:.2f}   (n = {len(df)} trimestres)")

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(df.index, df["q"].values, color=cols[0], lw=1.4)
ax.axhline(1.0, color="0.55", lw=1.0, ls=(0, (4, 2)), label="$q = 1$ (paridad)")
ax.set_xlabel("trimestre"); ax.set_ylabel("q de Tobin (agregada)")
ax.set_title("Q ratio: valor de mercado / costo de reposición del capital")
ax.legend(loc="upper left")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### LP-HAC: respuesta de la inversión a un choque de q
# Estimamos con **proyecciones locales** de Jordà (2005), con errores estándar HAC
# (Newey–West, ancho de banda $h+1$ según Plagborg-Møller–Wolf 2021), la respuesta
# acumulada del (log) de la inversión ante un choque de $\Delta\log q$:
# $$ \text{inv}_{t+h} - \text{inv}_{t-1} = \alpha_h + \beta_h\,\Delta\log q_t + \sum_l \gamma_l z_{t-l} + \varepsilon_{t,h}. $$
#
# > **Dos trampas de especificación.** (i) `lp_hac` **ya acumula**: construye internamente
# > $y_{t+h}-y_{t-1}$, así que se le pasa el **nivel** de $100\log(\text{inv})$, **no** su
# > diferencia; si le pasas $\Delta\log(\text{inv})$ estimas la respuesta del *crecimiento*,
# > que es otra cosa. (ii) Usamos `n_lags=4` y el nivel de `GPDIC1`, que es exactamente la
# > especificación de la figura del mazo (`fig_q_tobin_lp`), para que lo que veas aquí y lo
# > que veas proyectado en clase sean lo mismo. El resultado es robusto: con `n_lags=2`, o
# > arrancando la muestra en 1952 en vez de 1955, el pico sigue en $h=5$ con
# > $\beta$ entre $0.36$ y $0.39$.

# %% slideshow={"slide_type": "fragment"}
from puremacro.lp import lp_hac

irf = lp_hac(df.reset_index(drop=True), y="log_inv", x="dlog_q",
             horizons=range(0, 13), n_lags=4)   # misma spec que la figura del mazo
print(irf[["h", "beta", "se", "t", "lo", "hi"]].round(3).to_string(index=False))

peak = irf.loc[irf["beta"].idxmax()]
print(f"\npico en h = {int(peak['h'])} trimestres: beta = {peak['beta']:.3f} "
      f"(t = {peak['t']:.2f})")
assert peak["beta"] > 0 and peak["lo"] > 0        # respuesta positiva y significativa

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.6))
h = irf["h"].values
ax.fill_between(h, irf["lo"].values, irf["hi"].values, color="0.85",
                label="IC 90%")
ax.plot(h, irf["beta"].values, color=cols[0], lw=1.6, marker="o", ms=3,
        label=r"$\beta_h$")
ax.axhline(0, color="0.55", lw=0.8)
ax.set_xlabel("horizonte $h$ (trimestres)")
ax.set_ylabel("respuesta acumulada de la inversión (%)")
ax.set_title(r"IRF por LP-HAC: inversión ante un choque de 1% a $q$")
ax.legend(loc="lower right")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La q agregada explica *poco* la inversión
# La respuesta dinámica es positiva y significativa, pero la q **contemporánea** explica
# una fracción minúscula de la varianza del crecimiento de la inversión: la correlación
# simple es **negativa y prácticamente cero**, y el $R^2$ implícito no llega al 1%. Ese
# contraste —dinámica clara, $R^2$ diminuto— es el hecho estilizado central de esta
# literatura y el **fracaso empírico clásico** de la teoría q: en la teoría, $q$ debería ser
# un **estadístico suficiente** para la inversión.

# %% slideshow={"slide_type": "fragment"}
dlog_inv = df["log_inv"].diff().iloc[1:].values
dlog_q = df["dlog_q"].iloc[1:].values
rho = np.corrcoef(dlog_inv, dlog_q)[0, 1]
r2_contemp = rho ** 2
print(f"corr(Δ log inv, Δ log q)         = {rho:.3f}   (¡el signo ni siquiera es positivo!)")
print(f"R^2 contemporáneo                = {r2_contemp:.3f}")
assert r2_contemp < 0.05                          # la q agregada explica casi nada

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Choques específicos a la inversión
#
# El **precio relativo de los bienes de inversión** (respecto al consumo) ha caído de forma
# secular: producir una unidad de capital cuesta cada vez menos en términos de consumo.
# Esa caída es **cambio tecnológico específico a la inversión** (Greenwood–Hercowitz–Krusell
# 1997). Aquí construimos el **impulso ingenuo** $\text{IST}_t = -\Delta\log(p^{I}_t)$: una
# caída del precio relativo actúa como una mejora en la eficiencia de transformar producto en
# capital.
#
# > **`PIRIC` no es `PERIC`.** La serie del *bundle*, `PIRIC`, es el precio de la **inversión
# > total** relativo al consumo. GHK (1997), Fisher (2006) y Justiniano–Primiceri–Tambalotti
# > (2010) usan el precio del **equipo**, `PERIC`, que no viaja offline y exige red. No son
# > intercambiables: el equipo cae un factor $\approx41$ desde 1947 ($\approx4.8\%$ anual),
# > mientras la inversión total apenas cae un factor $\approx5$ ($\approx2.2\%$ anual), porque
# > el agregado arrastra estructuras y vivienda, que casi no se abaratan. La caída secular que
# > vas a ver es real, pero es **la del agregado**, no la cifra que citan esos artículos.

# %% slideshow={"slide_type": "fragment"}
pinv = fred("PIRIC").dropna()                     # precio de la inversión relativo al consumo
pinv = pinv[pinv.index >= START]
ist = (-100.0 * np.log(pinv).diff()).dropna()     # choque IST (variación %, signo positivo = mejora)
print(f"precio relativo de la inversión: {pinv.iloc[0]:.2f} (1955) -> {pinv.iloc[-1]:.2f} "
      f"({pinv.index[-1].year})")
print(f"caída acumulada = {100.0*np.log(pinv.iloc[0]/pinv.iloc[-1]):.0f} log-puntos")
print(f"IST promedio (anualizado) = {ist.mean()*4:.2f} pp/año,  sd trimestral = {ist.std():.2f}")

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(pinv.index, 100.0 * np.log(pinv.values), color=cols[0], lw=1.6)
ax.set_xlabel("trimestre")
ax.set_ylabel("precio relativo de la inversión (log $\\times$ 100)")
ax.set_title("Precio relativo de la inversión TOTAL (PIRIC) — el hecho de GHK usa PERIC")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### ¿Responde la inversión al "choque" IST? Lo que sale y lo que no
# Repetimos la LP-HAC usando $-\Delta\log p^{I}$ como impulso. La teoría dice que un
# abaratamiento del capital debería elevar la inversión —el canal detrás de los choques a la
# "eficiencia marginal de la inversión" (Justiniano–Primiceri–Tambalotti 2010)—, así que
# esperamos $\beta_h>0$.
#
# **Mira la tabla antes de creerte la teoría.** Los puntos estimados son positivos en los
# primeros horizontes, pero los errores estándar son enormes: la banda del 90% **contiene
# cero en todos los horizontes** y ningún $|t|$ llega a 1.5. Esta salida **no** establece que
# la inversión responda al choque IST. Tres razones, y las tres son la lección del ejercicio:
#
# 1. **No hay identificación.** $-\Delta\log p^{I}_t$ es una variable observada, no un choque
#    estructural: el precio relativo se mueve también por demanda, márgenes y composición. El
#    mazo identifica el choque IST **como es debido**, con Blanchard–Quah bivariado a la
#    Fisher (2006) —el IST es el único con efecto permanente sobre el precio relativo del
#    equipo—, no con la primera diferencia cruda.
# 2. **Serie equivocada para el hecho.** Usamos `PIRIC` (inversión total) porque `PERIC`
#    (equipo) no viaja offline; el agregado diluye justo la variación que identifica el canal.
# 3. **Señal débil.** $sd(\text{IST})\approx0.6$ pp trimestrales sobre ~279 trimestres, y
#    buena parte de esa variación es ruido de deflactores y revisiones: no hay potencia.
#
# Que el punto estimado tenga el signo de la teoría es sugerente y nada más. Publicar
# "la inversión sube ante el choque IST" con esta tabla sería exactamente el error que este
# curso persigue.

# %% slideshow={"slide_type": "fragment"}
dfi = pd.concat([100.0 * np.log(inv).rename("log_inv"), ist.rename("ist")], axis=1).dropna()
dfi = dfi[dfi.index >= START]
irf_ist = lp_hac(dfi.reset_index(drop=True), y="log_inv", x="ist",
                 horizons=range(0, 13), n_lags=2)
print(irf_ist[["h", "beta", "se", "t", "lo", "hi"]].round(3).to_string(index=False))
peak_ist = irf_ist.loc[irf_ist["beta"].abs().idxmax()]
n_signif = int(((irf_ist["lo"] > 0) | (irf_ist["hi"] < 0)).sum())
print(f"\nrespuesta máxima en h = {int(peak_ist['h'])}: beta = {peak_ist['beta']:.3f} "
      f"(t = {peak_ist['t']:.2f})")
print(f"horizontes con banda del 90% que EXCLUYE cero: {n_signif} de {len(irf_ist)} "
      f"  ->  el signo es el de la teoría, la precisión no acompaña")
assert n_signif == 0        # ninguna banda excluye cero: el texto lo dice así

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Ejercicios
# 1. **Utilización y Solow.** Si ignoramos $u_t$ (usamos $K_t$ en vez de $u_t K_t$), ¿en
#    qué dirección sesgamos la ciclicidad del residuo de Solow? Razona con el **signo** de
#    $\mathrm{corr}(u_t, \text{ciclo del PIB})$. Ojo con la trampa: la serie de arriba es
#    **simulada** y ese signo lo pusiste tú al construirla. ¿En qué evidencia **observada**
#    se apoya de verdad ese signo, según el mazo?
# 2. **q marginal vs. promedio.** Enumera dos supuestos de Hayashi (1982) que, al fallar,
#    hacen que $q^{a}\neq q^{m}$. ¿Cómo afecta eso a la regresión de inversión sobre q?
# 3. **Costos de ajuste.** Si los costos de ajuste fueran **más** convexos, ¿la IRF de la
#    inversión sería más rápida o más suave? Responde primero con la teoría. Después amplía
#    `horizons` a `range(0, 21)` y describe la forma de la IRF **estimada** — y explica por
#    qué esa figura, por sí sola, no mide $\phi$.
# 4. **q agregada y $R^2$.** Da tres razones por las que la q agregada explica tan poca
#    varianza de la inversión trimestral (pista: error de medición, burbujas en el precio
#    de las acciones, fricciones financieras, ajuste no convexo/*lumpy* a nivel planta).
# 5. **IST.** Estima la LP-HAC con `n_lags=4` para el impulso IST. ¿Cambia el pico? ¿Cambia
#    en algo la **conclusión** sobre la significancia? ¿Qué haría falta para poder afirmar
#    que la inversión responde al choque IST?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Como $u_t$ es procíclica y positivamente correlacionada con el ciclo, omitirla
#    **infla** la ciclicidad medida de la PTF: parte del auge de producción es más
#    utilización, no más tecnología. La evidencia observada no es la simulación de esta
#    lección: es `TCU` frente a `INDPRO` (mazo B1) y, sobre todo, la PTF **ajustada por
#    utilización** de Basu–Fernald–Kimball (2006) y Fernald (2014), que resulta menos volátil
#    y menos procíclica que el residuo de Solow crudo.
# 2. Falla de rendimientos constantes a escala (poder de mercado, rendimientos
#    decrecientes) y de competencia perfecta ⇒ $q^{a}$ incluye rentas/valor de mercado que
#    no gobiernan la inversión marginal ⇒ atenuación y error de medición en la regresión.
# 3. Más convexidad ⇒ ajuste más lento y suave: la IRF sube más gradualmente y su pico se
#    desplaza a horizontes mayores. Con `range(0, 21)` la IRF estimada sube durante ~3–5
#    trimestres, hace pico cerca de $h=5$ y luego decae despacio, quedándose positiva y con
#    banda por encima de cero hasta $h=20$. Esa forma es **compatible** con ajuste convexo,
#    pero no lo mide: $\beta_h$ es la respuesta a un movimiento **no identificado** de $q$;
#    para leer $\phi$ hay que confrontar la IRF con la del modelo (eso es justamente lo que
#    pide la Sección 3 del final teórico).
# 4. Error de medición ($q^{a}\neq q^{m}$), componentes no fundamentales/burbujas en el
#    precio accionario, fricciones financieras que rompen la equivalencia, y no convexidades
#    (inversión *lumpy*) que rompen la relación suave inversión–q a nivel micro.
# 5. El pico se queda en $h=3$, pero el punto baja de $\beta\approx2.11$ a $\approx1.74$
#    ($t$ de $1.28$ a $1.10$) y a horizontes largos ($h\ge6$) se vuelve **negativo**. La
#    conclusión no cambia: las bandas del 90% siguen cubriendo cero en **todos** los
#    horizontes, antes y después. El problema no son los rezagos ni los EE — es que
#    $-\Delta\log p^{I}$ no es un choque identificado. Para afirmar algo haría falta la
#    identificación de Fisher (2006) —Blanchard–Quah bivariado con el precio relativo del
#    **equipo** (`PERIC`), efecto permanente único— y esa serie exige red.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 5. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "En una frase, ¿por qué la q marginal puede diferir de la q promedio de Tobin?"
# - "Si la q agregada explica muy poca varianza de la inversión, ¿qué fricciones podrían
#   ser responsables?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase, ¿por qué la q marginal de la inversión puede diferir de la "
            "q promedio de Tobin?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** La **utilización variable** hace procíclicos los servicios de capital y sesga
# el residuo de Solow; los **costos de ajuste convexos** dan sentido a la inversión gradual y
# a la **q de Tobin** (marginal = promedio bajo Hayashi 1982). Con LP-HAC vimos que la
# inversión **sí** responde a $\Delta\log q$ —positiva y significativa, con pico cerca de
# $h=5$— pero que la q *agregada* explica **poca** varianza contemporánea (correlación simple
# casi nula, y de signo negativo): el hueco que motiva intangibles fuera del denominador,
# fricciones financieras, burbujas y ajuste *lumpy*. Del **precio relativo de la inversión**
# nos quedamos con el hecho secular —cae de forma sostenida— pero **no** con una IRF creíble:
# el impulso ingenuo $-\Delta\log p^{I}$ no está identificado y su LP no excluye cero en
# ningún horizonte. Ese contraste entre un hecho sólido y una estimación que no lo es
# también es parte de la lección.
#
# **Siguiente módulo:** la lección **10b — economía pequeña y abierta**
# (`10b_economia_abierta_es`), que cierra la semana 14 del mazo Slides07 (B1)
# llevando estos mecanismos a una economía que puede prestar y pedir prestado al resto del
# mundo. Las **rigideces nominales** llegan después, en el bloque de las semanas 15–16
# (lección **10c**), junto con las **fricciones de búsqueda y emparejamiento** del mercado
# laboral (lección **11**). Las fricciones *financieras* que el hueco del $R^2$ sugiere no
# tienen sesión propia en el programa; el hilo más cercano es la lección complementaria
# **23 — crecimiento en riesgo** (`23_growth_at_risk_es`), que mete las **condiciones
# financieras** (`NFCI`) en la distribución condicional del crecimiento.
