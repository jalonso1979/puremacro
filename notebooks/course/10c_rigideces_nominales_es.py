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
# # Módulo 10c — Rigideces nominales: de Rotemberg a la inflación de 2021–2023
#
# **Curso complementario · puremacro · mazo Slides08 (bloque B2) — mercados no competitivos:
# precios rígidos y fricciones laborales**
#
# El mazo B2 cubre las semanas 15–16 del calendario; **esta lección es la semana 15**
# (márgenes, rigidez de precios, no-neutralidad del dinero e inflación 2021–2023). La
# semana 16 —fricciones laborales, Mortensen–Pissarides, Hosios— es la lección `11_b2`.
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Mostrar **por qué con precios flexibles el dinero es neutral** —el margen constante
#    $\mathcal{M}=\varepsilon/(\varepsilon-1)$ clava el costo marginal real— y qué se rompe
#    exactamente cuando ajustar el precio cuesta recursos (**Rotemberg 1982**).
# 2. Derivar la **curva de Phillips neokeynesiana**
#    $\hat{\pi}_t=\beta E_t\hat{\pi}_{t+1}+\lambda\,\widehat{mc}_t$ con
#    $\lambda=(\varepsilon-1)/\vartheta$, y **verificarla numéricamente** diferenciando la
#    condición exacta de fijación de precios.
# 3. Resolver el **modelo de tres ecuaciones** (IS dinámica, Phillips, regla de Taylor) con
#    `puremacro.dsge.klein_solve`, leer sus IRF a un choque monetario y a uno de costos, y
#    dibujar el **principio de Taylor** como una frontera de determinación del equilibrio.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), leyendo **solo** CSV congelados del bundle del curso —
# nunca por red.

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
# ## 1. El gancho: 2021–2023, el episodio que obligó a volver a lo nominal
#
# Entre 2021 y 2023 la inflación general de Estados Unidos y de México llegó a niveles que
# ninguna de las dos economías había visto en **años**, y los dos bancos centrales
# subieron su tasa de política de forma abrupta. Conviene decir *cuántos* años, porque no es
# el mismo número en los dos países y la frase fácil («cuatro décadas») solo vale para uno:
# el pico estadounidense de 2022 no se veía desde **1981** —cuatro décadas— pero el mexicano
# sí se había visto tan tarde como en **diciembre de 2000**, apenas **dos** décadas antes.
# (Y hacia atrás la asimetría se agranda: México vivió inflaciones de dos y hasta tres
# dígitos en los ochenta y noventa, de otro orden de magnitud que 2022.) La celda de abajo
# calcula las dos fechas en vez de pedirte que las creas. El curso hasta aquí ha sido de **ciclos
# reales**: choques de productividad, cuñas, fricciones de búsqueda. En ese mundo el dinero
# no hace nada. Este episodio es el recordatorio de que sí hace.
#
# Los datos vienen del **paquete congelado del curso** (`data_curso/bundle_2026A`,
# descargados de FRED el 2026-08-01 y verificados por hash en su manifiesto):
#
# | serie | contenido | fuente |
# |---|---|---|
# | `CPIAUCSL` | IPC general EE. UU., índice mensual | FRED / BLS |
# | `CPILFESL` | IPC subyacente EE. UU. (sin alimentos ni energía) | FRED / BLS |
# | `FEDFUNDS` | tasa de fondos federales efectiva, % anual | FRED / Reserva Federal |
# | `CPALTT01MXM659N` | IPC general México, variación anual % | FRED / OCDE-MEI |
# | `CPGRLE01MXM659N` | IPC México **sin alimentos ni energía** (definición OCDE), variación anual % | FRED / OCDE-MEI |
# | `IR3TIB01MXM156N` | tasa interbancaria a 3 meses de México, % anual | FRED / OCDE-MEI |
#
# Y dos series más que **no** usa esta sección pero sí el mini-entregable de la sección 7:
#
# | serie | contenido | fuente |
# |---|---|---|
# | `JTSJOL` | vacantes de empleo EE. UU. (JOLTS), **nivel** en miles | FRED / BLS |
# | `UNEMPLOY` | desempleados EE. UU., **nivel** en miles | FRED / BLS |
#
# (Estas dos se congelaron el 2026-07-22, no el 2026-08-01 como las de arriba; la fecha de
# descarga y el `sha256` de cada archivo están en el `manifest.csv` del paquete.)
#
# **Advertencia honesta sobre los datos mexicanos.** Tres cosas que hay que saber antes de
# leer la figura:
#
# 1. **No es la subyacente de INEGI.** `CPGRLE01MXM659N` es el agregado de la OCDE *all items
#    excluding food and energy*, que **no** coincide con la **inflación subyacente oficial**
#    de INEGI/Banxico: la mexicana excluye agropecuarios, energéticos y tarifas autorizadas
#    por el gobierno, pero **sí incluye** los alimentos procesados. La brecha no es cosmética:
#    la subyacente oficial de INEGI tocó **≈8.5% en noviembre de 2022** (dato citado del SIE
#    de Banxico, **no** calculado en este cuaderno), casi **dos puntos** por encima del pico
#    que la celda de abajo imprime para la serie de la OCDE — y eso que las dos hacen pico el
#    mismo mes. La general, en cambio, sí calza con INEGI. Si verificas contra Banxico,
#    verifica contra el concepto correcto.
# 2. **Las series se descontinuaron.** Los índices de precios de México en FRED (procedentes
#    de OCDE-MEI) terminan en **julio de 2024**; el código imprime la última observación
#    disponible para que lo confirmes tú.
# 3. **La tasa es un *proxy*.** Graficamos la **interbancaria a 3 meses**, no la tasa objetivo
#    de Banxico. Para el nivel exacto de la tasa objetivo hay que ir al SIE de Banxico, no a
#    esta figura.

# %% slideshow={"slide_type": "fragment"}
def _serie(nombre):
    """Lee un CSV congelado del bundle (una columna de fechas + una de valores)."""
    d = pd.read_csv(DATA / f"{nombre}.csv", parse_dates=["observation_date"])
    return d.set_index("observation_date").iloc[:, 0]

cpi_us   = _serie("CPIAUCSL")            # índice
core_us  = _serie("CPILFESL")            # índice
ffr      = _serie("FEDFUNDS")            # % anual
pi_mx    = _serie("CPALTT01MXM659N")     # ya viene en variación anual %
pic_mx   = _serie("CPGRLE01MXM659N")     # sin alimentos ni energía (OCDE), variación anual %
                                         # OJO: NO es la subyacente oficial de INEGI/Banxico
tiie3m   = _serie("IR3TIB01MXM156N")     # % anual

pi_us  = 100.0 * (cpi_us  / cpi_us.shift(12)  - 1.0)   # variación anual %
pic_us = 100.0 * (core_us / core_us.shift(12) - 1.0)

VENT = slice("2019-01-01", "2024-07-01")   # ventana común: la mexicana termina en jul-2024

def liftoff(tasa, desde="2021-01-01", umbral=0.25):
    """Primer mes desde `desde` con un alza acumulada de 3 meses >= `umbral` (pp)."""
    d = (tasa - tasa.shift(3)).loc[desde:]
    return d[d >= umbral].index[0]

pico_us,  fecha_us  = pi_us[VENT].max(),  pi_us[VENT].idxmax()
picoc_us, fechac_us = pic_us[VENT].max(), pic_us[VENT].idxmax()
pico_mx,  fecha_mx  = pi_mx[VENT].max(),  pi_mx[VENT].idxmax()
picoc_mx, fechac_mx = pic_mx[VENT].max(), pic_mx[VENT].idxmax()
lift_us, lift_mx = liftoff(ffr), liftoff(tiie3m)

print(f"EE. UU.  pico inflación general    = {pico_us:.2f}%  en {fecha_us:%Y-%m}")
print(f"EE. UU.  pico inflación subyacente = {picoc_us:.2f}%  en {fechac_us:%Y-%m}")
print(f"México   pico inflación general    = {pico_mx:.2f}%  en {fecha_mx:%Y-%m}")
print(f"México   pico IPC sin alim. ni energia (def. OCDE) = {picoc_mx:.2f}%  en {fechac_mx:%Y-%m}")
print( "         (NO es la subyacente de INEGI: la oficial tocó ~8.5% en 2022-11, ~2 pp más;")
print( "          cifra citada del SIE de Banxico, no calculada aquí)")
# ¿cuánto hacía que no se veía ese nivel? Última observación ANTERIOR a 2021 que iguala o
# supera el pico del episodio. No es la misma respuesta en los dos países.
_prev_us = pi_us.loc[:"2020-12"];  _prev_mx = pi_mx.loc[:"2020-12"]
ult_us = _prev_us[_prev_us >= pico_us].index[-1]
ult_mx = _prev_mx[_prev_mx >= pico_mx].index[-1]
print(f"\n¿cuánto hacía que no se veía ese nivel? (última obs. previa a 2021 >= el pico)")
print(f"  EE. UU. {ult_us:%Y-%m}  ->  {(fecha_us - ult_us).days/365.25:4.1f} años"
      f"   = cuatro décadas")
print(f"  México  {ult_mx:%Y-%m}  ->  {(fecha_mx - ult_mx).days/365.25:4.1f} años"
      f"   = DOS décadas, no cuatro")

print(f"\ndespegue de la tasa (1er alza trimestral >= 25 pb):")
print(f"  México  {lift_mx:%Y-%m}   |   EE. UU. {lift_us:%Y-%m}"
      f"   -> {round((lift_us - lift_mx).days / 30.4)} meses de diferencia")
print(f"  inflación de EE. UU. cuando la Fed se movió:     {pi_us[lift_us]:.2f}%")
print(f"  inflación de México cuando Banxico se movió:    {pi_mx[lift_mx]:.2f}%")
print(f"  ...y la de EE. UU. ese mismo mes ({lift_mx:%Y-%m}):      {pi_us[lift_mx]:.2f}%"
      f"   <- México NO tenía la inflación más baja de las dos")
print(f"\núltima observación de precios de México en FRED: {pi_mx.index[-1]:%Y-%m}"
      f"   (serie descontinuada)")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 1 — la inflación se adelantó a la tasa
# En cada panel: inflación general (línea sólida), la medida sin alimentos ni energía
# (guiones) y tasa de política (punteada). Escala de grises; la identidad de cada serie está
# en el **tipo de línea**. En el panel A la línea de guiones **sí** es la subyacente oficial
# de EE. UU. (`CPILFESL`); en el panel B es el agregado de la **OCDE**, que no es la
# subyacente de INEGI —por eso las etiquetas de los dos paneles no dicen lo mismo.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)

for ax, (tit, gen, sub, tasa, etq, etq_sub) in zip(axes, [
    ("A. Estados Unidos", pi_us, pic_us, ffr, "fondos federales",
     "subyacente (sin alim. ni energía)"),
    ("B. México",         pi_mx, pic_mx, tiie3m, "interbancaria 3m (proxy)",
     "sin alim. ni energía (def. OCDE,\nno es la subyacente de INEGI)"),
]):
    ax.plot(gen[VENT].index,  gen[VENT].values,  color="0.10", lw=1.7, ls="-",
            label="inflación general")
    ax.plot(sub[VENT].index,  sub[VENT].values,  color="0.40", lw=1.5, ls=(0, (4, 2)),
            label=etq_sub)
    ax.plot(tasa[VENT].index, tasa[VENT].values, color="0.10", lw=1.5, ls=(0, (1, 1)),
            label=f"tasa de política: {etq}")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_title(tit); ax.set_xlabel("mes")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("% anual")
plt.tight_layout(); plt.show()

print(f"Pie: series congeladas de FRED (bundle 2026A, descarga 2026-08-01). Las series de\n"
      f"precios de México terminan en {pi_mx.index[-1]:%Y-%m} porque FRED las descontinuó;\n"
      f"la tasa mexicana es la interbancaria a 3 meses, proxy de la objetivo de Banxico;\n"
      f"y la linea de guiones del panel B es el agregado de la OCDE sin alimentos ni\n"
      f"energia, NO la inflacion subyacente de INEGI.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** Los dos picos son casi gemelos en magnitud pero no en fecha, y sobre todo: la
# **secuencia de la política monetaria fue distinta**. Banxico empezó a subir nueve meses
# antes que la Reserva Federal, y lo hizo con **su propia** inflación todavía en 5.8%, mientras
# que la Fed esperó a tener 8.2%. Cuidado con la lectura fácil de esa frase: en julio de 2021
# la inflación mexicana **no era menor que la estadounidense** —era mayor, 5.8% contra 5.2%,
# como imprime la celda—. Lo que difirió no fue el nivel de inflación de cada país sino el
# **umbral** con el que cada banco central decidió moverse. Un modelo de precios
# flexibles no tiene nada que decir sobre esta figura: ahí la tasa nominal y el nivel de
# precios son epifenómenos que no tocan una sola cantidad real. Construyamos el modelo en el
# que sí importan.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Con precios flexibles el dinero es neutral (y no es un accidente)
#
# El mazo ya dejó el mercado de productos en competencia monopolista: un continuo de
# empresas, demanda con elasticidad $\varepsilon$, y una regla de precios
# $P_{jt}=\mathcal{M}\,\lambda_{jt}$ con margen **constante**
# $\mathcal{M}=\varepsilon/(\varepsilon-1)$ y $\lambda_{jt}$ el costo marginal nominal.
# Definiendo el costo marginal **real** $mc_t\equiv\lambda_{jt}/P_t$ y usando el equilibrio
# simétrico $P_{jt}=P_t$:
#
# $$1=\mathcal{M}\,mc_t\qquad\Longrightarrow\qquad mc_t=\frac{1}{\mathcal{M}}=\frac{\varepsilon-1}{\varepsilon}\quad\text{para todo }t.$$
#
# El costo marginal real queda **clavado en una constante** que solo depende de
# $\varepsilon$. De la condición de primer orden del trabajo,
# $mc_t=(W_t/P_t)/\text{PMg}_L$, sale el salario real
# $w_t=\mathcal{M}^{-1}\,\text{PMg}_L$: el margen es un **impuesto implícito** sobre el pago
# a los factores, pero es un impuesto de tasa fija.
#
# Ninguna de las ecuaciones del bloque real contiene una variable nominal. Duplicar la
# cantidad de dinero duplica $P_t$ y $W_t$ y deja intacto todo lo real: la **dicotomía
# clásica**. La celda siguiente lo hace visible en una economía estática mínima
# ($y=A\,l$, utilidad $\log c-\chi l^{1+\varphi}/(1+\varphi)$, teoría cuantitativa
# $M V=P\,y$ para fijar el nivel de precios).

# %% slideshow={"slide_type": "fragment"}
def economia_flexible(M, *, A=1.0, eps=6.0, chi=1.0, varphi=1.0, sigma=1.0, V=1.0):
    """Economía estática con competencia monopolista y precios FLEXIBLES.

    Bloque real: mc = 1/markup (constante) -> w_real = mc*A; oferta de trabajo
    chi*l^varphi = w_real*c^-sigma con c = y = A*l. Bloque nominal: MV = P*y.
    Devuelve un diccionario con las cantidades reales y los precios nominales.
    """
    markup = eps / (eps - 1.0)
    mc = 1.0 / markup                    # costo marginal real: constante
    w_real = mc * A                      # salario real = PMgL / markup
    # chi*l^varphi = w_real*(A*l)^-sigma  ->  l = [w_real*A^-sigma/chi]^(1/(varphi+sigma))
    l = (w_real * A ** (-sigma) / chi) ** (1.0 / (varphi + sigma))
    y = A * l
    P = M * V / y                        # teoría cuantitativa
    return {"markup": markup, "mc": mc, "l": l, "y": y,
            "w_real": w_real, "P": P, "W_nominal": w_real * P}

base   = economia_flexible(M=1.0)
doble  = economia_flexible(M=2.0)

print(f"margen  M = eps/(eps-1) = {base['markup']:.4f}   ->  mc = 1/M = {base['mc']:.4f}"
      f"   ( = (eps-1)/eps = {5/6:.4f} )")
print("\n                    M = 1        M = 2      cambio")
for k, etq in [("l", "empleo"), ("y", "producto"), ("w_real", "salario real"),
               ("P", "nivel de precios"), ("W_nominal", "salario nominal")]:
    print(f"  {etq:18s} {base[k]:8.4f}  {doble[k]:8.4f}   x{doble[k]/base[k]:.2f}")

# La dicotomía clásica, verificada: lo real no se mueve, lo nominal escala uno a uno.
assert np.isclose(base["y"], doble["y"]) and np.isclose(base["w_real"], doble["w_real"])
assert np.isclose(doble["P"] / base["P"], 2.0) and np.isclose(doble["W_nominal"] / base["W_nominal"], 2.0)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **El punto que hay que retener.** La neutralidad de arriba no es un resultado numérico
# frágil: es **estructural**. El código no "encuentra" que el dinero es neutral, lo hereda de
# que `mc` es una constante y de que ninguna ecuación real ve a `M`. Para romperla no hace
# falta un mecanismo nuevo —hace falta que el margen **realizado**
# $\mathcal{M}_t=P_{jt}/\lambda_{jt}$ deje de ser constante. Ahí entra Rotemberg.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Rigidez a la Rotemberg: el costo de cambiar el precio
#
# **Rotemberg (1982):** cambiar el precio cuesta recursos. La empresa $j$ paga
# $\frac{\vartheta}{2}\left(P_{jt}/P_{jt-1}-1\right)^{2}Y_t$ en unidades del bien final.
# Sustituida la demanda $Y_{jt}=(P_{jt}/P_t)^{-\varepsilon}Y_t$, el beneficio real del
# periodo en función del precio relativo $x_{jt}\equiv P_{jt}/P_t$ es
#
# $$\Pi_{jt}=\underbrace{x_{jt}^{1-\varepsilon}Y_t}_{\text{ingreso}}-\underbrace{mc_t\,x_{jt}^{-\varepsilon}Y_t}_{\text{costo}}-\frac{\vartheta}{2}\left(\frac{P_{jt}}{P_{jt-1}}-1\right)^{2}Y_t .$$
#
# La empresa maximiza $E_0\sum_t Q_{0,t}\Pi_{jt}$. Como $P_{jt}$ aparece hoy **y** en el costo
# de ajuste de mañana (vía $P_{jt+1}/P_{jt}$), la fijación de precios se vuelve un problema
# **dinámico**. Derivando y multiplicando por $P_{jt}$, en el equilibrio simétrico
# ($x_{jt}=1$, $P_{jt}/P_{jt-1}=\pi_t$ la inflación bruta):
#
# $$(1-\varepsilon)+\varepsilon\,mc_t-\vartheta(\pi_t-1)\pi_t+E_t Q_{t,t+1}\,\vartheta(\pi_{t+1}-1)\pi_{t+1}\frac{Y_{t+1}}{Y_t}=0 .$$
#
# **Comprobación de cordura:** con $\vartheta=0$ queda $\varepsilon\,mc_t=\varepsilon-1$, es
# decir $mc=1/\mathcal{M}$, el caso flexible de la sección 2. La rigidez es una
# *perturbación* del caso flexible, no otro modelo.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### De la condición exacta a la curva de Phillips
# Linealizamos alrededor del estado estacionario de inflación cero: $\pi=1$,
# $mc=(\varepsilon-1)/\varepsilon$, $Q=\beta$, $Y_{t+1}/Y_t=1$. Escribimos
# $\pi_t=1+\hat{\pi}_t$ y $mc_t=mc\,(1+\widehat{mc}_t)$. Término a término:
#
# - $\varepsilon\,mc_t=(\varepsilon-1)(1+\widehat{mc}_t)$, que junto con el $(1-\varepsilon)$
#   deja $(\varepsilon-1)\widehat{mc}_t$;
# - $\vartheta(\pi_t-1)\pi_t$ vale cero en $\pi=1$ y su derivada ahí es
#   $\vartheta\left[(\pi_t-1)+\pi_t\right]_{\pi=1}=\vartheta$, así que $\approx\vartheta\hat{\pi}_t$;
# - el término adelantado, por lo mismo, $\approx\beta\vartheta\,E_t\hat{\pi}_{t+1}$.
#
# Sumando, $(\varepsilon-1)\widehat{mc}_t-\vartheta\hat{\pi}_t+\beta\vartheta E_t\hat{\pi}_{t+1}=0$, es decir
#
# $$\boxed{\;\hat{\pi}_t=\beta\,E_t\hat{\pi}_{t+1}+\lambda\,\widehat{mc}_t\;},\qquad \lambda=\frac{\varepsilon-1}{\vartheta}.$$
#
# La pendiente **crece con $\varepsilon$** (más competencia, más presión a mover el precio) y
# **cae con $\vartheta$** (ajustar cuesta más). Iterando hacia adelante,
# $\hat{\pi}_t=\lambda\sum_{k\ge0}\beta^k E_t\widehat{mc}_{t+k}$: la inflación de hoy es la
# suma descontada de los **costos marginales futuros esperados**. Y como
# $\mathcal{M}_t=1/mc_t$, la misma ecuación es
# $\hat{\pi}_t=\beta E_t\hat{\pi}_{t+1}-\lambda\widehat{\mathcal{M}}_t$: la rigidez de precios
# no añade una cuña, **endogeniza** la que la competencia monopolista ya había puesto ahí.
#
# En vez de creerle a la derivación, la **verificamos**: diferenciamos numéricamente la
# condición exacta en el estado estacionario y comparamos cada derivada parcial con el
# coeficiente analítico.

# %% slideshow={"slide_type": "fragment"}
BETA, EPS = 0.99, 6.0        # descuento trimestral; elasticidad de demanda (margen 20%)

def rotemberg_residual(pi_t, pi_next, mc, *, eps=EPS, vartheta=None, beta=BETA):
    """Condición EXACTA de fijación de precios de Rotemberg en equilibrio simétrico
    (con Q = beta y Y_{t+1}/Y_t = 1). Vale cero en el estado estacionario."""
    return ((1 - eps) + eps * mc
            - vartheta * (pi_t - 1) * pi_t
            + beta * vartheta * (pi_next - 1) * pi_next)

VARTHETA = 58.2524                     # costo de ajuste (calibrado abajo)
mc_ss = (EPS - 1.0) / EPS              # costo marginal real de estado estacionario
h = 1e-5

r0 = rotemberg_residual(1.0, 1.0, mc_ss, vartheta=VARTHETA)
d_pi   = (rotemberg_residual(1+h, 1.0, mc_ss, vartheta=VARTHETA)
          - rotemberg_residual(1-h, 1.0, mc_ss, vartheta=VARTHETA)) / (2*h)
d_pin  = (rotemberg_residual(1.0, 1+h, mc_ss, vartheta=VARTHETA)
          - rotemberg_residual(1.0, 1-h, mc_ss, vartheta=VARTHETA)) / (2*h)
d_mc   = (rotemberg_residual(1.0, 1.0, mc_ss+h, vartheta=VARTHETA)
          - rotemberg_residual(1.0, 1.0, mc_ss-h, vartheta=VARTHETA)) / (2*h)

print(f"residual en el estado estacionario           = {r0:.2e}   (debe ser 0)")
print(f"dR/dpi_t      numérico {d_pi:10.4f}   analítico {-VARTHETA:10.4f}")
print(f"dR/dpi_t+1    numérico {d_pin:10.4f}   analítico {BETA*VARTHETA:10.4f}")
print(f"dR/dmc_t      numérico {d_mc:10.4f}   analítico {EPS:10.4f}")

# El diferencial total  eps*dmc - vartheta*dpi_t + beta*vartheta*dpi_t+1 = 0
# se reescribe, con dmc = mc_ss*mc_hat, como  dpi_t = beta*dpi_t+1 + lambda*mc_hat.
lam_num = -(d_mc * mc_ss) / d_pi          # lambda implícito por diferenciación numérica
beta_num = -d_pin / d_pi                  # beta implícito
print(f"\nlambda  numérico = {lam_num:.6f}   fórmula (eps-1)/vartheta = {(EPS-1)/VARTHETA:.6f}")
print(f"beta    numérico = {beta_num:.6f}   calibrado                = {BETA:.6f}")
assert np.isclose(lam_num, (EPS - 1) / VARTHETA, rtol=1e-6)
assert np.isclose(beta_num, BETA, rtol=1e-8)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### $\vartheta$, $\varepsilon$ y $\lambda$: cómo se calibra la pendiente
# A primer orden la curva de Rotemberg es **idéntica** a la de Calvo, con
# $\lambda_{\text{Calvo}}=(1-\theta)(1-\beta\theta)/\theta$, donde $\theta$ es la
# probabilidad de **no** reoptimizar el precio. Igualar ambas pendientes traduce $\vartheta$
# en $\theta$ —y es como calibramos $\vartheta$ arriba: fijamos $\theta=0.75$ (precios que
# duran en promedio $1/(1-\theta)=4$ trimestres) y despejamos $\vartheta=(\varepsilon-1)/\lambda$.

# %% slideshow={"slide_type": "fragment"}
lam_calvo = lambda th, beta=BETA: (1 - th) * (1 - beta * th) / th

theta_obj = 0.75
lam_obj = lam_calvo(theta_obj)
vartheta_obj = (EPS - 1) / lam_obj
print(f"theta = {theta_obj:.2f}  ->  lambda_Calvo = {lam_obj:.6f}"
      f"  ->  vartheta = (eps-1)/lambda = {vartheta_obj:.4f}")
assert np.isclose(vartheta_obj, VARTHETA, atol=1e-3)   # es el valor usado arriba

print("\n  eps   vartheta   lambda=(eps-1)/vartheta   theta Calvo equivalente   duración (trim.)")
for eps_i in (4.0, 6.0, 11.0):
    for vth_i in (30.0, VARTHETA, 120.0):
        lam_i = (eps_i - 1) / vth_i
        # theta que iguala la pendiente de Calvo a lam_i (búsqueda por bisección, sin scipy)
        lo, hi = 1e-4, 1 - 1e-6
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if lam_calvo(mid) > lam_i: lo = mid
            else: hi = mid
        th_i = 0.5 * (lo + hi)
        print(f" {eps_i:5.1f} {vth_i:9.2f} {lam_i:22.4f} {th_i:22.3f} {1/(1-th_i):16.1f}")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura de la tabla.** Más competencia (mayor $\varepsilon$) o costos de ajuste menores
# empinan la curva de Phillips y, en la traducción a Calvo, acortan la duración implícita de
# los precios. La combinación que usamos —$\varepsilon=6$ (margen del 20%) y
# $\vartheta\approx58$— corresponde a precios que duran cuatro trimestres, el valor de
# referencia de la literatura de calibración. Es una **elección**, no un dato: la evidencia
# microeconómica de precios (Bils–Klenow 2004) sugiere duraciones más cortas, y esa tensión
# entre la pendiente que pide el macro y la que pide el micro sigue abierta.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. El modelo de tres ecuaciones
#
# Cerramos el modelo con tres ecuaciones en desviaciones (todo trimestral), con $\tilde{y}_t$
# la brecha del producto, $\hat{\pi}_t$ la inflación neta e $i_t$ la tasa nominal:
#
# $$\text{IS dinámica:}\qquad \tilde{y}_t=E_t\tilde{y}_{t+1}-\tfrac{1}{\sigma}\left(i_t-E_t\hat{\pi}_{t+1}-r^n_t\right)$$
# $$\text{Phillips:}\qquad \hat{\pi}_t=\beta E_t\hat{\pi}_{t+1}+\kappa\,\tilde{y}_t+u_t,\qquad \kappa=\lambda\psi$$
# $$\text{Taylor:}\qquad i_t=r^n_t+\phi_\pi\hat{\pi}_t+\phi_y\tilde{y}_t+v_t$$
#
# donde $\psi>0$ es el **traspaso de la brecha al costo marginal**
# ($\widehat{mc}_t=\psi\tilde{y}_t$; sin capital, $\psi=\sigma+\varphi$), $v_t$ es un choque
# de **política monetaria** y $u_t$ un choque de **costos** (*cost-push*), ambos AR(1). La
# tasa natural $r^n_t$ entra como dato y se cancela al sustituir la regla en la IS.
#
# Para que la lección permita también la sección 6, escribimos la Phillips en su versión
# **híbrida** $\hat{\pi}_t=\gamma\hat{\pi}_{t-1}+\beta(1-\gamma)E_t\hat{\pi}_{t+1}+\kappa\tilde{y}_t+u_t$;
# con $\gamma=0$ —el caso de esta sección— es exactamente la curva derivada arriba.
#
# El sistema se escribe en la forma de Klein $A\,E_t z_{t+1}=B\,z_t$ con
# $z_t=[v_t,\;u_t,\;\hat{\pi}_{t-1}\;|\;\tilde{y}_t,\;\hat{\pi}_t]$: tres variables
# **predeterminadas** y dos de **salto**.

# %% slideshow={"slide_type": "fragment"}
from puremacro.dsge import klein_solve

SIGMA, VARPHI = 1.0, 1.0                 # aversión al riesgo; inversa de la Frisch
PSI   = SIGMA + VARPHI                   # traspaso de la brecha al costo marginal
LAM   = (EPS - 1) / VARTHETA             # pendiente de la Phillips en el costo marginal
KAPPA = LAM * PSI                        # pendiente en la brecha
RHO_V, RHO_U = 0.5, 0.8                  # persistencia de los choques
IV, IU, IPL, IY, IPI = 0, 1, 2, 3, 4     # posiciones en z
N_PRE = 3

def nk_sistema(phi_pi, phi_y, gamma=0.0, kappa=KAPPA, beta=BETA, sigma=SIGMA):
    """Matrices (A, B) del modelo de tres ecuaciones en la forma A E_t z_{t+1} = B z_t."""
    A = np.zeros((5, 5)); B = np.zeros((5, 5))
    A[0, IV] = 1.0;  B[0, IV] = RHO_V                        # v_{t+1} = rho_v v_t
    A[1, IU] = 1.0;  B[1, IU] = RHO_U                        # u_{t+1} = rho_u u_t
    A[2, IPL] = 1.0; B[2, IPI] = 1.0                         # pi_lag_{t+1} = pi_t
    A[3, IY] = 1.0;  A[3, IPI] = 1.0 / sigma                 # IS (con Taylor sustituida)
    B[3, IV] = 1.0 / sigma; B[3, IY] = 1.0 + phi_y / sigma; B[3, IPI] = phi_pi / sigma
    A[4, IPI] = beta * (1 - gamma)                           # Phillips (híbrida si gamma>0)
    B[4, IU] = -1.0; B[4, IPL] = -gamma; B[4, IY] = -kappa; B[4, IPI] = 1.0
    return A, B

PHI_PI, PHI_Y = 1.5, 0.125               # Taylor (1993), trimestral: 1.5 y 0.5/4
sol = klein_solve(*nk_sistema(PHI_PI, PHI_Y), N_PRE)
print(f"calibración:  beta={BETA}  sigma={SIGMA}  varphi={VARPHI}  eps={EPS}"
      f"  vartheta={VARTHETA:.2f}")
print(f"              lambda={LAM:.4f}  psi={PSI}  kappa=lambda*psi={KAPPA:.4f}")
print(f"              phi_pi={PHI_PI}  phi_y={PHI_Y}  rho_v={RHO_V}  rho_u={RHO_U}")
print(f"\nklein_solve: eu = {sol.eu}   (1,1) = solución estable única")
print(f"módulos de los autovalores: {np.round(np.sort(np.abs(sol.eigenvalues)), 3)}")
print(f"  -> {int((np.abs(sol.eigenvalues) > 1 + 1e-9).sum())} fuera del círculo unitario;"
      f" hacen falta exactamente {5 - N_PRE} (las variables de salto)")
assert sol.eu == (1, 1)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Las IRF: el mismo modelo, dos choques, dos comovimientos
# Con la solución $x_{t+1}=Gx_t$, $y_t=Fx_t$, una IRF es simplemente poner el estado inicial
# en el choque y dejar correr el sistema. Normalizamos cada choque para que sea legible:
# el **monetario** eleva la tasa 100 pb anualizados en el impacto; el de **costos** eleva la
# inflación 100 pb anualizados en el impacto. Inflación y tasa se reportan **anualizadas**
# ($\times 4$); la brecha, en puntos porcentuales.

# %% slideshow={"slide_type": "fragment"}
H = 13   # horizonte en trimestres (0..12)

def irf(sol, choque, phi_pi=PHI_PI, phi_y=PHI_Y, H=H):
    """IRF por simulación determinista: x_0 = choque, x_{t+1} = G x_t, y_t = F x_t."""
    x = np.zeros((H, N_PRE)); x[0] = choque
    for t in range(1, H):
        x[t] = sol.G @ x[t - 1]
    y = (sol.F @ x.T).T                                   # columnas: [brecha, inflación]
    i = phi_pi * y[:, 1] + phi_y * y[:, 0] + x[:, IV]     # regla de Taylor
    return {"brecha": y[:, 0], "pi": 4 * y[:, 1], "i": 4 * i}   # pi e i anualizadas

mon = irf(sol, [1.0, 0.0, 0.0])
esc_m = 1.00 / mon["i"][0]                                # 100 pb anualizados en el impacto
mon = {k: v * esc_m for k, v in mon.items()}

cost = irf(sol, [0.0, 1.0, 0.0])
esc_c = 1.00 / cost["pi"][0]                              # 100 pb anualizados de inflación
cost = {k: v * esc_c for k, v in cost.items()}

print("CHOQUE MONETARIO (contractivo, +100 pb en el impacto)")
print(f"  brecha del producto en el impacto = {mon['brecha'][0]:+.3f} pp")
print(f"  inflación en el impacto           = {mon['pi'][0]:+.3f} pp anualizados")
print(f"  trimestre del mínimo de la brecha = {int(np.argmin(mon['brecha']))}")
print("CHOQUE DE COSTOS (+100 pb de inflación en el impacto)")
print(f"  brecha del producto en el impacto = {cost['brecha'][0]:+.3f} pp")
print(f"  inflación en el impacto           = {cost['pi'][0]:+.3f} pp anualizados")
print(f"  respuesta de la tasa en el impacto= {cost['i'][0]:+.3f} pp anualizados")

sig_mon  = np.sign(mon['brecha'][0] * mon['pi'][0])
sig_cost = np.sign(cost['brecha'][0] * cost['pi'][0])
print(f"\ncomovimiento producto-precios:  demanda (monetario) = {sig_mon:+.0f}"
      f"   |   oferta (costos) = {sig_cost:+.0f}")
assert sig_mon > 0 and sig_cost < 0    # el signo del comovimiento separa los dos choques

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 2 — el signo del comovimiento separa demanda de oferta
# Izquierda, choque **monetario** (un choque de demanda): la brecha y la inflación caen
# **juntas**, y el banco central no enfrenta disyuntiva. Derecha, choque de **costos**: la
# inflación sube mientras la brecha cae —direcciones **opuestas**— y ahí sí hay disyuntiva.
# Esta figura es la que usaremos en la sección 6 para pensar 2021–2023.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True)
h = np.arange(H)
for ax, d, tit in [(axes[0], mon,  "A. Choque monetario contractivo (+100 pb)"),
                   (axes[1], cost, "B. Choque de costos (+100 pb de inflación)")]:
    ax.plot(h, d["brecha"], color="0.10", lw=1.8, ls="-",          label=r"brecha $\tilde{y}_t$ (pp)")
    ax.plot(h, d["pi"],     color="0.35", lw=1.6, ls=(0, (4, 2)),  label=r"inflación $\hat{\pi}_t$ (pp anual.)")
    ax.plot(h, d["i"],      color="0.10", lw=1.4, ls=(0, (1, 1)),  label=r"tasa nominal $i_t$ (pp anual.)")
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_title(tit, fontsize=11); ax.set_xlabel("trimestres tras el choque")
    ax.legend(fontsize=8)
axes[0].set_ylabel("desviación")
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### El principio de Taylor: dónde el equilibrio deja de ser único
#
# El mazo lo deriva buscando una desviación **constante y autosostenida**
# ($\hat{\pi}_t=\hat{\pi}$, $\tilde{y}_t=\tilde{y}$, $v_t=0$). La Phillips pide
# $\tilde{y}=\frac{1-\beta}{\kappa}\hat{\pi}$; la Euler pide que la tasa real no se mueva,
# $i_t-\hat{\pi}=r^n_t$; y la regla entrega
# $i_t-r^n_t=\left[\phi_\pi+\phi_y\frac{1-\beta}{\kappa}\right]\hat{\pi}$. Las tres cosas solo
# son compatibles si
#
# $$\kappa\,(\phi_\pi-1)+(1-\beta)\,\phi_y=0\qquad\text{o bien}\qquad \hat{\pi}=0 .$$
#
# Si el banco central elige coeficientes que hacen esa expresión **positiva**, la única
# desviación constante posible es $\hat{\pi}=0$ y el equilibrio es **determinado**. Ese es el
# **principio de Taylor**: con $\phi_y=0$ se reduce a $\phi_\pi>1$. La intuición es directa:
# si la tasa nominal sube **menos** que la inflación, la tasa **real baja** cuando la
# inflación sube, la demanda se calienta y la expectativa se valida sola.
#
# Barremos la malla $(\phi_\pi,\phi_y)$ y, en cada punto, preguntamos a `klein_solve` si la
# condición de Blanchard–Kahn se cumple. La frontera numérica debe coincidir con
# $\phi_\pi^\*(\phi_y)=1-\frac{(1-\beta)}{\kappa}\phi_y$.
#
# Dos advertencias de lectura de la salida, ambas sobre el **filo** de la condición y ninguna
# cosmética. Primera: la bandera `eu` de esta versión de `klein_solve` vale `(0,0)` en
# cualquier fallo de Blanchard–Kahn, tanto si **faltan** autovalores inestables
# (indeterminación) como si **sobran** (inexistencia); quien distingue los dos casos es el
# conteo de autovalores, no la bandera. Segunda: justo **en** la frontera hay un autovalor de
# módulo exactamente 1, y ahí la condición se satisface con **igualdad**: el punto
# $\phi_\pi=1,\ \phi_y=0$ no es determinado por teoría, aunque el redondeo lo pinte de un lado
# o del otro. Por eso la celda imprime cuántos puntos de filo hay.

# %% slideshow={"slide_type": "fragment"}
g_pi = np.linspace(0.0, 2.5, 101)      # malla de phi_pi
g_y  = np.linspace(0.0, 2.0, 81)       # malla de phi_y
det = np.zeros((g_y.size, g_pi.size))  # 1 = equilibrio único
nun = np.zeros_like(det)               # autovalores fuera del círculo unitario
eus = {}                               # recuento de banderas eu devueltas
for a, py in enumerate(g_y):
    for b, pp in enumerate(g_pi):
        s_ab = klein_solve(*nk_sistema(pp, py), N_PRE)
        det[a, b] = 1.0 if s_ab.eu == (1, 1) else 0.0
        nun[a, b] = int((np.abs(s_ab.eigenvalues) > 1 + 1e-9).sum())
        eus[s_ab.eu] = eus.get(s_ab.eu, 0) + 1

pendiente = (1 - BETA) / KAPPA          # cuánto phi_pi "compra" cada unidad de phi_y
frontera = 1.0 - pendiente * g_y

print(f"fracción de la malla con equilibrio único = {det.mean():.3f}")
print("autovalores inestables en la malla (hacen falta 2): "
      + ", ".join(f"{int(k)} en {int((nun == k).sum())} puntos"
                  for k in np.unique(nun)))
print("  -> en esta malla el fallo es siempre por DEFECTO: indeterminación, no inexistencia")
print("banderas eu devueltas: "
      + ", ".join(f"{k} en {v} puntos" for k, v in sorted(eus.items())))
print("  OJO con la bandera: esta versión de klein_solve devuelve eu=(0,0) en TODO fallo de")
print("  Blanchard-Kahn, también cuando sobran soluciones. El diagnóstico económico es el")
print("  conteo de autovalores de la línea anterior (1 < 2 = indeterminación), no la bandera.")
_filo = int((det.sum()) - (nun == 2).sum())
print(f"  y hay {_filo} punto(s) de FILO donde los dos conteos discrepan: son los de módulo")
print("  exactamente 1 (phi_pi=1 con phi_y=0). Ahí la condición se cumple con IGUALDAD y la")
print("  teoría no promete unicidad; el signo del redondeo decide, no la economía.")
print(f"pendiente de la frontera  (1-beta)/kappa  = {pendiente:.4f}")
print("\n phi_y   frontera teórica   primer phi_pi determinado en la malla (paso "
      f"{g_pi[1]-g_pi[0]:.3f})")
for py in (0.0, 0.5, 1.0, 2.0):
    fila = det[int(np.argmin(np.abs(g_y - py)))]
    primero = g_pi[np.where(fila > 0)[0][0]]
    print(f" {py:5.2f} {1 - pendiente*py:18.4f} {primero:32.4f}")

# La frontera numérica reproduce la analítica dentro de la resolución de la malla.
assert abs(g_pi[np.where(det[0] > 0)[0][0]] - 1.0) <= (g_pi[1] - g_pi[0])

# Autovalores como función de phi_pi (con phi_y = 0): el cruce del círculo unitario.
g_pi2 = np.linspace(0.5, 2.5, 81)
mods = np.array([np.sort(np.abs(klein_solve(*nk_sistema(pp, 0.0), N_PRE).eigenvalues))[-2:]
                 for pp in g_pi2])

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 3 — el corazón de la lección
# **Izquierda:** la región donde el equilibrio es **único** (gris claro) y donde no lo es
# (gris oscuro), con la frontera analítica superpuesta. **Derecha:** los dos módulos de
# autovalor más grandes; para que la condición de Blanchard–Kahn se cumpla hacen falta
# **dos** por encima de 1 (hay dos variables de salto), y el segundo cruza el círculo
# unitario justo en $\phi_\pi=1$.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

ax = axes[0]
ax.pcolormesh(g_pi, g_y, det, cmap="Greys_r", vmin=-0.35, vmax=1.35, shading="auto")
ax.plot(frontera, g_y, color="0.0", lw=2.0, ls="-")
ax.plot(PHI_PI, PHI_Y, marker="o", ms=7, mfc="white", mec="0.0", mew=1.6, ls="none")
ax.annotate(r"Taylor (1993): $\phi_\pi=1.5,\ \phi_y=0.125$",
            xy=(PHI_PI, PHI_Y), xytext=(1.30, 0.52), fontsize=8, color="0.10",
            arrowprops=dict(arrowstyle="->", color="0.10", lw=0.9))
ax.text(1.05, 1.92, r"frontera $\kappa(\phi_\pi-1)+(1-\beta)\phi_y=0$",
        fontsize=8, color="0.10", va="top", ha="left")
ax.set_xlim(g_pi[0], g_pi[-1]); ax.set_ylim(g_y[0], g_y[-1])
ax.set_xlabel(r"$\phi_\pi$  (respuesta a la inflación)")
ax.set_ylabel(r"$\phi_y$  (respuesta a la brecha)")
ax.set_title("A. Determinación del equilibrio")
ax.text(1.80, 1.25, "equilibrio\núnico", ha="center", fontsize=10, color="0.10")
ax.text(0.48, 1.25, "indeterminado\n(equilibrios\nmúltiples)", ha="center", fontsize=10,
        color="white")
ax.grid(False)

ax = axes[1]
ax.plot(g_pi2, mods[:, 1], color="0.10", lw=1.8, ls="-",         label="módulo mayor")
ax.plot(g_pi2, mods[:, 0], color="0.45", lw=1.6, ls=(0, (4, 2)), label="segundo módulo")
ax.axhline(1.0, color="0.0", lw=1.0, ls=(0, (1, 1)))
ax.axvline(1.0, color="0.55", lw=1.0, ls=(0, (3, 1, 1, 1)))
ax.text(1.03, mods[:, 1].max() * 0.97, r"$\phi_\pi=1$", fontsize=9, color="0.35")
ax.set_xlabel(r"$\phi_\pi$   (con $\phi_y=0$)"); ax.set_ylabel("módulo del autovalor")
ax.set_title("B. Blanchard–Kahn: se necesitan dos módulos > 1")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lo que hay que ver en el panel A.** La frontera es **casi vertical**. La pendiente
# $(1-\beta)/\kappa$ impresa arriba dice cuánto $\phi_\pi$ te "compra" cada unidad de
# $\phi_y$: es diminuta porque $1-\beta\approx0.01$. Responder al producto ayuda, pero
# apenas; en la práctica el principio de Taylor **es** $\phi_\pi>1$. Y no es un tecnicismo:
# a la izquierda de esa línea el modelo admite equilibrios de **profecía autocumplida** —la
# inflación puede subir simplemente porque todos esperan que suba, y la política monetaria la
# valida. Es la lectura estándar de la *Gran Inflación* de los años setenta (Clarida, Galí y
# Gertler 2000).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Confrontación con la evidencia que el curso ya produjo
#
# El curso identifica choques monetarios por tres vías independientes —la **narrativa de
# Romer y Romer (2004)**, las **sorpresas de alta frecuencia** alrededor de los anuncios de
# la Reserva Federal (Gertler–Karadi 2015) y las sorpresas equivalentes en los anuncios de
# Banxico— y las tres coinciden en el mismo hecho: **un endurecimiento monetario no
# anticipado contrae el producto durante varios trimestres y baja los precios más tarde y
# menos**. Ese hecho no cabe en ningún modelo de mercados completos y precios flexibles:
# ahí, como vimos en la sección 2, un choque nominal cambia $P_t$ y nada más.
#
# En el modelo con rigidez a la Rotemberg sí cabe, y con la aritmética exacta:
# la contracción sube la tasa **real** (principio de Taylor), enfría la demanda y **baja**
# $\widehat{mc}_t$; bajar $\widehat{mc}_t$ es **subir** el margen $\widehat{\mathcal{M}}_t$, y
# el margen aprieta la cuña sobre el salario y el rendimiento del capital: caen el empleo y la
# inversión, y con ellos el producto.
#
# **Dónde falla, dicho sin adornos.** Las IRF *estimadas* tienen forma de **joroba**, con el
# máximo entre el cuarto y el octavo trimestre, tanto en el producto como en los precios
# (Ramey 2016; Christiano, Eichenbaum y Evans 2005). La versión que acabamos de resolver
# responde **de inmediato** y decae de forma monótona: la curva derivada es puramente
# prospectiva, $\hat{\pi}_t=\lambda\sum_{k\ge0}\beta^k E_t\widehat{mc}_{t+k}$, así que una
# $\lambda$ finita amortigua el **tamaño** de la respuesta de la inflación, no su **fecha**.
# La celda siguiente lo comprueba sobre nuestra propia IRF.
#
# **Nota de honestidad sobre los datos.** El bundle congelado de estas lecciones **no**
# incluye una IRF monetaria estimada, así que aquí no reportamos ninguna cifra empírica
# calculada: el horizonte del máximo (trimestres 4–8) es una **cita** a Ramey (2016), no un
# número producido por este cuaderno. Lo único que calculamos es la fecha del máximo del
# **modelo**.

# %% slideshow={"slide_type": "fragment"}
q_min_y  = int(np.argmin(mon["brecha"]))     # trimestre del mínimo de la brecha
q_min_pi = int(np.argmin(mon["pi"]))         # trimestre del mínimo de la inflación
monot_y  = bool(np.all(np.diff(mon["brecha"][q_min_y:]) >= -1e-12))

print("MODELO (choque monetario contractivo)")
print(f"  máximo efecto sobre la brecha    en el trimestre {q_min_y}")
print(f"  máximo efecto sobre la inflación en el trimestre {q_min_pi}")
print(f"  ¿decae de forma monótona tras el máximo?  {'sí' if monot_y else 'no'}")
print(f"  vida media de la brecha (rho_v = {RHO_V}): "
      f"{np.log(0.5)/np.log(RHO_V):.1f} trimestres")
print("\nDATOS (cita, no cálculo de este cuaderno)")
print("  Ramey (2016) y CEE (2005): máximo de la respuesta entre los trimestres 4 y 8.")
print("  -> el desfase no es un detalle de calibración: el modelo prospectivo puro NO")
print("     puede generar joroba. CEE lo cierran con hábitos en el consumo, costos de")
print("     ajuste de la inversión e indexación de precios. ¿Mecanismos o parches?")
assert q_min_y == 0 and q_min_pi == 0        # el modelo responde en el impacto

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 6. La inflación de 2021–2023: tres explicaciones y lo que el modelo puede arbitrar
#
# Tres relatos compitieron por el mismo repunte que graficamos en la sección 1:
#
# - **Demanda.** Estímulo fiscal y monetario sin precedente: el gasto empuja $\tilde{y}_t$ y
#   con él $\widehat{mc}_t$ y la inflación. Predicción distintiva: **producto y precios se
#   mueven en la misma dirección** (panel A de la Figura 2).
# - **Oferta.** Energía, alimentos y cuellos de botella en las cadenas de suministro: sube el
#   costo marginal sin que suba la demanda. Predicción distintiva: **producto y precios se
#   mueven en direcciones opuestas** (panel B), y aparece una disyuntiva para la política
#   monetaria que con demanda no existe.
# - **Expectativas.** Si $E_t\hat{\pi}_{t+1}$ se **desancla**, la curva de Phillips se
#   desplaza sola y la desinflación exige recesión. Si sigue anclada, la desinflación es
#   barata —que es, de hecho, lo que ocurrió en ambos países.
#
# **Bernanke y Blanchard (2023)** descomponen el episodio con un sistema de precios y
# salarios en cuatro bloques: energía y alimentos, indicadores de escasez, tensión laboral
# ($v/u$) y expectativas. Su veredicto: el **arranque** fue de oferta; la **persistencia** de
# 2022–23, laboral; las expectativas **nunca se desanclaron**.
#
# ### Qué representa y qué puede arbitrar el modelo que construimos
# **Representa los tres; arbitra dos.** Demanda y oferta entran por $\widehat{mc}_t$ con
# signos opuestos sobre el producto, y el comovimiento producto-precios es **observable**:
# eso los separa, y es exactamente el signo que la Figura 2 calculó. El tercero entra por
# $E_t\hat{\pi}_{t+1}$, que bajo expectativas racionales **no es un dato sino parte de la
# solución del modelo**; adjudicarlo exige encuestas de expectativas, que este cuaderno no
# tiene. Representar un canal no es poder atribuirle el episodio.
#
# **Lo que el modelo no puede hacer, dicho sin adornos:** hablar de sectores ni de cuellos de
# botella ($\widehat{mc}_t$ es una vara agregada); producir una curva de Phillips **no
# lineal**, que parte de la literatura de 2021–23 considera indispensable
# (Benigno–Eggertsson 2023); pronunciarse sobre el **traspaso cambiario**, decisivo en
# México; ni decir nada sobre la tasa real natural, que aquí entra como dato.
#
# La no linealidad está **en disputa**, y conviene no confundir dos objeciones distintas:
#
# - **Ball, Leigh y Mishra (2022)** ajustan el **propio episodio** 2020–2022 con una curva de
#   Phillips **lineal** en la tensión laboral $v/u$ (más un término de expectativas de largo
#   plazo y otro de precios relativos): si con una especificación recta se explica el
#   repunte, la no linealidad deja de ser indispensable.
# - **Hazell, Herreño, Nakamura y Steinsson (2022)** son otra cosa: estiman la pendiente con
#   un panel de **estados de EE. UU., 1978–2018**, es decir **antes** del episodio, y lo que
#   encuentran es que la curva es muy **plana** —pendiente pequeña, no necesariamente
#   recta—. Es evidencia previa **contra un empinamiento**, no un ajuste de 2021–23.
#
# Plano y recto no son sinónimos: se puede tener una curva plana en el tramo normal y
# empinada en el tramo de escasez, que es justo lo que sostienen Benigno y Eggertsson.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lo que sí podemos cuantificar: el precio de desanclar las expectativas
# La versión híbrida $\hat{\pi}_t=\gamma\hat{\pi}_{t-1}+\beta(1-\gamma)E_t\hat{\pi}_{t+1}+\kappa\tilde{y}_t+u_t$
# nos da una manera barata de poner número al tercer relato. Con $\gamma=0$ las expectativas
# son puramente prospectivas —**ancladas**—; con $\gamma>0$ una parte de la inflación de hoy
# se hereda de la de ayer, que es lo que hacen la indexación y las expectativas adaptativas.
# Sometemos ambas economías a un choque de costos **normalizado a la misma inflación de
# impacto** (+100 pb anualizados) y comparamos.
#
# **Cuidado con qué se mantiene fijo.** Lo que igualamos entre economías **no** es el tamaño
# del choque $u_t$ sino la inflación que produce en el impacto: como una $\gamma$ mayor
# amortigua la respuesta contemporánea, hace falta un $u_t$ **más grande** para llegar al
# mismo +100 pb, y la celda imprime cuánto más grande. Es la comparación pertinente para la
# pregunta que nos interesa —*revertir la misma inflación observada*, ¿cuánto producto
# cuesta?—, pero no es «el mismo choque»: si igualáramos $u_t$, la economía con $\gamma$ alta
# arrancaría además desde una inflación de impacto menor.
#
# **Advertencia:** $\gamma$ es un parámetro que elegimos, no una medición del anclaje de
# 2021–23. Lo que sigue es una *comparación de modelos*, no una estimación del episodio.

# %% slideshow={"slide_type": "fragment"}
H2 = 25
print(" gamma   u necesario   pico inflación   trim. hasta 1/2 del pico   costo acum. producto")
res = {}
for gamma in (0.0, 0.25, 0.5):
    s_g = klein_solve(*nk_sistema(PHI_PI, PHI_Y, gamma=gamma), N_PRE)
    assert s_g.eu == (1, 1)
    d = irf(s_g, [0.0, 1.0, 0.0], H=H2)
    esc_u = 1.0 / d["pi"][0]          # tamaño de u que da +100 pb de inflación en el impacto
    d = {k: v * esc_u for k, v in d.items()}
    pico = d["pi"].max()
    mitad = int(np.argmax(d["pi"] < pico / 2.0))
    costo = d["brecha"].sum()
    res[gamma] = (pico, mitad, costo, esc_u)
    print(f" {gamma:5.2f} {esc_u:13.4f} {pico:16.3f} {mitad:26d} {costo:23.3f}")

r0_, r5_ = res[0.0], res[0.5]
print(f"\nPara la MISMA inflación de impacto, gamma=0.5 exige un choque u"
      f" {r5_[3]/r0_[3]:.2f} veces mayor que gamma=0 (no es el mismo choque: es el mismo impacto).")
print(f"Desanclar a la mitad (gamma 0 -> 0.5) multiplica el costo acumulado en producto"
      f" por {r5_[2]/r0_[2]:.2f}")
print(f"y alarga de {r0_[1]} a {r5_[1]} trimestres el tiempo hasta que la inflación cae a la"
      f" mitad de su pico.")
assert abs(r5_[2]) > abs(r0_[2])   # con expectativas menos ancladas, la desinflación cuesta más

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** Partiendo de **la misma inflación de impacto**, la economía con expectativas
# menos ancladas termina con un pico más alto, una desinflación más lenta y —sobre todo— un
# episodio **mucho más caro de revertir** en producto. Esa es, en el lenguaje del modelo, la
# razón por la que el veredicto
# de Bernanke y Blanchard sobre el **anclaje** importa tanto: si las expectativas se hubieran
# desanclado, la desinflación de 2022–23 habría exigido la recesión que muchos pronosticaron
# y que no llegó. Lo que el modelo **no** dice es si estuvieron ancladas: eso lo dicen las
# encuestas y los *breakevens*, no estas ecuaciones.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 7. Preguntas para el debate
# 1. **Si el curso es de ciclos reales, ¿para qué esto?** En el RBC el dinero es neutral por
#    construcción y las fluctuaciones vienen de la tecnología. Pero las tres estrategias de
#    identificación del curso dicen que un choque nominal mueve cantidades reales. ¿Es la
#    rigidez de precios un **mecanismo nuevo** o solo la endogeneización de una cuña que la
#    competencia monopolista ya había puesto en el modelo? Defiende tu respuesta con la
#    ecuación $\hat{\pi}_t=\beta E_t\hat{\pi}_{t+1}-\lambda\widehat{\mathcal{M}}_t$.
# 2. **La salvedad de Nekarda–Ramey.** El canal de esta lección exige que el margen sea
#    **contracíclico** (que caiga en las expansiones). La evidencia empírica lo mide acíclico
#    o incluso procíclico. ¿Qué le pasa al mecanismo si el dato tiene razón? ¿Se salva
#    cambiando la medición del margen, o hay que cambiar el modelo?
# 3. **La joroba.** El modelo responde en el impacto; los datos, en el trimestre 4–8.
#    Christiano, Eichenbaum y Evans cierran el hueco con hábitos en el consumo, costos de
#    ajuste de la inversión e **indexación de precios a la inflación pasada**. ¿Cuál de los
#    tres te parece un mecanismo y cuál un parche? Fíjate en que la indexación es
#    exactamente el $\gamma>0$ de la sección 6: ¿es plausible como comportamiento, o solo
#    como forma reducida que hace calzar la IRF?
# 4. **La figura de la sección 1.** Banxico subió su tasa nueve meses antes que la Fed, con su
#    inflación en 5.8% frente al 8.2% que ya tenía EE. UU. cuando la Fed se movió —aunque en
#    ese julio de 2021 la inflación mexicana era **mayor** que la estadounidense (5.8% contra
#    5.2%): lo que difirió fue el umbral de reacción, no el nivel de inflación—.
#    En el lenguaje del modelo de tres ecuaciones, ¿qué diferencia entre
#    las dos economías podría justificarlo: una $\phi_\pi$ mayor, una $\lambda$ mayor, o algo
#    que el modelo simplemente no tiene (traspaso cambiario, historia de credibilidad)?
#
# ### Mini-entregable (30 min)
# Construye la **tensión del mercado laboral** $\theta_t=V_t/U_t$ de EE. UU. y grafica contra
# ella la inflación **subyacente** (`CPILFESL`, variación anual) en dos submuestras:
# 2001–2019 y 2021 en adelante.
#
# **Cuidado con las unidades, que es medio entregable.** $V_t$ y $U_t$ tienen que ser los dos
# **niveles**: `JTSJOL.csv` (vacantes, miles) y `UNEMPLOY.csv` (desempleados, miles), ambos ya
# congelados en `data/` (JOLTS arranca en dic-2000, así que la submuestra 2001–2019 sale
# completa). **No** uses `UNRATE.csv` en el denominador: es una **tasa** (%), y el nivel es
# $U_t=u_t\,LF_t/100$, de modo que
#
# $$\frac{\texttt{JTSJOL}}{\texttt{UNRATE}}=\frac{V_t}{u_t}=\theta_t\cdot\frac{LF_t}{100},$$
#
# es decir $\theta_t$ **multiplicado** por la fuerza laboral. Y la fuerza laboral crece a lo
# largo de la muestra, con lo que ese cociente le mete una tendencia a la variable cuya
# **pendiente** justamente te pedimos medir. (Si de todos modos quieres graficar $V/u$, hazlo,
# pero dilo en el texto y no lo llames $\theta$.)
#
# La pregunta: ¿sale una recta plana o una curva que se **empina** cuando $\theta$ es alta
# (Benigno–Eggertsson)? Si te sale no lineal, argumenta por dónde entraría esa no linealidad
# en la derivación de la sección 3: ¿por $\lambda=(\varepsilon-1)/\vartheta$, por el traspaso
# $\psi$, o por ninguna de las dos —es decir, la log-linealización misma es lo que se rompe?
# Entrega una figura y un párrafo.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 8. Explora con IA
# Prueba estas indicaciones con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué con precios flexibles el dinero es neutral aunque haya competencia
#   monopolista? Responde en una frase, usando el margen."
# - "Explica el principio de Taylor sin fórmulas: ¿qué pasa con la tasa **real** si el banco
#   central sube la nominal menos que uno a uno con la inflación?"

# %% slideshow={"slide_type": "fragment"}
print(tutor(
    "En dos frases: por que con precios rigidos a la Rotemberg un choque monetario mueve "
    "cantidades reales, si con precios flexibles y margen constante era neutral?",
    context=(f"Modelo de tres ecuaciones: lambda=(eps-1)/vartheta={LAM:.4f}, "
             f"kappa=lambda*psi={KAPPA:.4f}, phi_pi={PHI_PI}, phi_y={PHI_Y}. "
             f"IRF monetaria: brecha en el impacto {mon['brecha'][0]:+.3f} pp, "
             f"inflacion {mon['pi'][0]:+.3f} pp anualizados."),
))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Partimos del repunte inflacionario de 2021–2023 en EE. UU. y México con series
# congeladas de FRED; mostramos que con **precios flexibles** y margen constante
# $\mathcal{M}=\varepsilon/(\varepsilon-1)$ el costo marginal real queda clavado y el dinero
# es neutral **por construcción**; introdujimos el costo cuadrático de **Rotemberg**,
# derivamos la **curva de Phillips neokeynesiana** $\hat{\pi}_t=\beta E_t\hat{\pi}_{t+1}+\lambda\widehat{mc}_t$
# con $\lambda=(\varepsilon-1)/\vartheta$ y la **verificamos** diferenciando numéricamente la
# condición exacta; resolvimos el **modelo de tres ecuaciones** con
# `puremacro.dsge.klein_solve` y vimos que el **signo del comovimiento producto-precios**
# separa demanda de oferta; dibujamos el **principio de Taylor** como frontera de
# determinación —casi vertical, porque $1-\beta$ es diminuto— y cuantificamos el precio de
# desanclar las expectativas. El modelo falla en la **joroba** de las IRF estimadas y no
# puede arbitrar el papel de las expectativas en 2021–23: eso sigue siendo debate abierto.
#
# **Referencias.** Rotemberg (1982), *Sticky prices in the United States*, JPE 90. · Calvo
# (1983), *Staggered prices in a utility-maximizing framework*, JME 12. · Galí (2015),
# *Monetary Policy, Inflation, and the Business Cycle*, cap. 3. · Taylor (1993). · Bils y
# Klenow (2004), *Some evidence on the importance of sticky prices*, JPE 112. · Clarida,
# Galí y Gertler (2000), QJE 115. · Gertler y Karadi (2015), AEJ: Macroeconomics 7. ·
# Christiano, Eichenbaum y Evans (2005), JPE 113. · Ramey
# (2016), *Macroeconomic shocks and their propagation*, Handbook of Macroeconomics. ·
# Nekarda y Ramey (2020), JMCB 52. · Bernanke y Blanchard (2023), *What caused the U.S.
# pandemic-era inflation?*, Brookings. · Benigno y Eggertsson (2023), NBER WP 31197. ·
# Ball, Leigh y Mishra (2022), NBER WP 30613. · Hazell, Herreño, Nakamura y Steinsson (2022),
# QJE 137.
