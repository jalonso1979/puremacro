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
# # Módulo 4 — RBC: momentos, Slutsky y el reto de México
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Construir la **tabla de momentos del ciclo** (volatilidades relativas, comovimiento,
#    persistencia) con el filtro de **Hamilton** sobre datos reales de EE. UU.
# 2. Entender el **efecto Slutsky**: una media móvil de ruido blanco fabrica un ciclo que
#    a simple vista no se distingue del ciclo real.
# 3. Contrastar el RBC canónico con el **reto mexicano**, enunciado como toca: no que
#    $\sigma_c/\sigma_y$ cruce el $1$ —con la convención canónica del curso **no lo cruza**—
#    sino que esté *tan pegado* al $1$ ($0.96$) cuando el modelo cerrado entrega $\approx0.44$.
#    Es un fallo de **magnitud**, no de signo. Y repasar sus tres salidas: choques de
#    productividad más persistentes ($\sigma_a$), choques a la **tendencia**, y la **tasa de
#    interés país** ($q_t$).
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`): sin conexión y sin costo.

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
# ## 1. Los momentos del ciclo económico
#
# El modelo de **ciclos económicos reales** (RBC) se juzga por su capacidad de reproducir
# un puñado de *momentos*: la **desviación estándar** del producto, las **volatilidades
# relativas** de sus componentes ($\sigma_x/\sigma_y$), el **comovimiento** de cada
# componente con el producto y la **persistencia** (autocorrelación de primer orden) del
# ciclo. Antes de calcularlos hay que separar tendencia y ciclo; aquí usamos el filtro de
# **Hamilton (2018)**, que proyecta $y_{t+h}$ sobre sus rezagos recientes y toma el residuo
# como ciclo (evita las dinámicas espurias del filtro HP).
#
# Datos reales del *bundle*: `GDPC1` (PIB real) y `GPDIC1` (inversión **privada bruta
# interna**, que incluye la variación de existencias), ambos trimestrales, 1947Q1–2026Q1.

# %% slideshow={"slide_type": "fragment"}
from puremacro.cycles import hamilton_filter
from puremacro.data import hp_filter
from puremacro.spectral import business_cycle_band_power

y = pd.read_csv(DATA / "GDPC1.csv")
inv = pd.read_csv(DATA / "GPDIC1.csv")
panel = y.merge(inv, on="observation_date")

log_y = 100.0 * np.log(panel["GDPC1"].to_numpy())     # log-PIB en puntos porcentuales
log_i = 100.0 * np.log(panel["GPDIC1"].to_numpy())    # log-inversión

cyc_y, _ = hamilton_filter(log_y)   # (ciclo, tendencia); las primeras h+p-1 obs son NaN
cyc_i, _ = hamilton_filter(log_i)

ok = ~np.isnan(cyc_y) & ~np.isnan(cyc_i)              # descartar el calentamiento del filtro
cyc_y, cyc_i = cyc_y[ok], cyc_i[ok]

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La tabla de momentos
# Sobre el ciclo de Hamilton: volatilidad absoluta del producto, volatilidad relativa de la
# inversión ($\sigma_i/\sigma_y$), su comovimiento con el producto, la persistencia del PIB
# y la cuota de varianza que cae en la banda de negocios de 6–32 trimestres.

# %% slideshow={"slide_type": "fragment"}
sigma_y = cyc_y.std()
rel_vol_inv = cyc_i.std() / sigma_y
comov_inv = np.corrcoef(cyc_i, cyc_y)[0, 1]
persist_y = np.corrcoef(cyc_y[1:], cyc_y[:-1])[0, 1]
band_share = business_cycle_band_power(cyc_y)

print(f"observaciones usadas                       = {cyc_y.size}")
print(f"sigma_y  (desv. est. del ciclo del PIB, %) = {sigma_y:.2f}")
print(f"sigma_i / sigma_y  (volatilidad relativa)  = {rel_vol_inv:.2f}")
print(f"comovimiento  corr(inversión, PIB)         = {comov_inv:.2f}")
print(f"persistencia  corr(y_t, y_t-1)             = {persist_y:.2f}")
print(f"cuota espectral en banda 6-32 trimestres   = {band_share:.2f}")

# Hechos canónicos: la inversión es mucho más volátil que el producto y muy procíclica.
assert rel_vol_inv > 2.0
assert comov_inv > 0.6
assert 0.0 < band_share <= 1.0

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** La inversión es $3.8$ veces más volátil que el producto y fuertemente
# procíclica ($0.82$); el producto es muy persistente ($0.89$). Estos son justo los blancos
# que un RBC bien calibrado debe acertar.
#
# *Aviso de comparabilidad con el mazo A4.* La tabla del mazo reporta
# $\sigma_i/\sigma_y=1.98$ porque está construida sobre el **agregado unisectorial**, y su
# nota anuncia $\sim3$ con la definición estándar de **inversión fija privada**. Aquí sale
# $3.8$ porque `GPDIC1` es inversión privada bruta **con existencias**, y la variación de
# existencias es el componente más volátil de la contabilidad nacional. Las tres cifras son
# correctas: son tres series distintas. Es exactamente el punto de la ficha de medición.
#
# El componente que *falta* aquí es el **consumo**: en EE. UU. es
# más suave que el producto ($\sigma_c/\sigma_y<1$), el rasgo que la hipótesis de renta
# permanente predice. La pregunta de la parte 3 es **cuánto** más suave, y contra qué vara:
# ahí es donde México y el RBC canónico se separan.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. El ciclo que fabrica el azar (efecto Slutsky)
#
# Slutsky (1927) mostró algo incómodo: si tomas **ruido blanco** —puro azar, sin ninguna
# estructura cíclica— y lo pasas por una **media móvil**, aparecen ondas suaves y
# recurrentes que *parecen* un ciclo económico. La suma de choques independientes,
# promediada, induce persistencia y ondulación de la nada. La pregunta filosa: ¿podemos
# distinguir ese ciclo espurio del ciclo real del PIB?

# %% slideshow={"slide_type": "fragment"}
rng = np.random.default_rng(20260721)   # datos SIMULADOS: ruido blanco (declarado)
T = cyc_y.size
k = 8                                   # ventana de la media móvil, en trimestres (= el mazo A4)
# Pedimos T+k-1 innovaciones y convolucionamos en modo "valid": así la media móvil tiene
# exactamente T observaciones y NINGUNA está contaminada por los bordes. Con mode="same"
# las primeras y últimas k/2 serían sumas parciales divididas por k, artificialmente planas.
ruido = rng.standard_normal(T + k - 1)               # ruido blanco iid
ma = np.convolve(ruido, np.ones(k) / k, mode="valid")
ma = ma / ma.std()                                   # normalizar a varianza 1
cyc_y_norm = cyc_y / cyc_y.std()                     # ciclo real normalizado, para comparar
assert ma.size == T

persist_ma = np.corrcoef(ma[1:], ma[:-1])[0, 1]
band_ruido = business_cycle_band_power(ruido)
band_ma = business_cycle_band_power(ma)
print(f"persistencia  ciclo real del PIB           = {persist_y:.2f}")
print(f"persistencia  media móvil de ruido blanco  = {persist_ma:.2f}")
print(f"banda 6-32t   ruido blanco crudo           = {band_ruido:.2f}")
print(f"banda 6-32t   media móvil de ruido blanco  = {band_ma:.2f}")
print(f"banda 6-32t   ciclo real del PIB           = {band_share:.2f}")

# El azar promediado alcanza una persistencia comparable a la del ciclo real...
assert persist_ma > 0.7
assert abs(persist_ma - persist_y) < 0.05
# ...y una cuota espectral en la banda de negocios mucho más cerca del ciclo real que del
# ruido crudo: la media móvil es un filtro pasa-bajas, y ahí es donde vive el ciclo.
assert band_ruido < band_ma < band_share

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Los números, antes de la figura
# La media móvil de $8$ trimestres (la misma ventana del mazo A4) alcanza persistencia
# $0.88$ contra $0.89$ del ciclo real: **indistinguibles**. Y el promediado sube la cuota
# espectral en la banda de negocios de $0.29$ —el valor del ruido crudo, que es sólo el
# ancho relativo de la banda $6$–$32$ en $[0,0.5]$ ciclos por trimestre— a $0.53$, contra
# $0.63$ del ciclo real: una media móvil es un filtro **pasa-bajas** y deposita la potencia
# justo donde vive el ciclo. Ninguno de los dos momentos univariados separa el azar
# promediado del ciclo del PIB. Ojo con la ventana: **es $k$ quien fija el "periodo" del
# ciclo espurio**. Con $k=20$ la persistencia sube a $0.95$ y la cuota de banda **cae** a
# $0.29$, porque el ciclo fabricado se va más allá de los $32$ trimestres. Cambiar el
# filtrado cambia el ciclo: ésa es la incomodidad de Slutsky.
#
# ### ¿Cuál es cuál?
# Arriba, el ciclo real del PIB (Hamilton); abajo, la media móvil de ruido blanco. Ambos
# ondulan con periodos de auge y recesión aparentes. La moraleja de Slutsky: *ver* ondas no
# prueba que exista un mecanismo cíclico — un RBC debe ganarse la vida en los **momentos**,
# no en el parecido visual.

# %% slideshow={"slide_type": "slide"}
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True)
ax0.plot(cyc_y_norm, color="0.15", lw=1.4)
ax0.axhline(0, color="0.85", lw=0.6)
ax0.set_ylabel("desv. est.")
ax0.set_title("A. Ciclo real del PIB (filtro de Hamilton)")
ax1.plot(ma, color="0.55", lw=1.4, ls=(0, (4, 2)))
ax1.axhline(0, color="0.85", lw=0.6)
ax1.set_ylabel("desv. est."); ax1.set_xlabel("trimestre")
ax1.set_title("B. Media móvil de ruido blanco (efecto Slutsky)")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. El reto mexicano: no es el signo, es la magnitud
#
# Aquí hay una frase de libro de texto que conviene desarmar antes de repetirla: *"en las
# emergentes el consumo es más volátil que el producto, $\sigma_c/\sigma_y>1$"*. Para México
# **es falsa** con la convención de medición canónica del curso. Vamos a medirlo, no a
# creerlo.
#
# ### La ficha de medición (sin ella el número no es replicable)
#
# | decisión | convención del curso |
# |---|---|
# | **fuente y edición** | OCDE QNA vía SDMX, en el caché congelado `data/oecd_qna_apertura.csv` que viaja con las lecciones (última observación 2026Q2) |
# | serie de consumo | **hogares e ISFLSH** (P3, S1M), volúmenes |
# | serie de producto | PIB (B1GQ, S1), volúmenes |
# | **base de precios** | **base FIJA** (`PRICE_BASE=Q`, precios constantes, año base 2018) para México — el único de los 21 así; los demás, volúmenes **encadenados** (`L`) |
# | transformación | $100\log$ |
# | **muestra** | **1995Q1–2019Q4** |
# | orden de las operaciones | **recortar a la ventana y luego filtrar** |
# | filtro | HP $\lambda=1600$ (principal) y Hamilton $(8,4)$ (robustez) |
#
# Muevan un solo campo y el $0.96$ mexicano llega hasta $1.11$ —ahí sí cruza el 1—. Que el
# *signo* del "hecho estilizado" dependa de la ficha es el ejercicio, no un accidente.
#
# Sobre la base de precios: el campo está **declarado y además contestado**. El mazo
# Slides07 reconstruye un encadenado para México y el cociente se mueve $0.001$, así que la
# base fija no es la explicación del $0.96$; pero declararla es obligatorio (es uno de los
# seis campos de la ficha) y evita comparar peras con manzanas contra los otros 20 países.

# %% slideshow={"slide_type": "fragment"}
crudo = pd.read_csv(DATA / "oecd_qna_apertura.csv", parse_dates=["date"])

# La base de precios es un campo de la ficha: se COMPRUEBA, no se afirma de palabra.
bases = crudo.groupby("code")["price_base"].apply(lambda s: set(s.unique()) - {"V"})
print("base de precios por país en el caché:", {c: sorted(b) for c, b in bases.items()})
assert bases["MEX"] == {"Q"}                                   # México, base FIJA
assert all(b == {"L"} for c, b in bases.items() if c != "MEX")  # los demás, encadenados

qna = crudo.pivot_table(index=["code", "date"], columns="variable", values="value")

INI_C, FIN_C = "1995-01-01", "2019-10-01"     # ventana CANÓNICA del curso
RBC_CERRADO = 0.44        # sigma_c/sigma_y del RBC cerrado calibrado a México (mazo Slides04,
                          # .mod de Dynare de la parte modelo de la Tarea 2)


def cociente(code, filtro="hp"):
    # sigma_c/sigma_y con la convención canónica: RECORTAR a la ventana y luego FILTRAR.
    d = qna.loc[code].sort_index().dropna(subset=["gdp_vol", "conh_vol"])
    d = d[(d.index >= INI_C) & (d.index <= FIN_C)]
    ly = 100 * np.log(d["gdp_vol"].to_numpy())
    lc = 100 * np.log(d["conh_vol"].to_numpy())
    if filtro == "hp":
        cy, _ = hp_filter(ly); cc, _ = hp_filter(lc)
        cy, cc = np.asarray(cy), np.asarray(cc)
    else:
        cy, _ = hamilton_filter(ly); cc, _ = hamilton_filter(lc)
        ok = ~np.isnan(cy) & ~np.isnan(cc)
        cy, cc = np.asarray(cy)[ok], np.asarray(cc)[ok]
    return cy.std(), cc.std() / cy.std()


print("sigma_c/sigma_y, OCDE QNA, hogares vs PIB, volúmenes, 1995Q1-2019Q4")
print("(base FIJA 'Q', precios de 2018, para MEX; encadenada 'L' para los demás)")
print("(recorte ANTES de filtrar; ambas series a la misma ventana)\n")
print(f"{'país':<6}{'HP(1600)':>12}{'Hamilton(8,4)':>16}{'sigma_y (HP)':>15}")
dato = {}
for code in ["MEX", "USA"]:
    sy_hp, r_hp = cociente(code, "hp")
    _, r_ha = cociente(code, "hamilton")
    dato[code] = r_hp
    print(f"{code:<6}{r_hp:>12.3f}{r_ha:>16.3f}{sy_hp:>15.2f}")
print(f"\nRBC cerrado calibrado a México (mazo Slides04):  {RBC_CERRADO:.2f}")
print(f"brecha modelo cerrado vs México = {dato['MEX'] - RBC_CERRADO:.2f} "
      f"({dato['MEX'] / RBC_CERRADO:.1f} veces)")

# EL HECHO, enunciado como toca: el consumo mexicano NO es más volátil que el producto con
# esta convención; lo que sorprende es lo cerca que está del 1 y lo lejos que está el modelo.
assert dato["MEX"] < 1.0
assert abs(dato["MEX"] - 0.959) < 0.01 and abs(dato["USA"] - 0.795) < 0.01
assert dato["MEX"] > dato["USA"] > RBC_CERRADO

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lo que dicen los números
# México: $\sigma_c/\sigma_y=0.96$ con HP y $0.97$ con Hamilton. EE. UU., con la **misma**
# serie y la **misma** ventana: $0.79$ y $0.85$. El RBC cerrado calibrado a México entrega
# $\approx0.44$. Entonces:
#
# - El fallo del modelo **no es de signo**: el modelo predice $<1$ y el dato es $<1$. Los dos
#   están del mismo lado del 1.
# - El fallo es de **magnitud, y es enorme**: el modelo suaviza el consumo más del **doble**
#   de lo que lo suaviza México. Esa brecha de ~$0.5$ es el reto.
# - El contraste con EE. UU. sobrevive intacto ($0.96$ contra $0.79$): México suaviza menos.
#   Eso es lo que hay que explicar, y no necesita ningún $>1$.
#
# Dos advertencias de honestidad, porque el curso las va a cobrar:
#
# 1. **En sección cruzada México es mediano.** En un panel de 21 países con esta misma
#    convención, la mediana de $\sigma_c/\sigma_y$ es $0.957$ (Costa Rica) y México sale
#    $0.959$: **el lugar 12 de 21 ordenando de menor a mayor**, es decir, prácticamente la
#    mediana misma. El cociente por sí solo no separa emergentes de
#    avanzadas. Sí hay emergentes claramente por encima del 1 (Corea $1.55$, Chile $1.31$),
#    pero México no es uno de ellos con esta ficha.
# 2. **El comovimiento es alto, pero tampoco es récord.** $\mathrm{corr}(c,y)$ cíclica con
#    HP: Corea $0.92$, Chile $0.90$, España $0.88$, EE. UU. $0.87$, Portugal $0.86$, México
#    $0.85$ — México es el **sexto** de los 21, con mediana $0.78$. Alto, sí; excepcional, no.
#
# Moraleja metodológica: un hecho estilizado enunciado como desigualdad (*"$>1$"*) es frágil
# —lo decide la ficha de medición—; enunciado como magnitud (*"$0.96$ contra $0.44$ del
# modelo"*) es robusto y además dice cuánto tiene que trabajar la teoría.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El mecanismo candidato: "the cycle is the trend"
# ¿De dónde puede salir un cociente tan alto? Aguiar y Gopinath (2007) proponen que en las
# emergentes los choques golpean la **tasa de crecimiento tendencial**, no sólo el nivel. Si
# el choque es casi permanente, el ingreso futuro esperado sube tanto como el actual y el
# hogar **no tiene motivo para suavizar**: el consumo sigue al producto de cerca, o incluso
# lo rebasa.
#
# Lo ilustramos con una simulación de renta permanente (datos **SIMULADOS**, declarados).
# Ojo con la lectura: las economías simuladas son **regímenes de choques**, no retratos de
# países, y esto **no** es el RBC: es una regla de consumo escrita a mano, no la solución de
# un problema de optimización. Lo único que autoriza el ejercicio es la **estática
# comparativa** —qué le pasa al cociente cuando sube el peso del choque de tendencia—, no el
# nivel de ninguna de las barras. Ajustar el nivel a la cifra mexicana es trabajo del mazo de
# economía abierta.

# %% slideshow={"slide_type": "fragment"}
def economia(sig_g, sig_z, *, rho_g=0.8, theta=0.35, seed=0, T=400):
    """Renta permanente à la Aguiar-Gopinath (2007), SIMULADA.

    - g_t: choque persistente a la TASA de crecimiento (AR(1)), acumulado en la tendencia.
    - z_t: choque transitorio al NIVEL del producto.
    El consumo sigue el componente permanente y anticipa el crecimiento futuro (kappa*g_t),
    por lo que sobre-reacciona cuando dominan los choques a la tendencia.

    OJO, esto NO es el RBC resuelto: es una regla de consumo POSTULADA. En particular
    `theta` es la fracción del choque transitorio que se consume, un PARÁMETRO LIBRE, no un
    resultado. Con sig_g=0 se tiene log_con = theta*log_out término a término y, como el HP
    es lineal, el cociente sale EXACTAMENTE theta. Lo informativo aquí es cómo se mueve el
    cociente al variar sig_g, no su nivel.
    """
    r = np.random.default_rng(seed)
    g = np.zeros(T)
    for t in range(1, T):
        g[t] = rho_g * g[t - 1] + r.normal(0.0, sig_g)   # choque persistente al crecimiento
    X = np.cumsum(g)                                      # tendencia estocástica (log-nivel)
    z = r.normal(0.0, sig_z, T)                           # choque transitorio al nivel
    log_out = X + z
    kappa = rho_g / (1.0 - rho_g)                         # valor presente del crecimiento
    log_con = X + kappa * g + theta * z                  # consumo de renta permanente
    cy, _ = hp_filter(log_out); cc, _ = hp_filter(log_con)
    cy, cc = np.asarray(cy), np.asarray(cc)
    return cc.std() / cy.std(), np.corrcoef(cy, cc)[0, 1]

rel_pu, corr_pu = economia(sig_g=0.0, sig_z=1.0, seed=101)    # sin tendencia: transitorio puro
rel_tr, corr_tr = economia(sig_g=0.15, sig_z=1.0, seed=101)   # mezcla: algo de tendencia
rel_te, corr_te = economia(sig_g=1.0, sig_z=0.4, seed=202)    # régimen: tendencia domina

print(f"transitorio PURO (sig_g=0):   sigma_c/sigma_y = {rel_pu:.2f}   corr(c,y) = {corr_pu:.2f}")
print(f"mezcla (sig_g=0.15):          sigma_c/sigma_y = {rel_tr:.2f}   corr(c,y) = {corr_tr:.2f}")
print(f"tendencia domina (sig_g=1):   sigma_c/sigma_y = {rel_te:.2f}   corr(c,y) = {corr_te:.2f}")
print(f"\ndato México (HP, ventana canónica) = {dato['MEX']:.2f}"
      f"   |   RBC cerrado del mazo = {RBC_CERRADO:.2f}")

# Comprobación de que el 0.35 es theta y no un hallazgo: con sig_g=0 el cociente ES theta.
assert abs(rel_pu - 0.35) < 1e-10 and abs(corr_pu - 1.0) < 1e-10
assert rel_pu < rel_tr < rel_te        # LO QUE SÍ dice el ejercicio: la tendencia sube el
                                       # cociente, y mucho (estática comparativa)
assert corr_tr < corr_pu and corr_te < corr_pu   # ...y a costa de bajar el comovimiento
assert rel_tr < dato["MEX"] < rel_te   # el dato mexicano queda ENTRE la mezcla y la tendencia

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La escala del problema, en una figura
# Las cinco barras están en la misma escala y ordenan el argumento entero. **Léanlas como
# pendientes, no como niveles.** Sin choques a la tendencia el cociente simulado es $0.35$,
# que es *exactamente* el $\theta$ que le pusimos a la regla de consumo: con $\sigma_g=0$ el
# consumo es $\theta$ veces el producto en cada periodo y el filtro HP es lineal, así que ese
# $0.35$ es un **supuesto**, no un resultado — está ahí sólo para fijar la escala del caso
# transitorio, y **no valida** el $0.44$ del RBC (ese sí sale de resolver el modelo
# calibrado). Lo que sí es un resultado: basta un componente **pequeño** de tendencia
# ($\sigma_g=0.15$ contra $\sigma_z=1$) para subir a $0.91$, y con la tendencia dominando el
# cociente se dispara a $1.75$, muy por encima de México — el mismo $\theta$ en los tres
# casos.
#
# Nótese dónde queda la línea del $1$: el dato mexicano **no la cruza**, y aun así la
# distancia al modelo cerrado es de **medio punto**. El reto es esa distancia, no la línea.
#
# Y una honestidad más, que la salida imprime y conviene no saltarse: el canal de la
# tendencia sube el cociente pero **baja el comovimiento**, de $1.00$ en el caso transitorio
# puro a $0.61$ y $0.69$, mientras México tiene $\mathrm{corr}(c,y)=0.85$. Es decir, el canal
# empuja el momento que nos interesa en la dirección correcta y otro momento en la dirección
# equivocada. Cerrar los dos a la vez es justo lo que exige el modelo de economía abierta.

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(8.4, 3.9))
labels = [r"simulación: transitorio" "\n" r"puro (= $\theta$, supuesto)", "RBC cerrado\n(mazo)",
          "simulación:\nmezcla", "DATO México\n(HP, 1995–2019)",
          "simulación:\ntendencia domina"]
vals = [rel_pu, RBC_CERRADO, rel_tr, dato["MEX"], rel_te]
colores = ["0.72", "0.86", "0.58", "0.10", "0.40"]
bars = ax.bar(labels, vals, color=colores, width=0.58)
ax.axhline(1.0, color="0.0", lw=1.0, ls=(0, (4, 2)))
ax.text(4.44, 1.02, r"$\sigma_c/\sigma_y=1$", va="bottom", ha="right", fontsize=10)
ax.annotate("", xy=(1, RBC_CERRADO), xytext=(1, dato["MEX"]),
            arrowprops=dict(arrowstyle="<->", color="0.0", lw=1.1))
ax.text(1.12, (RBC_CERRADO + dato["MEX"]) / 2, "el reto\n(magnitud)", fontsize=8, va="center")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
ax.tick_params(axis="x", labelsize=8)
ax.set_ylabel(r"$\sigma_c\,/\,\sigma_y$")
ax.set_title("El reto no es cruzar el 1: es la brecha entre $0.44$ y $0.96$")
plt.tight_layout()
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Las tres salidas del reto
# Para que un RBC cierre la brecha de magnitud —de $\approx0.44$ hasta el $0.96$ mexicano— la
# literatura ofrece tres puertas (no excluyentes):
# 1. **$\sigma_a$ / persistencia de la productividad.** Choques de productividad más
#    grandes y persistentes hacen que la **renta del capital** y el ingreso permanente se
#    muevan más, elevando la volatilidad del consumo.
# 2. **La tendencia.** Choques a la *tasa de crecimiento* tendencial (Aguiar–Gopinath): el
#    ingreso futuro esperado sube y el consumo sobre-reacciona hoy. Es la salida que
#    simulamos arriba.
# 3. **La tasa de interés país ($q_t$).** Un *spread* soberano contracíclico (sube en
#    recesión) más fricciones financieras y **tipo de cambio** amplifican los choques y
#    desincronizan consumo y producto — canal central en la macro de economías abiertas
#    emergentes.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Preguntas para pensar
# 1. En la parte 1, la persistencia del ciclo real del PIB (~0.9) y la de la media móvil de
#    ruido blanco (parte 2) son casi iguales. ¿Qué momento **adicional** propondrías para
#    distinguir un ciclo con mecanismo económico de uno puramente estadístico (Slutsky)?
# 2. La renta permanente predice $\sigma_c/\sigma_y<1$ con choques transitorios pero puede
#    dar $>1$ con choques a la tendencia. ¿Por qué el consumo **anticipa** el crecimiento
#    futuro y salta por encima del producto? Explica el papel de $\kappa=\rho/(1-\rho)$.
# 3. **El hecho, bien enunciado.** El dato mexicano es $0.96$ y el modelo cerrado $0.44$:
#    ¿por qué es más informativo reportar esa brecha que la desigualdad "$\sigma_c/\sigma_y>1$"?
#    Escribe las dos versiones del hecho estilizado y di, para cada una, qué tendría que
#    ocurrir en los datos para **falsarla**. Pista: una depende de la ficha de medición y la
#    otra no.
# 4. De las tres salidas ($\sigma_a$, tendencia, $q_t$), ¿cuál te parece más plausible para
#    México en la crisis de 1994–95 y por qué? Piensa en el **tipo de cambio** y la **tasa
#    de interés país**. Ojo con la salida 1: subir $\sigma_a$ mueve numerador **y**
#    denominador — ¿por qué eso la hace mala candidata para cerrar la brecha?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Notas para las preguntas
# 1. La persistencia no discrimina porque cualquier suavizamiento la fabrica ($0.88$ contra
#    $0.89$). Tampoco basta con irse al dominio de la frecuencia: la lección imprime la
#    cuota de banda $6$–$32$ y da $0.53$ para la media móvil contra $0.63$ para el ciclo
#    real — más cerca del ciclo que del ruido crudo ($0.29$), y además esa cifra se mueve a
#    voluntad cambiando la ventana $k$. **Ningún momento univariado de una sola serie va a
#    resolverlo**, porque el problema es que la media móvil tiene, por construcción, los
#    mismos dos grados de libertad (varianza y persistencia) que basta ajustar.
#    La salida son los momentos **multivariados**: comovimiento $\mathrm{corr}(c,y)$,
#    $\mathrm{corr}(i,y)$ y las volatilidades **relativas** ($\sigma_i/\sigma_y\approx4$,
#    $\sigma_c/\sigma_y<1$; aquí $\sigma_i/\sigma_y=3.8$). Una media móvil de ruido blanco
#    no tiene ninguna razón para
#    ordenar así **varias series a la vez**; un mecanismo económico sí. Adicionalmente, la
#    **forma** de la función de impulso-respuesta a un choque identificado (lección 05).
# 2. Con $g_t$ AR(1) de coeficiente $\rho$, un choque hoy implica crecimiento esperado
#    también mañana: el valor presente del crecimiento futuro es
#    $\kappa=\rho/(1-\rho)$ veces el choque corriente. El ingreso **permanente** sube más
#    que el producto corriente, así que el consumo salta por encima de $y_t$ y
#    $\sigma_c/\sigma_y$ puede pasar de 1. Con $\rho=0$ ($\kappa=0$) se recupera el caso
#    transitorio y el suavizamiento clásico.
# 3. Versión-desigualdad: "$\sigma_c/\sigma_y>1$ en emergentes". Se falsa con **una sola**
#    ficha de medición distinta (México da $0.96$ con la canónica y $1.11$ moviendo un
#    campo): es frágil porque el signo depende de convenciones. Versión-magnitud: "el dato
#    es $0.96$ y el modelo cerrado $0.44$". Se falsa sólo si alguna ficha razonable llevara
#    el dato cerca de $0.44$ o el modelo cerca de $0.96$ — nada de eso ocurre, así que el
#    hecho sobrevive a la ficha y además cuantifica cuánto debe trabajar la teoría.
# 4. Para 1994–95 la salida 3 ($q_t$) es la más plausible: el colapso del tipo de cambio y
#    el salto del *spread* soberano son observables y contracíclicos, y desincronizan
#    consumo y producto sin necesidad de tocar la tecnología. La salida 1 es mala candidata
#    justamente por ser un cociente: subir $\sigma_a$ escala numerador y denominador casi en
#    la misma proporción, así que sube $\sigma_y$ sin mover $\sigma_c/\sigma_y$. La salida 2
#    (tendencia) sí mueve el cociente —lo vimos en la simulación— pero exige creer que los
#    choques mexicanos son casi permanentes, algo que el propio Aguiar–Gopinath discute.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 5. Explora con IA
# Prueba estas indicaciones con el tutor sin conexión (o cualquier asistente de IA):
# - "En una frase, ¿por qué el efecto Slutsky implica que 'ver' ciclos no prueba que exista
#   un mecanismo cíclico?"
# - "El consumo mexicano tiene $\sigma_c/\sigma_y=0.96$ y el RBC cerrado da $0.44$. ¿Por qué
#   el problema del modelo es de magnitud y no de signo? Da la intuición de Aguiar–Gopinath
#   en dos frases."

# %% slideshow={"slide_type": "fragment"}
print(tutor(
    "En una o dos frases, explica por qué el consumo de México es casi tan volátil como su "
    "producto (sigma_c/sigma_y = 0.96) cuando el RBC cerrado predice 0.44, y por qué el "
    "problema es de MAGNITUD y no de signo.",
    context=(f"Dato México (HP, hogares, 1995Q1-2019Q4, base fija 'Q') = {dato['MEX']:.2f}; "
             f"EE. UU. misma convención (base encadenada 'L') = {dato['USA']:.2f}; "
             f"RBC cerrado del mazo = {RBC_CERRADO:.2f}. "
             f"Simulación: transitorio puro={rel_pu:.2f}, mezcla={rel_tr:.2f}, tendencia={rel_te:.2f}. "
             f"Momentos reales EE. UU. (Hamilton): sigma_i/sigma_y={rel_vol_inv:.2f}, "
             f"persistencia PIB={persist_y:.2f}."),
))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Armamos la tabla de momentos del ciclo con el filtro de Hamilton sobre datos
# reales (inversión $3.8\times$ más volátil que el producto, muy procíclica y persistente),
# vimos que una media móvil de ruido blanco de $8$ trimestres fabrica un ciclo que ningún
# momento **univariado** distingue del real —misma persistencia ($0.88$ contra $0.89$) y
# cuota de banda del mismo orden ($0.53$ contra $0.63$)— y que sólo los momentos
# **multivariados** lo delatan (**Slutsky**); y medimos el **reto mexicano** con su ficha
# completa: $\sigma_c/\sigma_y=0.96$
# (HP) o $0.97$ (Hamilton) sobre 1995Q1–2019Q4, contra $0.79$/$0.85$ de EE. UU. con la misma
# serie y ventana, y contra $\approx0.44$ del RBC cerrado. El consumo mexicano **no** es más
# volátil que el producto: el fallo del modelo es de **magnitud** —suaviza más del doble de
# lo que suaviza México— y ésa, no la desigualdad "$>1$", es la brecha que las tres salidas
# ($\sigma_a$, tendencia, $q_t$) tienen que cerrar. Todo en `puremacro` (Python puro),
# sobre tu instalación local.
