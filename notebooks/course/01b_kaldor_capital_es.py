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
# # Módulo 1b — Kaldor, capital y participación del trabajo
#
# **Curso complementario · puremacro · mazo Slides01 — medición del ciclo (semanas 1–2)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Enunciar los **hechos estilizados de Kaldor** y **comprobar en los datos** que la
#    razón $I/Y$ *real* de EUA **sí tiene tendencia** (el hecho 2 es de razones
#    nominales; la real y la nominal solo se reconcilian por el precio relativo).
# 2. Construir el **acervo de capital por inventario perpetuo** y descubrir que el
#    cociente $K/Y$ así reconstruido **se duplica**: el hecho 4 aguanta en el acervo de
#    la BEA, **no** en cualquier reconstrucción hecha en casa.
# 3. Distinguir **senda de crecimiento balanceada** de **transición**: por qué la
#    condición inicial de $K$ se lava — y por qué eso no basta para salvar $K/Y$.
# 4. Medir la participación del trabajo con EU-KLEMS y ver que en estas seis economías
#    **sube en cinco de seis** (1995–2020): el signo del hecho 3 depende de la medición.
# 5. Detectar una **trampa de agregación** (ponderar por valor agregado en monedas
#    distintas) y situar bien el hecho de **Karabarbounis–Neiman (2014)**.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), con los datos congelados del *bundle*: sin conexión y sin costo.

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


def fred(name):
    """Lee un CSV FRED del bundle (columnas observation_date, VALOR) como serie
    trimestral/anual indexada por fecha. Nunca toca la red."""
    d = pd.read_csv(DATA / f"{name}.csv")
    d.columns = ["date", name]
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[name].astype(float)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. Los hechos de Kaldor
#
# Kaldor (1961) resumió el crecimiento de largo plazo en unos pocos hechos que un
# modelo debe reproducir. Los que nos importan hoy son las **grandes razones**
# aproximadamente constantes:
# $$\frac{C}{Y},\quad \frac{I}{Y},\quad \frac{K}{Y}\ \text{(constante)},\qquad
# \text{y la participación del trabajo } s_L = \frac{wL}{Y}.$$
# En una **senda balanceada**, $Y$, $C$, $K$ crecen a la misma tasa, así que estas
# razones no muestran tendencia. La pregunta empírica: **¿se cumplen?** Y la respuesta,
# ya lo verá, depende de **con qué serie se midan** — que es la lección del mazo A1.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Datos: un ancla empírica y un panel simulado
# El *bundle* trae la inversión y el producto **reales** de **EUA** (`GPDIC1`, `GDPC1`,
# volúmenes encadenados, ambos a tasa anual), con los que calculamos la razón $I/Y$
# *real*. **Ojo con lo que sale: no es plana.** La razón real **sube** de forma secular
# (véase la salida). Ese es exactamente el punto de la *segunda capa* del mazo A1: la
# $I/Y$ **nominal** de EUA sí es aproximadamente constante, la **real crece**, y ambas
# solo se reconcilian si el **precio relativo de la inversión cae** (sección 3). El
# *bundle* no trae las series nominales, así que aquí verificamos **solo la mitad real**
# del argumento; la mitad nominal viaja en el mazo.
#
# Para el **corte entre países** de $C/Y$ e $I/Y$ **no hay serie en el bundle** (el
# archivo real, `data_curso/oecd_qna_kaldor.csv` con MEX/USA/KOR, acompaña al mazo, no
# al ZIP del alumno), así que **simulamos** un panel con `np.random.default_rng`
# (semilla fija). Es un **dibujo, no evidencia**: no autoriza ninguna afirmación sobre
# ningún país. Lo calibramos para que enseñe la distinción que importa y que sí está en
# los datos reales del mazo: cuatro países con razón **sin tendencia** (senda
# balanceada), **Corea** con $I/Y$ alta y **decreciente** (transición) y **México** con
# media parecida pero **el doble de volátil**. El asterisco de las etiquetas recuerda
# que son series inventadas.

# %% slideshow={"slide_type": "fragment"}
# --- EUA: razón inversión/producto real desde el bundle (ancla empírica) ---
us = pd.concat([fred("GDPC1"), fred("GPDIC1")], axis=1).dropna()
iy_us = (us["GPDIC1"] / us["GDPC1"]).rename("I/Y")   # ambas a tasa anual: la razón es válida
dec_us = iy_us.groupby(iy_us.index.year // 10 * 10).mean()   # medias por década

# --- Corte entre países: SIMULADO y declarado como tal (el * lo recuerda) ---
rng = np.random.default_rng(20260721)
paises = ["MEX*", "USA*", "DEU*", "KOR*", "BRA*", "IND*"]   # * = simulado, NO son datos
anios = np.arange(1962, 2021)
iy_mean = {"MEX*": 0.22, "USA*": 0.20, "DEU*": 0.21, "KOR*": 0.30, "BRA*": 0.19, "IND*": 0.28}
cy_mean = {"MEX*": 0.66, "USA*": 0.67, "DEU*": 0.55, "KOR*": 0.55, "BRA*": 0.63, "IND*": 0.60}
sd = dict.fromkeys(paises, 0.012); sd["MEX*"] = 0.024       # MEX*: el doble de volátil
pend = dict.fromkeys(paises, 0.0);  pend["KOR*"] = -0.10    # KOR*: transición, -10 pts
x = (anios - anios.mean()) / (anios[-1] - anios[0])         # reloj en [-0.5, 0.5]
IY_sim, CY_sim = {}, {}
for c in paises:
    e = np.zeros(anios.size)
    for t in range(1, anios.size):                      # ruido AR(1) persistente, sin deriva
        e[t] = 0.80 * e[t - 1] + rng.standard_normal()
    e = sd[c] * e / e.std()
    IY_sim[c] = iy_mean[c] + pend[c] * x + e
    CY_sim[c] = cy_mean[c] - pend[c] * x - e             # C e I se mueven en sentido opuesto
IY_sim, CY_sim = pd.DataFrame(IY_sim, index=anios), pd.DataFrame(CY_sim, index=anios)

print("EUA, I/Y REAL por década (bundle) — NO es plana:")
print(dec_us.round(3).to_string())
print(f"  {iy_us.index[0].year} = {iy_us.iloc[0]:.3f}  ->  {iy_us.index[-1].year} = "
      f"{iy_us.iloc[-1]:.3f}   (x{iy_us.iloc[-1] / iy_us.iloc[0]:.2f});  "
      f"media {iy_us.mean():.3f}, coef. variación {iy_us.std() / iy_us.mean():.2f}")
print("\nsimulado (NO son datos) — I/Y: tendencia impuesta y desv. típica, en puntos:")
print(pd.DataFrame({"tendencia": 100 * pd.Series(pend),
                    "sd": 100 * pd.Series(sd)}).round(1).to_string())
# El hecho 2 NO se cumple en la razón REAL de EUA: casi se duplica entre décadas.
assert dec_us.iloc[-1] > 1.5 * dec_us.iloc[0]
planos = [c for c in paises if pend[c] == 0.0]
assert (IY_sim[planos].max() - IY_sim[planos].min()).max() < 0.12   # los planos, sin tendencia
assert np.polyfit(anios, IY_sim["KOR*"].to_numpy(), 1)[0] < -0.001  # KOR*: transición a la baja
assert IY_sim["MEX*"].std() > 1.5 * IY_sim["USA*"].std()            # MEX*: más volátil

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(len(paises))
fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.2, 3.6))
a0.plot(iy_us.index, iy_us.values, color="0.15", lw=1.3)
a0.plot(pd.to_datetime([f"{y}-07-01" for y in dec_us.index]), dec_us.values,
        color="0.55", lw=1.0, ls=(0, (4, 2)), marker="o", ms=3, label="media decenal")
a0.set_title("EUA: I/Y real (bundle) — sube"); a0.set_xlabel("año")
a0.set_ylabel("participación"); a0.legend(loc="upper left", fontsize=8)
for c, col in zip(paises, cols):
    a1.plot(anios, IY_sim[c].values, color=col, lw=1.2, label=c)
a1.set_title("I/Y: panel SIMULADO (no son datos)"); a1.set_xlabel("año")
a1.legend(loc="upper left", ncol=2, fontsize=8)
fig.suptitle("La I/Y real de EUA tiene tendencia; el panel simulado ilustra "
             "constancia vs transición")
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# La serie real de EUA (izquierda) **no** es plana: la media decenal pasa de $\approx
# 0.106$ en los cuarenta a $\approx 0.185$ en los veinte — casi el doble. El hecho 2 de
# Kaldor está enunciado sobre razones **nominales**; medido en **volúmenes encadenados**
# el resultado se invierte, y solo se reconcilia con el desplome del precio relativo de
# la inversión (sección 3). *Antes de invocar un hecho estilizado, pregunte con qué
# serie se midió.*
#
# El panel de la derecha es **simulado**: no dice nada de México ni de Corea. Lo que
# ilustra es la distinción conceptual —cuatro razones sin tendencia (senda balanceada),
# una decreciente (transición), una más volátil— que en los datos reales de la OCDE del
# mazo A1 aparece como *Corea con $I/Y$ alta y decreciente, México baja y volátil*. **Los
# hechos de Kaldor son hechos de senda balanceada, no leyes.**

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Capital por inventario perpetuo
#
# No observamos $K$ directamente, pero sí la inversión. El **método de inventario
# perpetuo** lo acumula:
# $$K_{t+1} = (1-\delta)\,K_t + I_t.$$
# Con $I_t$ la inversión real trimestral (flujo = tasa anual $/4$) y $\delta$ la
# depreciación trimestral. La **condición inicial** $K_0$ se fija en su valor de senda
# balanceada, $K_0 = I_0/(g+\delta)$, con $g$ el crecimiento medio de la inversión —
# es el **método de Hall**, y **supone que EUA ya estaba en su senda balanceada en
# 1947**. Guarde ese supuesto: la validación con $K/Y$ va a delatarlo.

# %% slideshow={"slide_type": "fragment"}
I_q = us["GPDIC1"].to_numpy() / 4.0        # flujo trimestral (la fuente está a tasa anual)
Y = us["GDPC1"].to_numpy()                 # producto a tasa anual
delta_a = 0.06                             # depreciación anual del acervo agregado
delta_q = 1 - (1 - delta_a) ** 0.25        # ~0.0153 trimestral
g_q = np.mean(np.diff(np.log(us["GPDIC1"].to_numpy())))   # crecimiento medio de la inversión


def inventario_perpetuo(I, K0, d):
    K = np.empty(I.size); K[0] = K0
    for t in range(1, I.size):
        K[t] = (1 - d) * K[t - 1] + I[t - 1]
    return K


K0_bgp = I_q[0] / (g_q + delta_q)          # arranque de senda balanceada
KY = {}
for etq, mult in [("K0 = BGP", 1.0), ("K0 = ½·BGP", 0.5), ("K0 = 2·BGP", 2.0)]:
    KY[etq] = inventario_perpetuo(I_q, mult * K0_bgp, delta_q) / Y

fin = {k: v[-1] for k, v in KY.items()}
ky_bgp = pd.Series(KY["K0 = BGP"], index=us.index)
print(f"delta trimestral = {delta_q:.4f}   crecimiento medio inversión g = {g_q:.4f}")
print(f"K/Y en {us.index[-1].year}T{us.index[-1].quarter} según la condición inicial:")
for k, v in fin.items():
    print(f"  {k:12s} -> {v:.2f}")
print("\nPERO la senda común NO es plana. K/Y (K0 = BGP) por década:")
print(ky_bgp.groupby(ky_bgp.index.year // 10 * 10).mean().round(2).to_string())
print(f"  {us.index[0].year} = {ky_bgp.iloc[0]:.2f}  ->  {us.index[-1].year} = "
      f"{ky_bgp.iloc[-1]:.2f}   (x{ky_bgp.iloc[-1] / ky_bgp.iloc[0]:.2f})")
print("  referencia: el acervo a coste corriente de la BEA se mueve entre 1.9 y 2.6.")
assert 1.0 < np.mean(list(fin.values())) < 3.0                 # orden de magnitud correcto
assert max(fin.values()) - min(fin.values()) < 0.02           # las 3 sendas convergen...
assert ky_bgp.iloc[-1] > 1.8 * ky_bgp.iloc[0]                 # ...pero la senda común SE DUPLICA

# %% slideshow={"slide_type": "slide"}
sty = _nbstyle.styles(len(KY))
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for (etq, ky), s, col in zip(KY.items(), sty, _nbstyle.palette(len(KY))):
    ax.plot(us.index, ky, color=col, lw=1.4, ls=s, label=etq)
ax.axhspan(1.9, 2.6, color="0.85", zorder=0, label="rango BEA (coste corriente)")
ax.set_xlabel("año"); ax.set_ylabel("K / Y (Y a tasa anual)")
ax.set_title("Las 3 condiciones iniciales convergen — a una senda que se duplica")
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Senda balanceada vs transición: lo que sí sale y lo que no
# **Sí sale**: las tres curvas parten de niveles distintos de $K_0$ pero **convergen a
# la misma senda**. La condición inicial se **lava**, a tasa $(1-\delta)^t$ por
# trimestre; la depreciación fija la *velocidad* con que se olvida el pasado. Con
# $\delta_q\approx0.0153$ esa velocidad es **lentísima**: la vida media del error inicial
# es $\ln(0.5)/\ln(1-\delta_q)\approx 45$ trimestres, y llegar al 1% pide ~300.
#
# **No sale** —y esto es lo importante— **la constancia de $K/Y$**. La senda común
# **crece de forma monótona y se duplica**, de $\approx 1.05$ en 1947 a $\approx 2.0$ hoy.
# El cuarto hecho de Kaldor **no** se verifica en esta reconstrucción. Dos razones, y
# ninguna es que el código esté mal:
#
# 1. **La condición inicial.** $K_0=I_1/(g+\delta)$ supone que EUA ya estaba en su senda
#    balanceada en 1947. En volúmenes encadenados la inversión real crece **más rápido**
#    que el PIB (es la sección 1: $I/Y$ real sube), así que ese $K_0$ queda demasiado
#    **bajo** y el inventario perpetuo tarda décadas en olvidar el error. En 1947 la
#    reconstrucción está a un factor de dos por debajo de la BEA.
# 2. **La base de precios.** $K^{nom}/Y^{nom}=(P_K/P_Y)\cdot K^{real}/Y^{real}$: un precio
#    relativo del capital con tendencia separa las dos series. Eso explica un desnivel
#    **que se abre**, no que la reconstrucción *arranque* en la mitad.
#
# **Lo que se llevan**: el hecho 4 aguanta en el acervo medido a **coste corriente** por
# la BEA (banda gris, 1.9–2.6 en ochenta años) y **no** en cualquier reconstrucción hecha
# en casa. Hall y Kehoe se replican entre sí, y aun así ambos se duplican.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. La participación del trabajo
#
# La participación del trabajo $s_L = wL/Y$ es el hecho 3 de Kaldor, y el que más se
# discute hoy: **Karabarbounis–Neiman (2014)** documentan una **caída global**. Vamos a
# intentar reproducirla con `klems_labor_share.csv` (extracto crudo de EU-KLEMS 2023,
# `comp_total`/`va`, 6 economías avanzadas, 1995–2020) — y **no va a salir**. Ese fracaso
# es el contenido de la sección: el signo del hecho 3 depende del **numerador**, del
# **denominador**, del **sector**, de la **ventana** y del **ponderador**.
#
# Dos trampas de medición que el código deja al descubierto:
#
# - **La ventana.** No es lo mismo medir desde 1995 que desde el máximo de 2000.
# - **El ponderador.** La columna `va` viene en **moneda nacional** (yenes, euros,
#   dólares). Ponderar por `va` sin convertir a una moneda común no agrega nada:
#   **elige a Japón**. Lo comprobamos abajo con los pesos implícitos.
#
# Y una ruptura de fuente: en EUA `comp_total` salta $\approx 20\%$ entre 1996 y 1997
# contra $\approx 6\%$ del valor agregado, así que la "subida" 1995–2020 de EUA **no es un
# movimiento económico**.

# %% slideshow={"slide_type": "fragment"}
ls = pd.read_csv(DATA / "klems_labor_share.csv")
W = ls.pivot(index="year", columns="code", values="labor_share")       # niveles por país
rango_intra = (W.max() - W.min())                                      # variación dentro de país
disp_entre = W.mean(axis=0).max() - W.mean(axis=0).min()               # dispersión de niveles
media_simple = W.mean(axis=1)                                          # 6 países, mismo peso

# TRAMPA: 'va' está en MONEDA NACIONAL. Ponderar por va sin convertir no agrega: elige a Japón.
w_va = ls.loc[ls.year == 1995].set_index("code")["va"]
w_va = w_va / w_va.sum()
pond_va = ls.groupby("year").apply(lambda d: np.average(d.labor_share, weights=d.va),
                                   include_groups=False)

print("cambio de la participación del trabajo, 1995 -> 2020 (puntos):")
print((100 * (W.loc[2020] - W.loc[1995])).round(1).to_string())
print("y desde el máximo de EUA en 2000 (puntos):")
print((100 * (W.loc[2020] - W.loc[2000])).round(1).to_string())
print(f"\nmedia simple: {media_simple.loc[1995]:.3f} (1995) -> {media_simple.loc[2020]:.3f} "
      f"(2020), cambio {media_simple.loc[2020] - media_simple.loc[1995]:+.3f}  -> SUBE")
print(f"rango intra-país máximo = {rango_intra.max():.3f} ({rango_intra.idxmax()}, "
      f"que es la ruptura 1996/97);  sin EUA = {rango_intra.drop('USA').max():.3f}")
print(f"dispersión de niveles entre países = {disp_entre:.3f}")
print("\npesos implícitos de 'va' en 1995 (MONEDA NACIONAL, %):")
print((100 * w_va).round(1).to_string())
print(f"  -> el 'agregado ponderado' {pond_va.iloc[0]:.3f} -> {pond_va.iloc[-1]:.3f} es "
      f"Japón con otro nombre (dista {np.abs(pond_va - W['JPN']).max():.4f} de JPN).")

assert (W.loc[2020] > W.loc[1995]).sum() == 5      # SUBE en 5 de 6 países, 1995-2020
assert media_simple.loc[2020] > media_simple.loc[1995]      # y la media simple también sube
assert (W.loc[2020] < W.loc[2000]).sum() == 2      # desde 2000 solo bajan EUA y Japón
assert np.abs(pond_va - W["JPN"]).max() < 0.01     # el "global" ponderado por va ES Japón
assert rango_intra.max() < disp_entre              # más variación entre países que dentro

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.8, 3.8))
for c, col in zip(W.columns, _nbstyle.palette(W.shape[1])):
    ax.plot(W.index, W[c].values, color=col, lw=1.1, alpha=0.9, label=c)
ax.plot(media_simple.index, media_simple.values, color="0.0", lw=2.6, label="media simple")
ax.plot(pond_va.index, pond_va.values, color="0.0", lw=1.4, ls=(0, (2, 2)),
        label="'pond. por va' (= Japón: trampa)")
ax.set_xlabel("año"); ax.set_ylabel("participación del trabajo $s_L$")
ax.set_title("Con estos seis países la caída no se ve: sube en cinco (1995–2020)")
ax.legend(loc="lower left", ncol=4, fontsize=8)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lectura: ¿se rompió Karabarbounis–Neiman? No — se midió de otra manera
# En el extracto crudo, 1995–2020, $s_L$ **sube en cinco de seis** (EUA $52\to57$,
# DEU $60\to61$, FRA $57\to58$, ESP $52\to54$, ITA $43\to45$) y **solo baja en Japón**
# ($56\to54$). La subida de EUA es la **ruptura de fuente** de 1996/97, no economía;
# medida desde su máximo de 2000, EUA **sí cae** ($61\to57$) y es el único de los seis
# donde la caída sostenida se ve. Lo que sí está en la figura son dos cosas: **niveles
# muy distintos** (casi 16 puntos entre Italia y Alemania en 2020) y **trayectorias que
# no van todas en la misma dirección**.
#
# El hecho de Karabarbounis–Neiman se mide **de otra manera**: participación del trabajo
# del sector **corporativo**, ajustada por ingreso mixto y autoempleo (Gollin 2002),
# **decenas de países** y ventana que arranca en los **setenta**. Chocar el extracto con
# el titular es el ejercicio: no invalida a KN, delimita a qué serie se refiere.
#
# Tres mecanismos rivales para la caída, cuando se mide donde sí aparece: **precio de la
# inversión** con $\sigma_{KL}>1$ (KN, abajo); **contable** —capitalización de la
# propiedad intelectual en las NIPA (Rognlie 2015; Koh, Santaeulàlia-Llopis y Zheng
# 2020)—; y **reasignación** hacia empresas superestrella con bajo $s_L$ (Autor et al.
# 2020).

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El precio de los bienes de capital como motor
# El precio **relativo de la inversión** (`PIRIC`) cae de forma secular: bienes
# de capital cada vez más baratos frente al consumo. Con una elasticidad de sustitución
# capital–trabajo $\sigma_{KL}$, la contabilidad de la participación implica
# $$\Delta\log s_L = s_K\,(\sigma_{KL}-1)\,\Delta\log\!\big(P_K/P_C\big),$$
# con $s_K$ la participación del capital. (Es la forma en **participación del trabajo**;
# `puremacro.korv_gmm.fit_sigma_kl_pooled` la escribe sobre el **cociente**
# $s_K/s_L$, donde el coeficiente es $1-\sigma_{KL}$ — de hecho **rechaza** que le pasen
# una participación del trabajo, precisamente para que nadie invierta el signo.)
# Una **pendiente positiva** implica $\sigma_{KL}>1$: capital y trabajo son
# **sustitutos**, así que cuando el capital se abarata la participación del trabajo
# **cae** — el mecanismo de Karabarbounis–Neiman.
#
# Estimamos la pendiente con una regresión **puramente ilustrativa**: agrupada, con el
# `PIRIC` de EUA como precio **común a todos los países**. Ojo con lo que eso implica:
# como el regresor es idéntico para los seis, añadir **efectos fijos de país no cambia
# nada** — hay 150 filas pero solo **25 observaciones independientes**. Y en este panel
# $s_L$ **sube** mientras $P_K/P_C$ **baja**, así que el ajuste carga la subida en la
# constante. Imprimimos el estadístico $t$ y el $R^2$ para que se vea la fragilidad.

# %% slideshow={"slide_type": "fragment"}
pk = fred("PIRIC").resample("YE").mean(); pk.index = pk.index.year      # precio rel. de la inversión, anual
dlp = np.log(pk[(pk.index >= 1995) & (pk.index <= 2020)]).diff()
M = pd.concat([pd.concat([np.log(W[c]).diff().rename("dls"),
                          dlp.rename("dlp")], axis=1).assign(code=c).dropna()
               for c in W.columns])
b1, b0 = np.polyfit(M["dlp"], M["dls"], 1)
slope = float(b1)
resid = M["dls"] - (b0 + b1 * M["dlp"])                  # error estándar MCO de la pendiente
se = float(np.sqrt((resid @ resid) / (len(M) - 2) / ((M["dlp"] - M["dlp"].mean()) ** 2).sum()))
r2 = float(np.corrcoef(M["dlp"], M["dls"])[0, 1] ** 2)
sK = 1 - float(W.stack().mean())         # participación media del capital en el panel (~0.46)
sigma_kl = 1 + slope / sK                # de dlog s_L = s_K (sigma-1) dlog(P_K/P_C)
tasa = 100 * np.log(pk.loc[2024] / pk.loc[1947]) / (2024 - 1947)

print(f"PIRIC (P_K/P_C): {pk.loc[1947]:.2f} (1947) -> {pk.loc[2024]:.2f} (2024)  "
      f"[caída de {100*(1 - pk.loc[2024]/pk.loc[1947]):.0f}%, es decir {tasa:+.1f}% anual]")
print(f"pendiente  dlog s_L ~ dlog(P_K/P_C) = {slope:+.3f}  (e.e. {se:.3f}, t = "
      f"{slope/se:.2f}, R2 = {r2:.3f}, n = {len(M)} filas / 25 años)")
print(f"  -> con s_K = {sK:.2f}, sigma_KL implícita ~ {sigma_kl:.2f} "
      f"(>1 = sustitutos, el mecanismo de Karabarbounis-Neiman)")
print("ADVERTENCIA: el signo es el ESPERADO, pero |t| < 2 y R2 ~ 0.01: con estos seis")
print("países y 25 años ni el signo está identificado. Es una ilustración del álgebra,")
print("NO una estimación. La versión seria es korv_gmm (sección 4).")
assert pk.loc[2024] < pk.loc[1947]                       # la inversión se abarata secularmente
assert slope > 0                                         # el signo va en la dirección de KN...
assert abs(slope / se) < 2.0                             # ...pero NO es significativo

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.plot(pk.index, pk.values, color="0.15", lw=1.6)
ax.axhline(1.0, color="0.8", lw=0.7)
ax.set_yscale("log")
ax.set_xlabel("año"); ax.set_ylabel("$P_K/P_C$  (escala log)")
ax.set_title("El precio relativo de la inversión cae ~82% (1947–2024, ~2.2% anual) — PIRIC")
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### 2025–26: la IA encarece el capital — ¿se rompe el hecho estilizado?
# La serie de arriba cuenta la historia secular: la inversión se abarata ~2.2% anual durante
# casi ocho décadas (1947–2024; el equipo, PERIC, cae más rápido, ~4.8% anual, un factor de
# ~41 — véase A1), y la IA es el episodio vigente de esa historia. Pero en el corto plazo la misma
# IA lo *encarece*: los centros de datos reasignaron la capacidad mundial de fabricación
# de memoria hacia HBM y los precios de contrato de la DRAM subieron ≈90–110% *en un solo
# trimestre* (2026T1), con escasez proyectada hasta 2027. Tres empresas fabrican ≈90% de
# la DRAM del mundo.
#
# **Externalidad pecuniaria (Scitovsky 1954).** La demanda de las empresas de IA sube el
# precio de la memoria para **todos** los demás compradores. Opera *a través de precios*:
# con competencia y mercados completos no es una ineficiencia —es el sistema de precios
# racionando capacidad escasa—. Tiene consecuencias de bienestar solo con fricciones
# *nombrables*: oligopolio con capacidad rígida (3 firmas ≈90%), oferta de corto plazo
# casi vertical (una fábrica tarda años), contratos que racionan sin precio, y mercados
# incompletos —en la lección de heterogeneidad y HANK (semana 11) veremos que ahí las
# externalidades pecuniarias rompen la eficiencia restringida (Dávila et al. 2012)—.
#
# Tres preguntas para discusión:
# 1. **Medición (esta lección):** ¿aparecerá en PERIC/PIRIC? El deflactor hedónico
#    ajusta por calidad y llega con rezago; el precio de lista no es el deflactor.
# 2. **Identificación (lecciones de RBC y SVAR):** el precio relativo del equipo revela
#    choques *tecnológicos* a la inversión; aquí sube sin retroceso tecnológico: es
#    demanda. ¿Qué supuesto se rompe?
# 3. **Distribución:** cómputo más caro para las demás empresas; México importa su
#    memoria: choque de términos de intercambio a su deflactor de inversión.
#
# *Cifras: precios de contrato reportados (TrendForce vía prensa especializada,
# feb–jun 2026).*

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Laboratorio
# Para llevar la contabilidad de la participación a estimación estructural:
# - **`puremacro.korv_gmm`** — CES anidada (KORV): momentos $m_3,\,m_4$ estiman
#   $\sigma_{KL}$ (`fit_sigma_kl_pooled`) y el costo de uso Hall–Jorgenson del equipo
#   (`build_usercost_column`); es la versión rigurosa de la regresión de arriba. El
#   momento $m_4$ va sobre $\Delta\log(s_K/s_L)$, **no** sobre $\Delta\log s_L$: pásele
#   la columna que construye `log_share_ratio` o la función lanza `ValueError`.
# - **`puremacro.labor_share.gollin_adjusted_ls`** — ajuste de Gollin (2002) por
#   ingreso mixto/cuenta propia, clave para *medir* bien $s_L$. **No corre sobre este
#   extracto**: exige columnas del SNA que EU-KLEMS no produce (se arman con
#   `fetch.sdmx_get`, es decir, con red).
# - **`puremacro.klems.load_klems_panel`** — el panel EU-KLEMS completo (más países e
#   industrias) del que sale `klems_labor_share.csv`. **Tampoco viaja en el bundle**:
#   lee el árbol crudo de EU-KLEMS en `data/raw/euklems` (tres CSV que hay que descargar
#   y montar a mano), así que es extensión fuera de clase, no ejercicio de sesión.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Ejercicios
# 1. **BGP vs transición.** Arranca el inventario perpetuo con $K_0=0$. ¿Cuántos
#    trimestres tarda $K/Y$ en acercarse a $1\%$ de la senda común? Relaciónalo con
#    $1-\delta$.
# 2. Sube $\delta$ anual a $0.10$ y bájalo a $0.03$. ¿Cómo cambian el **nivel** de $K/Y$
#    y la **velocidad** de convergencia?
# 3. **¿Cuánto $K_0$ haría falta?** Busca el multiplicador $m$ de $K_0=m\cdot I_1/(g+\delta)$
#    que pone el $K/Y$ de 1947 dentro del rango de la BEA (1.9–2.6). ¿Cuántas décadas
#    tarda la senda en volver a converger con la de $m=1$? ¿Salva eso la constancia
#    de $K/Y$, o solo mueve el problema al otro extremo de la muestra?
# 4. **La trampa del ponderador.** Recalcula el "agregado global" ponderando por `va` en
#    moneda nacional y compáralo con la serie de Japón: ¿por qué coinciden? Repítelo
#    convirtiendo cada `va` con un tipo de cambio (o una PPA) del año base que elijas.
#    ¿Cambia la *dirección* del agregado? Escribe la ficha de medición de tu agregado.
# 5. **Efectos fijos.** Reestima la pendiente $\Delta\log s_L$ sobre $\Delta\log(P_K/P_C)$
#    con **efectos fijos de país**. Predice el resultado **antes** de correrlo, sabiendo
#    que $P_K/P_C$ es el mismo para los seis países. Después prueba a excluir a EUA
#    (la ruptura 1996/97): ¿cuánto se mueve la pendiente?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Con $K_0=0$ el sesgo inicial decae como $(1-\delta)^t$; con $\delta_q\approx0.015$,
#    llegar a $1\%$ toma $\ln(0.01)/\ln(1-\delta_q)\approx 300$ trimestres — la transición
#    es lenta. La condición inicial se lava, pero despacio.
# 2. Mayor $\delta$ baja el nivel de senda $K/Y=(I/Y)/(g+\delta)$ y **acelera** la
#    convergencia; menor $\delta$ lo sube y la **frena**.
# 3. Hace falta $m\approx 2$ para arrancar dentro del rango de la BEA ($K/Y$ de 1947
#    pasa de $1.05$ a $2.10$). Pero las dos sendas vuelven a juntarse en unas **tres
#    décadas** (la brecha cae de $0.38$ en 1957 a $0.05$ en 1977: el error decae como
#    $(1-\delta_q)^t$), así que desde los ochenta el $K/Y$ reconstruido es el mismo y
#    sigue creciendo. Peor aún: con $m=2$ la razón primero **baja** a $\approx1.25$ en
#    los setenta y luego sube — tampoco es constante, solo se equivoca en el otro
#    sentido. **No salva el hecho 4**: el problema no es solo $K_0$, es que la inversión
#    real crece más rápido que el PIB real. La reconstrucción y la BEA miden cosas
#    distintas (volúmenes encadenados vs coste corriente).
# 4. Coinciden porque `va` está en **moneda nacional**: el yen le da a Japón el ~98% del
#    peso, así que el "agregado global" **es** Japón —el único de los seis que cae—, y
#    la caída es un artefacto de unidades, no un hecho. Con una conversión a moneda
#    común el peso pasa a EUA y el agregado **sube** (EUA sube en el extracto crudo).
#    La ficha debe declarar: fuente/serie, muestra, sector, tratamiento del ingreso
#    mixto, ponderador y moneda de conversión.
# 5. La pendiente es **idéntica** ($+0.213$): como $P_K/P_C$ es común a los seis países,
#    demediar por país no altera la variación del regresor; los efectos fijos solo
#    absorben las constantes. Es el diagnóstico de que hay 25 observaciones, no 150.
#    Excluyendo a EUA la pendiente sube a $\approx+0.32$ — un tercio de movimiento por
#    quitar un país confirma que la magnitud no es informativa.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 6. Explora con IA
# Prueba con el tutor sin conexión (o cualquier asistente de IA):
# - "En una senda de crecimiento balanceada, ¿por qué el cociente $K/Y$ es constante
#   aunque $K$ crezca? Da la intuición en una frase."
# - "Si el precio relativo del equipo cae, ¿cuándo baja la participación del trabajo —
#   con $\sigma_{KL}$ mayor o menor que 1?"
# - "El $K/Y$ que reconstruí por inventario perpetuo se duplica entre 1947 y hoy,
#   mientras el acervo de la BEA a coste corriente es plano. Dame dos explicaciones
#   posibles y dime cómo distinguirlas empíricamente."
# - "¿Por qué el alza del precio de la RAM causada por la demanda de los centros de datos
#   de IA es una externalidad *pecuniaria* y no tecnológica? ¿Bajo qué fricciones tiene
#   consecuencias de bienestar?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase: ¿por qué en una senda de crecimiento balanceada el cociente "
            "capital-producto K/Y es constante aunque el capital crezca?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen — tres hechos de Kaldor y tres sustos de medición.** (1) La $I/Y$ **real** de
# EUA **sí tiene tendencia**: casi se duplica entre décadas; el hecho 2 es un hecho de
# razones **nominales**, y las dos versiones solo se reconcilian por el desplome del
# precio relativo de la inversión. (2) El **capital por inventario perpetuo** converge a
# una senda común sea cual sea $K_0$ (transición vs BGP), pero esa senda **se duplica**
# de $\approx 1.05$ a $\approx 2.0$: el hecho 4 aguanta en el acervo de la **BEA** a coste
# corriente (1.9–2.6), no en una reconstrucción casera. (3) En el extracto crudo de
# EU-KLEMS la participación del trabajo **sube en cinco de seis** países 1995–2020, y el
# "agregado global ponderado por `va`" resultó ser **Japón disfrazado** (monedas sin
# convertir): el hecho de Karabarbounis–Neiman existe, pero se mide en otro sector, otra
# muestra y otra ventana. La regresión ilustrativa da el signo esperado ($\sigma_{KL}>1$)
# con $|t|<2$: sirve para leer el álgebra de KN, no para estimarla.
# **La regla del curso**: ningún momento se publica sin su ficha de medición.
# **Siguiente módulo:** la lección **02 — modelo neoclásico: Galí BQ e iteración de la
# función de valor** (`02_neoclasico_scb_es`), que abre el bloque de las semanas 3–4 con el
# mazo Slides02.
