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
# # Módulo 10b — Economía pequeña y abierta: México como objeto de estudio
#
# **Curso complementario · puremacro · mazo Slides07 — mecanismos y economía abierta (semanas 13–14)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. **Medir** los cuatro hechos que separan a una economía emergente de una avanzada
#    —producto más volátil, $\sigma_c/\sigma_y$ pegado al $1$ (no necesariamente por encima),
#    balanza comercial contracíclica y tasa de interés contracíclica— con el filtro de
#    **Hamilton** sobre el panel de la OCDE, declarando la **ventana** y el orden de las
#    operaciones antes de citar ninguna cifra, y demostrar con la identidad del gasto por qué
#    el modelo **cerrado** no puede generar $\sigma_c>\sigma_y$.
# 2. Diagnosticar el **problema del cierre** de una economía pequeña y abierta —con
#    $r$ exógeno y $\beta(1+r)=1$ la deuda tiene raíz unitaria— y resolverlo con la **prima
#    creciente en la deuda** de Schmitt-Grohé y Uribe (2003), verificando numéricamente que
#    Blanchard–Kahn falla con $\psi=0$ y se cumple con $\psi>0$.
# 3. **Resolver** el modelo log-linealizado con `puremacro.dsge.klein_solve`, leer las
#    funciones de impulso-respuesta a un choque de productividad y a uno de **prima de
#    riesgo**, y arbitrar el debate Aguiar–Gopinath (2007) contra García-Cicco, Pancrazi y
#    Uribe (2010) con las dos curvas que el propio modelo produce.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), sin red, leyendo los CSV congelados del *bundle*: $0.

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
# --------------------------------------------------------------------------------------
# DATOS CONGELADOS (todos locales; esta lección NUNCA usa la red).
#
#  oecd_qna_apertura.csv  OCDE, Cuentas Nacionales Trimestrales, dataflow
#                         DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR (P3 hogares, B1GQ,
#                         P6 exportaciones, P7 importaciones), nominal ("_nom") y volumen
#                         ("_vol": base ENCADENADA `PRICE_BASE=L` en todos los países
#                         salvo MÉXICO, que sólo se publica en base FIJA `Q`, precios
#                         constantes de 2018 — verificable en la columna price_base del
#                         propio CSV), 8 países, 1994Q1–2026Q2 (el último
#                         trimestre varía por país: MEX/USA/DEU llegan a 2026Q2, el
#                         resto a 2026Q1; CHL arranca en 1996Q1, TUR en 1995Q1 y COL
#                         en 2005Q1).
#                         Descargado 2026-08-01 (paquete del curso bundle_2026A).
#  IR3TIB01MXM156N.csv    FRED: tasa interbancaria a 3 meses de México, mensual, % anual.
#                         La serie congelada ARRANCA EN 1997-01 (por eso la sección 4 mide
#                         sobre 1997Q1–2019Q4). Descargado 2026-08-01.
#  CPALTT01MXM659N.csv    FRED: IPC de México, variación % respecto al mismo mes del año
#                         previo, mensual. Serie descontinuada en 2024-07. Desc. 2026-07-22.
#  FEDFUNDS.csv           FRED: tasa de fondos federales de EE.UU., mensual, % anual.
#  VIXCLS.csv             FRED/CBOE: índice VIX, diario. Precio del riesgo global.
#
# NOTA DE HONESTIDAD: el EMBI+ México (J.P. Morgan) es propietario y NO está en el paquete
# congelado. La sección 4 lo dice y usa aproximaciones documentadas en su lugar.
# --------------------------------------------------------------------------------------


def fred(name: str) -> pd.Series:
    """Lee un CSV estilo FRED del paquete congelado y lo devuelve como serie trimestral."""
    df = pd.read_csv(DATA / f"{name}.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = df.set_index("observation_date")[name].astype(float)
    return s.resample("QS").mean()


def ciclo(x, filtro="hamilton"):
    """Componente cíclico de una serie (Hamilton 2018 por omisión; HP como robustez)."""
    from puremacro.cycles import hamilton_filter
    from puremacro.data import hp_filter
    x = np.asarray(x, dtype=float)
    if filtro == "hamilton":
        c, _ = hamilton_filter(x)          # las primeras h+p-1 = 11 obs quedan en NaN
        return np.asarray(c)
    c, _ = hp_filter(x)                    # lambda = 1600
    return np.asarray(c)


# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Puente de notación con el mazo Slides07 (léelo antes de la sección 5)
#
# El código de esta lección reutiliza tres letras griegas que la lámina «Notación: un
# símbolo, un significado» del mazo Slides07 asigna a **otras** cosas dentro de estas mismas
# semanas. No las renombramos en el código —la calibración es la estándar de la literatura
# de economía abierta— pero la traducción es obligatoria para no cruzar cables:
#
# | símbolo en esta lección | qué es aquí | qué es en Slides07 | cómo se llama allá lo de aquí |
# |---|---|---|---|
# | $\psi$ | pendiente de la prima de deuda, $r_t=r^{*}+\psi(d_{t-1}-\bar d)$ | elasticidad *adicional* del producto en las horas (bloque de márgenes) | $p'(\bar d)$ — en el mazo la pendiente no tiene letra propia |
# | $\omega$ | curvatura de la desutilidad GHH del trabajo (Frisch $=1/(\omega-1)$) | curvatura de $\delta(u)$ isoelástica (bloque de utilización) | la curvatura del ocio, $\nu$ |
# | $\gamma$ | curvatura CRRA del consumo, $u'(c)=c^{-\gamma}$ | — | $\sigma$ (sin subíndice) |
#
# $\alpha$ y $\phi$ **sí** coinciden con el mazo ($\alpha=0.32$ es la participación del
# capital; $\phi$ es únicamente el costo de ajuste). Ojo: la lección 11, del mazo Slides08,
# usa $\alpha$ para la elasticidad del emparejamiento — ahí hay otro puente de notación.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. Los cuatro hechos de una economía emergente
#
# El ciclo mexicano no es el estadounidense con más ruido. Cuatro regularidades lo separan,
# y las cuatro son problemas para el modelo neoclásico cerrado que hemos usado hasta aquí:
#
# 1. **El producto es más volátil.** $\sigma_y$ del ciclo emergente casi duplica al avanzado.
# 2. **El consumo casi no se suaviza:** $\sigma_c/\sigma_y$ se pega al $1$, cuando la
#    hipótesis de renta permanente predice un cociente claramente $<1$ y los datos de las
#    economías avanzadas la confirman ($0.85$ en EE. UU., $0.54$ en Alemania). Ojo con cómo
#    se enuncia este hecho: **no** es que el consumo mexicano sea *más* volátil que el
#    producto —con la ventana canónica del curso no lo es—, sino que está mucho **más cerca**
#    del producto de lo que cualquier modelo de suavizamiento admite. Es un hecho de
#    **magnitud**, y por eso hay que decir con qué serie, qué ventana y qué filtro se midió.
# 3. **La balanza comercial es contracíclica:** $\mathrm{corr}(nx_t/y_t,\,y_t)<0$. El país
#    absorbe más de lo que produce en los auges y devuelve en las recesiones.
# 4. **La tasa de interés real es contracíclica** (sección 4).
#
# Medimos los tres primeros con el panel trimestral de la OCDE, filtrando con **Hamilton
# (2018)** —la convención del curso— sobre $100\log$ de los volúmenes. Cuidado con la base
# de precios: los volúmenes son **encadenados** (`L`) en todos los países del panel **salvo
# México**, que la OCDE publica sólo en **base fija** (`Q`, precios constantes, año base 2018).
# Aquí vemos ocho países; en el panel de 21 del mazo Slides07, México es el **único** medido
# así. La balanza comercial va en niveles: $nx_t/y_t$ a precios corrientes, en % del PIB.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Ficha de medición (léela antes de citar cualquier número de esta lección)
#
# Un cociente $\sigma_c/\sigma_y$ sin ficha de medición no es un dato, es una anécdota.
# El mazo Slides07 exige **seis campos**: (1) fuente y serie, (2) muestra, (3) filtro y su
# parámetro, (4) base de precios, (5) orden de las operaciones y (6) **edición** (*vintage*)
# del dato. Los seis, con la convención canónica del curso que esta lección usa en **todas**
# sus cifras principales:
#
# | campo | convención del curso | por qué importa |
# |---|---|---|
# | (1) serie de consumo | **hogares** (P3, S1M), volúmenes | añadir gobierno *baja* el cociente (a $0.85$ en el mazo) |
# | (1) serie de producto | PIB (B1GQ), volúmenes | |
# | (2) **ventana** | **1995Q1–2019Q4** | 1994 mete el pico previo a la crisis del Tequila |
# | (3) filtro principal | Hamilton (2018), $h=8$, $p=4$ | pierde las primeras $h+p-1=11$ obs. |
# | (3) filtro de robustez | HP, $\lambda=1600$ | no pierde observaciones |
# | (4) **base de precios** | **base FIJA** (`PRICE_BASE=Q`, precios constantes, año base 2018) para México; los demás del panel, **encadenados** (`L`) | el mazo Slides07 reconstruye el encadenado mexicano y el cociente se mueve $0.001$: campo declarado *y* contestado |
# | (5) orden de las operaciones | **recortar a la ventana y luego filtrar** | filtrar antes es *look-ahead* |
# | (6) **edición (*vintage*)** | OCDE QNA descargada el **2026-08-01** (paquete `bundle_2026A`) | es el campo que más se olvida: el mazo documenta un salto de $0.092$ en el cociente **entre ediciones consecutivas** (oct-2023 → ene-2024) con ventana, filtro y país fijos |
#
# (Transformación: $100\log$ de los volúmenes en los dos casos.)
#
# La línea que más se olvida es la penúltima. Si filtras la serie **completa** (1994Q1–2026Q2
# en este panel) y sólo después recortas el ciclo a 1995–2019, la tendencia de 2003 se estimó
# con datos de 2025 que en 2003 no existían: es **contaminación de futuro** (*look-ahead*), la
# misma enfermedad que motiva el trabajo con *vintages* y datos en tiempo real del mazo A1.
# No es "calentamiento del filtro": el filtro HP no tiene calentamiento, devuelve ciclo para
# **todas** las observaciones. Hamilton sí pierde $h+p-1$ observaciones iniciales, pero eso es
# otra cosa. Aquí `ventana()` recorta **primero** y `ciclo()` filtra **después**, siempre.

# %% slideshow={"slide_type": "fragment"}
qna = pd.read_csv(DATA / "oecd_qna_apertura.csv", parse_dates=["date"])
qna = qna.pivot_table(index=["code", "date"], columns="variable", values="value")

EMERGENTES = ["MEX", "CHL", "COL", "TUR"]
AVANZADAS = ["USA", "CAN", "AUS", "DEU"]
INI, FIN = "1995-01-01", "2019-10-01"      # ventana CANÓNICA del curso, corte pre-COVID
INI_ROB = "1994-01-01"                     # robustez: el arranque mismo de la QNA (§1 bis)


COLS = ["gdp_vol", "conh_vol", "gdp_nom", "conh_nom", "exp_nom", "imp_nom",
        "exp_vol", "imp_vol"]


def ventana(code, ini=INI, fin=FIN):
    """Sub-panel de un país en la ventana, sin huecos (TUR arranca el consumo en 1996)."""
    d = qna.loc[code].sort_index().dropna(subset=COLS)
    return d[(d.index >= ini) & (d.index <= fin)]


def momentos(code, ini=INI, fin=FIN):
    d = ventana(code, ini, fin)
    ly = 100 * np.log(d["gdp_vol"].to_numpy())
    lc = 100 * np.log(d["conh_vol"].to_numpy())
    nx = (100 * (d["exp_nom"] - d["imp_nom"]) / d["gdp_nom"]).to_numpy()
    cy, cc, cn = ciclo(ly), ciclo(lc), ciclo(nx)
    ok = ~np.isnan(cy) & ~np.isnan(cc) & ~np.isnan(cn)
    cy, cc, cn = cy[ok], cc[ok], cn[ok]
    # robustez: la misma corr(nx/y, y) con HP(1600), que descarta más baja frecuencia
    hy, hn = ciclo(ly, "hp"), ciclo(nx, "hp")
    return dict(n=int(ok.sum()), sd_y=cy.std(), sc_sy=cc.std() / cy.std(),
                corr_cy=np.corrcoef(cc, cy)[0, 1], corr_nxy=np.corrcoef(cn, cy)[0, 1],
                corr_nxy_hp=np.corrcoef(hn, hy)[0, 1],
                rho_y=np.corrcoef(cy[1:], cy[:-1])[0, 1])


tabla = pd.DataFrame({c: momentos(c) for c in EMERGENTES + AVANZADAS}).T
tabla.insert(0, "grupo", ["emergente"] * 4 + ["avanzada"] * 4)
tabla["n"] = tabla["n"].astype(int)
print("Momentos del ciclo, filtro de Hamilton, OCDE QNA 1995Q1–2019Q4 (ventana canónica)")
print("(COL arranca en 2005Q1; CHL y TUR en 1996Q1: sus muestras son más cortas)\n")
print(tabla.round(2).to_string())

# Hecho 1: México es bastante más volátil que EE. UU.
assert tabla.loc["MEX", "sd_y"] > tabla.loc["USA", "sd_y"]
# Hecho 2, enunciado como toca: el cociente mexicano NO cruza el 1 con la ventana canónica,
# pero está muy por encima del estadounidense y a un pelo del 1.
assert 0.90 < tabla.loc["MEX", "sc_sy"] < 1.00
assert tabla.loc["MEX", "sc_sy"] > tabla.loc["USA", "sc_sy"]
assert abs(tabla.loc["MEX", "sc_sy"] - 0.968) < 0.01     # cifra canónica del curso
assert abs(tabla.loc["USA", "sc_sy"] - 0.852) < 0.01

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Control de calidad: ¿de qué depende realmente el cociente?
# La transparencia *"El ciclo mexicano no es el de Estados Unidos con más informalidad"*
# reporta, con HP sobre 1995Q1–2019Q4, $\sigma_y=1.91$ para México contra $1.03$ para
# EE. UU., y $\sigma_c/\sigma_y=0.96$ contra $0.79$. Reproducimos esas cifras y, de paso,
# hacemos el experimento que separa las dos decisiones que suelen confundirse: **cambiar de
# filtro** con la ventana fija, y **cambiar de ventana** con el filtro fijo. Si no sale,
# el resto de la lección no vale nada.
#
# **Aviso sobre el sexto campo.** El cociente sale idéntico ($0.959$), pero la $\sigma_y$
# mexicana sale $1.90$ aquí y $1.911$ en el mazo: el mazo la calculó con la edición de
# enero-2024 de la QNA y el paquete congelado trae la de agosto-2026. Es la **edición**, no
# la ventana ni el filtro, y es exactamente el sexto campo de la ficha. En lo que sigue
# citamos siempre la cifra que **imprime esta celda**.

# %% slideshow={"slide_type": "fragment"}
def par(code, ini, filtro):
    """(sigma_y, sigma_c/sigma_y, n) recortando PRIMERO a la ventana y filtrando DESPUÉS."""
    d = ventana(code, ini, FIN)
    cy = ciclo(100 * np.log(d["gdp_vol"].to_numpy()), filtro)
    cc = ciclo(100 * np.log(d["conh_vol"].to_numpy()), filtro)
    ok = ~np.isnan(cy) & ~np.isnan(cc)
    cy, cc = cy[ok], cc[ok]
    return cy.std(), cc.std() / cy.std(), int(ok.sum())


chequeo = {}
print("MÉXICO — el cociente por ventana y por filtro (recorte ANTES de filtrar)\n")
print(f"{'ventana':<26}{'filtro':<11}{'n':>4}{'sigma_y':>10}{'sigma_c/sigma_y':>18}")
for ini, etiq in [(INI, "1995Q1-2019Q4 (canonica)"), (INI_ROB, "1994Q1-2019Q4 (robustez)")]:
    for filtro in ["hp", "hamilton"]:
        sy, ratio, n = par("MEX", ini, filtro)
        chequeo[(ini, filtro)] = (sy, ratio)
        print(f"{etiq:<26}{filtro:<11}{n:>4}{sy:>10.3f}{ratio:>18.3f}")

ef_filtro = chequeo[(INI, "hamilton")][1] - chequeo[(INI, "hp")][1]
ef_ventana = chequeo[(INI_ROB, "hamilton")][1] - chequeo[(INI, "hamilton")][1]
print(f"\nefecto FILTRO  (HP -> Hamilton, ventana canónica fija) = {ef_filtro:+.3f}")
print(f"efecto VENTANA (1995Q1 -> 1994Q1, Hamilton fijo)       = {ef_ventana:+.3f}")
print(f"la ventana pesa {ef_ventana / ef_filtro:.1f} veces más que el filtro")

for code in ["MEX", "USA"]:
    sy, ratio, _ = par(code, INI, "hp")
    print(f"{code}  HP 1995Q1–2019Q4:  sigma_y = {sy:.2f}   sigma_c/sigma_y = {ratio:.2f}")


def corr_nx(code, ini, filtro):
    """corr(nx/y, y) recortando PRIMERO a la ventana y filtrando DESPUÉS."""
    d = ventana(code, ini, FIN)
    cy = ciclo(100 * np.log(d["gdp_vol"].to_numpy()), filtro)
    cn = ciclo((100 * (d["exp_nom"] - d["imp_nom"]) / d["gdp_nom"]).to_numpy(), filtro)
    ok = ~np.isnan(cy) & ~np.isnan(cn)
    return np.corrcoef(cn[ok], cy[ok])[0, 1]


print("\nMÉXICO — el hecho 3, corr(nx/y, y), por ventana y por filtro")
for ini, etiq in [(INI, "1995Q1-2019Q4 (canonica)"), (INI_ROB, "1994Q1-2019Q4 (robustez)")]:
    print(f"  {etiq:<26}HP = {corr_nx('MEX', ini, 'hp'):+.2f}   "
          f"Hamilton = {corr_nx('MEX', ini, 'hamilton'):+.2f}")

# El hecho 3 SÍ depende del filtro y de la ventana; las cuatro cifras se citan en la lectura.
assert corr_nx("MEX", INI, "hp") < -0.2 and abs(corr_nx("MEX", INI, "hamilton")) < 0.1
assert corr_nx("MEX", INI_ROB, "hp") < -0.5

# Cifras canónicas del mazo Slides07: MEX 1.91 / 0.96 ; USA 1.03 / 0.79 (HP, 1995Q1–2019Q4).
assert abs(par("MEX", INI, "hp")[0] - 1.91) < 0.05 and abs(par("MEX", INI, "hp")[1] - 0.959) < 0.01
assert abs(par("USA", INI, "hp")[0] - 1.03) < 0.05 and abs(par("USA", INI, "hp")[1] - 0.795) < 0.01
# Y lo que este bloque existe para demostrar: la ventana manda, el filtro casi no.
assert abs(ef_filtro) < 0.02 and ef_ventana > 3 * abs(ef_filtro)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** Con Hamilton y la ventana canónica, México tiene $\sigma_y=3.51$ contra $2.36$
# de EE. UU. —un ciclo **50% más ancho**— y $\sigma_c/\sigma_y=0.97$ contra $0.85$. El hecho 1
# aparece limpio. El hecho 2 aparece **en magnitud**: el cociente mexicano está a tres
# centésimas del 1 ($0.968$) y doce por encima del estadounidense ($0.852$), pero **no lo
# cruza**.
#
# **La atribución correcta del salto.** Es tentador decir que el $0.96$ del mazo se convierte
# en $1.00$ "porque cambiamos de filtro". Es falso, y la tabla de arriba lo demuestra. Con la
# ventana canónica fija, pasar de HP a Hamilton mueve el cociente **menos de una centésima**
# ($0.959\to0.968$): los dos filtros dicen prácticamente lo mismo. Lo que mueve el número es
# la **ventana**: añadir 1994 —cuatro trimestres, el pico de absorción previo al colapso del
# Tequila— sube el cociente casi **cuatro centésimas** ($0.968\to1.004$), y de paso infla
# $\sigma_y$ de $3.51$ a $3.57$ con Hamilton y de $1.90$ a $2.32$ con HP. La ventana pesa más
# de **cuatro veces** lo que pesa el filtro.
# Guarda la lección: cuando dos personas reportan cociente distinto, pregunta primero por la
# muestra y sólo después por el filtro. El $1.004$ de la ventana 1994Q1 es una **caja de
# robustez etiquetada**, no la cifra principal — y ni siquiera es un caso de $>1$ robusto:
# depende de que entren cuatro trimestres concretos.
#
# El hecho 3 sí **depende del filtro**, y conviene decirlo en voz alta. Con HP(1600) la
# balanza comercial mexicana es contracíclica ($-0.29$); con Hamilton queda esencialmente
# ortogonal al ciclo ($+0.01$). No es contradicción: el ciclo de Hamilton retiene mucha más
# variación de baja frecuencia, y $nx/y$ en México está dominado por movimientos lentos
# —apertura comercial, petróleo, maquila— que HP manda a la tendencia y Hamilton deja dentro
# del ciclo. Los dos números son correctos; describen objetos distintos, y la literatura del
# área (Aguiar–Gopinath 2007) reporta el de HP. (Con la ventana 1994Q1 la correlación de HP
# es $-0.62$: casi toda esa contraciclicidad la aporta la reversión de 1994–95 de la sección
# 7. Otro hecho que es tan sobre la muestra como sobre el país.) Cuando un hecho estilizado
# vive o muere según el filtro o según cuatro trimestres, el hecho estilizado es sobre la
# medición tanto como sobre el país.
#
# Nota adicional, y es la pregunta que hace el mazo: **Australia**, país avanzado, supera en
# $\sigma_c/\sigma_y$ ($1.53$) a **los cuatro** emergentes —incluidos los dos que sí cruzan el
# 1, Chile ($1.42$) y Turquía ($1.03$)—. Lo logra por el denominador —su
# $\sigma_y=0.92$ es el más bajo del panel, un cuarto de siglo sin recesión hasta 2020— y
# comparte con México y Chile ser exportador de materias primas con términos de intercambio
# muy volátiles (sección 7). El cociente no es un pasaporte: es una razón entre dos
# volatilidades, y conviene mirar siempre las dos por separado.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

ax = axes[0]
ax.set_axisbelow(True)
orden = tabla.sort_values("sc_sy").index.tolist()
vals = tabla.loc[orden, "sc_sy"].to_numpy()
grises = ["0.15" if tabla.loc[c, "grupo"] == "emergente" else "0.68" for c in orden]
ax.bar(range(len(orden)), vals, color=grises, width=0.62)
ax.axhline(1.0, color="0.0", lw=1.0, ls=(0, (4, 2)))
ax.set_xticks(range(len(orden))); ax.set_xticklabels(orden, fontsize=9)
ax.set_ylabel(r"$\sigma_c\,/\,\sigma_y$")
ax.set_title("A. El consumo no se suaviza (oscuro = emergente)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

ax = axes[1]
for c in tabla.index:
    em = tabla.loc[c, "grupo"] == "emergente"
    ax.scatter(tabla.loc[c, "sd_y"], tabla.loc[c, "corr_nxy_hp"],
               s=70, marker="o" if em else "^",
               facecolor="0.15" if em else "white", edgecolor="0.15", zorder=3)
    ax.annotate(c, (tabla.loc[c, "sd_y"], tabla.loc[c, "corr_nxy_hp"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.axhline(0.0, color="0.5", lw=0.8)
ax.set_xlabel(r"$\sigma_y$ del ciclo de Hamilton (%)")
ax.set_ylabel(r"corr$(nx/y,\;y)$, filtro HP")
ax.set_title("B. Más volátil y con balanza contracíclica")
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. El margen que el modelo cerrado no tiene
#
# Una economía pequeña y abierta toma la tasa de interés mundial como dada y **separa lo que
# produce de lo que absorbe**. La restricción de recursos deja de igualar producto y gasto
# interno, y aparece un acervo nuevo: la deuda externa neta $d_t$, que acumula el exceso de
# absorción sobre producto,
#
# $$c_t+i_t+nx_t=y_t,\qquad d_{t+1}=(1+r_t)\,d_t-nx_t.$$
#
# Endeudarse no es gratis: la condición de **no-Ponzi** prohíbe pagar intereses emitiendo
# deuda para siempre,
#
# $$\lim_{T\to\infty}E_t\!\left[\frac{d_{t+T+1}}{\prod_{j=0}^{T}(1+r_{t+j})}\right]\le 0.$$
#
# Iterando la ley de movimiento hacia adelante con $r$ constante y usando que en el óptimo la
# condición se cumple con igualdad, queda la **restricción presupuestaria intertemporal del
# país**:
#
# $$(1+r)\,d_t=\sum_{j=0}^{\infty}\frac{E_t[nx_{t+j}]}{(1+r)^{j}}.$$
#
# La deuda de hoy es el valor presente de los superávits comerciales futuros: la absorción se
# desacopla del producto **periodo a periodo**, pero no en valor presente.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Por qué el modelo cerrado no puede dar $\sigma_c>\sigma_y$
#
# Escribamos la identidad del gasto en desviaciones, con participaciones $s_c=\bar C/\bar Y$
# y $s_a=\bar A/\bar Y$ (donde $A$ es la absorción no-consumo: inversión más gobierno):
#
# $$\hat y_t=s_c\hat c_t+s_a\hat a_t+\widehat{nx}_t.$$
#
# En la economía **cerrada** $\widehat{nx}_t\equiv0$ y $s_c+s_a=1$, de modo que
# $\sigma_y^2=s_c^2\sigma_c^2+s_a^2\sigma_a^2+2s_cs_a\rho_{ca}\sigma_c\sigma_a$. Pedir
# $\sigma_c>\sigma_y$ equivale, dividiendo entre $s_a>0$ y escribiendo $m=\sigma_a/\sigma_c$, a
#
# $$(1+s_c)>s_a\,m^2+2s_c\,\rho_{ca}\,m.$$
#
# Con $s_c\approx0.75$ y $\rho_{ca}=1$ —el caso del modelo de ciclos reales, donde el mismo
# choque mueve consumo e inversión— la condición se reduce a $m^2+6m-7<0$, es decir $m<1$:
# la inversión tendría que ser **menos** volátil que el consumo. Nunca lo es.
#
# La prueba numérica: tomamos las series mexicanas, calculamos el lado derecho, y además
# construimos el **contrafactual cerrado** $\hat y^{\text{cerr}}_t=s_c\hat c_t+s_a\hat a_t$
# —lo que sería el PIB si las exportaciones netas no existieran— para ver cuánto del
# cociente observado compra la apertura por sí sola.

# %% slideshow={"slide_type": "fragment"}
def aritmetica_cerrada(code, ini=INI, fin=FIN):
    d = ventana(code, ini, fin)
    # absorción no-consumo (inversión + gobierno), por residuo de la identidad del gasto
    a_nom = d["gdp_nom"] - d["conh_nom"] - d["exp_nom"] + d["imp_nom"]
    a_vol = d["gdp_vol"] - d["conh_vol"] - d["exp_vol"] + d["imp_vol"]
    s_c = (d["conh_nom"] / d["gdp_nom"]).mean()
    s_a = (a_nom / d["gdp_nom"]).mean()
    s_c, s_a = s_c / (s_c + s_a), s_a / (s_c + s_a)      # renormalizar a economía cerrada
    cy = ciclo(100 * np.log(d["gdp_vol"].to_numpy()))
    cc = ciclo(100 * np.log(d["conh_vol"].to_numpy()))
    ca = ciclo(100 * np.log(a_vol.to_numpy()))
    ok = ~np.isnan(cy) & ~np.isnan(cc) & ~np.isnan(ca)
    cy, cc, ca = cy[ok], cc[ok], ca[ok]
    m = ca.std() / cc.std()
    rho = np.corrcoef(cc, ca)[0, 1]
    cerrada = s_c * cc + s_a * ca                        # PIB contrafactual sin nx
    return dict(s_c=s_c, s_a=s_a, m=m, rho_ca=rho,
                izq=1 + s_c, der=s_a * m**2 + 2 * s_c * rho * m,
                sc_sy_real=cc.std() / cy.std(), sc_sy_cerrada=cc.std() / cerrada.std(),
                # control de aditividad del volumen (exacta sólo en base fija, o sea México)
                sh_vol=(a_vol / d["gdp_vol"]).mean(), sh_nom=(a_nom / d["gdp_nom"]).mean())


for code in ["MEX", "USA"]:
    r = aritmetica_cerrada(code)
    print(f"{code}:  s_c={r['s_c']:.3f}  s_a={r['s_a']:.3f}  "
          f"m=sigma_a/sigma_c={r['m']:.2f}  rho_ca={r['rho_ca']:.2f}")
    print(f"      [control] participación media del residuo: volumen {r['sh_vol']:.3f}  "
          f"vs. precios corrientes {r['sh_nom']:.3f}")
    print(f"      condición cerrada  (1+s_c)={r['izq']:.2f}  >  "
          f"s_a m^2 + 2 s_c rho m = {r['der']:.2f}?   "
          f"{'SÍ' if r['izq'] > r['der'] else 'NO — imposible cerrada'}")
    print(f"      sigma_c/sigma_y observado = {r['sc_sy_real']:.2f}   "
          f"contrafactual CERRADO = {r['sc_sy_cerrada']:.2f}\n")

# ROBUSTEZ ETIQUETADA (no es la cifra principal): la misma cuenta con la ventana 1994Q1.
_rob = aritmetica_cerrada("MEX", INI_ROB, FIN)
print(f"[robustez 1994Q1–2019Q4]  MEX: sigma_c/sigma_y observado = {_rob['sc_sy_real']:.2f}   "
      f"contrafactual CERRADO = {_rob['sc_sy_cerrada']:.2f}")

# En ambos países la desigualdad de economía cerrada se viola: el modelo cerrado tiene
# prohibido aritméticamente generar sigma_c > sigma_y con estas participaciones.
assert aritmetica_cerrada("MEX")["izq"] < aritmetica_cerrada("MEX")["der"]

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura, y una advertencia honesta.** La desigualdad se viola en México y en EE. UU.:
# con $m\approx1.9$ y $\rho_{ca}\approx0.4$, una economía cerrada con estas participaciones
# **no puede** tener el consumo más volátil que el producto. Abrir la economía reintroduce
# $\widehat{nx}_t$ y levanta la prohibición. Nótese qué tipo de resultado es éste: la
# aritmética prohíbe el $>1$ en una economía cerrada, y con la ventana canónica México
# tampoco lo exhibe. Lo que hay que explicar no es un cruce del 1; es que el cociente esté en
# $0.97$ cuando el suavizamiento intertemporal predice mucho menos.
#
# Y levantar la prohibición no es lo mismo que explicar la magnitud. El contrafactual dice
# que quitar las exportaciones netas mexicanas bajaría $\sigma_c/\sigma_y$ de $0.97$ a
# $0.94$: la apertura aporta unas **tres centésimas**, una fracción pequeña de la distancia
# entre México y una economía avanzada (EE. UU. está en $0.85$, Alemania en $0.54$). La
# cuenta externa es la *condición de posibilidad*; hace falta además un **mecanismo** que
# mueva el consumo —choques a la tendencia o una prima de riesgo contracíclica—. Ése es el
# resto de la lección.
#
# (Con la ventana de robustez 1994Q1–2019Q4 el contrafactual va de $1.00$ a $0.93$: siete
# centésimas. Que la contribución de la apertura se duplique al añadir cuatro trimestres
# indica que buena parte de ella la aporta la reversión de 1994–95, no la apertura "en
# general". Un dato más para la ficha de medición.)
#
# (La absorción no-consumo se construye por residuo de la identidad del gasto e incluye al
# gobierno: la OCDE QNA congelada en el paquete no trae la formación bruta de capital por
# separado para los ocho países. La sustitución está declarada; el orden de magnitud de
# $m$ no cambia. Segunda advertencia, y es de base de precios: el residuo en **volúmenes**
# sólo es exacto donde el volumen es **aditivo**, es decir en base fija — o sea, en México.
# En los países **encadenados** la suma de componentes no reproduce el PIB fuera del año de
# referencia y el residuo arrastra la discrepancia del encadenamiento. La celda imprime, como
# control de magnitud, la participación media del residuo en volúmenes junto a la misma
# participación a precios corrientes. Ojo con cómo se lee: que las dos coincidan en EE. UU.
# ($0.362$ y $0.364$) y se separen en México ($0.357$ contra $0.319$) **no** dice que el
# residuo mexicano esté peor medido —en base fija el volumen es aditivo y el residuo es
# exacto—; la brecha mexicana es cambio de **precios relativos** respecto de su año base 2018,
# que es justo lo que un volumen a precios de 2018 debe mostrar. Lo que no se puede
# comprobar con esta sola cuenta es la deriva del encadenamiento, y por eso la fila de
# EE. UU. de este bloque es ilustrativa y no una cifra de la ficha de medición.)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. El problema del cierre: la deuda con raíz unitaria
#
# Abrir la economía tiene un precio técnico. El hogar que elige $d_{t+1}$ sujeto a
# $c_t=y_t+d_{t+1}-(1+r_t)d_t$ satisface la condición de Euler
#
# $$u'(c_t)=\beta\,(1+r_{t+1})\,E_t[u'(c_{t+1})].$$
#
# Con $r_t=r$ constante y exógeno, $\beta(1+r)<1$ lleva a desahorrar sin límite y
# $\beta(1+r)>1$ a acumular activos sin límite: sólo $\beta(1+r)=1$ admite estado
# estacionario, y entonces $u'(c_t)=E_t[u'(c_{t+1})]$. La utilidad marginal es una
# **martingala**: el consumo tiene raíz unitaria, $\bar d$ queda **indeterminado**, y los
# momentos de segundo orden que queríamos comparar con la sección 1 sencillamente no existen.
#
# **La solución de Schmitt-Grohé y Uribe (2003):** hacer la prima creciente en la deuda,
#
# $$r_t=r^{*}+\psi\left(e^{\,d_t-\bar d}-1\right),$$
#
# tomada como dada por el hogar porque depende de la deuda **agregada**. El estado
# estacionario determinista queda fijado por $1=\beta[1+r^{*}+p(\bar d)]$ y, linealizando la
# Euler con $u'(c)=c^{-\gamma}$ alrededor de $\bar d$,
#
# $$\gamma\left(\log c_{t+1}-\log c_t\right)\simeq\frac{p'(\bar d)}{1+\bar r}\left(d_t-\bar d\right):$$
#
# cuando la deuda excede su nivel de largo plazo el consumo **crece** —el país ahorra— y la
# deuda regresa a $\bar d$.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La demostración numérica
# Log-linealizamos una economía **de dotación** (sin capital) alrededor del estado
# estacionario. Con $D_t\equiv(d_t-\bar d)/\bar y$ y $\hat y_t$ un AR(1), el sistema es
#
# $$D_t=\Big[1+\bar r+\tfrac{\bar d}{\bar y}\psi\bar y\Big]D_{t-1}+\tfrac{\bar c}{\bar y}\hat c_t-\hat y_t,
# \qquad E_t\hat c_{t+1}=\hat c_t+\frac{\psi\bar y}{\gamma(1+\bar r)}D_t,$$
#
# y lo resolvemos con `klein_solve`. El estado es $[D_{t-1},\hat y_t]$ y el control $\hat c_t$;
# el veredicto está en dos números: las banderas de existencia/unicidad `eu` y el mayor valor
# propio de la matriz de transición $G$.

# %% slideshow={"slide_type": "fragment"}
from puremacro.dsge import klein_solve


def dotacion(psi, rstar=0.01, gamma=2.0, rho_y=0.85, dbar_y=1.4):
    """Economía pequeña y abierta de dotación, log-linealizada. Estados [D_{t-1}, y_t]."""
    ybar = 1.0
    cbar = ybar - rstar * dbar_y * ybar          # nx = r*d en estado estacionario
    psi_y = psi * ybar                            # sensibilidad de r a D (en unidades de y)
    iD, iY, iC = 0, 1, 2
    Dn = np.zeros(3)                              # D_t como función de (D_{t-1}, y_t, c_t)
    Dn[iD] = 1 + rstar + dbar_y * psi_y
    Dn[iC] = cbar / ybar
    Dn[iY] = -1.0
    A, B = np.zeros((3, 3)), np.zeros((3, 3))
    A[0, iD] = 1.0; B[0, :] = Dn                              # ley de movimiento de la deuda
    A[1, iY] = 1.0; B[1, iY] = rho_y                          # dotación AR(1)
    A[2, iC] = 1.0; B[2, iC] = 1.0                            # Euler del consumo
    B[2, :] += psi_y * Dn / (gamma * (1 + rstar))
    return klein_solve(A, B, n_pre=2)


print("psi        eu        valores propios de G")
for psi in [0.0, 1e-8, 1e-4, 1e-3, 1e-2, 5e-2]:
    s = dotacion(psi)
    print(f"{psi:<10.0e} {str(s.eu):9s} {np.round(np.linalg.eigvals(s.G), 7)}")

# Sin prima (psi = 0) Blanchard-Kahn falla: NO hay solución estable. Con psi > 0 sí la hay,
# y el valor propio dominante —la persistencia de la deuda— se despega de 1.
assert dotacion(0.0).eu == (0, 0)
assert dotacion(1e-3).eu == (1, 1)
assert np.abs(np.linalg.eigvals(dotacion(1e-8).G)).max() > 0.99999

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El abanico de trayectorias
# Simulamos 200 economías **con los mismos parámetros**, cada una con su propia sucesión de
# choques de dotación extraída del mismo proceso, durante 200 trimestres, y graficamos la
# deuda. (La dispersión del abanico es justamente eso: economías idénticas a las que la
# suerte les tocó distinta.) Con $\psi\to0$ la dispersión **crece sin cota** —es un paseo
# aleatorio, cada historia recuerda para siempre su suerte pasada—. Con $\psi=10^{-3}$ la
# dispersión se estabiliza: existe un $\bar d$ y los momentos existen.

# %% slideshow={"slide_type": "slide"}
def abanico(psi, T=200, n=200, sig=0.02, seed=3):
    sol = dotacion(psi)
    rng = np.random.default_rng(seed)            # datos SIMULADOS (declarados)
    D = np.zeros((n, T))
    eps = rng.normal(0.0, sig, (n, T))
    for i in range(n):
        x = np.zeros(2)
        for t in range(1, T):
            x = sol.G @ x
            x[1] += eps[i, t]
            D[i, t] = x[0]
    return D


fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
for ax, psi, tit in zip(axes, [1e-8, 1e-3],
                        [r"A. $\psi\to 0$: raíz unitaria", r"B. $\psi=10^{-3}$: prima creciente"]):
    D = abanico(psi)
    for i in range(0, 200, 4):
        ax.plot(D[i], color="0.72", lw=0.5)
    ax.plot(D.std(axis=0), color="0.0", lw=1.8, label=r"$\pm$ desv. est. de corte transversal")
    ax.plot(-D.std(axis=0), color="0.0", lw=1.8)
    ax.axhline(0, color="0.3", lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel("trimestre"); ax.set_title(tit)
    print(f"psi={psi:.0e}:  sd(D) en t=50: {D[:, 50].std():.2f}   "
          f"t=100: {D[:, 100].std():.2f}   t=199: {D[:, -1].std():.2f}")
axes[0].set_ylabel(r"deuda externa $D_t=(d_t-\bar d)/\bar y$")
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()

# ¿Qué tan grande es la prima que basta? Aquí ybar = 1 es el PIB TRIMESTRAL, así que
# +25% del PIB ANUAL de deuda son dD = 0.25 * 4 = 1.0 unidades. La prima linealizada
# sube psi*ybar*dD por trimestre; la exacta, psi*(exp(dD) - 1).
psi_ref, dD = 1e-3, 0.25 * 4
lin, exa = psi_ref * 1.0 * dD, psi_ref * (np.exp(dD) - 1.0)
print(f"\npsi={psi_ref:.0e} y deuda +25% del PIB anual (dD={dD:.1f}):")
print(f"  prima linealizada: {1e4*lin:5.1f} pb trimestrales = {4e4*lin:5.1f} pb anuales")
print(f"  prima exacta:      {1e4*exa:5.1f} pb trimestrales = {4e4*exa:5.1f} pb anuales")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** En el panel A la desviación estándar de corte transversal crece
# aproximadamente como $\sqrt{t}$ —la firma de un paseo aleatorio— y no hay nivel de deuda al
# cual volver. En el panel B se aplana en pocas décadas. Nótese cuán **pequeña** es la prima
# que basta: como imprime la celda anterior, con $\psi=10^{-3}$ subir la deuda en 25% del PIB
# **anual** encarece el crédito unos **40 puntos base anuales** (10 pb por trimestre; 69 pb
# anuales si se usa la prima exponencial exacta en vez de su linealización, porque
# $\psi(e^{1}-1)\simeq1.7\psi$). Recuerda que aquí $\bar y=1$ es el PIB *trimestral*, así que
# 25% del PIB anual equivale a $\Delta D=1.0$. El cierre no es una fuerza económica grande;
# es una tachuela que fija el modelo al pizarrón. El mazo lo recuerda: hay al menos otros tres cierres
# —costos de cartera cuadráticos, factor de descuento endógeno, mercados completos— y a
# frecuencias de ciclo producen dinámicas prácticamente indistinguibles. **Difieren en el
# largo plazo, no en el ciclo.**

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Tasa de interés y riesgo país
#
# La prima no sólo ancla la deuda: es una fuente de fluctuaciones por derecho propio.
# Separemos sus dos componentes,
#
# $$r_t=r^{*}+p_t,\qquad p_t=p(d_t)+\tilde p_t,\qquad \tilde p_t=\rho_p\tilde p_{t-1}+\epsilon_{pt},$$
#
# donde $\tilde p_t$ es exógeno y **contracíclico**: la prima sube cuando la actividad cae.
# **Neumeyer y Perri (2005)** le dan mordida real con **capital de trabajo**: la empresa
# prepaga una fracción $\zeta$ de su nómina y resuelve
# $\max\,a_tk_{t-1}^{\alpha}l_t^{1-\alpha}-w_tl_t(1+\zeta r_t)-r_{kt}k_{t-1}$, de donde
# $(1-\alpha)a_tk^{\alpha}_{t-1}l^{-\alpha}_t=w_t(1+\zeta r_t)$. Como el hogar iguala su tasa
# marginal de sustitución a $w_t$, el equilibrio cumple
#
# $$\mathrm{TMS}_t=\frac{\mathrm{PML}_t}{1+\zeta r_t},\qquad 1-\tau_{lt}=\frac{1}{1+\zeta r_t}:$$
#
# **una tasa de interés contracíclica es una cuña laboral contracíclica.** Mueve las horas sin
# mover la productividad total de los factores — y por eso predice
# $\mathrm{corr}(h,\,y/h)<0$, el criterio empírico que ningún mecanismo de preferencias había
# logrado. **Uribe y Yue (2006)** añaden que la causalidad corre en los dos sentidos: la tasa
# mundial mueve las primas y la actividad emergente, pero las primas también responden a las
# condiciones internas y amplifican el choque original.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Lo que sí podemos medir, y lo que no
# El **EMBI+ México** de J.P. Morgan es propietario y **no está** en el paquete congelado del
# curso; no podemos calcular la correlación de Neumeyer–Perri con su propia serie. Usamos tres
# aproximaciones, cada una con su defecto declarado:
#
# 1. **VIX** (precio del riesgo global). En la literatura el factor común de los *spreads*
#    emergentes es abrumadoramente global; el VIX es su representante estándar. Defecto: es
#    global, no mexicano — mide el $r^{*}$ y el apetito de riesgo del mundo, no el riesgo país.
# 2. **Diferencial interbancario** $i^{MX}_t-i^{FF}_t$. Defecto grave: por paridad descubierta
#    mezcla riesgo país con **depreciación esperada** del peso.
# 3. **Tasa real ex post en pesos** $i^{MX}_t-\pi_t$ (IPC interanual). Defecto: es la tasa de
#    política deflactada hacia atrás, no la tasa a la que el país se financia en dólares.
#
# La ventana es **1997Q1–2019Q4**, y no es una elección estética: la interbancaria mexicana
# congelada del paquete arranca en **1997-01**. Usamos la misma ventana para las tres
# aproximaciones —también para el VIX, que existe desde 1990— para que las tres
# correlaciones sean comparables; de ahí que las tres tengan la misma $n$.

# %% slideshow={"slide_type": "fragment"}
i_mx = fred("IR3TIB01MXM156N")          # interbancaria 3 meses, México, % anual
pi_mx = fred("CPALTT01MXM659N")         # IPC México, variación interanual, %
i_ff = fred("FEDFUNDS")                 # fondos federales EE.UU., % anual
vix = fred("VIXCLS")                    # VIX, promedio trimestral

y_mx = pd.Series(100 * np.log(qna.loc["MEX"].sort_index()["gdp_vol"].to_numpy()),
                 index=qna.loc["MEX"].sort_index().index)

proxies = {"tasa real ex post (pesos)": i_mx - pi_mx,
           "diferencial i_MX - i_FF": i_mx - i_ff,
           "VIX (riesgo global)": vix}

print("Correlación con el ciclo del PIB de México (Hamilton), 1997Q1–2019Q4\n")
corr_riesgo = {}
for nombre, serie in proxies.items():
    j = pd.concat([y_mx.rename("y"), serie.rename("x")], axis=1, sort=True).dropna()
    j = j[(j.index >= "1997-01-01") & (j.index <= "2019-10-01")]
    cy, cx = ciclo(j["y"].to_numpy()), ciclo(j["x"].to_numpy())
    ok = ~np.isnan(cy) & ~np.isnan(cx)
    corr_riesgo[nombre] = np.corrcoef(cx[ok], cy[ok])[0, 1]
    print(f"  {nombre:28s} n={ok.sum():3d}   corr = {corr_riesgo[nombre]:+.2f}")

assert corr_riesgo["VIX (riesgo global)"] < -0.3      # el riesgo global es contracíclico

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura, incluida la sorpresa.** El VIX es netamente **contracíclico** frente al ciclo
# mexicano ($-0.50$): cuando el precio global del riesgo sube, México se contrae. Es la
# lectura de Uribe–Yue en su versión más cruda, y es consistente con Neumeyer–Perri.
#
# El diferencial interbancario es contracíclico pero débil ($-0.22$), y la **tasa real ex
# post en pesos es procíclica** ($+0.44$) — lo contrario de lo que dice el modelo. No es un
# error: es un recordatorio de qué tasa aparece en la teoría. La $r_t$ de Neumeyer–Perri es
# la tasa **en dólares a la que el país se endeuda con el resto del mundo**; la que medimos
# aquí es la **tasa de política** del Banco de México deflactada, y en un régimen de metas de
# inflación el banco central sube tasas en los auges. Confundir las dos es el error más común
# al llevar este modelo a datos mexicanos. Sin el EMBI no podemos cerrar el punto: lo dejamos
# dicho, no resuelto.

# %% slideshow={"slide_type": "slide"}
j = pd.concat([y_mx.rename("y"), vix.rename("vix")], axis=1, sort=True).dropna()
j = j[(j.index >= "1997-01-01") & (j.index <= "2019-10-01")]
cy, cv = ciclo(j["y"].to_numpy()), ciclo(j["vix"].to_numpy())
ok = ~np.isnan(cy) & ~np.isnan(cv)
fechas = j.index[ok]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7),
                         gridspec_kw={"width_ratios": [1.9, 1]})
ax = axes[0]
ax.plot(fechas, cy[ok] / cy[ok].std(), color="0.10", lw=1.5, label="ciclo del PIB de México")
ax.plot(fechas, -cv[ok] / cv[ok].std(), color="0.45", lw=1.5, ls=(0, (4, 2)),
        label="VIX (signo invertido)")
ax.axhline(0, color="0.8", lw=0.7)
ax.set_ylabel("desviaciones estándar"); ax.set_xlabel("trimestre")
ax.set_title("A. El riesgo global sube cuando México cae")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
ax.set_axisbelow(True)
nm = list(corr_riesgo)
ax.barh(range(len(nm)), [corr_riesgo[k] for k in nm], color=["0.70", "0.45", "0.15"])
ax.set_yticks(range(len(nm)))
ax.set_yticklabels(["real ex post\n(pesos)", "difer.\n$i_{MX}-i_{FF}$", "VIX"], fontsize=8)
ax.axvline(0, color="0.2", lw=0.9)
ax.set_xlabel(r"corr con el ciclo del PIB")
ax.set_title("B. Depende de qué tasa midas")
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. El modelo completo, resuelto con Klein
#
# Juntamos las piezas: preferencias **GHH** (Greenwood–Hercowitz–Huffman), que eliminan el
# efecto riqueza sobre la oferta de trabajo y hacen que las horas dependan sólo del salario;
# capital con **costos de ajuste** convexos $\Phi_t=\frac{\phi}{2}(k_t-k_{t-1})^2/\bar k$ (la
# lección 10 del mismo mazo); **capital de trabajo** $\zeta$; y la **prima de riesgo**
# $r_t=r^{*}+\psi(d_{t-1}-\bar d)+\tilde p_t$ del cierre. El equilibrio log-linealizado es
#
# $$\hat y_t=\hat a_t+\alpha\hat k_{t-1}+(1-\alpha)\hat h_t,\qquad
# \omega\hat h_t=\hat y_t-\frac{\zeta}{1+\zeta\bar r}\,(r_t-\bar r),$$
#
# $$\hat\lambda_t=E_t\hat\lambda_{t+1}+\frac{E_t(r_{t+1}-\bar r)}{1+\bar r},\qquad
# \hat\lambda_t+\phi(\hat k_t-\hat k_{t-1})=E_t\hat\lambda_{t+1}
# +\beta(\bar r+\delta)\big(E_t\hat y_{t+1}-\hat k_t\big)+\beta\phi\big(E_t\hat k_{t+1}-\hat k_t\big),$$
#
# más la ley de la deuda. Estados: $[\hat k_{t-1},D_{t-1},\hat a_t,\tilde p_t]$; controles:
# $[\hat c_t,\hat k_t]$. Calibración trimestral estándar: $\alpha=0.32$, $\delta=0.025$,
# $r^{*}=1\%$, $\gamma=2$, $\omega=1.6$ (Frisch $\approx1.7$), $\phi=8$, $\zeta=1$,
# $\psi=10^{-3}$, $\bar d/\bar y=1.4$ (35% del PIB anual).
#
# **Recuerda el puente de notación** del inicio de la lección: aquí $\gamma$ es la CRRA
# (en el mazo, $\sigma$), $\omega$ es la curvatura GHH del trabajo (en el mazo, $\nu$; allí
# $\omega$ es la curvatura de $\delta(u)$) y $\psi$ es la pendiente de la prima de deuda
# (en el mazo, $p'(\bar d)$; allí $\psi$ es la elasticidad adicional del producto en las
# horas). $\alpha$ y $\phi$ sí significan lo mismo en ambos sitios.

# %% slideshow={"slide_type": "fragment"}
def modelo_soe(alpha=0.32, delta=0.025, rstar=0.01, gamma=2.0, omega=1.6,
               psi=1e-3, phi=8.0, zeta=1.0, rho_a=0.90, rho_p=0.85, dbar_y=1.4):
    """Modelo neoclásico de economía pequeña y abierta, log-linealizado (forma de Klein)."""
    beta = 1.0 / (1.0 + rstar)
    ky = alpha / (rstar + delta)
    ybar = ky ** (alpha / (1.0 - alpha))                 # normalización: h=1, a=1
    kbar, ibar = ky * ybar, delta * ky * ybar
    dbar = dbar_y * ybar
    cbar = ybar - ibar - rstar * dbar
    chi = (1.0 - alpha) * ybar / (1.0 + zeta * rstar)    # escala de la desutilidad GHH
    sbar = cbar - chi / omega                            # argumento de la utilidad GHH
    chi_c, chi_h = gamma * cbar / sbar, gamma * chi / sbar
    zeff, psi_y, den = zeta / (1 + zeta * rstar), psi * ybar, omega - 1.0 + alpha
    iK, iD, iA, iP, iC, iKn = range(6)
    e = np.eye(6)

    R = psi_y * e[iD] + e[iP]                            # r_t - rbar (depende de D_{t-1}, p_t)
    H = (e[iA] + alpha * e[iK] - zeff * R) / den         # horas
    Y = e[iA] + alpha * e[iK] + (1 - alpha) * H          # producto
    Inv = (e[iKn] - (1 - delta) * e[iK]) / delta         # inversión
    LAM = -chi_c * e[iC] + chi_h * H                     # utilidad marginal
    Dn = ((1 + rstar) * e[iD] + dbar_y * R               # deuda elegida en t
          + (cbar / ybar) * e[iC] + (ibar / ybar) * Inv - Y)

    A, B, = np.zeros((6, 6)), np.zeros((6, 6))
    A[0, iK] = 1.0; B[0, iKn] = 1.0                                  # k_t pasa a ser estado
    A[1, iD] = 1.0; B[1, :] = Dn                                     # ley de la deuda
    A[2, iA] = 1.0; B[2, iA] = rho_a                                 # productividad AR(1)
    A[3, iP] = 1.0; B[3, iP] = rho_p                                 # prima exógena AR(1)
    A[4, :] = LAM                                                    # Euler de bonos
    B[4, :] = LAM - (psi_y * Dn + rho_p * e[iP]) / (1 + rstar)
    A[5, :] = LAM + beta * (rstar + delta) * Y + beta * phi * e[iKn]  # Euler del capital
    B[5, :] = (LAM + phi * e[iKn] - phi * e[iK]
               + beta * (rstar + delta) * e[iKn] + beta * phi * e[iKn])

    sol = klein_solve(A, B, n_pre=4)
    ss = dict(Y=Y, H=H, Inv=Inv, R=R, cbar=cbar, ibar=ibar, ybar=ybar)
    return sol, ss


sol, ss = modelo_soe()
print(f"existencia/unicidad eu = {sol.eu}   (1,1) = solución estable y única")
print(f"mayor valor propio de G = {np.abs(np.linalg.eigvals(sol.G)).max():.7f}")
print(f"con psi -> 0:            {np.abs(np.linalg.eigvals(modelo_soe(psi=1e-9)[0].G)).max():.7f}"
      "   <- la raíz unitaria de la sección 3, en el modelo completo")
assert sol.eu == (1, 1)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Funciones de impulso-respuesta
# Dos experimentos: (a) un choque de **productividad** transitorio de $+1\%$ con
# $\rho_a=0.90$; (b) un choque de **prima de riesgo** de $+100$ puntos base anuales con
# $\rho_p=0.85$. Ambos son puramente exógenos; el interés está en el contraste.

# %% slideshow={"slide_type": "fragment"}
def irf(sol, ss, choque, tam, T=25):
    """Respuesta al choque 'a' (productividad) o 'p' (prima), en % (r en % anual)."""
    x, y = np.zeros((T, 4)), np.zeros((T, 2))
    x[0, 2 if choque == "a" else 3] = tam
    y[0] = sol.F @ x[0]
    for t in range(1, T):
        x[t] = sol.G @ x[t - 1]
        y[t] = sol.F @ x[t]
    z = np.column_stack([x, y])
    return dict(y=100 * (z @ ss["Y"]), c=100 * z[:, 4], i=100 * (z @ ss["Inv"]),
                h=100 * (z @ ss["H"]), r=400 * (z @ ss["R"]))


irf_a = irf(sol, ss, "a", 0.01)        # +1% de PTF
irf_p = irf(sol, ss, "p", 0.0025)      # +100 pb anuales de prima

print("impacto (t=0), en %:            y        c        i        h      r (% anual)")
for nm, o in [("choque de PTF  +1%", irf_a), ("choque de prima +100pb", irf_p)]:
    print(f"  {nm:24s}{o['y'][0]:+7.2f}  {o['c'][0]:+7.2f}  {o['i'][0]:+7.2f}  "
          f"{o['h'][0]:+7.2f}  {o['r'][0]:+7.2f}")

# El choque de prima hace caer el consumo MAS que el producto: la clave de un cociente alto.
assert abs(irf_p["c"][0]) > abs(irf_p["y"][0])
assert abs(irf_a["c"][0]) < abs(irf_a["y"][0])

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.6), sharex=True)
h = np.arange(20)
for col, (o, tit) in enumerate([(irf_a, "Choque de productividad ($+1\\%$)"),
                                (irf_p, "Choque de prima ($+100$ pb anuales)")]):
    ax = axes[0, col]
    for k, lab, gr, ls in [("y", "producto", "0.05", "-"),
                           ("c", "consumo", "0.40", (0, (4, 2))),
                           ("h", "horas", "0.62", (0, (1, 1)))]:
        ax.plot(h, o[k][:20], color=gr, lw=1.6, ls=ls, label=lab)
    ax.axhline(0, color="0.8", lw=0.7); ax.set_title(tit)
    if col == 0:
        ax.set_ylabel("desviación, %")
    ax.legend(fontsize=8)

    ax = axes[1, col]
    ax.plot(h, o["i"][:20], color="0.05", lw=1.6, label="inversión (%)")
    ax.plot(h, o["r"][:20], color="0.50", lw=1.6, ls=(0, (4, 2)),
            label="tasa de interés (pp anuales)")
    ax.axhline(0, color="0.8", lw=0.7); ax.set_xlabel("trimestres")
    if col == 0:
        ax.set_ylabel("desviación")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** Ante el choque de **productividad** todo sube y el consumo sube **menos** que
# el producto: el hogar reconoce que el choque es transitorio, ahorra, y la deuda externa
# baja. Es el mecanismo de suavizamiento de siempre, con signo equivocado para México.
#
# Ante el choque de **prima**, en cambio, el consumo cae unas **tres veces** más que el
# producto. Dos canales operan a la vez: la Euler de bonos hace caer el consumo hoy contra un
# consumo futuro más caro, y el capital de trabajo encarece la nómina y **contrae las horas**
# sin tocar la productividad total de los factores — la cuña laboral de Neumeyer–Perri. La
# inversión se desploma. Éste, y no el choque de PTF, es el candidato natural a explicar
# un $\sigma_c/\sigma_y$ tan pegado al 1 como el mexicano.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Los momentos del modelo contra los de la sección 1
# Simulamos el modelo, **filtramos la simulación con Hamilton igual que los datos** (si no, la
# comparación es tramposa) y calculamos los mismos momentos. Se reportan tres versiones: sólo
# choques de productividad, sólo de prima, y ambos.

# %% slideshow={"slide_type": "fragment"}
from puremacro.cycles import hamilton_filter

SIG_A, SIG_P = 0.009, 0.0016   # calibrados para que sigma_y del modelo quede junto al de México


def momentos_modelo(sol, ss, sig_a=SIG_A, sig_p=SIG_P, T=4000, seed=11):
    rng = np.random.default_rng(seed)                    # datos SIMULADOS (declarados)
    x, y = np.zeros((T, 4)), np.zeros((T, 2))
    ea, ep = rng.normal(0, sig_a, T), rng.normal(0, sig_p, T)
    for t in range(1, T):
        x[t] = sol.G @ x[t - 1]
        x[t, 2] += ea[t]; x[t, 3] += ep[t]
        y[t] = sol.F @ x[t]
    z = np.column_stack([x, y])[100:]
    Y, C = 100 * (z @ ss["Y"]), 100 * z[:, 4]
    I, R = 100 * (z @ ss["Inv"]), 400 * (z @ ss["R"])
    NX = Y - (ss["cbar"] / ss["ybar"]) * C - (ss["ibar"] / ss["ybar"]) * I
    s = {k: ciclo(v) for k, v in dict(y=Y, c=C, i=I, r=R, nx=NX).items()}
    m = ~np.isnan(s["y"])
    s = {k: v[m] for k, v in s.items()}
    sy = s["y"].std()
    return dict(sd_y=sy, sc_sy=s["c"].std() / sy, si_sy=s["i"].std() / sy,
                corr_cy=np.corrcoef(s["c"], s["y"])[0, 1],
                corr_nxy=np.corrcoef(s["nx"], s["y"])[0, 1],
                corr_ry=np.corrcoef(s["r"], s["y"])[0, 1])


filas = {"modelo: sólo PTF": momentos_modelo(sol, ss, sig_p=0.0),
         "modelo: sólo prima": momentos_modelo(sol, ss, sig_a=0.0),
         "modelo: ambos choques": momentos_modelo(sol, ss),
         "DATOS México": dict(sd_y=tabla.loc["MEX", "sd_y"], sc_sy=tabla.loc["MEX", "sc_sy"],
                              si_sy=aritmetica_cerrada("MEX")["m"] * tabla.loc["MEX", "sc_sy"],
                              corr_cy=tabla.loc["MEX", "corr_cy"],
                              corr_nxy=tabla.loc["MEX", "corr_nxy"],
                              corr_ry=corr_riesgo["VIX (riesgo global)"]),
         "DATOS EE. UU.": dict(sd_y=tabla.loc["USA", "sd_y"], sc_sy=tabla.loc["USA", "sc_sy"],
                               si_sy=aritmetica_cerrada("USA")["m"] * tabla.loc["USA", "sc_sy"],
                               corr_cy=tabla.loc["USA", "corr_cy"],
                               corr_nxy=tabla.loc["USA", "corr_nxy"], corr_ry=np.nan)}
print("Momentos del ciclo (Hamilton). 'si_sy' en los datos es absorción no-consumo;")
print("'corr_ry' en los datos es la proxy VIX. Ambas sustituciones están declaradas.\n")
print(pd.DataFrame(filas).T.round(2).to_string())

# %% [markdown] slideshow={"slide_type": "subslide"}
# **El veredicto.** Con los dos choques el modelo acierta $\sigma_y$ (por construcción) y
# queda cerca en $\sigma_i/\sigma_y$ ($2.07$ contra $1.84$, y el $1.84$ del dato es la
# absorción no-consumo, no la inversión), pero **falla el hecho central**: entrega
# $\sigma_c/\sigma_y=0.74$
# contra $0.97$ en los datos —una brecha de **veintitrés centésimas**, que es exactamente el
# tamaño del reto: no "el modelo no cruza el 1", sino "el modelo se queda muy corto"— y una
# balanza comercial **procíclica** ($+0.26$) contra $+0.01$ (Hamilton) o $-0.29$ (HP) en los
# datos. El renglón *sólo prima* muestra por qué: aislado, el
# choque de prima genera $\sigma_c/\sigma_y=1.49$ y $\mathrm{corr}(nx/y,y)=-0.73$ — los dos
# hechos, con el signo correcto. Aislado, el choque de PTF transitorio genera lo contrario.
# El modelo no falla por falta de mecanismo; falla por la **mezcla** de choques que le
# impusimos. La sección 6 convierte esa observación en el debate central de la literatura.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 6. "El ciclo es la tendencia" — y el contrapunto honesto
#
# **Aguiar y Gopinath (2007)** proponen la explicación que da nombre a su artículo: las
# economías emergentes no reciben choques transitorios más grandes, sino choques a la **tasa
# de crecimiento tendencial** de la productividad,
# $\Delta\log A_t=(1-\rho_a)g_a+\rho_a\Delta\log A_{t-1}+\epsilon_t$. Si la productividad de
# mañana **hereda** el choque de hoy, la renta permanente sube tanto como la corriente y el
# hogar **no tiene motivo para suavizar**: el consumo salta con el producto, o por encima.
# El modelo hay que deflactarlo por $A_t^{1/(1-\alpha)}$ para eliminar la tendencia
# estocástica (el álgebra está en el mazo), y el descuento efectivo hereda los choques
# permanentes.
#
# Nuestro modelo es estacionario, así que no podemos meter una tendencia estocástica
# genuina. Lo que sí podemos hacer —y es la **aproximación estacionaria** de la hipótesis— es
# subir $\rho_a$ hacia 1 y preguntar cuánta persistencia hace falta para que
# $\sigma_c/\sigma_y$ alcance el **valor observado** de México ($0.97$; imprimimos también el
# precio de cruzar el $1$ redondo, que es la cifra de la ventana de robustez, para que se vea
# cuánto cuesta esa diferencia). Y hacer la misma pregunta a la hipótesis rival: cuánta
# volatilidad de la prima hace falta. Las dos curvas, en el mismo modelo, con el mismo
# filtro. (Como el modelo es lineal, $\sigma_c/\sigma_y$ depende sólo de la escala
# **relativa** de los dos choques: en el panel A, donde sólo hay choque de productividad, no
# depende de la escala en absoluto; en el panel B el eje horizontal es precisamente esa
# escala relativa, con $\sigma_a$ fija en su valor calibrado.)

# %% slideshow={"slide_type": "fragment"}
RHOS = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.997, 0.999]
SIGPS = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.020, 0.030]

curva_rho = [momentos_modelo(*modelo_soe(rho_a=r), sig_p=0.0)["sc_sy"] for r in RHOS]
curva_p = [momentos_modelo(sol, ss, sig_p=sp)["sc_sy"] for sp in SIGPS]

dato_mx = tabla.loc["MEX", "sc_sy"]      # 0.97 con la ventana canónica (NO es > 1)
cruce_rho = np.interp(dato_mx, curva_rho, RHOS)
cruce_p = np.interp(dato_mx, curva_p, SIGPS)
print(f"sigma_c/sigma_y en los datos de México (1995Q1–2019Q4, Hamilton) = {dato_mx:.3f}")
print(f"  persistencia necesaria (sólo PTF):     rho_a   = {cruce_rho:.4f}")
print(f"  volatilidad de prima necesaria:        sigma_p = {cruce_p:.4f} trimestral"
      f"  ({40000 * cruce_p:.0f} pb anuales por innovación)")
# El objetivo es el DATO observado, no un 1 redondo: si además quisiéramos cruzar el 1
# (la cifra de la ventana de robustez 1994Q1) el precio sube, y conviene verlo.
print(f"  [para cruzar el 1.00]                  rho_a   = {np.interp(1.0, curva_rho, RHOS):.4f}"
      f"   sigma_p = {40000 * np.interp(1.0, curva_p, SIGPS):.0f} pb anuales")
assert cruce_rho > 0.99      # sólo una raíz casi unitaria basta por el canal de persistencia

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
ax = axes[0]
ax.plot(RHOS, curva_rho, color="0.10", lw=1.8, marker="o", ms=4)
ax.axhline(dato_mx, color="0.0", lw=1.1, ls=(0, (4, 2)))
ax.text(0.705, dato_mx + 0.02, f"México (dato: {dato_mx:.2f})", fontsize=8)
ax.axvline(cruce_rho, color="0.55", lw=1.0, ls=(0, (1, 1)))
ax.set_xlabel(r"persistencia de la productividad $\rho_a$")
ax.set_ylabel(r"$\sigma_c/\sigma_y$ del modelo")
ax.set_title("A. Hipótesis de la tendencia (Aguiar–Gopinath)")

ax = axes[1]
ax.plot([40000 * s for s in SIGPS], curva_p, color="0.10", lw=1.8, marker="s", ms=4)
ax.axhline(dato_mx, color="0.0", lw=1.1, ls=(0, (4, 2)))
ax.axvline(40000 * cruce_p, color="0.55", lw=1.0, ls=(0, (1, 1)))
ax.text(40000 * cruce_p + 30, 0.70, f"{40000 * cruce_p:.0f} pb", fontsize=8)
ax.set_xlabel(r"desv. est. de la innovación a la prima (pb anuales)")
ax.set_title("B. Hipótesis de la prima (Neumeyer–Perri)")
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lo que dicen las curvas.** El canal de persistencia alcanza el $0.97$ mexicano sólo con
# $\rho_a\approx0.994$ —y el $1.00$ de la ventana de robustez, con $\rho_a\approx0.997$—:
# prácticamente una **raíz unitaria** en la productividad. Ése es el sentido preciso —y
# exigente— de *"el ciclo es la tendencia"*: no basta con choques "bastante persistentes",
# hace falta que sean casi permanentes. El canal de la prima llega al mismo cociente con
# innovaciones de unos **265** puntos base anuales (unos 290 si se le pide cruzar el 1), un
# orden de magnitud plausible para un emergente (el EMBI de México se movió cientos de puntos
# base en 1995, 2008 y 2020) pero nada barato. **Ninguna de las dos puertas es gratis**; ése
# es el estado real del debate, no un empate cómodo. Y fíjate en la sensibilidad: mover el
# dato tres centésimas mueve la $\rho_a$ exigida tres milésimas, que a esa altura de la escala
# es muchísimo. La ficha de medición de la sección 1 no era un formalismo.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El contrapunto: García-Cicco, Pancrazi y Uribe (2010)
#
# Aguiar y Gopinath estiman con muestras de dos décadas. García-Cicco, Pancrazi y Uribe
# repiten el ejercicio con series **anuales de 1900 a 2005** para Argentina y México, y el
# veredicto se invierte justo en el momento que motivaba la hipótesis. Las cifras que siguen
# son **citadas del artículo**, no calculadas aquí:
#
# - Estimado por el método generalizado de momentos con México 1900–2005, el modelo de
#   tendencia predice $\sigma(\Delta c)/\sigma(\Delta y)=3.1/5.2=0.60$ —consumo **menos**
#   volátil que el producto— cuando el siglo de datos da $6.2/4.1=1.50$. El fallo es de
#   **subpredicción**: es el error que no hay que reintroducir al contar esta historia.
# - Con Argentina, la autocorrelación de $nx/y$ del modelo es casi plana en la unidad
#   ($0.99$) frente a $0.58$ en el dato, y su desviación estándar es $106.6$ contra $5.2$.
#   Acierta el signo de $\mathrm{corr}(nx/y,y)$ ($-0.02$ contra $-0.04$) pero **subestima**
#   las correlaciones con consumo e inversión.
# - La muestra larga **no apaga por sí sola** el choque de tendencia: sin fricciones la
#   mediana posterior es $\sigma_g=0.030$ con $\rho_g=0.83$; sólo al admitir la **fricción
#   financiera** cae a $\sigma_g=0.0071$ con $\rho_g=0.35$, y la verosimilitud marginal
#   logarítmica sube de $547.6$ a $600.6$. Apaga la tendencia el modelo aumentado, no el
#   siglo de datos.
#
# El desacuerdo sigue abierto en dos frentes: sobre la **muestra** —un siglo mezcla
# regímenes cambiarios, comerciales y de política monetaria muy distintos— y sobre **qué
# momentos** deben disciplinar la estimación. Nuestra figura no arbitra el debate: lo
# encuadra. Muestra que las dos hipótesis pueden generar el hecho, y a qué precio en
# parámetros cada una.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 7. Términos de intercambio y paradas súbitas
#
# Supongamos que el financiamiento externo se corta: la deuda deja de poder crecer y queda
# sujeta a un tope, $d_{t+1}\le d_{\max}$. La ley de movimiento deja de ser una elección y se
# vuelve una **restricción**,
#
# $$nx_t=(1+r_t)d_t-d_{t+1}\ \ge\ (1+r_t)d_t-d_{\max},$$
#
# y como $c_t+i_t=y_t-nx_t$ con un producto casi predeterminado en el corto plazo, la mejora
# forzada de las exportaciones netas se paga **peso por peso con absorción**. No hace falta un
# choque a la productividad total de los factores para producir una recesión profunda: basta
# con que el acreedor cambie de opinión.
#
# Con dos bienes el mecanismo se enriquece. Los **términos de intercambio** $p^x_t/p^m_t$
# trasladan a la renta real lo que no está en los volúmenes: una caída reduce el ingreso real
# para un volumen dado de exportaciones. Si se percibe transitoria el país se endeuda y la
# cuenta corriente se deteriora; si se percibe permanente, la absorción cae. El ajuste exige
# además una depreciación real.
#
# El retrato canónico es México 1994–95. Lo calculamos con el mismo panel.

# %% slideshow={"slide_type": "fragment"}
anual = qna.loc["MEX"].resample("YS").sum()
nx_y = 100 * (anual["exp_nom"] - anual["imp_nom"]) / anual["gdp_nom"]
g_y = 100 * anual["gdp_vol"].pct_change()
g_c = 100 * anual["conh_vol"].pct_change()

print("México, episodios de reversión de la cuenta externa (OCDE QNA, anual)\n")
def col(v, w):
    return f"{'—':>{w}}" if not np.isfinite(v) else f"{v:+{w}.2f}"


print("año    nx/y (%)   Δ nx/y (pp)   PIB real (%)   consumo hogares (%)")
for a in [1994, 1995, 2008, 2009, 2020]:
    t = pd.Timestamp(f"{a}-01-01")
    print(f"{a}   {col(nx_y[t], 8)}   {col(nx_y.diff()[t], 10)}   "
          f"{col(g_y[t], 11)}   {col(g_c[t], 16)}")

rev = nx_y[pd.Timestamp("1995-01-01")] - nx_y[pd.Timestamp("1994-01-01")]
print(f"\nReversión 1994 -> 1995: {rev:.1f} puntos del producto en un año, "
      f"con el PIB real cayendo {abs(g_y[pd.Timestamp('1995-01-01')]):.1f}% "
      f"y el consumo {abs(g_c[pd.Timestamp('1995-01-01')]):.1f}%.")
assert rev > 5.0        # la parada súbita del Tequila, medida

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** La reversión de $5.8$ puntos del producto en un año, con el PIB cayendo
# $5.9\%$ y el consumo $6.0\%$, es la firma de una parada súbita: la cuenta externa mejora
# porque **tiene que** mejorar, y la absorción paga la cuenta. El contraste está en las otras
# dos recesiones: en **2009** el PIB cae $6.1\%$ —casi lo mismo que en 1995— pero la cuenta
# externa apenas se mueve ($+0.7$ puntos): fue una recesión **importada** por la demanda
# estadounidense, no un corte de financiamiento. En **2020** el PIB cae más ($8.6\%$) con una
# reversión de $2.0$ puntos, un tercio de la de 1995. La misma caída de producto puede venir
# con o sin ajuste externo forzado, y el tamaño de $\Delta(nx/y)$ es justamente lo que las
# distingue. No todas las recesiones mexicanas son paradas súbitas; la de 1995 sí lo fue, y
# es la norma de
# las crisis latinoamericanas que documentan Calvo, Izquierdo y Mejía (2004) y Mendoza
# (2010): reversiones abruptas de la cuenta corriente, depreciación real fuerte y colapso del
# producto, **agrupadas en el tiempo**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 8. Preguntas para pensar
#
# 1. **El hecho que depende del filtro.** En la sección 1, $\mathrm{corr}(nx/y,y)$ para
#    México es $-0.29$ con HP y $+0.01$ con Hamilton sobre la ventana canónica (y $-0.62$ con
#    HP si añades 1994). Un árbitro te pide elegir uno. ¿Cuál eliges y con qué argumento
#    *económico* —no estadístico—? Pista: ¿a qué frecuencia opera el mecanismo de
#    suavizamiento intertemporal que el modelo describe?
# 1 bis. **La atribución.** El cociente mexicano pasa de $0.959$ a $1.004$ entre la primera
#    fila y la última de la tabla de control de calidad. Descompón ese salto en la parte que
#    aporta el filtro y la que aporta la ventana, y explica por qué llamarle "calentamiento
#    del filtro" sería incorrecto para HP. ¿Qué tendría que pasar con la muestra para que el
#    $>1$ mexicano fuera un hecho robusto y no un artefacto de cuatro trimestres?
# 2. **Qué tasa es $r_t$.** La tasa real ex post en pesos resultó **procíclica** ($+0.44$) y
#    el VIX contracíclico ($-0.50$). Escribe la relación de paridad descubierta y explica
#    qué componente de $i^{MX}_t$ estaría contaminando la primera medición. ¿Qué serie
#    pedirías para hacerlo bien, y por qué el EMBI+ no basta tampoco?
# 3. **El precio de cada puerta.** La figura de la sección 6 dice que el canal de la
#    tendencia necesita $\rho_a\approx0.994$ y el de la prima innovaciones de ~265 pb
#    anuales para alcanzar el $0.97$ observado. ¿Qué **momento adicional** —no
#    $\sigma_c/\sigma_y$— separaría empíricamente las
#    dos hipótesis? Pista: mueven las horas por canales distintos; uno pasa por la
#    productividad y el otro por la cuña laboral, así que $\mathrm{corr}(h_t,\,y_t/h_t)$
#    debería discriminar.
# 4. **García-Cicco y el siglo.** GCPU muestran que la fricción financiera, no la muestra
#    larga, es lo que apaga el choque de tendencia. ¿Es eso evidencia contra Aguiar–Gopinath,
#    o evidencia de que los dos modelos están mal identificados con los momentos usuales?
# 5. **Leer el ciclo mexicano.** Con todo lo anterior: cuando el PIB de México cae, ¿qué
#    mirarías primero —la productividad total de los factores, el *spread* soberano, los
#    términos de intercambio o el ciclo estadounidense— y en qué orden? Justifica con al
#    menos dos números de esta lección.

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Notas para las preguntas
# 1. Argumento económico, no estadístico: el suavizamiento intertemporal opera en la banda
#    de negocios (6–32 trimestres), y Hamilton con $h=8$ deja pasar más frecuencia baja que
#    HP con $\lambda=1600$. Si lo que quieres contrastar es la respuesta de la cuenta
#    corriente a choques de esa banda, HP es el filtro alineado con el mecanismo; si te
#    interesa la respuesta a choques casi permanentes (la hipótesis de Aguiar–Gopinath),
#    Hamilton es el adecuado. Lo indefendible es elegir el que da el signo que uno quería.
# 1 bis. Del salto $0.959\to1.004$, prácticamente todo lo aporta la **ventana** (añadir 1994,
#    con el pico previo al Tequila) y sólo ~una centésima el filtro. Llamarle "calentamiento
#    del filtro" es incorrecto para HP porque HP **no** tiene calentamiento: devuelve ciclo
#    para todas las observaciones. Para que el $>1$ fuera robusto tendría que sobrevivir a
#    quitar 1994–95, a cambiar de filtro y a extender la muestra: hoy no sobrevive a ninguno.
# 2. Paridad descubierta: $i^{MX}_t\simeq i^{US}_t+E_t\Delta e_{t+1}+\text{prima}_t$. La tasa
#    real ex post en pesos mezcla la **inflación realizada** y la depreciación esperada, así
#    que en las crisis (cuando la inflación salta) su componente real cae y la serie sale
#    procíclica por construcción. Lo que se quiere es el **spread soberano en dólares**
#    (EMBI+ o un rendimiento soberano en dólares menos el del Tesoro). El EMBI+ tampoco basta
#    solo: mezcla riesgo de crédito con riesgo global (el VIX), y aquí interesa el precio que
#    enfrenta el prestatario privado, no sólo el soberano.
# 3. Sirve $\mathrm{corr}(h_t,\,y_t/h_t)$. El choque de tendencia entra por la productividad
#    y mueve horas y producto por hora en el mismo sentido; el choque de prima entra como
#    **cuña laboral** y los mueve en sentidos opuestos. Es un momento que las dos hipótesis
#    predicen con signo distinto, a diferencia de $\sigma_c/\sigma_y$, que ambas pueden
#    ajustar con el parámetro adecuado.
# 4. Es lo segundo, y conviene decirlo así: GCPU muestran que con los momentos usuales
#    ($\sigma$'s y correlaciones de primer orden) el choque de tendencia y la fricción
#    financiera son casi observacionalmente equivalentes. La muestra larga ayuda pero no
#    identifica; hace falta información adicional (los momentos de la pregunta 3, datos de
#    *spreads*, o restricciones de signo sobre las respuestas).
# 5. Orden razonable, y hay que justificarlo con cifras: (i) el **ciclo estadounidense**,
#    porque es el choque externo dominante para México y llega por comercio; (ii) el
#    **spread**/riesgo global — el VIX correlaciona $-0.50$ con el ciclo mexicano, y la
#    sección 6 muestra que la prima necesita innovaciones de ~265 pb para explicar el
#    cociente; (iii) los **términos de intercambio**; (iv) la **PTF**, que en el modelo
#    exigiría $\rho_a\approx0.994$ —casi una raíz unitaria— para dar el $0.97$ observado,
#    lo que la vuelve la explicación menos parsimoniosa de las cuatro.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Mini-entregable (una página)
# Replica la tabla de la sección 1 **cambiando una sola cosa a la vez** y reporta qué hechos
# sobreviven: (a) la ventana (1995–2019 canónica, contra 1994–2019 y contra 1995–2026 con
# COVID dentro); (b) el filtro (Hamilton contra HP); (c) el país (México contra Chile,
# Colombia y Turquía, cuidando que sus muestras arrancan más tarde); (d) el **orden de las
# operaciones** (recortar-y-filtrar, la convención del curso, contra filtrar-y-recortar, que
# usa datos del futuro para estimar la tendencia del pasado).
#
# Un quinto campo queda **fuera de tu alcance, y hay que decirlo en el entregable**: la
# *definición* del consumo. El panel congelado sólo trae consumo de **hogares** (P3, S1M);
# el consumo total (P3, S1) exigiría descargar otra serie, y el curso corre sin red. Según
# el mazo Slides07, moverlo baja el cociente mexicano de $0.96$ a $0.85$ —más que cualquiera
# de los cuatro campos que sí puedes mover—. Anótalo como límite declarado de tu ejercicio.
# Después responde: de los cuatro hechos de la sección 1, ¿cuáles
# son **robustos** y cuáles son artefactos de una elección? Tu conclusión tiene que hacerse
# cargo de las dos hipótesis rivales de la sección 6 — choque de tendencia contra prima
# contracíclica — y decir cuál de ellas queda mejor parada tras tu prueba de robustez.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 9. Explora con IA
# Prueba estas indicaciones con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué con $\beta(1+r)=1$ y $r$ exógeno la deuda externa tiene raíz unitaria?"
# - "¿Por qué una tasa de interés contracíclica funciona como una cuña laboral?"

# %% slideshow={"slide_type": "fragment"}
print(tutor(
    "En dos frases: por qué en una economía pequeña y abierta con tasa de interés exógena "
    "y constante la deuda externa tiene raíz unitaria, y cómo la prima creciente en la "
    "deuda de Schmitt-Grohé y Uribe (2003) resuelve el problema.",
    context=(f"En el modelo de dotación de esta lección, con psi=0 klein_solve devuelve "
             f"eu=(0,0) (Blanchard-Kahn falla) y con psi=1e-3 devuelve eu=(1,1) con valor "
             f"propio dominante {np.abs(np.linalg.eigvals(dotacion(1e-3).G)).max():.4f}. "
             f"En los datos, sigma_c/sigma_y de México = {dato_mx:.2f}."),
))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Medimos con la OCDE congelada, sobre la ventana canónica del curso
# (1995Q1–2019Q4, recorte **antes** de filtrar), los hechos que separan a México de EE. UU.
# —producto más volátil ($3.51$ contra $2.36$), $\sigma_c/\sigma_y=0.97$ contra $0.85$,
# balanza comercial contracíclica *según el filtro*— y vimos que el salto a $1.00$ que a veces
# se cita lo produce la **ventana** (añadir 1994), no el filtro, que apenas mueve una
# centésima. Demostramos con la identidad del gasto que una economía **cerrada** con esas
# participaciones tiene prohibido generar $\sigma_c>\sigma_y$; abrirla levanta la prohibición,
# pero aporta apenas tres centésimas del cociente. Diagnosticamos el
# **problema del cierre** —con $\psi=0$, `klein_solve` devuelve `eu=(0,0)`: no existe
# solución estable— y lo resolvimos con la prima creciente de Schmitt-Grohé y Uribe (2003).
# Con el modelo completo resuelto por Klein vimos que el choque de **prima** hunde el consumo
# tres veces más que el producto y contrae las horas por la cuña laboral de Neumeyer–Perri,
# mientras el choque de productividad transitorio hace lo contrario. Y pusimos las dos
# hipótesis rivales en la misma escala: la tendencia necesita $\rho_a\approx0.994$, la prima
# necesita innovaciones de ~265 pb — con el contrapunto de García-Cicco, Pancrazi y Uribe
# (2010), para quienes con un siglo de datos mexicanos el modelo de tendencia **subpredice**
# el cociente ($0.60$ contra $1.50$). El ciclo mexicano no es el estadounidense con más ruido;
# es un objeto con su propia física.
#
# **Referencias.** Schmitt-Grohé y Uribe (2003), *Closing small open economy models*, JIE 61.
# · Neumeyer y Perri (2005), *Business cycles in emerging economies: the role of interest
# rates*, JME 52. · Uribe y Yue (2006), *Country spreads and emerging countries: who drives
# whom?*, JIE 69. · Aguiar y Gopinath (2007), *Emerging market business cycles: the cycle is
# the trend*, JPE 115. · García-Cicco, Pancrazi y Uribe (2010), *Real business cycles in
# emerging countries?*, AER 100. · Mendoza (1991), *Real business cycles in a small open
# economy*, AER 81. · Calvo, Izquierdo y Mejía (2004), NBER WP 10520. · Mendoza (2010),
# *Sudden stops, financial crises, and leverage*, AER 100. · Hamilton (2018), *Why you should
# never use the Hodrick-Prescott filter*, REStat 100. · Klein (2000), JEDC 24.
