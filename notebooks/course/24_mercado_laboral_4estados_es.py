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
# # Módulo (electivo) — El mercado laboral tiene cuatro estados, no dos: informalidad y margen de participación
#
# **Curso complementario · puremacro**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Construir y leer la **matriz de transición de cuatro estados** F/I/U/N
#    (formal / informal / desempleado / inactivo) de México con
#    `puremacro.labor_flows_enoe`, y calcular las **tasas de encuentro y de
#    separación por estado**.
# 2. **Confrontar** el modelo de búsqueda-emparejamiento de **dos estados** (E/U)
#    —el caballo de batalla de la teoría de búsqueda, calibrado a EE.UU.— con un hecho que
#    **no puede ver**: en una economía dual el desempleo casi no se mueve porque
#    el ajuste cíclico corre por la **informalidad** y por el **margen de
#    participación** (Elsby–Hobijn–Şahin 2015).
# 3. Entender por qué la **tasa de desempleo**, cuando más de la mitad del empleo
#    es informal, es un termómetro de escala demasiado corta: señala el ciclo
#    (es netamente contracíclica) pero no mide el volumen del ajuste, que ocurre
#    en otros márgenes (Fernández–Meza 2015). Y aprender a distinguir esas dos
#    propiedades —**amplitud** y **ciclicidad**— que suelen confundirse.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), leyendo los parquets **ya procesados** del bundle —
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
# ## 1. Dos estados no alcanzan
#
# El modelo estándar de búsqueda (Diamond–Mortensen–Pissarides, DMP) parte
# de un **agente representativo** que solo puede estar en
# **dos estados**: empleado ($E$) o desempleado ($U$). Toda la dinámica del
# mercado laboral se resume en dos tasas —hallazgo de empleo $f$ y separación
# $s$— y la tasa de desempleo obedece
# $$\dot u = s\,(1-u) - f\,u, \qquad u^\* = \frac{s}{s+f}.$$
# Ese marco describe razonablemente a EE.UU. Pero proyectarlo sobre México
# ignora dos márgenes que allí **son el centro de la acción**:
#
# - **Informalidad (I).** Estar empleado no es un estado único: hay empleo
#   **formal** (con acceso a seguridad social vía el empleador) e **informal**
#   (sin él). En México el empleo informal es **más de la mitad** del total.
# - **Participación (N).** La frontera entre buscar trabajo y salir de la fuerza
#   laboral es porosa. Elsby, Hobijn y Şahin (2015) muestran que ignorar este
#   **margen de participación** sesga cualquier lectura de las fluctuaciones.
#
# `puremacro.labor_flows_enoe` codifica los **cuatro** estados
# `STATES = ('F', 'I', 'U', 'N')` a partir del panel rotatorio de la ENOE (INEGI).
#
# > **Aviso de notación.** En el bloque de tres estados E/U/I de EE.UU. (la CPS), la
# > letra $I$ significa **inactividad**. Aquí, en el bloque mexicano F/I/U/N, $I$ es
# > empleo **informal** y la inactividad pasa a $N$. Es la misma convención del mazo
# > del curso; no las mezcles.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Los datos del bundle
# El microdato de la ENOE (cientos de MB por trimestre) **no** viaja en el
# bundle, así que `transitions_from_enoe` / `load_enoe_quarter` —que leen los
# `.dta` de INEGI— quedan fuera de esta lección. Lo que sí traemos, **ya
# procesado** por esa tubería, son dos parquets: las **transiciones trimestrales
# observadas** (matriz $4\times4$ por mes de referencia) y los **acervos
# mensuales** F/I/U/N. Trabajamos con ellos.

# %% slideshow={"slide_type": "fragment"}
# Dos requisitos que conviene comprobar ANTES de nada, porque fallan de forma poco
# informativa: (i) el módulo labor_flows_enoe, que no está en todas las ruedas publicadas
# de puremacro; (ii) un motor de parquet (pyarrow o fastparquet), que pandas necesita para
# read_parquet y que no viene con las dependencias base del paquete.
try:
    from puremacro.labor_flows_enoe import STATES   # ('F', 'I', 'U', 'N')
except ModuleNotFoundError:                          # rueda de PyPI sin este módulo
    STATES = ("F", "I", "U", "N")
    print("AVISO: puremacro.labor_flows_enoe no está en tu instalación de puremacro.\n"
          "       La lección sigue: los cuatro estados se declaran aquí y los datos ya\n"
          "       vienen procesados en el bundle. Para tener el módulo, instala puremacro\n"
          "       desde el árbol del repositorio (`pip install -e .`) en vez de PyPI.")
try:
    import pyarrow  # noqa: F401  -- motor de parquet
except ModuleNotFoundError:
    try:
        import fastparquet  # noqa: F401
    except ModuleNotFoundError as _e:
        raise ModuleNotFoundError(
            "Esta lección lee dos archivos .parquet y pandas necesita un motor para ello. "
            "Instala uno con `pip install pyarrow` (o `pip install fastparquet`) y vuelve "
            "a ejecutar la celda."
        ) from _e

tr = pd.read_parquet(DATA / "enoe_transitions_quarterly_observed.parquet")  # p_ab por mes
st = pd.read_parquet(DATA / "enoe_stocks_monthly.parquet")                  # F/I/U/N (personas)
print(f"transiciones: {tr.shape[0]} meses de referencia, {tr.index.min().date()} .. {tr.index.max().date()}")
print(f"acervos:      {st.shape[0]} meses, estados = {list(st.columns)}")
print(f"estados del módulo: {STATES}")

# Matriz de transición trimestral MEDIA (cada p_ab es Prob(j en t | i en t-3)).
# La suspensión de la ENOE (abril-junio de 2020) deja fuera 6 meses de referencia: 3 no
# están en el índice (feb.-abr. 2020) y 3 vienen enmascarados como NaN (ene., may. y jun.
# de 2020), que .mean() de pandas ignora. La media se calcula, pues, sobre 229 meses.
P = np.array([[tr[f"p_{a}{b}"].mean() for b in STATES] for a in STATES])
print("\nmatriz de transición media 4x4 (filas suman a 1):", P.sum(axis=1).round(3))
assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Tasas de encuentro y de separación, **por estado**
# Con cuatro estados, "encontrar empleo" y "separarse" dejan de ser dos números
# y se vuelven **flujos por origen**. Definimos (todo trimestral):
# - **Encuentro** hacia el empleo ($F\cup I$): desde $U$, $f_U=p_{UF}+p_{UI}$;
#   desde $N$, $f_N=p_{NF}+p_{NI}$.
# - **Separación** hacia el no-empleo ($U\cup N$): desde $F$, $s_F=p_{FU}+p_{FN}$;
#   desde $I$, $s_I=p_{IU}+p_{IN}$.
# - **Rotación dual**: formalización $p_{IF}$ e informalización $p_{FI}$ —el
#   churn $F\leftrightarrow I$ que un modelo de dos estados colapsa en "empleo".

# %% slideshow={"slide_type": "fragment"}
f_U = (tr["p_UF"] + tr["p_UI"]).mean()   # U -> empleo
f_N = (tr["p_NF"] + tr["p_NI"]).mean()   # N -> empleo
s_F = (tr["p_FU"] + tr["p_FN"]).mean()   # F -> no-empleo
s_I = (tr["p_IU"] + tr["p_IN"]).mean()   # I -> no-empleo
p_IF = tr["p_IF"].mean()                 # formalización
p_FI = tr["p_FI"].mean()                 # informalización

print("ENCUENTRO (hacia empleo, trimestral)")
print(f"  desde U:  f_U = p_UF+p_UI = {f_U:.3f}")
print(f"  desde N:  f_N = p_NF+p_NI = {f_N:.3f}")
print("SEPARACIÓN (hacia no-empleo, trimestral)")
print(f"  desde F:  s_F = p_FU+p_FN = {s_F:.3f}")
print(f"  desde I:  s_I = p_IU+p_IN = {s_I:.3f}   <- el empleo informal es mucho más frágil")
print("ROTACIÓN DUAL")
print(f"  formalización   I->F = {p_IF:.3f}")
print(f"  informalización F->I = {p_FI:.3f}   <- salir hacia informalidad es más probable que formalizarse")
assert s_I > s_F        # el empleo informal se separa mucho más que el formal
assert p_FI > p_IF      # hazards, no flujo neto: la tasa de formalización (I->F) es baja — la informalidad no es un trampolín fácil

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 1 — la matriz $4\times4$ y sus tasas por estado
# Izquierda: la matriz de transición media (más oscuro = más probable). La
# **diagonal** domina —cada estado es persistente ($p_{FF}=0.78$, $p_{II}=0.71$,
# $p_{NN}=0.84$), salvo $U$, que no lo es ($p_{UU}=0.19$)—, y en la fila $I$
# compara las dos salidas que importan: $p_{IN}=0.17$ contra $p_{IF}=0.10$. La
# informalidad desemboca en inactividad más que en formalidad.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))

ax = axes[0]
im = ax.imshow(P, cmap="Greys", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(4)); ax.set_xticklabels(STATES)
ax.set_yticks(range(4)); ax.set_yticklabels(STATES)
ax.set_xlabel("estado en $t$ (destino)"); ax.set_ylabel("estado en $t-3$ (origen)")
ax.set_title("Matriz de transición trimestral ENOE (media)")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                color="white" if P[i, j] > 0.5 else "0.10", fontsize=9)

ax = axes[1]
labels = ["$f_U$\nU→emp", "$f_N$\nN→emp", "$s_F$\nF→no-emp", "$s_I$\nI→no-emp",
          "$p_{IF}$\nformaliz.", "$p_{FI}$\ninformaliz."]
vals = [f_U, f_N, s_F, s_I, p_IF, p_FI]
grays = ["0.15", "0.35", "0.55", "0.55", "0.72", "0.72"]
ax.bar(range(len(vals)), vals, color=grays)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("probabilidad trimestral")
ax.set_title("Tasas de encuentro / separación por estado")
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** El desempleo $U$ es un estado **pequeño y de paso**: solo el 19%
# de quienes están en $U$ siguen ahí un trimestre después ($p_{UU}=0.19$), y de
# los que salen hacia el empleo ($f_U\approx0.52$) el destino **no** es
# indistinto: entran a la informalidad **el doble de veces** que a la formalidad
# ($p_{UI}=0.35$ contra $p_{UF}=0.17$). El grueso del ajuste ocurre además **sin
# pasar por el desempleo**: la separación del empleo informal ($s_I=0.19$) casi
# triplica la del formal ($s_F=0.06$), y buena parte de esos flujos van directo a
# la **inactividad** ($N$), no a la fila de desempleo. Un modelo E/U no tiene
# dónde poner ninguno de estos movimientos.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. La confrontación: el desempleo que casi no se mueve
#
# El DMP de dos estados, calibrado a EE.UU., hace del **desempleo** la
# variable de estado que absorbe el ciclo: cae en las expansiones, salta en las
# recesiones. Si aplicamos esa lente a México deberíamos ver un desempleo que
# absorbe el ciclo con **amplitud** comparable. **No lo vemos.** Cuidado con la
# formulación, que es donde se cuela el error fácil: lo que falla no es que $u$
# no *siga* al ciclo —lo sigue, y muy de cerca, como mediremos—, sino que su
# recorrido es demasiado corto para dar cabida al ajuste. Comparemos la tasa de
# desempleo de EE.UU. (el mundo de dos estados) con la de México, calculada desde
# los acervos $u = U/(F+I+U)$, en la **misma ventana** 2005–2024.

# %% slideshow={"slide_type": "fragment"}
from puremacro.data import hp_filter   # (cycle, trend) = hp_filter(y, lamb=1600)

# México: márgenes desde los acervos (trimestral, promediando los meses).
# OJO con el hueco: la ENOE se suspendió en abril-junio de 2020, así que 2020Q2 no existe
# en los acervos y .dropna() lo elimina. La serie trimestral queda con 78 observaciones y
# el filtro HP la trata como si fuera contigua (2020Q1 pegado a 2020Q3).
q = st.resample("QS").mean().dropna()
LF  = q["F"] + q["I"] + q["U"]            # fuerza laboral
WAP = LF + q["N"]                         # población en edad de trabajar
u_mx   = 100 * q["U"] / LF               # tasa de desempleo (% de la PEA)
infml  = 100 * q["I"] / (q["F"] + q["I"])  # informalidad (% del empleo)
inact  = 100 * q["N"] / WAP               # inactividad (% de la WAP) = margen de participación

# EE.UU.: tasa de desempleo congelada del bundle (FRED UNRATE), misma ventana.
us = pd.read_csv(DATA / "UNRATE.csv"); us.columns = ["date", "u"]
us["date"] = pd.to_datetime(us["date"]); us = us.set_index("date")["u"].resample("QS").mean()
lo, hi = u_mx.index.min(), u_mx.index.max()
u_us = us[(us.index >= lo) & (us.index <= hi)]

def hp_cycle(x):
    """Componente cíclico HP (lambda=1600) como array."""
    c, _ = hp_filter(np.asarray(x, dtype=float)); return np.asarray(c, dtype=float)

def cyc_sd(x):
    """Desviación estándar del ciclo HP (lambda=1600)."""
    return float(np.std(hp_cycle(x)))

sd = {"u EE.UU.": cyc_sd(u_us), "u México": cyc_sd(u_mx),
      "informalidad": cyc_sd(infml), "inactividad": cyc_sd(inact)}
print("nivel medio:  u_MX = %.1f%%   informalidad = %.0f%% del empleo   participación = %.0f%% de la WAP"
      % (u_mx.mean(), infml.mean(), 100 - inact.mean()))
print("volatilidad cíclica (sd del ciclo HP), CADA UNA SOBRE SU PROPIA BASE:")
for k, base in [("u EE.UU.", "% de la PEA "), ("u México", "% de la PEA "),
                ("informalidad", "% del empleo"), ("inactividad", "% de la WAP ")]:
    print(f"  {k:14s} {base}  {sd[k]:.2f}")

# --- Base COMÚN. Los cuatro números de arriba tienen DENOMINADORES DISTINTOS (PEA, empleo,
# WAP), así que sus "puntos %" no son comparables entre sí tal cual. La comparación limpia
# expresa los cuatro estados mexicanos como % de la MISMA población: la WAP.
sd_wap = {s: cyc_sd(100 * q[s] / WAP) for s in STATES}
print("\nbase COMÚN (los cuatro estados como % de la WAP), sd del ciclo HP:")
for s in STATES:
    print(f"  {s}: {sd_wap[s]:.2f}")
print(f"  -> I se mueve {sd_wap['I']/sd_wap['U']:.1f}x más que U, y N {sd_wap['N']/sd_wap['U']:.1f}x más:")
print( "     el orden se mantiene en base común, no era un artefacto de los denominadores.")
rho_FI = float(np.corrcoef(hp_cycle(100 * q["F"] / WAP), hp_cycle(100 * q["I"] / WAP))[0, 1])
print(f"  corr(ciclo F/WAP, ciclo I/WAP) = {rho_FI:.2f}  <- F e I son imágenes espejo:")
print( "     el volumen del movimiento está en la recomposición F<->I, no en la entrada a U")
print( "     (si ese movimiento es o no ciclo agregado, lo medimos en la celda siguiente).")

# --- Cociente EE.UU./México: dos lecturas, porque las muestras NO son idénticas.
# 2020Q2 (el trimestre de u=13.0% en EE.UU.) falta en México: la ENOE estaba suspendida.
r_todo = sd["u EE.UU."] / sd["u México"]
sin20  = lambda x: x[x.index.year != 2020]
r_sin20 = cyc_sd(sin20(u_us)) / cyc_sd(sin20(u_mx))
print(f"\ncociente sd(u EE.UU.)/sd(u México) = {r_todo:.1f}x  (muestra completa; EE.UU. incluye")
print( "  el 2020Q2 de 13.0% que en México no existe porque la ENOE estuvo suspendida)")
print(f"  excluyendo 2020 en AMBOS países:    {r_sin20:.1f}x  <- el contraste sobrevive, más modesto")
assert sd["u EE.UU."] > sd["u México"]     # el 2-estados de EUA vive de un u volátil
assert r_sin20 > 1.0                       # y no es un artefacto del COVID asimétrico
assert min(sd_wap, key=sd_wap.get) == "U"  # en base común, U es el margen MENOS volátil
assert rho_FI < -0.5                       # el ajuste es la recomposición F<->I
assert infml.mean() > 50                   # economía dual: >50% del empleo es informal

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Amplitud no es ciclicidad: correlación con el PIB
# Lo anterior mide **cuánto se mueve** cada margen, no **con qué** se mueve. Son
# preguntas distintas y aquí dan respuestas distintas, así que conviene medir las
# dos. Correlacionamos el ciclo HP de cada margen con el ciclo HP del **PIB
# mexicano** (OCDE QNA, volúmenes, CSV congelado del bundle; para México la OCDE
# solo publica base fija `Q`, por eso se lee el CSV y no `fetch_qna_expenditure`).

# %% slideshow={"slide_type": "fragment"}
_qna = pd.read_csv(DATA / "oecd_qna_apertura.csv")
_gdp = (_qna[(_qna["code"] == "MEX") & (_qna["variable"] == "gdp_vol")]
        .assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["value"].sort_index())
com = _gdp.index.intersection(q.index)          # ventana común: 78 trimestres (sin 2020Q2)
c_gdp = hp_cycle(100 * np.log(_gdp.loc[com]))   # ciclo del PIB, en % (log x 100)

rho = {"u México (% PEA)": np.corrcoef(c_gdp, hp_cycle(u_mx.loc[com]))[0, 1]}
for s in STATES:
    rho[f"{s}/WAP"] = float(np.corrcoef(c_gdp, hp_cycle(100 * q[s].loc[com] / WAP.loc[com]))[0, 1])
rho["informalidad (% empleo)"] = float(np.corrcoef(c_gdp, hp_cycle(infml.loc[com]))[0, 1])

print(f"ciclo del PIB de México: sd = {np.std(c_gdp):.2f}%  ({len(com)} trimestres, 2005Q2-2024Q4)")
print("correlación contemporánea con el ciclo del PIB:")
for k, v in rho.items():
    print(f"  {k:24s} {v:+.2f}")
print("\nLECTURA, y corrige la intuición fácil:")
print("  * la tasa de desempleo mexicana NO es acíclica: es el margen MÁS contracíclico")
print("    de los cuatro. Registra el ciclo con fidelidad; lo que no hace es ABSORBERLO,")
print("    porque su amplitud es minúscula (0.22 pp de la WAP).")
print("  * la inactividad N es claramente contracíclica: en las recesiones la gente sale")
print("    de la fuerza laboral. Ése es el margen de participación de Elsby-Hobijn-Sahin.")
print("  * el par F<->I es el de mayor amplitud (1.13 y 1.04 pp de la WAP: dos caras de un")
print("    mismo margen, corr = -0.74) pero el MENOS alineado con el PIB:")
print("    buena parte de su movimiento no es ciclo agregado (composición, reformas,")
print("    formalización de largo plazo). Amplitud grande no implica ciclicidad grande.")
assert rho["u México (% PEA)"] < -0.5           # u es contracíclica y con fuerza
assert abs(rho["informalidad (% empleo)"]) < abs(rho["u México (% PEA)"])
assert rho["N/WAP"] < 0                         # la participación cae en las recesiones

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Ficha de medición (regla del curso: ningún segundo momento sin sus seis campos)
# | campo | valor |
# |---|---|
# | **fuente / serie** | México: acervos mensuales F/I/U/N del panel rotatorio de la ENOE (INEGI), ya procesados por `puremacro.labor_flows_enoe` (`enoe_stocks_monthly.parquet`). EE.UU.: FRED `UNRATE`, CSV congelado del bundle. PIB de México: OCDE QNA, volúmenes (`oecd_qna_apertura.csv`, `gdp_vol`). |
# | **muestra** | 2005Q2–2024Q4 en ambos países (78 trimestres en México: **2020Q2 no existe**, la ENOE estuvo suspendida abril–junio de 2020). |
# | **filtro** | Hodrick–Prescott, $\lambda=1600$. Las tasas y participaciones se filtran **en niveles (%)**; el PIB, en **log × 100**. |
# | **base de precios** | Series laborales: no aplica (son proporciones de personas). PIB: volúmenes encadenados, base fija `Q` —la única que la OCDE publica para México, por eso se lee el CSV congelado y no `fetch_qna_expenditure`, que filtra `PRICE_BASE=='L'` y devolvería vacío—. |
# | **orden recortar–filtrar** | Primero recortar, después filtrar, en todas las series: el `UNRATE` de EE.UU. y el PIB se recortan a la ventana común (y el PIB, además, al mismo índice trimestral que la ENOE, sin 2020Q2) **antes** de pasar por el HP. |
# | **edición / vintage** | Parquets del bundle del curso (ENOE hasta 2024Q4), `UNRATE.csv` y `oecd_qna_apertura.csv` congelados; sin red. |
#
# **Y una advertencia de nivel.** Estas cifras salen del panel rotatorio con sus propias
# definiciones y pesos, así que **no coinciden con los agregados publicados por INEGI**:
# la tasa de informalidad laboral oficial (TIL1) ronda 55–58% y la tasa de participación
# de la ENOE ~59.5%, frente al 62% y 56% de aquí. Para el argumento de la lección importan
# el **orden de magnitud** y la **volatilidad relativa**, no el decimal del nivel.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 2 — lo que el modelo de dos estados ve, y lo que se pierde
# Izquierda: la línea que el DMP de dos estados toma como *todo* el mercado
# laboral —el desempleo— en EE.UU. y en México. Derecha: la volatilidad cíclica de
# los cuatro estados mexicanos **en base común** (todos como % de la misma
# población, la WAP), que es la única forma de compararlos entre sí. En México el
# desempleo es, con diferencia, el margen **menos** volátil.

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

ax = axes[0]
# Reindexar México al calendario trimestral completo deja 2020Q2 como NaN, de modo que la
# línea se ROMPE en el hueco de la ENOE en vez de fingir una interpolación que no existe.
u_mx_plot = u_mx.reindex(pd.date_range(u_mx.index.min(), u_mx.index.max(), freq="QS"))
ax.plot(u_us.index, u_us.values, color="0.10", lw=1.6, label="EE.UU. (2 estados: E/U)")
ax.plot(u_mx_plot.index, u_mx_plot.values, color="0.55", lw=1.6, ls=(0, (4, 2)),
        label="México (F/I/U/N)")
ax.axvline(pd.Timestamp("2020-04-01"), color="0.75", lw=6, alpha=0.45, zorder=0)
ax.annotate("2020Q2: ENOE\nsuspendida", xy=(pd.Timestamp("2020-04-01"), 7.2),
            xytext=(pd.Timestamp("2013-01-01"), 11.0), fontsize=7, color="0.35",
            ha="center", arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
ax.set_ylabel("tasa de desempleo, %"); ax.set_xlabel("trimestre")
ax.set_title("El desempleo: volátil en EE.UU., plano en México")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
order = ["F", "I", "U", "N"]
names = ["F\nformal", "I\ninformal", "U\ndesempleo", "N\ninactivo"]
ax.bar(range(len(order)), [sd_wap[k] for k in order],
       color=["0.20", "0.45", "0.80", "0.65"], edgecolor="0.25", lw=0.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("sd del ciclo HP (puntos % de la WAP)")
ax.set_title("¿Dónde ocurre el ajuste cíclico? (base común)")
ax.set_ylim(0, max(sd_wap.values()) * 1.18)
for i, k in enumerate(order):
    ax.text(i, sd_wap[k] + 0.02, f"{sd_wap[k]:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **El puñetazo.** En EE.UU. el desempleo trimestral salta de 3.6% (2019Q4) a
# **13.0%** (2020Q2) y de 4.8% (2007Q4) a **9.9%** (2009Q4): el modelo E/U
# calibrado a esos datos *necesita* un $u$ muy cíclico. En México, en las mismas
# fechas, la tasa de desempleo se mueve en una banda estrecha alrededor de su
# media de 4.0%: su **máximo de toda la muestra** es 6.1% (2009Q3) y en el
# trimestre COVID medido (2020Q3) llega solo a 4.9%. No es que el mercado laboral
# mexicano no fluctúe: fluctúa por **otros márgenes**. En base común, el de mayor
# amplitud es la recomposición formal↔informal (1.13 y 1.04 pp de la WAP: son dos
# caras del mismo margen, con correlación $-0.74$ entre sus ciclos), casi cinco
# veces el desempleo; la participación (entrar/salir de $N$) le sigue, con más del
# triple. Ojo con el
# salto lógico que ya desmontamos arriba: *amplitud* no es *ciclicidad*. La que
# marca el ciclo con fidelidad es $u$ ($-0.76$ con el PIB); la que mueve gente es
# la informalidad. Y la que hace las dos cosas a la vez —amplitud grande y signo
# claro ($-0.51$)— es la **inactividad**: es exactamente el **margen de
# participación** de Elsby–Hobijn–Şahin (2015),
# amplificado por la dualidad formal/informal (Fernández–Meza 2015). Un modelo de
# dos estados, estructuralmente, **no tiene la fila ni la columna** para registrarlo.
#
# *Dos cautelas honestas.* (i) El trimestre más violento del episodio COVID
# (2020Q2) está en la serie de EE.UU. y **falta** en la de México, así que el
# cociente de volatilidades de la muestra completa exagera el contraste; por eso
# la celda de arriba lo recalcula excluyendo 2020 en ambos países y el contraste
# sobrevive, aunque más modesto. (ii) La comparación entre márgenes solo tiene
# sentido con **denominador común**: en puntos porcentuales de bases distintas
# (PEA, empleo, WAP) uno puede fabricar casi cualquier ordenamiento.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 3 — el desempleo es una rendija
# Composición de la población en edad de trabajar por estado. La banda de
# **desempleo (U)** es una rendija delgada y quieta; el volumen y el movimiento
# están en $F$, $I$ y $N$. Ahí es donde vive el ciclo laboral mexicano.

# %% slideshow={"slide_type": "slide"}
shares = q[list(STATES)].div(WAP, axis=0) * 100.0   # % de la WAP, por estado
fig, ax = plt.subplots(figsize=(8.0, 3.9))
ax.stackplot(shares.index,
             shares["F"], shares["I"], shares["U"], shares["N"],
             labels=["F formal", "I informal", "U desempleo", "N inactivo"],
             colors=["0.20", "0.45", "0.80", "0.65"], edgecolor="white", lw=0.3)
ax.set_ylim(0, 100); ax.set_ylabel("% de la población en edad de trabajar")
ax.set_xlabel("trimestre")
ax.set_title("México: los cuatro estados de la WAP (el desempleo es la rendija clara)")
ax.legend(loc="lower center", ncol=4, fontsize=8, framealpha=0.9)
plt.show()

u_band = shares["U"].mean()
print(f"la banda U promedia solo {u_band:.1f}% de la WAP;  I promedia {shares['I'].mean():.0f}%  y  N promedia {shares['N'].mean():.0f}%")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Preguntas para pensar
# 1. **¿Qué mide la tasa de desempleo con ~62% de informalidad?** Si perder el
#    empleo formal casi nunca lleva a "desempleado que busca" sino a "informal" o
#    a "inactivo", ¿un desempleo de 4% significa pleno empleo, o significa que el
#    desempleo abierto es un lujo que pocos pueden pagarse? ¿Qué indicador
#    pondrías en su lugar para leer el ciclo laboral mexicano?
# 2. **El agente representativo.** El DMP de dos estados tiene un solo tipo de trabajador y
#    un solo tipo de empleo. ¿Qué supuesto exacto se rompe al añadir el estado
#    $I$: la matching function, la ecuación de Bellman del empleo, o la regla de
#    reparto salarial (Nash)? ¿Bastaría con "dos sectores DMP en paralelo" o hace
#    falta permitir flujos $F\leftrightarrow I$?
# 3. **Margen de participación.** Elsby–Hobijn–Şahin (2015) descomponen las
#    fluctuaciones incluyendo entradas/salidas de la fuerza laboral. En la Figura
#    2, la inactividad es más volátil que el desempleo (más del triple, en base
#    común). ¿Por qué un modelo que fija la participación —uno en el que toda la
#    población relevante es la fuerza laboral, $E+U$, sin estado $N$— atribuiría
#    erróneamente ese movimiento a la tasa de hallazgo $f$ o a la de separación $s$?

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Notas para las preguntas
# 1. Con esa matriz de transición, perder el empleo formal desemboca sobre todo en $I$ o en
#    $N$, no en $U$: el desempleo abierto exige poder financiar la búsqueda sin ingreso, y en
#    México pocos pueden. Por eso un $4\%$ no es pleno empleo, es un indicador que casi no
#    registra el margen por donde ocurre el ajuste. Indicadores mejores: la **tasa de
#    informalidad**, la **razón empleo formal / WAP**, la **tasa de separación por origen**
#    ($s_F$, $s_I$) o una tasa de subocupación que incluya a los informales involuntarios.
#    Y cuidado con el diagnóstico fácil: la tasa de desempleo mexicana **no** es acíclica
#    —correlaciona $-0.76$ con el ciclo del PIB, más que ningún otro margen—, es
#    **contracíclica y diminuta**. Señala el ciclo pero no lo absorbe: 0.22 pp de la WAP
#    de amplitud frente a 1.04 de la informalidad y 0.71 de la inactividad. Por eso hace
#    falta acompañarla con un indicador de **volumen** del ajuste, no sustituirla por uno.
# 2. Se rompe la **matching function**, y con ella la definición misma del estado: con $I$ el
#    trabajador tiene tres opciones de destino, no dos, y el emparejamiento deja de ser un
#    proceso único $u\times v$. Las Bellman y el reparto de Nash se pueden reescribir sector
#    por sector sin drama. Pero "dos DMP en paralelo" **no** basta: el hecho central de los
#    datos es la rotación $F\leftrightarrow I$ (los flujos $p_{IF}$ y $p_{FI}$), es decir el
#    tránsito directo entre sectores sin pasar por el desempleo, que dos mercados
#    independientes prohíben por construcción.
# 3. Si el modelo fija la participación, la identidad $u=s/(s+f)$ tiene que absorber todo el
#    movimiento con sólo dos parámetros. Una salida masiva hacia $N$ (que reduce el
#    denominador de la tasa de desempleo sin que nadie encuentre empleo) se lee entonces como
#    si $f$ hubiera **subido**, y una entrada desde $N$ hacia $U$ se lee como si $s$ hubiera
#    subido. La atribución queda invertida: el modelo le echa la culpa a la tecnología de
#    emparejamiento de lo que en realidad es un margen de participación. Ése es exactamente el
#    punto de Elsby–Hobijn–Şahin, y por eso la descomposición correcta necesita los cuatro
#    estados.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 4. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué en una economía con alta informalidad la tasa de desempleo subestima
#   el deterioro del mercado laboral en una recesión? Responde en una frase."
# - "¿Qué le falta a un modelo de búsqueda de dos estados (E/U) para representar
#   el margen de participación de Elsby–Hobijn–Şahin?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase: ¿por qué la tasa de desempleo mide mal el ciclo "
            "laboral en una economía con más de la mitad del empleo informal?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Construimos la matriz de transición de **cuatro estados** F/I/U/N
# de México (`puremacro.labor_flows_enoe`) y sus tasas de encuentro/separación
# por origen; el empleo informal se separa mucho más que el formal y desemboca en
# inactividad, no en formalidad. Al **confrontar** el DMP de **dos estados** (E/U)
# con los datos, y midiendo los cuatro márgenes sobre una **base común**, el
# desempleo mexicano resulta ser el margen de **menor amplitud** con diferencia
# (0.22 pp de la WAP, contra 1.04 de la informalidad y 0.71 de la inactividad):
# el volumen del ajuste corre por la **informalidad** y el **margen de
# participación** (Elsby–Hobijn–Şahin 2015; Fernández–Meza 2015), que un modelo
# E/U no puede representar. Matiz importante y medido aquí: $u$ **sí** es
# fuertemente contracíclica ($-0.76$ con el ciclo del PIB); lo que no es, es
# grande. Es un termómetro fiable con una escala demasiado corta: registra la
# fiebre, pero no dice cuántos enfermos hay.
#
# **Referencias.** Elsby, Hobijn y Şahin (2015), *On the importance of the
# participation margin for labor market fluctuations*, JME 72. · Shimer (2012),
# *Reassessing the ins and outs of unemployment*, RED 15. · Fernández y Meza
# (2015), *Informal employment and business cycles in emerging economies: The
# case of Mexico*, RED 18.
