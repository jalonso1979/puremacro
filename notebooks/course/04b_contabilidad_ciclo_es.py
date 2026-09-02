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
# # Módulo 4b — Contabilidad del ciclo económico: cuatro cuñas y un veredicto
#
# **Curso complementario · puremacro · mazo Slides04 — ciclos reales y contabilidad del ciclo (semanas 7–8)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Situar cada una de las **cuatro cuñas** de Chari, Kehoe y McGrattan (2007) —eficiencia
#    $A_t$, trabajo $1-\tau_{\ell,t}$, inversión $1+\tau_{x,t}$ y gasto $g_t$— en la ecuación
#    de equilibrio del modelo neoclásico que rompe, y **medirlas** como residuos sobre datos
#    trimestrales de la OCDE.
# 2. Resolver el **prototipo log-linealizado** con `puremacro.dsge.klein_solve` y correr el
#    **experimento contrafactual** de CKM: alimentar el modelo con **una cuña a la vez** y
#    medir qué fracción del **desvío del producto respecto a su tendencia** reproduce cada una.
# 3. Leer el resultado con honestidad: por qué las cuñas **descartan familias** de modelos
#    pero no identifican una fricción, y **auditar la medición** de *este* panel —dos de sus
#    cuatro cuñas están contaminadas: la de inversión por el atajo de la Euler *ex post*, y la
#    de trabajo porque en tres de los seis países el insumo de trabajo **no son horas**, algo
#    que la identidad $\log(1-\tau_{\ell,t})=\log(c_t/y_t)+3\log(h_t/\text{EMP}_t)+\text{cte}$
#    convierte en un problema exacto y no en una sospecha.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), leyendo datos **congelados** del bundle del curso —
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
# ## 1. La idea: toda economía con fricciones tiene un gemelo neoclásico con cuñas
#
# Un investigador escribe un modelo con fricción financiera; otro, uno con salarios rígidos;
# otro, uno con poder sindical. ¿Cómo decidimos cuál merece el esfuerzo? Chari, Kehoe y
# McGrattan (2007) proponen un rodeo: demuestran **resultados de equivalencia**. Dada una
# economía detallada de una lista corta y explícita, construyen una economía **prototipo**
# —el neoclásico de siempre, aumentado con cuatro cuñas que varían en el tiempo— cuyas
# asignaciones de equilibrio **coinciden exactamente** con las de la economía detallada.
#
# Las cuatro equivalencias efectivamente demostradas:
#
# | cuña | dónde entra | fricción detallada equivalente |
# |---|---|---|
# | **eficiencia** $A_t$ | función de producción | financiamiento de insumos intermedios: el costo del crédito para pagar insumos aparece como pérdida de eficiencia |
# | **trabajo** $1-\tau_{\ell,t}$ | condición intratemporal | salarios rígidos con choques monetarios; poder monopólico de un sindicato sobre el salario |
# | **inversión** $1+\tau_{x,t}$ | ecuación de Euler | financiamiento de la inversión con costos de verificación del estado (agencia) |
# | **gasto** $g_t$ | restricción de recursos | economía abierta: exportaciones netas junto con las compras públicas —eso es el teorema; el panel que mediremos **no** es así: sus compras públicas viajan dentro de $c_t$ y la cuña sólo trae el sector externo (parte 3) |
#
# Todo lo demás que suele leerse en las cuñas es **analogía, no teorema**: la mala asignación
# entre unidades heterogéneas se lee en $A_t$ (Restuccia–Rogerson; Hsieh–Klenow) y las
# fricciones de búsqueda en $1-\tau_{\ell,t}$ (Shimer), pero ninguna viene con equivalencia
# probada en CKM. El mapa es de **muchos a uno**: la cuña que se mueve *nombra una familia*
# de modelos; no elige uno dentro de ella.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. Dónde entra cada cuña: las tres ecuaciones del prototipo
#
# El hogar descentralizado resuelve el problema de siempre con dos precios distorsionados
# —el salario neto $(1-\tau_{\ell,t})w_t$ y el costo de la inversión $(1+\tau_{x,t})$— y el
# gobierno se lleva $g_t$ de la restricción de recursos. Las tres condiciones son las que ya
# derivamos en el mazo, con una cuña cada una:
#
# **Intratemporal** (relación marginal de sustitución ocio–consumo $=$ producto marginal del
# trabajo, distorsionada por $\tau_\ell$):
#
# $$\psi\,l_t^{\nu}\,c_t^{\sigma}=(1-\tau_{\ell,t})\,(1-\alpha)\,\frac{y_t}{l_t}$$
#
# **Euler** (el costo de una unidad de capital hoy contra su retorno mañana, distorsionado
# por $\tau_x$ en **ambas** fechas):
#
# $$E_t\left\{\left[\alpha e^{a_{t+1}}k_t^{\alpha-1}l_{t+1}^{1-\alpha}+(1-\delta)(1+\tau_{x,t+1})\right]\frac{\beta c_t^{\sigma}}{c_{t+1}^{\sigma}}\right\}=(1+\tilde{g})\,(1+\tau_{x,t})$$
#
# **Restricción de recursos** (donde vive la cuña de gasto):
#
# $$c_t+\big[(1+g)k_t-(1-\delta)k_{t-1}\big]+g_t=e^{a_t}k_{t-1}^{\alpha}l_t^{1-\alpha}$$
#
# Con $\tau_{\ell,t}=\tau_{x,t}=g_t=0$ se recuperan **exactamente** las tres condiciones del
# modelo neoclásico deflactado de las **semanas 3–4** (mazo Slides02mav, lección 02). La cuña
# de eficiencia es $e^{a_t}$, ahora
# sin imponerle un AR(1). Usamos las preferencias del panel del curso,
# $U=c_t^{1-\sigma}/(1-\sigma)-\psi\,l_t^{1+\nu}/(1+\nu)$, con $\sigma=1$ (aversión al riesgo)
# y $\nu=2$ (**inversa** de la elasticidad de Frisch).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Medición: las cuñas son residuos, no estimaciones
#
# Dados $\alpha,\delta,\beta,\nu,\sigma,\psi$, una serie de capital por **inventario
# permanente** y los datos $\{y_t,c_t,x_t,h_t\}$, tres cuñas se despejan sin más. Las
# escribimos **con el fechado exacto con que se generó el panel congelado**, que no es el del
# libro de texto:
#
# $$A_t=\frac{y_t}{k_t^{\alpha_c}\,h_t^{1-\alpha_c}},\qquad
# \frac{1}{1-\tau_{\ell,t}}=\frac{(1-\alpha_c)\,y_t}{\psi\,\tilde h_t^{1+\nu}\,c_t^{\sigma}},\qquad
# g_t=y_t-c_t-x_t$$
#
# Tres avisos sobre la primera fórmula, porque cada uno cambia el número:
#
# 1. el capital es **contemporáneo** $k_t$, no $k_{t-1}$;
# 2. el exponente es $\alpha_c$, una **constante por país** —la media (winsorizada) de la
#    columna `alpha` sobre la base 2000–2006—, no la columna `alpha` trimestral;
# 3. $h_t$ es el **insumo de trabajo** que trae el panel, y no siempre son horas. Sobre esto
#    volvemos en un minuto, porque es el problema serio de esta lección.
#
# Y uno sobre la segunda: el insumo que entra en la intratemporal es $\tilde h_t$ —el trabajo
# **normalizado**, no el agregado $h_t$— y entra elevado a $1+\nu$, no a $\nu$. Es exactamente
# la fórmula del libro de texto con $l_t=\tilde h_t$: despejando de
# $\psi\,l_t^{\nu}c_t^{\sigma}=(1-\tau_{\ell,t})(1-\alpha)y_t/l_t$ queda una $l_t$ en el
# denominador que se junta con la otra. Lo verificamos abajo contra la columna `logS`, junto con
# la fórmula de $A_t$.
#
# ($\tilde h_t=h_t/(\text{EMP}_t\cdot\bar H)$ es la fracción del tiempo disponible que trabaja
# el ocupado promedio, con $\bar H=1152$ horas al trimestre —96 horas semanales de tiempo
# disponible por 12 semanas—; las cuñas se normalizan a 1 en la base 2000–2006, así que sólo
# sus **desvíos** tienen significado, y cualquier constante —$\psi$, $\bar H$, las unidades de
# $h_t$— se va con la normalización.)
#
# La de inversión es la única que vive dentro de una esperanza condicional. CKM tratan
# $s_t=(\log A_t,\tau_{\ell,t},\tau_{x,t},\log g_t)$ como un VAR(1) estimado por máxima
# verosimilitud junto con el prototipo. **El panel del curso toma un atajo declarado**: quita
# $E_t$ (previsión perfecta), impone $\tau_{x,t+1}=\tau_{x,t}$ y despeja, **fechando el
# residuo en $t+1$ y con el capital de $t+1$ en el producto marginal**:
#
# $$\frac{1}{1+\tau_{x,t+1}}\;\simeq\;
# \frac{\beta^{-1}(c_{t+1}/c_t)^{\sigma}-(1-\delta)}{\alpha_c\,y_{t+1}/k_{t+1}}$$
#
# Es un **residuo de Euler ex post**, no la cuña filtrada: hereda entera la sorpresa de
# $c_{t+1}$. Guarda esa frase; en la parte 7 nos cobrará la factura.
#
# **Y una incoherencia que hay que declarar de una vez**, porque nos acompañará hasta el final:
# el panel mide con capital **contemporáneo** ($k_t$ en $A_t$, $k_{t+1}$ en la Euler) y el
# prototipo que resolveremos en la parte 5 —el del mazo, el de la parte 2— produce con capital
# **rezagado** $k_{t-1}$. Un trimestre de desfase entre la regla que generó los residuos y la
# regla con la que los interpretamos. Rehacer la medición con el fechado del modelo está fuera
# del alcance de la lección (habría que regenerar el panel de investigación), así que lo
# tratamos como lo que es: una fuente más de distancia entre nuestro ejercicio y el de CKM,
# apuntada en la lista de diferencias declaradas del resumen.
#
# ### El archivo congelado
# `wedges_bca_curso.csv` es un subconjunto del panel de investigación del profesor
# (`wedges_panel_BGP_clean_nu2.csv`, 33 países de la OCDE, sin los agregados
# EA20/EU27/EU27_2020), recortado a seis países para que viaje ligero. Columnas:
#
# - `Y`, `C`, `I`: PIB, consumo e inversión reales trimestrales (con los **bienes duraderos**
#   reclasificados de consumo a inversión, como pide el modelo).
# - `K`: acervo de capital por inventario permanente. `alpha`: $1-$ participación laboral
#   trimestral; `alpha_c`: la constante por país que **de hecho** se usó en las fórmulas.
# - `EMP`: ocupados. `H`: insumo de trabajo. `use_hours`: 1 si `H` son horas de verdad.
# - `logA` $=\log A_t$; `logS` $=\log\!\big[1/(1-\tau_{\ell,t})\big]$;
#   `logD` $=\log\!\big[1/(1+\tau_{x,t})\big]$.

# %% slideshow={"slide_type": "fragment"}
# Panel congelado: subconjunto de wedges_panel_BGP_clean_nu2.csv (proyecto BCA del profesor),
# extraído el 2026-08-01; ver manifest.csv del bundle_2026A.
pan = pd.read_csv(DATA / "wedges_bca_curso.csv", parse_dates=["date"])
PAISES = ["USA", "ESP", "DEU", "GBR", "CAN", "KOR"]
NU, SIGMA = 2.0, 1.0            # inversa de Frisch y aversión al riesgo del panel
BASE = ("2000-01-01", "2006-12-31")     # ventana de normalización de las cuñas
FIN_MUESTRA = "2019-12-31"              # tendencia y VAR se estiman antes del COVID

print(f"{pan.shape[0]} filas · países = {sorted(pan.code.unique())}")
print(f"periodo: {pan.date.min().date()} .. {pan.date.max().date()}")
assert not pan.code.isin(["EA20", "EU27", "EU27_2020"]).any()   # sin agregados

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Antes de creerle al panel: reproducir sus propias fórmulas
# Si las ecuaciones de arriba son las que generaron el archivo, tenemos que poder
# **reconstruir `logA` y `logS` desde cero** con $Y$, $C$, $K$, $H$, $\text{EMP}$ y $\alpha_c$.
# Si el fechado de $A_t$ fuera el del libro de texto ($k_{t-1}$, `alpha` trimestral), o si en la
# cuña de trabajo pusiéramos el insumo agregado $h_t$ en vez del normalizado $\tilde h_t$, la
# reconstrucción fallaría. Comprobémoslo: no es un detalle de contabilidad, es la diferencia
# entre saber y suponer qué mide cada columna. (Comparamos en **desvíos de la base 2000–2006**,
# porque las constantes $\psi$, $\bar H$ y las unidades de $h_t$ no son observables.)

# %% slideshow={"slide_type": "fragment"}
HBAR = 1152.0                    # horas disponibles por trimestre (96 semanales x 12 semanas)


def _dev(s: pd.Series) -> pd.Series:
    """Desvío respecto a la media de la base 2000–2006 (quita cualquier constante)."""
    return s - s.loc[BASE[0]:BASE[1]].mean()


errA_ok, errA_mal, errS_ok, errS_mal = {}, {}, {}, {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    ac = float(u["alpha_c"].iloc[0])
    ht = u.H / (u.EMP * HBAR)                                          # insumo normalizado
    bien = np.log(u.Y / (u.K ** ac * u.H ** (1 - ac)))                 # K_t, alpha_c
    mal = np.log(u.Y / (u.K.shift(1) ** u.alpha * u.H ** (1 - u.alpha)))  # k_{t-1}, alpha_t
    errA_ok[code] = float(np.nanmax(np.abs(bien - u.logA)))
    errA_mal[code] = float(np.nanmax(np.abs(_dev(mal) - _dev(u.logA))))
    # cuña de trabajo: log[(1-a)y / (psi * ht^(1+nu) * c^sigma)], salvo constantes
    bienS = np.log(u.Y) - (1 + NU) * np.log(ht) - SIGMA * np.log(u.C)
    malS = np.log(u.Y / u.H) - NU * np.log(ht) - SIGMA * np.log(u.C)   # h_t agregado, ht^nu
    errS_ok[code] = float(np.nanmax(np.abs(_dev(bienS) - _dev(u.logS))))
    errS_mal[code] = float(np.nanmax(np.abs(_dev(malS) - _dev(u.logS))))

print("error máximo al reconstruir las columnas del panel (puntos logarítmicos)")
print(pd.DataFrame({"logA: K_t, alpha_c": errA_ok, "logA: K_{t-1}, alpha_t": errA_mal,
                    "logS: ht^(1+nu)": errS_ok, "logS: (y/h)·ht^nu": errS_mal}).to_string(
    float_format=lambda v: f"{v:.2e}"))
print("\nLas columnas 1 y 3 son cero a precisión de máquina: ésas son las fórmulas reales.")
print("La 2 no lo es: el fechado del libro de texto NO genera este panel.")
print("La 4 tampoco: en la cuña de trabajo el insumo es el NORMALIZADO ht, elevado a 1+nu.")
assert max(errA_ok.values()) < 1e-10 and max(errS_ok.values()) < 1e-10
assert min(errA_mal.values()) > 1e-3 and min(errS_mal.values()) > 1e-3

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El problema serio: en tres de los seis países el insumo de trabajo **no son horas**
# Tres de los **seis** que miramos aquí, no tres del mundo: en el panel de investigación del
# que sale este archivo (33 países) son siete —Australia, Suiza, el Reino Unido, Japón, Corea,
# Nueva Zelanda y EUA—, y de esos siete nos tocan tres. La cifra es de esta selección; no la
# cites como un dato del panel.
#
# El generador del panel usa horas trabajadas cuando la OCDE las publica y, cuando no,
# **impone** $h_t=\text{EMP}_t\times 480$ (40 horas semanales $\times$ 12 semanas). Ahí el
# margen **intensivo** desaparece por construcción: $h_t/\text{EMP}_t$ es exactamente
# constante, y la cuña de trabajo sólo puede ver despidos, nunca reducciones de jornada.
# Esto no hay que creérselo: se ve en el coeficiente de variación de $h_t/\text{EMP}_t$.
#
# **Aviso de unidades antes de mirar la tabla:** el nivel de la columna `H` no es homogéneo
# entre países —donde se impuso la jornada viene en miles de horas y donde hay horas de verdad,
# en millones—, así que la columna `h/EMP medio` sale como 480 en unos y como 0.3 en otros. Son
# las **mismas** unidades salvo un factor 1000: los tres países con horas trabajan entre 290 y
# 370 horas por trimestre, no 0.3. Nada de esto afecta a las cuñas, porque toda constante
# multiplicativa se va con la normalización a la base; la única columna que hay que leer es la
# de la derecha, el **coeficiente de variación**, que sí es adimensional.

# %% slideshow={"slide_type": "fragment"}
hpe = {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date")
    r = (u.H / u.EMP).dropna()
    esc = 1.0 if int(u.use_hours.iloc[0]) == 0 else 1000.0   # H en miles vs millones de horas
    hpe[code] = {"h/EMP crudo": r.mean(), "horas/trim.": r.mean() * esc,
                 "coef. de variación": r.std() / r.mean(),
                 "use_hours": int(u.use_hours.iloc[0])}
hpe = pd.DataFrame(hpe).T
print(hpe.to_string(float_format=lambda v: f"{v:,.4f}" if abs(v) > 1e-6 else f"{v:.1e}"))
print("('h/EMP crudo' es la columna del archivo tal cual; 'horas/trim.' la pone en horas por")
print(" ocupado. Las 480 impuestas quedan por encima de las horas reales de ESP/DEU/CAN: es un")
print(" nivel arbitrario, y no importa, porque la normalización a la base se lleva el nivel.)")

CON_HORAS = [c for c in PAISES if hpe.loc[c, "use_hours"] == 1]
SIN_HORAS = [c for c in PAISES if hpe.loc[c, "use_hours"] == 0]
print(f"\nhoras verdaderas: {CON_HORAS}")
print(f"EMP x 480 horas impuestas: {SIN_HORAS}  <-- aquí NO hay margen intensivo")
# En los países sin horas la razón h/EMP es constante a precisión de máquina.
assert all(hpe.loc[c, "coef. de variación"] < 1e-12 for c in SIN_HORAS)
assert all(hpe.loc[c, "coef. de variación"] > 1e-2 for c in CON_HORAS)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Léelo despacio, porque contamina el resultado principal.** EUA —el caso central de la
# lección y de CKM— está en el grupo **sin horas**: su $h_t$ es empleo puro. Como
#
# $$\frac{1}{1-\tau_{\ell,t}}\;\propto\;\frac{y_t}{\tilde h_t^{1+\nu}\,c_t^{\sigma}},$$
#
# un insumo de trabajo que **no cae** por la vía de la jornada deja la cuña de trabajo más
# **alta** de lo que estaría con horas verdaderas. Y "la cuña de trabajo sube en la Gran
# Recesión" es justamente el hallazgo que la parte 7 va a destacar. En la parte 7 medimos
# cuánto pesa esta contaminación, usando los tres países que **sí** traen horas.
#
# **México no está en el panel** y hay que decirlo: falta un acervo de capital trimestral
# homogéneo y una serie de empleo y horas armonizada con la OCDE en el mismo archivo.
# Trabajamos con EUA y España —el par del mazo— y dejamos la extensión a México como
# entregable al final.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La identidad que hay detrás: el empleo **no entra** en la cuña de trabajo
# Despejando $1-\tau_{\ell,t}$ de la intratemporal, con las preferencias del panel
# ($\sigma=1$, $\nu=2$) y $\tilde h_t=h_t/(\text{EMP}_t\bar H)$:
#
# $$\log(1-\tau_{\ell,t})=\log\frac{\psi\,\tilde h_t^{1+\nu}c_t^{\sigma}}{(1-\alpha_c)\,y_t}
# =\log\frac{c_t}{y_t}+(1+\nu)\log\tilde h_t+\text{cte}
# =\log\frac{C_t}{Y_t}+3\log\frac{H_t}{\text{EMP}_t}+\text{cte}$$
#
# Dos líneas de álgebra y sale un resultado incómodo: la cuña de trabajo medida es la razón
# consumo–producto más tres veces la **jornada**, y nada más. El empleo aparece únicamente
# dentro de $H_t/\text{EMP}_t$, así que **el margen extensivo se cancela**: una economía que
# despide al 10% de su gente sin tocarle la jornada tiene, con el mismo consumo y el mismo
# producto, exactamente la misma cuña de trabajo que antes. No es una aproximación de primer
# orden ni un resultado del modelo; es una identidad de la medición, y se comprueba al bit.

# %% slideshow={"slide_type": "fragment"}
def _logS(u: pd.DataFrame, H, EMP) -> pd.Series:
    """log[1/(1-tau_l)] del panel, salvo constantes, con un insumo (H, EMP) cualquiera."""
    return np.log(u.Y) - (1 + NU) * np.log(H / (EMP * HBAR)) - SIGMA * np.log(u.C)


ident, invar, solo_cy = {}, {}, {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    d = (-u.logS) - (np.log(u.C / u.Y) + (1 + NU) * np.log(u.H / u.EMP))
    ident[code] = float(np.max(np.abs(d - d.mean())))
    f = np.where(u.index >= pd.Timestamp("2008-01-01"), 0.90, 1.0)   # despidos puros en 2008Q1
    invar[code] = float(np.max(np.abs(_logS(u, u.H, u.EMP) - _logS(u, u.H * f, u.EMP * f))))
    e = (-100.0 * u.logS) - 100.0 * np.log(u.C / u.Y)                # cuña L menos consumo/PIB
    solo_cy[code] = float(np.max(np.abs(e - e.mean())))

print("la identidad log(1-tau_l) = log(C/Y) + 3 log(H/EMP) + cte, y qué la rompe")
print(pd.DataFrame({"error de la identidad": ident,
                    "despedir al 10% (jornada intacta)": invar,
                    "cuña de trabajo menos log(C/Y)": solo_cy}).to_string(
    float_format=lambda v: f"{v:.2e}"))
print("\nColumna 1: la identidad es exacta a precisión de máquina en los seis países.")
print("Columna 2: despedir al 10% de los ocupados sin tocar la jornada NO mueve la cuña.")
print("Columna 3: en los países SIN horas la cuña de trabajo ES la razón consumo-producto")
print("           (difieren en una constante, que el detrend se lleva); en los otros tres, no.")
assert max(ident.values()) < 1e-10 and max(invar.values()) < 1e-10
assert all(solo_cy[c] < 1e-9 for c in SIN_HORAS)
assert all(solo_cy[c] > 1.0 for c in CON_HORAS)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Es la subsección anterior vista del otro lado.** Allí dijimos que en EUA, el Reino Unido y
# Corea el insumo de trabajo es empleo por 480 horas. La identidad traduce esa frase a algo
# mucho más fuerte: si $H_t/\text{EMP}_t$ es constante, la cuña de trabajo de esos tres países
# es $100\log(C_t/Y_t)$ **y nada más** —la tercera columna lo confirma a $10^{-13}$—. Su cuña
# de trabajo no contiene un solo dato del mercado laboral: es la razón consumo–producto con
# otro nombre.
#
# Guárdalo para las partes 5 y 7. Cuando el reparto diga que la cuña de trabajo no explica la
# Gran Recesión en EUA, lo que estará diciendo es que la razón consumo–producto estadounidense,
# sola, no la explica. Y el factor $-(1+\nu)=-3$ que aparecerá en la auditoría de la parte 7 no
# es una elección de modelización: sale de esta identidad.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Orientación común y desvíos de la senda de crecimiento
# El panel guarda las cuñas con signos heterogéneos. Las volteamos a la convención del mazo
# —**caer = frenar la economía**— y las expresamos en **puntos logarítmicos** ($\times100$):
#
# $$\text{cuña}A=100\log A_t,\qquad \text{cuña}L=100\log(1-\tau_{\ell,t}),\qquad
# \text{cuña}X=-100\log(1+\tau_{x,t})$$
#
# La cuña de gasto no admite logaritmo (las exportaciones netas pueden ser negativas), así que
# la medimos como **porcentaje del PIB**: $\check g_t=100\,(y_t-c_t-x_t)/y_t$.
#
# Después le quitamos a **cada** serie —cuñas y producto— una tendencia lineal ajustada sobre
# 2000–2019. Es la versión operativa de "desvíos de la senda de crecimiento balanceado": el
# prototipo es estacionario, los datos no.

# %% slideshow={"slide_type": "fragment"}
def detrend(s: pd.Series) -> pd.Series:
    """Desvíos de una tendencia lineal ajustada sobre 2000Q1–2019Q4 (excluye COVID)."""
    t = np.arange(len(s))
    m = (s.index <= FIN_MUESTRA) & np.isfinite(s.to_numpy())
    b, a = np.polyfit(t[m], s.to_numpy()[m], 1)
    return pd.Series(s.to_numpy() - (a + b * t), index=s.index, name=s.name)


def cunas(code: str) -> pd.DataFrame:
    """Las cuatro cuñas y el producto, en desvíos y con orientación 'caer = frenar'."""
    u = pan[pan.code == code].sort_values("date").set_index("date")
    w = pd.DataFrame(index=u.index)
    w["cunaA"] = 100.0 * u["logA"]                       # eficiencia
    w["cunaL"] = -100.0 * u["logS"]                      # trabajo: 100*log(1-tau_l)
    w["cunaX"] = 100.0 * u["logD"]                       # inversión: -100*log(1+tau_x)
    w["cunaG"] = 100.0 * (u["Y"] - u["C"] - u["I"]) / u["Y"]   # gasto, % del PIB
    w["y"] = 100.0 * np.log(u["Y"])                      # producto
    return w.apply(detrend)


W = {c: cunas(c) for c in PAISES}
sd = pd.DataFrame({c: W[c].loc[:FIN_MUESTRA].std() for c in PAISES}).T.round(2)
print("desviación estándar 2000–2019 (cuñas A/L/X y el PIB en puntos logarítmicos; "
      "cuñaG en % del PIB)")
print(sd.to_string())
print("\nLa cuña de inversión es de lejos la más volátil: es un residuo de Euler ex post.")
assert (sd["cunaX"] > sd["cunaA"]).all()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Salvedad de unidades, antes de comparar nada.** Los tamaños **no** son comparables entre
# cuñas: cada una entra en una ecuación distinta y con una elasticidad distinta. Que la cuña
# de inversión tenga una desviación estándar entre casi cuatro (Canadá) y once veces (Alemania)
# mayor que la de eficiencia no significa que "haga cuatro veces más". Para eso hace falta el
# experimento de la parte 5: meterlas al modelo y ver qué producto sale.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Una nota sobre los datos: aquí la cuña de gasto **no** trae compras públicas
# La tabla de la parte 1 dice —y es el teorema— que la equivalencia de economía abierta pone en
# $g_t$ las exportaciones netas **junto con** las compras públicas. Este panel no funciona así,
# y conviene saberlo antes de leer un solo resultado: su $C$ es el consumo final **total**,
# hogares más gobierno, de modo que las compras públicas ya están dentro de $c_t$. Lo que queda
# en $g_t=y_t-c_t-x_t$ son las **exportaciones netas** más la variación de existencias y el
# residuo de encadenamiento. Política fiscal, ninguna.
#
# No hay que creérselo: el bundle lo comprueba. `oecd_qna_apertura.csv` —el archivo de la
# lección 10b— trae exportaciones, importaciones y consumo de los **hogares** para tres de
# nuestros seis países. Si la cuña fuera exportaciones netas *más* compras públicas, tendría
# que ir unos quince puntos del PIB por encima de las exportaciones netas solas.

# %% slideshow={"slide_type": "fragment"}
ap = pd.read_csv(DATA / "oecd_qna_apertura.csv", parse_dates=["date"])
ap = ap[ap.price_base == "L"]                    # volúmenes encadenados, como el panel de cuñas
COMUNES = [c for c in PAISES if c in set(ap.code)]

gasto, todos_g = {}, []
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    gY = 100.0 * (u.Y - u.C - u.I) / u.Y                       # la cuña de gasto, % del PIB
    todos_g.append(gY)
    fila = {"g/Y mediana": float(gY.median()),
            "C/Y base": 100.0 * float((u.C / u.Y).loc[BASE[0]:BASE[1]].mean())}
    if code in COMUNES:
        d = ap[ap.code == code].pivot_table(index="date", columns="variable", values="value")
        i = u.index.intersection(d.index)
        xn = 100.0 * (d.exp_vol - d.imp_vol).loc[i] / d.gdp_vol.loc[i]     # exportaciones netas
        hog = 100.0 * (d.conh_vol / d.gdp_vol).loc[i]                      # consumo de hogares
        fila.update({"XN/Y mediana": float(xn.median()),
                     "|g - XN| mediana": float(np.median(np.abs(gY.loc[i] - xn))),
                     "corr(g, XN)": float(np.corrcoef(gY.loc[i], xn)[0, 1]),
                     "hogares C/Y base": float(hog.loc[BASE[0]:BASE[1]].mean())})
        fila["C - hogares"] = fila["C/Y base"] - fila["hogares C/Y base"]
    gasto[code] = fila
gasto = pd.DataFrame(gasto).T
gtodo = pd.concat(todos_g)

print("qué contiene la cuña de gasto de este panel (todo en % del PIB, volúmenes encadenados)")
print(gasto.round({"corr(g, XN)": 3}).round(
    {c: 2 for c in gasto.columns if c != "corr(g, XN)"}).to_string(na_rep="    ·"))
print(f"\nel archivo de apertura cubre {sorted(ap.code.unique())}; comunes con PAISES: {COMUNES}")
print(f"mediana de g/Y en los seis países: {float(gtodo.median()):+.2f}% del PIB")
print(f"g y las exportaciones netas son la MISMA serie: se separan "
      f"{gasto.loc[COMUNES, '|g - XN| mediana'].max():.2f} puntos del PIB en la mediana y")
print(f"correlacionan {gasto.loc[COMUNES, 'corr(g, XN)'].min():.3f} o más; lo que sobra son "
      "existencias y encadenamiento.")
print(f"Y el consumo del panel va entre {gasto.loc[COMUNES, 'C - hogares'].min():.1f} y "
      f"{gasto.loc[COMUNES, 'C - hogares'].max():.1f} puntos del PIB por encima del")
print("consumo de los HOGARES: ese hueco es el consumo público, que viaja dentro de C. Si")
print("estuviera en la cuña, g/Y rondaría el +15% del PIB en vez del 0%.")
assert (gasto.loc[COMUNES, "|g - XN| mediana"] < 0.5).all()
assert (gasto.loc[COMUNES, "corr(g, XN)"] > 0.99).all()
assert (gasto.loc[COMUNES, "C - hogares"] > 10.0).all()
assert abs(float(gtodo.median())) < 1.0

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lo que esto cambia al leer la parte 5.** Cuando dentro de dos secciones la cuña de gasto no
# explique nada, lo que no explicará nada es el **sector externo**, con las existencias detrás;
# no el gasto público. Este panel no puede acusar ni exculpar a las compras públicas porque no
# las ve. En el prototipo de CKM sí entrarían ahí, y con consumo privado en vez de total la
# cuña de gasto sería otra serie.
#
# **Y una segunda salvedad sobre la misma cuña, ésta de unidades.** El mazo la define como un
# **nivel**: «$g_t$ … es la cuña de gasto: un nivel ($g_t=y_t-c_t-x_t$), a diferencia del $g_t$
# del bloque fiscal, que es una participación del PIB». Nosotros la medimos como participación,
# $\check g_t=100\,g_t/y_t$, y la restricción de recursos log-linealizada de la parte 5 pide la
# otra: $g_t$ en unidades del producto de la **senda**, $100\,g_t/\hat y_t$. La diferencia no es
# retórica —la participación se mueve **sola** cuando cae el denominador— y es exacta:
#
# $$\underbrace{100\,\frac{g_t}{y_t}}_{\text{participación}}-\underbrace{100\,\frac{g_t}{\hat y_t}}_{\text{nivel}}
# =100\,\frac{g_t}{\hat y_t}\left(\frac{\hat y_t}{y_t}-1\right)$$
#
# el nivel de la cuña multiplicado por la brecha del producto. Midámoslo donde el producto cae.

# %% slideshow={"slide_type": "fragment"}
EPISODIOS_G = [("Gran Recesión", "2007-10-01", "2009-04-01"),
               ("COVID", "2019-10-01", "2020-04-01")]
niv = {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    t = np.arange(len(u)); m = np.asarray(u.index <= FIN_MUESTRA)
    b, a = np.polyfit(t[m], np.log(u.Y.to_numpy())[m], 1)
    ytend = pd.Series(np.exp(a + b * t), index=u.index)         # producto de la senda
    g = u.Y - u.C - u.I
    part, nivel = 100.0 * g / u.Y, 100.0 * g / ytend
    brecha = 100.0 * (np.log(u.Y) - np.log(ytend))              # desvío del producto, pts log.
    # la separación entre las dos medidas es exactamente nivel x brecha del producto
    assert np.allclose(part - nivel, nivel * (ytend / u.Y - 1.0), atol=1e-10)
    for nom, t0, t1 in EPISODIOS_G:
        niv[(code, nom)] = {"brecha PIB (fondo)": float(brecha[t1]),
                            "cambio participación": float(part[t1] - part[t0]),
                            "cambio nivel": float(nivel[t1] - nivel[t0]),
                            "diferencia": float((part[t1] - part[t0]) - (nivel[t1] - nivel[t0]))}
niv = pd.DataFrame(niv).T
print("cuña de gasto: cambio pico -> fondo medido como participación y como nivel (% del PIB)")
print("('brecha PIB (fondo)' es el NIVEL del desvío del producto en el trimestre del")
print(" fondo, que es lo que multiplica a la cuña en la fórmula de arriba; las otras")
print(" tres columnas son CAMBIOS entre el pico y el fondo)")
print(niv.round(2).to_string())
peor = niv["diferencia"].abs().idxmax()
fila = niv.loc[peor]
print(f"\nmayor discrepancia: {peor[0]} en el {peor[1]}, {fila['diferencia']:+.2f} puntos del PIB,")
print(f"con el producto {fila['brecha PIB (fondo)']:.1f} puntos por debajo de su tendencia: la")
print(f"participación se mueve {fila['cambio participación']:+.2f} y el nivel {fila['cambio nivel']:+.2f}.")
assert float(niv["diferencia"].abs().max()) > 0.5

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Muerde donde el producto se desploma, y sólo ahí.** En 2008–09, con el producto entre tres
# puntos por encima y cinco por debajo de su tendencia en el trimestre del fondo, las dos
# medidas se separan poco más de medio punto del PIB como mucho —$0.52$ en Alemania, la peor
# de las seis— y da igual cuál uses. En 2020Q2 no da igual: el Reino Unido tiene el producto veinticinco puntos por
# debajo de su tendencia y las dos medidas se separan 0.8 puntos del PIB —más de la mitad de
# lo que se mueve el nivel, que son 1.5—. Esos 0.8 puntos no son gasto: son denominador. La
# regla: **cuando el episodio mueve el producto, la cuña de gasto en participación deja de
# ser la cuña de gasto**.
#
# Seguimos con la participación en el resto de la lección —es la serie con la que están hechas
# todas las cifras anteriores— y en la parte 5 comprobamos que alimentar el nivel no mueve el
# reparto de la Gran Recesión. Pero la definición correcta es el nivel, y en un episodio como
# el de 2020 habría que usarlo.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 1 — ¿qué cuña se mueve en cada episodio?
# Cuatro paneles: EUA y España, en la Gran Recesión y en el COVID. Las tres cuñas están
# suavizadas con una media móvil centrada de cuatro trimestres (el residuo de Euler es muy
# ruidoso). Todas caen cuando frenan a la economía.
#
# **Pregunta activa, antes de mirar el eje vertical:** ¿qué cuña se desploma en 2008–09 y cuál
# en 2020? ¿Y qué **familia** de modelos acusa cada episodio?

# %% slideshow={"slide_type": "slide"}
VENTANAS = [("Gran Recesión", "2006-01-01", "2012-12-31"),
            ("COVID", "2018-07-01", "2023-12-31")]
ETQ = [("cunaA", "eficiencia $A$"), ("cunaL", r"trabajo $1-\tau_\ell$"),
       ("cunaX", r"inversión $-(1+\tau_x)$")]

fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4))
for i, code in enumerate(["USA", "ESP"]):
    s = W[code][["cunaA", "cunaL", "cunaX"]].rolling(4, center=True, min_periods=2).mean()
    for j, (nom, t0, t1) in enumerate(VENTANAS):
        ax = axes[i, j]
        v = s.loc[t0:t1]
        for k, (col, lab) in enumerate(ETQ):
            ax.plot(v.index, v[col], color=_nbstyle.GRAYS[k], lw=1.5,
                    ls=_nbstyle.LINESTYLES[k], label=lab)
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_title(f"{code} — {nom}", fontsize=11)
        if j == 0:
            ax.set_ylabel("puntos logarítmicos")
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.show()

# Cambio de cada cuña entre el pico previo y el trimestre en que el PIB toca fondo,
# sobre las MISMAS series suavizadas que dibuja la figura.
EPISODIOS = [("Gran Recesión", "2007-10-01", "2009-04-01"),
             ("COVID", "2019-10-01", "2020-04-01")]

filas = {}
for code in ["USA", "ESP"]:
    s = W[code][["cunaA", "cunaL", "cunaX"]].rolling(4, center=True, min_periods=2).mean()
    for nom, pico, fondo in EPISODIOS:
        filas[(code, nom)] = s.loc[fondo] - s.loc[pico]
print("cambio de cada cuña, del pico al trimestre de mínimo del PIB (puntos logarítmicos)")
print(pd.DataFrame(filas).T.round(1).to_string())

for code in ["USA", "ESP"]:
    s = W[code]["cunaX"].rolling(4, center=True, min_periods=2).mean()
    print(f"{code}: mínimo de la cuña de inversión dentro de 2008–2010 = "
          f"{s.loc['2008-01-01':'2010-12-31'].min() - s.loc['2007-10-01']:.1f} pts")

# Aviso: en 2020 el residuo de Euler ex post ni siquiera existe en varios países.
falta = {c: [str(d.date()) for d in pan[(pan.code == c) & pan.logD.isna()].date]
         for c in PAISES}
print("\ntrimestres SIN cuña de inversión medida (logD = NaN):")
for c in PAISES:
    print(f"  {c}: {falta[c]}")
print("El de 2000Q1 es mecánico (la fórmula necesita t+1). Los demás no: el rendimiento bruto")
print("implícito, beta^-1 (c_{t+1}/c_t) - (1-delta), se vuelve NEGATIVO y el logaritmo no")
print("existe. La media móvil de cuatro trimestres tapa el hueco, así que las cifras del COVID")
print("de arriba se apoyan en los trimestres vecinos, no en el trimestre del fondo.")
# Los seis países pierden al menos un trimestre de 2020.
assert all(any(d.startswith("2020") for d in falta[c]) for c in PAISES)

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura.** En 2008–09 se mueven la **Euler** y la producción, no la intratemporal: la cuña
# de eficiencia cede unos 3 puntos en ambos países y la de **inversión** cae 12 puntos en
# España contra 2 en EUA en el trimestre del fondo (y toca mínimos de $-21$ y $-10$ dentro del
# episodio). La cuña de **trabajo**, en cambio, **sube respecto de su propia tendencia**: las
# horas cayeron *menos* de lo que la intratemporal exigía dado el desplome del consumo.
# Diagnóstico: la familia de las **fricciones financieras**, que es la lectura convencional de
# esa crisis. (Ojo con el "respecto de su propia tendencia": la celda siguiente explica por
# qué esa salvedad decide el **signo** del resultado en España.)
#
# En 2020 el cuadro se invierte. La cuña de **trabajo** española se hunde 20 puntos —una
# economía cuyos trabajadores no pueden ir a trabajar aparece, ante la ecuación intratemporal,
# como un impuesto brutal al trabajo— mientras la de inversión **salta hacia arriba** (+20 en
# EUA, +27 en España). Ese salto no es una relajación financiera: es el desplome del consumo
# confinado colándose por el residuo de la Euler. La cuña mide **la distancia al modelo**, no
# una fricción de la economía.
#
# Y el atajo de la Euler *ex post* enseña ahí su límite duro, no sólo su sesgo: la celda de
# arriba imprime que en 2020 **la cuña de inversión no existe** —los seis países pierden al
# menos un trimestre, y cinco de ellos pierden justo 2020Q2, el trimestre del fondo—. El salto
# del consumo es tan grande que el rendimiento bruto implícito se vuelve negativo y el
# logaritmo no está definido. Las cifras $+20$ y $+27$ salen de la media móvil, que rellena el
# hueco con los vecinos: son órdenes de magnitud, no mediciones del trimestre del fondo. Con la
# cuña filtrada de CKM esto no pasaría, porque $\tau_x$ no tendría que absorber entera la
# sorpresa de $c_{t+1}$.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Por qué estas cifras no son las del mazo (y las dos están bien)
#
# El mazo **Slides04**, en el pie de la misma figura de cuatro paneles, reporta para 2008–09
# que «la cuña de inversión cae 14 puntos logarítmicos en EUA y 22 en España, mientras la de
# eficiencia cede 1.7 y 4.0», y para 2020 que «la cuña de trabajo española se desploma 29
# puntos y la de inversión salta 65». Aquí salen $2$/$12$, $\approx3$/$\approx3$, $-20$ y
# $+27$. No es un error de ninguno de los dos: son **dos normalizaciones distintas sobre las
# mismas cuñas**. El `wedges_bca_slim.csv` con que se dibuja la figura del mazo (33 países) y el
# `wedges_bca_curso.csv` que leen esta lección y el recuadro de código del mazo salen del mismo
# panel de investigación y sus cuñas coinciden hasta $3\times10^{-14}$; lo que cambia es la
# receta de medición:
#
# | | mazo Slides04 | esta lección |
# |---|---|---|
# | tendencia | **no** se quita | detrend lineal 2000–2019 |
# | referencia | media del **año previo** (2006 / 2019) | valor en el **pico** (2007Q4 / 2019Q4) |
# | medida | **mínimo/máximo** dentro de 2006–2011 y 2019–2021 | cambio **pico → fondo** (2009Q2 / 2020Q2) |
#
# La misma advertencia vale para la tabla de desviaciones estándar de arriba. El mazo ancla la
# lección con «en EUA, 2000–2019, la desviación estándar de la cuña de inversión es de 6.7
# puntos logarítmicos, contra 4.4 de la de eficiencia y 1.4 de la de trabajo». Nuestra tabla da
# $6.67$, $0.97$ y $1.27$: la de inversión coincide (no tiene tendencia que quitar) y la de
# **eficiencia no**, porque $\log A_t$ sí la tiene y aquí se la quitamos. En niveles la cuña de
# eficiencia parece tres veces más volátil que la de trabajo ($4.4$ contra $1.4$); en desvíos de
# tendencia es algo **menos** volátil ($0.97$ contra $1.27$). Es la misma serie: cambia lo que
# se llama "el ciclo".
#
# Y una consecuencia que hay que declarar, porque cambia el diagnóstico: **el signo de la
# cuña de trabajo española en 2008–09 depende de la normalización**. Sin quitar tendencia
# cae (el recuadro de código del mazo imprime $-8.5$); en desvíos de la tendencia 2000–2019
# sube ($+2.4$). La razón es que la cuña laboral española tiene tendencia **descendente** a lo
# largo de la muestra: al removerla, lo que en niveles era una caída pasa a ser un movimiento
# por **encima** de la senda.
#
# Cuál de las dos lecturas quieres depende de la pregunta. El prototipo de CKM es
# estacionario, así que para *alimentarlo* con las cuñas (parte 5) hay que trabajar en
# desvíos; para describir la magnitud bruta del episodio, el nivel sin detrend dice más. Lo
# que no se vale —y es exactamente lo que la auditoría del curso cobra— es citar una cifra
# sin decir cuál de las dos recetas la produjo.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Calibrar el prototipo con los propios datos
#
# Antes del experimento hay que cerrar el modelo. No inventamos parámetros: los sacamos del
# mismo panel.
#
# - $\alpha$: la constante $\alpha_c$ del panel (columna `alpha_c`), que es la misma que entró
#   en la medición de las cuñas. Difiere de la media simple de `alpha` en la base en menos de
#   $2\times10^{-4}$, porque el generador winsoriza antes de promediar.
# - $\delta$: se recupera **exactamente** de la serie de capital, porque fue construida por
#   inventario permanente **con la inversión rezagada** —$K_t=(1-\delta)K_{t-1}+I_{t-1}$— de
#   modo que $\delta=1-(K_t-I_{t-1})/K_{t-1}$. El fechado importa: con la ley de libro de texto
#   $K_t=(1-\delta)K_{t-1}+I_t$ el mismo cálculo ya no es exacto, y lo verificamos abajo.
# - $k/y$, $c/y$, $x/y$: razones medias de la base 2000–2006. La razón de gasto es el residuo
#   $g/y=1-c/y-x/y$ (negativa donde las importaciones netas pesan).
# - $g$: pendiente de la tendencia lineal de $\log y$ (crecimiento trimestral de la senda).
# - $\beta$: **no** se calibra aparte; se elige para que la Euler se cumpla en el estado
#   estacionario, $\beta=(1+g)/\bar R$ con $\bar R=\alpha\,(y/k)+(1-\delta)$. Así el prototipo
#   es internamente consistente con las razones observadas.

# %% slideshow={"slide_type": "fragment"}
def calibra(code: str) -> dict:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    b = u.loc[BASE[0]:BASE[1]]
    alpha = float(u["alpha_c"].iloc[0])                        # constante del panel
    K, I = u["K"].to_numpy(), u["I"].to_numpy()
    delta = float(np.mean(1.0 - (K[1:] - I[:-1]) / K[:-1]))    # K_t = (1-d)K_{t-1} + I_{t-1}
    ky, cy, xy = float((b.K / b.Y).mean()), float((b.C / b.Y).mean()), float((b.I / b.Y).mean())
    m = np.asarray(u.index <= FIN_MUESTRA)
    g = float(np.polyfit(np.arange(m.sum()), np.log(u.Y.to_numpy()[m]), 1)[0])
    Rbar = alpha / ky + (1.0 - delta)
    return dict(alpha=alpha, delta=delta, ky=ky, cy=cy, xy=xy, gy=1 - cy - xy,
                g=g, Rbar=Rbar, beta=(1 + g) / Rbar)


cal = {c: calibra(c) for c in PAISES}
print(pd.DataFrame(cal).T[["alpha", "delta", "ky", "cy", "xy", "gy", "g", "beta"]]
      .round(4).to_string())
print("\ndelta = 0.015 trimestral (6% anual) en los seis países: es el valor con que se")
print("construyó el acervo K, recuperado aquí en vez de supuesto. Con el fechado correcto")
print("la recuperación es exacta a precisión de máquina; con el del libro de texto, no:")
chk = {}
for c in PAISES:
    u = pan[pan.code == c].sort_values("date")
    K, I = u["K"].to_numpy(), u["I"].to_numpy()
    chk[c] = {"delta con I_{t-1} (sd)": np.std(1.0 - (K[1:] - I[:-1]) / K[:-1]),
              "delta con I_t (media)": np.mean(1.0 - (K[1:] - I[1:]) / K[:-1]),
              "delta con I_t (sd)": np.std(1.0 - (K[1:] - I[1:]) / K[:-1])}
print(pd.DataFrame(chk).T.to_string(float_format=lambda v: f"{v:.2e}"))
assert all(abs(cal[c]["delta"] - 0.015) < 1e-9 for c in PAISES)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. El experimento contrafactual: una cuña a la vez
#
# Éste es el corazón del método. Se alimenta al prototipo con **una** cuña medida, se dejan las
# otras tres en su valor de la senda balanceada, y se compara la serie simulada con el dato.
# Lo hacemos de verdad, en tres pasos:
#
# **(a) Log-linealizar.** Con $\hat{}$ para desvíos logarítmicos, la producción y la
# intratemporal se resuelven estáticamente:
#
# $$\hat l_t=\frac{\hat\tau_{\ell,t}+\hat a_t+\alpha\hat k_{t-1}-\hat c_t}{\nu+\alpha},\qquad
# \hat y_t=\hat a_t+\alpha\hat k_{t-1}+(1-\alpha)\hat l_t$$
#
# la restricción de recursos entrega el capital y la Euler linealizada, el consumo:
#
# $$\hat c_t-E_t\hat c_{t+1}+\frac{1}{\bar R}\Big[\tfrac{\alpha}{k/y}\big(E_t\hat a_{t+1}+(\alpha-1)\hat k_t+(1-\alpha)E_t\hat l_{t+1}\big)+(1-\delta)E_t\hat\tau_{x,t+1}\Big]=\hat\tau_{x,t}$$
#
# **(b) Cerrar las expectativas.** El prototipo necesita saber cómo evolucionan las cuñas.
# CKM estiman el VAR(1) $s_{t+1}=Ps_t+Q\varepsilon_{t+1}$ por máxima verosimilitud junto con el
# modelo; nosotros lo estimamos por **mínimos cuadrados** sobre las cuñas medidas (2000–2019,
# suavizadas). Es una diferencia real y la declaramos.
#
# **(c) Resolver y simular.** `klein_solve` da $\hat k_t=G\,x_t$ y $\hat c_t=F\,x_t$ con
# $x_t=(\hat k_{t-1},\hat a_t,\hat\tau_{\ell,t},\hat\tau_{x,t},\check g_t)$. Alimentamos la
# **trayectoria medida** de una cuña, con las otras en cero, e iteramos.

# %% slideshow={"slide_type": "fragment"}
from puremacro.dsge import klein_solve

IK, IA, IL, IX, IG, IC = range(6)          # kk = k_{t-1}; c es la única variable de salto


def matrices(p: dict, P: np.ndarray):
    """A E_t z_{t+1} = B z_t, con z = (k_{t-1}, a, tau_l, tau_x, g, c)."""
    a, dl, ky, cy, g, Rb = p["alpha"], p["delta"], p["ky"], p["cy"], p["g"], p["Rbar"]
    n, e = 6, np.eye(6)
    A, B = np.zeros((n, n)), np.zeros((n, n))
    den = NU + a
    lv = (e[IL] + e[IA] + a * e[IK] - e[IC]) / den          # l_t como función de z_t
    yv = e[IA] + a * e[IK] + (1 - a) * lv                   # y_t como función de z_t
    # (1) recursos + acumulación: k_t = [y - (c/y)c - g + (1-delta)(k/y)k_{t-1}] / ((1+g)(k/y))
    A[0, IK] = 1.0
    B[0] = (yv - cy * e[IC] - e[IG] + (1 - dl) * ky * e[IK]) / ((1 + g) * ky)
    # (2)-(5) proceso exógeno de las cuñas: s_{t+1} = P s_t
    for r, i in enumerate([IA, IL, IX, IG]):
        A[1 + r, i] = 1.0
        for cc, j in enumerate([IA, IL, IX, IG]):
            B[1 + r, j] = P[r, cc]
    # (6) Euler linealizada
    A[5] = e[IC] - (1 / Rb) * (a / ky * (e[IA] + (1 - a) * lv) + (1 - dl) * e[IX])
    A[5, IK] += -(1 / Rb) * (a / ky) * (a - 1)              # (alpha-1) k_t
    B[5] = e[IC] - e[IX]
    return A, B


def prototipo(code: str, Wd: dict | None = None):
    """VAR(1) de las cuñas + solución de Klein del prototipo del país."""
    Wd = W if Wd is None else Wd
    s = Wd[code][["cunaA", "cunaL", "cunaX", "cunaG"]].rolling(4, center=True, min_periods=2).mean()
    # orientación del MODELO: a = log A, tau_l = log(1-tau_l), tau_x = log(1+tau_x)
    S = pd.DataFrame({"a": s.cunaA, "tl": s.cunaL, "tx": -s.cunaX, "g": s.cunaG})
    e = S.loc[:FIN_MUESTRA].dropna()
    P = np.linalg.lstsq(e.to_numpy()[:-1], e.to_numpy()[1:], rcond=None)[0].T
    A, B = matrices(cal[code], P)
    return S, P, klein_solve(A, B, n_pre=5)


def simula(sol, p: dict, S: pd.DataFrame, activas) -> pd.Series:
    """Producto simulado alimentando SOLO las cuñas de `activas` (las demás en cero)."""
    a, den = p["alpha"], NU + p["alpha"]
    X = S[["a", "tl", "tx", "g"]].copy()
    for c in X.columns:
        if c not in activas:
            X[c] = 0.0
    Xv, kk, out = X.to_numpy(), 0.0, np.zeros(len(X))
    for t in range(len(X)):
        x = np.r_[kk, Xv[t]]
        ch = float(sol.F[0] @ x)                                   # consumo (variable de salto)
        lh = (Xv[t, 1] + Xv[t, 0] + a * kk - ch) / den             # horas
        out[t] = Xv[t, 0] + a * kk + (1 - a) * lh                  # producto
        kk = float(sol.G[0] @ x)                                   # capital de mañana
    return pd.Series(out, index=X.index)


S_usa, P_usa, sol_usa = prototipo("USA")
print("EUA · flags de Blanchard–Kahn (existencia, unicidad) =", sol_usa.eu)
print("EUA · raíces del VAR(1) de las cuñas:",
      np.round(np.abs(np.linalg.eigvals(P_usa)), 3))
assert sol_usa.eu == (1, 1)          # solución estable y única

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### La prueba de humo: las cuatro cuñas juntas
# Por construcción, en el modelo **no lineal** de CKM las cuatro cuñas reproducen los datos
# *exactamente* — y por eso el ajuste no es evidencia de nada; el contenido está en el
# **reparto**. Nuestra versión es log-lineal y usa cuñas suavizadas, así que la réplica es
# buena pero no exacta. Vale la pena medir cuán buena antes de repartir.

# %% slideshow={"slide_type": "fragment"}
EXPERIMENTOS = [("todas", ["a", "tl", "tx", "g"]), ("A", ["a"]), ("L", ["tl"]),
                ("X", ["tx"]), ("G", ["g"])]
res = {}
for code in PAISES:
    S, P, sol = prototipo(code)
    m = S.loc[:FIN_MUESTRA].dropna().index                  # muestra 2000–2019 sin NaN
    sim = {lab: simula(sol, cal[code], S.loc[m], act) for lab, act in EXPERIMENTOS}
    res[code] = dict(sol=sol, yd=W[code]["y"].loc[m], sim=sim)

corr = pd.DataFrame({c: {k: np.corrcoef(v, res[c]["yd"])[0, 1]
                         for k, v in res[c]["sim"].items()} for c in PAISES}).T
vol = pd.DataFrame({c: {k: v.std() / res[c]["yd"].std()
                        for k, v in res[c]["sim"].items()} for c in PAISES}).T
print("correlación con el PIB observado (2000–2019), por conjunto de cuñas alimentadas")
print(corr.round(2).to_string())
print("\nvolatilidad simulada / volatilidad observada")
print(vol.round(2).to_string())

# %% [markdown] slideshow={"slide_type": "subslide"}
# **La réplica es buena, no exacta.** Con las cuatro cuñas juntas el prototipo reproduce entre
# el 91% (Canadá) y el 143% (Reino Unido) de la volatilidad observada del producto y
# correlaciona con él entre 0.51 (España) y 0.88 (Reino Unido). Nótese que sobrepasarse también
# es fallar: en el Reino Unido y Corea el prototipo genera **más** ciclo del que hubo.
# La brecha respecto al "exactamente" de CKM tiene tres culpables identificables:
# suavizamos las cuñas cuatro trimestres, linealizamos el modelo, y la $\tau_x$ del panel no es
# la cuña filtrada de CKM. Insistimos: que ajuste bien **no** es evidencia a favor de nada. Lo
# que sigue —el reparto— sí lo es.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Figura 2 — repartir la Gran Recesión
# Producto observado (línea gruesa) contra el producto que genera el prototipo alimentado con
# **una sola** cuña. Todo en puntos logarítmicos, normalizado a 0 en **2007Q4**, el pico previo.
#
# **Cuidado con la palabra "caída".** Lo que dibujamos y repartimos no es el PIB, sino su
# **desvío respecto a la tendencia lineal 2000–2019** (así lo definimos en la parte 3, porque
# el prototipo es estacionario). Un país puede tener un desvío muy negativo aunque su PIB
# apenas haya bajado: basta con que dejara de crecer al ritmo de su tendencia. La celda
# siguiente reporta **las dos** cifras lado a lado para que la diferencia no se pierda.

# %% slideshow={"slide_type": "slide"}
PICO, VALLE = "2007-10-01", "2009-04-01"
SERIES = [("todas", "las cuatro juntas", "0.55", (0, (6, 2))),
          ("A", "solo eficiencia", "0.15", (0, (1, 1.4))),
          ("L", "solo trabajo", "0.70", (0, (5, 1, 1, 1))),
          ("X", "solo inversión", "0.35", "-.")]

fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
for ax, code in zip(axes, ["USA", "ESP"]):
    r = res[code]
    v = slice("2006-01-01", "2011-12-31")
    ax.plot(r["yd"].loc[v].index, (r["yd"] - r["yd"][PICO]).loc[v],
            color="0.00", lw=2.4, label="PIB observado (desvío)")
    for lab, nom, gris, estilo in SERIES:
        y = r["sim"][lab]
        ax.plot(y.loc[v].index, (y - y[PICO]).loc[v], color=gris,
                lw=1.5, ls=estilo, label=nom)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_title(f"{code}: ¿qué cuña reproduce el desvío?", fontsize=11)
    ax.set_xlabel("trimestre")
axes[0].set_ylabel("PIB, desvío de tendencia\npuntos log. (2007Q4 = 0)")
axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.show()

# %% slideshow={"slide_type": "fragment"}
# Métrica canónica de CKM: fracción del desvío pico-valle que reproduce cada cuña sola.
cuota = {}
for code in PAISES:
    r = res[code]
    u = pan[pan.code == code].sort_values("date").set_index("date")
    dy = float(r["yd"][VALLE] - r["yd"][PICO])                 # desvío vs tendencia
    dy_bruto = float(100.0 * np.log(u.Y[VALLE] / u.Y[PICO]))   # PIB crudo, sin detrend
    cuota[code] = {"desvío vs tendencia": dy, "PIB crudo": dy_bruto,
                   **{lab: float(y[VALLE] - y[PICO]) / dy for lab, y in r["sim"].items()}}
cuota = pd.DataFrame(cuota).T
print("Gran Recesión 2007Q4 → 2009Q2, en puntos logarítmicos:")
print("  'desvío vs tendencia' = lo que el prototipo debe explicar (es lo que se reparte)")
print("  'PIB crudo'           = 100*log(Y_2009Q2/Y_2007Q4), sin quitar tendencia")
print("las columnas A/L/X/G son fracciones del DESVÍO, no del PIB crudo\n")
print(cuota.round(2).to_string())
print("\nsuma de las cuatro cuñas por separado vs la simulación conjunta:")
print((cuota[["A", "L", "X", "G"]].sum(axis=1) - cuota["todas"]).round(6).to_string())
# En una solución lineal la descomposición ES aditiva entre cuñas (superposición).
assert np.allclose(cuota[["A", "L", "X", "G"]].sum(axis=1), cuota["todas"], atol=1e-8)

print("\nLas dos columnas NO son la misma estadística; el caso extremo es Corea.")
print(f"  PIB crudo de Corea, 2007Q4 -> 2009Q2 : {cuota.loc['KOR', 'PIB crudo']:+.2f} pts log.")
print(f"  desvío de Corea vs su tendencia      : {cuota.loc['KOR', 'desvío vs tendencia']:+.2f} pts log.")
print(f"  crecimiento trimestral de tendencia  : {100 * cal['KOR']['g']:.2f}% (el mayor del panel)")
print(f"El PIB coreano practicamente no cayó; se quedó lejos de su tendencia porque dejó de")
print(f"crecer al ritmo al que venía. Cuando abajo digamos que allí la cuña de inversión")
print(f"reproduce el {100 * cuota.loc['KOR', 'X']:.0f}% del episodio, se lee sobre el desvío "
      "respecto a la tendencia,")
print("NO sobre una caída del PIB que casi no existió.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Cómo se lee esta tabla.** Una cuota de 0.65 significa que, alimentado **solo** con esa
# cuña, el prototipo genera el 65% del desvío observado del producto respecto a su
# tendencia entre 2007Q4 y 2009Q2 —no el 65% de una caída del PIB. Cuotas negativas
# significan que esa cuña, sola, empujaba al producto en la dirección **contraria** a la
# recesión. Las cuotas **no suman uno**, sino la fila `todas` —y eso lo verifica el `assert`
# de la celda: nuestra solución es lineal, así que la descomposición entre cuñas es **aditiva**
# por superposición. Fuera de la aproximación lineal ni siquiera esa aditividad se sostiene, y
# por eso CKM insisten en reportar también la simulación con las cuatro cuñas juntas. Que
# `todas` no dé exactamente 1.00 mide lo que al prototipo log-lineal se le escapa del dato.
#
# El patrón es nítido y sirve para descartar: la cuña de **gasto** no explica nada en ningún
# país, y la de **trabajo**, en la Gran Recesión, o no hace nada o empuja al revés —el insumo
# de trabajo *medido* cayó menos de lo que la intratemporal exigía dado el desplome del
# consumo. Toda la acción está entre eficiencia e inversión, y su reparto **cambia por país**:
# en Alemania la cuña de inversión es irrelevante y la eficiencia lo hace casi todo; en España
# y el Reino Unido la de inversión reproduce dos tercios o más del desvío.
#
# Ojo con la frase "el insumo de trabajo cayó menos": en EUA, el Reino Unido y Corea ese
# insumo **no puede** caer por reducción de jornada, porque el panel le impuso 480 horas
# constantes (parte 3). Lo cuantificamos en la parte 7 antes de sacar conclusiones.

# %% slideshow={"slide_type": "fragment"}
# ¿Cambia el reparto si la cuña de gasto entra como NIVEL, que es lo que pide la restricción de
# recursos (parte 3)? Se rehacen el VAR(1) y el contrafactual completos con la otra medida.
W_niv = {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    t = np.arange(len(u)); m = np.asarray(u.index <= FIN_MUESTRA)
    b, a = np.polyfit(t[m], np.log(u.Y.to_numpy())[m], 1)
    w2 = W[code].copy()
    w2["cunaG"] = detrend(100.0 * (u.Y - u.C - u.I) / np.exp(a + b * t))
    W_niv[code] = w2

dnivel = {}
for code in PAISES:
    S2, _, sol2 = prototipo(code, W_niv)
    m2 = S2.loc[:FIN_MUESTRA].dropna().index
    dy = float(W[code]["y"][VALLE] - W[code]["y"][PICO])
    fila = {}
    for lab, act in EXPERIMENTOS:
        y2 = simula(sol2, cal[code], S2.loc[m2], act)
        fila[lab] = float(y2[VALLE] - y2[PICO]) / dy - cuota.loc[code, lab]
    dnivel[code] = fila
dnivel = pd.DataFrame(dnivel).T
print("cambio de cada cuota al medir la cuña de gasto como NIVEL en vez de participación")
print(dnivel.round(4).to_string())
print(f"\nEl mayor cambio en cualquier cuota es {np.abs(dnivel.to_numpy()).max():.3f}. El reparto de")
print("la Gran Recesión no distingue las dos medidas —los desvíos del producto son de seis a")
print("ocho puntos—; en un episodio como el de 2020 sí las distinguiría (parte 3).")
assert np.abs(dnivel.to_numpy()).max() < 0.05

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 6. Salvedades: esto es un diagnóstico, no una explicación
#
# 1. **Por construcción las cuñas reproducen los datos.** Que el prototipo "ajuste" no es
#    evidencia a favor de nada. El contenido está en el reparto, no en el ajuste.
# 2. **El mapa fricción $\to$ cuña es de muchos a uno.** La cuña de trabajo es compatible con
#    impuestos, sindicatos, salarios rígidos o fricciones de emparejamiento, y el ejercicio no
#    puede escoger entre ellos. Nombra una familia; no elige un miembro.
# 3. **Christiano y Davis (2006).** La lectura de las cuñas no es robusta a dos decisiones que
#    parecen inocuas: (i) *cómo* se escribe la fricción dentro del prototipo —la misma fricción
#    financiera acaba en la cuña de inversión o dentro de la ecuación de acumulación de capital
#    según cómo se represente, y el veredicto cambia con ella—; y (ii) cómo se especifica y
#    estima el proceso conjunto $P,Q$ de las cuñas. Nuestro $P$ por mínimos cuadrados sobre
#    cuñas suavizadas es una elección más, y afecta al reparto.
# 4. **Todo lo que el prototipo no tiene reaparece disfrazado de cuña**: utilización variable
#    del capital, capital humano, informalidad, error de medición. La cuña mide la distancia al
#    modelo, no una fricción de la economía. Y lo que el prototipo sí tiene pero el **dato** no
#    —la jornada, en la mitad de este panel— reaparece igual: disfrazado de cuña de trabajo.
# 5. **Esto no es identificación causal.** No hay choque exógeno, ni instrumento, ni variación
#    plausiblemente aleatoria. Es una **descomposición contable** de la distancia entre datos y
#    un modelo de referencia. Comparar con el mazo Slides05 (semanas 9–10): allá se identifica;
#    aquí se diagnostica.
# 6. **La equivalencia va en la dirección cómoda.** *Dado* un modelo detallado se construye su
#    prototipo; de la cuña medida **no** se recupera el modelo.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 7. El veredicto: ¿confirma este panel el resultado canónico?
#
# El resultado canónico de CKM —Gran Depresión y recesión de 1982 en EUA— es que las cuñas de
# **eficiencia** y de **trabajo** reproducen casi todo el movimiento del producto y la de
# **inversión** casi nada, y a veces lo empuja en la dirección equivocada. Compruébalo tú: la
# figura siguiente pone lado a lado la cuota de la Gran Recesión que reproduce cada cuña, país
# por país.

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(8.4, 3.7))
ax.set_axisbelow(True)
etiq = [("A", "eficiencia", "0.10"), ("L", "trabajo", "0.40"),
        ("X", "inversión", "0.62"), ("G", "gasto", "0.86")]
anchura, xs = 0.2, np.arange(len(PAISES))
for k, (col, nom, gris) in enumerate(etiq):
    ax.bar(xs + (k - 1.5) * anchura, cuota.loc[PAISES, col].to_numpy(),
           width=anchura, color=gris, edgecolor="0.15", lw=0.5, label=nom, zorder=3)
ax.axhline(0, color="0.2", lw=0.8)
ax.axhline(1, color="0.4", lw=0.8, ls=(0, (4, 2)))
ax.text(len(PAISES) - 0.5, 1.03, "todo el desvío", fontsize=8, ha="right")
ax.set_xticks(xs)
ax.set_xticklabels([c + ("" if c in CON_HORAS else "*") for c in PAISES])
ax.set_ylabel("fracción del desvío del PIB\nrespecto a su tendencia, 2007Q4–2009Q2")
ax.set_title("Una cuña a la vez: reparto de la Gran Recesión")
ax.legend(fontsize=8, ncol=4, loc="upper left")
fig.text(0.01, -0.02, "* insumo de trabajo = empleo x 480 horas: sin margen intensivo",
         fontsize=7.5)
plt.tight_layout(); plt.show()

n_X = int((cuota.loc[PAISES, "X"] > 0.30).sum())
n_A = int((cuota.loc[PAISES, "A"] > 0.30).sum())
n_L = int((cuota.loc[PAISES, "L"] > 0.30).sum())
print(f"países (de {len(PAISES)}) donde la cuña sola reproduce >30% del desvío:")
print(f"  eficiencia {n_A}   ·   trabajo {n_L}   ·   inversión {n_X}")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **El veredicto, sin adornos.** Este panel **no** confirma la mitad del resultado canónico.
# La cuña de eficiencia sí hace mucho trabajo, como en CKM. Pero la de **trabajo** no explica
# la Gran Recesión en **ninguno** de los seis países, y la de **inversión** —que en CKM casi no
# importa— aquí reproduce más del 30% del desvío en cuatro de los seis (España, Reino Unido,
# Corea y, algo menos, EUA).
#
# Antes de anunciar que hemos refutado a CKM, hay que agotar las explicaciones aburridas, y hay
# tres muy fuertes. Dos de ellas son **defectos de nuestra medición**, una por cuña:
#
# - **El episodio es otro.** CKM estudian la Gran Depresión y 1982, dos recesiones sin crisis
#   financiera en el centro. 2008–09 sí lo fue. Que la cuña de inversión se mueva es
#   justamente lo que un economista esperaría, y Brinca, Chari, Kehoe y McGrattan (2016)
#   documentan cuñas de inversión activas en la Gran Recesión.
# - **$\tau_x$ está contaminada.** Nuestra $\tau_x$ es un residuo de Euler **ex post**: no
#   filtra expectativas, hereda entera la sorpresa del consumo de $t+1$. Si el consumo cae por
#   razones ajenas a la tasa de interés, esa caída aparece como fricción de inversión aunque no
#   lo sea. El COVID lo enseña en caricatura (Figura 1): la cuña de inversión **sube** en 2020,
#   lo que no puede leerse como una relajación financiera —y en el trimestre del fondo ni
#   siquiera existe, porque el logaritmo de un rendimiento negativo no está definido.
# - **$\tau_\ell$ también está contaminada, y justo donde más duele.** En EUA, el Reino Unido y
#   Corea el insumo de trabajo es empleo $\times\,480$: el margen **intensivo** no existe por
#   construcción. Como $1/(1-\tau_{\ell,t})\propto y_t/(\tilde h_t^{1+\nu}c_t^{\sigma})$, borrar la
#   caída de la jornada **sube** la cuña de trabajo medida — que es exactamente el hallazgo que
#   acabamos de destacar. La conclusión "la cuña de trabajo no hace nada" descansa sobre un
#   insumo mutilado en tres de los seis países. Cuantifiquémoslo.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Auditoría del margen intensivo
# No podemos inventarle horas a EUA. Lo que **sí** podemos hacer es el experimento inverso en
# los tres países que las traen (ESP, DEU, CAN): **borrarles** el margen intensivo igual que
# hizo el generador, sustituyendo $h_t$ por $\text{EMP}_t\times\overline{h/\text{EMP}}$, y
# volver a medir. Como la jornada entra en **dos** cuñas, hay que mover las dos. Escribiendo
# $\hat\jmath_t=\log(h_t/\text{EMP}_t)$ en desvíos de la base 2000–2006:
#
# $$\Delta\log A_t=+(1-\alpha_c)\,\hat\jmath_t,\qquad
# \Delta\log(1-\tau_{\ell,t})=-(1+\nu)\,\hat\jmath_t$$
#
# **Los signos son opuestos, y conviene entender por qué antes de mirar los números.** Al
# sustituir $h_t$ por $\text{EMP}_t\times\overline{h/\text{EMP}}$, el insumo medido cambia en
# $-\hat\jmath_t$: en una recesión con recorte de jornada ($\hat\jmath_t<0$) el insumo *sube*
# respecto del real. Un insumo mayor con el mismo producto da una $A_t$ medida **más baja**
# (entra con exponente $-(1-\alpha_c)$ en $\log A$) y una cuña de trabajo **más alta** (entra
# con exponente $-(1+\nu)$ en $\log(1-\tau_\ell)$, que es la orientación "caer $=$ frenar").
# Borrar la jornada, entonces, **hunde** la eficiencia medida y **levanta** el trabajo medido.
#
# En magnitud el factor de la cuña de trabajo es $1+\nu=3$, unas seis veces el de la de
# eficiencia ($1-\alpha_c\approx0.5$). Pero cuidado: eso es el efecto **directo sobre la cuña**,
# y la cuota que nos importa pasa además por el VAR(1) reestimado y por el equilibrio general
# del prototipo. No supongas la respuesta; rehacemos el contrafactual completo y la medimos.

# %% slideshow={"slide_type": "fragment"}
W_sin = {}
for code in PAISES:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    j = np.log(u.H / u.EMP)
    j = j - j.loc[BASE[0]:BASE[1]].mean()                # desvío de la jornada vs la base
    ac = float(u["alpha_c"].iloc[0])
    w2 = W[code].copy()
    w2["cunaA"] = detrend(100.0 * u["logA"] + 100.0 * (1 - ac) * j)   # +: sube h, baja A
    w2["cunaL"] = detrend(-100.0 * u["logS"] - 100.0 * (1 + NU) * j)  # -: sube h, sube 1-tau_l
    W_sin[code] = w2

# Comprobación: las dos cuñas mutiladas son las que salen de recalcularlas con h = EMP x cte.
for code in CON_HORAS:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    ac = float(u["alpha_c"].iloc[0])
    Hfijo = u.EMP * np.exp(np.log(u.H / u.EMP).loc[BASE[0]:BASE[1]].mean())
    A_ref = 100.0 * np.log(u.Y / (u.K ** ac * Hfijo ** (1 - ac)))
    S_ref = -100.0 * (np.log(u.Y) - (1 + NU) * np.log(Hfijo / (u.EMP * HBAR))
                      - SIGMA * np.log(u.C))
    assert np.allclose(W_sin[code]["cunaA"], detrend(A_ref), atol=1e-9)
    assert np.allclose(W_sin[code]["cunaL"], detrend(S_ref), atol=1e-9)
print("las cuñas mutiladas coinciden con recalcularlas desde cero con h = EMP x jornada fija")

aud = {}
for code in PAISES:
    S2, _, sol2 = prototipo(code, W_sin)
    m2 = S2.loc[:FIN_MUESTRA].dropna().index
    dy = float(W[code]["y"][VALLE] - W[code]["y"][PICO])
    fila = {"horas verdaderas": int(code in CON_HORAS)}
    for lab, act in [("A", ["a"]), ("L", ["tl"])]:
        y2 = simula(sol2, cal[code], S2.loc[m2], act)
        fila[f"cuota {lab} panel"] = cuota.loc[code, lab]
        fila[f"cuota {lab} sin jornada"] = float(y2[VALLE] - y2[PICO]) / dy
        fila[f"sesgo {lab}"] = fila[f"cuota {lab} sin jornada"] - fila[f"cuota {lab} panel"]
    aud[code] = fila
aud = pd.DataFrame(aud).T
print("¿cuánto cambia el reparto si se borra el margen intensivo (empleo x horas fijas)?")
print(aud.round(3).to_string())
# Donde el panel ya impuso 480 horas, borrar la jornada no cambia nada: no hay qué borrar.
assert np.allclose(aud.loc[SIN_HORAS, ["sesgo A", "sesgo L"]].to_numpy(float), 0.0, atol=1e-9)
print(f"\nEn {SIN_HORAS} el sesgo es cero porque el panel YA había borrado la jornada:")
print("no es que estén limpios, es que su contaminación no se puede deshacer desde aquí.")
print(f"En {CON_HORAS} —los que sí tienen horas— borrarla mueve la cuota de la cuña de")
print(f"trabajo entre {aud.loc[CON_HORAS, 'sesgo L'].min():+.2f} y "
      f"{aud.loc[CON_HORAS, 'sesgo L'].max():+.2f}, y la de eficiencia entre "
      f"{aud.loc[CON_HORAS, 'sesgo A'].min():+.2f} y {aud.loc[CON_HORAS, 'sesgo A'].max():+.2f}.")
print("Los sesgos son del mismo orden que las cuotas y tienen SIGNO SISTEMÁTICO: borrar la")
print("jornada le quita explicación a la cuña de trabajo y se la da a la de eficiencia, en los")
print("tres países y por una magnitud parecida. Es un traspaso, no ruido.")
# El sesgo es un traspaso: lo que pierde el trabajo lo gana (casi todo) la eficiencia.
assert (aud.loc[CON_HORAS, "sesgo L"] < -0.15).all()
assert (aud.loc[CON_HORAS, "sesgo A"] > +0.15).all()
print("\nLo que SÍ es robusto a la mutilación (vale con y sin jornada, en los seis países):")
qL = np.r_[aud["cuota L panel"].to_numpy(float), aud["cuota L sin jornada"].to_numpy(float)]
qA = np.r_[aud["cuota A panel"].to_numpy(float), aud["cuota A sin jornada"].to_numpy(float)]
print(f"  cuota de la cuña de TRABAJO     en [{qL.min():+.2f}, {qL.max():+.2f}]  -> siempre pequeña")
print(f"  cuota de la cuña de EFICIENCIA  en [{qA.min():+.2f}, {qA.max():+.2f}]  -> siempre grande")
assert qL.max() < 0.30 and qA.min() > 0.40
print("\nY la moraleja para los tres países SIN horas: si su jornada real hubiera entrado en la")
print("medición, su cuota de trabajo sería MAYOR y la de eficiencia MENOR que las de la tabla")
print("de la parte 5, por un margen del orden de las dos décimas que acabamos de medir.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### El experimento inverso: prestarle una jornada a quien no la tiene
# La auditoría de arriba borra la jornada donde la hay. La identidad de la parte 3 permite el
# movimiento contrario **sin simular nada**: devolverle una jornada $\hat\jmath_t$ a un país
# medido con $h_t=\text{EMP}_t\times480$ mueve sus dos cuñas en cantidades exactas,
#
# $$\Delta\log(1-\tau_{\ell,t})=+(1+\nu)\,\hat\jmath_t=+3\,\hat\jmath_t,\qquad
# \Delta\log A_t=-(1-\alpha_c)\,\hat\jmath_t.$$
#
# No tenemos la jornada de EUA, del Reino Unido ni de Corea: el archivo del curso no la trae.
# (El panel de investigación del que sale sí guarda una columna de horas para EUA y para el
# Reino Unido y aun así los marca `use_hours = 0`; para Corea no hay horas que descartar. Esa
# columna no viaja en el bundle, así que aquí no podemos usarla.) Lo que sí tenemos son **tres
# jornadas observadas** en el mismo episodio: las de España, Alemania y Canadá. Prestémoselas.

# %% slideshow={"slide_type": "fragment"}
jor = {}
for code in CON_HORAS:
    u = pan[pan.code == code].sort_values("date").set_index("date")
    j = np.log(u.H / u.EMP)
    jor[code] = 100.0 * float(j[VALLE] - j[PICO])
print("jornada observada: cambio del pico al valle, 2007Q4 -> 2009Q2 (puntos logarítmicos)")
print("  " + "   ".join(f"{c} {v:+.2f}" for c, v in jor.items()))

prest = {}
for code in SIN_HORAS:
    ac = float(pan[pan.code == code]["alpha_c"].iloc[0])
    dL = float(W[code]["cunaL"][VALLE] - W[code]["cunaL"][PICO])
    dA = float(W[code]["cunaA"][VALLE] - W[code]["cunaA"][PICO])
    conL = [dL + (1 + NU) * j for j in jor.values()]
    conA = [dA - (1 - ac) * j for j in jor.values()]
    prest[code] = {"cuña L medida": dL, "L prestada mín": min(conL), "L prestada máx": max(conL),
                   "cuña A medida": dA, "A prestada mín": min(conA), "A prestada máx": max(conA)}
prest = pd.DataFrame(prest).T
print("\ncambio de las cuñas del pico al valle, medido y con la jornada prestada")
print("(puntos logarítmicos, series SIN suavizar: no son las cifras de la figura 1)")
print(prest.round(2).to_string())
print("\nCon CUALQUIERA de las tres jornadas observadas, la cuña de trabajo de EUA, el Reino")
print("Unido y Corea cambia de signo en la Gran Recesión: deja de subir y cae. El '+3' de la")
print("identidad multiplica una jornada que en estos tres países vale 480 horas por decreto.")
assert all(prest.loc[c, "cuña L medida"] > 0 > prest.loc[c, "L prestada máx"] for c in SIN_HORAS)
print("Ojo con lo que esto es y lo que no. Es la CUÑA, exacta por identidad, no la CUOTA:")
print("la cuota pasa además por el VAR(1) y por el equilibrio general, y ahí seguimos con")
print("la conjetura del punto 3 de abajo. Y la jornada es PRESTADA: da el signo y el orden")
print("de magnitud del sesgo, no la cifra de cada país —la jornada estadounidense no tiene")
print("por qué parecerse a la alemana, y nadie la ha medido en esta lección.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Qué se salva y qué no del veredicto.** La auditoría deja tres conclusiones, en orden de
# solidez decreciente.
#
# 1. **Robusto.** Bajo *las dos* mediciones y en los seis países, la cuña de trabajo no pasa
#    de un tercio en valor absoluto —la celda imprime el rango, $[-0.33, +0.18]$— y la de
#    eficiencia se queda por encima de $0.5$ (rango impreso $[+0.51, +0.94]$). El titular
#    cualitativo —la eficiencia hace mucho, el trabajo poco— aguanta la mutilación del
#    insumo de trabajo.
# 2. **No robusto: los números, y el sesgo tiene dirección conocida.** Los sesgos medidos
#    ($-0.19$ a $-0.30$ en la cuota de trabajo, $+0.18$ a $+0.28$ en la de eficiencia) son del
#    mismo orden que las cuotas mismas, y borrar la jornada **traspasa** explicación del trabajo
#    a la eficiencia en los tres países, sin excepción. En Alemania y Canadá la cuota de trabajo
#    hasta cambia de signo (de $+0.07$ a $-0.14$ y de $+0.18$ a $-0.12$): la lectura fina "en
#    Canadá la cuña de trabajo amortiguó la recesión" no sobrevive a la mutilación. La lectura
#    que sí sobrevive en España es sólo la del signo, no la de la magnitud: pasa de $-0.14$ a
#    $-0.33$.
# 3. **Lo que no podemos saber... salvo su dirección.** En EUA, el Reino Unido y Corea el sesgo
#    aparece como cero sólo porque su jornada ya venía borrada del origen. No están limpios:
#    están **fuera del alcance de la auditoría**. Pero el punto 2 nos deja algo más que un
#    encogimiento de hombros: como el traspaso trabajo $\to$ eficiencia tiene el mismo signo en
#    los tres países auditables, lo esperable es que las cuotas de EUA, el Reino Unido y Corea
#    en la tabla de la parte 5 **subestimen** la del trabajo y **sobreestimen** la de la
#    eficiencia, por un margen del orden de dos décimas. Es una conjetura disciplinada por tres
#    observaciones, no una medición: no la escribas como si lo fuera. La celda de la jornada
#    prestada aprieta un poco más la tuerca: con la jornada de cualquiera de los tres países
#    auditables, la **cuña** de trabajo de los tres no auditables **cambiaría de signo** en el
#    episodio. El cálculo es exacto —es la identidad de la parte 3— pero la jornada es
#    prestada; lo que sigue siendo conjetura es la cuota, y la jornada verdadera de esos tres
#    países no la mide nadie aquí.
#
# Así que la frase honesta no es "la cuña de trabajo no explica la Gran Recesión en ninguno de
# los seis países", sino: *en ninguna de las mediciones que este panel permite construir la
# cuña de trabajo pasa de un tercio del desvío; y en tres de los seis países el ejercicio ni
# siquiera es una prueba limpia del resultado de CKM, con un sesgo cuya dirección conocemos y
# cuya magnitud sólo podemos acotar por analogía.*
#
# Ésta es la lección metodológica del módulo: el resultado del reparto depende de cómo se
# midió cada cuña, y un buen economista aplicado audita la medición antes que la teoría.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 8. Preguntas para pensar y mini-entregable
#
# 1. **Elige un país del panel y un episodio.** Mide el movimiento de cada cuña frente a su base y
#    su cuota en el contrafactual. ¿Qué familia de modelos acusa ese episodio? *Trampa
#    deliberada*: el salto de la cuña de inversión en 2020 hereda el desplome del consumo. Si
#    lo lees como fricción financiera, ¿qué predicción falsa harías sobre el crédito en 2020?
# 2. **Utilización variable.** Si la utilización del capital sube y baja con el ciclo y no se
#    observa, ¿en qué dirección se sesga la $A_t$ medida y qué le pasa a la cuota de la cuña de
#    eficiencia? ¿Es un problema de medición o una fricción real?
# 3. **El margen intensivo (el problema real de este panel).** En EUA, el Reino Unido y Corea
#    el insumo $h_t$ es empleo $\times\,480$: **sólo** margen extensivo. Si en una recesión las
#    empresas recortan la jornada antes que la plantilla, ¿en qué dirección se equivoca la cuña
#    de trabajo medida, y qué le pasa a su cuota? Compáralo con el sesgo que midió la
#    auditoría de la parte 7 y con el módulo 24 (cuatro estados) y la informalidad, donde el
#    ajuste corre por márgenes que un solo número de "trabajo" tampoco puede ver.
# 4. **¿Qué modelo vale la pena escribir?** Si en tu país la cuña de inversión no se mueve,
#    ¿puedes descartar las fricciones financieras como historia principal? ¿Y si sí se mueve,
#    puedes concluir que *son* la historia? Las dos preguntas no son simétricas: explica por
#    qué el poder del método está en **descartar**.

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Notas para las preguntas
# 1. La trampa: en 2020 la cuña de inversión **sube** porque es un residuo de Euler ex post
#    que hereda el desplome del consumo confinado. Leerlo como relajación financiera te
#    obligaría a predecir crédito abundante y barato al sector privado en 2020Q2 —justo
#    cuando el crédito se sostuvo por avales públicos y no por una mejora del descuento
#    privado—. La predicción falsa es observable, y por eso el episodio sirve de vacuna.
#    Segunda vuelta de tuerca, para quien mire el archivo: en 2020Q2 la columna `logD` es
#    **NaN** en cinco de los seis países, porque el rendimiento bruto implícito se vuelve
#    negativo. El "salto" que se ve en la figura lo pone la media móvil, no el dato.
# 2. Si la utilización es procíclica y no se observa, la $A_t$ medida absorbe sus
#    movimientos: se vuelve **más volátil y más procíclica** de lo que es la tecnología, y la
#    cuota de la cuña de eficiencia queda **sobreestimada**. Es un problema de **medición**
#    del insumo (capital efectivo $u_tK_t$), no una fricción; pero en el modelo se disfraza
#    de fricción, que es exactamente el punto de la parte 7.
# 3. Con $h_t$ = empleo $\times\,480$, un recorte de jornada no se registra: el insumo medido
#    cae **menos** de lo que cayó el insumo real. Como
#    $1/(1-\tau_\ell)\propto y/(\tilde h^{1+\nu}c^{\sigma})$, la cuña de trabajo medida queda **más
#    alta** (más "sube") de lo que corresponde, y su cuota se **subestima**; y la $A_t$ medida,
#    con un insumo mayor para el mismo producto, queda más baja, así que su cuota se
#    **sobreestima**. La parte 7 lo mide en los tres países que sí traen horas: borrar la
#    jornada mueve la cuota de trabajo entre $-0.19$ y $-0.30$ y la de eficiencia entre
#    $+0.18$ y $+0.28$. Es un traspaso, y su dirección es la que acabas de razonar. En la
#    lección 24 el problema aparece con otro disfraz: el ajuste mexicano corre por la
#    informalidad y la participación, márgenes que un solo agregado de "trabajo" no ve.
# 4. No son simétricas. Si la cuña de inversión **no** se mueve, cualquier modelo cuya
#    fricción entre por la ecuación de Euler está descartado: por construcción tendría que
#    haber dejado huella ahí. Si **sí** se mueve, no puedes concluir nada, porque la cuña es
#    un residuo y cualquier cosa que la Euler no explique —fricciones financieras, pero
#    también errores de medición del consumo o del capital, choques de riesgo, cambios en la
#    tasa de descuento— aparece en el mismo lugar. El método es un **filtro de rechazo**, no
#    un identificador.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Mini-entregable — meter a México en el panel.** México no está aquí porque falta el insumo,
# no porque no importe. Escribe la receta completa: (i) PIB, consumo y FBCF trimestrales reales
# de INEGI, con duraderos reclasificados a inversión; (ii) acervo de capital por inventario
# permanente con $\delta$ trimestral y un $K_0$ de la razón $k/y$ de la senda; (iii) horas
# trabajadas de la ENOE —y aquí la decisión fina: ¿horas totales de ocupados formales e
# informales, o sólo formales?—; (iv) $\alpha$ de la participación laboral de cuentas
# nacionales. Argumenta qué cuña quedaría más contaminada por la informalidad y por qué.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 9. Explora con IA
# Prueba estas indicaciones con el tutor sin conexión (o cualquier asistente de IA):
# - "¿Por qué que las cuatro cuñas reproduzcan los datos exactamente NO es evidencia a favor
#   del modelo neoclásico?"
# - "Explica la crítica de Christiano y Davis a la contabilidad del ciclo en dos frases."

# %% slideshow={"slide_type": "fragment"}
print(tutor(
    "En dos o tres frases: ¿por qué la contabilidad del ciclo económico (cuatro cuñas) sirve "
    "para DESCARTAR familias de modelos pero no para identificar una fricción concreta?",
    context=("Cuotas del desvío del PIB respecto a su tendencia, 2007Q4-2009Q2, "
             "reproducidas por cada cuña sola: "
             f"EUA A={cuota.loc['USA','A']:.2f} L={cuota.loc['USA','L']:.2f} "
             f"X={cuota.loc['USA','X']:.2f}; ESP A={cuota.loc['ESP','A']:.2f} "
             f"L={cuota.loc['ESP','L']:.2f} X={cuota.loc['ESP','X']:.2f}."),
))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Medimos las cuatro cuñas de Chari–Kehoe–McGrattan como residuos de las
# condiciones de equilibrio sobre un panel trimestral congelado de seis países, calibramos el
# prototipo con los propios datos, lo resolvimos log-linealizado con `klein_solve` y corrimos
# el **experimento contrafactual**: una cuña a la vez, midiendo qué fracción del **desvío del
# PIB respecto a su tendencia** entre 2007Q4 y 2009Q2 reproduce cada una (que no es lo mismo
# que la caída del PIB: en Corea el PIB casi no bajó y el desvío fue de $-5.7$ puntos).
# La cuña de eficiencia hace mucho trabajo, la de trabajo poco en este episodio, y la de
# inversión mucho más de lo que el resultado canónico de CKM haría esperar. Auditamos las
# **dos** contaminaciones de la medición: el residuo de Euler *ex post* con que este panel mide
# $\tau_x$ —que en 2020 ni siquiera está definida en el trimestre del fondo—, y el insumo de
# trabajo sin margen intensivo (empleo $\times\,480$) de EUA, el Reino Unido y Corea, cuyo
# efecto medido es un **traspaso** de dos décimas de cuota del trabajo a la eficiencia. Las
# cuñas son un **diagnóstico**: descartan familias enteras de modelos, y ese poder de descarte,
# no la identificación, es su aportación.
#
# **Diferencias con el experimento completo de CKM**, declaradas: (i) resolvemos el prototipo
# **log-linealizado**, no en niveles, de modo que las componentes suman **exactamente** por
# superposición —una comodidad de la aproximación, no una propiedad del método: fuera de ella
# la descomposición no es aditiva, y por eso CKM reportan también la simulación conjunta—;
# (ii) el VAR(1) de las cuñas se estima por **mínimos cuadrados** sobre las cuñas medidas y
# suavizadas, no por máxima verosimilitud conjunta con el modelo; (iii) $\tau_x$ viene del
# atajo *ex post* del panel, no de la Euler filtrada; (iv) en tres de los seis países el insumo
# de trabajo es empleo con jornada fija, no horas, de modo que $\tau_\ell$ (y en menor medida
# $A$) sólo ven el margen extensivo; (v) el panel mide con capital contemporáneo y el prototipo
# produce con capital rezagado: un trimestre de desfase entre medición e interpretación; y
# (vi) la cuña de gasto de este panel es sector externo y existencias, **sin** compras
# públicas —viven dentro de $C$—, y la alimentamos como participación del PIB y no como el
# nivel que pide la restricción de recursos (comprobado en la parte 5: mueve el reparto menos
# de 0.02).
#
# **Lo que esta lección no cubre** y sí hace `T04_C`, el cuaderno del escaparate de `puremacro`
# —que no viaja en el paquete del curso, pídeselo al profesor—: la solución **no lineal** con
# previsión perfecta, la cuña de trabajo por **población activa**, el filtro de Kehoe–Prescott
# para grandes depresiones, el costo del impago soberano y una sección de México —que no
# sustituye al mini-entregable de arriba: allí el insumo lo construyes tú.
#
# **Referencias.** Chari, Kehoe y McGrattan (2007), *Business cycle accounting*, Econometrica
# 75(3). · Christiano y Davis (2006), *Two flaws in business cycle accounting*, NBER WP 12647.
# · Brinca, Chari, Kehoe y McGrattan (2016), *Accounting for business cycles*, Handbook of
# Macroeconomics 2A. · Klein (2000), *Using the generalized Schur form…*, JEDC 24.
