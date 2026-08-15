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
# # Módulo (electivo) — ¿Es la IA un choque ΔA o ΔQ? Complementariedad capital-calificación y el premio por calificación
#
# **Curso complementario · puremacro · electivo (mazo de crecimiento y distribución)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Explicar por qué el **modelo neoclásico de un factor** (agente representativo, $Y=A\,F(K,L)$)
#    es **mudo sobre la distribución**: fija la participación del trabajo y no tiene premio por
#    calificación.
# 2. Enunciar el **hecho que no explica**: el **premio por calificación subió** desde ~1980
#    *aunque* la **oferta relativa de calificados también subió** (Katz–Murphy, Autor 2015).
# 3. **Estimar** la **complementariedad capital-calificación** de Krusell–Ohanian–Ríos-Rull–Violante
#    (2000) con `puremacro.korv_gmm` **sobre EU-KLEMS REAL** (25 economías, 1996–2021), y leer la
#    elasticidad **equipo–no-calificado** estimada frente al valor **estructural** de KORV.
# 4. Simular el **experimento IA**: si la IA abarata el equipo (precio relativo a la baja, como el
#    PIRIC del *bundle*), ¿qué pasa con el **premio por calificación** y con la **participación del
#    trabajo** bajo la elasticidad **estimada**?
# 5. Contestar el gancho: **¿la IA sustituye al calificado (cambia el signo) o lo complementa?** — y
#    ver por qué los datos agregados dejan la respuesta **abierta**.
#
# Todo corre en Python puro sobre tu **instalación local** de `puremacro`
# (`pip install puremacro`), con datos del *bundle* local: $0. La estimación KORV se
# **precomputa offline** desde el cache real de EU-KLEMS (ver §2); el notebook solo lee un CSV chico.

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
    indexada por fecha. Nunca toca la red."""
    d = pd.read_csv(DATA / f"{name}.csv")
    d.columns = ["date", name]
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[name].astype(float)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. El gancho: lo que el modelo estándar NO puede decir
#
# El corazón del curso —modelo neoclásico de crecimiento, RBC, agente representativo— escribe el
# producto como
# $$Y_t = A_t\,F(K_t,\,L_t),\qquad F\ \text{Cobb–Douglas},$$
# con **un solo tipo de trabajo** $L$. Ese modelo es una **máquina de $\Delta A$**: todo choque es
# un cambio de **productividad neutral** (TFP). Pero por construcción:
# - hay **un solo salario** y la **participación del trabajo** $s_L=1-\alpha$ es **constante**;
# - no existe **premio por calificación** $w_S/w_U$ — no hay a quién comparar.
#
# **El hecho que no explica.** Desde ~1980 el **premio por calificación** (universidad/preparatoria)
# **subió** de forma sostenida, *al mismo tiempo* que la **oferta relativa de calificados** también
# subió. Una demanda relativa con pendiente negativa —lo que da un modelo de un factor o de
# calificados y no-calificados como sustitutos simples— predice lo **contrario**: más oferta de
# calificados debería **bajar** su salario relativo. Algo desplazó la **demanda** de calificación.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### $\Delta A$ vs $\Delta Q$: dos maneras de que "llegue la IA"
# La macro moderna distingue dos choques tecnológicos (ver la contabilidad de cuñas de BCA:
# cuña de eficiencia vs cuña de inversión):
# - **$\Delta A$ — TFP neutral:** sube $A$ en $Y=A\,F(K,L)$. Levanta todo por igual; **distribución
#   neutral** (premio y $s_L$ sin cambio). El modelo de un factor solo ve esto.
# - **$\Delta Q$ — específico a la inversión:** la IA **abarata el equipo** (cae su precio relativo
#   $P_K/P_C$; sube la eficiencia $Q=1/(P_K/P_C)$, Greenwood–Hercowitz–Krusell). Si el equipo es
#   **complemento del calificado** y **sustituto del no-calificado** (KORV 2000), $\Delta Q$
#   **reescribe** el premio por calificación y la composición del ingreso — algo que el modelo de un
#   factor **no puede representar**.
#
# La pregunta del módulo: **¿la IA es $\Delta A$ o $\Delta Q$?** Necesitamos un modelo con más de un
# factor para siquiera plantearla.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Datos del bundle que el modelo de un factor no genera
# Dos hechos, ambos reales, ambos invisibles para $Y=A\,F(K,L)$:
# - **PIRIC** (`PIRIC`, FRED): precio relativo de la **inversión total** (inversión/consumo), que
#   **cae** de forma secular — el motor $\Delta Q$. El modelo de un factor no tiene $P_K/P_C$ que
#   caiga. **Cuidado con la etiqueta**: PIRIC es inversión *total* (incluye estructuras y
#   residencial); el precio del **equipo** propiamente dicho —PERIC, Fisher (2006)— cae **mucho** más
#   rápido (factor $\approx41$ desde 1947, $\approx4.8\%$ anual, como en el mazo 01/02, frente al
#   factor $\approx5$ y $\approx2\%$ anual de PIRIC), pero **no viene** en el *bundle* offline. Usamos PIRIC como **proxy conservador**: todo
#   $\Delta Q$ que midamos abajo es una **cota inferior** del choque al equipo.
# - **Participación del trabajo** (`klems_labor_share.csv`, EU-KLEMS, 6 economías): el agregado
#   ponderado **baja** — pero Cobb–Douglas la fija **constante**.

# %% slideshow={"slide_type": "fragment"}
pk = fred("PIRIC").resample("YE").mean(); pk.index = pk.index.year        # precio rel. INVERSIÓN (proxy del equipo)
caida_pk = 100 * (1 - pk.loc[2024] / pk.loc[1947])

ls = pd.read_csv(DATA / "klems_labor_share.csv")                           # EU-KLEMS, 1995-2020
glob = ls.groupby("year").apply(lambda d: np.average(d.labor_share, weights=d.va),
                                include_groups=False)                      # agregado pond. por VA

print(f"PIRIC (P_inversión/P_consumo): {pk.loc[1947]:.2f} (1947) -> {pk.loc[2024]:.2f} (2024)  "
      f"[cae {caida_pk:.0f}%]  -> proxy CONSERVADOR del choque ΔQ (el equipo, PERIC, cae mucho más)")
print(f"participación del trabajo (global pond.): {glob.iloc[0]:.3f} (1995) -> "
      f"{glob.iloc[-1]:.3f} (2020), cambio {glob.iloc[-1]-glob.iloc[0]:+.3f}")
print("Cobb–Douglas de un factor predice s_L CONSTANTE: no puede generar ninguna de las dos.")
assert pk.loc[2024] < pk.loc[1947]          # el equipo se abarata (ΔQ)
assert glob.iloc[-1] < glob.iloc[0]         # el agregado ponderado cae

# %% slideshow={"slide_type": "slide"}
fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.6))
a0.plot(pk.index, pk.values, color="0.15", lw=1.6)
a0.set_yscale("log"); a0.axhline(1.0, color="0.85", lw=0.7)
a0.set_title(f"ΔQ: precio rel. de la inversión, cae {caida_pk:.0f}% (PIRIC)", fontsize=9.5)
a0.set_xlabel("año"); a0.set_ylabel("$P_K/P_C$ (escala log)")
a1.plot(glob.index, glob.values, color="0.0", lw=2.2)
a1.set_title("Participación del trabajo (pond. VA): baja", fontsize=9.5)
a1.set_xlabel("año"); a1.set_ylabel("$s_L$")
fig.suptitle("Hechos reales del bundle que $Y=A\\,F(K,L)$ (un factor) no genera", fontsize=10.5)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. KORV (2000) estimado sobre EU-KLEMS REAL, con `puremacro.korv_gmm`
#
# Krusell, Ohanian, Ríos-Rull y Violante (2000) rompen el trabajo en **calificado** $S$ y
# **no-calificado** $U$, y anidan el **equipo** $K_e$ con el calificado:
# $$Y = A\Big[\,\mu\,U^{\nu} + (1-\mu)\big(\lambda K_e^{\rho} + (1-\lambda)S^{\rho}\big)^{\nu/\rho}\Big]^{1/\nu}.$$
# Dos elasticidades de sustitución:
# - **equipo–calificado** $\sigma_{es}=1/(1-\rho)$ (nido interno);
# - **equipo–no-calificado** $\sigma_{eu}=1/(1-\nu)$ (nido externo).
#
# *Puente de notación con el mazo 02*: ahí el exponente del nido **externo** se llama $\sigma$; aquí
# lo llamamos $\nu$ para reservar la letra $\sigma$ a las **elasticidades**. Es lo mismo: KORV estiman
# $\sigma_{\text{mazo}}\approx0.4\Rightarrow\sigma_{eu}\approx1.67$ y $\rho\approx-0.5\Rightarrow
# \sigma_{es}\approx0.67$, y la condición del mazo ($\sigma>\rho$) es exactamente
# $\sigma_{es}<\sigma_{eu}$. Ojo también: dentro de este nido, calificado y no-calificado se sustituyen
# con la elasticidad del nido **externo** ($=\sigma_{eu}$); el $\sigma_{su}$ que estima el GMM abajo es
# la elasticidad de **demanda relativa** de la FOC m1, no un parámetro extra del nido.
#
# **Complementariedad capital-calificación** $=\ \sigma_{es}<\sigma_{eu}$: el equipo es **más
# complemento** del calificado que del no-calificado, así que abaratarlo **sube** el premio. En su
# forma fuerte (KORV) $\sigma_{es}<1<\sigma_{eu}$: complementa al calificado y **sustituye** al
# no-calificado. `puremacro.korv_gmm.fit_korv_pooled` estima el sistema por **GMM** con tres
# condiciones de momento (FOC de razón de habilidades, de razón de equipo y de participación laboral)
# y reporta $\sigma_{su},\ \sigma_{eu}$, la Allen $\sigma_{es}$ y el **test J de Hansen**.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Nota de datos (ahora sí: EU-KLEMS REAL, no un panel sintético)
# Ya **no** usamos un panel calibrado. El panel de momentos `data/korv_panel_klems.csv` se
# **precomputa offline** desde el cache real de **EU-KLEMS 2023** con
# `puremacro.klems.load_klems_panel` (script `tools_curso_korv_build.py`), replicando el pipeline
# canónico de KORV: por país-año construimos horas y compensación por **calificación**
# (no-calificado = ISCED 0-2 = LOW; calificado = ISCED 3-8 = MED+HIGH), el **equipo** (inversión
# $I_e$) y su índice de precio `p_equip_index`, y el deflactor del consumo $P_C$ (OECD-QNA). De ahí
# salen las seis log-diferencias que consume el estimador
# (`dlog_ls_lu, dlog_ws_wu, dlog_ke_lu, dlog_wu_pk, dlog_lsushare, dlog_pk_pc`). El notebook solo lee
# ese CSV chico y corre `fit_korv_pooled` — **cero red, cero cache pesado**.
#
# **Honestidad metodológica.** El GMM agrupado identifica con nitidez la elasticidad
# **equipo–no-calificado** $\sigma_{eu}$ (momento m2). La FOC de calificación (m1) se
# **auto-instrumenta** y **no** trae un desplazador de oferta tipo Katz–Murphy, así que $\sigma_{su}$
# (y con ella la Allen $\sigma_{es}$) sale **mal identificada** y el test J de Hansen **rechaza** el
# sistema de 3 momentos — un resultado empírico real, no un defecto del notebook. Reportamos
# $\sigma_{eu}$ como titular y, para el nido interno equipo–calificado, tomamos el valor
# **estructural de KORV** ($\sigma_{es}\approx0.67$), que los datos agregados no fijan.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Ficha de medición (regla del curso: ningún número se publica sin ella)
# | campo | valor |
# |---|---|
# | fuente / serie | EU-KLEMS 2023 (cuentas nacionales, de capital y de trabajo), industria `TOT`; equipo = **flujo de inversión** `i_equip` y su índice `p_equip_index`; $P_C$ = deflactor del consumo OCDE-QNA (Q4 de cada año). Experimento §3: `PIRIC` (FRED, EUA, trimestral → media anual) |
# | muestra | 25 países, 1996–2021, 439 observaciones país-año (1995 se pierde al diferenciar) |
# | filtro | **ninguno**: log-diferencias anuales, no hay filtro de ciclo (no es un segundo momento) |
# | base de precios | índices de precio de EU-KLEMS en moneda nacional corriente, relativos a $P_C$ del mismo país; PIRIC es índice EUA inversión/consumo |
# | orden de operaciones | por país-año: construir razones ($L_S/L_U$, $w_S/w_U$, $K_e/L_U$, …) → logaritmo → **diferenciar dentro del país** → recortar NaN. Nunca diferenciar antes de agregar por calificación |
# | edición / vintage | cache congelado de EU-KLEMS 2023 (el CSV `korv_panel_klems.csv` del *bundle*); PIRIC del *bundle*, hasta 2024Q4 |

# %% slideshow={"slide_type": "fragment"}
from puremacro.korv_gmm import fit_korv_pooled

panel = pd.read_csv(DATA / "korv_panel_klems.csv")            # momentos KORV precomputados de EU-KLEMS
fit = fit_korv_pooled(panel)                                  # GMM de 2 pasos, pesos por país (REAL)

print("Estimación KORV sobre EU-KLEMS REAL (puremacro.korv_gmm.fit_korv_pooled):")
print(f"  panel: {fit.n_obs} obs, {fit.n_country} países, {int(panel.year.min())}-{int(panel.year.max())}")
print(f"  sigma_su (calificado-no calificado) = {fit.sigma_su:+.2f} (ee {fit.se_sigma_su:.2f})   <- SESGADA: m1 se auto-instrumenta (=MCO), no hay IV de oferta")
print(f"  sigma_eu (EQUIPO-no calificado)      = {fit.sigma_eu:+.2f} (ee {fit.se_sigma_eu:.2f})   <- TITULAR, bien identificada")
print(f"  sigma_es (EQUIPO-calificado, Allen)  = {fit.sigma_es:+.2f}                 <- depende de sigma_su -> frágil")
print(f"  Hansen J = {fit.hansen_J:.1f} (p = {fit.hansen_p:.3f})   -> el sistema de 3 momentos se RECHAZA")

# Elasticidades para el experimento IA (§3-§4):
SIG_ES     = 0.67          # equipo-calificado (nido interno): valor ESTRUCTURAL de KORV; el GMM agregado no lo fija
SIG_EU_HAT = fit.sigma_eu  # equipo-no calificado: ESTIMADA real sobre EU-KLEMS (titular)
SIG_EU_KORV = 1.67         # equipo-no calificado ESTRUCTURAL de KORV (referencia de la literatura)

print(f"\nLectura: en EU-KLEMS agregado sigma_eu={fit.sigma_eu:.2f} queda CERCA/DEBAJO de Cobb-Douglas (=1),")
print(f"  mucho más DÉBIL que el sigma_eu~1.67 estructural de KORV. Aun así se cumple la")
print(f"  complementariedad capital-calificación en sentido RELATIVO: sigma_es({SIG_ES}) < sigma_eu({fit.sigma_eu:.2f}),")
print(f"  es decir el equipo complementa MÁS al calificado que al no-calificado -> abaratarlo sube el premio.")
print(f"  (con la Allen ESTIMADA la desigualdad también se cumple: {fit.sigma_es:.2f} < {fit.sigma_eu:.2f}, pero ese")
print(f"   número hereda el sesgo de sigma_su, así que el experimento usa el 0.67 estructural.)")

assert fit.converged                       # el GMM convergió
assert np.isfinite(fit.sigma_eu) and fit.sigma_eu > 0
assert fit.n_obs > 300 and fit.n_country >= 20
assert SIG_ES < fit.sigma_eu               # complementariedad capital-calificación (relativa) en el dato REAL

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### 2b. ¿Rescata un instrumento de oferta (Katz–Murphy) a $\sigma_{su}$?
# El $\sigma_{su}\approx+0.44$ de arriba sale **mal identificado** porque `fit_korv_pooled`
# **auto-instrumenta** la FOC de calificación (m1): usa el propio salario relativo como
# "instrumento", con lo que el momento colapsa a la condición de **MCO** (la regresión agrupada de
# $\Delta\log(L_S/L_U)$ sobre $\Delta\log(w_S/w_U)$ da $\sigma_{su}=0.49$; el GMM conjunto la mueve a
# $0.44$). Con **oferta y demanda relativas moviéndose a la vez**, esa pendiente no es la de la
# demanda: mezcla las dos y queda **sesgada a la baja** frente al $\approx1.4$ de la literatura. Nota
# que el problema **no es de precisión** —el error estándar es 0.02— sino de **sesgo**: un número
# nítidamente estimado de algo que no es $\sigma_{su}$. La cura de la literatura es un **desplazador
# de oferta** tipo **Katz–Murphy**: instrumentar el salario relativo $\Delta\log(w_S/w_U)$ con el
# crecimiento de la **oferta relativa de calificados en cantidad** $\Delta\log(N_S/N_U)$.
# `puremacro.korv_gmm` expone justo esa ruta: `fit_sigma_su_pooled(iv_col='dlog_ns_nu')`, un **2SLS
# agrupado** que desmediana por país y usa la columna instrumento del panel.
#
# **Nota de datos (honestidad).** EU-KLEMS 2023 publica la calificación **solo** como participación
# del empleo (`Share_E`), de modo que la oferta en **cantidad** (personas) coincide con las **horas**
# de equilibrio: una oferta **contemporánea** sería idéntica a la variable dependiente de m1 y el 2SLS
# **degenera** (instrumento = regresando; da $\sigma_{su}\approx+4.6$, una cifra mecánica sin
# contenido económico). Por eso usamos la oferta **predeterminada** (rezagada un año), que sí es
# exógena al choque de demanda corriente —el supuesto identificador de Katz–Murphy—. Es un instrumento
# **válido pero débil**: no hay en el agregado un desplazador **demográfico** verdaderamente exógeno
# (cohortes/escolaridad) como el que usa KORV con series largas de EE. UU. Espera, por tanto, un
# error estándar grande y un punto poco informativo.

# %% slideshow={"slide_type": "fragment"}
from puremacro.korv_gmm import fit_sigma_su_pooled

iv = fit_sigma_su_pooled(panel, iv_col="dlog_ns_nu")          # 2SLS con IV de oferta (Katz–Murphy)
SIG_SU_SELF = fit.sigma_su                                    # +0.44, auto-instrumentado (m1 sin IV) = MCO
SIG_SU_KM = 1.40                                              # referencia de la LITERATURA: Katz-Murphy (1992),
#   elasticidad calificado-no calificado ~1.4. OJO: no es un parámetro propio del nido de KORV; ahí la
#   sustitución S-U la gobierna el nido EXTERNO, o sea sigma_su = sigma_eu = 1.67 por construcción.

print("Identificación de sigma_su: auto-instrumento (fit_korv_pooled) vs IV de oferta (Katz–Murphy):")
print(f"  sigma_su AUTO-INSTRUMENTADO (m1 sin IV)     = {SIG_SU_SELF:+.2f}   <- es MCO: sesgado a la baja")
print(f"  sigma_su con IV de oferta 'dlog_ns_nu'      = {iv.sigma_su:+.2f} (ee {iv.se_sigma_su:.2f})   "
      f"[2SLS, n={iv.n_obs}, {iv.n_country} países]")
print(f"  cambio con el IV                            = {iv.sigma_su - SIG_SU_SELF:+.2f}   "
      f"(se mueve AÚN MÁS cerca de 0, no hacia el ~{SIG_SU_KM:.1f} de Katz-Murphy)")
print(f"  intervalo ~95%: [{iv.sigma_su-1.96*iv.se_sigma_su:+.2f}, {iv.sigma_su+1.96*iv.se_sigma_su:+.2f}]"
      f"  -> incluye el 0 y NO incluye el {SIG_SU_KM:.1f} de Katz-Murphy")
print("\nLectura honesta: el IV de oferta quita la endogeneidad mecánica de m1 (instrumento = regresor),")
print("  pero NO rescata la")
print("  complementariedad FUERTE: el punto se mueve de +0.44 a +0.19 —o sea en la dirección CONTRARIA")
print("  a la que haría falta— y con un ee de 0.48 la estimación es simplemente NO INFORMATIVA:")
print("  indistinguible de 0 y también de 1. El instrumento (oferta rezagada) es válido pero DÉBIL.")
print("  En el agregado EU-KLEMS sigma_su SIGUE SIN IDENTIFICARSE -> por eso el nido interno")
print("  equipo-calificado se ancla en el valor ESTRUCTURAL de KORV (sigma_es=0.67), no en el dato.")

assert iv.converged and np.isfinite(iv.sigma_su)
assert abs(iv.sigma_su) < abs(SIG_SU_SELF)     # con el IV el punto queda aún más cerca de 0
assert iv.sigma_su < 1.0                        # no alcanza la sustituibilidad S-U de la literatura (~1.4)
assert iv.sigma_su + 1.96 * iv.se_sigma_su < SIG_SU_KM   # el IC95% ni siquiera cubre el 1.4
assert iv.n_obs > 300 and iv.n_country >= 20    # 2SLS sobre el panel real

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. El experimento IA: la IA abarata el equipo ($\Delta Q$)
#
# Con la elasticidad **estimada** $\sigma_{eu}$ (y el nido interno $\sigma_{es}\approx0.67$ de KORV)
# montamos el nido y hacemos que la **IA abarate el equipo**: su costo de uso cae en proporción al
# **PIRIC real** (proxy **conservador** del precio del equipo, §1), y la empresa **demanda más equipo** (demanda óptima $\partial Y/\partial K_e =
# r_e\propto P_K$). Con **oferta de trabajo fija**, leemos el **premio por calificación** $w_S/w_U$ y
# las **participaciones del ingreso**. Como referencia trazamos también la respuesta bajo el
# $\sigma_{eu}$ **estructural de KORV** (1.67). El contrafactual $\Delta A$ (TFP neutral) deja todo
# **plano**.

# %% slideshow={"slide_type": "fragment"}
from scipy.optimize import brentq          # raíz 1-D para la demanda óptima de equipo (puremacro no la cubre)

sig_es, sig_eu = SIG_ES, SIG_EU_HAT        # elasticidad ESTIMADA (nido externo) + KORV (nido interno)
LAM, MU = 0.35, 0.40                        # pesos del nido (calibración de niveles)


def korv_nest(Ke, S, U, sig_es=sig_es, sig_eu=sig_eu, lam=LAM, mu=MU, A=1.0):
    """Nido CES de KORV. Devuelve premio w_S/w_U y participaciones del ingreso.
    Productos marginales (=precios de factores) por forma cerrada; numpy puro."""
    rho = 1.0 - 1.0 / sig_es                 # equipo-calificado: sig_es<1 -> rho<0 (complementos)
    nu = 1.0 - 1.0 / sig_eu                   # no calif.-nido:    sig_eu<1 -> nu<0 (también complementos, débil)
    m = (lam * Ke**rho + (1 - lam) * S**rho) ** (1.0 / rho)
    P = mu * U**nu + (1 - mu) * m**nu
    G = A * P ** (1.0 / nu)
    dU = A * mu * U**(nu - 1) * P**(1.0/nu - 1)
    dS = A * (1 - mu) * (1 - lam) * P**(1.0/nu - 1) * m**(nu - rho) * S**(rho - 1)
    dKe = A * (1 - mu) * lam * P**(1.0/nu - 1) * m**(nu - rho) * Ke**(rho - 1)
    return dict(prem=dS / dU, share_S=dS*S/G, share_U=dU*U/G,
                share_Ke=dKe*Ke/G, share_L=(dU*U + dS*S)/G, mpk=dKe)


seg = pk[(pk.index >= 1995) & (pk.index <= 2019)]            # PIRIC real en el tramo con cobertura KLEMS
S0 = U0 = 1.0                                # oferta de trabajo normalizada y fija


def simula(sig_eu_use):
    """Ruta de equipo óptima y resultados del nido cuando el PIRIC cae, con el sig_eu dado."""
    r_bar = korv_nest(1.0, S0, U0, sig_eu=sig_eu_use)["mpk"] / seg.iloc[0]     # calibra Ke_1995 = 1
    Ke = np.array([brentq(lambda k: korv_nest(k, S0, U0, sig_eu=sig_eu_use)["mpk"] - r_bar * p, 1e-4, 1e7)
                   for p in seg.to_numpy()])
    S = [korv_nest(k, S0, U0, sig_eu=sig_eu_use) for k in Ke]
    return Ke, (np.array([x["prem"] for x in S]), np.array([x["share_S"] for x in S]),
                np.array([x["share_U"] for x in S]), np.array([x["share_L"] for x in S]))


Ke_path, (premio, share_S, share_U, share_L) = simula(SIG_EU_HAT)         # ESTIMADA (titular)
_, (premio_k, share_S_k, share_U_k, share_L_k) = simula(SIG_EU_KORV)      # KORV estructural (referencia)

d_prem = 100 * (premio[-1] / premio[0] - 1)
d_prem_k = 100 * (premio_k[-1] / premio_k[0] - 1)
print(f"IA abarata el equipo (proxy PIRIC, cae {100*(1-seg.iloc[-1]/seg.iloc[0]):.0f}% en 1995-2019): "
      f"acervo K_e {Ke_path[0]:.2f} -> {Ke_path[-1]:.2f}")
print(f"  [ESTIMADA sigma_eu={SIG_EU_HAT:.2f}]  premio w_S/w_U : {d_prem:+.1f}%   (ΔA neutral daría 0%)")
print(f"     participación CALIFICADOS   : {share_S[-1]-share_S[0]:+.3f}   (el equipo los complementa)")
print(f"     participación NO-CALIFICADOS: {share_U[-1]-share_U[0]:+.3f}   (sigma_eu<1: NO son desplazados; suben poco)")
print(f"     participación del trabajo   : {share_L[-1]-share_L[0]:+.3f}")
print(f"  [KORV sigma_eu={SIG_EU_KORV:.2f}]     premio w_S/w_U : {d_prem_k:+.1f}%   (con sustituibilidad fuerte: efecto GRANDE)")
print(f"     participación NO-CALIFICADOS: {share_U_k[-1]-share_U_k[0]:+.3f}   (aquí SÍ el equipo los SUSTITUYE)")
print(f"     participación del trabajo   : {share_L_k[-1]-share_L_k[0]:+.3f}")
print("  -> El dato agregado dice: complementariedad REAL pero DÉBIL; la versión fuerte de KORV necesita")
print("     más sustituibilidad (sigma_eu~1.67) para desplazar al no-calificado y mover mucho el premio.")
print("\nLÍMITE DEL EJERCICIO (leerlo, no taparlo): en AMBAS calibraciones la participación del TRABAJO")
print(f"  SUBE (de {share_L[0]:.3f} a {share_L[-1]:.3f}), al revés de la caída observada en §1. Es mecánico: con")
print("  sigma_es=0.67<1 el equipo y el calificado son complementos, así que cuando P_K cae el GASTO en")
print("  equipo (r_e·K_e) cae y su participación baja. KORV es un modelo del PREMIO, no de la caída de")
print("  s_L; para esa caída hacen falta otros ingredientes (márgenes, intangibles, vivienda).")
assert d_prem > 0                                    # bajo la elasticidad estimada el premio SUBE (modesto)
assert (share_S[-1]-share_S[0]) > (share_U[-1]-share_U[0])   # calificados ganan MÁS: complementariedad relativa
assert d_prem_k > d_prem                             # con el sigma_eu fuerte de KORV el efecto es mayor
assert share_L[-1] > share_L[0] and share_L_k[-1] > share_L_k[0]   # el modelo NO reproduce la caída de s_L

# %% slideshow={"slide_type": "slide"}
anios = seg.index.to_numpy()
fig, (b0, b1) = plt.subplots(1, 2, figsize=(9.6, 3.7))
b0.plot(anios, premio, color="0.10", lw=1.9, label=f"ESTIMADA $\\sigma_{{eu}}$={SIG_EU_HAT:.2f} ({d_prem:+.0f}%)")
b0.plot(anios, premio_k / premio_k[0] * premio[0], color="0.45", lw=1.5, ls=(0, (4, 2)),
        label=f"KORV $\\sigma_{{eu}}$=1.67 ({d_prem_k:+.0f}%)")
b0.axhline(premio[0], color="0.75", lw=1.0, ls=(0, (1, 1)), label="ΔA neutral (sin cambio)")
b0.set_title("Premio $w_S/w_U$ (ΔQ: equipo barato)", fontsize=9.5)
b0.set_xlabel("año"); b0.set_ylabel("$w_S/w_U$ (nivel del nido; 1995 en común)")
b0.legend(loc="upper left", fontsize=7.5)
b1.plot(anios, share_S, color="0.05", lw=1.8, label="calificados (↑ complemento)")
b1.plot(anios, share_U, color="0.45", lw=1.6, ls=(0, (4, 2)), label="no-calificados (~plano bajo est.)")
b1.plot(anios, share_L, color="0.15", lw=1.2, ls=(0, (1, 1)),
        label="trabajo agregado (SUBE: el modelo\nno reproduce la caída de $s_L$)")
b1.set_title("Participaciones del ingreso (elast. estimada)", fontsize=9.5)
b1.set_xlabel("año"); b1.set_ylabel("participación"); b1.legend(loc="center left", fontsize=8)
fig.suptitle("Experimento IA ($\\Delta Q$): con la elasticidad ESTIMADA el premio sube, pero MENOS que con KORV",
             fontsize=10.5)
plt.show()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Confrontación: un factor no puede; KORV sí (pero el dato agregado lo hace débil)
#
# Volvamos al gancho con la imagen oferta–demanda de Katz–Murphy / Autor. En el plano
# (oferta relativa $S/U$, premio $w_S/w_U$):
# - **Modelo estándar (demanda fija, equipo congelado):** una **sola** curva con pendiente
#   negativa. Más $S/U$ $\Rightarrow$ el premio **baja**.
# - **KORV con IA ($\Delta Q$):** el equipo barato **desplaza la demanda de calificación hacia
#   afuera** (complementariedad capital-calificación). La curva de 2020 (mucho equipo) queda
#   **arriba** de la de 1980 — y **cuánto** se desplaza depende de la elasticidad.
#
# El hecho observado (1980 → 2020): $S/U$ **subió** y el premio **también subió**. Eso exige un
# desplazamiento **grande** de la demanda: cabe sobre la curva desplazada con la complementariedad
# **estructural** de KORV, pero **no** sobre la curva con la complementariedad **estimada** (más
# débil) — la que sale del EU-KLEMS agregado alcanza para desplazar la demanda, mas no para vencer el
# alza de oferta.
#
# **Cómo NO leer la figura.** Lo observado es la **dirección** (ambos suben), no la altura: el nido
# está en unidades normalizadas, así que el punto "2020" se dibuja **sobre** la curva KORV — es su
# predicción, no un premio medido. Y el acervo del escenario ($K_e$ ×6) hace trabajo, así que abajo
# lo **anclamos** con la FOC del propio nido y dos precios distintos: con el **PIRIC** (inversión
# total, el proxy conservador del *bundle*) el modelo pide solo $\approx2.5\times$ y entonces **ni
# siquiera** la curva KORV alcanza el premio de 1980; con el precio del **equipo** al ritmo de PERIC
# ($\approx4.8\%$ anual) pide $\approx4.9\times$ y **sí** alcanza. Conclusión pedagógica: el veredicto
# de esta figura depende de **qué precio** llames "el del equipo" — dilo siempre que la reportes.

# %% slideshow={"slide_type": "fragment"}
SU = np.linspace(1.0, 1.4, 30)                                # oferta relativa de calificados (sube)
KE_1980, KE_2020 = 1.0, 6.0                                   # equipo congelado vs escenario ILUSTRATIVO (equipo+cómputo, 6x)
curva_std   = np.array([korv_nest(KE_1980, s, 1.0)["prem"] for s in SU])                  # demanda fija (est.)
curva_ia    = np.array([korv_nest(KE_2020, s, 1.0)["prem"] for s in SU])                  # desplazada, ESTIMADA
curva_ia_k  = np.array([korv_nest(KE_2020, s, 1.0, sig_eu=SIG_EU_KORV)["prem"] for s in SU])  # desplazada, KORV
dato_1980   = (SU[0],  curva_std[0])                          # (S/U bajo, premio bajo)
dato_2020   = (SU[-1], curva_ia_k[-1])                        # (S/U alto, premio ALTO) — sobre la curva KORV

print(f"modelo estándar (equipo fijo): premio {curva_std[0]:.3f} -> {curva_std[-1]:.3f}  (BAJA con la oferta)")
print(f"desplazamiento a S/U=1.2: ESTIMADA {100*(korv_nest(KE_2020,1.2,1)['prem']/korv_nest(KE_1980,1.2,1)['prem']-1):+.0f}%  |  "
      f"KORV {100*(korv_nest(KE_2020,1.2,1,sig_eu=SIG_EU_KORV)['prem']/korv_nest(KE_1980,1.2,1)['prem']-1):+.0f}%")
print(f"hecho a racionalizar (S/U sube Y premio sube): 1980 premio={dato_1980[1]:.3f}  ->  2020 premio={dato_2020[1]:.3f} "
      f"sobre la curva KORV  -> sube = {dato_2020[1] > dato_1980[1]}")
print(f"  (sobre la curva ESTIMADA el punto 2020 daría {curva_ia[-1]:.3f} < {dato_1980[1]:.3f}: el desplazamiento débil NO alcanza)")

# ¿De dónde sale el K_e=6x? Dos anclas, ambas vía la FOC del nido (cuánto equipo pide la empresa
# cuando su precio cae lo que cayó en el dato), para no dejar el 6x como número mágico:
r_bar80 = korv_nest(1.0, S0, U0)["mpk"] / pk.loc[1980]          # calibra K_e(1980) = 1
ke_de_precio = lambda ratio: brentq(
    lambda k: korv_nest(k, S0, U0)["mpk"] - r_bar80 * pk.loc[1980] * ratio, 1e-4, 1e9)

KE_PIRIC = ke_de_precio(pk.loc[2020] / pk.loc[1980])           # ancla CONSERVADORA: inversión total
KE_EQUIP = ke_de_precio(np.exp(-0.048 * 40))                   # ancla EQUIPO: ritmo PERIC ~4.8%/año, 40 años
prem_piric_k = korv_nest(KE_PIRIC, SU[-1], 1.0, sig_eu=SIG_EU_KORV)["prem"]
prem_equip_k = korv_nest(KE_EQUIP, SU[-1], 1.0, sig_eu=SIG_EU_KORV)["prem"]

print(f"\nDe dónde sale el K_e={KE_2020:.0f}x (no es magia, es el precio del equipo):")
print(f"  ancla PIRIC (inversión TOTAL, cae {100*(1-pk.loc[2020]/pk.loc[1980]):.0f}% 1980-2020)  -> K_e={KE_PIRIC:.1f}x  "
      f"-> curva KORV en S/U=1.4: {prem_piric_k:.3f} < {dato_1980[1]:.3f}  (NO alcanza)")
print(f"  ancla EQUIPO (ritmo PERIC ~4.8%/año, factor {np.exp(0.048*40):.0f}x en 40 años) -> K_e={KE_EQUIP:.1f}x  "
      f"-> curva KORV en S/U=1.4: {prem_equip_k:.3f} > {dato_1980[1]:.3f}  (SÍ alcanza)")
print("  -> El resultado depende de QUÉ precio llames 'el del equipo'. Con inversión total (el proxy")
print("     conservador del bundle) ni KORV racionaliza el hecho; con el precio del EQUIPO, sí, y el")
print("     escenario ilustrativo de 6x queda en el mismo orden. Dilo cuando reportes el ejercicio.")

assert curva_std[-1] < curva_std[0]           # demanda fija: solo puede predecir caída
assert np.all(curva_ia > curva_std)           # el equipo desplaza la demanda hacia ARRIBA (aun con elasticidad estimada)
assert np.all(curva_ia_k >= curva_ia)         # con la complementariedad de KORV el desplazamiento es MAYOR
assert dato_2020[1] > dato_1980[1]            # SOLO la curva KORV (con el K_e ilustrativo) sube el premio
assert prem_piric_k < dato_1980[1] < prem_equip_k   # el ancla de precios decide: inversión total NO, equipo SÍ

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.8, 4.2))
ax.plot(SU, curva_std, color="0.60", lw=1.6, ls=(0, (4, 2)),
        label="Demanda fija — modelo estándar (equipo congelado)")
ax.plot(SU, curva_ia, color="0.40", lw=1.6, ls=(0, (2, 1.5)),
        label=f"Demanda desplazada — elasticidad ESTIMADA ($\\sigma_{{eu}}$={SIG_EU_HAT:.2f})")
ax.plot(SU, curva_ia_k, color="0.08", lw=2.0,
        label="Demanda desplazada — KORV estructural ($\\sigma_{eu}$=1.67), $K_e$ ×6 ilustrativo")
ax.annotate("", xy=dato_2020, xytext=dato_1980,
            arrowprops=dict(arrowstyle="-|>", color="0.0", lw=1.8))
ax.scatter(*dato_1980, color="0.0", zorder=5); ax.scatter(*dato_2020, color="0.0", zorder=5)
ax.annotate("1980", dato_1980, textcoords="offset points", xytext=(6, -12), fontsize=9)
ax.annotate("2020 (dirección observada;\naltura = predicción KORV)", dato_2020,
            textcoords="offset points", xytext=(-8, -34), ha="right", fontsize=8)
ax.set_xlabel("oferta relativa de calificados  $S/U$")
ax.set_ylabel("premio por calificación  $w_S/w_U$")
ax.set_title("Oferta sube y premio sube: solo cabe con un desplazamiento GRANDE de la demanda (KORV, $\\Delta Q$)")
ax.legend(loc="upper right", fontsize=7.5)
plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Entonces, ¿la IA sustituye o complementa al calificado?
# La descomposición de la §3 lo contesta **con el signo**: bajo la elasticidad **estimada**, el equipo
# barato (IA) **complementa** al calificado (su participación **sube**) más de lo que toca al
# no-calificado, así que el **premio por calificación sube** —aunque de forma **modesta**—. La versión
# **fuerte** de KORV ($\sigma_{eu}>1$) además **sustituye** al no-calificado (su participación baja) y
# mueve el premio mucho más. En ambos casos lo interesante de $\Delta Q$ ocurre **dentro** del trabajo,
# en la desigualdad: un modelo de un factor, que solo mira $s_L$, concluiría que "no pasó nada" con la
# distribución.
#
# **Y hay que decir el reverso**, que la salida de §3 enseña sin pudor: en esta calibración el propio
# modelo predice que $s_L$ **sube** ~4 puntos cuando el equipo se abarata —con $\sigma_{es}<1$ el gasto
# en equipo cae al caer su precio—, justo **al revés** del hecho de §1. KORV está construido para el
# **premio**, no para la caída de la participación del trabajo; esa caída pide otros ingredientes
# (márgenes, intangibles, vivienda). Un modelo puede acertar en una dimensión y fallar en otra: hay
# que decir en cuál se le está creyendo.
#
# **Lo que el dato REAL agrega.** Nuestra estimación sobre EU-KLEMS dice que la complementariedad
# capital-calificación **existe pero es débil** en datos agregados ($\sigma_{eu}\approx0.7$–$0.9$,
# cerca de Cobb–Douglas), y el sistema se **rechaza** — por eso la pregunta "¿$\Delta A$ o $\Delta Q$?"
# **no** está cerrada: KORV necesitó identificación más rica (series de EE. UU. + instrumentos) para
# llegar a $\sigma_{eu}\approx1.67$.
#
# **Matiz de frontera (Acemoglu–Restrepo 2022):** si la IA **automatiza tareas** que antes hacían los
# calificados —y no solo abarata equipo complementario— podría **invertir el signo** ($\sigma_{es}>
# \sigma_{eu}$) en ciertas ocupaciones. Cuál domina es empírico y aún abierto: por eso se **mide**.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Laboratorio
# Lo que sigue **no** viaja en tu ZIP: el CSV de momentos sí, pero el script de construcción y el
# cache de EU-KLEMS viven en el repositorio del paquete y pesan. Léelo como el mapa de lo que hay
# detrás del CSV (y como guion para el proyecto, si consigues el cache).
# - **`tools_curso_korv_build.py`** (raíz del repo de `puremacro`) — el script que precomputa
#   `data/korv_panel_klems.csv` desde el cache real de EU-KLEMS con `load_klems_panel`; cambia
#   `equip_col='k_equip'` (acervo; en la corrida de referencia del script da $\sigma_{eu}\approx0.70$)
#   vs `'i_equip'` (inversión, titular $\approx0.89$) para ver la robustez.
# - **`puremacro.korv_gmm.fit_korv_pooled_2moments`** — quita la FOC de participación (m3) para aislar
#   si ese momento arrastra el rechazo de Hansen; **`fit_korv_pooled_usercost`** reemplaza $P_K$ por el
#   costo de uso Hall–Jorgenson (`build_usercost_column`, $r=0.04$, $\delta_e=0.13$).
# - **`fit_sigma_su_pooled(iv_col='dlog_ns_nu')`** — la ruta con **instrumento de oferta**
#   (Katz–Murphy) para $\sigma_{su}$ (§2b): quita el sesgo de auto-instrumentación de m1 pero, con
#   el agregado EU-KLEMS, $\sigma_{su}$ **sigue sin identificarse** (cerca de 0, con ee grande, lejos
#   del ~1.4 de Katz–Murphy). Esta ruta **sí** la puedes correr con el CSV del ZIP: prueba a cambiar
#   el rezago del desplazador de oferta (columna `dlog_ns_nu`) y observa cuánto se mueve. Para probar
#   otra definición de equipo hay que editar `build_moments` y pasarle `equip_def='ict'` a
#   `load_klems_panel` (el script no expone ese argumento), lo que ya exige el cache.
# - **`puremacro.klems.load_klems_panel(include_investment=True)`** — el micro-panel EU-KLEMS por
#   habilidad del que salen los momentos reales.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 6. Preguntas para pensar
# 1. **$\Delta A$ vs $\Delta Q$.** Un choque $\Delta A$ (TFP neutral) y uno $\Delta Q$ (equipo
#    barato) pueden subir el PIB **lo mismo**. ¿Qué **observable de distribución** los distingue, y
#    por qué el modelo de un factor no puede usarlo?
# 2. **El signo de $\sigma_{es}$ vs $\sigma_{eu}$.** Si mañana midieras $\sigma_{es}>\sigma_{eu}$
#    (equipo *más* sustituto del calificado que del no-calificado), ¿en qué dirección se movería el
#    premio cuando cae $P_K$? ¿Qué diría eso sobre "la IA viene por los profesionistas"?
# 3. **Estimado vs estructural.** El $\sigma_{eu}$ que estimamos ($\approx0.9$) está lejos del
#    $1.67$ de KORV. Da dos razones (identificación, muestra, medición) por las que el GMM agregado
#    subestima la sustituibilidad equipo–no-calificado, y qué harías para acercarte al valor de KORV.
# 4. **Lo que el modelo NO da.** En la simulación de §3 la participación del trabajo **sube** ~4
#    puntos cuando el equipo se abarata, al revés del hecho de §1. Explica el mecanismo (¿qué papel
#    juega $\sigma_{es}<1$?) y di qué **no** deberías concluir de este ejercicio sobre la caída de
#    $s_L$.

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Soluciones (esquema)
# 1. Los distingue el **premio por calificación** (y la composición del ingreso laboral): $\Delta A$
#    lo deja **igual**, $\Delta Q$ lo **mueve** vía complementariedad capital-calificación. El modelo
#    de un factor no tiene $w_S/w_U$ —fija $s_L$— así que ambos choques le lucen idénticos.
# 2. Con $\sigma_{es}>\sigma_{eu}$ el equipo se vuelve *más* sustituto del calificado: al caer $P_K$ y
#    subir $K_e$, el producto marginal del **calificado** cae **relativo** al del no-calificado y el
#    premio **bajaría**. Sería el escenario "la IA automatiza tareas cognitivas" de Acemoglu–Restrepo.
# 3. (i) **Identificación**: `fit_korv_pooled` auto-instrumenta m1; sin un desplazador de oferta
#    (Katz–Murphy, `dlog_ns_nu`) $\sigma_{su}$ queda sesgada y contamina el sistema. (ii) **Muestra**:
#    EU-KLEMS agregado 1995–2021 tiene poca variación de precios de equipo por país; KORV usa series
#    largas de EE. UU. (iii) **Medición**: acervo vs inversión, costo de uso Hall–Jorgenson, definición
#    de equipo (ICT). Usar `fit_korv_pooled_usercost` / `equip_def='ict'` y el IV de oferta. Añade
#    una cuarta: el precio que usamos en el experimento es el de la **inversión total** (PIRIC), no el
#    del **equipo** (PERIC), lo que **atenúa** el $\Delta Q$ medido.
# 4. Con $\sigma_{es}=0.67<1$, equipo y calificado son **complementos**: cuando $P_K$ cae, el **gasto**
#    en equipo $r_e K_e$ cae (la cantidad sube menos que proporcionalmente al precio), su participación
#    baja y por residuo $s_L$ **sube**. No debes concluir que "el $\Delta Q$ explica la caída de $s_L$":
#    en esta calibración predice lo contrario. KORV es un modelo del **premio**; la caída de $s_L$ pide
#    márgenes, intangibles o vivienda, fuera de este nido.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 7. Explora con IA
# Prueba con el tutor sin conexión (o cualquier asistente de IA):
# - "En una frase: ¿por qué la complementariedad capital-calificación hace que abaratar el equipo
#   *suba* el premio por calificación?"
# - "¿Cómo distinguirías empíricamente un choque de IA tipo $\Delta A$ (TFP) de uno tipo $\Delta Q$
#   (equipo barato)?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase: ¿por qué, con complementariedad capital-calificación (equipo y "
            "calificado complementos, sigma<1), abaratar el equipo SUBE el premio por calificación?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** El modelo neoclásico de **un factor** ($Y=A\,F(K,L)$) es una máquina de $\Delta A$:
# fija la participación del trabajo y no tiene premio por calificación, así que **no puede** explicar
# que el premio **subiera** mientras subía la oferta de calificados, ni la caída de $s_L$, ni el
# colapso del precio relativo de la inversión (PIRIC). **KORV (2000)** sí abre la puerta —al menos
# para el premio; la caída de $s_L$ **no** la reproduce, ver §4—: **estimamos** con `puremacro.korv_gmm`
# sobre **EU-KLEMS REAL** la elasticidad equipo–no-calificado ($\sigma_{eu}\approx0.9$, más débil que
# el $1.67$ estructural de KORV, con la Allen $\sigma_{es}$ mal identificada y Hansen rechazando),
# simulamos la **IA como $\Delta Q$** (equipo barato) y vimos el premio **subir** —modesto bajo la
# elasticidad estimada, grande bajo la de KORV— con la composición del ingreso reordenándose, y
# confrontamos: el hecho observado solo cabe con un **desplazamiento grande** de la demanda de
# calificación —la firma de $\Delta Q$, invisible para el agente representativo. **¿$\Delta A$ o
# $\Delta Q$?** Con un solo factor ni siquiera puedes preguntarlo; con datos agregados, la respuesta
# aún se **mide**.
#
# **Referencias.** Krusell, Ohanian, Ríos-Rull y Violante (2000, *Econometrica*); Autor (2015,
# *JEP*); Acemoglu y Restrepo (2022, *Econometrica*).
