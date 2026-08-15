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
# # Módulo 8 — Desempleo, vacantes, flujos y el efecto de la IA
#
# **Curso complementario · puremacro · mazo Slides08 — mercados no competitivos (semanas 15–16)**
#
# ### Objetivos de aprendizaje
# Al terminar esta lección podrás:
# 1. Leer la **curva de Beveridge** (vacantes contra desempleo) y distinguir
#    movimientos *a lo largo* de la curva (ciclo) de **desplazamientos** de la
#    curva (emparejamiento), incluido el corrimiento hacia afuera tras el COVID.
# 2. Reproducir el **puzzle de Shimer**: la tensión del mercado laboral
#    $\theta=V/U$ es ~25 veces más volátil que la productividad $A$ en la
#    muestra 2001–hoy (Shimer 2005, con su muestra y su filtro, obtuvo ~20) —
#    algo que el modelo de búsqueda estándar no genera.
# 3. Construir la **matriz de transición** de estados laborales E/U/N a partir
#    de los flujos brutos de la CPS, y leer la tasa de hallazgo de empleo y la
#    de separación; ojear las transiciones **F/I/U/N** de la ENOE para México.
# 4. Montar un **mini event-study** de exposición a la IA con proyecciones
#    locales de panel, **con prueba de pre-tendencias** — y entender por qué el
#    resultado exige **cautela**.
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


# %% slideshow={"slide_type": "skip"}
# Lector uniforme para los CSV congelados del bundle (formato FRED:
# observation_date, VALOR). Siempre desde disco — nunca por red — para que
# corra idéntico en cualquier máquina y sin conexión.
def load_series(name: str) -> pd.Series:
    df = pd.read_csv(DATA / f"{name}.csv")
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[name]


# %% [markdown] slideshow={"slide_type": "slide"}
# ## 1. La curva de Beveridge
#
# El mercado laboral empareja **buscadores** (desempleados $u$) con **puestos
# vacantes** ($v$) mediante una tecnología de emparejamiento — en la notación
# del mazo Slides08,
# $$ m_t = A_{mt}\, u_t^{\phi} v_t^{1-\phi}, \qquad \theta \equiv \frac{v}{u}, $$
# con $A_m$ la **eficiencia del emparejamiento** y $\phi\in[0.5,0.7]$ la
# elasticidad respecto de los buscadores (rango de Petrongolo–Pissarides).
# En estado estacionario los flujos de entrada al desempleo igualan las
# salidas, $s(1-u)=p(\theta)u$, de donde $u^{*}=s/[s+p(\theta)]$: un lugar
# geométrico **decreciente** entre $u$ y $v$, la **curva de Beveridge**. Los
# auges suben hacia la izquierda (muchas vacantes, poco desempleo); las
# recesiones bajan hacia la derecha.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### A lo largo vs desplazamientos
# - **A lo largo** de la curva: variación cíclica de la demanda. La economía
#   recorre el lugar geométrico en sentido antihorario tras un choque (las
#   vacantes se ajustan en semanas; el estanque de desempleo se drena despacio).
# - **Desplazamientos** de la curva: peor **emparejamiento** (menor $A_m$),
#   más desajuste sectorial o geográfico, o más separaciones $s$ mueven toda la
#   curva **hacia afuera** — más desempleo para la misma tasa de vacantes.
#
# El episodio de EE.UU. tras 2021 es el experimento natural más limpio: las
# vacantes cayeron desde máximos históricos con una subida del desempleo de
# menos de un punto porcentual.

# %% slideshow={"slide_type": "fragment"}
V = load_series("JTSJOL")          # vacantes JOLTS (miles), mensual SA, desde 2000-12
u = load_series("UNRATE")          # tasa de desempleo (%), mensual SA
bev = pd.concat([V.rename("V"), u.rename("u")], axis=1, sort=True).dropna()
bev = bev.loc["2001-01-01":]       # el mazo grafica de 2001 a hoy; JOLTS empieza en 2000-12
bev["v_mill"] = bev["V"] / 1000.0  # vacantes en millones
print(f"muestra Beveridge: {bev.index.min().date()} .. {bev.index.max().date()} "
      f"({len(bev)} meses)")

# Etiquetas y ventanas coinciden (antes la nube "2001-2007" empezaba en
# 2000-12 y la de "2008-2019", en 2007-12).
ERAS = [  # (etiqueta, desde, hasta, gris, marcador)
    ("2001-2007", "2001-01-01", "2007-12-31", "0.00", "o"),
    ("2008-2019", "2008-01-01", "2019-12-31", "0.55", "s"),
    ("COVID 2020", "2020-01-01", "2020-12-31", "0.30", "^"),
    ("2021+",      "2021-01-01", "2099-01-01", "0.72", "D"),
]

# %% slideshow={"slide_type": "slide"}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax = axes[0]
for name, lo, hi, c, mk in ERAS:
    m = (bev.index >= lo) & (bev.index <= hi)
    ax.plot(bev.loc[m, "u"], bev.loc[m, "v_mill"], "-", color=c, lw=0.8,
            marker=mk, ms=3.0, alpha=0.85, label=name)
ax.set_xlabel("tasa de desempleo, %"); ax.set_ylabel("vacantes, millones")
ax.set_title("Curva de Beveridge de EE.UU. (JOLTS)")
ax.legend(fontsize=8)

ax = axes[1]
bev["theta"] = bev["V"] / bev["u"]     # tensión proxy: vacantes por punto de u
ax.plot(bev.index, bev["theta"], color="0.15", lw=1.4)
ax.axhline(bev["theta"].loc["2001":"2019"].mean(), color="0.6", lw=0.8, ls=":",
           label="media 2001-2019")
ax.set_ylabel(r"tensión proxy  $\theta \propto V/U$")
ax.set_title("Tensión del mercado laboral")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

peak = bev["theta"].idxmax()
mult = bev["theta"].max() / bev["theta"].loc["2001":"2019"].mean()
print(f"tensión máxima en {peak.date()}: V = {bev.loc[peak, 'v_mill']:.1f} millones "
      f"con u = {bev.loc[peak, 'u']:.1f}%  ->  {mult:.1f} veces la media 2001-2019")
print(f"último dato ({bev.index[-1].date()}): V = {bev['v_mill'].iloc[-1]:.1f} millones, "
      f"u = {bev['u'].iloc[-1]:.1f}%")
assert mult > 3.0, "el pico de tensión de 2022 debe estar muy por encima de la media 2001-2019"

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Leyendo el gráfico.** La nube 2008-2019 traza un circuito antihorario: la
# recesión resbala abajo-derecha sobre una curva estable y la recuperación
# vuelve por un sendero visiblemente *exterior* (el desplazamiento de 2010-13
# que lanzó una literatura de desajuste). El COVID es el tramo casi
# **horizontal**: el desempleo salta de 4.4% a 14.8% **en un mes** mientras las
# vacantes caen mucho menos y muy brevemente (de 7.0 a 4.6 millones entre
# febrero y abril de 2020, y de vuelta en 6.4 en agosto) — el estanque de
# desempleo se llena de golpe y el eje que se mueve es el horizontal. El tramo
# 2021+ es el inverso y el histórico: la caída de vacantes desde el pico de
# 12.3 millones (marzo de 2022) hasta ~7.6 millones se produce con una subida
# del desempleo de menos de un punto (3.4%-4.3%), un descenso casi **vertical**
# que ninguno de los dos bandos del debate de 2022 (Blanchard–Domash–Summers
# contra Figura–Waller) predijo del todo. **Fetchers del paquete** para ir más lejos:
# `puremacro.fetch.fetch_jolts` (EE.UU., panel JOLTS completo) y
# `puremacro.fetch.fetch_eurostat_vacancies`
# (Europa) — se usan en `notebooks/18_beveridge_curve_es`, un cuaderno del paquete
# fuera del temario del curso (lectura opcional, y sí sale a la red).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 2. El puzzle de Shimer
#
# En el modelo de búsqueda-emparejamiento de caballo de batalla (Pissarides;
# Mortensen-Pissarides), un choque de productividad $A$ mueve la tensión
# $\theta$ casi uno a uno: la **elasticidad** de $\theta$ respecto de $A$ es
# cercana a 1. Pero en los datos $\theta$ es **enormemente** más volátil que
# $A$. Shimer (2005) lo cuantificó: la desviación estándar de $\theta$ es unas
# **~20 veces** la de la productividad laboral (su muestra 1951-2003, HP con
# $\lambda=10^{5}$). El modelo estándar, calibrado a lo razonable, genera un
# cociente de **1-2** — de ahí el *puzzle*. Abajo lo rehacemos con la muestra
# JOLTS (2001-hoy) y el $\lambda=1600$ del curso: el cociente sale ~25, la
# misma cifra de la lámina del mazo.

# %% slideshow={"slide_type": "fragment"}
from puremacro.data import hp_filter   # (cycle, trend) = hp_filter(y, lamb=1600)

prod = load_series("OPHNFB")           # producto por hora (nonfarm business), trimestral
theta_m = (V / u).dropna().loc["2001-01-01":]    # tensión proxy mensual ~ V/U
theta_q = theta_m.resample("QS").mean()          # a trimestral (media)
prod_q  = prod.resample("QS").mean().dropna()
idx = theta_q.index.intersection(prod_q.index)
log_theta = np.log(theta_q.loc[idx].to_numpy())
log_A     = np.log(prod_q.loc[idx].to_numpy())

c_theta, _ = hp_filter(log_theta)      # ciclos HP (log-desviaciones), lambda=1600
c_A, _     = hp_filter(log_A)
c_theta = np.asarray(c_theta); c_A = np.asarray(c_A)

sd_theta, sd_A = c_theta.std(), c_A.std()
ratio = sd_theta / sd_A
print(f"muestra: {idx.min().date()} .. {idx.max().date()} ({len(idx)} trimestres)")
print(f"sd( ciclo log theta ) = {sd_theta:.3f}")
print(f"sd( ciclo log A )     = {sd_A:.3f}")
print(f"cociente  sd(theta)/sd(A) = {ratio:.1f}")
# Sensibilidad al filtro: Shimer usó lambda = 1e5, el curso usa 1600.
c_t5, _ = hp_filter(log_theta, lamb=1e5)
c_A5, _ = hp_filter(log_A, lamb=1e5)
print(f"  con el lambda de Shimer (1e5): {np.std(c_t5) / np.std(c_A5):.1f}  "
      f"(el orden de magnitud no depende del filtro)")
print("  el modelo con Nash flexible genera un cociente de 1-2;")
print("  Shimer (2005), con su muestra 1951-2003, lo situó en ~20  ->  el puzzle.")
assert ratio > 5.0, "el cociente de amplitudes debe ser mucho mayor que 1"

# %% slideshow={"slide_type": "slide"}
qd = pd.PeriodIndex(pd.DatetimeIndex(idx), freq="Q").to_timestamp()
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.plot(qd, 100 * c_theta, color="0.10", lw=1.5, label=r"ciclo de $\log\theta$ (izq.)")
ax.set_ylabel(r"$\log\theta$, % desv.", color="0.10")
ax.axhline(0, color="0.85", lw=0.6)
ax2 = ax.twinx()
ax2.plot(qd, 100 * c_A, color="0.55", lw=1.5, ls=(0, (4, 2)),
         label=r"ciclo de $\log A$ (der.)")
ax2.set_ylabel(r"$\log A$, % desv.", color="0.55")
ax2.grid(False)
ax.set_title(f"Puzzle de Shimer: la tensión es ~{ratio:.0f}x más volátil que la productividad")
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Nota de método.** Usamos una *proxy* de tensión $\theta = V/U$ formada con
# vacantes JOLTS y la **tasa** de desempleo, no el nivel (el mazo pide
# UNEMPLOY): la diferencia entre ambas es $\log$ de la fuerza laboral, casi toda
# baja frecuencia, así que el **ciclo** HP la borra — con el nivel (UNEMPLOY) el
# cociente sale 24.9 en vez de 25.2, dentro del redondeo. Dos advertencias honestas: (i)
# Shimer filtró con $\lambda=10^{5}$ y nosotros con el $\lambda=1600$ del curso
# (la línea impresa arriba muestra que el cociente se mueve de ~25 a ~24: el
# resultado no vive del filtro); (ii) la muestra es 2001-hoy, no la de Shimer,
# y contiene el COVID, que infla ambas volatilidades. Las resoluciones
# propuestas — rigidez salarial (Hall 2005), calibración del excedente
# fundamental $A-b$ (Hagedorn-Manovskii 2008), ofertas alternantes
# (Hall-Milgrom 2008) — buscan amplificar la respuesta de $\theta$ a un choque
# pequeño de $A$.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 3. Flujos: la matriz de transición E/U/N
#
# La tasa de desempleo es un **acervo**; lo que la mueve son **flujos** entre
# tres estados: empleado (E), desempleado (U) y fuera de la fuerza laboral (N).
# La CPS publica los conteos brutos de transición mes a mes. Con ellos armamos
# la matriz $3\times3$ de probabilidades $p_{ij}=$ Prob(estado $j$ en $t$ \|
# estado $i$ en $t-1$), normalizando cada flujo por el acervo de origen.
# `puremacro.labor_flows.transitions_from_cps_flows(flows, stocks)` hace
# exactamente eso.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Ensamblando las entradas desde el bundle
# El bundle trae las 8 series de flujo de la CPS (`LNS17*`, en miles). La BLS no
# publica la novena celda (**NN**, permanecer fuera de la fuerza laboral), y la
# función pide además los **acervos** de origen E/U/N: de los tres, el freeze
# solo trae `UNEMPLOY`; `CE16OV` y `LNS15000000` no viajan en el bundle. Los
# reconstruimos con los propios flujos:
# - $E_t = EE_t + UE_t + NE_t$ y $U_t = EU_t + UU_t + NU_t$ — suma de **todas**
#   las entradas a cada estado, así que son el acervo de fin de mes y son
#   **exactos** salvo el error de margen de la BLS (la función retrasa un mes lo
#   que le pasamos, de modo que el denominador es el acervo de inicio de mes);
# - $N_t = N_{t-1} + (EN_t+UN_t) - (NE_t+NU_t)$ y $NN_t = N_{t-1}-NE_t-NU_t$.
#
# **Cuidado con la fila N.** Esa recursión **no** es contabilidad exacta: ignora
# la entrada neta de población en edad de trabajar (nadie "fluye" a $N$ al
# cumplir 16 años ni al inmigrar), así que el acervo reconstruido se desvía cada
# vez más del real — la celda de abajo imprime cuánto. Por eso aquí **solo
# leemos las filas E y U**, que sí son exactas; la matriz completa, con acervos
# de verdad, es lo que arma el mazo con `CE16OV`/`UNEMPLOY`/`LNS15000000`.
# Arrancamos en **1994** porque ahí entra el rediseño de la CPS.

# %% slideshow={"slide_type": "fragment"}
from puremacro.labor_flows import transitions_from_cps_flows, CPS_FLOW_SERIES

fl = pd.DataFrame({k: load_series(v) for k, v in CPS_FLOW_SERIES.items()}).dropna()
fl = fl.loc["1994-01-01":]
E = fl["EE"] + fl["UE"] + fl["NE"]                 # acervos por suma de entradas
U = fl["EU"] + fl["UU"] + fl["NU"]
N = 65000.0 + ((fl["EN"] + fl["UN"]) - (fl["NE"] + fl["NU"])).cumsum()
fl["NN"] = N.shift(1) - fl["NE"] - fl["NU"]        # permanencia en N (residual)
stocks = pd.DataFrame({"E": E, "U": U, "N": N}).dropna()
common = fl.dropna().index.intersection(stocks.index)
panel = transitions_from_cps_flows(fl.loc[common], stocks.loc[common])
tr = panel.monthly

f_rate = tr["p_UE"]     # tasa de hallazgo de empleo (U -> E)
s_rate = tr["p_EU"]     # tasa de separación a desempleo (E -> U)
print(f"muestra de flujos: {tr.index.min().date()} .. {tr.index.max().date()} "
      f"({len(tr)} meses)")
print(f"tasa de hallazgo de empleo  p_UE: media {f_rate.mean():.2f}, "
      f"rango [{f_rate.min():.2f}, {f_rate.max():.2f}]")
print(f"tasa de separación          p_EU: media {s_rate.mean():.3f}, "
      f"pico {s_rate.max():.3f} en {s_rate.idxmax().date()} (choque COVID)")
print(f"validación — u implícita = U/(E+U): media {(U/(E+U)*100).mean():.1f}%  "
      f"(compara con UNRATE ~ {u.loc[E.index.min():E.index.max()].mean():.1f}%)")
print(f"AVISO fila N — acervo reconstruido: {N.iloc[0]:,.0f} miles en "
      f"{N.index[0].date()} -> {N.iloc[-1]:,.0f} en {N.index[-1].date()}; "
      "la inactividad real SUBIÓ por encima de 100,000 miles en ese periodo.")
print("  la recursión no ve la entrada neta de población: p_NE/p_NU/p_NN NO son "
      "utilizables. Las filas E y U (las que leemos) sí lo son.")
assert 0.15 < f_rate.mean() < 0.40         # hallazgo mensual ~25-30% en la literatura
assert s_rate.max() > 0.05                 # el pico COVID dispara las separaciones

# %% slideshow={"slide_type": "slide"}
cols = _nbstyle.palette(2)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
ax = axes[0]
ax.plot(f_rate.index, f_rate, color=cols[0], lw=1.2, label=r"hallazgo $p_{UE}$")
ax.plot(s_rate.index, s_rate * 5, color="0.55", lw=1.2, ls=(0, (4, 2)),
        label=r"separación $p_{EU}\times 5$")
ax.set_title("Flujos CPS de EE.UU. (probabilidades mensuales)")
ax.set_ylabel("probabilidad"); ax.legend(fontsize=8)

# --- México: transiciones F/I/U/N de la ENOE (parquets del bundle) ---
enoe_tr = pd.read_parquet(DATA / "enoe_transitions_quarterly_observed.parquet")
enoe_st = pd.read_parquet(DATA / "enoe_stocks_monthly.parquet")
share = (enoe_st.div(enoe_st.sum(axis=1), axis=0)).mean()   # composición media
ax = axes[1]
order = ["F", "I", "U", "N"]
ax.bar(order, [share[s] for s in order], color=["0.15", "0.45", "0.70", "0.85"])
ax.set_title("México (ENOE): composición media de la\npoblación en edad de trabajar")
ax.set_ylabel("participación")
for i, s in enumerate(order):
    ax.text(i, share[s] + 0.01, f"{share[s]*100:.0f}%", ha="center", fontsize=8)
plt.tight_layout(); plt.show()

print(f"ENOE — panel rotatorio {enoe_tr.index.min().date()} .. "
      f"{enoe_tr.index.max().date()} ({len(enoe_tr)} meses de referencia, "
      "COVID enmascarado)")
print("  probabilidad media de PERMANECER en cada estado (diagonal trimestral):")
for s in order:
    print(f"    {s} -> {s}: {enoe_tr[f'p_{s}{s}'].mean():.2f}")
print(f"  formalización  p_IF = {enoe_tr['p_IF'].mean():.2f}   contra "
      f"informalización  p_FI = {enoe_tr['p_FI'].mean():.2f}")
print(f"  separaciones: s_F = p_FU+p_FN = "
      f"{(enoe_tr['p_FU'] + enoe_tr['p_FN']).mean():.2f}   contra   s_I = p_IU+p_IN = "
      f"{(enoe_tr['p_IU'] + enoe_tr['p_IN']).mean():.2f}")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lo que dicen los flujos.** En EE.UU. la tasa de **hallazgo** ($p_{UE}\approx
# 0.26$ mensual de media) es alta y muy procíclica; la de **separación**
# ($p_{EU}\approx 0.013$) es baja pero salta en las recesiones — el pico de
# abril de 2020 ($0.11$, más de ocho veces la media) es el evento más violento de la
# serie. Ojo con la lectura estándar: en EE.UU. lo que domina el ciclo del
# desempleo es la caída del **hallazgo**, no el salto de las separaciones, que
# es grande pero brevísimo. En México el estado **I (informal)** es enorme
# (33% de la población en edad de trabajar, contra 21% de empleo formal), y
# formalizarse es **menos** probable que informalizarse ($p_{IF}=0.10$ contra
# $p_{FI}=0.15$): la informalidad es **persistente**, no un trampolín. El empleo
# informal es además unas tres veces más frágil que el formal
# ($s_I\approx 0.19$ contra $s_F\approx 0.06$), las mismas cifras de la lámina
# de la matriz $4\times4$ del mazo. La matriz F/I/U/N que
# graficamos viene precalculada, pero se construye desde el microdato con
# `puremacro.labor_flows_enoe` (`transitions_from_enoe`, `load_enoe_quarter`,
# `quarterly_transitions_from_pairs`), que enlaza panel-personas entre
# trimestres consecutivos.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 4. Escaparate: ¿deja huella la IA en el desempleo estatal?
#
# La pregunta del momento: ¿los estados más **expuestos a la IA** vieron subir
# su desempleo tras el salto de la IA generativa (ChatGPT, nov-2022)? Montamos
# un **mini event-study de panel** con la receta del mazo. La exposición
# `STATE_AI_EXPOSURE_2019[estado]` es la participación del empleo estatal en
# ocupaciones de **cómputo y matemáticas** (SOC 15-0000, OES 2019): una *proxy*
# documentada del índice AIOE de Felten–Raj–Seamans (2021), transversal y fija
# en el tiempo — los valores del diccionario son aproximaciones que respetan el
# ordenamiento de la tabla del BLS, no la tabla oficial. El choque es un
# **impulso** en la fecha del evento, escalado por la exposición estandarizada;
# `puremacro.lp.panel_lp` estima
# $$ u^{(c)}_{t+h}-u^{(c)}_{t-1} = \mu_c + \tau_t + \beta_h\,\text{shock}^{(c)}_t + \dots $$
# con efectos fijos de estado ($\mu_c$) y de tiempo ($\tau_t$): $\tau_t$ absorbe
# el efecto nacional de la IA, así que $\beta_h$ mide la respuesta **diferencial**
# de los estados más expuestos.
#
# Dos decisiones de diseño, ambas del mazo: la ventana arranca en **2021m1**
# (dejar dentro 2019-2020 metería el rebote del COVID en las pre-tendencias) y
# estimamos también **horizontes negativos** como prueba de pre-tendencias. Con
# `n_lags=3` los horizontes $h=-1,-2,-3$ son **degenerados** — el lado izquierdo
# $u_{t+h}-u_{t-1}$ es una combinación exacta de los rezagos que ya están del
# lado derecho, así que el ajuste es perfecto y el error estándar, cero —; por
# eso las pre-tendencias van de $h=-6$ a $h=-4$.

# %% slideshow={"slide_type": "fragment"}
import glob, os
from puremacro.lp import panel_lp
from puremacro.fetch.state_industry_panel import STATE_AI_EXPOSURE_2019 as AIX

rows = []
for f in sorted(glob.glob(str(DATA / "??UR.csv"))):     # 50 estados + DC
    code = os.path.basename(f)[:2]
    if code not in AIX:
        continue
    s = load_series(os.path.basename(f)[:-4]).rename("ur").to_frame()
    s["code"] = code
    rows.append(s.reset_index()[["code", "date", "ur"]])
pan = pd.concat(rows, ignore_index=True)
pan = pan[pan["date"] >= "2021-01-01"].copy()   # ventana del mazo: fuera 2019-2020

aix = pd.Series(AIX)
aix_z = (pan["code"].map(AIX) - aix.mean()) / aix.std()   # exposición estandarizada (+1 sd)
EVENT = pd.Timestamp("2022-11-01")                        # lanzamiento de ChatGPT
pan["shock"] = np.where(pan["date"] == EVENT, aix_z, 0.0)
pan = pan.set_index(["code", "date"]).sort_index()
print(f"panel: {pan.index.get_level_values('code').nunique()} unidades "
      f"(50 estados + DC), {len(pan)} obs, "
      f"{pan.index.get_level_values('date').min().date()} .. "
      f"{pan.index.get_level_values('date').max().date()}, evento {EVENT.date()}")

H = list(range(-6, -3)) + list(range(0, 25))   # -3..-1 son degenerados con n_lags=3
irf = panel_lp(pan, y="ur", x="shock", horizons=H, n_lags=3)
pre, post = irf[irf["h"] < 0], irf[irf["h"] >= 0]
print("pre-tendencias (h < 0):")
print(pre[["h", "beta", "se", "lo", "hi"]].round(3).to_string(index=False))
print("respuesta (primeros horizontes):")
print(post[["h", "beta", "se", "lo", "hi"]].head(6).round(3).to_string(index=False))

# %% slideshow={"slide_type": "slide"}
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.fill_between(post["h"], post["lo"], post["hi"], color="0.75", alpha=0.5,
                label="IC 90%")
ax.plot(post["h"], post["beta"], color="0.10", lw=1.8, marker="o", ms=3,
        label=r"$\beta_h$: efecto por +1 sd de exposición a IA")
ax.errorbar(pre["h"], pre["beta"], yerr=(pre["hi"] - pre["lo"]) / 2,
            fmt="s", ms=4, color="0.45", lw=1.0, capsize=2,
            label="pre-tendencias ($h<0$)")
ax.axhline(0, color="0.4", lw=0.8)
ax.axvline(-0.5, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("meses respecto al lanzamiento de ChatGPT")
ax.set_ylabel("respuesta del desempleo, pp")
ax.set_title("Event-study de exposición a la IA (desempleo estatal)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

sig_pre = int((np.sign(pre["lo"]) == np.sign(pre["hi"])).sum())
sig_post = int((np.sign(post["lo"]) == np.sign(post["hi"])).sum())
print(f"pre-tendencias con IC 90% que excluye el cero: {sig_pre} de {len(pre)} "
      f"(signo: {'+' if pre['beta'].mean() > 0 else '-'})")
print(f"horizontes post-evento con IC 90% que excluye el cero: {sig_post} "
      f"de {len(post)}")
print("  el diseño NO pasa su propia prueba de pre-tendencias: lo que sigue "
      "no admite lectura causal.")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Lectura honesta (¡cautela!).** Con la muestra congelada del bundle, la
# celda imprime **3 de 3 pre-tendencias significativas y positivas** ($h=-6,-5,
# -4$) y **7 de 25 horizontes post-evento significativos, todos negativos**.
# Leído crudo, diría "los estados más expuestos a la IA vieron *bajar* su
# desempleo". No lo leas así: el diseño **suspende su propia prueba**. Los
# estados de exposición alta (DC, MA, WA, CA, MD…) venían normalizándose más
# rápido tras el COVID *antes* del evento, y los coeficientes negativos
# posteriores son la continuación de esa misma trayectoria, no una respuesta a
# ChatGPT. La lista de advertencias es larga:
# 1. **Pre-tendencias rotas.** Con el gráfico delante, es el punto que manda: si
#    los tratados ya se movían distinto antes, $\beta_h$ no identifica nada.
# 2. **Identificación transversal de un solo evento.** El impulso ocurre en una
#    fecha; $\beta_h$ se identifica del corte transversal de 50 estados y DC,
#    con poquísimos grados de libertad efectivos.
# 3. **Proxy tosca.** La participación en ocupaciones de cómputo/matemáticas de
#    2019 mide *quién construye* la IA, no *quién es sustituido* por ella —
#    posiblemente con el signo equivocado (esos estados podrían ganar empleo).
# 4. **Confusión brutal.** 2022-2024 mezcla la desinflación, el alza de tasas y
#    el ciclo de despidos tecnológicos de 2023; nada de eso lo aísla este diseño.
# 5. **El desempleo es lento.** Un reacomodo tecnológico tardaría años en
#    aparecer en la tasa agregada estatal.
#
# Es el diseño correcto para *hacer la pregunta*, y el recordatorio correcto de
# que una pregunta candente no garantiza una respuesta identificada. Prueba de
# robustez para ti: mueve la ventana a `2018-01-01` (el rebote del COVID entra
# en la estimación) y verás que todo se vuelve un nulo ruidoso — las
# pre-tendencias dejan de rechazarse porque los errores estándar se inflan, no
# porque el problema haya desaparecido. Esa es exactamente la trampa.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 5. Preguntas para pensar
# 1. **Beveridge.** Si tras el COVID la curva se desplazó hacia afuera, ¿qué
#    parámetro del modelo de emparejamiento ($A_m$, $s$, $\phi$) cambió, y qué
#    política lo revertiría? ¿Y si en realidad la economía solo se movió *a lo
#    largo* de una curva estable?
# 2. **Shimer.** El cociente $sd(\theta)/sd(A)\approx 25$ dice que la
#    tensión reacciona demasiado a la productividad para el modelo estándar.
#    ¿Por qué la **rigidez salarial** (el salario responde poco al choque)
#    amplifica la respuesta de las vacantes y ayuda a resolver el puzzle?
# 3. **IA y flujos.** Si la IA sustituyera tareas de forma gradual, ¿en cuál
#    tasa de transición esperarías ver la huella primero — hallazgo $p_{UE}$,
#    separación $p_{EU}$, o salidas a fuera de la fuerza laboral $p_{EN}$? ¿Por
#    qué el desempleo agregado sería el último indicador en moverse?

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 6. Explora con IA
# Prueba esto con el tutor sin conexión (o cualquier asistente de IA):
# - "Explica en una frase por qué un descenso casi vertical de las vacantes sin
#   subida del desempleo es un rompecabezas para la curva de Beveridge estándar."
# - "¿Por qué la tasa de separación salta en las recesiones mientras la de
#   hallazgo de empleo cae — y cuál domina el aumento del desempleo?"

# %% slideshow={"slide_type": "fragment"}
print(tutor("En una frase, ¿qué es el puzzle de Shimer y por qué el modelo de "
            "búsqueda estándar no lo genera?"))

# %% [markdown] slideshow={"slide_type": "slide"}
# **Resumen.** Recorrimos el mercado laboral por sus cuatro objetos centrales:
# la **curva de Beveridge** y su corrimiento post-COVID (vacantes vs desempleo),
# el **puzzle de Shimer** (la tensión ~25x más volátil que la productividad en
# 2001-hoy; ~20 en la muestra original de Shimer), los **flujos** E/U/N de la
# CPS —de los que aquí solo son exactas las filas E y U— más un vistazo
# F/I/U/N a la informalidad mexicana de la ENOE, y un **event-study de
# exposición a la IA** que es tanto un escaparate del método como una lección de
# humildad identificativa: el diseño reprueba su prueba de pre-tendencias, y ese
# es el resultado. Todo en `puremacro` (Python puro), ejecutable sin conexión en
# tu propia máquina.
#
# **El otro mercado no competitivo del mazo Slides08** es la lección
# **10c — rigideces nominales** (`10c_rigideces_nominales_es`): competencia
# monopolista, Rotemberg y la inflación de 2021-2023, que en el temario va en la
# semana 15, justo *antes* de esta. Después vienen la **11** (Mortensen-Pissarides
# y la condición de Hosios: el modelo detrás del puzzle de Shimer que vimos aquí) y la
# **24** (la matriz de cuatro estados F/I/U/N de la ENOE, que retoma en serio el vistazo a
# México de la §3).
