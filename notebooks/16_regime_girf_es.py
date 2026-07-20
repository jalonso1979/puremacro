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

# %% [markdown]
# # Transmisión dependiente del estado, bien hecha
#
# **¿Golpea más fuerte el mismo choque cuando la economía ya está en un mal
# estado?** Los modelos de regímenes — VAR de umbral, VAR con cambio de
# régimen markoviano, VECM de umbral — existen exactamente para responder
# eso. Pero ajustar uno y luego empujar sus coeficientes por régimen a
# través de una rutina de IRF *lineal* responde en silencio una pregunta
# distinta: "¿qué pasaría si la economía estuviera encerrada en ese régimen
# para siempre?" Durante mucho tiempo ese atajo de régimen congelado era la
# única respuesta al impulso que los ajustes de regímenes de este paquete
# podían ofrecer. Este notebook construye el objeto correcto — la **IRF
# generalizada** de Koop, Pesaran y Potter (1996, *J. Econometrics*) — y la
# usa para recuperar una asimetría plantada, contrastarla con una banda
# bootstrap y exponer la no linealidad en tamaño/signo al estilo de Kilian
# y Vigfusson (2011).

# %% [markdown]
# ## De la IRF lineal a la IRF generalizada en matemáticas
#
# Un VAR de umbral con dos regímenes cambia su dinámica según una variable
# de umbral rezagada $z_{t-d}$:
# $$ y_t = c^{(r_t)} + A^{(r_t)} y_{t-1} + \varepsilon_t^{(r_t)}, \qquad
#    r_t = \mathbb{1}\{z_{t-d} > c\}. $$
# En un VAR lineal la IRF es una única trayectoria determinista. Aquí la
# respuesta depende de la **historia** $\omega_{t-1}$ (en qué régimen
# empiezas) y del **tamaño y signo** del choque (un choque grande puede
# empujar a $z$ a cruzar el umbral en pleno vuelo), así que KPP definen la
# IRF como una esperanza condicional e integran el futuro por Monte Carlo:
# $$ GI(h, \delta, \omega_{t-1}) =
#    \mathbb{E}\!\left[y_{t+h} \mid \varepsilon_{j,t}+\delta,\ \omega_{t-1}\right]
#  - \mathbb{E}\!\left[y_{t+h} \mid \omega_{t-1}\right]. $$
# El estimador simula **trayectorias emparejadas**: una base con
# innovaciones $\varepsilon \sim N(0, I)$ mapeadas a través del factor de
# Cholesky del *régimen vigente*, y una gemela con choque que suma $\delta$
# al choque identificado $j$ en $h=0$ — mismos sorteos en todo lo demás. A
# lo largo de ambas trayectorias el régimen se reevalúa con la $z$ simulada
# en cada paso: **el cambio de régimen es endógeno**. Promediando sobre
# historias extraídas de la muestra, condicionales al régimen inicial, se
# obtienen $GI_{\text{calma}}(h)$, $GI_{\text{estrés}}(h)$ y el estadístico
# de asimetría de transmisión
# $$ \Delta(h) = GI_{\text{estrés}}(h) - GI_{\text{calma}}(h), $$
# con banda por bootstrap sobre las historias extraídas. El chequeo de
# Kilian-Vigfusson completa el instrumental: en un modelo lineal
# $GI(2\delta) = 2\,GI(\delta)$ y $GI(-\delta) = -GI(\delta)$, así que
# graficar $GI(\delta)/\delta$ entre tamaños y signos mide cuán no lineal
# es realmente la transmisión.

# %% [markdown]
# **Intuición.** Una IRF lineal es un solo número por horizonte porque un
# modelo lineal trata idénticamente cada trimestre y cada tamaño de choque.
# Un modelo de regímenes no: la transmisión depende de dónde *está* la
# economía — y, crucialmente, de adónde la *envía* el choque. Si un choque
# de estrés financiero arrastra a la economía a cruzar el umbral, el choque
# cambia la propia dinámica a través de la cual se propagará; congelar el
# régimen supone eliminar exactamente esa retroalimentación. La GIRF la
# conserva: las historias vienen de la muestra (así que "empezar en
# estrés" significa los trimestres de estrés que los datos realmente
# contienen), las trayectorias cambian de régimen endógenamente, y la IRF
# se convierte en un *promedio de futuros* en lugar de una recursión
# mecánica. El precio es ruido de simulación; la recompensa es un objeto
# que puede diferir honestamente entre regímenes, tamaños y signos — y una
# banda de diferencias que te dice si lo hace.

# %% [markdown]
# ## Preparación — una economía simulada con regímenes de calma y estrés
#
# Dos variables: un índice de condiciones financieras ($z_t$, columna 0 —
# la variable de umbral) y el crecimiento del producto (columna 1).
# **Plantamos** la dependencia del estado: en el régimen de estrés
# ($z_{t-1} > 0$) el ICF es más persistente (0.8 vs 0.5) y arrastra al
# crecimiento cinco veces más fuerte (−0.5 vs −0.1). La covarianza de las
# innovaciones es la *identidad en ambos regímenes*, de modo que toda
# asimetría de aquí en adelante es transmisión, no volatilidad de impacto.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf as var_irf
from puremacro.var.regime import girf, tvar_fit
from puremacro.var.regime.threshold import TVARResult

A_CALM = np.array([[0.5, 0.0],      # FCI moderately persistent
                   [-0.1, 0.4]])    # ...and a mild drag on growth
A_STRESS = np.array([[0.8, 0.0],    # FCI very persistent under stress
                     [-0.5, 0.4]])  # ...and a 5x stronger drag on growth

def simulate_economy(T=600, seed=20):
    """Self-exciting TVAR(1): regime decided by last quarter's FCI."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        A = A_STRESS if Y[t - 1, 0] > 0.0 else A_CALM
        Y[t] = A @ Y[t - 1] + rng.standard_normal(2)
    return Y

Y = simulate_economy()
stress_share = float((Y[:, 0] > 0.0).mean())
print(f"T = {len(Y)} quarters | share of stress quarters = {stress_share:.3f}")
assert 0.40 < stress_share < 0.70   # both regimes well populated

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(Y[:, 0], color="0.15", linewidth=0.9)
ax.axhline(0.0, color="0.55", linewidth=0.9, linestyle="--")
ax.fill_between(np.arange(len(Y)), Y[:, 0].min(), Y[:, 0].max(),
                where=Y[:, 0] > 0.0, color="0.85", zorder=0)
ax.set_xlabel("Quarter")
ax.set_ylabel("Financial conditions index $z_t$")
ax.set_title("Simulated FCI — shaded spans are stress quarters ($z_t > 0$)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ajustar el VAR de umbral
#
# `tvar_fit` busca en una malla el umbral $c$ y el rezago de decisión $d$
# (Tsay 1998) y corre OLS por régimen. Nada le dice que la partición
# verdadera está en cero con $d=1$ — tiene que encontrar eso, y la brecha
# de coeficientes plantada, por su cuenta.

# %%
fit = tvar_fit(Y, threshold_var_idx=0, p=1, delay_grid=(1, 2), n_threshold_grid=40)
print(f"threshold c = {fit.threshold:+.3f} (true 0.0) | delay d = {fit.delay} (true 1)")
print(f"regime split: {fit.n_low} calm / {fit.n_high} stress quarters")
print("A_low  (calm)  =", np.round(fit.A_low, 3).tolist())
print("A_high (stress)=", np.round(fit.A_high, 3).tolist())
assert fit.delay == 1
assert abs(fit.threshold) < 0.5                       # near the true zero
assert fit.A_high[1, 0] < fit.A_low[1, 0] - 0.25      # planted cross-effect gap

# %% [markdown]
# **Lee la salida.** La malla aterriza cerca de la verdad: umbral cercano a
# cero, rezago 1, y el régimen de estrés estimado lleva las dos huellas
# plantadas — mayor persistencia del ICF ($\approx 0.7$ vs $0.5$) y el
# arrastre sobre el crecimiento mucho más fuerte ($\approx -0.47$ vs
# $-0.11$). La atenuación del estimador de persistencia (0.7, no 0.8) es el
# precio habitual de las observaciones mal clasificadas cerca del umbral.
# Hasta aquí esto es solo estimación; la pregunta es qué *significan* estos
# dos bloques de coeficientes para un choque — y eso no se responde
# alimentando cualquiera de los bloques a una IRF lineal.

# %% [markdown]
# ## La GIRF: historias, cambio endógeno, simulación emparejada
#
# Una sola llamada ejecuta la construcción KPP completa: extraer 40
# historias por régimen inicial de la muestra, simular 80 trayectorias
# emparejadas por historia con el impulso sumado al choque identificado del
# ICF en $h=0$ (Cholesky dentro del régimen), dejar que el régimen cambie
# endógenamente a lo largo de cada trayectoria, y promediar.
# `shock_size=[1, 2, -1, -2]` añade el menú de tamaño/signo de
# Kilian-Vigfusson, y la diferencia entre regímenes viene con una banda
# bootstrap al 90%.

# %%
res = girf(fit, Y, shock=0, horizon=16, n_hist=40, n_sim=80,
           shock_size=[1.0, 2.0, -1.0, -2.0], n_boot=300, rng=16)
print(res.summary())

h = np.arange(res.horizon + 1)
g_calm = res.girf_by_regime[0, 0]     # (+1 sd shock) x (H+1, n)
g_strs = res.girf_by_regime[1, 0]
d_gr = res.difference[0, :, 1]
d_lo = res.difference_lo[0, :, 1]
d_hi = res.difference_hi[0, :, 1]
print(f"growth response at h=2: calm {g_calm[2, 1]:+.3f} | stress {g_strs[2, 1]:+.3f}")
print(f"difference (stress-calm) at h=2: {d_gr[2]:+.3f}, 90% band "
      f"[{d_lo[2]:+.3f}, {d_hi[2]:+.3f}]")
assert g_strs[2, 1] < g_calm[2, 1] - 0.10   # stress transmission is stronger...
assert d_hi[2] < -0.10                      # ...and the band excludes zero

# the naive frozen-regime IRF: pretend the economy never leaves stress
naive = var_irf([fit.A_high], np.linalg.cholesky(fit.Sigma_high), 16)[:, :, 0]
cum_naive, cum_girf = naive[:, 1].sum(), g_strs[:, 1].sum()
print(f"cumulative growth loss in stress: frozen-regime {cum_naive:+.2f} "
      f"vs GIRF {cum_girf:+.2f}")
assert cum_naive < cum_girf - 0.30          # freezing the regime overstates it

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
axes[0].axhline(0, color="0.6", linewidth=0.8, linestyle=":")
axes[0].plot(h, g_calm[:, 1], color="0.45", linewidth=1.8, label="start in calm")
axes[0].plot(h, g_strs[:, 1], color="0.00", linewidth=1.8, linestyle="--",
             label="start in stress")
axes[0].plot(h, res.girf_pooled[0, :, 1], color="0.30", linewidth=1.0,
             linestyle=":", label="pooled")
axes[0].plot(h, naive[:, 1], color="0.65", linewidth=1.0, linestyle="-.",
             label="frozen-stress (naive)")
axes[0].set_xlabel("Horizon (quarters)")
axes[0].set_ylabel("Growth response to a +1 sd FCI shock")
axes[0].set_title("Generalized IRF by starting regime")
axes[0].legend(fontsize=8)
axes[1].axhline(0, color="0.6", linewidth=0.8, linestyle=":")
axes[1].fill_between(h, d_lo, d_hi, color="0.75", alpha=0.6,
                     label="90% bootstrap band")
axes[1].plot(h, d_gr, color="0.00", linewidth=1.8, label="stress $-$ calm")
axes[1].set_xlabel("Horizon (quarters)")
axes[1].set_ylabel("Difference in growth response")
axes[1].set_title("Regime-dependent transmission test")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Lee la salida.** Panel izquierdo: el mismo choque financiero de +1
# desviación estándar cuesta alrededor de **−0.42** de crecimiento en $h=2$
# cuando aterriza en estrés frente a **−0.24** en calma — casi el doble de
# daño, puramente por dónde aterriza. La línea ingenua de estrés congelado
# sobrestima la respuesta de estrés (pérdida acumulada −2.6 vs −1.9): las
# trayectorias reales *escapan* al régimen de calma conforme el ICF
# revierte a la media, y la GIRF promedia esas salidas mientras que la
# recursión congelada las prohíbe. Panel derecho: la diferencia
# estrés-menos-calma es negativa y su banda al 90% queda **estrictamente
# por debajo de cero** durante los dos primeros años — un contraste
# bootstrap directo de transmisión dependiente del régimen, que aquí
# rechaza correctamente la simetría porque la asimetría la plantamos
# nosotros. Una sutileza que vale la pena saborear: la respuesta *propia*
# del ICF difiere entre regímenes mucho menos de lo que sugieren los
# coeficientes (0.7 vs 0.5), porque un choque positivo en calma empuja al
# ICF a cruzar el umbral y entonces se propaga de todos modos con la
# dinámica de estrés — el cambio endógeno en acción.

# %% [markdown]
# ## Asimetría de tamaño y signo, al estilo Kilian-Vigfusson
#
# En un modelo lineal $GI(\delta)/\delta$ es una sola curva, sea cual sea
# $\delta$. En un modelo de umbral no: un choque de +2 desviaciones pasa
# más parte de su vida en el régimen de estrés que uno de +1, y un choque
# de −1 activamente *rescata* a la economía del estrés. Dividir cada GIRF
# por su propio $\delta$ hace visibles esas desviaciones.

# %%
sc = res.scaled()                     # girf_pooled / delta, shape (S, H+1, n)
labels = [f"$\\delta = {d:+.0f}$ sd" for d in res.shock_sizes]
cols = _nbstyle.palette(4)
stys = _nbstyle.styles(4)
dev_size = np.abs(sc[1, :, 1] - sc[0, :, 1]).max()   # |GI(2d)/2 - GI(d)| gap
dev_sign = np.abs(sc[2, :, 1] - sc[0, :, 1]).max()   # |GI(-d)/(-d) - GI(d)| gap
print(f"max size deviation |GI(2)/2 - GI(1)|  (growth): {dev_size:.3f}")
print(f"max sign deviation |GI(-1)/-1 - GI(1)| (growth): {dev_sign:.3f}")
assert np.abs(sc[:, 0, :] - sc[0, 0, :]).max() < 1e-12  # impact IS proportional
assert dev_size > 0.02 and dev_sign > 0.05              # dynamics are NOT

fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":")
for s in range(4):
    ax.plot(h, sc[s, :, 1], color=cols[s], linestyle=stys[s], linewidth=1.6,
            label=labels[s])
ax.set_xlabel("Horizon (quarters)")
ax.set_ylabel("Scaled growth response  $GI(\\delta)/\\delta$")
ax.set_title("Kilian-Vigfusson size/sign check: one curve iff linear")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Lee la salida.** En $h=0$ las cuatro curvas coinciden *exactamente* —
# la respuesta de impacto es $\mathrm{chol}(\Sigma_r)e_j\delta$,
# proporcional por construcción — así que cualquier dispersión en
# $h \geq 1$ es pura no linealidad de transmisión. Y se dispersa: los
# choques positivos (que reclutan al régimen de estrés) transmiten más daño
# por desviación estándar que los negativos (que huyen de él), y los
# choques de $\pm 2$ desviaciones se desvían más que los de $\pm 1$, porque
# los impulsos mayores reubican más trayectorias al otro lado del umbral.
# Si alguien te entrega una sola IRF para un modelo de regímenes sin una
# $\delta$ en la etiqueta, este gráfico es la pregunta que hay que hacer.

# %% [markdown]
# ## El chequeo de cordura del límite lineal — validación que puedes repetir
#
# ¿Cómo sabemos que el simulador es correcto? Fuerza al modelo a ser lineal
# y exige de vuelta la forma cerrada. Construye un `TVARResult` cuyos dos
# regímenes son *copias idénticas* de un VAR lineal ajustado: el umbral
# sigue "cambiando" regímenes a lo largo de cada trayectoria, pero ambos
# regímenes llevan los mismos coeficientes, así que la GIRF debe colapsar a
# `var.irf` — y como el impulso se *suma* bajo números aleatorios comunes,
# la diferencia emparejada es determinista y la coincidencia es a precisión
# de máquina, no salvo ruido de Monte Carlo. Este es el mismo oráculo que
# fija la batería de tests.

# %%
est = estimate_var(Y, 1)
A_hat = np.hstack(est.A_list)
fit_lin = TVARResult(
    A_low=A_hat, intercept_low=est.c, Sigma_low=est.Sigma,
    A_high=A_hat.copy(), intercept_high=est.c.copy(), Sigma_high=est.Sigma.copy(),
    threshold=float(np.median(Y[:, 0])), delay=1, threshold_var_idx=0,
    n_low=0, n_high=0, rss_total=0.0, grid={},
)
chk = girf(fit_lin, Y, shock=0, horizon=16, n_hist=20, n_sim=30, n_boot=100, rng=1)
closed_form = var_irf(est.A_list, np.linalg.cholesky(est.Sigma), 16)[:, :, 0]
gap = np.abs(chk.girf_pooled[0] - closed_form).max()
print(f"max |GIRF - linear IRF| in the linear limit: {gap:.2e}")
print(f"max |regime difference| in the linear limit: {np.abs(chk.difference).max():.2e}")
assert gap < 1e-10
assert np.abs(chk.difference).max() < 1e-12

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":")
for j, (name, sty) in enumerate([("FCI", "-"), ("growth", "--")]):
    ax.plot(h, closed_form[:, j], color="0.15", linestyle=sty, linewidth=1.6,
            label=f"linear IRF — {name}")
    ax.plot(h[::2], chk.girf_pooled[0, ::2, j], "o", color="0.45", markersize=4,
            label=f"GIRF — {name}" if j == 0 else None)
ax.set_xlabel("Horizon (quarters)")
ax.set_ylabel("Response to a +1 sd FCI shock")
ax.set_title("Linear limit: the GIRF collapses onto the closed form")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **El momento de validación.** Esta es la disciplina que el notebook
# quiere que robes: todo estimador basado en simulación debería venir con
# un límite en el que su respuesta se conoce *exactamente*. Para la GIRF de
# KPP ese límite es "regímenes que no difieren", y la coincidencia a
# precisión de máquina de arriba (del orden de $10^{-16}$, no de $10^{-2}$)
# solo es posible porque el impulso se suma al choque identificado bajo
# números aleatorios comunes — una decisión de diseño tomada *para* la
# testeabilidad. Cuando el límite lineal se cumple y la asimetría plantada
# se recupera con el signo correcto, la salida interesante (la banda de
# diferencias) hereda esa credibilidad.

# %% [markdown]
# ## Tu turno — ¿qué tan grande debe ser un choque para romper la proporcionalidad?
#
# El ejercicio de abajo recalcula la GIRF para un tamaño de choque de tu
# elección y la compara, por desviación estándar, contra el punto de
# referencia de +1. La comparación de impacto debe coincidir *exactamente*
# para cualquier $\delta$ (esa es la proporcionalidad de Cholesky que
# verificaste arriba); el número interesante es cuánto se separan las
# curvas en horizontes de ciclo económico.

# %%
DELTA_TRY = 2.0   # ← change this: shock size in sd units (try 0.5, 3.0, -2.0, 5.0)
res_try = girf(fit, Y, shock=0, horizon=16, n_hist=30, n_sim=60,
               shock_size=[1.0, DELTA_TRY], n_boot=100, rng=2)
sc_try = res_try.scaled()
drift = np.abs(sc_try[1, :, 1] - sc_try[0, :, 1])
print(f"delta = {DELTA_TRY:+.1f} sd -> max per-sd growth drift vs +1 sd: "
      f"{drift.max():.3f} at h = {int(drift.argmax())}")
# Holds for the default and ANY delta you try: impact is exactly proportional.
assert np.allclose(sc_try[1, 0], sc_try[0, 0], atol=1e-10)

# %% [markdown]
# **Ejercicios.** (1) *Básico*: pon `DELTA_TRY = -2.0` y explica el signo
# de la desviación usando el umbral: ¿qué régimen reclutan los choques
# negativos del ICF, y por qué eso los hace *más débiles* por desviación
# estándar? (2) *Intermedio*: reajusta el modelo con `delay_grid=(2,)` y
# vuelve a correr la GIRF — ¿forzar el rezago equivocado encoge la banda de
# la diferencia estrés-calma hacia cero, y qué enseña eso sobre
# especificar mal $d$? (3) *Avanzado*: reemplaza `tvar_fit` por
# `ms_var_fit(Y, K=2, p=1)` y llama a la *misma* función `girf`. La banda
# de diferencias ahora refleja solo el Cholesky de impacto — explica por
# qué la especificación markoviana con $A$ compartida no puede generar
# asimetría de transmisión, y qué necesitaría el modelo para poder hacerlo.
#
# ## Por qué esto importa para la investigación de incertidumbre por regímenes
#
# La literatura empírica de incertidumbre por regímenes vive exactamente de
# este objeto. Caldara, Fuentes-Albero, Gilchrist y Zakrajšek (2016, *EER*)
# sostienen que los choques de incertidumbre se transmiten sobre todo
# cuando las condiciones financieras están tensas — una afirmación *sobre*
# $\Delta(h)$, contrastable solo con una construcción que deje a la
# economía moverse entre estados tensos y holgados a mitad de la respuesta,
# como aquí. Del lado estructural, el modelo Diamond-Mortensen-Pissarides
# con regímenes de este paquete (`puremacro.models.dmp_regime_dependent`,
# `dmp_irf(..., regime="H")` vs `regime="L"`) produce el mismo fenómeno
# desde la teoría — regímenes de volatilidad que cambian el cálculo de
# excedente de la creación de empleo — y la GIRF es la lente de forma
# reducida que apuntarías a datos generados por un modelo así. El flujo de
# trabajo honesto que ambos comparten: plantear el modelo de regímenes,
# simularlo hacia adelante *con el cambio de régimen intacto*, y reportar
# la diferencia entre regímenes con una banda, nunca una IRF de régimen
# congelado.
#
# **¿Qué tan completo es esto?** `girf` despacha sobre los tres ajustes de
# regímenes de `puremacro.var.regime` — `tvar_fit` (usado aquí),
# `tvecm_fit` (cointegración con umbral, respuestas en niveles) y
# `ms_var_fit` (trayectorias de régimen extraídas de la matriz de
# transición ajustada). Las alternativas uniecuacionales viven en
# `puremacro.lp` (`lp_state_dep` para proyecciones locales dependientes del
# estado al estilo Auerbach-Gorodnichenko); `puremacro.uncertainty.regimes`
# fecha los regímenes mismos (quiebres de Bai-Perron, regímenes de
# calendario); y el notebook 06 cubre las convenciones de identificación
# que hereda la elección de Cholesky dentro del régimen.
