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
# # Choques agregados en una economía heterogénea
#
# Al incorporar choques agregados de PTF a la economía de Aiyagari, la *distribución
# completa de la riqueza* pasa a ser una variable de estado. Krusell & Smith (1998)
# demostraron que los hogares pueden pronosticar el futuro con un único momento —
# el capital medio — con una precisión casi perfecta ("agregación aproximada").
# Resolvemos ese punto fijo, mostramos luego una transición con previsión perfecta
# y presentamos el referente de agente representativo.

# %% [markdown]
# ## El método en ecuaciones
#
# Al añadir la PTF agregada $Z$ a Aiyagari, la función de valor del hogar incorpora *dos*
# argumentos agregados — el nivel de capital $K$ y el choque $Z$:
# $$ V(a,z;\,K,Z) = \max_{a'\ge 0}\; u(c) + \beta\,\mathbb{E}\!\left[V(a',z';\,K',Z')\mid z,Z\right],
# \qquad c = w\,z + (1+r)\,a - a'. $$
# Los precios provienen de la tecnología agregada Cobb-Douglas, ahora escalada por $Z$:
# $$ r = \alpha\,Z\,(K/L)^{\alpha-1} - \delta, \qquad w = (1-\alpha)\,Z\,(K/L)^{\alpha}. $$
# El *verdadero* estado agregado es la distribución completa de la riqueza $\mu$, pues el
# capital de mañana $K'=\int a'\,d\mu$ depende del ahorro de cada hogar. La idea de Krusell &
# Smith consiste en sustituir $\mu$ por un único momento y una **regla de pronóstico log-lineal**,
# una por estado agregado:
# $$ \log K' = b_0(Z) + b_1(Z)\,\log K. $$
# Los hogares resuelven su problema tomando esta regla como dada; en equilibrio, la regla debe
# reproducir el $K'$ que la distribución simulada efectivamente genera. La **agregación
# aproximada** es el hallazgo empírico de que la regla ajusta con $R^2\approx 0{,}999$ — un único
# momento (el capital medio $K$) es un estadístico casi suficiente para toda la distribución.
#
# **Intuición.** Mantener $\mu$ como estado es inviable: es un objeto de dimensión infinita, de
# modo que la función de valor viviría en un espacio que ninguna computadora puede discretizar.
# ¿Por qué basta *un solo* momento? Porque la regla de ahorro $a'(a,z)$ es casi **lineal en la
# riqueza** en la región donde se concentra la masa, de manera que el ahorro agregado de la
# economía depende casi exclusivamente del capital *medio*, y no de cómo se reparte ese capital
# entre los hogares. Redistribuir riqueza entre dos agentes situados en el tramo lineal de la
# regla apenas altera el total que ahorran — por eso los momentos superiores de $\mu$ resultan
# casi irrelevantes para pronosticar $K'$. Las dos piezas siguientes someten esta idea a prueba:
# la **trayectoria de transición** (un choque MIT — una desviación puntual y perfectamente
# anticipada respecto del estado estacionario) muestra cómo viajan los precios y el capital
# *entre* estados estacionarios, y el modelo de **agente representativo** elimina por completo la
# heterogeneidad como referente límite.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import (
    krusell_smith, neoclassical_growth, aiyagari_steady_state,
    transition_path, VFIProblem, markov_stationary,
)

# %% [markdown]
# ## 1. Krusell–Smith: resolución del punto fijo de la regla de pronóstico
# Iteramos: resolvemos el problema del hogar (con K medio incorporado al estado
# exógeno) → simulamos la distribución de riqueza a lo largo de una trayectoria
# agregada extraída → re-estimamos el pronóstico log-lineal
# `log K' = b0[Z] + b1[Z] log K` → amortiguamos → repetimos.

# %%
ks = krusell_smith(n_a=150, n_K=5, T=2000, burn_in=300, seed=0)
print(f"converged={ks.converged} in {ks.n_outer} outer iters; "
      f"mean K = {ks.mean_K:.3f}, no-agg-risk K* = {ks.no_agg_risk_K:.3f}")
print(f"forecast R² per aggregate state: {np.round(ks.r_squared, 5)}")
assert ks.converged
assert (ks.r_squared > 0.95).all(), ks.r_squared          # approximate aggregation
assert abs(ks.mean_K / ks.no_agg_risk_K - 1.0) < 0.10

# Reconstruct the aggregate-state path with the same seed/defaults (the result
# object doesn't expose it — improvement backlog #2).
P_Z = np.array([[0.875, 0.125], [0.125, 0.875]])
rng = np.random.default_rng(0)
cdf = np.cumsum(P_Z, axis=1)
Z_path = np.empty(2000, dtype=int)
s = 0
for t in range(2000):
    Z_path[t] = s
    s = int(np.searchsorted(cdf[s], rng.random()))

# %% [markdown]
# **Lea el resultado.** Dos números resumen el hallazgo. Primero, el **$R^2$ del pronóstico por
# estado agregado** es $\approx 0{,}999$ tanto en el régimen de PTF baja como en el de PTF alta:
# regresar el $\log K$ del período siguiente sobre el $\log K$ de hoy no deja prácticamente
# varianza residual, de modo que un hogar que sigue únicamente el capital medio pronostica el
# agregado casi a la perfección. Esa es la agregación aproximada, cuantificada. Segundo, el
# **$K$ medio** simulado se sitúa a pocos puntos porcentuales del referente sin riesgo agregado
# $K^*$ (el cociente impreso es $\approx 1$): incorporar riesgo agregado reordena el capital a lo
# largo del ciclo, pero apenas desplaza su promedio de largo plazo, porque el motivo precautorio
# que ancla $K^*$ permanece en gran medida inalterado. Las pendientes de pronóstico $b_1(Z)$ son
# ambas algo inferiores a $1$ — el capital revierte a la media, regresando hacia $K^*$ tras un
# choque.

# %% [markdown]
# ### Figura principal — agregación aproximada
# El `log K` del período siguiente es esencialmente una línea en el `log K` de hoy,
# una por estado agregado, con R² ≈ 0,999. Un único momento resume la distribución
# completa.

# %%
burn = 300
logK = np.log(ks.K_path)
x, y, zt = logK[burn:-1], logK[burn + 1:], Z_path[burn:]
cols = _nbstyle.palette(2)
fig, ax = plt.subplots()
for iZ, (c, lab) in enumerate(zip(cols, ["low TFP", "high TFP"])):
    m = zt == iZ
    ax.scatter(x[m], y[m], s=6, color=c, alpha=0.35, label=f"{lab}  (R²={ks.r_squared[iZ]:.4f})")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, ks.b0[iZ] + ks.b1[iZ] * xs, color=c, linewidth=1.4)
ax.set_xlabel("log K  (today)"); ax.set_ylabel("log K′  (next period)")
ax.set_title("Krusell–Smith forecast rule"); ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### Figura complementaria — trayectoria simulada del capital
# El capital medio fluctúa en torno al nivel sin riesgo agregado K* a medida que la
# PTF alterna entre estados; las bandas sombreadas corresponden a períodos de PTF
# baja.

# %%
seg = slice(burn, burn + 400)
t = np.arange(400)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(t, ks.K_path[seg], color="0.0", linewidth=1.0)
ax.axhline(ks.no_agg_risk_K, color="0.5", linestyle="--", linewidth=0.8, label="K* (no agg risk)")
low = Z_path[seg] == 0
ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=low, color="0.85",
                step="mid", linewidth=0, label="low TFP")
ax.set_xlabel("Time"); ax.set_ylabel("Mean capital K")
ax.set_title("Aggregate capital over the cycle"); ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 2. Una transición con previsión perfecta (choque MIT)
# Partimos de una economía de Aiyagari con escasez de capital (todos pobres) y
# observamos su convergencia de regreso al estado estacionario; `r` cae y `K`
# aumenta a lo largo de la trayectoria.

# %%
ai = aiyagari_steady_state(n_z=5, n_a=100, a_max=50.0, gamma=1.0)
eqp = ai["equilibrium"].problem
a_g = np.asarray(eqp.a_grid, dtype=float)
z_g = np.asarray(eqp.z_grid, dtype=float)
P_g = np.asarray(eqp.P_z, dtype=float)
L = ai["L"]; alpha, delta, beta = 0.36, 0.08, 0.96
V_ss = ai["equilibrium"].solution.V
r_ss = ai["r"]

mu0 = np.zeros_like(ai["equilibrium"].distribution)
mu0[3, :] = markov_stationary(P_g)                        # all mass at a low asset node

def _wage(r):
    KL = (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))
    return (1.0 - alpha) * KL ** alpha

def build_problem(t, price_path):
    r = float(price_path[t]); w = _wage(r)
    def rf(ap, a, z, xp=np):
        c = w * xp.exp(z) + (1.0 + r) * a - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
    return VFIProblem(a_grid=a_g, z_grid=z_g, P_z=P_g, return_fn=rf, beta=beta,
                      options=dict(tol=1e-9, n_howard=30))

def implied_price_path(dists, policies, price_path):
    Ks = np.array([float(np.sum(d * a_g[:, None])) for d in dists[:-1]])
    Ks = np.maximum(Ks, 1e-6)
    return alpha * (Ks / L) ** (alpha - 1.0) - delta

T = 60
tp = transition_path(mu0, V_ss, build_problem, implied_price_path,
                     np.full(T, r_ss), damping=0.2, tol=2e-3, max_iter=600)
K_tp = np.array([float(np.sum(d * a_g[:, None])) for d in tp.distributions])
print(f"transition converged in {tp.n_iter} iters (gap {tp.gap:.1e}); "
      f"K: {K_tp[0]:.2f} → {K_tp[-1]:.2f}  (SS {ai['K']:.2f})")
assert K_tp[-1] > K_tp[0]                                  # capital rebuilds toward SS

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
a1.plot(K_tp, color="0.0"); a1.axhline(ai["K"], color="0.5", linestyle="--", linewidth=0.8)
a1.set_xlabel("Time"); a1.set_ylabel("Mean capital K"); a1.set_title("Capital transition")
a2.plot(tp.price_path, color="0.0"); a2.axhline(r_ss, color="0.5", linestyle="--", linewidth=0.8)
a2.set_xlabel("Time"); a2.set_ylabel("Interest rate r"); a2.set_title("Price transition")
plt.show()

# %% [markdown]
# **Lea el resultado.** Partimos de una economía con escasez de capital — toda la masa en un
# nodo de activos bajo —, de modo que el capital está por debajo de su estado estacionario y el
# producto marginal del capital, y por tanto $r$, es elevado. A medida que los hogares
# reconstruyen sus reservas precautorias, **$K$ aumenta de forma monótona hacia el estado
# estacionario** (panel izquierdo, convergiendo a la línea discontinua) mientras **$r$ cae**
# (panel derecho): una economía más abundante en capital obtiene un rendimiento menor, justo lo
# que indica $r=\alpha(K/L)^{\alpha-1}-\delta$, decreciente en $K$. Esta es la contraparte con
# previsión perfecta de las fluctuaciones estocásticas anteriores — la misma mecánica de precios,
# pero recorriendo una única trayectoria anticipada entre estados estacionarios en lugar de un
# ciclo estacionario.

# %% [markdown]
# ## 3. El referente de agente representativo
# Sin heterogeneidad, el modelo neoclásico de crecimiento estocástico posee una
# distribución ergódica del capital concentrada en torno a su estado estacionario
# analítico.

# %%
ng = neoclassical_growth(n_k=300)
mu_k = ng["distribution"].sum(axis=1)                      # marginal over TFP
print(f"rep-agent: mean K = {ng['mean_capital']:.3f}, analytical K_ss = {ng['K_ss']:.3f}")
assert abs(ng["mean_capital"] / ng["K_ss"] - 1.0) < 0.05
fig, ax = plt.subplots()
ax.fill_between(ng["k_grid"], mu_k, color="0.75", step="mid")
ax.axvline(ng["K_ss"], color="0.2", linestyle="--", linewidth=0.9, label="analytical K_ss")
ax.set_xlabel("Capital k"); ax.set_ylabel("Ergodic mass")
ax.set_title("Representative-agent stochastic growth"); ax.legend()
plt.show()

# %% [markdown]
# ## Su turno — ¿de verdad generaliza la regla de pronóstico?
#
# El R² anterior es *dentro de muestra*: la regla se ajustó sobre la misma trayectoria con la
# que se la evalúa, de modo que un R² alto podría, en principio, reflejar sobreajuste. La prueba
# honesta es *fuera de muestra*. Reutilizando el `ks` ya resuelto (sin volver a resolver el
# costoso punto fijo), dividimos la simulación posterior al período de calentamiento en un tramo
# de *ajuste* y un tramo *reservado*, reestimamos `b0[Z] + b1[Z]·logK` solo en el tramo de
# ajuste y la evaluamos en el tramo reservado. Si la agregación aproximada es real, el R² fuera
# de muestra debería mantenerse cerca del valor dentro de muestra. Cambie `split` abajo.

# %%
# ← Change this train/test split fraction (fit on the first `split` of the post-burn-in path,
# score on the rest). Keep it in (0, 1); the rule should generalize at any reasonable value.
split = 0.7
x_all, y_all, z_all = logK[burn:-1], logK[burn + 1:], Z_path[burn:]   # reuse the solved ks
k = int(split * x_all.size)
r2_test = np.empty(2)
for iZ in range(2):                                       # one rule per aggregate state
    fit = z_all[:k] == iZ
    A = np.vstack([np.ones(fit.sum()), x_all[:k][fit]]).T
    coef, *_ = np.linalg.lstsq(A, y_all[:k][fit], rcond=None)
    te = z_all[k:] == iZ                                  # held-out segment, same state
    yhat = coef[0] + coef[1] * x_all[k:][te]
    yt = y_all[k:][te]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2_test[iZ] = 1.0 - float(np.sum((yt - yhat) ** 2)) / ss_tot
print(f"split={split}: out-of-sample R² per state = {np.round(r2_test, 5)}  "
      f"(in-sample {np.round(ks.r_squared, 5)})")
assert r2_test.min() > 0.9, r2_test     # the rule generalizes: held-out fit stays high

# %% [markdown]
# **Ejercicios.** (1) Reduzca la ventana de ajuste (`split = 0.4`) — el R² fuera de muestra
# apenas debería moverse, pues la regla tiene solo dos parámetros por estado y la trayectoria es
# larga. (2) Compare `r2_test` con el `ks.r_squared` dentro de muestra: el desempeño fuera de
# muestra *no* es sistemáticamente peor, la firma de una agregación aproximada genuina y no de
# sobreajuste. (3) *Avanzado* (lento, vuelve a resolver el punto fijo): llame a
# `krusell_smith(n_a=150, n_K=5, T=2000, burn_in=300, seed=0, Z_vals=(0.96, 1.04))` para ampliar
# la dispersión de PTF agregada, y verifique que las dos pendientes de pronóstico `b1` se separan
# mientras el R² de cada estado se mantiene en ≈ 0,999.
#
# **¿Qué tan completo es esto?** Esto es apenas un rincón del instrumental de agentes
# heterogéneos de `puremacro.vfi`. La misma cadena `VFIProblem` → resolución → distribución →
# vaciado de mercado sustenta el **equilibrio general de Aiyagari** (NB01), el ciclo de vida /
# **OLG** con mortalidad (NB03), la entrada y salida de empresas de **Hopenhayn** (NB04) y las
# carteras de **dos activos** con EGM y preferencias de Epstein–Zin (NB05). `transition_path`
# resuelve transiciones deterministas de choque MIT entre estados estacionarios, y
# `neoclassical_growth` es el referente de agente representativo empleado más arriba.
