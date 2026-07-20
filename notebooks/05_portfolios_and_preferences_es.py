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
# # Riesgo, rendimientos y preferencias
#
# Tres de las capacidades más avanzadas de la librería: **múltiples estados
# endógenos** (un activo líquido + uno ilíquido), **preferencias de Epstein–Zin**
# que separan la aversión al riesgo de la elasticidad de sustitución
# intertemporal, y el **método de la cuadrícula endógena (EGM)** como
# alternativa rápida y precisa de resolución. Resolvemos un problema de cartera
# de dos activos, elevamos la aversión al riesgo bajo Epstein–Zin manteniendo
# fija la elasticidad, y mostramos que el EGM y la iteración en la función de
# valor recuperan la misma política de consumo. Todo con `puremacro.vfi`, en el
# navegador.

# %% [markdown]
# ## El método en tres piezas
#
# **Restricción presupuestaria de dos activos.** Los hogares mantienen un activo
# líquido $m$ (de libre ajuste, rendimiento bajo $r_m$) y un activo ilíquido $k$
# (rendimiento mayor $r_k>r_m$, pero reequilibrarlo cuesta $\kappa\,|k'-k|$ por
# unidad movida). Con renta AR(1) $e^{z}$ (discretizada por Tauchen) y sin
# posibilidad de endeudarse, el consumo es
# $$ c = (1+r_m)\,m + (1+r_k)\,k + e^{z} - m' - k' - \kappa\,|k'-k|,\qquad m',k'\ge 0. $$
# El estado es el *par* $(m,k)$ — dos estados endógenos continuos optimizados de forma conjunta.
#
# **Utilidad recursiva de Epstein–Zin.** La utilidad CRRA separable en el tiempo
# ata la aversión al riesgo a la inversa de la elasticidad de sustitución
# intertemporal (EIS). Epstein–Zin rompe ese vínculo, con la aversión al riesgo
# $\gamma$ y la EIS $\psi$ como parámetros separados:
# $$ V = \Big[(1-\beta)\,u^{\rho} + \beta\,\big(\mathbb{E}_t[\,V'^{\,1-\gamma}\,]\big)^{\rho/(1-\gamma)}\Big]^{1/\rho},
# \qquad \rho = 1-\tfrac{1}{\psi}. $$
# El término interior es el **equivalente cierto** $\mathrm{CE}=(\mathbb{E}_t[V'^{\,1-\gamma}])^{1/(1-\gamma)}$:
# el valor aleatorio del próximo período se colapsa a un equivalente sin riesgo
# a través de la curvatura de aversión al riesgo $\gamma$, y luego se contrapone
# con el presente mediante la curvatura de la EIS $\rho$. Cuando $\gamma=1/\psi$
# (es decir, $\rho=1-\gamma$) la recursión vuelve a colapsar a la utilidad
# esperada separable en el tiempo.
#
# **Método de la cuadrícula endógena (EGM).** Para el problema de un activo con
# CRRA, la política óptima satisface la ecuación de Euler $c^{-\gamma}=\beta(1+r)\,\mathbb{E}_t[c'^{-\gamma}]$.
# En lugar de buscar $a'$ en cada estado, el EGM *fija* una cuadrícula de activos
# del próximo período, evalúa el lado derecho e invierte para leer el consumo de
# hoy $c=\big(\beta(1+r)\,\mathbb{E}_t[c'^{-\gamma}]\big)^{-1/\gamma}$ y los activos
# corrientes endógenos que lo justifican — sin búsqueda de raíces, coste $O(n)$
# por iteración, y una política continua (interpolada).
#
# **Intuición.** Separar $\gamma$ de $\psi$ importa porque el riesgo y el tiempo
# son distintos: $\gamma$ gobierna cuánto le disgusta al hogar una *apuesta* sobre
# el valor del próximo período, mientras que $\psi$ gobierna con qué disposición
# *traslada* consumo entre fechas. Manténgase $\psi$ fija y elévese $\gamma$, y se
# refuerza únicamente el motivo precautorio — los hogares ahorran más para la
# *misma* disposición a sustituir en el tiempo, algo que un único parámetro CRRA
# no puede expresar. El EGM es rápido porque nunca optimiza por estado: explota la
# monotonía de la ecuación de Euler para invertir la política directamente. Y la
# división en dos activos es una disyuntiva riesgo–rendimiento — la prima de
# rendimiento del activo ilíquido se paga con fricciones de ajuste, de modo que
# los saldos líquidos son el amortiguador que absorbe el riesgo de renta.

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
    two_asset_profile, EpsteinZinProblem, solve_egm, VFIProblem,
    tauchen, markov_stationary, stationary_distribution,
)

# %% [markdown]
# ## 1. Dos activos: una reserva líquida y un depósito ilíquido de valor
# El activo líquido `m` es de libre ajuste pero ofrece un rendimiento bajo;
# el ilíquido `k` paga un rendimiento mayor pero tiene costes de reequilibrio.
# Resolver sobre la cuadrícula conjunta `(m, k)` es un problema de
# **múltiples estados endógenos** — dos activos continuos optimizados a la vez.

# %%
ta = two_asset_profile(n_m=18, n_k=20, n_z=5, r_illiquid=0.03, k_max=25.0)
print(f"E[liquid]={ta['mean_liquid']:.2f}, E[illiquid]={ta['mean_illiquid']:.2f}, "
      f"illiquid share={ta['illiquid_share']:.2f}, wealth Gini={ta['wealth_gini']:.2f}")
assert 0.0 < ta["illiquid_share"] < 1.0
assert 0.0 < ta["wealth_gini"] < 1.0

n_m, n_k = 18, 20
m_grid = np.linspace(0.0, 15.0, n_m)
k_grid = np.linspace(0.0, 25.0, n_k)
mk = ta["distribution"].sum(axis=1).reshape(n_m, n_k)     # joint (m,k) mass, marginal over income

# %% [markdown]
# ### Figura central — la distribución conjunta de cartera
# Los hogares concentran su riqueza en el activo ilíquido de mayor rendimiento
# `k`, manteniendo un saldo líquido `m` reducido como amortiguador frente al
# riesgo de renta. La cruz indica la tenencia media — la división óptima que
# el motor encuentra sobre dos estados endógenos.

# %%
fig, ax = plt.subplots()
im = ax.imshow(mk.T, origin="lower", aspect="auto", cmap="Greys",
               extent=[m_grid[0], m_grid[-1], k_grid[0], k_grid[-1]])
ax.scatter([ta["mean_liquid"]], [ta["mean_illiquid"]], color="0.0", marker="x", s=70,
           label=f"mean (m={ta['mean_liquid']:.1f}, k={ta['mean_illiquid']:.1f})")
ax.set_xlabel("Liquid m"); ax.set_ylabel("Illiquid k")
ax.set_title("Two-asset stationary distribution")
ax.legend(loc="upper right")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mass")
plt.show()

# %% [markdown]
# **Lee el resultado.** La cuota ilíquida impresa arriba (la masa en $k$ como
# fracción de la riqueza total) está muy por encima de la mitad: los hogares
# guardan la mayor parte de su riqueza en el activo ilíquido de mayor rendimiento
# y mantienen solo un saldo líquido $m$ reducido. Esa pequeña porción líquida es
# *precautoria* — les permite suavizar las perturbaciones de renta sin pagar el
# coste de ajuste $\kappa$ por tocar $k$. El índice de Gini de riqueza,
# estrictamente entre 0 y 1, confirma una distribución genuinamente dispersa
# sobre la cuadrícula conjunta $(m,k)$: la cruz de la figura marca la tenencia
# media, y la masa se inclina hacia $k$ alto y $m$ bajo, exactamente el rincón
# que la prima de rendimiento premia.

# %% [markdown]
# ## 2. Epstein–Zin: aversión al riesgo frente a la elasticidad de sustitución
# La utilidad CRRA separable en el tiempo fuerza que la aversión al riesgo sea
# igual a 1/EIS. Epstein–Zin las desvincula. Mantener la EIS fija y elevar la
# aversión al riesgo refuerza el motivo precautorio, por lo que los hogares
# ahorran más.

# %%
z_g, Pz = tauchen(n=5, rho=0.9, sigma=0.2)
a_g = np.linspace(0.01, 30.0, 120)
psi = 1.5                                                 # EIS (fixed)

def ez_mean_assets(gamma):
    def felicity(ap, a, z, xp=np):                        # period felicity must be > 0
        c = 1.0 + 0.5 * xp.exp(z) + 1.03 * a - ap
        return xp.where(c > 0.0, c, -np.inf)
    sol = EpsteinZinProblem(a_grid=a_g, z_grid=z_g, P_z=Pz, return_fn=felicity,
                            beta=0.95, gamma=gamma, psi=psi).solve()
    mu = stationary_distribution(sol.policy_aprime, Pz)
    return float(np.sum(mu * a_g[:, None])), a_g[sol.policy_aprime]

mean_lo, pol_lo = ez_mean_assets(2.0)
mean_hi, pol_hi = ez_mean_assets(8.0)
print(f"mean assets: risk aversion 2 → {mean_lo:.2f}, risk aversion 8 → {mean_hi:.2f}")
assert mean_hi > mean_lo                                  # more risk aversion → more saving

mid = len(z_g) // 2
fig, ax = plt.subplots()
ax.plot(a_g, a_g, color="0.6", linewidth=0.8, linestyle=":", label="45°")
ax.plot(a_g, pol_lo[:, mid], color=_nbstyle.palette(2)[0], label="risk aversion 2")
ax.plot(a_g, pol_hi[:, mid], color=_nbstyle.palette(2)[1], linestyle="--",
        label="risk aversion 8")
ax.set_xlabel("Assets today"); ax.set_ylabel("Assets tomorrow")
ax.set_title("Epstein–Zin: precautionary saving (EIS fixed)"); ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# **Lee el resultado.** Con la EIS mantenida en $\psi=1.5$, lo único que cambió
# entre las dos resoluciones es la aversión al riesgo ($\gamma=2$ frente a
# $\gamma=8$), y aun así los activos medios aumentan — un efecto precautorio puro
# que un único parámetro CRRA no podría aislar, ya que elevar la CRRA también
# habría movido la EIS. En la figura, la política de ahorro con $\gamma=8$ se
# sitúa *por encima* de la de $\gamma=2$ y más arriba aún de la línea de 45°: en
# cada nivel de activos, el hogar más averso al riesgo lleva más riqueza al
# próximo período como auto-seguro frente a la perturbación de renta AR(1).

# %% [markdown]
# ## 3. EGM frente a VFI: misma solución, distintos algoritmos
# El método de la cuadrícula endógena invierte la ecuación de Euler (política
# continua, coste $O(n)$ por iteración). Su política de consumo coincide
# estrechamente con la obtenida mediante iteración en la función de valor
# (VFI) con cuadrícula discreta.

# %%
gamma, r, beta = 2.0, 0.03, 0.96
income = np.exp(z_g)
egm = solve_egm(a_g, z_g, income, Pz, beta=beta, r=r, gamma=gamma)

def util(c, xp):
    cc = xp.maximum(c, 1e-12)
    return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

def rf(ap, a, z, xp=np):
    c = (1.0 + r) * a + xp.exp(z) - ap
    return xp.where(c > 0.0, util(c, xp), -np.inf)

vsol = VFIProblem(a_grid=a_g, z_grid=z_g, P_z=Pz, return_fn=rf, beta=beta,
                  options=dict(tol=1e-9, n_howard=40)).solve()
c_vfi = (1.0 + r) * a_g[:, None] + income[None, :] - a_g[vsol.policy_aprime]
max_gap = float(np.max(np.abs(c_vfi[:, mid] - egm.c[:, mid])))
print(f"max |c_EGM − c_VFI| at median income = {max_gap:.3f}")
assert max_gap < 0.5                                      # close (VFI is grid-discretized)

fig, ax = plt.subplots()
ax.plot(a_g, egm.c[:, mid], color="0.0", label="EGM (continuous)")
ax.plot(a_g, c_vfi[:, mid], color="0.45", linestyle="--", label="VFI (discretized)")
ax.set_xlabel("Assets a"); ax.set_ylabel("Consumption c")
ax.set_title("EGM vs. VFI consumption policy"); ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# **Lee el resultado.** La brecha máxima entre las políticas de consumo del EGM y
# de la VFI (impresa arriba) es una fracción pequeña del consumo — los dos
# algoritmos coinciden. El residuo no es error del EGM sino *discretización* de la
# VFI: la política del EGM es continua (interpola a partir de la ecuación de Euler
# invertida), mientras que la VFI debe elegir $a'$ entre los 120 nodos de la
# cuadrícula, de modo que su consumo es constante a tramos y ambas divergen más
# donde la cuadrícula es gruesa en relación con la pendiente de la política. En la
# figura, la curva continua del EGM atraviesa con suavidad la curva escalonada de
# la VFI. La misma economía, dos algoritmos — y el EGM llega allí con coste $O(n)$
# por iteración y sin optimización por estado.

# %% [markdown]
# ## Tu turno — ¿sigue coincidiendo el EGM con la VFI al cambiar las preferencias?
#
# La sección 3 usó `gamma = 2`. Sube la aversión al riesgo (una utilidad más
# curvada, una ecuación de Euler más rígida) y comprueba dos hechos estructurales
# que deberían sobrevivir a *cualquier* elección razonable: (i) las políticas de
# consumo del EGM y de la VFI siguen coincidiendo dentro de una tolerancia, y (ii)
# la política del EGM es monótona en la riqueza (consumir más cuando se tiene
# más). Mantenemos el rendimiento `r` y `beta` fijos en los valores de la sección
# 3, por lo que esto sigue siendo una re-solución barata de un activo. Cambia
# `gamma_you`.
#
# Consejo: mantén `gamma_you` muy por encima de 0 (la CRRA exige $\gamma>0$). Los
# valores muy grandes (digamos 20+) aplanan la política de consumo a medida que la
# precaución domina, pero el EGM y la VFI siguen acompañándose — la coincidencia
# es sobre los *algoritmos*, no sobre el parámetro.

# %%
# ← Change this risk-aversion parameter (CRRA curvature). Keep it > 0.
gamma_you = 5.0

egm_you = solve_egm(a_g, z_g, income, Pz, beta=beta, r=r, gamma=gamma_you)

def util_you(c, xp):
    cc = xp.maximum(c, 1e-12)
    return (cc ** (1.0 - gamma_you) - 1.0) / (1.0 - gamma_you)

def rf_you(ap, a, z, xp=np):
    c = (1.0 + r) * a + xp.exp(z) - ap
    return xp.where(c > 0.0, util_you(c, xp), -np.inf)

vsol_you = VFIProblem(a_grid=a_g, z_grid=z_g, P_z=Pz, return_fn=rf_you, beta=beta,
                      options=dict(tol=1e-9, n_howard=40)).solve()
c_vfi_you = (1.0 + r) * a_g[:, None] + income[None, :] - a_g[vsol_you.policy_aprime]
gap_you = float(np.max(np.abs(c_vfi_you[:, mid] - egm_you.c[:, mid])))
min_slope = float(np.min(np.diff(egm_you.c[:, mid])))     # monotone consumption ⇒ ≥ 0
print(f"gamma={gamma_you}: max |c_EGM − c_VFI| = {gap_you:.3f}, "
      f"min Δc (EGM, over wealth) = {min_slope:+.4f}")
assert gap_you < 0.5                                       # solvers agree (VFI is grid-discretized)
assert min_slope >= -1e-9                                  # EGM consumption rises with wealth

# %% [markdown]
# **Ejercicios.** (1) Fija `gamma_you = 1.5` y `gamma_you = 10.0`; la brecha sigue
# siendo pequeña en ambos casos — confirma que el EGM y la VFI coinciden
# independientemente de la curvatura. (2) Con `gamma_you` alto la política de
# consumo del EGM se aplana (la precaución domina): compara `egm_you.c[:, mid]` en
# la base frente al tope de `a_g` para `gamma=2` y `gamma=10`. (3) *Extensión:* lee
# `egm_you.n_iter` para varios `gamma_you` — *aumenta* con la curvatura (el punto fijo
# se vuelve más rígido), pero cada iteración sigue siendo $O(n)$ en la malla (sin
# búsqueda de raíces por estado), que es la verdadera ventaja de velocidad del EGM.
#
# **¿Qué tan completa es la biblioteca?** `puremacro.vfi` es un kit de
# herramientas completo para agentes heterogéneos, y el EGM, Epstein–Zin y los
# múltiples estados endógenos son todos miembros de primera clase del mismo motor.
# Los demás cuadernos de muestra se apoyan en él: la desigualdad de riqueza de
# **Aiyagari / Huggett** en equilibrio general (NB01), **Krusell–Smith** con
# riesgo agregado y trayectorias de transición (NB02), **ciclo de vida / OLG** con
# mortalidad (NB03), y **Hopenhayn** con entrada y salida de firmas (NB04). Los
# tipos permanentes, el vaciamiento de mercado en EG y la dinámica de transición
# completan el kit.
