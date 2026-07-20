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
# # Las empresas también son heterogéneas: entrada, salida y selección
#
# Los métodos de agentes heterogéneos no se aplican únicamente a los hogares. En **Hopenhayn (1992)**,
# las empresas extraen productividad persistente, pagan un costo fijo de operación y salen del mercado cuando
# su valor esperado se vuelve negativo — un umbral endógeno. La libre entrada determina el precio del producto.
# El resultado es selección: los incumbentes que sobreviven son más productivos que los entrantes.

# %% [markdown]
# ## El modelo en cuatro ecuaciones
#
# **Incumbentes.** El único estado de una empresa es su productividad $s$, un AR(1) discretizado
# mediante Tauchen en una grilla con transición $P$. Cada período obtiene un beneficio de operación
# $\pi(s;p)$ al precio del producto $p$ y luego decide, *al final del período*, si continúa o cierra
# por un valor de salida de $0$:
# $$ W(s) = \pi(s;p) + \beta\,\max\!\bigl\{0,\ \mathbb{E}\!\left[W(s')\mid s\right]\bigr\}. $$
# Con rendimientos decrecientes, $\pi(s;p)=\max_{n}\,p\,s\,n^{\alpha}-n-c_f$ tiene la forma cerrada
# $\pi(s;p)=(1-\alpha)\,(\alpha p s)^{\frac{1}{1-\alpha}}/\alpha-c_f$ —
# creciente en $s$, de modo que la empresa continúa si y solo si $\mathbb{E}[W(s')\mid s]\ge 0$, es decir,
# por encima de un **umbral de salida** $s^\*$ (selección).
#
# **Libre entrada.** Un entrante potencial paga $c_e$ y extrae su productividad de $\nu$. La entrada
# ocurre hasta que el valor esperado de entrar iguala su costo, lo que determina el precio $p^\*$:
# $$ \mathbb{E}_{s\sim\nu}\!\left[W(s;p)\right] = c_e. $$
# Como $\mathbb{E}_\nu[W]$ es creciente en $p$, esto se equilibra en un único $p^\*$.
#
# **Distribución estacionaria.** Los sobrevivientes transitan según $P$; cada período entra una masa fija
# de entrantes con distribución $\nu$ y salen los que cierran, por lo que la medida de empresas $g$ resuelve
# $$ g = g\,S + \nu,\qquad S[s,s'] = \mathbb{1}\{s\ \text{sobrevive}\}\;P[s,s'], $$
# un punto fijo que *no* conserva la masa ($S$ es subestocástica), normalizado a una densidad.
#
# **Intuición.** El beneficio crece con la productividad, por lo que solo las empresas por encima del umbral
# $s^\*$ esperan un valor de continuación no negativo — el mercado *elimina* a las empresas de baja
# productividad, dejando incumbentes sistemáticamente más productivos que la distribución de entrantes $\nu$.
# La libre entrada es el margen de beneficio nulo del modelo: si entrar valiera más que $c_e$, los entrantes
# afluirían, aumentarían la oferta y empujarían el precio $p$ a la baja hasta que el valor de entrar se ajuste
# exactamente a $c_e$. Los dos márgenes se mueven en conjunto bajo la estática comparativa — aumentar el
# **costo fijo** $c_f$ reduce el beneficio de cada empresa, de modo que el umbral $s^\*$ sube y la distribución
# estacionaria se desplaza hacia sobrevivientes más productivos; aumentar el **costo de entrada** $c_e$, en
# cambio, exige un precio $p^\*$ más alto para que la entrada siga siendo rentable.

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
    tauchen, markov_stationary,
    firm_value_with_exit, firm_stationary_distribution, free_entry_price,
)

# %% [markdown]
# ## 1. Solución del equilibrio de libre entrada
# Construimos el modelo directamente para conservar todos los objetos (grilla de productividad, distribución
# de entrantes, función de valor, máscara de supervivencia) para los gráficos. El beneficio con rendimientos
# decrecientes a escala tiene una forma cerrada.

# %%
alpha, beta, cf, ce = 2.0 / 3.0, 0.8, 20.0, 40.0
rho, sigma, log_zbar, n_s = 0.9, 0.2, 1.4, 101
z_grid, P = tauchen(n=n_s, rho=rho, sigma=sigma)
s = np.exp(z_grid + log_zbar)                              # productivity levels
nu = markov_stationary(P)                                 # entrant productivity draw

def profit_at(p):
    n_star = (alpha * p * s) ** (1.0 / (1.0 - alpha))
    return p * s * n_star ** alpha - n_star - cf

eq = free_entry_price(profit_at, nu, ce, (0.5, 20.0), P_z=P, beta=beta)
V, survive = firm_value_with_exit(profit_at(eq.price), P, beta)
g = firm_stationary_distribution(P, survive, nu)
exit_rate = float(g @ (~survive).astype(float))
mean_inc = float(g @ s); mean_ent = float(nu @ s)
print(f"price = {eq.price:.3f}, free-entry residual = {eq.residual:.2e}")
print(f"exit rate = {exit_rate:.3f}; mean productivity: incumbents {mean_inc:.2f} "
      f"vs entrants {mean_ent:.2f}")
assert abs(eq.residual) < 1e-2                            # free entry clears
assert 0.0 < exit_rate < 1.0
assert mean_inc > mean_ent                                # selection

# %% [markdown]
# **Lectura del resultado.** El residuo de libre entrada $\mathbb{E}_\nu[W]-c_e$ es ~0, así que el
# precio impreso es el $p^\*$ que hace que entrar valga exactamente su costo — no queda entrada rentable
# sobre la mesa. La tasa de salida (la masa estacionaria ubicada en productividades por debajo del umbral
# $s^\*$) está estrictamente entre 0 y 1: algunas empresas extraen mal y cierran, pero el mercado no
# colapsa. El dato decisivo es la brecha de productividad — los *incumbentes* son en promedio más
# productivos que la distribución de *entrantes* $\nu$. Esa brecha es la selección: los entrantes de baja
# productividad o no sobreviven a sus primeras malas extracciones o son expulsados, de modo que la población
# sobreviviente queda inclinada hacia arriba respecto de quién cruza la puerta.

# %% [markdown]
# ### Figura principal — selección
# Los incumbentes (distribución estacionaria) se ubican a la derecha de la distribución de entrantes: el
# mercado elimina a las empresas de baja productividad. El umbral de salida es el nivel mínimo de
# productividad en el que una empresa aún elige continuar operando.

# %%
s_thresh = s[survive][0]                                  # lowest surviving productivity
cols = _nbstyle.palette(2)
fig, ax = plt.subplots()
ax.plot(s, g, color=cols[0], label="incumbents (stationary)")
ax.plot(s, nu, color=cols[1], linestyle="--", label="entrant draw ν")
ax.axvline(s_thresh, color="0.3", linewidth=0.9, linestyle=":",
           label=f"exit threshold s={s_thresh:.2f}")
ax.set_xlim(s.min(), np.quantile(s, 0.97))
ax.set_xlabel("Productivity s"); ax.set_ylabel("Density")
ax.set_title("Firm productivity: selection at work"); ax.legend()
plt.show()

# %% [markdown]
# ### Complementaria — función de valor y la región de salida
# Por debajo del umbral, el valor de continuación es negativo, por lo que la empresa sale del mercado
# (la función de valor queda truncada en 0).

# %%
cont = P @ V                                              # E[V'|s]
fig, ax = plt.subplots()
ax.plot(s, V, color="0.0", label="incumbent value V(s)")
ax.axhline(0.0, color="0.5", linewidth=0.8)
ax.fill_between(s, V.min(), 0.0, where=~survive, color="0.85",
                label="exit region")
ax.set_xlim(s.min(), np.quantile(s, 0.9))
ax.set_xlabel("Productivity s"); ax.set_ylabel("Value")
ax.set_title("Value function and exit decision"); ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### Complementaria — estática comparativa en el costo de entrada
# Un mayor costo de entrada requiere un precio del producto más alto para que la entrada sea rentable.

# %%
ces = np.linspace(20.0, 80.0, 7)
prices = [free_entry_price(profit_at, nu, c, (0.5, 30.0), P_z=P, beta=beta).price
          for c in ces]
assert np.all(np.diff(prices) > 0)                        # price increasing in ce
fig, ax = plt.subplots()
ax.plot(ces, prices, color="0.0", marker="o", markersize=4)
ax.set_xlabel("Entry cost cₑ"); ax.set_ylabel("Equilibrium output price")
ax.set_title("Comparative statics: entry cost → price")
plt.show()

# %% [markdown]
# **Lectura del resultado.** La línea tiene pendiente estrictamente positiva, y el `assert` lo confirma:
# un costo de entrada $c_e$ más alto corresponde a un precio de equilibrio $p^\*$ más alto. La lógica es la
# condición de libre entrada $\mathbb{E}_\nu[W(s;p)]=c_e$ — si encarece la entrada, la única forma de
# mantener el *valor* de entrar igual a ese costo mayor es que el precio del producto (y por ende el beneficio
# de cada empresa) sea más alto. Así, una industria más cara de penetrar es una en la que los incumbentes
# ganan más por unidad y el piso de precios es más alto.

# %% [markdown]
# ## Tu turno — el costo fijo desplaza el umbral de supervivencia
#
# Mantén fijo el precio de equilibrio $p^\*$ y pregúntate: ¿quién sobrevive si operar una empresa se vuelve
# más costoso? Aumentar el costo de operación $c_f$ reduce el beneficio de cada empresa, de modo que el
# sobreviviente marginal es empujado hacia arriba en la escala de productividad — el umbral de salida $s^\*$
# sube y la proporción de supervivencia cae. Esto es barato (sin recalcular el equilibrio general): solo
# recomputamos la función de valor al *mismo* precio con un costo fijo más pesado. Cambia `cf_you`.

# %%
# ← Change this operating cost (baseline cf = 20.0). Try values at or above the baseline.
cf_you = 30.0
profit_you = profit_at(eq.price) - (cf_you - cf)          # same price, heavier fixed cost
V_you, survive_you = firm_value_with_exit(profit_you, P, beta)
g_you = firm_stationary_distribution(P, survive_you, nu)
cutoff_you = s[survive_you][0]                            # lowest surviving productivity
share_you = float(survive_you.mean())
print(f"cf = {cf_you}: exit cutoff s* = {cutoff_you:.3f} (baseline {s_thresh:.3f}), "
      f"survival share = {share_you:.3f} (baseline {survive.mean():.3f})")
# Raising the fixed cost weakly raises the cutoff and weakly lowers the survival share:
assert cutoff_you >= s_thresh - 1e-9                      # cutoff (weakly) climbs
assert share_you <= survive.mean() + 1e-9                 # fewer firms survive
assert abs(g_you.sum() - 1.0) < 1e-9                      # still a valid density
assert s.min() <= cutoff_you <= s.max()                   # cutoff stays on the grid

# %% [markdown]
# **Ejercicios.** (1) Lleva `cf_you` a `45.0` — el umbral sube aún más y la proporción de supervivencia
# sigue cayendo; confirma que ambos `assert` se mantienen. (2) Fija `cf_you = 20.0` (el valor base): el
# umbral y la proporción deben coincidir exactamente con los números principales. (3) *Avanzado* (recalcula
# el equilibrio): construye un nuevo cierre `profit_at` con el nuevo costo fijo, vuelve a ejecutar
# `free_entry_price` y comprueba que el *precio de equilibrio* también sube con `cf` — el margen de entrada
# y el margen de salida se mueven en conjunto.
#
# **¿Qué tan completo es esto?** `puremacro.vfi` es una caja de herramientas completa de agentes
# heterogéneos, y el bloque de empresas de aquí es una de sus esquinas: la misma cadena de resolución →
# distribución estacionaria → vaciado de mercado impulsa a los hogares de mercados incompletos de
# **Aiyagari** (NB01), el riesgo agregado y las trayectorias de transición de **Krusell-Smith** (NB02), el
# **ciclo de vida / OLG** con mortalidad (NB03) y las carteras de **dos activos** con grillas endógenas
# (EGM) y preferencias de Epstein-Zin (NB05).
