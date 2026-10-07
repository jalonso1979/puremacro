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
# # Modelos HANK en Espacio de Secuencias — Heterogeneidad sin Explosión de Estados
#
# **¿Cómo se transmiten los choques de política monetaria en una economía donde los hogares enfrentan riesgo de ingreso idiosincrásico no asegurable y restricciones de endeudamiento?**
#
# En un modelo Nuevo Keynesiano de Agente Representativo (RANK), el consumo agregado sigue una sola ecuación de Euler de un ahorrador sin restricciones. La política monetaria actúa entonces casi solo a través de la sustitución intertemporal: una tasa real más alta hace que los hogares pospongan su gasto.
#
# Con riesgo de ingreso no asegurable y un límite de endeudamiento, los hogares difieren en cuánto gastan de un peso adicional. Los que tienen poca riqueza gastan buena parte de inmediato; los hogares ricos gastan poco. En un modelo Nuevo Keynesiano de Agentes Heterogéneos (HANK) esta dispersión abre un segundo canal, indirecto: una política más restrictiva reduce el producto y el ingreso laboral, y los hogares con propensiones marginales a consumir (PMC) altas recortan aún más su gasto. Qué tan grande es ese canal depende de quién recibe el ingreso perdido, y este cuaderno lo mide.
#
# Resolver un modelo así exigía seguir toda la distribución de la riqueza como variable de estado (Krusell & Smith 1998). Auclert, Bardóczy, Rognlie y Straub (2021, *Econometrica*) linealizan en cambio en el *espacio de secuencias*: el bloque de los hogares se resume en jacobianos que transforman trayectorias completas de ingreso y de tasas de interés en una trayectoria de consumo, y el Algoritmo de Noticias Falsas (*Fake News Algorithm*) calcula cada jacobiano con un solo paso hacia atrás del problema del hogar. El equilibrio general se reduce entonces a un sistema lineal de $T \times T$.
#
# Calculamos la distribución estacionaria de la riqueza con el Método de Malla Endógena (EGM), obtenemos los jacobianos de consumo $\mathcal{J}_{C, Y}$ y $\mathcal{J}_{C, r}$ y resolvemos la respuesta de equilibrio general a un choque monetario de $0.25$ puntos porcentuales sobre la tasa de política trimestral (cerca de un punto en tasa anual) con `puremacro.models.solve_hank_sequence_space`. El modelo es una calibración de libro de texto incluida en la librería, no una economía estimada.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.models import solve_hank_sequence_space

# %% [markdown]
# ## 1. Estado Estacionario y Método de Malla Endógena (EGM)
#
# Un continuo de hogares con horizonte infinito difiere en su productividad laboral $s_t \in \{0.5, 1.5\}$, que sigue una cadena de Markov de dos estados que permanece en su estado actual con probabilidad $0.9$ cada trimestre (ambos valores están fijos en el código de la librería). Los hogares maximizan
#
# $$ \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \frac{c_t^{1-\gamma} - 1}{1-\gamma} $$
#
# sujetos a la restricción presupuestaria y a un límite de endeudamiento que excluye activos negativos:
#
# $$ c_t + a_{t+1} = (1 + r_t) a_t + (1 - \tau_t) w_t s_t, \quad a_{t+1} \ge 0 $$
#
# Los hogares tienen deuda pública $B$; los intereses $r_t B$ se pagan con un impuesto proporcional al ingreso laboral $\tau_t$. La condición de primer orden es la desigualdad de Euler:
#
# $$ u'(c_t) \ge \beta (1 + r_{t+1}) \mathbb{E}_t \left[ u'(c_{t+1}) \right], \quad \text{con igualdad si } a_{t+1} > 0 $$
#
# Para evitar la búsqueda no lineal de raíces, `puremacro` usa el **Método de Malla Endógena (EGM)** de Carroll (2006). Sobre una malla de activos al final del periodo $a_{t+1} \in [0, a_{\max}]$, el consumo se obtiene invirtiendo la utilidad marginal:
#
# $$ c_t(a_{t+1}, s_t) = \left( \beta (1 + r_{ss}) \sum_{s'} \Pi(s_t, s') u'\left(c_{t+1}^*(a_{t+1}, s')\right) \right)^{-1/\gamma} $$
#
# Los activos corrientes se obtienen de la restricción presupuestaria, $a_t = \frac{c_t + a_{t+1} - (1-\tau) w_{ss} s_t}{1 + r_{ss}}$, y la política se interpola de vuelta sobre la malla de activos. La distribución estacionaria $\mathcal{D}^*(a, s)$ es el vector invariante de la matriz de transición con loterías de Young (2010), $\mathcal{T}^* \mathcal{D}^* = \mathcal{D}^*$.

# %%
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,               # quarterly real rate (about 4% a year)
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,  # +0.25 pp on the quarterly policy rate
    shock_rho=0.7,
    n_a=60,
)
print(res.summary())

# %% [markdown]
# ## 2. Distribución de la Propensión Marginal a Consumir (PMC)
#
# La PMC trimestral ante una transferencia $m$ inesperada y única es
#
# $$ \text{PMC}(a, s) \equiv \lim_{m \to 0} \frac{c(a + m, s) - c(a, s)}{m} = \frac{\partial c(a, s)}{\partial a} \cdot \frac{1}{1 + r_{ss}} $$
#
# Un consumidor de ingreso permanente gasta aproximadamente el valor de anualidad $r/(1+r) \approx 1\%$ de una transferencia por trimestre. Los hogares en el límite de endeudamiento, o cerca de él, no pueden suavizar su consumo, así que gastan mucho más de inmediato.
#
# La celda siguiente imprime la PMC por decil de riqueza y lee el gradiente a partir de esos números. En esta calibración las PMC altas se concentran solo en el decil inferior: el decil siguiente ya se comporta mucho más como el resto de la distribución.

# %%
mpc = res.mpc_distribution
high = mpc[mpc > 0.5]
print(f"Deciles with a quarterly MPC above 0.5: {', '.join(high.index)} ({', '.join(f'{v:.4f}' for v in high)})")
print(f"Next decile: {mpc.index[len(high)]} ({mpc.iloc[len(high)]:.4f}); top decile: {mpc.iloc[-1]:.4f}; "
      f"aggregate MPC {res.steady_state_mpc:.4f}")

fig, ax = _nbstyle.figura(figsize=(8.5, 4.2))
mpc.plot(kind="bar", ax=ax, color=_nbstyle.TINTA, edgecolor=_nbstyle.SPINE, alpha=0.9)
ax.set_title("Marginal Propensity to Consume (MPC) by Wealth Decile", fontsize=11, fontweight="bold")
ax.set_ylabel("Quarterly MPC", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 3. Jacobianos de Consumo en Espacio de Secuencias $\mathcal{J}_{C, r}$ y $\mathcal{J}_{C, Y}$
#
# En el espacio de secuencias, la trayectoria del consumo agregado $\mathbf{C} = (C_0, C_1, \dots, C_{T-1})^\top$ es una función de las trayectorias del ingreso agregado $\mathbf{Y}$ y de la tasa de interés real $\mathbf{r}$. Linealizar alrededor del equilibrio estacionario da
#
# $$ d\mathbf{C} = \mathcal{J}_{C, Y} \, d\mathbf{Y} + \mathcal{J}_{C, r} \, d\mathbf{r} $$
#
# donde $\mathcal{J}_{C, Y}[t, s] = \partial C_t / \partial Y_s$ es la respuesta del consumo en la fecha $t$ a un aumento del ingreso agregado en la fecha $s$, anunciado en la fecha 0. El ingreso llega a los hogares a través del salario.
#
# El Algoritmo de Noticias Falsas calcula estas matrices de $T \times T$ con un paso hacia atrás del problema del hogar por cada insumo:
# 1. $\mathcal{J}_{C, Y}$ (panel izquierdo) tiene una diagonal marcada: los hogares gastan parte de un aumento de ingreso en el trimestre en que llega. Por encima de la diagonal ($t < s$), el consumo sube un poco por anticipación; por debajo ($t > s$), se mantiene más alto mientras los hogares gastan los ahorros que acumularon.
# 2. $\mathcal{J}_{C, r}$ (panel derecho) recoge la sustitución intertemporal y los efectos ingreso: una tasa real esperada más alta en la fecha $s$ lleva a los hogares sin restricciones a ahorrar más antes de $s$.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.6))

im1 = ax1.imshow(res.jacobian_c_y[:15, :15], cmap=_nbstyle.CMAP_SEQ, origin="upper")
ax1.set_title(r"Income Jacobian $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Shock Horizon s (quarters)", color=_nbstyle.TEXTO)
ax1.set_ylabel("Response Horizon t (quarters)", color=_nbstyle.TEXTO)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors=_nbstyle.NOTA)

im2 = ax2.imshow(res.jacobian_c_r[:15, :15], cmap=_nbstyle.CMAP_SEQ_R, origin="upper")
ax2.set_title(r"Interest Rate Jacobian $\mathcal{J}_{C, r}$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Shock Horizon s (quarters)", color=_nbstyle.TEXTO)
ax2.set_ylabel("Response Horizon t (quarters)", color=_nbstyle.TEXTO)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors=_nbstyle.NOTA)

# %% [markdown]
# ## 4. Respuestas de Equilibrio General a una Restricción Monetaria
#
# El bloque de los hogares se cierra con tres ecuaciones nuevo keynesianas estándar en el espacio de secuencias:
#
# 1. **Curva de Phillips Nuevo Keynesiana** con pendiente $\kappa$:
#    $$ \mathbf{\pi}_t = \beta \mathbb{E}_t \mathbf{\pi}_{t+1} + \kappa \left( \mathbf{Y}_t - Y_{ss} \right) $$
# 2. **Regla de Taylor** con una perturbación exógena de política $\mathbf{\epsilon}_t = 0.0025 \cdot 0.7^t$:
#    $$ \mathbf{i}_t = r_{ss} + \phi_{\pi} \mathbf{\pi}_t + \mathbf{\epsilon}_t $$
# 3. **Ecuación de Fisher** para la tasa real ex ante: $d\mathbf{r}_t = d\mathbf{i}_t - \mathbb{E}_t d\mathbf{\pi}_{t+1}$.
# 4. **Vaciado del Mercado de Bienes**: $\mathbf{H}(\mathbf{Y}) \equiv \mathbf{C}(\mathbf{Y}, \mathbf{r}(\mathbf{Y})) - \mathbf{Y} = \mathbf{0}$.
#
# Diferenciar el vaciado de mercado da un solo sistema lineal en la trayectoria del producto:
#
# $$ \left( \mathbf{I} - \mathcal{J}_{C, Y} - \mathcal{J}_{C, r} \mathbf{M}_{r, Y} \right) d\mathbf{Y} = \mathcal{J}_{C, r} d\mathbf{\epsilon} $$
#
# donde $\mathbf{M}_{r, Y} = \partial \mathbf{r} / \partial \mathbf{Y}$ reúne la curva de Phillips y la regla de Taylor. Como $d\mathbf{C} = d\mathbf{Y}$, la respuesta del consumo se divide exactamente en un canal **directo** (de tasa de interés) $\mathcal{J}_{C, r} \, d\mathbf{r}$ y un canal **indirecto** (de ingreso) $\mathcal{J}_{C, Y} \, d\mathbf{Y}$. La celda siguiente calcula ambos, para ver qué parte de la recesión explica el canal de ingreso en esta calibración.

# %%
dY = res.irf_output
dr = res.irf_rate                          # ex-ante real rate path
direct = res.jacobian_c_r @ dr             # interest-rate channel, income held at steady state
indirect = res.jacobian_c_y @ dY           # income channel, rates held at steady state
share_indirect_0 = indirect[0] / dY[0]
share_indirect_2y = indirect[:8].sum() / dY[:8].sum()   # first two years; later sums depend on the horizon T

print(f"Impact (quarter 0), % of steady-state output:")
print(f"  output = consumption : {dY[0] * 100:+.4f}")
print(f"  direct (rate) channel: {direct[0] * 100:+.4f}")
print(f"  indirect (income)    : {indirect[0] * 100:+.4f} ({share_indirect_0:.1%} of the impact)")
print(f"Sum over the first 8 quarters: indirect channel {share_indirect_2y:.1%} of the output loss")
print(f"Impact real rate {dr[0] * 100:+.4f} pp per quarter; impact inflation {res.irf_inflation[0] * 100:+.4f} pp per quarter")
print(f"J_CY[0,0] = {res.jacobian_c_y[0, 0]:.4f} against an aggregate MPC of {res.steady_state_mpc:.4f}")

# Internal accounting checks (both hold by construction of the GE solve, so a failure means a bug)
assert np.allclose(res.irf_consumption, dY, atol=1e-14), "goods market: dC = dY"
assert np.allclose(direct + indirect, dY, atol=1e-12), "dC = J_CY dY + J_Cr dr"
assert dY[0] < 0 and dr[0] > 0 and res.irf_inflation[0] < 0, "a tightening raises the real rate and lowers output and inflation"

fig, ax = _nbstyle.figura(figsize=(8.5, 4.6))
h = np.arange(len(dY))
ax.plot(h, dY * 100, **_nbstyle.S1, label=r"Output = consumption $d\mathbf{Y}$ (% of $Y_{ss}$)")
ax.plot(h, direct * 100, **_nbstyle.S2, label=r"Direct channel $\mathcal{J}_{C,r}\,d\mathbf{r}$ (% of $Y_{ss}$)")
ax.plot(h, indirect * 100, **_nbstyle.S3, label=r"Indirect channel $\mathcal{J}_{C,Y}\,d\mathbf{Y}$ (% of $Y_{ss}$)")
ax.plot(h, res.irf_inflation * 100, **_nbstyle.S4, label=r"Inflation $d\mathbf{\pi}$ (pp per quarter)")
ax.plot(h, dr * 100, **_nbstyle.S5, label=r"Ex-ante real rate $d\mathbf{r}$ (pp per quarter)")
ax.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax.set_title("HANK Responses to a +0.25 pp Quarterly Policy-Rate Shock", fontsize=11, fontweight="bold")
ax.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax.set_ylabel("Percent / percentage points", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE, fontsize=8, loc="lower right")
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.**
#
# 1. **Quién gasta.** La PMC trimestral agregada es $0.1022$, unas diez veces la referencia del ingreso permanente de $r/(1+r) \approx 1\%$. Ese promedio proviene de un solo decil. Solo el Decil 1 tiene una PMC mayor que un medio ($0.6643$); el Decil 2 ya baja a $0.0880$, y el decil superior gasta $0.0216$. Los hogares con PMC alta son los que no tienen riqueza y están en el límite de endeudamiento.
# 2. **Quién recibe el ingreso.** El primer elemento del jacobiano del ingreso, $\mathcal{J}_{C,Y}[0,0] = 0.0679$, está muy por debajo de la PMC agregada de $0.1022$, aunque ambos miden el gasto ante una ganancia sorpresa de un trimestre. El ingreso agregado llega a través del salario, en proporción a la productividad, de modo que la mayor parte va a los hogares de alta productividad, cuyas PMC son bajas. La indicación 3 reproduce la diferencia.
# 3. **La respuesta de equilibrio general.** El choque de política eleva la tasa real ex ante en $0.1338$ puntos porcentuales por trimestre en el impacto, menos que el choque de $0.25$ puntos, porque la inflación cae $0.1489$ puntos y la regla de Taylor responde en contra. El producto y el consumo caen $0.4335\%$ en el impacto. El canal directo de tasa de interés explica $-0.3684$ de esa caída, y el canal de ingreso $-0.0652$, es decir, $15.0\%$ del impacto. En los primeros 8 trimestres el canal de ingreso explica $29.8\%$ de la pérdida de producto. En esta calibración el canal de ingreso amplifica la recesión pero no la domina, porque los hogares que gastarían el ingreso perdido no son los que ganan la mayor parte de él.
# 4. **Qué muestran las verificaciones.** Las dos aserciones (el consumo es igual al producto, y los dos canales suman ese total) son identidades contables internas de la solución lineal. Muestran que la descomposición es completa, no que la calibración describa alguna economía. No se reportan sumas en horizontes más largos, porque con $T = 40$ todavía dependen de dónde se corta el horizonte.

# %% [markdown]
# ## Tu turno
#
# Los jacobianos de los hogares son objetos de equilibrio parcial: nunca ven la regla de Taylor. Por eso se puede cambiar la regla monetaria sin volver a resolver el problema del hogar. La celda reconstruye $\mathbf{M}_{r,Y}$ a partir de la curva de Phillips y la regla de Taylor, $d\mathbf{r} = (\phi_\pi \mathbf{K} - \mathbf{K}_{+1})\, d\mathbf{Y} + \boldsymbol{\epsilon}$ con $\mathbf{K}[t,s] = \kappa \beta^{s-t}$ para $s \ge t$, y resuelve el sistema lineal de la sección 4 con los jacobianos de la sección 1.
#
# **Prediga primero:** si el banco central responde con más fuerza a la inflación ($\phi_\pi > 1.5$), ¿la recesión en el impacto es más profunda o más leve que en la sección 4? La primera verificación compara la solución construida a mano con la que calcula la propia librería (una verificación interna). La segunda es su predicción: falla si la dirección es incorrecta.

# %%
# Your turn: change the monetary rule and reuse the Jacobians
phi_turn = 2.5   # ← change this: Taylor-rule response to inflation, 1.1 to 3.0
assert 1.1 <= phi_turn <= 3.0

Jy, Jr = res.jacobian_c_y, res.jacobian_c_r
T = Jy.shape[0]
t_idx = np.arange(T)
beta_nk, kappa_nk = 0.985, 0.1
K_pi = np.where(t_idx[None, :] >= t_idx[:, None], kappa_nk * beta_nk ** (t_idx[None, :] - t_idx[:, None]), 0.0)
K_next = np.vstack([K_pi[1:], np.zeros((1, T))])     # maps dY into E_t pi_{t+1}
eps = 0.0025 * 0.7 ** t_idx

def ge_output(phi):
    """Output path from (I - J_CY - J_Cr M) dY = J_Cr eps with dr = M dY + eps."""
    M = phi * K_pi - K_next
    return np.linalg.solve(np.eye(T) - Jy - Jr @ M, Jr @ eps)

dY_15, dY_turn = ge_output(1.5), ge_output(phi_turn)
ref = solve_hank_sequence_space(T=T, beta=0.985, gamma=1.0, r_ss=0.01, phi_pi=phi_turn, kappa=0.1,
                                shock_magnitude=0.0025, shock_rho=0.7, n_a=60)
print(f"Impact output: phi = 1.5 -> {dY_15[0] * 100:+.4f}%, phi = {phi_turn} -> {dY_turn[0] * 100:+.4f}% "
      f"(library re-solve {ref.irf_output[0] * 100:+.4f}%)")
print(f"Cumulative output over {T} quarters: {dY_15.sum() * 100:+.3f} vs {dY_turn.sum() * 100:+.3f} (% of quarterly Y_ss)")

# Internal check: the hand-built GE system is the library's
assert np.max(np.abs(dY_15 - res.irf_output)) < 1e-12
assert np.max(np.abs(dY_turn - ref.irf_output)) < 1e-12
# Your prediction: a more hawkish rule gives a shallower impact recession, a more dovish one a deeper one
assert np.sign(dY_turn[0] - dY_15[0]) == np.sign(phi_turn - 1.5), "the direction of the change is wrong"

# %% [markdown]
# **Indicaciones.**
# 1. *Básico.* Un aumento del ingreso agregado en el trimestre 0 tiene que gastarse tarde o temprano. Antes de calcular, use la restricción presupuestaria intertemporal para predecir $x = \sum_t (1 + r_{ss})^{-t} \mathcal{J}_{C,Y}[t, 0]$. Calcúlelo con `q = (1 + res.r_ss) ** -np.arange(T); x = q @ Jy[:, 0]` y luego resuelva de nuevo con `T=200` (lo demás igual que en la sección 1). ¿A dónde va la parte que falta con $T = 40$? Autoverificación: `assert x < 0.8` con $T = 40$ y `abs(x - 1) < 0.01` con $T = 200$.
# 2. *Intermedio.* Explique la dirección que encontró. Grafique `ge_output(phi)` y la tasa real implícita `(phi * K_pi - K_next) @ ge_output(phi) + eps` para $\phi_\pi \in \{1.25, 1.5, 2.5\}$. ¿Por qué el mismo choque de política eleva menos la tasa real bajo una regla más restrictiva?
# 3. *Avanzado.* La PMC agregada que imprime la sección 1 y $\mathcal{J}_{C,Y}[0,0]$ miden ambos el gasto ante una ganancia sorpresa de ingreso de un trimestre, pero difieren. Calcule la PMC de cada hogar con `m = np.diff(res.policy_c, axis=0) / ((1 + res.r_ss) * np.diff(res.asset_grid))[:, None]`, repita la última fila, recórtela a $[0, 1]$ y promédiela con pesos `res.distribution` (la PMC simple) y con pesos `res.distribution * s` para $s = (0.5, 1.5)$ (la parte de un aumento salarial que recibe cada hogar). Autoverificación: `abs(mpc_plain - res.steady_state_mpc) < 1e-10` y `abs(mpc_earn - Jy[0, 0]) / Jy[0, 0] < 0.05`. ¿Qué dice esto sobre el tamaño del canal indirecto?
#
# ## ¿Qué tan exhaustivo es esto?
#
# - `puremacro.models.solve_hank_sequence_space` también devuelve `fake_news`, `simulate_transfer` (transferencias fiscales focalizadas) y `solve_nonlinear` (una transición MIT no lineal que parte de esta solución lineal).
# - `puremacro.dsge.hank` conecta los jacobianos de los hogares con el conjunto de herramientas DSGE, y el cuaderno 44 los vincula a un modelo DSGE; el cuaderno 46 usa la versión de dos activos.
# - El cuaderno 52 resuelve la transición no lineal de una economía de Aiyagari con riqueza continua después de un choque MIT, la contraparte no lineal de las respuestas lineales de aquí.
