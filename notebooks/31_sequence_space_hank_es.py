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
# **¿Cómo se transmiten los choques de política monetaria en una economía donde los hogares enfrentan riesgo de ingreso no asegurable y restricciones de endeudamiento?**
#
# En los modelos tradicionales de Agente Representativo Nuevo Keynesiano (RANK), el consumo agregado está gobernado por una única ecuación de Euler correspondiente a un ahorrador no restringido con acceso perfecto a mercados financieros completos. Bajo estas condiciones, se cumple la equivalencia ricardiana frente a transferencias de suma fija y el mecanismo de transmisión de la política monetaria opera casi exclusivamente a través de la sustitución intertemporal: los hogares postergan su gasto corriente ante un aumento en la tasa de interés real.
#
# La evidencia microeconómica, sin embargo, revela una marcada heterogeneidad en los balances patrimoniales, la liquidez y las respuestas de gasto. Empíricamente, los hogares con escasa riqueza líquida presentan una **Propensión Marginal a Consumir (PMC)** de entre 40% y 60% ante choques transitorios de ingreso, mientras que los hogares de altos patrimonios exhiben una PMC inferior al 5%. En los modelos Heterogéneos Nuevo Keynesianos (HANK), esta dispersión genera potentes canales de retroalimentación indirecta: el endurecimiento monetario deprime la demanda laboral agregada, reduciendo desproporcionadamente el ingreso disponible de los trabajadores con restricciones de liquidez y desatando una contracción endógena del consumo agregado que amplifica sustancialmente el impacto directo de las tasas.
#
# Históricamente, resolver equilibrios generales en entornos con agentes heterogéneos requería seguir la evolución temporal de la distribución conjunta de riqueza y productividad, enfrentando la denominada *maldición de la dimensionalidad* (Krusell & Smith 1998). Adrien Auclert, Bence Bardóczy, Matthew Rognlie y Ludwig Straub (2021, *Econometrica*) transformaron este campo mediante el método de **Jacobianos en Espacio de Secuencias (SSJ)**. En lugar de resolver funciones de valor hacia atrás en cada punto del espacio de estados agregado, SSJ evalúa la respuesta lineal de dimensión infinita de los bloques heterogéneos directamente en el espacio de secuencias temporales en $O(T^3)$ operaciones mediante el Algoritmo de Noticias Falsas (*Fake News Algorithm*).
#
# En este tutorial interactivo calculamos la distribución estacionaria de riqueza mediante el Método de Cuadrícula Endógena (EGM), derivamos los Jacobianos de consumo en espacio de secuencias $\mathcal{J}_{C, Y}$ y $\mathcal{J}_{C, r}$, y resolvemos la transición de equilibrio general ante un choque contractivo de 25 puntos base usando `puremacro.models.solve_hank_sequence_space`.

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
# ## 1. Estado Estacionario y Método de Cuadrícula Endógena (EGM)
#
# Consideramos un continuo de hogares con horizonte infinito indexados por productividad laboral idiosincrásica $s_t \in \{s_L, s_H\}$, la cual sigue una cadena de Markov de dos estados con probabilidades de transición $\Pi(s, s')$. Los hogares maximizan su utilidad intertemporal esperada descontada:
#
# $$ \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \frac{c_t^{1-\gamma} - 1}{1-\gamma} $$
#
# sujeta a la restricción presupuestaria y a un límite estricto de endeudamiento que prohíbe posiciones netas de activos negativas:
#
# $$ c_t + a_{t+1} = (1 + r_{ss}) a_t + w_{ss} s_t, \quad a_{t+1} \ge 0 $$
#
# La condición de primer orden para la acumulación óptima de activos satisface la desigualdad de Euler:
#
# $$ u'(c_t) \ge \beta (1 + r_{ss}) \mathbb{E}_t \left[ u'(c_{t+1}) \right], \quad \text{con igualdad si } a_{t+1} > 0 $$
#
# Para resolver este problema sin recurrir a costosos algoritmos de optimización no lineal, `puremacro` implementa el **Método de Cuadrícula Endógena (EGM)** de Christopher Carroll (2006). A partir de una cuadrícula exógena de activos futuros $a_{t+1} \in [0, a_{\max}]$, el consumo óptimo se obtiene de manera analítica invirtiendo la función de utilidad marginal:
#
# $$ c_t(a_{t+1}, s_t) = \left( \beta (1 + r_{ss}) \sum_{s'} \Pi(s_t, s') u'\left(c_{t+1}^*(a_{t+1}, s')\right) \right)^{-1/\gamma} $$
#
# Los activos corrientes se recuperan entonces de forma endógena a través de la restricción presupuestaria $a_t = \frac{c_t + a_{t+1} - w_{ss} s_t}{1 + r_{ss}}$, interpolando posteriormente sobre la cuadrícula de estados de referencia. La distribución estacionaria de activos $\mathcal{D}^*(a, s)$ se calcula como el autovector invariante del operador de transición de Markov $\mathcal{T}^* \mathcal{D}^* = \mathcal{D}^*$ mediante la simulación no estocástica de Young (2010).

# %%
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,
    shock_rho=0.7,
    n_a=60,
)
print(res.summary())

# %% [markdown]
# ## 2. Distribución de la Propensión Marginal a Consumir (PMC) por Decil
#
# Un pilar fundamental de la macroeconomía de agentes heterogéneos es la distribución transversal de la Propensión Marginal a Consumir trimestral. Para cualquier hogar con activos $a$ y productividad $s$, la PMC ante una transferencia transitoria e inesperada $m$ se define como:
#
# $$ \text{PMC}(a, s) \equiv \lim_{m \to 0} \frac{c(a + m, s) - c(a, s)}{m} = \frac{\partial c(a, s)}{\partial a} \cdot \frac{1}{1 + r_{ss}} $$
#
# En una economía de agente representativo, la PMC agregada equivale al valor de anualidad de la riqueza $r / (1 + r) \approx 1\%$, lo que implica que los agentes suavizan cualquier transferencia transitoria a lo largo de su horizonte de vida completo. En HANK, los hogares situados en la frontera de endeudamiento $a \approx 0$ enfrentan restricciones activas; su utilidad marginal es sumamente empinada, forzándolos a consumir más del 50% de cualquier dólar adicional dentro del mismo trimestre.
#
# Como se observa en el gráfico por deciles a continuación, el 20% más pobre de la distribución presenta PMCs trimestrales superiores al 55%, mientras que los deciles superiores convergen hacia niveles reducidos consistentes con la hipótesis del ingreso permanente. Este gradiente microeconómico constituye el principal canal de transmisión de los multiplicadores fiscales y la retroalimentación monetaria.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.2))
res.mpc_distribution.plot(kind="bar", ax=ax, color=_nbstyle.TINTA, edgecolor=_nbstyle.SPINE, alpha=0.9)
ax.set_title("Propensión Marginal a Consumir (PMC) por Decil de Riqueza", fontsize=11, fontweight="bold")
ax.set_ylabel("PMC Trimestral", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 3. Jacobianos de Consumo en Espacio de Secuencias $\mathcal{J}_{C, r}$ y $\mathcal{J}_{C, Y}$
#
# En el espacio de secuencias, el bloque de consumo agregado $\mathbf{C} = (C_0, C_1, \dots, C_{T-1})^\top$ se modela como una función no lineal explícita de las trayectorias temporales completas del ingreso agregado $\mathbf{Y}$ y de la tasa de interés real $\mathbf{r}$. Linealizando alrededor del estado estacionario se obtienen las matrices Jacobianas:
#
# $$ d\mathbf{C} = \mathcal{J}_{C, Y} \, d\mathbf{Y} + \mathcal{J}_{C, r} \, d\mathbf{r} $$
#
# donde cada elemento $\mathcal{J}_{C, Y}[t, s] = \frac{\partial C_t}{\partial Y_s}$ cuantifica la variación del consumo agregado en el horizonte $t$ provocada por un aumento anticipado de una unidad en el ingreso laboral agregado en el horizonte $s$.
#
# El algoritmo de noticias falsas (*Fake News Algorithm*) evalúa estas matrices de dimensión $T \times T$ calculando eficientemente cómo se actualizan las expectativas de los hogares a medida que avanza la información:
# 1. $\mathcal{J}_{C, Y}$ (Panel Izquierdo): Presenta una clara dominancia diagonal. Debido a los hogares con alta PMC, un choque de ingreso en la fecha $s$ produce una respuesta contemporánea inmediata ($t = s$). Para $t < s$, el consumo responde moderadamente debido a los ahorradores prospectivos, mientras que para $t > s$ el consumo permanece elevado debido a los ahorros precautorios acumulados.
# 2. $\mathcal{J}_{C, r}$ (Panel Derecho): Modela la sustitución intertemporal y los efectos de flujo de caja. Un incremento en la tasa de interés en el horizonte $s$ induce a los hogares no restringidos a ahorrar, deprimiendo el consumo corriente ($t \le s$).

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.6))

im1 = ax1.imshow(res.jacobian_c_y[:15, :15], cmap=_nbstyle.CMAP_SEQ, origin="upper")
ax1.set_title(r"Jacobiano del Ingreso $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizonte del Choque s", color=_nbstyle.TEXTO)
ax1.set_ylabel("Horizonte de Respuesta t", color=_nbstyle.TEXTO)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors=_nbstyle.NOTA)

im2 = ax2.imshow(res.jacobian_c_r[:15, :15], cmap=_nbstyle.CMAP_SEQ_R, origin="upper")
ax2.set_title(r"Jacobiano de la Tasa de Interés $\mathcal{J}_{C, r}$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizonte del Choque s", color=_nbstyle.TEXTO)
ax2.set_ylabel("Horizonte de Respuesta t", color=_nbstyle.TEXTO)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors=_nbstyle.NOTA)

# %% [markdown]
# ## 4. Respuestas al Impulso en Equilibrio General ante un Alza de 25 pb
#
# Para cerrar el modelo en equilibrio general, el bloque heterogéneo se acopla a las ecuaciones neokeynesianas de producción y política monetaria en el espacio de secuencias:
#
# 1. **Curva de Phillips Neokeynesiana**: La inflación $\mathbf{\pi}$ responde al costo marginal mediante fijación de precios tipo Calvo con pendiente $\kappa$:
#    $$ \mathbf{\pi}_t = \beta \mathbb{E}_t \mathbf{\pi}_{t+1} + \kappa \left( \mathbf{Y}_t - Y_{ss} \right) $$
# 2. **Regla de Taylor**: El banco central determina la tasa nominal de política $\mathbf{i}_t$ reaccionando a la inflación con persistencia y una perturbación exógena $\mathbf{\epsilon}$:
#    $$ \mathbf{i}_t = r_{ss} + \phi_{\pi} \mathbf{\pi}_t + \mathbf{\epsilon}_t $$
# 3. **Ecuación de Fisher**: Tasa real ex-ante: $d\mathbf{r}_t = d\mathbf{i}_t - \mathbb{E}_t d\mathbf{\pi}_{t+1}$.
# 4. **Vaciado del Mercado de Bienes**: El producto agregado iguala al consumo privado: $\mathbf{H}(\mathbf{Y}) \equiv \mathbf{C}(\mathbf{Y}, \mathbf{r}(\mathbf{Y})) - \mathbf{Y} = \mathbf{0}$.
#
# Diferenciando la condición de vaciado con respecto a la secuencia de choques monetarios $\mathbf{\epsilon}$ se obtiene el sistema lineal en espacio de secuencias:
#
# $$ \left( \mathbf{I} - \mathcal{J}_{C, Y} - \mathcal{J}_{C, r} \mathbf{M}_{r, Y} \right) d\mathbf{Y} = \mathcal{J}_{C, r} d\mathbf{\epsilon} $$
#
# donde $\mathbf{M}_{r, Y} = \frac{\partial \mathbf{r}}{\partial \mathbf{Y}}$ resume los ajustes de equilibrio general de precios y política monetaria. El vector de transición $d\mathbf{Y}$ se obtiene mediante una única inversión matricial $(I - \mathbf{J}_{total})^{-1}$, requiriendo apenas milisegundos de cómputo.
#
# A diferencia de RANK (donde la contracción ocurre esencialmente por sustitución intertemporal pura), en HANK la caída del producto ($d\mathbf{Y}$) y del consumo ($d\mathbf{C}$) es más severa debido a que el canal indirecto refuerza la recesión: la menor producción reduce los ingresos salariales, forzando recortes inmediatos de gasto entre los hogares con restricciones de liquidez.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.4))
h = np.arange(len(res.irf_output))
ax.plot(h, res.irf_output * 100, **_nbstyle.S1, label=r"Producto $d\mathbf{Y}$ (%)")
ax.plot(h, res.irf_consumption * 100, **_nbstyle.S2, label=r"Consumo $d\mathbf{C}$ (%)")
ax.plot(h, res.irf_inflation * 100, **_nbstyle.S3, label=r"Inflación $d\mathbf{\pi}$ (%)")
ax.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax.set_title("Respuestas en Equilibrio General HANK ante Alza de 25 pb", fontsize=11, fontweight="bold")
ax.set_xlabel("Horizonte (Trimestres)", color=_nbstyle.TEXTO)
ax.set_ylabel("Desviación Porcentual (%)", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
