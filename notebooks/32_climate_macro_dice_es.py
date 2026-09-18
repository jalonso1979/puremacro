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
# # Macroeconomía y Cambio Climático — El Modelo DICE y la Fijación de Precios al Carbono
#
# **¿Cómo interactúan el crecimiento económico, las emisiones de gases de efecto invernadero y la retroalimentación climática a horizontes de un siglo, y cuál es la trayectoria óptima del Costo Social del Carbono?**
#
# Durante el próximo siglo, los agregados macroeconómicos y la dinámica biofísica del clima estarán estrechamente interconectados. La combustión de combustibles fósiles genera emisiones antropogénicas de gases de efecto invernadero que se acumulan en reservorios atmosféricos, intensifican el forzamiento radiativo y elevan la temperatura media de la superficie terrestre. Este calentamiento, a su vez, genera severos daños macroeconómicos: caída en rendimientos agrícolas, depreciación acelerada del capital físico ante fenómenos meteorológicos extremos, pérdida de productividad laboral en zonas cálidas y colapso de servicios ecosistémicos.
#
# Para analizar este ciclo de retroalimentación a escala multisecular, William Nordhaus (Premio Nobel de Economía 2018) desarrolló el modelo de **Clima y Economía Integrados Dinámicamente (DICE)**. En DICE, un planificador de tipo Ramsey resuelve el dilema intertemporal entre consumo presente, acumulación de capital y mitigación de emisiones:
# 1. **Costos de Mitigación**: Reducir emisiones exige desplegar tecnologías limpias, sustituir capital fósil e incrementar la eficiencia energética, lo cual desvía recursos del consumo corriente y la inversión productiva.
# 2. **Daños Climáticos Evitados**: Abatir emisiones disminuye la concentración atmosférica de carbono a largo plazo, conteniendo el calentamiento global y preservando la capacidad productiva futura.
#
# Extensiones modernas (Golosov, Hassler, Krusell y Tsyvinski 2014, *Econometrica*; Nordhaus 2017, *PNAS*) formalizan el concepto del **Costo Social del Carbono (CSC)**: el valor presente descontado de todos los daños marginales futuros causados sobre el producto bruto mundial por una tonelada adicional de $\text{CO}_2$ emitida hoy. El CSC determina el precio pigouviano óptimo que debe fijarse sobre las emisiones mediante impuestos ambientales o sistemas de permisos negociables.
#
# En este tutorial interactivo simulamos el modelo DICE-2016R a lo largo de un horizonte de 150 años (2020–2170) mediante `puremacro.climate.simulate_dice_model`, contrastando una trayectoria base moderada frente a un programa acelerado de impuestos al carbono orientado a cumplir con las metas climáticas internacionales.

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

from puremacro.climate import simulate_dice_model

# %% [markdown]
# ## 1. Simulación del Modelo DICE: Escenario Base vs. Impuesto Acelerado al Carbono
#
# La estructura geofísica y económica del modelo DICE se compone de cuatro módulos acoplados:
#
# 1. **Producción Global y Producto Bruto**:
#    El producto mundial bruto $Y_{gross, t}$ sigue una tecnología agregada Cobb-Douglas sobre capital $K_t$ y trabajo $L_t$, potenciada por la productividad total de los factores exógena $A_t$:
#    $$ Y_{gross, t} = A_t K_t^\gamma L_t^{1-\gamma} $$
# 2. **Dinámica del Ciclo del Carbono (Sistema de Tres Reservorios)**:
#    Las emisiones industriales $E_t = \sigma_t (1 - \mu_t) Y_{gross, t} + E_{land, t}$ fluyen a la atmósfera, donde $\sigma_t$ es la intensidad de carbono del producto y $\mu_t \in [0, 1]$ es la tasa de control de emisiones. El vector de reservas de carbono $\mathbf{M}_t = (M_{AT, t}, M_{UP, t}, M_{LO, t})^\top$ evoluciona mediante la matriz lineal de transferencia $\mathbf{\Phi}_M$:
#    $$ \mathbf{M}_{t+1} = \mathbf{\Phi}_M \mathbf{M}_t + \begin{pmatrix} E_t \\ 0 \\ 0 \end{pmatrix} $$
# 3. **Forzamiento Radiativo y Océano Térmico de Dos Capas**:
#    La concentración atmosférica se traduce en forzamiento radiativo $F_t = \eta \log_2 \left( \frac{M_{AT, t}}{M_{AT, pre}} \right) + F_{EX, t}$, gobernando las temperaturas atmosférica $T_{AT, t}$ y del océano profundo $T_{LO, t}$.
# 4. **Producto Neto con Costos de Abatimiento y Daños Climáticos**:
#    El producto disponible para consumo e inversión neta es:
#    $$ Y_{net, t} = \left[ 1 - \Lambda_t(\mu_t) \right] \left[ 1 - \Omega(T_{AT, t}) \right] Y_{gross, t} $$
#    donde la función de daño cuadrático de Nordhaus es $\Omega(T_t) = \frac{\pi_1 T_t + \pi_2 T_t^2}{1 + \pi_1 T_t + \pi_2 T_t^2}$ y la función de costos de abatimiento es $\Lambda_t(\mu_t) = \theta_{1, t} \mu_t^{\theta_2}$.
#
# A continuación simulamos:
# - **Escenario Base**: Impuesto inicial al carbono de $\$40/\text{tCO}_2$ con crecimiento anual del $2.0\%$.
# - **Escenario con Impuesto Acelerado**: Impuesto inicial de $\$80/\text{tCO}_2$ con crecimiento anual del $3.5\%$.

# %%
res_base = simulate_dice_model(
    n_periods=30,
    time_step_years=5,
    start_year=2020,
    carbon_tax_initial=40.0,
    carbon_tax_growth=0.02,
)
print("=== Escenario Base ===")
print(res_base.summary())

res_policy = simulate_dice_model(
    n_periods=30,
    time_step_years=5,
    start_year=2020,
    carbon_tax_initial=80.0,
    carbon_tax_growth=0.035,
)
print("\n=== Escenario con Impuesto Acelerado ===")
print(res_policy.summary())

# %% [markdown]
# ## 2. Anomalías de Calentamiento Superficial y Metas del Acuerdo de París
#
# En el marco del Acuerdo de París de 2015, la comunidad internacional se comprometió a mantener el incremento de la temperatura media global muy por debajo de $2.0^\circ\text{C}$ respecto a niveles preindustriales y a desplegar esfuerzos para limitar el calentamiento a $1.5^\circ\text{C}$.
#
# En el Escenario Base ($\$40/\text{tCO}_2$ creciendo al $2\%$), la mitigación es insuficiente para evitar un cambio climático peligroso: la anomalía térmica global rebasa el umbral de $1.5^\circ\text{C}$ hacia 2040, cruza la barrera de $2.0^\circ\text{C}$ antes de 2060 y alcanza $+3.36^\circ\text{C}$ para 2100, extendiéndose por encima de $+4.1^\circ\text{C}$ en el siglo XXII.
#
# En contraste, el Escenario de Impuesto Acelerado frena rápidamente la curva de emisiones: el calentamiento máximo se contiene cerca de $2.1^\circ\text{C}$ hacia 2085 antes de estabilizarse, mitigando sustancialmente los riesgos de puntos de inflexión planetarios irreversibles.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.4))
ax.plot(res_base.trajectories.index, res_base.trajectories["temperature_anomaly"], **_nbstyle.S1, label=r"Escenario Base ($\$40/\mathrm{t}, +2\%/\mathrm{año}$)")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["temperature_anomaly"], **_nbstyle.S2, label=r"Política Acelerada ($\$80/\mathrm{t}, +3.5\%/\mathrm{año}$)")
ax.axhline(1.5, color=_nbstyle.SPINE, linestyle=":", label=r"Objetivo París $1.5^\circ\mathrm{C}$")
ax.axhline(2.0, color=_nbstyle.NOTA, linestyle="-.", label=r"Límite Crítico $2.0^\circ\mathrm{C}$")
ax.set_title("Anomalía de Temperatura Superficial Global (°C sobre nivel preindustrial)", fontsize=11, fontweight="bold")
ax.set_xlabel("Año", color=_nbstyle.TEXTO)
ax.set_ylabel("Anomalía de Calentamiento (°C)", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 3. Emisiones Globales de CO2 y Trayectorias de Descarbonización
#
# El cronograma de descarbonización ilustra el impacto directo del precio del carbono sobre las emisiones industriales. La tasa de control de emisiones $\mu_t$ responde de forma endógena igualando el costo marginal de abatimiento con el impuesto vigente:
#
# $$ \text{MAC}_t(\mu_t) = \frac{\partial \Lambda_t}{\partial \mu_t} \cdot \frac{Y_{gross, t}}{\sigma_t Y_{gross, t}} = \frac{\theta_{1, t} \theta_2 \mu_t^{\theta_2 - 1}}{\sigma_t} = \tau_t $$
#
# En el Escenario Base, las emisiones permanecen por encima de $30\,\text{GtCO}_2/\text{año}$ hasta 2080, ya que el crecimiento económico contrarresta la moderada tasa de abatimiento. En el escenario acelerado, las emisiones anuales alcanzan su punto máximo de forma casi inmediata (hacia 2025 en $32\,\text{GtCO}_2/\text{año}$) y colapsan hacia cero neto en las últimas décadas del siglo, logrando un desacoplamiento estructural entre crecimiento del PIB y emisiones contaminantes.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.4))
ax.plot(res_base.trajectories.index, res_base.trajectories["emissions"], **_nbstyle.S1, label="Emisiones Anuales Base")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["emissions"], **_nbstyle.S2, label="Descarbonización Acelerada")
ax.set_title("Emisiones Globales Antropogénicas de CO2 (GtCO2/año)", fontsize=11, fontweight="bold")
ax.set_xlabel("Año", color=_nbstyle.TEXTO)
ax.set_ylabel("Emisiones (GtCO2)", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 4. Daños Macroeconómicos (% del Producto Mundial Bruto)
#
# Los daños económicos en el modelo DICE cuantifican la producción perdida como consecuencia del deterioro ambiental y el estrés térmico:
#
# $$ \text{Daños}_t = \Omega(T_{AT, t}) \, Y_{gross, t} $$
#
# Bajo la trayectoria Base, el calentamiento continuo impone un costo creciente sobre la economía global: los daños climáticos escalan del $0.38\%$ del PIB en 2025 al $2.66\%$ en 2100 y superan el $4.2\%$ hacia 2150. En términos acumulados, esto representa billones de dólares de producto económico destruido anualmente.
#
# Bajo el Impuesto Acelerado al Carbono, si bien el gasto inicial en mitigación $\Lambda_t Y_{gross, t}$ es más alto en las primeras décadas, los daños climáticos quedan acotados de forma permanente por debajo del $1.1\%$ del PIB. La ganancia neta en bienestar social—medida por la integral de utilidad descontada $\sum_{t} \beta^t u(C_t)$—confirma que una política climática temprana y decidida genera retornos netos altamente positivos para la economía global.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.4))
ax.plot(res_base.trajectories.index, res_base.trajectories["damage_fraction"] * 100, **_nbstyle.S1, label="Proporción de Daños Base")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["damage_fraction"] * 100, **_nbstyle.S2, label="Proporción de Daños con Política Acelerada")
ax.set_title("Daños Climáticos Macroeconómicos (% del Producto Mundial Bruto)", fontsize=11, fontweight="bold")
ax.set_xlabel("Año", color=_nbstyle.TEXTO)
ax.set_ylabel("Proporción de Daños (% del PIB)", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
