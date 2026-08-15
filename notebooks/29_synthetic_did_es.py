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
# # Diferencias-en-Diferencias Sintéticas — Unificando Control Sintético y TWFE
#
# **¿Cómo evaluar políticas causales cuando fallan las tendencias paralelas y los controles difieren de la unidad tratada?**
# Dos de las metodologías de evaluación causal más influyentes en economía empírica son:
# 1. **Diferencias en Diferencias (DiD)**: Supone que los resultados potenciales no tratados siguen tendencias paralelas a lo largo del tiempo, apoyándose en efectos fijos aditivos de unidad y tiempo.
# 2. **Método de Control Sintético (SCM)**: Resuelve el incumplimiento de tendencias paralelas buscando ponderaciones no negativas $\omega$ de donantes que reproduzcan la trayectoria pre-tratamiento. Sin embargo, SCM sufre cuando la unidad tratada se encuentra fuera de la envolvente convexa de controles y carece de invarianza ante traslaciones de nivel.
#
# Dmitry Arkhangelsky, Susan Athey, David Hirshberg, Guido Imbens y Stefan Wager (2021, *American Economic Review*) introdujeron las **Diferencias en Diferencias Sintéticas (SDID)**, que unifican SCM y DiD.
#
# Este tutorial demuestra cómo utilizar `puremacro.did.synthetic_did` para evaluar intervenciones de política mediante ponderaciones de unidades $\hat{\omega}$, ponderaciones temporales $\hat{\lambda}$ y ajuste de intercepto.

# %% [markdown]
# ## El método en matemáticas — Mínimos Cuadrados con Doble Ponderación
#
# Sea $Y_{i,t}$ el resultado para la unidad $i \in \{1, \dots, N\}$ en el período $t \in \{1, \dots, T\}$, con una cohorte tratada en el período $T_{\text{post}}$.
#
# ### 1. Ponderaciones de Unidades $\hat{\omega}$ (Alineación de Trayectorias)
# Buscamos ponderaciones de donantes $\hat{\omega} \ge 0, \sum_{i=2}^N \hat{\omega}_i = 1$ para emparejar la trayectoria pre-tratamiento de la unidad tratada:
# $$ \hat{\omega} = \arg\min_{\omega \in \Delta^{N-1}} \sum_{t < T_{\text{post}}} \left( \sum_{i \in \text{Donantes}} \omega_i Y_{i,t} - Y_{\text{tratada},t} \right)^2 + \zeta \|\omega\|_2^2 $$
#
# ### 2. Ponderaciones Temporales $\hat{\lambda}$ (Balance Pre vs. Post Períodos)
# Buscamos ponderaciones temporales $\hat{\lambda} \ge 0, \sum_{t < T_{\text{post}}} \hat{\lambda}_t = 1$ para balancear los períodos previos frente a los períodos posteriores en los donantes:
# $$ \hat{\lambda} = \arg\min_{\lambda \in \Delta^{T_{\text{pre}}-1}} \sum_{i \in \text{Donantes}} \left( \sum_{t < T_{\text{post}}} \lambda_t Y_{i,t} - \bar{Y}_{i,\text{post}} \right)^2 $$
#
# ### 3. Estimador del Efecto de Tratamiento SDID
# El efecto causal estimado $\hat{\tau}_{\text{SDID}}$ es:
# $$ \hat{\tau}_{\text{SDID}} = \left(\bar{Y}_{\text{tratada},\text{post}} - \sum_{i \in \text{Donantes}} \hat{\omega}_i \bar{Y}_{i,\text{post}}\right) - \sum_{t < T_{\text{post}}} \hat{\lambda}_t \left(Y_{\text{tratada},t} - \sum_{i \in \text{Donantes}} \hat{\omega}_i Y_{i,t}\right) $$
# Este estimador es doblemente robusto: es insesgado si las ponderaciones de unidades o las temporales logran el balance, y absorbe diferencias de nivel permanente.

# %% [markdown]
# ## Configuración — Simulación con Controles no Paralelos

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

from puremacro.did import synthetic_did

rng = np.random.default_rng(2021)

N_donors = 15
N_units = N_donors + 1  # La unidad 0 es tratada
T_periods = 20
T_treat = 12

# Factores latentes comunes
time_trend = np.linspace(10, 25, T_periods)
macro_factor = np.sin(np.linspace(0, 3 * np.pi, T_periods)) * 3.0

# Cargas factoriales
unit_fixed_effects = rng.uniform(5, 20, size=N_units)
factor_loadings = rng.uniform(0.5, 2.0, size=N_units)
factor_loadings[0] = 1.4  # Carga de la unidad tratada

# Efecto de Tratamiento Verdadero (tau = -4.5 tras T_treat)
tau_true = -4.5

records = []
for i in range(N_units):
    for t in range(T_periods):
        y_it = unit_fixed_effects[i] + time_trend[t] + factor_loadings[i] * macro_factor[t]
        y_it += rng.normal(0, 0.4)
        
        is_treated = (i == 0) and (t >= T_treat)
        if is_treated:
            y_it += tau_true
            
        records.append({
            "unit": f"unit_{i:02d}",
            "time": t,
            "y": y_it,
            "treat_time": T_treat if i == 0 else np.nan,
        })

df_panel = pd.DataFrame(records)

# %% [markdown]
# ## Estimación de DiD Sintético con puremacro
#
# Ajustamos SDID mediante `synthetic_did`:

# %%
res_sdid = synthetic_did(
    df=df_panel,
    unit="unit",
    time="time",
    outcome="y",
    treat_time="treat_time",
    n_boot=100,
    seed=42,
)

print(f"=== Estimación de DiD Sintético (Arkhangelsky et al. 2021) ===")
print(f"Efecto de Tratamiento (tau): {res_sdid.tau:.4f} (Verdadero: {tau_true:.4f})")
print(f"Error Estándar:              {res_sdid.se:.4f}")
print(f"Intervalo de Confianza 90%:  [{res_sdid.lo:.4f}, {res_sdid.hi:.4f}]")
print(f"Período de Tratamiento:      t = {res_sdid.treatment_time}")

assert abs(res_sdid.tau - tau_true) < 1.0, "SDID recupera el efecto de tratamiento verdadero"
assert np.isclose(np.sum(res_sdid.omega), 1.0), "Ponderaciones de unidades deben sumar 1"
assert np.isclose(np.sum(res_sdid.lambda_w), 1.0), "Ponderaciones temporales deben sumar 1"

# %% [markdown]
# ## Figura Principal — Trayectorias de SDID vs. Control Tradicional
#
# A continuación graficamos la trayectoria tratada, la trayectoria sintética y las ponderaciones $\hat{\omega}$.

# %%
# Pivotear datos
piv = df_panel.pivot(index="time", columns="unit", values="y")
treated_traj = piv["unit_00"].values
donor_matrix = piv.drop(columns=["unit_00"]).values

# Trayectoria de control sintético
synthetic_path = donor_matrix @ res_sdid.omega

# Promedio simple no ponderado
unweighted_control_path = donor_matrix.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [2.2, 1.2]})

# Panel 1: Comparación de trayectorias
ax = axes[0]
time_axis = np.arange(T_periods)
ax.plot(time_axis, treated_traj, color="0.00", lw=2.0, label="Unidad Tratada ($Y_{1,t}$)")
ax.plot(time_axis, synthetic_path, color="0.40", ls="--", lw=1.8, label="Control Sintético SDID ($\sum \hat{\omega}_i Y_{i,t}$)")
ax.plot(time_axis, unweighted_control_path, color="0.75", ls=":", lw=1.5, label="Controles no Ponderados (DiD Ingenuo)")
ax.axvline(T_treat - 0.5, color="0.60", ls="-.", lw=1.0, label="Inicio de Tratamiento")

ax.set_title("(a) Trayectorias: Tratada vs. Sintética vs. Ingenua", loc="left", fontsize=9.5, fontweight="bold")
ax.set_xlabel("Período")
ax.set_ylabel("Resultado $Y_{i,t}$")
ax.legend(loc="upper left", fontsize=8)

# Panel 2: Ponderaciones de donantes
ax_w = axes[1]
top_donors = res_sdid.omega.nlargest(6).iloc[::-1]
ax_w.barh(top_donors.index, top_donors.values, color="0.30", edgecolor="0.00")
ax_w.set_title(r"(b) Donantes Principales $\hat{\omega}_i$", loc="left", fontsize=9.5, fontweight="bold")
ax_w.set_xlabel(r"Ponderación $\hat{\omega}_i$")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretación Económica
#
# 1. **Alineación de Trayectorias**: Mientras que el promedio no ponderado diverge debido a diferentes cargas factoriales, el control sintético de SDID reproduce fielmente la trayectoria tratada en todo el período pre-tratamiento ($t < 12$).
# 2. **Divergencia Post-Tratamiento**: Tras la intervención en $t=12$, la unidad tratada experimenta una caída persistente ($\hat{\tau} \approx -4.5$).
# 3. **Ponderaciones Temporales ($\hat{\lambda}$)**: Re-ponderan los períodos previos para absorber shocks macroeconómicos comunes transitorios.

# %% [markdown]
# ## Tu Turno — Inspección de Ponderaciones Temporales $\hat{\lambda}$

# %%
fig, ax = plt.subplots(figsize=(6.0, 2.8))

pre_times = np.arange(T_treat)
ax.bar(pre_times, res_sdid.lambda_w, color="0.35", edgecolor="0.00", width=0.6, label="Ponderaciones Temporales $\hat{\lambda}_t$")
ax.axhline(1.0 / T_treat, color="0.60", ls="--", label=f"Ponderación Uniforme (1/{T_treat})")

ax.set_title("Ponderaciones Temporales Pre-Tratamiento $\hat{\lambda}_t$", loc="left", fontsize=10, fontweight="bold")
ax.set_xlabel("Período Pre-Tratamiento $t$")
ax.set_ylabel("Ponderación $\hat{\lambda}_t$")
ax.set_xticks(pre_times)
ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.show()

print("Suma de ponderaciones temporales:", np.sum(res_sdid.lambda_w))

# %% [markdown]
# ## Conclusión Metodológica
#
# - Utilice **`synthetic_did`** al evaluar políticas regionales, estatales o sectoriales donde no se cumplan las tendencias paralelas y los estimadores convencionales de efectos fijos bidireccionales presenten sesgo.
# - SDID combina las ventajas del **Control Sintético** y **DiD** en un único marco doblemente robusto e invariante a desplazamientos de nivel.
