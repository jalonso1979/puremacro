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
# # Novedades en puremacro 2.0: Guía Interactiva
#
# `puremacro 2.0` consolida la caja de herramientas macroeconométrica en un motor de cómputo
# de nivel de producción que corre en **100% Python puro** (cero compiladores C/Fortran) y cumple
# estrictamente con el **contrato Pyodide para navegador** (`numpy`, `scipy`, `pandas`, `matplotlib`).
#
# Este notebook interactivo recorre las principales innovaciones de la versión 2.0:
# 1. **Arquitectura Unificada `LPResult`** — Visualización `.plot()` en 1 línea y exportación instantánea a LaTeX / Typst.
# 2. **LP-IV Dependiente del Estado** (Ramey & Zubairy 2018) — Multiplicadores fiscales con instrumentos externos.
# 3. **VAR Aumentado por Factores (FAVAR)** (Bernanke, Boivin & Eliasz 2005) — Extracción de factores latentes e IRFs de panel.
# 4. **Diferencias en Diferencias Escalonadas Modernas** — Estudios de eventos robustos a heterogeneidad.
# 5. **Pipeline de Reportes Académicos** — Estrellas de significancia académica y tablas para manuscritos.
# 6. **Análisis de Sensibilidad DiD Honesto** (Rambachan & Roth 2023) — Violaciones a tendencias paralelas y valores de quiebre $M^*$.
# 7. **Perturbación DSGE de 2º Orden con Poda (Pruning)** (Kim et al. 2008) — Simulaciones no explosivas e IRFs asimétricas.
# 8. **Descarga de Cómputo a Google Colab** — Puente desde Juno e iPad a aceleradores en la nube.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import puremacro

print(f"Versión de puremacro cargada: {puremacro.__version__}")

# %% [markdown]
# ## 1. Arquitectura Unificada `LPResult`
#
# En 2.0, los estimadores de proyecciones locales devuelven un objeto `LPResult` (subclase de `pd.DataFrame`).
# Se obtienen todas las capacidades de pandas más métodos econométricos especializados:
# - `.plot()` — genera curvas de respuesta al impulso con bandas de confianza sombreadas en 1 línea.
# - `.summary()` — reporte ASCII limpio y estructurado de la regresión.
# - `.to_latex()`, `.to_typst()`, `.to_markdown()` — exportación inmediata a Overleaf, Typst y Quarto.

# %%
from puremacro.lp import lp_hac

# Serie macroeconómica sintética: respuesta del PIB real a una sorpresa monetaria
rng = np.random.default_rng(2026)
T = 180
shock = rng.standard_normal(T)
gdp_growth = np.cumsum(0.75 * shock + 0.35 * rng.standard_normal(T))
df_lp = pd.DataFrame({"gdp": gdp_growth, "shock": shock})

# Estimar LP hasta el horizonte 16 con 4 rezagos de control e intervalo de confianza al 90%
res_lp = lp_hac(df_lp, y="gdp", x="shock", horizon=16, lags=4, ci=0.90)

# Mostrar resumen tabular
print(res_lp.summary())

# Graficar IRF en una sola línea
fig, ax = plt.subplots(figsize=(7, 4))
res_lp.plot(ax=ax, title="Respuesta del PIB Real al Choque Monetario (LP-HAC)")
plt.show()

# Exportar tabla lista para LaTeX
print("Salida LaTeX (primeras 4 filas):")
print("\n".join(res_lp.to_latex().splitlines()[:6]))

# %% [markdown]
# ## 2. LP-IV Dependiente del Estado (Ramey & Zubairy 2018)
#
# ¿Cuál es la magnitud del multiplicador del gasto público durante períodos de holgura económica
# (alto desempleo) versus épocas normales? La estimación requiere:
# 1. Interacciones dependientes del estado: $F(s_t) x_t$ y $(1 - F(s_t)) x_t$.
# 2. Variables instrumentales externas: $F(s_t) z_t$ y $(1 - F(s_t)) z_t$ para resolver la endogeneidad de la política.
# 3. Regresiones 2SLS horizonte por horizonte con inferencia HAC Newey-West.

# %%
from puremacro.lp import lp_state_dep_iv

# Simulación de datos estilo Ramey-Zubairy
T = 200
unemployment = 6.5 + 1.5 * rng.standard_normal(T)       # Variable de estado (desempleo)
military_news = rng.standard_normal(T)                  # Instrumento exógeno de noticias
gov_spending = 0.8 * military_news + 0.3 * rng.standard_normal(T) # Gasto endógeno

# Transmisión con variación según el estado: multiplicador mayor en holgura alta
is_slack = (unemployment > 6.5).astype(float)
y = np.cumsum(0.9 * (is_slack * gov_spending) + 0.4 * ((1.0 - is_slack) * gov_spending) + 0.2 * rng.standard_normal(T))
df_rz = pd.DataFrame({"gdp": y, "spending": gov_spending, "news": military_news, "unemp": unemployment})

# Estimar multiplicadores con umbral de 6.5% de desempleo
res_rz = lp_state_dep_iv(
    df_rz,
    y="gdp",
    x="spending",
    z="news",
    state="unemp",
    threshold=6.5,
    transition="threshold",
    horizon=12,
    lags=2,
    ci=0.90,
)

print(res_rz[["h", "beta_H", "first_stage_f_H", "beta_L", "first_stage_f_L"]].head())

# %% [markdown]
# ## 3. VAR Aumentado por Factores (FAVAR, Bernanke et al. 2005)
#
# Los modelos VAR estándar están limitados a 3-6 variables para evitar la proliferación de parámetros.
# `favar` sintetiza paneles de alta dimensión (más de 50 indicadores macroeconómicos)
# en componentes principales latentes, estima un VAR conjunto sobre `[política, factores]`, y
# proyecta las respuestas al impulso estructurales sobre todas las series del panel con bandas bootstrap.

# %%
from puremacro.var import favar

# Panel informativo sintético de alta dimensión (20 series, T=150)
N = 20
F_true = rng.standard_normal((T, 2))
policy_rate = np.zeros(T)
for t in range(1, T):
    policy_rate[t] = 0.6 * policy_rate[t-1] + 0.4 * F_true[t-1, 0] + 0.2 * rng.standard_normal()

loadings = rng.uniform(-1.0, 1.0, size=(N, 3))
Z = np.column_stack([policy_rate, F_true])
panel_X = Z @ loadings.T + 0.4 * rng.standard_normal((T, N))
panel_df = pd.DataFrame(panel_X, columns=[f"Macro_Var_{i+1}" for i in range(N)])

# Estimar FAVAR con 2 factores latentes y bandas bootstrap al 90%
favar_res = favar(panel_df, policy_rate, n_factors=2, p=1, horizon=12, n_boot=50, seed=42)

print(favar_res.summary())

# Graficar respuestas seleccionadas
favar_res.plot(variables=["Macro_Var_1", "Macro_Var_2"])
plt.show()

# %% [markdown]
# ## 4. Diferencias en Diferencias Escalonadas Modernas
#
# Las regresiones clásicas de efectos fijos bidireccionales (TWFE) sesgan las estimaciones cuando
# la adopción del tratamiento es escalonada y los efectos son dinámicos o heterogéneos.
# `puremacro.did` implementa los estimadores robustos de última generación:
# - `callaway_santanna` — $ATT(g, t)$ de grupo-tiempo y estudios de eventos dinámicos.
# - `synthetic_did` — Ponderadores sintéticos de unidades ($\omega$) y de tiempo ($\lambda$).

# %%
from puremacro.did import callaway_santanna

# Panel de adopción escalonada: 12 unidades en 8 períodos
units = []
for u in range(12):
    treat_yr = 2012 if u < 4 else (2014 if u < 8 else np.nan) # NaN = nunca tratado
    for yr in range(2008, 2016):
        d = 1.0 if not np.isnan(treat_yr) and yr >= treat_yr else 0.0
        y = 2.0 * d + 0.5 * (yr - 2008) + rng.standard_normal()
        units.append({"unit": f"U{u}", "year": yr, "treat_time": treat_yr, "outcome": y})

panel_did = pd.DataFrame(units)

# Ajustar Callaway-Sant'Anna
res_did = callaway_santanna(panel_did, unit="unit", time="year", outcome="outcome", treat_time="treat_time", ci=0.90)
print(res_did.summary())
print(res_did.att_event_study)

# %% [markdown]
# ## 5. Pipeline de Reportes Académicos y Estrellas de Significancia
#
# Formatee coeficientes de regresión directamente en tablas académicas con estrellas de significancia:
# - `***` $p < 0.01$
# - `**` $p < 0.05$
# - `*` $p < 0.10$

# %%
from puremacro.reports import coef_table

betas = np.array([0.524, -1.892, 0.043])
ses = np.array([0.082, 0.410, 0.065])
varnames = ["Tasa de Política", "Gasto Fiscal", "Apertura Comercial"]

# Tabla en formato LaTeX
print("Formato LaTeX:")
print(coef_table(betas, ses, names=varnames, stars=True, fmt="latex"))

# Tabla en formato Typst
print("\nFormato Typst:")
print(coef_table(betas, ses, names=varnames, stars=True, fmt="typst"))

# %% [markdown]
# ## 6. Análisis de Sensibilidad DiD Honesto (Rambachan & Roth 2023)
#
# Evalúe la robustez de las estimaciones ante posibles violaciones del supuesto de tendencias paralelas.
# Calcula conjuntos identificados, intervalos de confianza robustos (Imbens & Manski 2004) y
# el valor de quiebre $M^*$ (el multiplicador que anula la significancia estadística).

# %%
from puremacro.did import honest_did_sensitivity

sens_res = honest_did_sensitivity(res_did, target_horizon=0, ci=0.90)
print(sens_res.summary())
print("\n" + sens_res.plot_ascii())

# %% [markdown]
# ## 7. Perturbación DSGE de 2º Orden con Poda (Kim et al. 2008)
#
# La perturbación estándar de segundo orden en modelos DSGE genera trayectorias explosivas
# debido a variedades cuadráticas espurias. El método de poda (pruning) de Kim, Kim,
# Schaumburg & Sims (2008) descompone el espacio de estados en componentes de primer y
# segundo orden, garantizando simulaciones estacionarias, respuestas al impulso generalizadas (GIRF)
# asimétricas y medias ergódicas analíticas con ajuste por riesgo.

# %%
from puremacro.dsge import canonical_growth_2nd_order

dsge_sol = canonical_growth_2nd_order()
print("Autovalores de primer orden (dentro del círculo unitario):", np.round(np.abs(dsge_sol.eigenvalues), 3))

# Simular 100 períodos con poda
sim = dsge_sol.simulate(periods=100, seed=42)
print("\nTrayectorias de simulación (primeras 5 filas):")
print(sim.to_frame().head())

# Respuesta al impulso generalizada (GIRF) que demuestra asimetría no lineal
girf_pos = dsge_sol.girf("eps", size=+2.0, horizon=8)
girf_neg = dsge_sol.girf("eps", size=-2.0, horizon=8)
print("\nGIRF Asimétrica del Consumo (choque positivo vs negativo):")
print(pd.DataFrame({
    "Choque +2σ": girf_pos["c"],
    "Choque -2σ": girf_neg["c"],
    "Suma (Asimetría)": girf_pos["c"] + girf_neg["c"],
}))

# Media ergódica con ajuste de riesgo (estado estacionario estocástico)
sss = dsge_sol.stochastic_steady_state(sigma=1.0)
print("\nEstado Estacionario Estocástico (Ajuste Precautorio):")
print("Desplazamiento ergódico del capital:", float(sss["states"]["k"]))

# %% [markdown]
# ## 8. Descarga de Cómputo a Google Colab desde Juno y Pyodide
#
# Trabaje de manera fluida en iPad (Juno / Juno Connect / Safari) y descargue simulaciones
# pesadas o bootstraps grandes directamente a GPUs/TPUs de Google Colab sin transferir archivos manualmente.

# %%
from puremacro.runtime import generate_colab_notebook, colab_auth_guide

# Imprimir guía de autenticación para iPad
print("Guía de autenticación para iPad / Safari móvil:")
print(colab_auth_guide()[:300] + "...\n")

# Generar notebook autónomo con carga de datos integrada
nb_json = generate_colab_notebook(
    code="""
import puremacro
from puremacro.dsge import canonical_growth_2nd_order
sol = canonical_growth_2nd_order()
sim = sol.simulate(periods=10000, seed=42)
print("Simulación pesada en Colab completada. Capital medio:", sim.states['k'].mean())
""",
    title="puremacro_tarea_colab",
    save_path=None,
    mount_drive=False,
)
print("Longitud de la cadena JSON del notebook de Colab:", len(str(nb_json)))

# %% [markdown]
# ## Conclusión
#
# `puremacro 2.0` combina la velocidad del cómputo vectorizado moderno con la accesibilidad
# de Python puro y despliegue sin instalación en el navegador.
#
# - **Documentación**: [https://jalonso1979.github.io/puremacro/](https://jalonso1979.github.io/puremacro/)
# - **Código Fuente**: [https://github.com/jalonso1979/puremacro](https://github.com/jalonso1979/puremacro)
