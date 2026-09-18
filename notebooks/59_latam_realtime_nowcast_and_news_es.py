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
# # Nowcasting en tiempo real y atribución de noticias de Bańbura-Modugno en América Latina: datos escalonados, descomposición de revisiones y calibración de densidad
#
# **¿Cómo pueden los bancos centrales y departamentos de investigación económica en América Latina estimar el crecimiento del PIB trimestral en tiempo real a partir de paneles mensuales con datos escalonados, descomponer las actualizaciones del nowcast en sorpresas de publicación y revisiones estadísticas mediante la identidad exacta de Bańbura y Modugno (2014), y validar la calibración de la densidad predictiva utilizando pruebas de uniformidad PIT de Berkowitz (2001)?**
#
# **El panel en este cuaderno es simulado para ejecución fuera de línea. No extraiga hechos históricos sobre América Latina a partir de él.** Cada número se genera en la primera celda de código a partir de la semilla determinista fija `np.random.default_rng(42)`: el factor latente común sigue un proceso normal acumulativo, mientras que las series de indicadores combinan cargas factoriales con choques idiosincrásicos. Lo que es completamente genuino y auténtico es el *esquema* institucional — los nombres de los proveedores (`inegi`, `banxico`, `bcb`), los identificadores canónicos de series (`735848`, `736184`, `628197`, `SF61745` para México; `22099`, `24363`, `433`, `432` para Brasil), las frecuencias y las unidades de medida que retornan los conectores de `puremacro.fetch.realtime`. Esto garantiza una ejecución 100% reproducible y desconectada en entornos Pyodide y WebAssembly sin requerir acceso a redes externas, credenciales o tokens de API de bancos centrales. El cartucho de datos portátil `.pmz` lleva `SIMULATED` en sus notas de procedencia, garantizando total transparencia.
#
# La vigilancia macroeconómica en economías emergentes —especialmente en América Latina— opera en un entorno caracterizado por desfases de publicación asíncronos y llegadas no sincronizadas de información estadística. Las autoridades de política monetaria en el Banco de México (Banxico) y el Banco Central do Brasil (BCB) no pueden esperar las publicaciones trimestrales de las cuentas nacionales, que suelen difundirse entre 60 y 90 días después del cierre del trimestre. En su lugar, las mesas de seguimiento monitorean indicadores mensuales de alta frecuencia: indicadores oportunos de actividad económica (IGAE en México, IBC-Br en Brasil), producción industrial, índices de precios al consumidor (INPC e IPCA) y tasas de interés de política monetaria (TIIE y Selic). Debido a que los institutos de estadística publican estos indicadores bajo calendarios heterogéneos, las matrices de datos en tiempo real presentan un patrón desbalanceado o "borde irregular" (*ragged edge*) al final de la muestra.
#
# Para extraer una señal coherente y oportuna del estado de la actividad económica, los bancos centrales implementan Modelos de Factores Dinámicos (DFM) en espacio de estados, estimados mediante el suavizador de Kalman (Giannone, Reichlin & Small 2008; Doz, Giannone & Reichlin 2011). Cuando se publican nuevos datos o las agencias estadísticas revisan retrospectivamente cifras previas, el modelo actualiza su estimación en tiempo real (*nowcast*) del PIB. Explicar los factores económicos determinantes de esa actualización es fundamental para la deliberación de política: ¿se revisó el nowcast porque la producción industrial superó las expectativas, o porque el instituto de estadística revisó a la baja la actividad del mes anterior? La descomposición analítica de noticias de Bańbura y Modugno (2014) resuelve este desafío al descomponer la revisión del nowcast en sorpresas de publicación (innovaciones respecto a las expectativas del modelo) y revisiones retrospectivas de datos, cumpliendo una identidad matemática exacta. Finalmente, la credibilidad de un banco central exige bandas de incertidumbre rigurosamente calibradas: las pruebas de uniformidad de la transformada de probabilidad integral (PIT de Berkowitz 2001) y los gráficos de abanico (*fan charts*) aseguran que los comités de política monetaria operen con densidades predictivas probabilísticas bien calibradas y no con predicciones puntuales engañosamente estrechas.

# %% [markdown]
# ## El método en matemáticas — Factores dinámicos, descomposición de noticias y calibración de densidad
#
# **1. Modelo de Factores Dinámicos en Forma de Espacio de Estados.** Sea $X_t = [x_{1, t}, \dots, x_{n, t}]^\top$ un panel $n$-dimensional de indicadores macroeconómicos mensuales estandarizados con media cero y varianza unitaria. Los indicadores comparten $r$ factores latentes no observados $F_t \in \mathbb{R}^r$ sujetos a errores de medición idiosincrásicos $\xi_t$:
# $$ X_t = \Lambda F_t + \xi_t, \quad \xi_t \sim \text{i.i.d.} \, \mathcal{N}(0, R), \quad R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2). $$
# Los factores latentes siguen un proceso autorregresivo vectorial estacionario de orden $p$:
# $$ F_t = A_1 F_{t-1} + \dots + A_p F_{t-p} + u_t, \quad u_t \sim \text{i.i.d.} \, \mathcal{N}(0, Q). $$
# Para paneles desbalanceados con bordes irregulares, los parámetros $(\Lambda, A, Q, R)$ se estiman mediante el algoritmo en dos etapas de componentes principales y filtro/suavizador de Kalman de Doz, Giannone y Reichlin (2011). La expectativa condicional $\hat{y}_{t^*|v} = \mathbb{E}[y_{t^*} \mid \Omega_v]$ de la variable objetivo $y_{t^*}$ (PIB trimestral) se extrae mediante el suavizador de Kalman sobre el conjunto de información disponible en la edición histórica $\Omega_v$.
#
# **2. Atribución Exacta de Noticias de Bańbura y Modugno (2014).** Sean $\Omega_{v-1}$ y $\Omega_v$ los conjuntos de información en dos ediciones sucesivas de datos, con $\Omega_{v-1} \subset \Omega_v$. La nueva información disponible se compone de nuevas publicaciones para periodos recientes $j \in \mathcal{I}_{\text{new}}$ y revisiones a observaciones históricas $k \in \mathcal{I}_{\text{rev}}$:
# $$ I_{j, v} \equiv x_{j, t_j} - \mathbb{E}[x_{j, t_j} \mid \Omega_{v-1}], \quad R_{k, v} \equiv x_{k, t_k}^{(v)} - x_{k, t_k}^{(v-1)}. $$
# La revisión del nowcast $\Delta \hat{y}_{t^*|v} \equiv \hat{y}_{t^*|v} - \hat{y}_{t^*|v-1}$ se descompone analíticamente en:
# $$ \Delta \hat{y}_{t^*|v} = \sum_{j \in \mathcal{I}_{\text{new}}} \omega_j \cdot I_{j, v} + \sum_{k \in \mathcal{I}_{\text{rev}}} \omega_k \cdot R_{k, v}, $$
# donde las ponderaciones $\omega$ están determinadas por la ganancia de Kalman y las autocovarianzas de los estados:
# $$ \omega = \operatorname{Cov}\left(y_{t^*}, \begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right) \left[\operatorname{Var}\left(\begin{bmatrix} I_v \\ R_v \end{bmatrix} \mid \Omega_{v-1}\right)\right]^{-1}. $$
# Esta descomposición satisface la identidad matemática exacta con residuo nulo:
# $$ \text{Error de Descomposición} \equiv \left| \Delta \hat{y}_{t^*|v} - \left(\sum \text{Impacto}_{\text{publicaciones}} + \sum \text{Impacto}_{\text{revisiones}}\right) \right| < 10^{-10}. $$
#
# **3. Evaluación de Densidad Predictiva: Prueba de Razón de Verosimilitud de Berkowitz (2001).** Sean $\{y_t\}_{t=1}^T$ las realizaciones observadas y $\{\mu_t, \sigma_t\}_{t=1}^T$ la secuencia de medias condicionales y desviaciones estándar del nowcast a un paso adelante. La Transformada de Probabilidad Integral (PIT) es:
# $$ p_t = \Phi\left(\frac{y_t - \mu_t}{\sigma_t}\right), \quad z_t = \Phi^{-1}(p_t), $$
# donde $\Phi(\cdot)$ es la función de distribución acumulada normal estándar. Bajo la hipótesis nula de correcta calibración e independencia serial, $p_t \sim \text{i.i.d.} \, \mathcal{U}(0, 1)$ y $z_t \sim \text{i.i.d.} \, \mathcal{N}(0, 1)$. Berkowitz modela el error transformado $z_t$ mediante un proceso autorregresivo:
# $$ (z_t - \mu) = \rho (z_{t-1} - \mu) + \varepsilon_t, \quad \varepsilon_t \sim \text{i.i.d.} \, \mathcal{N}(0, \sigma_\varepsilon^2). $$
# La prueba de razón de verosimilitud contrasta $H_0: \mu = 0, \sigma_\varepsilon^2 = 1, \rho = 0$ frente a la alternativa no restringida:
# $$ \text{LR} = -2 \left[ \ln L(0, 1, 0) - \ln L(\hat{\mu}, \hat{\sigma}_\varepsilon^2, \hat{\rho}) \right] \sim \chi^2(3). $$

# %% [markdown]
# ## Intuición
#
# **Intuición.** Los comités de política monetaria en los bancos centrales se reúnen en calendarios preestablecidos, independientemente de si las estadísticas oficiales del PIB trimestral ya han sido publicadas. Esperar las cifras trimestrales deja a las autoridades "a ciegas" durante puntos de inflexión críticos del ciclo económico. Los indicadores mensuales de alta frecuencia llegan con mayor rapidez, pero cada uno ofrece una visión parcial de la economía: la producción industrial abarca únicamente manufacturas, minería y servicios públicos; la inflación mide precios y no producto real; y las tasas de interés reflejan la postura monetaria y no el nivel de actividad. Además, cada indicador posee su propio retraso de divulgación, generando un borde irregular en el que algunas variables están actualizadas hasta el mes pasado mientras que otras presentan rezagos de dos o tres meses.
#
# El Modelo de Factores Dinámicos resuelve esta fricción informativa. Al postular que unas cuantas fuerzas macroeconómicas comunes impulsan los co-movimientos de todos los indicadores, el filtro de Kalman estima el estado económico subyacente a pesar de los datos faltantes. Cuando un instituto de estadística publica un nuevo dato, el motor de nowcasting calcula la *sorpresa* —la diferencia entre el número anunciado y lo que los factores latentes anticipaban—. Un incremento en la actividad económica no necesariamente eleva el nowcast: si el modelo y el mercado esperaban un repunte del 2.0% y el dato publicado fue de sólo 1.2%, la sorpresa es negativa (-0.8%) y el nowcast del PIB se revisa a la baja.
#
# El marco de Bańbura y Modugno proporciona un desglose contable auditable para cada revisión del nowcast. El equipo técnico puede presentar un gráfico de atribución claro a la junta de gobierno: el nowcast se revisó en +0.15 puntos porcentuales, explicado por +0.22 pp provenientes de sorpresas positivas en el índice de actividad económica, compensados por -0.04 pp en producción industrial y -0.03 pp por revisiones a la baja en datos del trimestre anterior. Simultáneamente, los bancos centrales comunican la incertidumbre mediante gráficos de abanico: en lugar de transmitir una falsa precisión con un único número puntual, bandas de probabilidad crecientes (como el verde de Banxico o el azul marino del BCB) ilustran la expansión de los intervalos de confianza en horizontes futuros. La prueba de uniformidad de Berkowitz garantiza que estas bandas no sean ni excesivamente estrechas (sobreconfiadas) ni excesivamente amplias (poco informativas), consolidando la credibilidad institucional ante los mercados financieros y la sociedad.

# %%
# Preamble: numerical libraries, plotting style, and realtime nowcast modules
import sys
from pathlib import Path
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.fetch.realtime import (
    VintagePanel,
    pack_realtime_cartridge,
    load_realtime_cartridge,
)
from puremacro.nowcast import (
    DynamicFactorModel,
    realtime_nowcast,
    banbura_modugno_news,
    fan_chart,
    pit_uniformity_test,
)

# Set deterministic random seed for Pyodide reproducibility
rng = np.random.default_rng(42)

print("puremacro Latin America Real-Time Nowcasting & News Attribution Engine")

# %%
# --- Experiment 1: Assemble Multi-Country Latin America Vintage Panel ---
# ALL OBSERVATIONS BELOW ARE DETERMINISTICALLY GENERATED IN THIS CELL.
# The provider names, canonical series IDs (735848, 736184, SF61745, 22099, 24363, 432),
# and units are the authentic ones defined by INEGI, Banxico, and BCB.
# The numbers themselves are synthetic paths generated from seed 42 to guarantee
# 100% offline execution in Pyodide without live network sockets or API keys.
#
# Reference sample: 36 monthly periods (2022-01-01 to 2024-12-01).
# Vintages: 2 snapshot captures (2024-11-01 and 2024-12-01).
dates = pd.date_range("2022-01-01", periods=36, freq="MS")
v1 = pd.Timestamp("2024-11-01")
v2 = pd.Timestamp("2024-12-01")

mex_series = [
    ("gdp", "inegi", "735848", "index"),
    ("activity", "inegi", "736184", "index"),
    ("ip", "inegi", "736184_IP", "index"),
    ("cpi", "inegi", "628197", "index"),
    ("policy_rate", "banxico", "SF61745", "rate"),
]

bra_series = [
    ("gdp", "bcb", "22099", "index"),
    ("activity", "bcb", "24363", "index"),
    ("ip", "bcb", "21859", "index"),
    ("cpi", "bcb", "433", "index"),
    ("policy_rate", "bcb", "432", "rate"),
]

rows = []
for country, series_list in [("MEX", mex_series), ("BRA", bra_series)]:
    # Common persistent business cycle factor for the economy
    f_latent = np.cumsum(rng.normal(scale=0.25, size=len(dates)))
    for var, prov, sid, un in series_list:
        load = rng.uniform(0.7, 1.3)
        noise = rng.normal(scale=0.15, size=len(dates))
        y_sim = 100.0 + 1.5 * f_latent * load + noise if un == "index" else 8.0 - 0.1 * f_latent * load + noise

        for t_idx, d in enumerate(dates):
            # Vintage 1: asynchronous ragged edge at the last 2 periods
            val_v1 = float(y_sim[t_idx])
            if t_idx == len(dates) - 1 and var != "gdp":
                val_v1 = np.nan
            elif t_idx == len(dates) - 2 and var in [series_list[2][0], series_list[3][0]]:
                val_v1 = np.nan

            if not np.isnan(val_v1):
                rows.append({
                    "country": country, "variable": var, "date": d, "vintage": v1,
                    "value": val_v1, "provider": prov, "series_id": sid, "units": un,
                })

            # Vintage 2: releases ragged values and revises historical activity
            val_v2 = float(y_sim[t_idx])
            if t_idx == len(dates) - 1 and var == series_list[3][0]:
                val_v2 = np.nan
            if t_idx == 20 and var == series_list[1][0]:
                val_v2 += 0.45  # Statistical agency retrospective revision

            if not np.isnan(val_v2):
                rows.append({
                    "country": country, "variable": var, "date": d, "vintage": v2,
                    "value": val_v2, "provider": prov, "series_id": sid, "units": un,
                })

df_panel = pd.DataFrame(rows)
panel_raw = VintagePanel(df_panel)

# Package into portable self-verifying .pmz cartridge and reload
with tempfile.TemporaryDirectory() as td:
    cart_path = Path(td) / "latam_realtime_nowcast.pmz"
    pack_realtime_cartridge(
        panel_raw,
        cart_path,
        source="Banxico, INEGI, BCB",
        notes="SIMULATED panel on authentic Latin America central bank identifiers",
    )
    loaded_panel = load_realtime_cartridge(cart_path)

print("Constructed Multi-Country Vintage Panel:")
print("  Data Provenance    : SIMULATED (stylized paths on real provider/series identifiers)")
print(f"  Total Observations : {len(panel_raw):,}")
print(f"  Countries Included : {panel_raw.countries}")
print(f"  Macro Variables    : {panel_raw.variables}")
print(f"  Reference Periods  : {len(dates)} months ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
print(f"  Vintage Snapshots  : {len([v1, v2])} captures ({v1.strftime('%Y-%m-%d')} and {v2.strftime('%Y-%m-%d')})")

# Verify panel integrity and roundtrip fidelity
assert panel_raw.countries == ["BRA", "MEX"]
assert "gdp" in panel_raw.variables
assert "activity" in panel_raw.variables
assert "policy_rate" in panel_raw.variables
assert len(loaded_panel) == len(panel_raw)

# %%
# --- Experiment 2: Dynamic Factor Model Nowcasting & Exact News Attribution ---
# Execute high-level real-time orchestrator for Mexico (Banxico/INEGI) and Brazil (BCB).
# Automatically extracts ragged edges, fits DFM with Kalman smoothing, and computes
# the exact Bańbura & Modugno (2014) news decomposition between Vintage 1 and Vintage 2.
res_mex = realtime_nowcast(country="MEX", panel=loaded_panel, method="dfm", n_factors=1)
res_bra = realtime_nowcast(country="BRA", panel=loaded_panel, method="dfm", n_factors=1)

print(res_mex.summary())
print("\n" + "=" * 74)
print(f"Brazil (BCB) Point Nowcast: {res_bra.nowcast:+.4f} (SE: {res_bra.forecast_sd:.4f})")
print("=" * 74)

# Headline assertions verifying Mexico nowcast and Bańbura-Modugno news identity
nd_mex = res_mex.news_decomposition
assert res_mex.country == "MEX"
assert res_mex.palette == "banxico"
assert not np.isnan(res_mex.nowcast)
assert res_mex.forecast_sd > 0.0
assert nd_mex is not None
assert nd_mex.decomposition_error < 1e-10, f"Decomposition error {nd_mex.decomposition_error} exceeds 1e-10"
assert abs(nd_mex.revision - nd_mex.total_impact) < 1e-10, "Total impact must match nowcast revision"
assert len(nd_mex.news_table) > 0, "News table must contain indicator surprise releases"

# Headline assertions verifying Brazil results and institutional palette
nd_bra = res_bra.news_decomposition
assert res_bra.country == "BRA"
assert res_bra.palette == "bcb"
assert not np.isnan(res_bra.nowcast)
assert nd_bra is not None
assert nd_bra.decomposition_error < 1e-10, f"Brazil decomp error {nd_bra.decomposition_error} exceeds 1e-10"

# %%
# --- Experiment 3: Out-of-Sample Density Forecast Evaluation & Fan Charts ---
# Evaluate density calibration using Berkowitz (2001) Probability Integral Transform
# (PIT) likelihood ratio test and Kolmogorov-Smirnov test over a rolling out-of-sample sample.
T_eval = 60
mu_eval = rng.normal(loc=2.0, scale=0.5, size=T_eval)
sd_eval = rng.uniform(0.4, 0.8, size=T_eval)
y_eval = mu_eval + sd_eval * rng.normal(size=T_eval)

pit_res = pit_uniformity_test(realised=y_eval, mu=mu_eval, sigma=sd_eval)

print(pit_res.summary())

# Assertions verifying well-calibrated predictive distribution
assert isinstance(pit_res.lr_pvalue, float)
assert pit_res.is_uniform is True, "Calibrated predictive density must satisfy uniformity"
assert pit_res.lr_pvalue > 0.05, f"Berkowitz LR p-value {pit_res.lr_pvalue:.4f} rejected at 5%"
assert pit_res.ks_pvalue > 0.05, f"Kolmogorov-Smirnov p-value {pit_res.ks_pvalue:.4f} rejected at 5%"

# Construct central bank fan charts: Banxico national green and BCB navy
hist_mex = pd.Series([100.1, 100.4, 100.3, 100.5], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
fc_mean_mex = pd.Series([res_mex.nowcast, res_mex.nowcast + 0.15, res_mex.nowcast + 0.30], index=["2025Q1", "2025Q2", "2025Q3"])
fc_sd_mex = pd.Series([res_mex.forecast_sd, res_mex.forecast_sd * 1.25, res_mex.forecast_sd * 1.55], index=fc_mean_mex.index)
fc_mex = fan_chart(hist_mex, fc_mean_mex, fc_sd_mex, levels=(0.3, 0.6, 0.9), palette="banxico")

hist_bra = pd.Series([98.2, 98.5, 98.7, 98.8], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
fc_mean_bra = pd.Series([res_bra.nowcast, res_bra.nowcast + 0.10, res_bra.nowcast + 0.25], index=["2025Q1", "2025Q2", "2025Q3"])
fc_sd_bra = pd.Series([res_bra.forecast_sd, res_bra.forecast_sd * 1.20, res_bra.forecast_sd * 1.45], index=fc_mean_bra.index)
fc_bra = fan_chart(hist_bra, fc_mean_bra, fc_sd_bra, levels=(0.3, 0.6, 0.9), palette="bcb")

assert fc_mex.palette == "banxico"
assert fc_bra.palette == "bcb"
assert len(fc_mex.intervals) == 3
assert len(fc_bra.intervals) == 3

# %%
# --- Hero Visualization Dashboard: Latin America Real-Time Nowcasting ---
# 4-panel comprehensive figure displaying latent factors, news attribution waterfall,
# central bank fan chart projections, and Berkowitz PIT calibration diagnostics.
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: Latent Dynamic Factor Trajectories
axes[0, 0].plot(res_mex.factors.index, res_mex.factors.iloc[:, 0], color=_nbstyle.S1["color"], lw=2.0, label="Mexico DFM Factor 1 (Banxico/INEGI)")
axes[0, 0].plot(res_bra.factors.index, res_bra.factors.iloc[:, 0], color=_nbstyle.S2["color"], lw=2.0, ls="--", label="Brazil DFM Factor 1 (BCB)")
axes[0, 0].set_title("Latin America Dynamic Common Factors (2022–2024)", fontsize=11, fontweight="semibold")
axes[0, 0].set_xlabel("Reference Period")
axes[0, 0].set_ylabel("Latent Factor Index (Std. Units)")
axes[0, 0].grid(True, ls=":", color=_nbstyle.REJILLA)
axes[0, 0].legend(loc="best", fontsize=8)

# Panel 2: Bańbura & Modugno News Attribution Waterfall
res_mex.news_decomposition.plot(ax=axes[0, 1], title="Mexico GDP Nowcast Revision: Bańbura-Modugno News Waterfall")

# Panel 3: Banco de México GDP Growth Fan Chart
fc_mex.plot(ax=axes[1, 0], title="Banco de México: Headline GDP Growth Fan Chart Projection")

# Panel 4: Out-of-Sample Predictive Density Calibration (Berkowitz PIT)
pit_res.plot(ax=axes[1, 1], title="Forecast Density Calibration: Berkowitz (2001) PIT Distribution")

fig.suptitle("Latin America Real-Time Nowcasting & News Attribution Dashboard", fontsize=13, fontweight="bold")

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Cada cifra empírica y cuadro de diagnóstico obtenido arriba ilustra los mecanismos centrales de la vigilancia macroeconómica en tiempo real bajo paneles con datos escalonados:
#
# 1. **Nowcasts puntuales en tiempo real (Experimento 2 y Figura Panel 1):** El Modelo de Factores Dinámicos estima con éxito los nowcasts puntuales para México y Brasil a partir del panel con bordes irregulares. Para México, el nowcast del índice del PIB se ubica en $+100.6377$ con un error estándar condicional de $0.1123$, produciendo un intervalo de confianza al 90% de $[+100.4530, +100.8224]$. Para Brasil, el nowcast puntual se sitúa en $+98.9262$ con un error estándar de $0.1791$. El Panel 1 del tablero ilustra los factores latentes comunes del ciclo económico: el factor mexicano (verde Banxico) y el factor brasileño (azul marino BCB) evolucionan suavemente a lo largo de los 36 meses muestrales, filtrando el ruido idiosincrásico de los indicadores mensuales más volátiles.
# 2. **Cascada exacta de noticias de Bańbura-Modugno (Experimento 2 y Figura Panel 2):** Entre la captura del 1 de noviembre de 2024 ($v-1$) y la del 1 de diciembre de 2024 ($v$), el nowcast del PIB mexicano se actualiza de $+100.6322$ a $+100.6377$, acumulando una revisión total de $\Delta \hat{y} = +0.0055$ puntos del índice. El Panel 2 descompone esta revisión en componentes económicos específicos: las nuevas publicaciones de producción industrial y actividad económica aportaron sorpresas positivas respecto a las expectativas previas del modelo, mientras que la revisión retrospectiva en el mes 20 aportó un ajuste complementario. De manera crucial, el error analítico impreso es de $5.97 \times 10^{-15}$, demostrando que la identidad de Bańbura y Modugno cuadra a precisión de máquina ($< 10^{-10}$).
# 3. **Bandas de incertidumbre en gráficos de abanico institucionales (Experimento 3 y Figura Panel 3):** El Panel 3 despliega el gráfico de abanico del Banco de México anclado en el nowcast puntual. Tres bandas de confianza anidadas (30%, 60% y 90%) se expanden a lo largo del horizonte de proyección de 2025 ($h = 1, 2, 3$ trimestres) a medida que el error estándar pasa de $0.1123$ a $0.1740$. La paleta gráfica adopta los estándares de la institución: el verde Banxico (`#006847`) presenta la proyección central y las franjas de incertidumbre, brindando a las autoridades un rango probabilístico en lugar de una estimación puntual engañosa.
# 4. **Diagnóstico de uniformidad PIT de Berkowitz (Experimento 3 y Figura Panel 4):** La muestra de evaluación con $T = 60$ nowcasts fuera de muestra supera tanto la prueba de razón de verosimilitud de Berkowitz (2001) ($LR = 3.93, p = 0.2695 > 0.05$) como la prueba de Kolmogórov-Smirnov ($KS = 0.096, p = 0.6241 > 0.05$). En el Panel 4, el histograma empírico de las transformadas integrales de probabilidad $p_t = \Phi((y_t - \mu_t)/\sigma_t)$ se alinea estrechamente con la línea de referencia uniforme de $1.0$. Los parámetros estimados en el espacio transformado ($\hat{\mu} = 0.082$, $\hat{\sigma} = 0.941$, $\hat{\rho} = 0.165$) confirman que el motor de nowcasting produce densidades predictivas sin sesgo ($\mu \approx 0$), sin subdispersión ($\sigma \approx 1$) y libres de autocorrelación persistente ($\rho \approx 0$).

# %%
# Your turn: customize country selection, factor dimensions, and fan chart levels
# Modify the parameters below to explore different Latin American economies
# and evaluate how dynamic factor ranks modulate news decomposition attribution.

# ← change this: target economy code ('MEX' or 'BRA')
country_custom = "MEX"

# ← change this: number of common dynamic factors (1 or 2)
n_factors_custom = 1

# ← change this: target variable to nowcast ('gdp' or 'activity')
target_var_custom = "gdp"

# ← change this: central bank fan chart confidence levels
fan_levels_custom = (0.50, 0.70, 0.90)

# Re-run real-time nowcast orchestrator with custom parameters
res_custom = realtime_nowcast(
    country=country_custom,
    panel=loaded_panel,
    target_variable=target_var_custom,
    n_factors=n_factors_custom,
)

# Generate custom fan chart
fig_custom = res_custom.plot_fan_chart(levels=fan_levels_custom)

print(f"Custom Nowcasting Results ({country_custom} - {target_var_custom}):")
print(f"  Point Nowcast       : {res_custom.nowcast:+.4f}")
print(f"  Forecast Std. Error : {res_custom.forecast_sd:.4f}")
print(f"  Visual Theme        : {res_custom.palette.upper()}")
print(f"  Decomposition Error : {res_custom.news_decomposition.decomposition_error:.2e}")

# Downstream assertions validating user parameters and execution
assert country_custom in ("MEX", "BRA"), "Supported countries are MEX or BRA"
assert n_factors_custom in (1, 2), "Factors must be 1 or 2"
assert target_var_custom in ("gdp", "activity"), "Target variable must be gdp or activity"
assert not np.isnan(res_custom.nowcast), "Nowcast must be a valid float"
assert res_custom.forecast_sd > 0.0, "Forecast standard deviation must be positive"
assert res_custom.news_decomposition is not None, "News decomposition must be present"
assert res_custom.news_decomposition.decomposition_error < 1e-10, "Decomposition error must be < 1e-10"

# %% [markdown]
# **Prompts.**
# 1. *Básico:* Cambie `country_custom` de `"MEX"` a `"BRA"`. Observe cómo la fuente institucional se actualiza a Banco Central do Brasil, la serie objetivo se asigna al PIB de IBGE y la paleta visual cambia del verde de Banxico al azul marino del BCB (`#0b3b60`).
# 2. *Intermedio:* Modifique `fan_levels_custom` a `(0.40, 0.80)`. Observe cómo las bandas de proyección central se ajustan para mostrar dos franjas más amplias en lugar de tres, modificando la comunicación visual del balance de riesgos.
# 3. *Avanzado:* Aumente `n_factors_custom` a `2`. Compare la varianza explicada y el error estándar condicional frente al modelo base de un solo factor, y observe cómo los factores comunes secundarios modulan las ponderaciones del filtro de Kalman aplicadas a las sorpresas de actividad económica.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro` ofrece un ecosistema completo para nowcasting regional y vigilancia macroeconómica en tiempo real:
# - `puremacro.fetch.realtime`: Conectores de datos en tiempo real de primera clase para Banxico, INEGI, BCB y BCCh (`VintagePanel`, `pack_realtime_cartridge`, `load_realtime_cartridge`).
# - `puremacro.nowcast.dfm`: Modelos de Factores Dinámicos con filtro y suavizador de Kalman (`DynamicFactorModel`, `DynamicFactorModelResult`).
# - `puremacro.nowcast.news`: Atribución analítica de noticias frente a revisiones basada en Bańbura y Modugno (2014) (`banbura_modugno_news`, `NewsDecompositionResult`).
# - `puremacro.nowcast.evaluation`: Evaluación probabilística de pronósticos y calibración de densidades (`fan_chart`, `pit_uniformity_test`, `crps_gaussian`, `log_score_gaussian`).
# - `puremacro.nowcast.realtime_nowcast`: Orquestador unificado de alto nivel que integra paneles de datos históricos con estimación en espacio de estados (`realtime_nowcast`, `RealtimeNowcastResult`).
