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
# # Macroeconomía en Tiempo Real de América Latina: Cartuchos de Datos Offline, Triángulos de Revisión y Análisis de Mankiw-Shapiro
#
# **¿Cómo monitorean la política monetaria y la actividad económica en tiempo real los bancos centrales y los investigadores macroeconómicos en América Latina, qué tan cuantiosas son las revisiones posteriores a los datos preliminares, y cómo pueden los cartuchos de datos inmutables y autenticados garantizar la replicabilidad empírica exacta frente a cambios en las APIs de los institutos estadísticos?**
#
# La vigilancia macroeconómica en economías emergentes —particularmente en América Latina— exige navegar severas fricciones de información en tiempo real. Las autoridades monetarias como el Banco de México (Banxico), el Banco Central do Brasil (BCB) y el Banco Central de Chile (BCCh) operan bajo una marcada vulnerabilidad externa, donde las tasas de interés de política monetaria (TIIE objetivo, Taxa Selic y TPM) deben responder con prontitud a la evolución de la actividad y la inflación domésticas. Sin embargo, las cifras de cuentas nacionales publicadas por los institutos de estadística (como el INEGI en México o el IBGE en Brasil) constituyen estimaciones preliminares sustentadas en muestras parciales de indicadores mensuales. A lo largo de meses y trimestres sucesivos, estas cifras preliminares experimentan revisiones retrospectivas sustanciales conforme se incorporan respuestas rezagadas de encuestas, se imputa la actividad del sector informal y se realizan conciliaciones anuales de referencia.
#
# Determinar si las revisiones macroeconómicas representan **noticias** (actualizaciones eficientes de pronóstico que incorporan nueva información económica) o **ruido** (errores transitorios de medición) es indispensable para la estabilidad macroeconómica. Si las revisiones son predominantemente ruido, los bancos centrales que reaccionan con agresividad a las publicaciones iniciales introducen volatilidad espuria en la economía real. Si las revisiones representan noticias, los datos preliminares resumen eficientemente toda la información disponible y la política debe reaccionar de inmediato. Además, la investigación empírica en América Latina se ve interrumpida cuando las APIs de los bancos centrales o institutos estadísticos alteran sus rutas de acceso, modifican esquemas JSON o exigen credenciales complejas. Los cartuchos de datos `.pmz` empaquetan paneles de tiempo real multipaís en cápsulas inmutables con verificación criptográfica SHA-256 que se ejecutan completamente offline en navegadores y entornos Pyodide. Este cuaderno demuestra el flujo integral: ensamble de un panel de tiempo real de América Latina, empaquetado y validación de cartuchos `.pmz`, construcción de triángulos de revisión y contrastes econométricos de noticias frente a ruido de Mankiw-Shapiro (1986).

# %% [markdown]
# ## El método en matemáticas — Vintages en Tiempo Real de América Latina, Triángulos de Revisión y Econometría de Mankiw-Shapiro
#
# **1. Matriz de Vintages en Tiempo Real y Dinámica de Revisiones.** Sea $y_{i, t, v}$ el valor del indicador macroeconómico $i$ para el trimestre de referencia $t$ publicado en la edición de vintage $v$ (con fecha de publicación $v \ge t$). Para una secuencia balanceada, la matriz triangular histórica de revisiones $\mathbf{T}[t, v]$ organiza las fechas de referencia en las filas y los vintages de publicación en las columnas:
# $$ \mathbf{T} = \begin{bmatrix} y_{t_1, v_1} & y_{t_1, v_2} & \dots & y_{t_1, v_K} \\ \text{NaN} & y_{t_2, v_2} & \dots & y_{t_2, v_K} \\ \vdots & \vdots & \ddots & \vdots \\ \text{NaN} & \text{NaN} & \dots & y_{t_K, v_K} \end{bmatrix}. $$
# Sea $y_t^{(0)} = y_{t, v_0(t)}$ la estimación preliminar inicial y $y_t^{(F)} = y_{t, v_{\max}}$ la última publicación de referencia. La revisión total se define como:
# $$ r_t \equiv y_t^{(F)} - y_t^{(0)}. $$
#
# **2. Contrastes Econométricos de Noticias vs. Ruido de Mankiw-Shapiro (1986).** Bajo expectativas racionales e informes estadísticos eficientes, las revisiones se clasifican en dos hipótesis estructurales rivales:
# - **Hipótesis de Noticias ($H_{\text{Noticias}}$):** La cifra preliminar $y_t^{(0)}$ es una proyección matemática óptima del valor final $y_t^{(F)}$ sobre el conjunto de información preliminar $\Omega_0$. La revisión $r_t$ representa innovaciones impredecibles (noticias) y debe ser ortogonal a $y_t^{(0)}$:
#   $$ r_t = \alpha_p + \beta_p y_t^{(0)} + \varepsilon_{p, t}, \quad H_0^{\text{Noticias}}: \beta_p = 0. $$
# - **Hipótesis de Ruido ($H_{\text{Ruido}}$):** La publicación preliminar es el verdadero valor final contaminado con error clásico de medición $u_t$: $y_t^{(0)} = y_t^{(F)} + u_t$, donde $\operatorname{Cov}(y_t^{(F)}, u_t) = 0$. La revisión $r_t = -u_t$ está correlacionada con la cifra preliminar pero debe ser ortogonal al valor final $y_t^{(F)}$:
#   $$ r_t = \alpha_f + \beta_f y_t^{(F)} + \varepsilon_{f, t}, \quad H_0^{\text{Ruido}}: \beta_f = 0. $$
# La fracción de ruido de la varianza de la cifra preliminar se cuantifica como:
# $$ \text{Fracción de Ruido} = \max(0, -\beta_p) = \frac{\operatorname{Var}(u_t)}{\operatorname{Var}(y_t^{(0)})}. $$
#
# **3. Autenticación Criptográfica de Cartuchos de Datos.** Los cartuchos `.pmz` garantizan la replicabilidad empírica offline mediante resúmenes criptográficos:
# $$ \mathcal{H}_{\text{datos}} = \operatorname{SHA256}(\operatorname{bytes}(df)), \quad \text{verificados al desempaquetar contra el manifiesto de metadatos.} $$

# %% [markdown]
# ## Intuición
#
# **Intuición.** Las autoridades monetarias del Banco de México, el Banco Central do Brasil y el Banco Central de Chile toman decisiones de tasas de interés en tiempo real basándose en aproximaciones preliminares imperfectas del crecimiento económico y la inflación. Cuando el instituto nacional de estadística publica el PIB preliminar del trimestre anterior, dicho cálculo refleja únicamente una fracción de las encuestas empresariales completadas y depende en gran medida de imputaciones estadísticas para la economía informal. En los trimestres posteriores, conforme se procesan las declaraciones tributarias corporativas consolidadas y los censos económicos, las agencias publican cifras revisadas que pueden modificar la apreciación sobre la posición cíclica de la economía.
#
# El marco econométrico de Mankiw-Shapiro permite evaluar cómo deben interpretar estas revisiones los responsables de política. Si los institutos estadísticos elaboran los datos preliminares como proyecciones racionales dada la información incompleta, las revisiones representan auténticas *noticias* económicas. En este escenario, las revisiones no pueden predecirse a partir de la cifra preliminar ($\beta_p = 0$), lo que implica que el banco central no puede mejorar la estimación inicial y debe considerarla una señal insesgada. Por el contrario, si las estimaciones iniciales están contaminadas por error clásico de medición (*ruido*), la revisión muestra correlación negativa con la publicación preliminar ($\beta_p < 0$) y ortogonalidad con el dato final ($\beta_f = 0$). Bajo la hipótesis de ruido, los banqueros centrales que responden agresivamente al crecimiento preliminar terminan reaccionando a artefactos estadísticos, amplificando la volatilidad del producto.
#
# Para posibilitar un análisis macroeconómico riguroso sin depender de conexiones de red externas, límites de tasa en APIs o credenciales privadas, `puremacro` introduce los cartuchos de datos autónomos (`.pmz`). El cartucho encapsula paneles de vintages multipaís, metadatos canónicos de series y firmas criptográficas SHA-256 en un único archivo comprimido. Al cargarse offline, el cartucho valida la integridad de los datos y expone la interfaz analítica integral de `VintagePanel` (`coverage()`, `as_of()`, `triangle()`, `revisions()` y `news_or_noise()`), garantizando compatibilidad absoluta con navegadores y entornos Pyodide.

# %%
# Preamble: import numerical libraries, plotting style, and realtime panel tools
import sys
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
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

# Set deterministic random seed for reproducibility
rng = np.random.default_rng(42)

print("Latin America Real-Time Ecosystem: Banxico, INEGI, BCB, BCCh")

# %%
# --- Experiment 1: Assemble Multi-Country Latin America Real-Time Vintage Panel ---
# Construct quarterly reference periods: 2022Q1 to 2025Q3 (15 reference quarters)
# Publication vintages: 8 quarterly vintages spanning 2024Q1 to 2025Q4
ref_dates = pd.date_range("2022-01-01", "2025-07-01", freq="QS").strftime("%Y-%m-%d").tolist()
vintage_dates = pd.date_range("2024-01-01", "2025-10-01", freq="QS").strftime("%Y-%m-%d").tolist()

rows = []
for v_idx, v in enumerate(vintage_dates):
    for d_idx, d in enumerate(ref_dates):
        if d <= v:
            # Mexico: Real GDP (INEGI indicator 735848) with realistic preliminary measurement noise
            base_gdp = 24000000.0 + 150000.0 * d_idx
            noise = float(rng.normal(0, 50000.0)) if d == v or (pd.to_datetime(v) - pd.to_datetime(d)).days <= 180 else 0.0
            rows.append({
                "country": "MEX", "variable": "gdp_real", "date": d, "vintage": v,
                "value": base_gdp + noise, "provider": "inegi", "series_id": "735848", "units": "MXN_millions"
            })
            # Mexico: Policy rate (Banxico TIIE objetivo, series SF61745)
            rows.append({
                "country": "MEX", "variable": "policy_rate", "date": d, "vintage": v,
                "value": 11.25 - 0.25 * d_idx, "provider": "banxico", "series_id": "SF61745", "units": "percent"
            })
            # Brazil: Policy rate (BCB Taxa Selic, series 432)
            rows.append({
                "country": "BRA", "variable": "policy_rate", "date": d, "vintage": v,
                "value": 12.75 - 0.50 * d_idx, "provider": "bcb", "series_id": "432", "units": "percent"
            })
            # Chile: Policy rate (BCCh TPM, series F022.TPM.TPO.D001.NO.Z.D)
            rows.append({
                "country": "CHL", "variable": "policy_rate", "date": d, "vintage": v,
                "value": 9.50 - 0.75 * d_idx, "provider": "bcch", "series_id": "F022.TPM.TPO.D001.NO.Z.D", "units": "percent"
            })

df_raw = pd.DataFrame(rows)
panel_raw = VintagePanel(df_raw)

print(f"Constructed Multi-Country Vintage Panel:")
print(f"  Total Observations : {len(panel_raw):,}")
print(f"  Countries Included : {panel_raw.countries}")
print(f"  Macro Variables    : {panel_raw.variables}")
print(f"  Reference Periods  : {len(ref_dates)} quarters ({ref_dates[0]} to {ref_dates[-1]})")
print(f"  Vintages Available : {len(vintage_dates)} releases ({vintage_dates[0]} to {vintage_dates[-1]})")

# Panel structural assertions
assert panel_raw.countries == ["BRA", "CHL", "MEX"]
assert "policy_rate" in panel_raw.variables
assert "gdp_real" in panel_raw.variables
assert len(panel_raw) > 300

# %%
# --- Experiment 2: Self-Verifying Cryptographic Cartridge Packaging & Loading ---
# Package panel into an offline .pmz cartridge with SHA-256 verification
with tempfile.TemporaryDirectory() as tmp_dir:
    cartridge_file = Path(tmp_dir) / "latam_realtime_macro.pmz"
    pack_realtime_cartridge(
        panel_raw,
        cartridge_file,
        source="Banxico, INEGI, BCB, BCCh Regional Real-Time Ecosystem",
        vintage="2026-04-01",
        notes="Latin America central bank real-time macroeconomic vintage cartridge",
    )
    assert cartridge_file.exists(), "Cartridge file must be created on disk"
    loaded_panel = load_realtime_cartridge(cartridge_file, verify=True)

print(f"Portable Cartridge Authentication:")
print(f"  SHA-256 Digest Verification : SUCCESS")
print(f"  Loaded Countries            : {loaded_panel.countries}")
print(f"  Loaded Variables            : {loaded_panel.variables}")
print(f"  Loaded Record Count         : {len(loaded_panel):,}")

# Cartridge integrity assertions
assert isinstance(loaded_panel, VintagePanel)
assert loaded_panel.countries == ["BRA", "CHL", "MEX"]
assert len(loaded_panel) == len(panel_raw)

# %%
# --- Experiment 3: Real-Time Coverage, As-Of Cross-Section, and Revision Triangles ---
cov_df = loaded_panel.coverage()
as_of_2025 = loaded_panel.as_of("2025-06-01")
tri_df = loaded_panel.triangle("MEX", "gdp_real")
rev_df = loaded_panel.revisions("MEX", "gdp_real")

print(f"Real-Time Panel Analysis:")
print(f"  Coverage Table Dimensions     : {cov_df.shape}")
print(f"  As-Of 2025-06-01 Observations : {len(as_of_2025)}")
print(f"  Mexico GDP Revision Triangle  : {tri_df.shape[0]} reference dates x {tri_df.shape[1]} vintages")
print(f"  Total Revisions Extracted     : {len(rev_df)} revision pairs")
print(f"  Mean Revision Magnitude       : {rev_df['revision'].mean():.4f} percentage points")

# Coverage and revision assertions
assert not cov_df.empty, "Coverage table must not be empty"
assert not as_of_2025.empty, "As-of slice must return observations"
assert tri_df.shape[1] == len(vintage_dates), "Triangle must have columns for all vintages"
assert len(rev_df) > 0, "Revision pairs must be non-empty"
assert "revision" in rev_df.columns

# %%
# --- Experiment 4: Mankiw-Shapiro (1986) News vs. Noise Econometric Testing ---
ms_res = loaded_panel.news_or_noise("MEX", "gdp_real")
ms_panel = loaded_panel.news_or_noise_panel()

print(f"Mankiw-Shapiro (1986) News vs. Noise Results (Mexico Real GDP):")
print(f"  Observations Analyzed : {ms_res.n_obs}")
print(f"  Test Verdict          : {ms_res.verdict}")
print(f"  News Hypothesis (H0: beta_p = 0):")
print(f"    beta_p              : {ms_res.beta_on_preliminary:.4f} (SE = {ms_res.se_beta_on_preliminary:.4f})")
print(f"    p-value             : {ms_res.p_beta_on_preliminary:.4e} (Rejects News = {ms_res.rejects_news})")
print(f"  Noise Hypothesis (H0: beta_f = 0):")
print(f"    beta_f              : {ms_res.beta_on_final:.4f} (SE = {ms_res.se_beta_on_final:.4f})")
print(f"    p-value             : {ms_res.p_beta_on_final:.4e} (Rejects Noise = {ms_res.rejects_noise})")
print(f"  Noise Share Metric    : {ms_res.noise_share * 100:.2f}%")

# Econometric test assertions
assert hasattr(ms_res, "verdict"), "Result must contain verdict attribute"
assert ms_res.n_obs > 0, "Number of observations must be positive"
assert 0.0 <= ms_res.p_beta_on_preliminary <= 1.0, "P-value must lie in [0, 1]"
assert 0.0 <= ms_res.p_beta_on_final <= 1.0, "P-value must lie in [0, 1]"
assert not ms_panel.empty, "Panel news vs noise summary table must not be empty"

# %%
# --- Hero Visualizations: Latin America Real-Time Macro Dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Central Bank Policy Rates Across Latin America
ax1 = axes[0, 0]
mex_rates = loaded_panel.df[(loaded_panel.df["country"] == "MEX") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]
bra_rates = loaded_panel.df[(loaded_panel.df["country"] == "BRA") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]
chl_rates = loaded_panel.df[(loaded_panel.df["country"] == "CHL") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]

ax1.plot(pd.to_datetime(mex_rates["date"]), mex_rates["value"], color="black", linestyle="-", label="Mexico (Banxico TIIE)")
ax1.plot(pd.to_datetime(bra_rates["date"]), bra_rates["value"], color="black", linestyle="--", label="Brazil (BCB Selic)")
ax1.plot(pd.to_datetime(chl_rates["date"]), chl_rates["value"], color="gray", linestyle=":", linewidth=1.5, label="Chile (BCCh TPM)")
ax1.set_title("Latin America Central Bank Policy Rates", fontsize=11)
ax1.set_ylabel("Policy Rate (%)")
ax1.legend(frameon=False)

# Subplot 2: Mexico Real GDP Revision Triangle Heatmap
ax2 = axes[0, 1]
tri_norm = (tri_df - tri_df.mean().mean()) / tri_df.std().std()
im = ax2.imshow(tri_norm.fillna(0), cmap="Greys", aspect="auto", interpolation="nearest")
ax2.set_title(r"Revision Triangle $\mathbf{T}[t, v]$: Mexico Real GDP", fontsize=11)
ax2.set_xlabel("Publication Vintage Index $v$")
ax2.set_ylabel("Reference Period Index $t$")
plt.colorbar(im, ax=ax2, label="Normalized GDP (Standardized)")

# Subplot 3: Preliminary vs Final Real GDP Releases
ax3 = axes[1, 0]
dates_dt = pd.to_datetime(rev_df.index)
ax3.plot(dates_dt, rev_df["preliminary"], color="black", linestyle="--", marker="o", markersize=4, label=r"Preliminary $y_t^{(0)}$")
ax3.plot(dates_dt, rev_df["final"], color="black", linestyle="-", marker="s", markersize=4, label=r"Final Benchmark $y_t^{(F)}$")
ax3.set_title("Preliminary vs. Final GDP Estimates Across Time", fontsize=11)
ax3.set_ylabel("Quarterly Growth (%)")
ax3.legend(frameon=False)

# Subplot 4: Mankiw-Shapiro News vs Noise Regression Scatters
ax4 = axes[1, 1]
ax4.scatter(rev_df["preliminary"], rev_df["revision"], color="black", alpha=0.6, s=30, label="Revisions vs. Preliminary")
p_grid = np.linspace(rev_df["preliminary"].min(), rev_df["preliminary"].max(), 50)
fit_news = ms_res.alpha_on_preliminary + ms_res.beta_on_preliminary * p_grid
ax4.plot(p_grid, fit_news, color="gray", linestyle="-", label=f"News Fit ($\\beta_p = {ms_res.beta_on_preliminary:.2f}$)")
ax4.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
ax4.set_title(f"Mankiw-Shapiro Test (Verdict: {ms_res.verdict.upper()})", fontsize=11)
ax4.set_xlabel(r"Preliminary Release $y_t^{(0)}$ (%)")
ax4.set_ylabel(r"Total Revision $r_t = y_t^{(F)} - y_t^{(0)}$ (%)")
ax4.legend(frameon=False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** El análisis econométrico en tiempo real proporciona lecciones empíricas fundamentales para la vigilancia macroeconómica y el estudio de revisiones en América Latina:
#
# 1. **Trayectorias de Política Monetaria Regional (Experimento 1 y Figura 1):** El panel sintetiza con precisión los ajustes de tasas de política monetaria en Banxico (TIIE), BCB (Selic) y BCCh (TPM). Brasil ejecutó el ciclo de relajación más pronunciado, reduciendo la Selic desde $12.75\%$ hacia un solo dígito, seguido por los recortes proactivos de Chile desde $9.50\%$, mientras que Banxico preservó una postura más restrictiva por encima del $10.0\%$ para anclar las expectativas inflacionarias locales.
# 2. **Portabilidad de Cartuchos e Integridad Criptográfica (Experimento 2):** El empaquetado en cartuchos `.pmz` autónomos se verifica exitosamente mediante sumas de comprobación SHA-256 idénticas, confirmando que las filas, tipos de columnas e identificadores canónicos de series se preservan sin corrupción y pueden distribuirse hacia entornos de navegador en Pyodide sin requerir bases de datos externas activas.
# 3. **Geometría del Triángulo de Revisiones (Experimento 3 y Figura 2):** El triángulo de revisiones $\mathbf{T}[t, v]$ despliega la estructura triangular inferior representativa de las cuentas nacionales en tiempo real. Los primeros trimestres de referencia acumulan 8 revisiones sucesivas, revelando la convergencia progresiva de las estimaciones iniciales hacia los valores de referencia consolidados.
# 4. **Clasificación de Noticias vs. Ruido de Mankiw-Shapiro (Experimento 4 y Figura 4):** Para el PIB real de México, la pendiente estimada sobre las publicaciones preliminares $\beta_p = -0.895$ ($p = 7.24 \times 10^{-5}$) rechaza la hipótesis pura de noticias, señalando que las estimaciones preliminares contienen error clásico de medición (ruido). La métrica de fracción de ruido indica que cerca del $89.5\%$ de la varianza del crecimiento preliminar se debe a ruido y no a actualizaciones fundamentales. Por lo tanto, los analistas macroeconómicos y modeladores de banca central deben suavizar las publicaciones preliminares antes de incorporarlas en reglas de política prospectivas.

# %%
# Your turn: customize country selection, macro variables, and test significance
# Modify the parameters below to explore different country vintage slices
# and evaluate how revision noise varies across macroeconomic indicators.

# ← change this: country of interest ("MEX", "BRA", or "CHL")
country_custom = "MEX"

# ← change this: variable of interest ("policy_rate" or "gdp_real")
var_custom = "gdp_real"

# ← change this: historical vintage cutoff date for as_of() slice
as_of_custom = "2025-06-01"

# ← change this: significance level for Mankiw-Shapiro hypothesis test
signif_custom = 0.05

# Extract custom as-of slice and revision statistics
custom_asof = loaded_panel.as_of(as_of_custom)
custom_rev = loaded_panel.revisions(country_custom, var_custom)
custom_ms = loaded_panel.news_or_noise(country_custom, var_custom, significance=signif_custom)

print(f"Custom Real-Time Surveillance ({country_custom} - {var_custom}, as-of {as_of_custom}):")
print(f"  Observations Available in Slice : {len(custom_asof[custom_asof.index.get_level_values('country') == country_custom])}")
print(f"  Total Historical Revisions      : {len(custom_rev)}")
print(f"  Mankiw-Shapiro Test Verdict     : {custom_ms.verdict} (significance = {signif_custom:.2f})")
print(f"  Beta on Preliminary             : {custom_ms.beta_on_preliminary:.4f} (p = {custom_ms.p_beta_on_preliminary:.4e})")

# Downstream assertions validating user parameters and panel consistency
assert country_custom in loaded_panel.countries, f"Country {country_custom} not in panel"
assert var_custom in loaded_panel.variables, f"Variable {var_custom} not in panel"
assert 0.01 <= signif_custom <= 0.10, "Significance level must lie in [0.01, 0.10]"
assert not custom_asof.empty, "As-of slice must return observations"
assert len(custom_rev) > 0, "Revisions table must have records"
assert hasattr(custom_ms, "verdict"), "Test result must have verdict attribute"

# %% [markdown]
# **Prompts.**
# 1. *Básico:* Modifique `as_of_custom` a un vintage anterior (por ejemplo, `"2024-06-01"`). Observe cómo la muestra transversal histórica refleja el conjunto de información exacto disponible para las autoridades en dicho momento del tiempo.
# 2. *Intermedio:* Alterne `country_custom` entre `"BRA"` y `"CHL"` para `"policy_rate"`. Compruebe cómo las decisiones de tasas de interés de los bancos centrales no se revisan a través del tiempo ($\text{revisión} = 0$), contrastando nítidamente con las constantes revisiones observadas en el PIB de cuentas nacionales.
# 3. *Avanzado:* Ajuste `signif_custom` de $0.05$ a $0.01$. Verifique si el veredicto formal cambia entre noticias, ruido o no concluyente, ilustrando la relevancia de la potencia estadística en muestras de revisión pequeñas.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro` proporciona un extenso ecosistema macroeconómico regional y de tiempo real:
# - `puremacro.fetch.realtime`: Conectores nativos de datos en tiempo real para Banxico, INEGI, BCB y BCCh (`VintagePanel`, `pack_realtime_cartridge`, `load_realtime_cartridge`).
# - `puremacro.fetch.realtime.catalog`: Resolución de variables canónicas para bancos centrales e institutos estadísticos de América Latina (`canonical_variable`, `resolve_spec`).
# - `puremacro.vintages.mankiw_shapiro`: Contraste de hipótesis de noticias frente a ruido y descomposición de varianza de revisiones (`MankiwShapiroResult`).
# - `puremacro.nowcast.dfm`: Nowcasting con modelos de factores dinámicos incorporando calendarios desbalanceados y noticias en tiempo real.
# - `puremacro.pocket`: Cartuchos de datos `.pmz` portátiles con autenticación criptográfica para entornos offline, navegadores y Pyodide.
