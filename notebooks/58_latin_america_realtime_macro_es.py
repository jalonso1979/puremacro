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
# **El panel de este cuaderno es simulado. No extraiga de él ningún hecho sobre América Latina.** Cada número se genera en la primera celda de código a partir de la semilla fija `np.random.default_rng(42)`: el nivel del PIB mexicano es una tendencia lineal más extracciones de `rng.normal`, y las tres trayectorias de tasa de política son líneas rectas, con la chilena aplanada en un piso. Lo que sí es real es el *esquema*: los nombres de los proveedores, los identificadores de series (`735848`, `SF61745`, `432`, `F022.TPM.TPO.D001.NO.Z.D`) y las unidades que devuelven los conectores de `puremacro`, de modo que el cuaderno ejercita la maquinaria genuina de `VintagePanel` y `.pmz` sin ninguna llamada de red ni credenciales. El cartucho `.pmz` que escribe lleva `SIMULATED` en su cadena de procedencia, así que quien reciba únicamente el cartucho se entera de lo mismo. Nada de lo que sigue es historia de Banxico, INEGI, BCB ni BCCh.
#
# **Aquí los vintages son fechas de captura, no ediciones publicadas.** Ninguna de estas cuatro fuentes conserva un archivo de publicaciones superadas: el SIE de Banxico, el BIE del INEGI, el SGS del BCB y el SIETE del BCCh sobrescriben la serie en su lugar. Por tanto un panel de tiempo real para ellas debe *acumularse*: se captura la edición vigente, se espera y se vuelve a capturar, y el historial de revisiones solo alcanza hasta la primera captura propia. Las ocho columnas de vintage que siguen representan ocho capturas de ese tipo, en ocho días distintos.
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
# Para Banxico, el INEGI, el BCB y el BCCh el índice $v$ es una **fecha de captura**, no una fecha de publicación: cada proveedor sirve únicamente la edición vigente de la serie y la sobrescribe en su lugar, de modo que $v$ registra el día en que una descarga se almacenó localmente y $v_0(t)$ es la primera captura posterior al cierre del trimestre $t$, no el día en que la agencia lo publicó por primera vez.
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
#
# El panel que se ejercita a continuación es simulado precisamente para que esta demostración pueda verificarse. Como conocemos el proceso generador de datos —una tendencia determinista, ruido de medición sembrado solo en las ediciones recientes, líneas de tasa de política sin revisar— podemos decir exactamente qué *debería* reportar cada diagnóstico, y las aserciones de cada celda obligan a la biblioteca a cumplirlo. Una captura real de los cuatro conectores mostraría la misma maquinaria sobre números que nadie controla; también requeriría red, credenciales para Banxico y meses de capturas acumuladas antes de que existiera revisión alguna que contrastar.

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
# --- Experiment 1: Assemble a SIMULATED Multi-Country Latin America Vintage Panel ---
# EVERY VALUE BELOW IS GENERATED IN THIS CELL. The provider names, series identifiers
# and units are the real ones the puremacro connectors return, so the panel exercises
# the genuine schema offline (no network call, no credentials anywhere in this file).
# The numbers themselves are stylized paths invented for this notebook and are NOT
# Banxico, INEGI, BCB or BCCh history -- do not quote them as facts about the region.
#
# Construct quarterly reference periods: 2022Q1 to 2025Q3 (15 reference quarters).
# Publication vintages: 8 quarterly SNAPSHOT DATES spanning 2024Q1 to 2025Q4. All four
# of these sources publish only the current edition of a series, so a real panel is
# accumulated one capture at a time and its revision history reaches back only as far
# as the first capture; the 8 columns here stand in for 8 such captures.
ref_dates = pd.date_range("2022-01-01", "2025-07-01", freq="QS").strftime("%Y-%m-%d").tolist()
vintage_dates = pd.date_range("2024-01-01", "2025-10-01", freq="QS").strftime("%Y-%m-%d").tolist()

rows = []
for v_idx, v in enumerate(vintage_dates):
    for d_idx, d in enumerate(ref_dates):
        if d <= v:
            # Mexico: Real GDP level (INEGI indicator 735848). Simulated preliminary
            # measurement noise is planted only in editions published within 180 days
            # of the reference quarter; older editions repeat the same trend value.
            base_gdp = 24000000.0 + 150000.0 * d_idx
            noise = float(rng.normal(0, 50000.0)) if d == v or (pd.to_datetime(v) - pd.to_datetime(d)).days <= 180 else 0.0
            rows.append({
                "country": "MEX", "variable": "gdp_real", "date": d, "vintage": v,
                "value": base_gdp + noise, "provider": "inegi", "series_id": "735848", "units": "level"
            })
            # Mexico: Policy rate (Banxico TIIE objetivo, series SF61745).
            # Simulated straight line, 11.25% down to 7.75% at 25 bp per quarter.
            rows.append({
                "country": "MEX", "variable": "policy_rate", "date": d, "vintage": v,
                "value": 11.25 - 0.25 * d_idx, "provider": "banxico", "series_id": "SF61745", "units": "rate"
            })
            # Brazil: Policy rate (BCB Taxa Selic, series 432).
            # Simulated straight line, 12.75% down to 5.75% at 50 bp per quarter.
            rows.append({
                "country": "BRA", "variable": "policy_rate", "date": d, "vintage": v,
                "value": 12.75 - 0.50 * d_idx, "provider": "bcb", "series_id": "432", "units": "rate"
            })
            # Chile: Policy rate (BCCh TPM, series F022.TPM.TPO.D001.NO.Z.D).
            # Simulated straight line, 9.50% down to a 3.00% floor, so the synthetic
            # rate never reaches zero or turns negative.
            rows.append({
                "country": "CHL", "variable": "policy_rate", "date": d, "vintage": v,
                "value": max(3.00, 9.50 - 0.50 * d_idx), "provider": "bcch", "series_id": "F022.TPM.TPO.D001.NO.Z.D", "units": "rate"
            })

df_raw = pd.DataFrame(rows)
panel_raw = VintagePanel(df_raw)
min_policy_rate = float(df_raw.loc[df_raw["variable"] == "policy_rate", "value"].min())

print(f"Constructed Multi-Country Vintage Panel:")
print("  Data Provenance    : SIMULATED (stylized paths on real provider/series identifiers)")
print(f"  Total Observations : {len(panel_raw):,}")
print(f"  Countries Included : {panel_raw.countries}")
print(f"  Macro Variables    : {panel_raw.variables}")
print(f"  Reference Periods  : {len(ref_dates)} quarters ({ref_dates[0]} to {ref_dates[-1]})")
print(f"  Vintage Snapshots  : {len(vintage_dates)} captures ({vintage_dates[0]} to {vintage_dates[-1]})")
print(f"  Lowest Policy Rate : {min_policy_rate:.2f}% (simulated floor)")

# Panel structural assertions
assert panel_raw.countries == ["BRA", "CHL", "MEX"]
assert "policy_rate" in panel_raw.variables
assert "gdp_real" in panel_raw.variables
assert len(panel_raw) > 300
# A simulated policy rate that goes non-positive would be economically absurd, and it
# would be undefined under the log transform the revision tools apply to a series
# declared units="level"; these three declare units="rate", which is read in levels.
assert min_policy_rate > 0.0, "Simulated policy rates must stay strictly positive"

# %%
# --- Experiment 2: Self-Verifying Cryptographic Cartridge Packaging & Loading ---
# Package panel into an offline .pmz cartridge with SHA-256 verification
with tempfile.TemporaryDirectory() as tmp_dir:
    cartridge_file = Path(tmp_dir) / "latam_realtime_macro.pmz"
    pack_realtime_cartridge(
        panel_raw,
        cartridge_file,
        source="SIMULATED panel on Banxico, INEGI, BCB and BCCh series identifiers",
        vintage="2026-04-01",
        notes=(
            "Latin America real-time vintage cartridge for puremacro showcase 58. "
            "SYNTHETIC DATA: values are generated from a fixed seed, not fetched from "
            "any provider. Vintage columns are snapshot dates, not published editions."
        ),
    )
    assert cartridge_file.exists(), "Cartridge file must be created on disk"
    loaded_panel = load_realtime_cartridge(cartridge_file, verify=True)

print(f"Portable Cartridge Authentication:")
print(f"  SHA-256 Digest Verification : SUCCESS")
print(f"  Declared Provenance         : {loaded_panel.metadata['provenance_source']}")
print(f"  Loaded Countries            : {loaded_panel.countries}")
print(f"  Loaded Variables            : {loaded_panel.variables}")
print(f"  Loaded Record Count         : {len(loaded_panel):,}")

# Cartridge integrity assertions
assert isinstance(loaded_panel, VintagePanel)
assert loaded_panel.countries == ["BRA", "CHL", "MEX"]
assert len(loaded_panel) == len(panel_raw)
# A reader who receives only the .pmz must still learn the data is simulated.
assert "SIMULATED" in loaded_panel.metadata["provenance_source"]

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
print(f"  Mean Revision (signed)        : {rev_df['revision'].mean():.4f} percentage points")
print(f"  Mean Absolute Revision        : {rev_df['revision'].abs().mean():.4f} percentage points")

# Coverage and revision assertions
assert not cov_df.empty, "Coverage table must not be empty"
assert not as_of_2025.empty, "As-of slice must return observations"
assert tri_df.shape[1] == len(vintage_dates), "Triangle must have columns for all vintages"
assert len(rev_df) > 0, "Revision pairs must be non-empty"
assert "revision" in rev_df.columns

# %%
# --- Experiment 4: Mankiw-Shapiro (1986) News vs. Noise Econometric Testing ---
ms_res = loaded_panel.news_or_noise("MEX", "gdp_real")
# min_obs defaults to 12; this simulated panel has 7 observable revision pairs per series.
ms_panel = loaded_panel.news_or_noise_panel(min_obs=len(rev_df))
print(ms_panel[["country", "variable", "n_obs", "beta_on_preliminary", "verdict", "ok", "note"]].to_string(index=False))

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
gdp_row = ms_panel[(ms_panel["country"] == "MEX") & (ms_panel["variable"] == "gdp_real")]
assert bool(gdp_row["ok"].iloc[0]), "The panel must estimate the Mexican GDP test, not just list it"
assert np.isclose(gdp_row["beta_on_preliminary"].iloc[0], ms_res.beta_on_preliminary), "Panel row must match the single-series test"
rate_rows = ms_panel[ms_panel["variable"] == "policy_rate"]
assert not rate_rows["ok"].any(), "Unrevised policy rates have no revisions to test"
assert rate_rows["note"].str.contains("zero").all(), "Each skipped row must say why"

# %%
# --- Hero Visualizations: Latin America Real-Time Macro Dashboard (SIMULATED panel) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Central Bank Policy Rates Across Latin America
ax1 = axes[0, 0]
mex_rates = loaded_panel.df[(loaded_panel.df["country"] == "MEX") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]
bra_rates = loaded_panel.df[(loaded_panel.df["country"] == "BRA") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]
chl_rates = loaded_panel.df[(loaded_panel.df["country"] == "CHL") & (loaded_panel.df["variable"] == "policy_rate") & (loaded_panel.df["vintage"] == vintage_dates[-1])]

ax1.plot(pd.to_datetime(mex_rates["date"]), mex_rates["value"], color=_nbstyle.S1["color"], linestyle=_nbstyle.S1["linestyle"], label="Mexico (Banxico TIIE)")
ax1.plot(pd.to_datetime(bra_rates["date"]), bra_rates["value"], color=_nbstyle.S2["color"], linestyle=_nbstyle.S2["linestyle"], label="Brazil (BCB Selic)")
ax1.plot(pd.to_datetime(chl_rates["date"]), chl_rates["value"], color=_nbstyle.S3["color"], linestyle=_nbstyle.S3["linestyle"], linewidth=1.5, label="Chile (BCCh TPM)")
ax1.set_title("Simulated Latin America Policy Rates", fontsize=11)
ax1.set_ylabel("Policy Rate (%)")
ax1.legend(frameon=False)

# Subplot 2: Mexico Real GDP Revision Triangle, as the revision still to come in each cell
ax2 = axes[0, 1]
latest = tri_df.ffill(axis=1).iloc[:, -1]
still_to_come = 100.0 * (latest.to_numpy()[:, None] / tri_df.to_numpy() - 1.0)   # % of the edition's value
im = ax2.imshow(np.abs(still_to_come), cmap=_nbstyle.CMAP_SEQ, aspect="auto", interpolation="nearest")
ax2.set_title(r"Revision Triangle $\mathbf{T}[t, v]$: Simulated Mexico Real GDP", fontsize=11)
ax2.set_xlabel("Snapshot Index $v$")
ax2.set_ylabel("Reference Period Index $t$")
plt.colorbar(im, ax=ax2, label="|Revision still to come| (% of level)")

# Subplot 3: Preliminary vs Final Real GDP Releases
ax3 = axes[1, 0]
dates_dt = pd.to_datetime(rev_df.index)
ax3.plot(dates_dt, rev_df["preliminary"], color=_nbstyle.S2["color"], linestyle="--", marker="o", markersize=4, label=r"Preliminary $y_t^{(0)}$")
ax3.plot(dates_dt, rev_df["final"], color=_nbstyle.S1["color"], linestyle="-", marker="s", markersize=4, label=r"Final Benchmark $y_t^{(F)}$")
ax3.set_title("Simulated Preliminary vs. Final GDP Estimates", fontsize=11)
ax3.set_ylabel("Quarterly Growth (%)")
ax3.legend(frameon=False)

# Subplot 4: Mankiw-Shapiro News vs Noise Regression Scatters
ax4 = axes[1, 1]
ax4.scatter(rev_df["preliminary"], rev_df["revision"], color=_nbstyle.S1["color"], alpha=0.6, s=30, label="Revisions vs. Preliminary")
p_grid = np.linspace(rev_df["preliminary"].min(), rev_df["preliminary"].max(), 50)
fit_news = ms_res.alpha_on_preliminary + ms_res.beta_on_preliminary * p_grid
ax4.plot(p_grid, fit_news, color=_nbstyle.S2["color"], linestyle="-", label=f"News Fit ($\\beta_p = {ms_res.beta_on_preliminary:.2f}$)")
ax4.axhline(0.0, color=_nbstyle.SPINE, linestyle=":", linewidth=0.8)
ax4.set_title(f"Mankiw-Shapiro Test (Verdict: {ms_res.verdict.upper()})", fontsize=11)
ax4.set_xlabel(r"Preliminary Release $y_t^{(0)}$ (%)")
ax4.set_ylabel(r"Total Revision $r_t = y_t^{(F)} - y_t^{(0)}$ (%)")
ax4.legend(frameon=False)

fig.suptitle(
    "SIMULATED real-time panel: values are generated from a fixed seed, not fetched",
    fontsize=12, fontweight="bold", y=1.00,
)

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Todo lo que sigue es una propiedad del panel **simulado** construido en el Experimento 1, no una medición de ninguna economía latinoamericana. Lo que se demuestra es la maquinaria —el objeto panel, el cartucho, el triángulo y la econometría— funcionando correctamente sobre datos cuya verdad controlamos:
#
# 1. **Trayectorias Simuladas de Tasa de Política (Experimento 1 y Figura 1):** La Figura 1 grafica las tres trayectorias simuladas de tasa de política en la última captura. México desciende en línea recta de $11.25\%$ a $7.75\%$ ($25$ pb por trimestre), Brasil de $12.75\%$ a $5.75\%$ ($50$ pb por trimestre) y Chile de $9.50\%$ hasta un piso de $3.00\%$. Son líneas inventadas, elegidas para dar algo que dibujar a la maquinaria multipaís; el valor impreso `Lowest Policy Rate` confirma que el piso mantiene toda tasa simulada estrictamente positiva, de modo que ninguna trayectoria simulada resulta económicamente absurda. El catálogo declara las tres series con `units="rate"`, que las herramientas de revisiones leen en niveles y no en diferencias logarítmicas, así que el piso es una salvaguarda de plausibilidad y no una exigencia del contraste. Sus niveles, su orden y sus pendientes no informan nada sobre la TIIE, la Selic ni la TPM.
# 2. **Portabilidad de Cartuchos e Integridad Criptográfica (Experimento 2):** El empaquetado en cartuchos `.pmz` autónomos se verifica exitosamente mediante sumas de comprobación SHA-256 idénticas, confirmando que las filas, tipos de columnas e identificadores canónicos de series se preservan sin corrupción y pueden distribuirse hacia entornos de navegador en Pyodide sin requerir bases de datos externas activas. El viaje de ida y vuelta también conserva la cadena de procedencia: el panel cargado reporta `SIMULATED panel on Banxico, INEGI, BCB and BCCh series identifiers` y la celda lo verifica con una aserción, de modo que el cartucho no puede circular despojado de esa advertencia.
# 3. **Geometría del Triángulo de Revisiones (Experimento 3 y Figura 2):** El triángulo $\mathbf{T}[t, v]$ del PIB mexicano tiene $15$ fechas de referencia $\times$ $8$ capturas. Como la primera captura (2024T1) es posterior al primer trimestre de referencia (2022T1), la geometría es la inversa de un archivo clásico: las filas *más antiguas* están completas en las ocho columnas y la más reciente solo tiene dos. `revisions()` devuelve $7$ pares, uno por cada trimestre de referencia de 2024T1 a 2025T3, porque `require_observable_first` censura todo trimestre que terminó antes de la captura más temprana: para esos, la columna disponible más antigua ya es una cifra revisada, y tomarla como primera publicación subestimaría toda revisión calculada a partir de ella. Cada uno de esos siete trimestres fue perturbado por el generador, que siembra ruido solo en ediciones publicadas dentro de los $180$ días posteriores al trimestre de referencia: en total se perturban ocho trimestres de referencia, de 2023T4 a 2025T3, y la censura descarta el primero de ellos. La revisión media de $0.0367$ puntos porcentuales tiene signo, así que las revisiones positivas y negativas se compensan; el tamaño que conviene leer es la revisión absoluta media que también se imprime, y también es un hecho sobre la simulación. La Figura 2 sombrea cada celda según cuánto se revisaría todavía esa edición hasta llegar a la captura más reciente: las celdas oscuras están en la diagonal de ediciones recientes, y toda fila anterior a 2023T4 queda en blanco porque nunca se revisó.
# 4. **Clasificación de Noticias vs. Ruido de Mankiw-Shapiro (Experimento 4 y Figura 4):** El contraste devuelve el veredicto **`neither`** (ninguna de las dos), y el cuarto panel se titula en consecuencia. La pendiente sobre la publicación preliminar es $\beta_p = -0.8947$ (EE $0.0749$, $p = 7.24 \times 10^{-5}$), que rechaza la hipótesis de noticias, y la fracción de ruido $\max(0, -\beta_p)$ es $89.47\%$. Pero la pendiente sobre la publicación *final* también dista mucho de cero, $\beta_f = -2.0301$ (EE $0.5327$, $p = 0.0125$), de modo que la hipótesis de ruido también se rechaza, y una revisión que no es ortogonal a ninguna de las dos publicaciones no es ni noticia pura ni ruido puro. Leer el rechazo de $\beta_p$ por sí solo como "las revisiones son ruido" es exactamente el error contra el que advierte `docs/real_time_data.md`: $\beta_p$ identifica la *fracción* de ruido, y el veredicto proviene del par de regresiones. La razón mecánica de que la rama de ruido rechace aquí es la muestra: solo $n = 7$ trimestres de referencia sobreviven al filtro de observabilidad, y para el más reciente de ellos, 2025T3, el valor "final" es todavía una edición temprana y ruidosa —la última captura cae dentro de la misma ventana de $180$ días—, así que el error de medición sembrado contamina $y_t^{(F)}$ tanto como $y_t^{(0)}$. La Indicación 3 más abajo estrecha el nivel de significancia a $0.01$, con el cual $\beta_f$ ya no rechaza y el veredicto sí cambia a `noise`. La versión de panel del contraste, `news_or_noise_panel(min_obs=7)`, reproduce exactamente esta fila del PIB mexicano y reporta las tres tasas de política como `ok=False` con una nota: se republican sin cambios, así que sus revisiones son exactamente cero y no hay nada que contrastar.

# %%
# Your turn: customize country selection, macro variables, and test significance
# Modify the parameters below to explore different country vintage slices
# and evaluate how revision noise varies across macroeconomic indicators.

# ← change this: country of interest ("MEX", "BRA", or "CHL")
country_custom = "MEX"

# ← change this: variable of interest ("gdp_real", simulated for MEX only, or "policy_rate")
var_custom = "gdp_real"

# ← change this: historical vintage cutoff date for as_of() slice (clamped to the
# first snapshot: an earlier date has no information set to report)
as_of_custom = "2025-06-01"

# ← change this: significance level for Mankiw-Shapiro hypothesis test
signif_custom = 0.05

# Guard the knob against its own options. Three combinations are legitimate and
# still cannot be tested: (a) a date before the first snapshot, where as_of()
# returns an empty frame with no country index; (b) a (country, variable) pair the
# panel does not carry -- only Mexico has gdp_real here; and (c) a series that is
# never revised, which every simulated policy rate is by construction. In case (c)
# news_or_noise() raises rather than returning a meaningless slope, so the test is
# reported as skipped instead of run.
first_vintage = min(vintage_dates)
as_of_effective = max(as_of_custom, first_vintage)
custom_asof = loaded_panel.as_of(as_of_effective)
n_in_slice = int((custom_asof.index.get_level_values("country") == country_custom).sum())

pairs_available = set(zip(loaded_panel.df["country"], loaded_panel.df["variable"]))
has_series = (country_custom, var_custom) in pairs_available
custom_rev = loaded_panel.revisions(country_custom, var_custom) if has_series else None
is_revised = custom_rev is not None and len(custom_rev) >= 3 and float(custom_rev["revision"].abs().max()) > 0.0
custom_ms = (
    loaded_panel.news_or_noise(country_custom, var_custom, significance=signif_custom)
    if is_revised else None
)

print(f"Custom Real-Time Surveillance ({country_custom} - {var_custom}, as-of {as_of_effective}):")
print(f"  Observations Available in Slice : {n_in_slice}")
if not has_series:
    print(f"  Series Not In Panel             : {country_custom} carries no {var_custom} column")
else:
    print(f"  Total Historical Revisions      : {len(custom_rev)}")
if custom_ms is None:
    skip_reason = "series not in panel" if not has_series else "series is unrevised or has < 3 revision pairs"
    print(f"  Mankiw-Shapiro Test             : skipped ({skip_reason})")
else:
    print(f"  Mankiw-Shapiro Test Verdict     : {custom_ms.verdict} (significance = {signif_custom:.2f})")
    print(f"  Beta on Preliminary             : {custom_ms.beta_on_preliminary:.4f} (p = {custom_ms.p_beta_on_preliminary:.4e})")

# Downstream assertions validating user parameters and panel consistency
assert country_custom in loaded_panel.countries, f"Country {country_custom} not in panel"
assert var_custom in loaded_panel.variables, f"Variable {var_custom} not in panel"
assert 0.01 <= signif_custom <= 0.10, "Significance level must lie in [0.01, 0.10]"
assert as_of_effective >= first_vintage, "As-of date must be on or after the first snapshot"
assert not custom_asof.empty, "As-of slice must return observations"
assert custom_rev is None or "revision" in custom_rev.columns, "Revisions table must carry a revision column"
assert custom_ms is None or hasattr(custom_ms, "verdict"), "Test result must have verdict attribute"

# %% [markdown]
# **Indicaciones.**
# 1. *Básico:* Modifique `as_of_custom` a una captura anterior (por ejemplo, `"2024-06-01"`). Observe cómo la muestra transversal se reduce al conjunto de información que habría estado disponible en esa captura. Póngala antes de la primera captura (`"2023-06-01"`) y la celda la ajusta a la primera captura e imprime esa fecha ajustada en su encabezado, porque antes de la primera captura no hay conjunto de información que reportar.
# 2. *Intermedio:* Cambie `country_custom` a `"BRA"` o `"CHL"` y `var_custom` a `"policy_rate"`. Toda tasa de política simulada es una línea determinista, idéntica en las ocho capturas, así que `revisions()` devuelve siete filas de exactamente cero y el contraste de Mankiw-Shapiro se reporta como **omitido**: una regresión degenerada no identifica pendiente alguna. Si deja `var_custom = "gdp_real"` y cambia el país, se imprime `Series Not In Panel`, porque aquí solo México lleva una serie de PIB simulada.
# 3. *Avanzado:* Ajuste `signif_custom` de $0.05$ a $0.01$. El $p = 0.0125$ de la rama de ruido queda ahora por encima del umbral mientras el $p = 7.24 \times 10^{-5}$ de la rama de noticias sigue por debajo, de modo que el veredicto pasa de `neither` a `noise`: un recordatorio vívido de que con $n = 7$ el veredicto habla tanto de potencia estadística como de los datos.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro` proporciona un extenso ecosistema macroeconómico regional y de tiempo real:
# - `puremacro.fetch.realtime`: Conectores nativos de datos en tiempo real para Banxico, INEGI, BCB y BCCh (`VintagePanel`, `pack_realtime_cartridge`, `load_realtime_cartridge`).
# - `puremacro.fetch.realtime.catalog`: Resolución de variables canónicas para bancos centrales e institutos estadísticos de América Latina (`canonical_variable`, `resolve_spec`).
# - `puremacro.vintages`: Contraste de hipótesis de noticias frente a ruido y descomposición de varianza de revisiones (`mankiw_shapiro`, `MankiwShapiroResult`).
# - `puremacro.nowcast.dfm`: Nowcasting con modelos de factores dinámicos incorporando calendarios desbalanceados y noticias en tiempo real.
# - `puremacro.pocket`: Cartuchos de datos `.pmz` portátiles con autenticación criptográfica para entornos offline, navegadores y Pyodide.
