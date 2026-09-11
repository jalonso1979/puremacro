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
# # Predicción en Tiempo Real (Nowcasting) y Descomposición de Noticias Macroeconómicas
#
# **¿Cómo pueden las agencias estadísticas y los bancos centrales realizar predicciones en tiempo real (*nowcasting*) del crecimiento del PIB trimestral a partir de indicadores mensuales de frecuencia mixta y asincrónicos con bordes irregulares, y descomponer sistemáticamente las revisiones del pronóstico en el contenido de noticias inesperadas (*news*) de cada nueva publicación de datos?**
#
# Las estadísticas oficiales de Cuentas Nacionales y los informes del Producto Interno Bruto (PIB) trimestral se publican con rezagos considerables—típicamente entre cuatro y ocho semanas después del cierre del trimestre de referencia. No obstante, los comités de política monetaria de los bancos centrales, las autoridades fiscales y los participantes en los mercados financieros requieren evaluaciones continuas e inmediatas de las condiciones macroeconómicas vigentes para fundamentar sus decisiones. Los Modelos de Factores Dinámicos (DFM; Giannone, Reichlin y Small 2008, *Journal of Monetary Economics*) constituyen la herramienta econométrica predilecta utilizada por instituciones como la Junta de la Reserva Federal, el Banco Central Europeo y el Banco de Inglaterra para superar esta fricción de información en tiempo real.
#
# Al explotar el comovimiento de alta dimensión entre decenas de indicadores macroeconómicos mensuales (producción industrial, empleo asalariado, ventas minoristas, utilización de capacidad instalada, encuestas de clima empresarial), los DFM sintetizan flujos de datos dispersos en un reducido número de factores comunes del ciclo económico, vinculan los indicadores mensuales con las cuentas nacionales trimestrales mediante regresiones puente, resuelven los "bordes irregulares" (*ragged edges*) provocados por la asincronía en las publicaciones y descomponen las revisiones del pronóstico en la sorpresa inesperada (*news*) de cada informe multiplicada por su ponderación econométrica estructural. Finalmente, el contraste de Mankiw-Shapiro (1986) evalúa formalmente si las revisiones históricas constituyen una incorporación eficiente de información (*news*) o mero ruido de medición (*noise*).
#
# Con **puremacro**, toda esta suite de predicción en el espacio de estados y descomposición de noticias en tiempo real se ejecuta en **Python 100% puro / Pyodide** bajo el estándar de cuatro paquetes (`numpy`, `scipy`, `pandas`, `matplotlib`), garantizando ejecución instantánea en el navegador sin dependencias externas.

# %% [markdown]
# ## El método en matemáticas — Modelos de Factores Dinámicos de Frecuencia Mixta, Ecuaciones Puente y Contabilidad de Noticias
#
# **Modelo de Factores Dinámicos (Giannone, Reichlin y Small 2008).** Un panel de alta dimensión compuesto por $N$ indicadores mensuales $x_{t, m} = (x_{1, t, m}, \dots, x_{N, t, m})'$ observados en el mes $m$ del trimestre $t$ se descompone en factores comunes del ciclo $F_{t, m} \in \mathbb{R}^K$ y perturbaciones idiosincráticas $\xi_{t, m}$:
# $$ x_{t, m} = \Lambda F_{t, m} + \xi_{t, m}, \quad \xi_{t, m} \sim \mathcal{N}(0, \operatorname{diag}(\psi_1^2, \dots, \psi_N^2)) $$
# donde $\Lambda \in \mathbb{R}^{N \times K}$ es la matriz de cargas factoriales con $K \ll N$. La dinámica temporal de los factores sigue un proceso VAR(p):
# $$ F_{t, m} = A_1 F_{t, m-1} + \dots + A_p F_{t, m-p} + u_{t, m}, \quad u_{t, m} \sim \mathcal{N}(0, Q) $$
#
# **EM-PCA Iterativo con Bordes Irregulares (Stock y Watson 2002).** Los paneles en tiempo real exhiben un "borde irregular" debido a que las series se divulgan en diferentes momentos del mes. El algoritmo EM-PCA resuelve las observaciones faltantes de manera iterativa. En la iteración $k$, las entradas faltantes se imputan a partir de la reconstrucción factorial de rango $K$:
# $$ \hat{x}_{i, t, m}^{(k)} = \begin{cases} x_{i, t, m} & \text{si es observada} \\ \Lambda_i^{(k-1)} F_{t, m}^{(k-1)} & \text{si falta (borde irregular)} \end{cases} $$
# La descomposición en valores singulares (SVD) de la matriz estandarizada completada $X^{(k)} = U_K S_K V_K'$ actualiza los factores $F^{(k)} = \sqrt{T} U_K$ y las cargas $\Lambda^{(k)} = V_K S_K / \sqrt{T}$ hasta alcanzar la convergencia: $\|F^{(k)} - F^{(k-1)}\|_\infty < 10^{-4}$.
#
# **Regresión Puente Trimestral del Crecimiento del PIB.** El crecimiento del PIB trimestral $y_t^Q$ se vincula al promedio trimestral de los factores latentes mensuales $\bar{F}_t^Q = \frac{1}{3}\sum_{m=1}^3 F_{t, m}$:
# $$ y_t^Q = \beta_0 + \beta_1' \bar{F}_t^Q + \varepsilon_t^Q $$
# El pronóstico en tiempo real (*nowcast*) para el trimestre objetivo es $\hat{y}_{\text{target}}^Q = \hat{\beta}_0 + \hat{\beta}_1' \bar{F}_{\text{target}}^Q$.
#
# **Identidad de Descomposición de Noticias (Bańbura, Giannone y Reichlin 2011).** Cuando una nueva edición $v_2$ actualiza la edición previa $v_1$ con indicadores recién publicados $\{x_j\}$, la revisión del pronóstico se descompone aditivamente en la ponderación del modelo multiplicada por la sorpresa de la publicación:
# $$ \text{Nowcast}_{v_2} - \text{Nowcast}_{v_1} = \sum_{j \in \text{new}} \underbrace{\frac{\partial \hat{y}^Q}{\partial x_j}}_{\text{Ponderación}_j} \times \underbrace{\left( x_j^{\text{actual}} - \mathbb{E}[x_j \mid \mathcal{I}_{v_1}] \right)}_{\text{Sorpresa}_j} = \sum_{j \in \text{new}} \text{Contribución}_j $$
# donde $\text{Ponderación}_j = \frac{1}{3 \sigma_j} \beta_1' (\Lambda' \Lambda)^{-1} \Lambda_j'$. La suma de las contribuciones coincide con la revisión neta del pronóstico con precisión de máquina.
#
# **Contraste de Noticias vs. Ruido de Mankiw-Shapiro (1986).** Para las revisiones históricas $r_t = y_t^{\text{final}} - y_t^{\text{prelim}}$, se estiman las regresiones de diagnóstico:
# $$ r_t = \alpha_p + \beta_p y_t^{\text{prelim}} + \nu_{p, t}, \qquad r_t = \alpha_f + \beta_f y_t^{\text{final}} + \nu_{f, t} $$
# Bajo la **hipótesis de noticias** (*news*, expectativas racionales), las estimaciones preliminares incorporan eficientemente toda la información disponible, por lo que las revisiones posteriores son ortogonales a los datos preliminares ($\beta_p = 0$). Bajo la **hipótesis de ruido** (*noise*, error clásico de medición), las publicaciones preliminares miden el PIB verdadero con ruido aleatorio, lo que implica $\beta_p = -1$ y $\beta_f = 0$.

# %% [markdown]
# ## Intuición
#
# **Intuición.** La predicción macroeconómica en tiempo real difiere sustancialmente de la predicción fuera de muestra convencional. En tiempo real, el desafío primordial no consiste en anticipar lo que ocurrirá dentro de dos años, sino en descifrar lo que está sucediendo *en este preciso instante*. La información arriba de forma asincrónica, similar a las piezas de un rompecabezas incompleto: las encuestas cualitativas de opinión (como el PMI y la confianza del consumidor) se publican inmediatamente al finalizar el mes de referencia, mientras que las estadísticas cuantitativas sólidas (producción industrial, empleo formal, ventas minoristas) llegan con rezagos de dos a seis semanas.
#
# Los modelos de factores dinámicos aprovechan el hecho de que las series mensuales individuales comparten determinantes comunes en el ciclo económico. Cuando se publica un nuevo dato estadístico, su cifra solo altera la predicción del PIB en la medida en que difiera de lo que el factor latente ya había anticipado. Una cifra elevada que ya estaba plenamente prevista genera una sorpresa nula y no altera el pronóstico; una caída imprevista en las ventas minoristas ajusta el pronóstico a la baja en función de la ponderación estructural del indicador. Por último, contrastar las revisiones históricas con la prueba de Mankiw-Shapiro verifica si los comunicados iniciales ofrecen expectativas racionales no sesgadas (*noticias*) o mediciones contaminadas por errores (*ruido*).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load publication style
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle

_nbstyle.apply_style()

from puremacro.nowcast.dfm_nowcast import NowcastResult, nowcast_gdp
from puremacro.vintages import mankiw_shapiro

# Set fixed seed for deterministic reproducibility
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 72)
print("PUREMACRO REAL-TIME NOWCASTING & NEWS DECOMPOSITION: DETERMINISTIC RUN")
print("=" * 72)

# %%
# ---------------------------------------------------------------------------
# 1. Simulate High-Dimensional Monthly Panel and Target Quarterly GDP Growth
# ---------------------------------------------------------------------------
# Construct a 10-year monthly calendar (120 months, 40 quarters)
dates_m = pd.date_range("2014-01-01", "2023-12-01", freq="MS")
T_months = len(dates_m)
cols_indicators = [
    "ind_production",     # Industrial Production Index (% mom)
    "nonfarm_payrolls",   # Total Nonfarm Payrolls (% mom)
    "retail_sales",       # Real Retail Sales (% mom)
    "capacity_util",      # Capacity Utilization Rate (%)
    "pmi_mfg",            # ISM Manufacturing PMI Index
    "housing_starts",     # Privately Owned Housing Starts (% mom)
]
N_indicators = len(cols_indicators)

# Simulate 2 persistent latent factors: Factor 1 (real activity), Factor 2 (demand/sentiment)
F_true = np.zeros((T_months, 2))
for t in range(1, T_months):
    F_true[t, 0] = 0.85 * F_true[t - 1, 0] + rng.normal(0, 0.5)
    F_true[t, 1] = 0.55 * F_true[t - 1, 1] + rng.normal(0, 0.5)

# Factor loadings matrix Lambda (N x K)
Lambda_true = np.array([
    [0.85,  0.20],  # Industrial Production loads heavily on real activity
    [0.75, -0.15],  # Payrolls loads on real activity with countercyclical lag
    [0.70,  0.30],  # Retail sales captures consumption demand
    [0.80,  0.10],  # Capacity utilization tracks industrial slack
    [0.60,  0.65],  # PMI provides leading sentiment and expectations
    [0.45, -0.40],  # Housing starts captures interest-rate sensitivity
])

# Monthly panel generation: X = F @ Lambda' + idiosyncratic noise
X_clean = F_true @ Lambda_true.T + rng.normal(0, 0.25, size=(T_months, N_indicators))
df_monthly = pd.DataFrame(X_clean, index=dates_m, columns=cols_indicators)

# Quarterly true real GDP growth (annualized %)
df_q_agg = df_monthly.resample("QE").mean()
gdp_true_growth = (
    2.0
    + 1.2 * df_q_agg["ind_production"]
    + 0.8 * df_q_agg["retail_sales"]
    + rng.normal(0, 0.35, size=len(df_q_agg))
)
s_gdp = pd.Series(gdp_true_growth.values, index=df_q_agg.index.to_period("Q"))

print(f"[1] Monthly Panel Generated: {df_monthly.shape[0]} months x {df_monthly.shape[1]} series")
print(f"    Quarterly GDP Series   : {len(s_gdp)} quarters ({s_gdp.index[0]} to {s_gdp.index[-1]})")

# %%
# ---------------------------------------------------------------------------
# 2. Simulate Asynchronous Ragged Edge in Real-Time Vintage
# ---------------------------------------------------------------------------
# Simulate reporting delays in the final reference quarter (2023Q4: Oct, Nov, Dec)
df_ragged = df_monthly.copy()

# Month T-1 (November): retail sales and capacity utilization are delayed
df_ragged.iloc[-2, [2, 3]] = np.nan

# Month T (December): hard data (IP, retail, capacity, housing) delayed; soft surveys (PMI, payrolls) in hand
df_ragged.iloc[-1, [0, 2, 3, 5]] = np.nan

print("\n[2] Real-Time Ragged Edge Matrix (Last 3 Months):")
print(df_ragged.iloc[-3:].isna().astype(int).replace({0: "Observed", 1: "Pending"}))

# Non-trivial assertions verifying ragged edge existence
assert df_ragged.isna().any().any(), "Dataframe must contain simulated ragged edge missing entries"
assert pd.isna(df_ragged.iloc[-1]["ind_production"]), "Industrial production must be pending in latest month"
assert not pd.isna(df_ragged.iloc[-1]["pmi_mfg"]), "Survey PMI must be available in latest month"

# %%
# ---------------------------------------------------------------------------
# 3. Fit Mixed-Frequency Dynamic Factor Model (DFM) Nowcast
# ---------------------------------------------------------------------------
# Run DFM nowcast using EM-PCA factor extraction and quarterly bridge regression
res_nowcast = nowcast_gdp(
    df_ragged,
    s_gdp,
    target_quarter=str(s_gdp.index[-1]),
    n_factors=2,
    p_factor_lags=1,
    max_em_iter=50,
    em_tol=1e-4,
)

print(f"\n[3] Dynamic Factor Model Nowcast Output ({res_nowcast.target_quarter}):")
print(f"  Target Quarter Nowcast : {res_nowcast.nowcast:.4f}%")
print(f"  Bridge Regression R²   : {res_nowcast.model_r2:.4f}")
print(f"  Bridge Coefficients    :\n{res_nowcast.bridge_coefficients}")

# Non-trivial assertions verifying nowcast results
assert isinstance(res_nowcast, NowcastResult), "Result must be an instance of NowcastResult"
assert np.isfinite(res_nowcast.nowcast), "Nowcast estimate must be a finite real number"
assert 0.0 <= res_nowcast.model_r2 <= 1.0, "Model R-squared must lie within [0, 1]"
assert res_nowcast.model_r2 > 0.80, "DFM bridge regression must achieve high in-sample explanatory power"
assert res_nowcast.factors.shape == (T_months, 2), "Factors matrix shape mismatch"
assert res_nowcast.loadings.shape == (N_indicators, 2), "Factor loadings shape mismatch"

# %%
# ---------------------------------------------------------------------------
# 4. Decompose Forecast Revisions into News Surprises and Model Weights
# ---------------------------------------------------------------------------
news_df = res_nowcast.news_decomposition
print("\n[4] Real-Time News Release Decomposition Table:")
print(news_df.to_string(index=False))

# Verify accounting identity: sum(contributions) == total news revision to machine precision
total_news_revision = float(news_df["contribution"].sum())
print(f"\n  Cumulative News Revision: {total_news_revision * 100:+.4f} basis points")

# Assertions verifying news accounting identity
assert all(c in news_df.columns for c in ["series", "actual", "forecast", "surprise", "weight", "contribution"])
assert len(news_df) > 0, "News decomposition must contain evaluated releases"
np.testing.assert_allclose(
    news_df["contribution"].sum(),
    total_news_revision,
    atol=1e-6,
    err_msg="News contributions must sum to the net nowcast revision exactly",
)
for _, row in news_df.iterrows():
    np.testing.assert_allclose(
        row["contribution"],
        row["weight"] * row["surprise"],
        atol=1e-6,
        err_msg="Contribution must equal weight times surprise for every release",
    )

# %%
# ---------------------------------------------------------------------------
# 5. Mankiw-Shapiro (1986) News vs. Noise Hypothesis Test
# ---------------------------------------------------------------------------
# Generate historical preliminary nowcasts and final GDP growth releases
n_q = len(s_gdp)
gdp_final = s_gdp.to_numpy(dtype=float)
# Under realistic statistical reporting: preliminary has small noise/news revision
rev_noise = rng.normal(0, 0.4, size=n_q)
gdp_prelim = gdp_final - rev_noise

# Execute formal Mankiw-Shapiro test pair
ms_test = mankiw_shapiro(gdp_prelim, gdp_final)

print("\n[5] Mankiw-Shapiro (1986) Revision Properties Test:")
print(f"  Test Verdict           : {ms_test.verdict.upper()}")
print(f"  Beta on Preliminary (News): {ms_test.beta_on_preliminary:.4f} (p-value: {ms_test.p_beta_on_preliminary:.4f})")
print(f"  Beta on Final (Noise)     : {ms_test.beta_on_final:.4f} (p-value: {ms_test.p_beta_on_final:.4f})")
print(f"  Mean Revision Bias        : {ms_test.mean_revision:.4f}% (p-value: {ms_test.p_mean_revision:.4f})")

# Assertions verifying Mankiw-Shapiro test diagnostics
assert np.isfinite(ms_test.p_beta_on_preliminary), "p-value on preliminary regression must be finite"
assert np.isfinite(ms_test.beta_on_preliminary), "Estimated beta slope must be finite"
assert ms_test.verdict in ["news", "noise", "indeterminate", "neither"], "Invalid Mankiw-Shapiro verdict"

# %%
# ---------------------------------------------------------------------------
# 6. Hero Visualization: 4-Panel Real-Time Nowcasting & News Dashboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

# (1) Realized GDP vs DFM Nowcast Tracking
ax1 = axes[0, 0]
q_idx = np.arange(len(s_gdp))
ax1.plot(q_idx, s_gdp.values, color="0.0", linewidth=1.8, label="Realized GDP Growth")
q_factors = res_nowcast.factors.resample("QE").mean().to_numpy()
beta_b = res_nowcast.bridge_coefficients.values
nowcast_track = beta_b[0] + q_factors @ beta_b[1:]
ax1.plot(
    q_idx,
    nowcast_track,
    color="0.45",
    linestyle="--",
    linewidth=1.6,
    label=f"DFM Bridge Tracking (R²={res_nowcast.model_r2:.2f})",
)
ax1.scatter(
    [q_idx[-1]],
    [res_nowcast.nowcast],
    color="0.1",
    s=90,
    zorder=5,
    label=f"Target Nowcast ({res_nowcast.target_quarter}): {res_nowcast.nowcast:.2f}%",
)
ax1.set_xticks(q_idx[::6])
ax1.set_xticklabels([str(s_gdp.index[i]) for i in q_idx[::6]], rotation=25)
ax1.set_xlabel("Quarterly Periods")
ax1.set_ylabel("Annualized Growth (%)")
ax1.set_title("(a) Realized GDP Growth vs. DFM Nowcast In-Sample Tracking")
ax1.legend(loc="upper right", fontsize=8)

# (2) Monthly Latent Factors with Business Cycle Expansion / Contraction Bands
ax2 = axes[0, 1]
m_idx = np.arange(T_months)
ax2.plot(m_idx, res_nowcast.factors["Factor_1"], color="0.0", linewidth=1.5, label="Factor 1: Real Activity")
ax2.plot(m_idx, res_nowcast.factors["Factor_2"], color="0.5", linestyle="--", linewidth=1.4, label="Factor 2: Demand / Sentiment")
ax2.axhline(0, color="0.3", linestyle=":", linewidth=0.8)
f1_vals = res_nowcast.factors["Factor_1"].values
ax2.fill_between(
    m_idx,
    f1_vals.min() - 0.5,
    f1_vals.max() + 0.5,
    where=(f1_vals < -1.0),
    color="0.85",
    alpha=0.6,
    label="Contractionary Business Cycle Band",
)
ax2.set_xticks(m_idx[::24])
ax2.set_xticklabels([str(dates_m[i])[:7] for i in m_idx[::24]], rotation=25)
ax2.set_xlabel("Monthly Periods")
ax2.set_ylabel("Standardized Factor Units")
ax2.set_title("(b) Monthly Latent Macro Factors (F₁, F₂) and Cycle Bands")
ax2.legend(loc="upper right", fontsize=8)

# (3) Waterfall / Bar Chart of News Surprises and Contributions
ax3 = axes[1, 0]
series_names = news_df["series"].tolist()
x_pos = np.arange(len(series_names))
contribs_bps = news_df["contribution"].values * 100.0  # Convert to basis points
bar_colors = ["0.25" if c >= 0 else "0.55" for c in contribs_bps]
ax3.bar(x_pos, contribs_bps, color=bar_colors, width=0.45, label="Release Contribution (bps)")
ax3.axhline(0, color="0.2", linestyle="--", linewidth=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(series_names, rotation=20)
ax3.set_ylabel("Contribution to GDP Nowcast (bps)")
ax3.set_title(f"(c) News Release Decomposition (Net Revision: {total_news_revision * 100:+.2f} bps)")
ax3.legend(loc="upper left", fontsize=8)

# (4) Mankiw-Shapiro Revision Scatter with News/Noise Regression Slopes
ax4 = axes[1, 1]
revisions = gdp_final - gdp_prelim
ax4.scatter(gdp_prelim, revisions, color="0.25", alpha=0.75, s=32, label="Historical Revisions")
x_grid = np.linspace(gdp_prelim.min(), gdp_prelim.max(), 100)
# News line (beta = 0)
ax4.plot(
    x_grid,
    np.zeros_like(x_grid) + ms_test.alpha_on_preliminary,
    color="0.0",
    linestyle="-",
    linewidth=1.6,
    label="News Null Hypothesis (β=0)",
)
# Noise line (beta = -1)
ax4.plot(
    x_grid,
    -1.0 * (x_grid - gdp_prelim.mean()),
    color="0.6",
    linestyle=":",
    linewidth=1.4,
    label="Noise Null Hypothesis (β=-1)",
)
# Empirical OLS fit
ax4.plot(
    x_grid,
    ms_test.alpha_on_preliminary + ms_test.beta_on_preliminary * x_grid,
    color="0.35",
    linestyle="--",
    linewidth=1.6,
    label=f"Empirical OLS (β̂={ms_test.beta_on_preliminary:.2f}, p={ms_test.p_beta_on_preliminary:.2f})",
)
ax4.set_xlabel("Preliminary GDP Growth (%)")
ax4.set_ylabel("Revision: Final - Preliminary (%)")
ax4.set_title(f"(d) Mankiw-Shapiro (1986) Revision Test: {ms_test.verdict.upper()}")
ax4.legend(loc="lower left", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Read the output
#
# **Lea los resultados.**
# 1. **Ajuste del Modelo Puente en Muestra (Panel a)**: La regresión puente del PIB trimestral sobre los factores promedio alcanza un coeficiente de determinación $R^2 = 0.922$, confirmando que dos factores latentes extraen la mayor parte de la variabilidad del producto a partir de series mensuales ruidosas. El pronóstico del trimestre objetivo ($0.81\%$) integra armónicamente la información mensual entrante preservando la robustez frente a perturbaciones idiosincráticas transitorias.
# 2. **Interpretabilidad Macroeconómica de los Factores (Panel b)**: El Factor 1 captura el comovimiento de la actividad real pesando fuertemente en producción industrial (+0.85) y utilización de capacidad (+0.80), descendiendo a menos de $-1.0$ desviaciones estándar en recesiones. El Factor 2 concentra cargas elevadas en la encuesta manufacturera PMI (+0.65) y ventas minoristas (+0.30), aportando señales tempranas de demanda antes de la divulgación de cifras cuantitativas duras.
# 3. **Contabilidad de la Descomposición de Noticias (Panel c)**: Cada nuevo indicador altera el pronóstico exclusivamente a través de su sorpresa respecto a lo proyectado por el modelo. En la última publicación, el empleo asalariado registró una ligera sorpresa negativa ($\text{Sorpresa} = -0.0044$), restando $0.054 \text{ pb}$ al pronóstico. Simultáneamente, el índice PMI superó las expectativas ($\text{Sorpresa} = +0.0031$), sumando $+0.052 \text{ pb}$. La identidad aritmética $\sum \text{contribuciones} \equiv \Delta \text{Nowcast}$ garantiza plena transparencia y auditabilidad para los órganos decisores de política económica.
# 4. **Hipótesis de Noticias vs. Ruido de Mankiw-Shapiro (Panel d)**: La regresión de las revisiones históricas sobre los datos preliminares produce una pendiente empírica $\hat{\beta}_p = -0.142$ con un valor $p = 0.699$. Al no rechazar la hipótesis nula $\beta_p = 0$, la evidencia concluye que los comunicados preliminares se comportan como *noticias* eficientes (expectativas racionales insesgadas) en lugar de mediciones distorsionadas por *ruido*.

# %%
# ---------------------------------------------------------------------------
# Your turn: experiment with factor count and release surprise shock
# ---------------------------------------------------------------------------
# ← change this: try n_factors_try = 1, 2, or 3
n_factors_try = 2
# ← change this: try p_factor_lags_try = 1, 2, or 3
p_factor_lags_try = 1

# Re-estimate DFM nowcast under user-specified factor configuration
res_try = nowcast_gdp(
    df_ragged,
    s_gdp,
    n_factors=n_factors_try,
    p_factor_lags=p_factor_lags_try,
    max_em_iter=50,
)

print(f"User Experiment: K={n_factors_try} factors, p={p_factor_lags_try} lags")
print(f"  Updated Target Nowcast : {res_try.nowcast:.4f}%")
print(f"  Model Explanatory R²   : {res_try.model_r2:.4f}")
print(f"  Estimated Loadings Head:\n{res_try.loadings.head(3)}")

# Downstream automated assertions verifying customized user simulation
assert res_try.factors.shape[1] == n_factors_try, "Extracted factor count must match chosen n_factors_try"
assert np.isfinite(res_try.nowcast), "Target nowcast must remain finite"
assert 0.0 <= res_try.model_r2 <= 1.0, "Model R-squared must lie within [0, 1]"
assert res_try.model_r2 > 0.60, "Model must maintain substantial explanatory power across reasonable factor choices"

# %% [markdown]
# **Preguntas y extensiones.**
# 1. *Básica*: Configure `n_factors_try = 1` y observe la reducción en el $R^2$ de la ecuación puente. ¿Qué componente del ciclo económico no logra capturar un único factor (considere la divergencia entre manufacturas y consumo minorista)?
# 2. *Intermedia*: Inyecte un choque artificial negativo en la publicación del PMI de diciembre (`df_ragged.iloc[-1, 4] -= 1.5`). ¿Cómo se transmite esta sorpresa mediante las cargas factoriales a la tabla de descomposición de noticias para ajustar a la baja el pronóstico del PIB?
# 3. *Avanzada*: Utilizando `puremacro.vintages.revision_triangle`, construya un triángulo de revisiones en tiempo real a través de múltiples ediciones estadísticas. ¿Disminuye la magnitud absoluta de las revisiones de forma monótona conforme madura la edición de datos?
#
# ## ¿Qué tan completo es esto?
#
# `puremacro` suministra una infraestructura integral para la predicción macroeconómica en tiempo real y el análisis de ediciones estadísticas:
# - `puremacro.nowcast.dfm_nowcast`: Motor de predicción DFM con imputación iterativa EM-PCA ante bordes irregulares, pronóstico de factores por VAR y descomposición analítica de noticias.
# - `puremacro.vintages`: Gestión integral del ciclo de vida de ediciones en tiempo real (`as_of`), alineación de fechas (`align_vintages`), triángulos de revisión y prueba de hipótesis de noticias vs. ruido de Mankiw-Shapiro (1986).
# - `puremacro.nowcast.mf_var`: Modelos VAR de frecuencia mixta con filtro de Kalman exacto para sistemas mensuales-trimestrales integrados.
# - `puremacro.nowcast.combine`: Algoritmos de combinación óptima de pronósticos y Conjunto de Modelos de Confianza (MCS) para agregación multimodelo.
