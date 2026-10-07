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
# # Datos en Tiempo Real, Ediciones Históricas y la Prueba de Mankiw-Shapiro
#
# **¿Cómo afectan las revisiones de datos al análisis macroeconómico, y cómo distinguir si una primera publicación estadística es un pronóstico eficiente de la cifra final ("noticia") o la cifra final más un error de medición ("ruido")?**
#
# **El panel de ediciones de este cuaderno es simulado, y el proceso que lo genera es conocido: ruido puro.** Los nombres de países y variables son solo etiquetas. No se descarga nada, y ningún número de abajo describe a Estados Unidos, Alemania, el Reino Unido ni México.
#
# Los agregados macroeconómicos como el PIB, el consumo y la inversión se revisan durante años a medida que las agencias estadísticas incorporan datos administrativos, encuestas de referencia y cambios metodológicos:
#
# 1. **Información en tiempo real y errores de política.** Orphanides (2001, *American Economic Review*) mostró que los errores de política de los años setenta reflejaron en parte estimaciones de la brecha del producto hechas con datos en tiempo real que después se revisaron fuertemente. Croushore & Stark (2001, *Journal of Econometrics*) construyeron la base de datos en tiempo real que volvió rutinario ese análisis.
#
# 2. **La prueba de noticia contra ruido de Mankiw & Shapiro (1986).** Sea $y_{0,t}$ la primera publicación del crecimiento del trimestre $t$ y $y_{T,t}$ la cifra final revisada. La revisión es
#    $$ r_t = y_{T,t} - y_{0,t} $$
#    Mankiw & Shapiro (1986, *Survey of Current Business*; NBER Working Paper 1939) regresan la revisión sobre cada publicación:
#    $$ r_t = \alpha_0 + \beta_0 \, y_{0,t} + \varepsilon_t, \qquad r_t = \alpha_T + \beta_T \, y_{T,t} + \eta_t $$
#
#    - **Noticia** ($\beta_0 = 0$): la primera publicación es un pronóstico eficiente, $y_{0,t} = \mathbb{E}[y_{T,t} \mid \Omega_t]$, de modo que la revisión es información nueva, no correlacionada con lo publicado.
#    - **Ruido** ($\beta_T = 0$): la primera publicación es la cifra final más un error de medición clásico, $y_{0,t} = y_{T,t} + v_t$ con $\text{Cov}(y_{T,t}, v_t) = 0$. La revisión $r_t = -v_t$ no está correlacionada con la cifra final pero sí, negativamente, con la primera publicación, con pendiente $\beta_0 = -\text{Var}(v)/\text{Var}(y_0)$. Esa pendiente está entre $-1$ y $0$; se acerca a $-1$ solo cuando el ruido domina la varianza de la primera publicación.
#
# Revisamos el catálogo de series en tiempo real de `puremacro`, construimos un panel simulado de ediciones, dibujamos triángulos de revisión, estimamos ambas regresiones con `QNAVintagePanel.revision_stats` (que las delega en `puremacro.vintages.mankiw_shapiro`) y extraemos conjuntos de datos en un punto del tiempo con `puremacro.fetch.QNAVintagePanel`.

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

from puremacro.fetch import (
    get_qna_vintage_catalog,
    QNAVintagePanel,
)
from puremacro.vintages import mankiw_shapiro

# %% [markdown]
# ## 1. El Catálogo de Ediciones de Cuentas Nacionales
#
# La investigación en tiempo real necesita cada observación tal como se publicó en una fecha dada. Archivos como ALFRED (ArchivaL Federal Reserve Economic Data) del Banco de la Reserva Federal de St. Louis guardan cada serie junto con sus fechas de publicación (ediciones).
#
# `get_qna_vintage_catalog()` enumera los identificadores de series que pedirían los descargadores en tiempo real de `puremacro`: una fila por país y variable de cuentas nacionales, con identificadores construidos con el patrón de nombres de los Main Economic Indicators de la OCDE en FRED (Estados Unidos usa sus propias series de FRED). El catálogo es una tabla de la librería; llamarlo no descarga nada, y si ALFRED tiene ediciones para un identificador dado solo se sabe al descargarlo.

# %%
catalog = get_qna_vintage_catalog()
print(f"Catalogue: {len(catalog)} series, {catalog['country'].nunique()} countries, {catalog['variable'].nunique()} variables")
print("\nSample of catalogue entries:")
print(catalog[["country", "country_name", "region", "variable", "series_id"]].head(10).to_string(index=False))

# Summary by variable
var_counts = catalog.groupby("variable")["country"].count().reset_index()
var_counts.columns = ["Variable", "Country Count"]
print("\nVariable coverage across countries:")
print(var_counts.to_string(index=False))

# %% [markdown]
# ## 2. Un Panel Simulado de Ediciones en Tiempo Real
#
# Simulamos 40 trimestres de referencia ($T = 40$) y 50 ediciones trimestrales de publicación ($V = 50$) para cuatro etiquetas de país y dos variables. Sea $y_t^*$ la tasa de crecimiento verdadera del trimestre $t$, extraída de forma independiente con media $2.2$ y desviación estándar $\sigma_* = 1.5$. La primera publicación aparece un trimestre después y contiene un error de medición clásico $v_t \sim \mathcal{N}(0, \sigma_v^2)$, $\sigma_v = 0.6$, independiente de $y_t^*$:
#
# $$ y_{0, t} = y_t^* + v_t $$
#
# Cada edición posterior $j \ge 1$ elimina parte del error:
#
# $$ y_{j, t} = y_t^* + e^{-j / \kappa} \, v_t, \qquad \kappa = 3 $$
#
# Es un proceso de **ruido puro**: no hay noticias, porque cada revisión solo elimina error. Tras las 49 ediciones disponibles para el primer trimestre, el error restante es despreciable; el trimestre más reciente ha tenido 9 revisiones y conserva $e^{-3} \approx 5\%$ de su error. La sección 5 usa esta verdad conocida para calificar las pruebas.

# %%
n_quarters = 40
obs_dates = pd.date_range("2010-01-01", periods=n_quarters, freq="QS")
vintage_dates = pd.date_range("2010-04-01", periods=50, freq="QS")
countries = ["USA", "DEU", "GBR", "MEX"]   # labels only: every series is simulated
variables = ["gdp_real", "gfcf_real"]
mean_growth, sd_true, sd_noise, kappa_decay = 2.2, 1.5, 0.6, 3.0

def simulate_vintages(sd_v=sd_noise, dgp="noise", seed=1986):
    """Simulated vintages for every country label and variable.

    dgp="noise": y_0 = y* + v and each vintage removes part of v (classical measurement error).
    dgp="news" : y_0 = y* is the efficient first estimate and each vintage adds part of the news v.
    """
    rng = np.random.default_rng(seed)
    records = []
    for c in countries:
        for var in variables:
            y_star = rng.normal(loc=mean_growth, scale=sd_true, size=n_quarters)
            v = rng.normal(loc=0.0, scale=sd_v, size=n_quarters)
            for i, obs in enumerate(obs_dates):
                for j, vint in enumerate(vintage_dates):
                    if j >= i + 1:   # first published one quarter after the reference quarter
                        decay = np.exp(-(j - i - 1) / kappa_decay)
                        if dgp == "noise":
                            val = y_star[i] + decay * v[i]
                        else:
                            val = y_star[i] + (1.0 - decay) * v[i]
                        records.append({"country": c, "variable": var, "date": obs, "vintage": vint, "value": float(val)})
    return QNAVintagePanel(df=pd.DataFrame(records))

panel = simulate_vintages()
print(f"Simulated QNAVintagePanel: {len(panel.df):,} records, {len(countries)} country labels x {len(variables)} variables")

# %% [markdown]
# ## 3. Visualización del Triángulo de Revisiones $(T \times V)$
#
# Los datos por edición suelen guardarse como una matriz triangular de revisiones $\mathbf{R} \in \mathbb{R}^{T \times V}$:
#
# $$ \mathbf{R} = \begin{bmatrix} y_{1, v_1} & y_{1, v_2} & y_{1, v_3} & \dots & y_{1, v_V} \\ \text{NaN} & y_{2, v_2} & y_{2, v_3} & \dots & y_{2, v_V} \\ \text{NaN} & \text{NaN} & y_{3, v_3} & \dots & y_{3, v_V} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \text{NaN} & \text{NaN} & \text{NaN} & \dots & y_{T, v_V} \end{bmatrix} $$
#
# Cada fila es un trimestre de referencia $t$ y cada columna una edición de publicación $v$. La diagonal principal contiene la primera publicación de cada trimestre; leer a lo largo de una fila muestra cómo se revisó la estimación de ese trimestre.

# %%
tri_usa = panel.revision_matrix("USA", "gdp_real")
print("Simulated 'USA gdp_real' revision matrix (first 6 quarters x 6 vintages):")
print(tri_usa.iloc[:6, :6].round(3).to_string())

# %%
fig, ax = _nbstyle.figura(figsize=(9.2, 5.2))

im = ax.imshow(tri_usa.iloc[:24, :24].to_numpy(), cmap=_nbstyle.CMAP_SEQ, aspect="auto")
ax.set_title("Simulated 'USA gdp_real': Revision Triangle (T × V)", fontsize=11, fontweight="bold")
ax.set_xlabel("Publication Vintage", color=_nbstyle.TEXTO)
ax.set_ylabel("Reference Quarter", color=_nbstyle.TEXTO)

ax.set_xticks(range(0, 24, 4))
ax.set_xticklabels([f"{d.year}Q{d.quarter}" for d in tri_usa.columns[:24:4]], rotation=45)
ax.set_yticks(range(0, 24, 4))
ax.set_yticklabels([f"{d.year}Q{d.quarter}" for d in tri_usa.index[:24:4]])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Simulated growth rate (%)", color=_nbstyle.TEXTO)

# %% [markdown]
# ## 4. Primera Publicación contra Última Estimación
#
# Comparar la primera publicación con la última estimación muestra el tamaño y el signo de las revisiones:
#
# $$ r_t = y_{T, t} - y_{0, t} $$
#
# Si las revisiones no son cero en promedio ($\bar{r} \neq 0$), la primera publicación está sesgada. Si están correlacionadas con el ciclo económico, las decisiones basadas en datos sin revisar pueden equivocarse de forma sistemática (Orphanides, 2001).

# %%
s_first = panel.first_release("USA", "gdp_real")
s_latest = panel.latest_release("USA", "gdp_real")

fig, (ax1, ax2) = _nbstyle.figura(2, 1, figsize=(9.5, 6.0), sharex=True)

ax1.plot(s_first.index, s_first.values, **_nbstyle.S1, label="First release")
ax1.plot(s_latest.index, s_latest.values, **_nbstyle.S2, label="Latest estimate")
ax1.set_title("Simulated 'USA gdp_real': First Release vs. Latest Estimate", fontsize=11, fontweight="bold")
ax1.set_ylabel("Growth rate (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

revision = s_latest - s_first
ax2.bar(revision.index, revision.values, color=_nbstyle.S3["color"], width=60, edgecolor=_nbstyle.SPINE, alpha=0.8, label="Revision ($y_T - y_0$)")
ax2.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="-")
ax2.set_title("Total Revision", fontsize=11, fontweight="bold")
ax2.set_xlabel("Reference quarter", color=_nbstyle.TEXTO)
ax2.set_ylabel("Revision (pp)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 5. La Prueba de Mankiw-Shapiro (1986) sobre un Proceso de Ruido Conocido
#
# ### Las dos hipótesis
# 1. **Noticia ($H_{\text{news}}$).** La agencia publica su mejor pronóstico de la cifra final dada la información $\Omega_t$ que tiene:
#    $$ y_{0, t} = \mathbb{E}[y_{T, t} \mid \Omega_t] \implies y_{T, t} = y_{0, t} + \nu_t, \quad \mathbb{E}[\nu_t \mid \Omega_t] = 0 $$
#    La revisión $r_t = \nu_t$ no se puede predecir con la primera publicación, así que en $r_t = \alpha_0 + \beta_0 y_{0,t} + \varepsilon_t$ se tiene $\beta_0 = 0$. Sí está correlacionada con la cifra final: $\beta_T = \text{Var}(\nu)/\text{Var}(y_T) > 0$.
#
# 2. **Ruido ($H_{\text{noise}}$).** La agencia observa la cifra final con un error de medición clásico:
#    $$ y_{0, t} = y_{T, t} + v_t, \quad \text{Cov}(y_{T, t}, v_t) = 0 $$
#    La revisión $r_t = -v_t$ no está correlacionada con la cifra final, así que $\beta_T = 0$. Sobre la primera publicación,
#    $$ \beta_0 = \frac{\text{Cov}(-v_t, \, y_{T, t} + v_t)}{\text{Var}(y_{0, t})} = \frac{-\sigma_v^2}{\sigma_{y_T}^2 + \sigma_v^2}, $$
#    que está en $(-1, 0)$ y vale $-1$ solo en el límite en que la varianza del ruido domina. Una pendiente de $-1$ no es la hipótesis nula de ruido.
#
# En nuestra simulación $\sigma_{y_T} = \sigma_* = 1.5$ y $\sigma_v = 0.6$, así que la pendiente poblacional es $\beta_0 = -0.36/2.61 \approx -0.14$ (la celda la imprime). Por eso la nula de ruido se prueba sobre la publicación **final**, $\beta_T = 0$, y la nula de noticia sobre la primera, $\beta_0 = 0$.
#
# `QNAVintagePanel.revision_stats` estima ambas regresiones mediante `puremacro.vintages.mankiw_shapiro`, con errores estándar robustos a heterocedasticidad (Newey-West con `hac_lags`) y valores $p$ de la $t$ de Student, y combina las dos pruebas al 5% en una etiqueta:
#
# | Pata de noticia ($\beta_0 = 0$) | Pata de ruido ($\beta_T = 0$) | Etiqueta |
# |---|---|---|
# | no se rechaza | se rechaza | "news" |
# | se rechaza | no se rechaza | "noise" |
# | se rechaza | se rechaza | "mixed" |
# | no se rechaza | no se rechaza | "indeterminate" |
#
# "Indeterminate" es un no rechazo de ambas nulas: la muestra no distingue noticia de ruido. No es evidencia a favor de ninguna. El diccionario también reporta la proporción de ruido $\max(0, -\hat\beta_0)$, una magnitud que se lee junto a los dos valores $p$.

# %%
beta_pop = -sd_noise**2 / (sd_true**2 + sd_noise**2)
stats_usa = panel.revision_stats("USA", "gdp_real")

print("==================================================================")
print("  MANKIW & SHAPIRO (1986) TEST, simulated 'USA gdp_real'")
print("==================================================================")
print(f"Known process:            pure noise, population slope on y_0 = {beta_pop:.4f}")
print(f"Sample size (N):          {stats_usa['n_obs']}")
print(f"Mean revision:            {stats_usa['mean_revision']:.4f} (p = {stats_usa['p_mean_revision']:.4f})")
print(f"Revision std. dev.:       {stats_usa['std_revision']:.4f}")
print("------------------------------------------------------------------")
print(f"News leg,  r on y_0:      beta_0 = {stats_usa['mankiw_shapiro_beta']:.4f} (s.e. {stats_usa['mankiw_shapiro_se']:.4f}, p = {stats_usa['mankiw_shapiro_pvalue']:.4f})")
print(f"Noise leg, r on y_T:      beta_T = {stats_usa['mankiw_shapiro_beta_final']:.4f} (s.e. {stats_usa['mankiw_shapiro_se_final']:.4f}, p = {stats_usa['mankiw_shapiro_pvalue_final']:.4f})")
print(f"Noise share:              {stats_usa['noise_share']:.4f} (population {-beta_pop:.4f})")
print(f"Label:                    {stats_usa['hypothesis']}")
print("==================================================================")

# Internal check: revision_stats runs the same regressions as mankiw_shapiro
ms_usa = mankiw_shapiro(s_first, s_latest)
assert abs(ms_usa.beta_on_preliminary - stats_usa["mankiw_shapiro_beta"]) < 1e-10
assert abs(ms_usa.p_beta_on_final - stats_usa["mankiw_shapiro_pvalue_final"]) < 1e-10

# %%
# Cross-country comparison: both legs and the label, from revision_stats
results_all = []
for c in countries:
    for v in variables:
        st = panel.revision_stats(c, v)
        results_all.append({
            "Country": c,
            "Variable": v,
            "N": st["n_obs"],
            "Mean rev.": round(st["mean_revision"], 4),
            "beta_0": round(st["mankiw_shapiro_beta"], 4),
            "p(beta_0=0)": round(st["mankiw_shapiro_pvalue"], 4),
            "beta_T": round(st["mankiw_shapiro_beta_final"], 4),
            "p(beta_T=0)": round(st["mankiw_shapiro_pvalue_final"], 4),
            "Label": st["hypothesis"],
        })

df_test_summary = pd.DataFrame(results_all)
print("Mankiw-Shapiro test on 8 simulated series (known process: pure noise)")
print(df_test_summary.to_string(index=False))
print(f"\nPopulation slopes: beta_0 = {beta_pop:.4f}, beta_T = 0")
print(f"Mean of the 8 estimates: beta_0 = {df_test_summary['beta_0'].mean():.4f}, beta_T = {df_test_summary['beta_T'].mean():.4f}")
print(f"Labels: {df_test_summary['Label'].value_counts().to_dict()}")

assert (df_test_summary["beta_0"] < 0).all(), "under classical noise every slope on y_0 should be negative here"
assert abs(df_test_summary["beta_0"].mean() - beta_pop) < 0.05, "the average slope should be near the population slope"
assert abs(df_test_summary["beta_T"].mean()) < 0.05, "the average slope on y_T should be near zero"
assert (df_test_summary["beta_0"] > -1).all(), "a slope of -1 is not what classical noise produces"

# %% [markdown]
# ## 6. El Diagrama de Dispersión de Mankiw-Shapiro
#
# Graficar la primera publicación $y_{0, t}$ contra la revisión $r_t = y_{T, t} - y_{0, t}$ muestra directamente la regresión de noticia:
#
# - Una recta horizontal ($\beta_0 = 0$) es lo que predice la hipótesis de **noticia**.
# - Con **ruido** clásico, la recta baja con el cociente de varianzas $\beta_0 = -\sigma_v^2 / (\sigma_{y_T}^2 + \sigma_v^2)$, aquí cerca de $-0.14$, no $-1$.
# - La recta de MCO es la estimación con esta muestra de 40 trimestres.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.8))

x_vals = s_first.values
y_vals = (s_latest - s_first).values
ax.scatter(x_vals, y_vals, color=_nbstyle.S1["color"], edgecolors=_nbstyle.SPINE, s=50, alpha=0.85, label="Quarters ($y_{0,t}, r_t$)")

x_grid = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 100)
y_fit = stats_usa["mankiw_shapiro_alpha"] + stats_usa["mankiw_shapiro_beta"] * x_grid
ax.plot(x_grid, y_fit, **_nbstyle.S2, label=f"OLS fit ($\\beta_0={stats_usa['mankiw_shapiro_beta']:.2f}$, p={stats_usa['mankiw_shapiro_pvalue']:.3f})")

# Population slope of the known noise process, drawn through the sample means
y_noise = y_vals.mean() + beta_pop * (x_grid - x_vals.mean())
ax.plot(x_grid, y_noise, color=_nbstyle.NOTA, lw=1.5, linestyle=":", label=f"Noise process, population slope ($\\beta_0={beta_pop:.2f}$)")

ax.axhline(0, color=_nbstyle.SPINE, lw=1.2, linestyle="--", label="News hypothesis ($\\beta_0=0$)")

ax.set_title("Mankiw & Shapiro (1986): Revision against First Release", fontsize=11, fontweight="bold")
ax.set_xlabel("First release $y_{0,t}$ (%)", color=_nbstyle.TEXTO)
ax.set_ylabel("Revision $y_{T,t} - y_{0,t}$ (pp)", color=_nbstyle.TEXTO)
ax.legend(loc="lower left", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE, fontsize=8)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 7. Conjuntos de Datos en un Punto del Tiempo (`.as_of()`)
#
# Los pronósticos pseudo fuera de muestra y las descomposiciones históricas deben usar solo lo que estaba publicado en su momento; usar datos revisados introduce información posterior en el pasado (sesgo de anticipación).
#
# `panel.as_of(date)` reconstruye el conjunto de datos disponible en una fecha de publicación. Para una edición $V^*$ conserva, para cada serie y trimestre de referencia, la última estimación publicada en o antes de $V^*$:
#
# $$ \mathcal{I}_{V^*} = \left\{ y_{v, t} \;\Big|\; v \le V^* \text{ and } v = \max_{u \le V^*} u \right\} $$

# %%
df_2018 = panel.as_of("2018-04-01")
print("Point-in-time panel as of 2018-04-01 (first 10 rows):")
print(df_2018.head(10).round(3).to_string())

df_2022 = panel.as_of("2022-01-01")
print(f"\nObservations available in the 2018 snapshot: {len(df_2018)}")
print(f"Observations available in the 2022 snapshot: {len(df_2022)}")

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** El panel es ruido puro por construcción, así que cada veredicto se puede calificar contra la verdad.
#
# 1. **La pendiente poblacional es pequeña.** Con $\sigma_* = 1.5$ y $\sigma_v = 0.6$ la pendiente de la revisión sobre la primera publicación es $-0.1379$, no $-1$. Es una forma cerrada derivada a mano para el proceso conocido, independiente de puremacro. Las ocho pendientes estimadas son todas negativas y promedian $-0.1471$; las pendientes sobre la última estimación promedian $-0.0108$, cerca de su valor poblacional de cero.
# 2. **Una serie, una muestra.** Para la serie simulada 'USA gdp_real' la pendiente sobre la primera publicación es $-0.0477$ (e.e. $0.0583$, $p = 0.4187$), así que esta muestra de 40 trimestres no puede rechazar la noticia aunque la verdad sea ruido. La pata de ruido tampoco rechaza ($p = 0.1487$), y `revision_stats` etiqueta la serie "indeterminate": la respuesta honesta, porque no se rechaza ninguna de las dos nulas. Su proporción de ruido estimada, $0.0477$, es un tercio del valor poblacional $0.1379$. Las mismas pendientes salen de `mankiw_shapiro` directamente (una verificación interna).
# 3. **Ocho series.** `revision_stats` acierta ("noise") en 4 de las 8 series y no puede decidir en 2 ('USA gdp_real' y 'DEU gdp_real'). Rechaza la nula verdadera de ruido en las dos series 'GBR' ($p = 0.0068$ y $p = 0.0487$), lo que da un "mixed" y un "news": al 5% se esperan esos rechazos falsos en más o menos una serie de cada veinte, y dos de ocho es mala suerte en este sorteo. Con una proporción de ruido de solo $0.1379$ y 40 trimestres, la prueba tiene poca potencia, y una etiqueta sobre una serie es evidencia débil. La revisión media de 'USA gdp_real' ($0.1286$, $p = 0.1232$) no es significativa, como corresponde: este proceso no tiene sesgo.
# 4. **Ninguna pendiente cerca de $-1$.** Todas las pendientes sobre la primera publicación están entre $-0.2431$ y $-0.0477$. Una regla que esperara una pendiente cercana a $-1$ para llamar ruido a una serie no llamaría ruido a ninguna de estas ocho; con ruido clásico esa pendiente exige que el error explique la mayor parte de la varianza de la primera publicación (indicación 2).
# 5. **Conjuntos de información.** La instantánea del 2018-04-01 tiene 128 observaciones frente a 160 en 2022: un análisis fechado en abril de 2018 debe usar el panel más pequeño y sin revisar.

# %% [markdown]
# ## Tu turno
#
# Cambie qué tan ruidosa es la primera publicación, o cambie el proceso de ruido a noticia, y **prediga ambas pendientes antes de ejecutar**: la pendiente de la revisión sobre la primera publicación ($\beta_0$) y sobre la última estimación ($\beta_T$). La celda reconstruye el panel con las mismas extracciones aleatorias, estima ambas pendientes para las ocho series y verifica que sus promedios estén a menos de tres errores estándar de sus predicciones. La predicción escrita en la celda es la fórmula poblacional; reemplácela con la suya. Poner $\beta_0 = -1$ para el proceso de ruido hace fallar la verificación.

# %%
# Your turn: predict both Mankiw-Shapiro slopes, then let the simulation grade you
sd_v_turn = 0.6        # ← change this: s.d. of the first-release error (noise) or of the news, 0.3 to 3.0
dgp_turn = "noise"     # ← change this: "noise" (y_0 = y* + v) or "news" (y_0 = y*, revisions add v)
assert 0.3 <= sd_v_turn <= 3.0 and dgp_turn in ("noise", "news")

share = sd_v_turn**2 / (sd_true**2 + sd_v_turn**2)
my_beta_0 = -share if dgp_turn == "noise" else 0.0   # slope of r on y_0: replace with your prediction
my_beta_T = 0.0 if dgp_turn == "noise" else share    # slope of r on y_T: replace with your prediction

panel_turn = simulate_vintages(sd_v_turn, dgp_turn)
b0, bT, labels = [], [], []
for c in countries:
    for v in variables:
        st = panel_turn.revision_stats(c, v)
        b0.append(st["mankiw_shapiro_beta"])
        bT.append(st["mankiw_shapiro_beta_final"])
        labels.append(st["hypothesis"])
b0, bT = np.array(b0), np.array(bT)
se0, seT = b0.std(ddof=1) / np.sqrt(len(b0)), bT.std(ddof=1) / np.sqrt(len(bT))

print(f"Process: {dgp_turn}, sd_v = {sd_v_turn}, noise share of Var(y_0 or y_T) = {share:.4f}")
print(f"  beta_0: mean {b0.mean():+.4f} (s.e. {se0:.4f}), your prediction {my_beta_0:+.4f}")
print(f"  beta_T: mean {bT.mean():+.4f} (s.e. {seT:.4f}), your prediction {my_beta_T:+.4f}")
print(f"  labels: {pd.Series(labels).value_counts().to_dict()}")

assert abs(b0.mean() - my_beta_0) < 3 * se0 + 0.01, "slope on the first release is far from your prediction"
assert abs(bT.mean() - my_beta_T) < 3 * seT + 0.01, "slope on the latest estimate is far from your prediction"

# %% [markdown]
# **Indicaciones.**
# 1. *Básico.* Con los valores por omisión, derive $\beta_0 = \text{Cov}(-v, y^* + v)/\text{Var}(y^* + v)$ como número antes de ejecutar. ¿Por qué está lejos de $-1$ aunque la primera publicación sea ruido puro? Puede ignorar el pequeño error que queda en las últimas ediciones.
# 2. *Intermedio.* Suponga una regla que llama "noise" a una serie solo cuando $|\hat\beta_0 + 1| < 0.3$. Invierta la fórmula de la pendiente para encontrar la desviación estándar del ruido que hace $\beta_0 = -0.8$, asígnela a `sd_v_turn` y vuelva a ejecutar. ¿Qué tan ruidosa debe ser la primera publicación para que esa regla pueda decir "noise", y cuántas de las ocho series llamaría ruido con los valores por omisión? ¿Qué dicen las etiquetas de `revision_stats` en ambos casos, y por qué la prueba de dos patas no necesita conocer la proporción de ruido?
# 3. *Avanzado.* Ponga `dgp_turn = "news"`. Prediga ambas pendientes y explique por qué comparar las dos regresiones distingue noticia de ruido cuando la regresión única sobre $y_0$ con un umbral de $-1$ no puede. Con 40 trimestres por serie, ¿cuántas de las ocho etiquetas son erróneas bajo cada proceso, y qué tipo de error es cada una (un rechazo falso o un no rechazo)? Vuelva a ejecutar con `panel_turn.revision_stats(c, v, hac_lags="auto")` dentro del bucle: ¿cambian los errores de Newey-West alguna etiqueta?
#
# ## ¿Qué tan exhaustivo es esto?
#
# - `puremacro.vintages`: `mankiw_shapiro` (ambas patas, con errores estándar HAC mediante `hac_lags`), `revision_test` sobre paneles largos de ediciones, y triángulos de revisión.
# - `puremacro.fetch.QNAVintagePanel` y `get_qna_vintage_catalog`: cortes en un punto del tiempo, primeras y últimas publicaciones, `revision_stats` (ambas patas de Mankiw-Shapiro y una etiqueta news/noise/mixed/indeterminate, mediante `mankiw_shapiro`), y el catálogo que usan los descargadores en tiempo real (que descargan datos y no se llaman aquí).
# - `puremacro.fetch.realtime.VintagePanel`: las mismas operaciones para paneles de bancos centrales latinoamericanos, incluido `news_or_noise` (cuadernos 58 y 59); el cuaderno 58 aplica la prueba de dos patas a un panel con pocos pares de revisión, y el cuaderno 48 aplica la nula de ruido corregida en un ejemplo de nowcasting.
