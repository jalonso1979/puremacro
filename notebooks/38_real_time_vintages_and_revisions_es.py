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
# # Datos en Tiempo Real, Vendimias Históricas y el Test de Mankiw-Shapiro
#
# **¿Cómo afectan las revisiones estadísticas al análisis macroeconómico y cómo podemos evaluar si las publicaciones iniciales reflejan pronósticos racionales ("News") o errores de medición ("Noise")?**
#
# Los agregados macroeconómicos como el Producto Interno Bruto (PIB), el consumo y la inversión son revisados periódicamente por los institutos de estadística conforme se incorporan fuentes censales, registros administrativos y actualizaciones metodológicas:
#
# 1. **Información en Tiempo Real y Errores de Política**:
#    Athanasios Orphanides (2001, *American Economic Review*) demostró que los errores de política monetaria en los años setenta se debieron a estimaciones de la brecha del producto sobre datos en tiempo real que sufrieron revisiones masivas posteriores. Croushore & Stark (2001, *Journal of Economic Literature*) formalizaron la arquitectura de datos en tiempo real.
#
# 2. **El Test Econométrico de Mankiw & Shapiro (1986)**:
#    Sea $y_{0,t}$ la estimación inicial (avance) del crecimiento del PIB en el trimestre $t$, y $y_{T,t}$ la serie final revisada. La revisión total es:
#    $$ r_t = y_{T,t} - y_{0,t} $$
#    Mankiw & Shapiro (1986, *Journal of Business & Economic Statistics*) formulan la regresión MCO:
#    $$ r_t = \alpha + \beta \, y_{0,t} + \varepsilon_t $$
#
#    - **Hipótesis de Noticias (News, $\beta = 0$, $R^2 \approx 0$)**: La publicación inicial es un pronóstico condicional óptimo ($y_{0,t} = \mathbb{E}[y_{T,t} \mid \Omega_t]$). La revisión $r_t$ representa información estadística nueva y no correlacionada con $y_{0,t}$.
#    - **Hipótesis de Ruido (Noise, $\beta = -1$, $\alpha = 0$)**: La publicación inicial contiene error de medición clásico ($y_{0,t} = y_{T,t} + v_t$, con $\text{Cov}(y_{T,t}, v_t) = 0$). Por lo tanto, $r_t = -v_t$ tiene correlación negativa con $y_{0,t}$.
#
# En este cuaderno, exploramos las vendimias de Cuentas Nacionales Trimestrales (QNA) para más de 45 países, construimos triángulos de revisión $(T \times V)$, generamos cortes en tiempo real y ejecutamos el test de Mankiw-Shapiro con `puremacro.fetch` y `puremacro.vintages`.

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

# %% [markdown]
# ## 1. Inspección del Catálogo de Vendimias QNA (45+ Países)
#
# La investigación macroeconómica en tiempo real exige contar con la información estadística tal como estaba disponible para las autoridades y los analistas al momento de adoptar decisiones. En bases de datos de archivo como ALFRED (ArchivaL Federal Reserve Economic Data) de la Reserva Federal de St. Louis, cada indicador macroeconómico se almacena junto con su fecha de publicación o vendimia.
#
# `puremacro` ofrece un catálogo estandarizado que vincula a más de 45 economías (los 38 miembros de la OCDE, el G7, el G20 y mercados emergentes clave) con las vendimias históricas disponibles en ALFRED. Cada registro incluye el código del país, su región, la definición homogénea de la variable (PIB real, formación bruta de capital fijo o consumo privado) y el identificador oficial de la serie en ALFRED.

# %%
catalog = get_qna_vintage_catalog()
print(f"Total de series en el catálogo: {len(catalog)}")
print("\nMuestra de Entradas del Catálogo:")
print(catalog[["country", "country_name", "region", "variable", "series_id"]].head(10))

# Resumen por variable
var_counts = catalog.groupby("variable")["country"].count().reset_index()
var_counts.columns = ["Variable", "Número de Países"]
print("\nCobertura de Variables:")
print(var_counts)

# %% [markdown]
# ## 2. Construcción de un Panel de Vendimias en Tiempo Real
#
# Para ilustrar el pipeline analítico en tiempo real, simulamos una estructura histórica de revisiones para 4 países a lo largo de 40 trimestres ($T=40$) y 50 fechas de publicación o vendimias ($V=50$).
#
# Sea $y_t^*$ la verdadera tasa subyacente de crecimiento económico en el trimestre $t$. La publicación inicial o avance $y_{0, t}$ se difunde con un trimestre de rezago ($t+1$) y contiene tanto información del estado latente como un error inicial de medición por muestreo $v_t \sim \mathcal{N}(0, \sigma_v^2)$:
#
# $$ y_{0, t} = y_t^* + v_t $$
#
# Conforme transcurren las vendimias posteriores $j \ge 1$, las agencias estadísticas van incorporando información censal, registros fiscales y factores de ajuste estacional actualizados. La estimación en la vendimia $y_{j, t}$ converge paulatinamente hacia el benchmark definitivo mediante una tasa de decaimiento geométrico:
#
# $$ y_{j, t} = y_t^* + e^{-j / \kappa} \, v_t, \qquad \kappa > 0 $$

# %%
rng = np.random.default_rng(1986)
n_quarters = 40
obs_dates = pd.date_range("2010-01-01", periods=n_quarters, freq="QS")
vintage_dates = pd.date_range("2010-04-01", periods=50, freq="QS")

countries = ["USA", "DEU", "GBR", "MEX"]
variables = ["gdp_real", "gfcf_real"]

records = []
for c in countries:
    for var in variables:
        true_growth = rng.normal(loc=2.2, scale=1.5, size=n_quarters)
        noise = rng.normal(loc=0.0, scale=0.6, size=n_quarters)
        y_0 = true_growth + noise
        
        for i, obs in enumerate(obs_dates):
            first_pub_idx = i + 1
            for j, vint in enumerate(vintage_dates):
                if j >= first_pub_idx:
                    lag = j - first_pub_idx
                    decay = np.exp(-lag / 3.0)
                    val = true_growth[i] + decay * noise[i]
                    records.append({
                        "country": c,
                        "variable": var,
                        "date": obs,
                        "vintage": vint,
                        "value": float(val),
                    })

df_raw = pd.DataFrame(records)
panel = QNAVintagePanel(df=df_raw)
print(f"Panel QNAVintagePanel creado con {len(df_raw):,} registros en {len(countries)} países.")

# %% [markdown]
# ## 3. Visualización del Triángulo de Revisión $(T \times V)$
#
# La estructura canónica de datos de vendimia se organiza en una matriz triangular inferior de revisiones $\mathbf{R} \in \mathbb{R}^{T \times V}$:
#
# $$ \mathbf{R} = \begin{bmatrix} y_{1, v_1} & y_{1, v_2} & y_{1, v_3} & \dots & y_{1, v_V} \\ \text{NaN} & y_{2, v_2} & y_{2, v_3} & \dots & y_{2, v_V} \\ \text{NaN} & \text{NaN} & y_{3, v_3} & \dots & y_{3, v_V} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \text{NaN} & \text{NaN} & \text{NaN} & \dots & y_{T, v_V} \end{bmatrix} $$
#
# Cada fila representa el periodo de referencia u observación $t$ (trimestre), mientras que cada columna corresponde a la fecha de la vendimia publicada $v$. La diagonal principal recoge la primera estimación (avance) para cada periodo. Al desplazarse horizontalmente a lo largo de una fila, se traza la trayectoria histórica de revisiones que experimenta la cifra de un trimestre particular conforme la agencia estadística refina sus cálculos.

# %%
tri_mex = panel.revision_matrix("MEX", "gdp_real")
print("Matriz de Revisión del PIB Real de México (primeros 6 trimestres × 6 vendimias):")
print(tri_mex.iloc[:6, :6])

# %%
fig, ax = _nbstyle.figura(figsize=(9.2, 5.2))

im = ax.imshow(tri_mex.iloc[:24, :24].to_numpy(), cmap=_nbstyle.CMAP_SEQ, aspect="auto")
ax.set_title("México PIB Real: Triángulo Histórico de Revisión (T × V)", fontsize=11, fontweight="bold")
ax.set_xlabel("Índice de Fecha de Publicación (Vendimia)", color=_nbstyle.TEXTO)
ax.set_ylabel("Índice del Trimestre de Observación", color=_nbstyle.TEXTO)

ax.set_xticks(range(0, 24, 4))
ax.set_xticklabels([d.strftime("%YQ%q") for d in tri_mex.columns[:24:4]], rotation=45)
ax.set_yticks(range(0, 24, 4))
ax.set_yticklabels([d.strftime("%YQ%q") for d in tri_mex.index[:24:4]])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Crecimiento Anualizado del PIB Real (%)", color=_nbstyle.TEXTO)

# %% [markdown]
# ## 4. Primera Estimación (Avance) vs. Última Revisión
#
# La comparación entre el avance inicial y el benchmark final disponible revela la magnitud, el signo y la persistencia cíclica de las revisiones macroeconómicas:
#
# $$ r_t = y_{T, t} - y_{0, t} $$
#
# Si el promedio de las revisiones difiere sistemáticamente de cero ($\bar{r} \neq 0$), la estimación preliminar exhibe sesgo estadístico. Asimismo, si las revisiones muestran correlación procíclica o anticíclica con las fluctuaciones económicas, las decisiones de política calibradas con cifras iniciales pueden inducir perturbaciones desestabilizadoras involuntarias (Orphanides, 2001).

# %%
s_first = panel.first_release("MEX", "gdp_real")
s_latest = panel.latest_release("MEX", "gdp_real")

fig, (ax1, ax2) = _nbstyle.figura(2, 1, figsize=(9.5, 6.0), sharex=True)

ax1.plot(s_first.index, s_first.values, **_nbstyle.S1, label="Primera Estimación (Avance)")
ax1.plot(s_latest.index, s_latest.values, **_nbstyle.S2, label="Último Benchmark Revisado")
ax1.set_title("México PIB Real: Serie Inicial vs. Serie Revisada", fontsize=11, fontweight="bold")
ax1.set_ylabel("Tasa de Crecimiento (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

revision = s_latest - s_first
ax2.bar(revision.index, revision.values, color=_nbstyle.S3["color"], width=60, edgecolor=_nbstyle.SPINE, alpha=0.8, label="Revisión ($y_T - y_0$)")
ax2.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="-")
ax2.set_title("Serie Histórica de Revisiones", fontsize=11, fontweight="bold")
ax2.set_xlabel("Fecha de Observación", color=_nbstyle.TEXTO)
ax2.set_ylabel("Revisión (puntos porcentuales)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 5. Ejecución del Test de Mankiw-Shapiro (1986)
#
# Gregory Mankiw y Matthew Shapiro (1986) formularon el marco econométrico fundamental para contrastar la racionalidad y eficiencia de las cifras estadísticas preliminares.
#
# Estimamos la regresión por Mínimos Cuadrados Ordinarios (MCO) de la revisión $r_t = y_{T, t} - y_{0, t}$ respecto a la estimación inicial $y_{0, t}$:
#
# $$ r_t = \alpha + \beta \, y_{0, t} + \varepsilon_t $$
#
# ### Hipótesis Teóricas
# 1. **Hipótesis de Noticias ($H_{\text{news}}$)**:
#    Los institutos estadísticos elaboran pronósticos racionales del valor final condicionando en todo el conjunto de información $\Omega_t$ disponible en ese momento:
#    $$ y_{0, t} = \mathbb{E}[y_{T, t} \mid \Omega_t] \implies y_{T, t} = y_{0, t} + \nu_t, \quad \text{donde } \mathbb{E}[\nu_t \mid \Omega_t] = 0 $$
#    Bajo expectativas racionales, la revisión $r_t = \nu_t$ constituye una *noticia pura* completamente ortogonal e impredecible a partir del dato preliminar:
#    $$ \alpha = 0, \qquad \beta = 0, \qquad R^2 \approx 0 $$
#
# 2. **Hipótesis de Ruido ($H_{\text{noise}}$)**:
#    El instituto de estadística observa el verdadero benchmark contaminado por un error de medición clásico $u_t$:
#    $$ y_{0, t} = y_{T, t} + u_t, \quad \text{con } \text{Cov}(y_{T, t}, u_t) = 0, \quad u_t \sim \text{i.i.d.}(0, \sigma_u^2) $$
#    En tal situación, la revisión es el negativo exacto del ruido de medición ($r_t = -u_t$). Al estimar la regresión de $r_t$ sobre $y_{0, t}$, el estimador de la pendiente resulta:
#    $$ \beta = \frac{\text{Cov}(-u_t, y_{T, t} + u_t)}{\text{Var}(y_{0, t})} = \frac{-\sigma_u^2}{\sigma_y^2 + \sigma_u^2} < 0 $$
#    Cuando el error de medición predomina, $\beta \to -1$.

# %%
stats_mex = panel.revision_stats("MEX", "gdp_real")
print("==================================================================")
print("  TEST ECONOMÉTRICO DE NOTICIAS VS RUIDO DE MANKIW Y SHAPIRO (1986)")
print("==================================================================")
print(f"País:                  México (MEX)")
print(f"Variable:              PIB Real (gdp_real)")
print(f"Tamaño Muestral (N):   {stats_mex['n_obs']}")
print(f"Revisión Media (alpha):{stats_mex['mean_revision']:.4f}")
print(f"Revisión Abs. Media:   {stats_mex['abs_mean_revision']:.4f}")
print(f"Desv. Est. Revisión:   {stats_mex['std_revision']:.4f}")
print("------------------------------------------------------------------")
print(f"Constante MCO (alpha): {stats_mex['mankiw_shapiro_alpha']:.4f}")
print(f"Pendiente MCO (beta):  {stats_mex['mankiw_shapiro_beta']:.4f}")
print(f"Error Estándar (SE):   {stats_mex['mankiw_shapiro_se']:.4f}")
print(f"Estadístico t:         {stats_mex['mankiw_shapiro_tstat']:.4f}")
print(f"Valor p:               {stats_mex['mankiw_shapiro_pvalue']:.4f}")
print(f"Conclusión:            {stats_mex['hypothesis']}")
print("==================================================================")

# %%
# Resumen para todos los países
results_all = []
for c in countries:
    for v in variables:
        st = panel.revision_stats(c, v)
        results_all.append({
            "País": c,
            "Variable": v,
            "N": st["n_obs"],
            "Revisión Media": st["mean_revision"],
            "Desv. Est.": st["std_revision"],
            "Beta": st["mankiw_shapiro_beta"],
            "t-stat": st["mankiw_shapiro_tstat"],
            "p-value": st["mankiw_shapiro_pvalue"],
            "Hipótesis": st["hypothesis"],
        })

df_test_summary = pd.DataFrame(results_all)
print("\nTabla Resumen del Test de Mankiw-Shapiro por País:")
print(df_test_summary.to_string(index=False))

# %% [markdown]
# ## 6. Gráfico Diagnóstico de la Regresión de Mankiw-Shapiro
#
# La representación gráfica de la estimación inicial $y_{0, t}$ en el eje horizontal frente a la revisión $r_t = y_{T, t} - y_{0, t}$ en el eje vertical aporta una perspectiva geométrica reveladora:
#
# - Una línea horizontal ($\beta = 0$) define la **Referencia de Noticias Puras**, señalando estimaciones eficientes y no sesgadas.
# - Una recta decreciente con pendiente $\beta = -1$ define la **Referencia de Ruido Puro**, revelando que el avance preliminar contiene puro error de medición.
# - La pendiente estimada por MCO evidencia si las revisiones compensan excesiva o insuficientemente las lecturas extremas iniciales.

# %%
fig, ax = _nbstyle.figura(figsize=(8.5, 4.8))

x_vals = s_first.values
y_vals = (s_latest - s_first).values
ax.scatter(x_vals, y_vals, color=_nbstyle.S1["color"], edgecolors=_nbstyle.SPINE, s=50, alpha=0.85, label="Observaciones ($y_{0,t}, r_t$)")

x_grid = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 100)
y_fit = stats_mex["mean_revision"] + stats_mex["mankiw_shapiro_beta"] * x_grid
ax.plot(x_grid, y_fit, **_nbstyle.S2, label=f"Ajuste MCO ($\\beta={stats_mex['mankiw_shapiro_beta']:.2f}$, p={stats_mex['mankiw_shapiro_pvalue']:.3f})")

y_noise = -1.0 * x_grid
ax.plot(x_grid, y_noise, color=_nbstyle.NOTA, lw=1.5, linestyle=":", label="Referencia de Ruido Puro ($\\beta=-1$)")
ax.axhline(0, color=_nbstyle.SPINE, lw=1.2, linestyle="--", label="Referencia de Noticias Puras ($\\beta=0$)")

ax.set_title("Gráfico Diagnóstico de Noticias vs. Ruido (Mankiw & Shapiro)", fontsize=11, fontweight="bold")
ax.set_xlabel("Estimación Inicial $y_{0,t}$ (%)", color=_nbstyle.TEXTO)
ax.set_ylabel("Revisión Total $y_{T,t} - y_{0,t}$ (puntos %)", color=_nbstyle.TEXTO)
ax.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 7. Cortes en Tiempo Real del Panel (`.as_of()`)
#
# Al efectuar ejercicios de pronóstico pseudo fuera de muestra o al estimar la descomposición histórica de modelos SVAR, el empleo de series revisadas contemporáneamente genera sesgo de anticipación o fuga de información futura.
#
# La función `panel.as_of(date)` reconstruye de manera fidedigna la información de corte transversal y de series temporales que estaba disponible para el investigador en una fecha de publicación determinada. Para cualquier vendimia histórica $V^*$, se seleccionan todas las observaciones que satisfacen:
#
# $$ \mathcal{I}_{V^*} = \left\{ y_{v, t} \;\Big|\; v \le V^* \text{ y } v = \max_{u \le V^*} u \right\} $$

# %%
df_2018 = panel.as_of("2018-04-01")
print("Panel de Corte en Tiempo Real al 2018-04-01:")
print(df_2018.head(10))

# Slicing as of 2022-01-01
df_2022 = panel.as_of("2022-01-01")
print(f"\nObservaciones disponibles en el corte 2018: {len(df_2018)}")
print(f"Observaciones disponibles en el corte 2022: {len(df_2022)}")
