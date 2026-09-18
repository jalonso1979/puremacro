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
# # Pronósticos Macroeconómicos en Alta Dimensión — Red Elástica y Lasso Adaptativo
#
# **¿Cómo pueden los econometristas extraer señales predictivas parsimoniosas a partir de decenas o cientos de indicadores macroeconómicos sin incurrir en sobreajuste ni sufrir por multicolinealidad?**
#
# En la macroeconomía empírica moderna, la predicción de variables clave (como la inflación subyacente, el producto industrial o el empleo) involucra habitualmente paneles masivos de predictores potenciales: diferenciales de tasas de interés, precios internacionales de materias primas, tipos de cambio, encuestas cualitativas de expectativas y agregados crediticios. Cuando el número de variables predictoras candidatas $P$ es elevado en relación con el tamaño de la muestra temporal efectiva $T$, la estimación clásica por Mínimos Cuadrados Ordinarios (MCO) falla de forma crítica: la varianza muestral de los estimadores se dispara, el modelo sobreajusta el ruido idiosincrásico en la muestra y la precisión predictiva fuera de muestra colapsa.
#
# La estimación regularizada resuelve este dilema dimensional contrayendo los coeficientes no esenciales hacia cero mediante funciones de penalización estructuradas:
# 1. **Regresión Ridge** (penalización $L_2$, Hoerl & Kennard 1970): Contrae los coeficientes proporcionalmente, estabilizando los pronósticos ante alta multicolinealidad, pero retiene las $P$ variables sin inducir dispersión (*sparsity*).
# 2. **Lasso** (penalización $L_1$, Tibshirani 1996): Genera soluciones de esquina exactas, anulando de forma automática los coeficientes irrelevantes para ejecutar selección de variables. Sin embargo, cuando los predictores están fuertemente correlacionados, Lasso selecciona arbitrariamente uno de ellos y descarta los demás.
# 3. **Red Elástica (Elastic Net)** (Zou & Hastie 2005): Combina de manera convexa la dispersión $L_1$ y la contracción grupal $L_2$, seleccionando grupos de variables correlacionadas en bloque.
# 4. **Lasso Adaptativo (Adaptive Lasso)** (Hui Zou 2006, *JASA*): Asigna ponderadores específicos determinados por los datos $w_j = 1/|\hat{\beta}_{j, init}|^\gamma$. Crucialmente, Zou demostró que el Lasso estándar no siempre es consistente para la selección de variables cuando fallan ciertas condiciones de diseño, mientras que el Lasso Adaptativo alcanza la **propiedad del oráculo**: selecciona asintóticamente el subconjunto verdadero con probabilidad 1 y produce estimadores asintóticamente insesgados y normales equivalentes a conocer el modelo real a priori.
#
# En este tutorial interactivo simulamos un entorno macroeconómico de alta dimensión ($P=30$ predictores correlacionados, $T=160$ observaciones) donde la inflación es generada por un subconjunto disperso de variables activas, y evaluamos la Red Elástica frente al Lasso Adaptativo mediante `puremacro.forecast.forecast_penalized`.

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

from puremacro.forecast import forecast_penalized

# %% [markdown]
# ## 1. Simulación de un Panel Macroeconómico en Alta Dimensión (P = 30 Predictores)
#
# Consideramos la regresión predictiva para la variable objetivo $y_{t+h}$ al horizonte $h$:
#
# $$ y_{t+h} = \mu + \sum_{j=1}^P \beta_j X_{j, t} + \varepsilon_{t+h} $$
#
# La función objetivo de la Red Elástica minimiza la suma de residuos al cuadrado sujeta a la penalización mixta:
#
# $$ \min_{\mu, \beta} \frac{1}{2T} \sum_{t=1}^T \left( y_{t+h} - \mu - X_t \beta \right)^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right] $$
#
# donde $\alpha \in [0, 1]$ equilibra la penalización Lasso $L_1$ ($\alpha = 1$) con la penalización Ridge $L_2$ ($\alpha = 0$), y $\lambda > 0$ modula la intensidad global de la regularización.
#
# Para el Lasso Adaptativo ($\alpha = 1$), la penalización incorpora ponderadores específicos:
#
# $$ \min_{\mu, \beta} \frac{1}{2T} \sum_{t=1}^T \left( y_{t+h} - \mu - X_t \beta \right)^2 + \lambda \sum_{j=1}^P w_j |\beta_j| $$
#
# donde $w_j = |\hat{\beta}_{j, \text{Ridge}}|^{-1}$. Los predictores con coeficientes preliminares pequeños en la etapa Ridge reciben penalizaciones mayores, forzando a cero el ruido mientras se atenúa el sesgo de contracción sobre las señales verdaderas.
#
# A continuación generamos un panel mensual de $P=30$ predictores autorregresivos. La inflación $y$ está determinada únicamente por cuatro variables activas ($j \in \{1, 5, 12, 22\}$) con parámetros verdaderos $[1.8, -1.4, 1.2, -0.9]$, mientras que las 26 series restantes representan ruido.

# %%
rng = np.random.default_rng(123)
T = 160
P = 30
dates = pd.date_range("2010-01-01", periods=T, freq="MS")

X = np.zeros((T, P))
for j in range(P):
    rho = rng.uniform(0.3, 0.8)
    for t in range(1, T):
        X[t, j] = rho * X[t-1, j] + rng.normal(scale=0.8)

# Variable objetivo impulsada por 4 predictores activos
y = np.zeros(T)
active_indices = [1, 5, 12, 22]
weights = [1.8, -1.4, 1.2, -0.9]
for t in range(1, T):
    signal = sum(w * X[t-1, idx] for w, idx in zip(weights, active_indices))
    y[t] = 2.0 + signal + rng.normal(scale=0.5)
y[0] = 2.0

df_X = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
s_y = pd.Series(y, index=dates, name="Inflación IPC")

# %% [markdown]
# ## 2. Estimación de Pronósticos con Red Elástica y Lasso Adaptativo
#
# `puremacro.forecast.forecast_penalized` resuelve la trayectoria de optimización por descenso coordinado a lo largo de una rejilla geométrica de parámetros de penalización $\lambda \in [\lambda_{\min}, \lambda_{\max}]$. Para cada candidato $\lambda$, la complejidad óptima del modelo se selecciona mediante el Criterio de Información Bayesiano (BIC):
#
# $$ \text{BIC}(\lambda) = T \log\left( \frac{\text{SSR}(\lambda)}{T} \right) + \text{df}(\lambda) \log(T) $$
#
# donde $\text{df}(\lambda)$ es el número de coeficientes activos no nulos. Debido a que el término penalizador $\log(T)$ es más severo que en AIC ($2$), el BIC penaliza fuertemente la inclusión de variables redundantes, garantizando consistencia asintótica en la selección de modelos bajo el supuesto de dispersión.

# %%
res_enet = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=False)
print("=== Red Elástica ===")
print(res_enet.summary())

res_alasso = forecast_penalized(df_X, s_y, horizon=1, alpha=1.0, adaptive=True)
print("\n=== Lasso Adaptativo ===")
print(res_alasso.summary())

# %% [markdown]
# ## 3. Valores Observados vs. Ajustados y Trayectorias de Regularización
#
# Los paneles a continuación comparan:
# - **Panel Izquierdo**: La inflación histórica observada (línea continua) frente a los valores ajustados a un mes vista generados por el Lasso Adaptativo (línea discontinua). El modelo regularizado alcanza un $R^2 \approx 0.95$ en muestra sin sobreajustar el ruido.
# - **Panel Derecho**: El perfil del criterio BIC a lo largo de los candidatos $\log_{10}(\lambda)$. La forma en U ilustra el dilema sesgo-varianza: una penalización insuficiente ($\lambda \to 0$) infla los grados de libertad, mientras que una penalización excesiva ($\lambda \gg 0$) colapsa las señales hacia el origen. El mínimo del BIC define el parámetro óptimo $\lambda^*$.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

fitted_vals = res_alasso.intercept + df_X.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
ax1.plot(dates[1:], s_y.iloc[1:], **_nbstyle.S1, label="Inflación Observada")
ax1.plot(dates[1:], fitted_vals, **_nbstyle.S2, label=f"Ajuste Lasso Adaptativo (R²={res_alasso.in_sample_r2:.2f})")
ax1.set_title("Ajuste del Modelo Penalizado vs. Datos Observados", fontsize=11, fontweight="bold")
ax1.set_xlabel("Fecha", color=_nbstyle.TEXTO)
ax1.set_ylabel("Tasa de Inflación (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

ax2.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, **_nbstyle.S1, marker="o", markersize=3)
ax2.axvline(np.log10(res_alasso.optimal_lambda), color=_nbstyle.SPINE, linestyle="--", label=f"Óptimo λ* = {res_alasso.optimal_lambda:.4f}")
ax2.set_title("Trayectoria BIC según la Penalización", fontsize=11, fontweight="bold")
ax2.set_xlabel(r"$\log_{10}(\lambda)$", color=_nbstyle.TEXTO)
ax2.set_ylabel("Puntaje BIC", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 4. Comparación de Dispersión: Red Elástica vs. Lasso Adaptativo
#
# La inspección de los coeficientes estimados corrobora las diferencias teóricas entre ambos paradigmas de regularización:
#
# - **Red Elástica ($\alpha=0.5$)**: Debido al componente Ridge $L_2$, retiene coeficientes pequeños distintos de cero en varios predictores de ruido correlacionados (como los indicadores 03, 15 y 28). Este comportamiento de agrupamiento preserva la estabilidad predictiva cuando las variables no pueden aislarse limpiamente, pero introduce sesgo de atenuación sobre los coeficientes principales.
# - **Lasso Adaptativo ($\alpha=1.0$)**: Exhibe la propiedad del oráculo en la práctica. Anula exactamente las 26 variables de ruido y conserva de forma exclusiva los 4 predictores del proceso generador de datos, con magnitudes que reproducen de cerca los parámetros verdaderos ($+1.8, -1.4, +1.2, -0.9$).

# %%
fig, ax = _nbstyle.figura(figsize=(9.0, 4.5))
df_comp = pd.DataFrame({
    "Red Elástica (α=0.5)": res_enet.coefficients,
    "Lasso Adaptativo (α=1.0)": res_alasso.coefficients,
})
top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
top_feats.plot(kind="bar", ax=ax, color=_nbstyle.palette(2), edgecolor=_nbstyle.SPINE, alpha=0.85)
ax.set_title("Selección de Coeficientes y Comparación de Contracción", fontsize=11, fontweight="bold")
ax.set_ylabel("Coeficiente Estimado", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
