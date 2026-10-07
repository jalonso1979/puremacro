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
# 4. **Lasso Adaptativo** (Hui Zou 2006, *JASA*): Asigna pesos específicos por predictor $w_j = 1/|\hat{\beta}_{j, init}|^\gamma$. Bajo condiciones adecuadas del diseño, del estimador inicial y de la penalización, alcanza una **propiedad del oráculo**: la selección y la inferencia asintóticas se comportan como si conociéramos las variables activas. Esto no garantiza selección exacta ni coeficientes insesgados en muestras finitas.
#
# Simulamos $P=30$ indicadores persistentes en el tiempo con innovaciones independientes
# durante $T=160$ períodos. Cuatro indicadores generan la inflación. Estimamos Red Elástica
# y Lasso Adaptativo mediante `puremacro.forecast.forecast_penalized` con los primeros
# 120 períodos y evaluamos pronósticos a un mes en los últimos 40 períodos.

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
# La implementación usa $w_j \propto (|\hat{\beta}_{j,\text{Ridge}}|+10^{-3})^{-1}$,
# normalizados por su mediana. Los predictores con coeficientes Ridge preliminares pequeños
# reciben penalizaciones mayores.
#
# **Intuición.** La contracción acepta algo de sesgo a cambio de menor varianza de estimación.
# Los pesos adaptativos penalizan más las señales inicialmente débiles. Un buen ajuste o un
# modelo parsimonioso no demuestran habilidad predictiva: la muestra reservada la evalúa aparte.
#
# Los cuatro índices de Python activos son `[1, 5, 12, 22]` (base cero), correspondientes
# a las etiquetas 02, 06, 13 y 23, con coeficientes `[1.8, -1.4, 1.2, -0.9]`.
# Los otros 26 indicadores tienen coeficientes poblacionales iguales a cero.

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

# Target variable driven by 4 key predictors
y = np.zeros(T)
active_indices = [1, 5, 12, 22]
weights = [1.8, -1.4, 1.2, -0.9]
for t in range(1, T):
    signal = sum(w * X[t-1, idx] for w, idx in zip(weights, active_indices))
    y[t] = 2.0 + signal + rng.normal(scale=0.5)
y[0] = 2.0

df_X = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
s_y = pd.Series(y, index=dates, name="CPI Inflation")

# %% [markdown]
# ## 2. Estimación de Pronósticos con Red Elástica y Lasso Adaptativo
#
# `puremacro.forecast.forecast_penalized` resuelve la trayectoria de optimización por descenso coordinado a lo largo de una rejilla geométrica de parámetros de penalización $\lambda \in [\lambda_{\min}, \lambda_{\max}]$. Para cada candidato $\lambda$, la complejidad óptima del modelo se selecciona mediante el Criterio de Información Bayesiano (BIC):
#
# $$ \text{BIC}(\lambda) = T \log\left( \frac{\text{SSR}(\lambda)}{T} \right) + \text{df}(\lambda) \log(T) $$
#
# Aquí $T$ cuenta los pares de entrenamiento alineados y $\text{df}(\lambda)$ cuenta
# los coeficientes activos más el intercepto. BIC elige una penalización dentro de la rejilla;
# la parsimonia por sí sola no garantiza selección consistente. La estandarización, los pesos
# adaptativos y BIC usan exclusivamente la muestra de entrenamiento.

# %%
TRAIN_END = 120
X_train, y_train = df_X.iloc[:TRAIN_END], s_y.iloc[:TRAIN_END]
assert X_train.index.equals(y_train.index)  # this API aligns inputs by position
res_enet = forecast_penalized(X_train, y_train, horizon=1, alpha=0.5, adaptive=False)
print("=== Elastic Net ===")
print(res_enet.summary())

res_alasso = forecast_penalized(X_train, y_train, horizon=1, alpha=1.0, adaptive=True)
print("\n=== Adaptive Lasso ===")
print(res_alasso.summary())

# %% [markdown]
# ## 3. Valores Observados vs. Ajustados y Trayectorias de Regularización
#
# Los paneles a continuación comparan:
# - **Panel izquierdo**: Inflación observada y ajustada en entrenamiento. El $R^2$ impreso
# mide ajuste dentro de muestra; estos valores no son pronósticos fuera de muestra.
# - **Panel derecho**: BIC de entrenamiento en la rejilla de penalizaciones. El mínimo
# equilibra ajuste residual y tamaño del modelo; la trayectoria no tiene que ser una U suave.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

fitted_vals = res_alasso.intercept + X_train.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
ax1.plot(dates[1:TRAIN_END], y_train.iloc[1:], **_nbstyle.S1, label="Actual Inflation")
ax1.plot(dates[1:TRAIN_END], fitted_vals, **_nbstyle.S2, label=f"Adaptive Lasso Fit (R²={res_alasso.in_sample_r2:.2f})")
ax1.set_title("Actual vs. Penalized Model Fitted Path", fontsize=11, fontweight="bold")
ax1.set_xlabel("Date", color=_nbstyle.TEXTO)
ax1.set_ylabel("Inflation Rate (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

ax2.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, **_nbstyle.S1, marker="o", markersize=3)
ax2.axvline(np.log10(res_alasso.optimal_lambda), color=_nbstyle.SPINE, linestyle="--", label=f"Optimal λ* = {res_alasso.optimal_lambda:.4f}")
ax2.set_title("BIC Regularisation Path Across Candidate Penalties", fontsize=11, fontweight="bold")
ax2.set_xlabel(r"$\log_{10}(\lambda)$", color=_nbstyle.TEXTO)
ax2.set_ylabel("BIC Score", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 4. Comparación de Dispersión: Red Elástica vs. Lasso Adaptativo
#
# Comparamos los coeficientes estimados con la verdad conocida de la simulación. La tabla
# cuenta señales recuperadas y predictores de ruido seleccionados usando `selected_features`.
# Evalúa una muestra finita, no la propiedad asintótica del oráculo. La gráfica muestra
# coeficientes mayores que 0.05 en valor absoluto en algún ajuste, junto con todas las señales
# verdaderas; los coeficientes seleccionados más pequeños sí se cuentan en la tabla.

# %%
true_coefficients = pd.Series(0.0, index=df_X.columns)
true_coefficients.iloc[active_indices] = weights
true_features = set(true_coefficients[true_coefficients != 0].index)
selection_rows = []
for name, result in [("Elastic Net", res_enet), ("Adaptive Lasso", res_alasso)]:
    selected = set(result.selected_features)
    selection_rows.append({"Model": name, "Signals recovered": len(selected & true_features),
                           "Noise selected": len(selected - true_features)})
print(pd.DataFrame(selection_rows).to_string(index=False))

fig, ax = _nbstyle.figura(figsize=(9.0, 4.5))
df_comp = pd.DataFrame({
    "True coefficient": true_coefficients,
    "Elastic Net (α=0.5)": res_enet.coefficients,
    "Adaptive Lasso (α=1.0)": res_alasso.coefficients,
})
top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
top_feats.plot(kind="bar", ax=ax, color=_nbstyle.palette(3), edgecolor=_nbstyle.SPINE, alpha=0.85)
ax.set_title("Coefficient Selection & Shrinkage Comparison", fontsize=11, fontweight="bold")
ax.set_ylabel("Estimated Coefficient", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 5. Pronósticos a un mes con una muestra cronológica reservada
#
# Fijamos los coeficientes al terminar el entrenamiento. Para cada objetivo reservado $y_t$,
# usamos exclusivamente $X_{t-1}$, que suponemos ya observado. Los resultados posteriores
# nunca entran en la estimación, la estandarización ni BIC. Son pronósticos sucesivos a un mes
# con coeficientes fijos, no un pronóstico a 40 meses elaborado en el corte inicial.
# La referencia es el promedio de los resultados de entrenamiento usados en la estimación.

# %%
X_test_lagged = df_X.iloc[TRAIN_END - 1:-1]
actual_test = s_y.iloc[TRAIN_END:]
predictions = pd.DataFrame(index=actual_test.index)
for name, result in [("Elastic Net", res_enet), ("Adaptive Lasso", res_alasso)]:
    predictions[name] = result.intercept + X_test_lagged.to_numpy() @ result.coefficients.to_numpy()
    assert np.isclose(predictions[name].iloc[0], result.forecast)
predictions["Training mean"] = y_train.iloc[1:].mean()
assert len(predictions) == T - TRAIN_END and np.isfinite(predictions).all().all()
errors = predictions.sub(actual_test, axis=0)
metrics = pd.DataFrame({"RMSE": np.sqrt(errors.pow(2).mean()), "MAE": errors.abs().mean()})
print(metrics.round(3).to_string())
assert metrics.loc["Adaptive Lasso", "RMSE"] < metrics.loc["Training mean", "RMSE"]

fig, ax = _nbstyle.figura(figsize=(10.0, 4.0))
ax.plot(actual_test.index, actual_test, **_nbstyle.S1, label="Observed holdout")
ax.plot(predictions.index, predictions["Adaptive Lasso"], **_nbstyle.S2, label="Adaptive Lasso forecast")
ax.plot(predictions.index, predictions["Training mean"], **_nbstyle.S3, label="Training-mean benchmark")
ax.set_title("Forecast evaluation: the final 40 months were held out")
ax.set_xlabel("Target month")
ax.set_ylabel("Inflation rate (%)")
ax.legend()

# %% [markdown]
# **Lectura del resultado.** RMSE y MAE describen ahora errores en objetivos no vistos.
# Compare ambos modelos con el promedio de entrenamiento, no con el $R^2$ dentro de muestra.
# Las señales plantadas permiten que el Lasso Adaptativo supere esa referencia con esta
# semilla; las diferencias entre modelos penalizados siguen siendo específicas de la muestra.
# La simulación no incluye revisiones de datos ni retrasos de publicación.
#
# ## Su turno — cambie la mezcla de penalizaciones
#
# Mantenga la división cronológica y cambie el parámetro de mezcla de la Red Elástica.
# La aserción verifica predicciones válidas; no exige que todas las opciones ganen.

# %%
your_alpha = 0.8  # ← change this penalty mix in [0, 1]; 0 = Ridge, 1 = Lasso
your_result = forecast_penalized(X_train, y_train, horizon=1, alpha=your_alpha, adaptive=False)
your_predictions = your_result.intercept + X_test_lagged.to_numpy() @ your_result.coefficients.to_numpy()
your_rmse = float(np.sqrt(np.mean((your_predictions - actual_test.to_numpy()) ** 2)))
assert your_predictions.shape == actual_test.shape and np.isfinite(your_predictions).all()
print(f"alpha={your_alpha:.2f}: holdout RMSE={your_rmse:.3f}, selected={len(your_result.selected_features)}")

# %% [markdown]
# **Ejercicios.** (1) Pruebe `your_alpha = 0` y `1`; compare variables seleccionadas y RMSE.
# (2) Active los pesos adaptativos y compare selecciones falsas con los predictores verdaderos.
# (3) Agregue una evaluación con ventana creciente que reestime usando únicamente observaciones
# disponibles en cada origen. Si usa esta muestra reservada para ajustar sus decisiones,
# reserve otro período posterior para la evaluación final.
#
# **¿Qué tan completo es esto?** `forecast_penalized` también admite horizontes directos y Ridge.
# El cuaderno 19 compara pérdidas predictivas con el conjunto de confianza de modelos; el 33
# introduce nowcasting con bordes irregulares, donde las fechas de publicación son parte del modelo.
