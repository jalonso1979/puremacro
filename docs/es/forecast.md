> 🇬🇧 [English](../forecast.md) · 🇪🇸 Español

# Pronóstico Macroeconómico Penalizado

`puremacro.forecast.forecast_penalized` responde a una pregunta concreta: a partir de un panel amplio de indicadores macroeconómicos candidatos, ¿cuáles aportan información predictiva genuina a un horizonte de $h$ pasos y cuál es el pronóstico puntual para el período $T+h$?

Es la herramienta adecuada para situaciones en las que el estimador de MCO deja de ser viable — por ejemplo, 200 indicadores mensuales frente a 120 trimestres históricos ($P > T$) — o cuando MCO está matemáticamente definido pero ajusta ruido muestral (*sobreajuste*). Devuelve un vector disperso (*sparse*) de coeficientes y una predicción puntual.

```python
import numpy as np
import pandas as pd
from puremacro.forecast import forecast_penalized

# El panel de 30 predictores del ejemplo: indicadores AR(1), cuatro de los cuales determinan y un paso adelante.
rng = np.random.default_rng(123)
T, P = 160, 30
X = np.zeros((T, P))
for j in range(P):
    rho = rng.uniform(0.3, 0.8)
    for t in range(1, T):
        X[t, j] = rho * X[t - 1, j] + rng.normal(scale=0.8)
y = np.full(T, 2.0)
for t in range(1, T):
    y[t] = 2.0 + 1.8 * X[t-1, 1] - 1.4 * X[t-1, 5] + 1.2 * X[t-1, 12] - 0.9 * X[t-1, 22] + rng.normal(scale=0.5)
dates = pd.date_range("2010-01-01", periods=T, freq="MS")
X_panel = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
y_target = pd.Series(y, index=dates, name="CPI Inflation")

res = forecast_penalized(X_panel, y_target, horizon=1, alpha=1.0, adaptive=True)
print("Pronóstico puntual:", res.forecast)
print("Predictores seleccionados:", res.selected_features)
print(res.summary())
```

El cuaderno 34 (`notebooks/34_penalized_macro_forecasting_es.py`) y `puremacro/examples/penalized_macro_forecasting.py` ejecutan el flujo completo sobre ese panel. Nada en esta página accede a la red.

---

## 1. Familia de estimadores y función objetivo

El algoritmo implementa descenso por coordenadas (*coordinate descent*) con umbralización suave (*soft-thresholding*) en NumPy puro, con la opción de ponderaciones adaptativas. Dos argumentos seleccionan el miembro de la familia:

| `alpha` | `adaptive` | Modelo estimado |
|---|---|---|
| `1.0` | `False` | Lasso estándar |
| `1.0` | `True` | **Lasso Adaptativo** (Zou 2006) |
| `0 < \alpha < 1` | `False` | Elastic Net (Zou y Hastie 2005) |
| `0 < \alpha < 1` | `True` | Elastic Net Adaptativo |
| `0.0` | `False` | Ridge — trayectoria en forma cerrada, véase la sección 3 |
| `0.0` | `True` | Ridge ponderado (penalización $w_j \beta_j^2$) |

`alpha` debe pertenecer a $[0, 1]$; cualquier otro valor lanza `ValueError`. La función objetivo sobre variables estandarizadas es:

$$\min_{\beta_0, \beta} \frac{1}{2T} \sum_{t=1}^{T-h} \left( y_{t+h} - \beta_0 - x_t'\beta \right)^2 + \lambda \sum_{j=1}^P w_j \left[ \alpha |\beta_j| + \frac{1}{2} (1 - \alpha) \beta_j^2 \right]$$

Cuando `adaptive=True`, los pesos $w_j$ se obtienen a partir de una estimación preliminar regularizada de Ridge:
$$w_j = \frac{1}{|\hat{\beta}_{\text{ridge}, j}| + 10^{-3}}$$
re-escalados para que la mediana de los pesos sea igual a 1. El suelo de $10^{-3}$ acota los pesos máximos en 1000, evitando la exclusión forzosa prematura de variables cuyos coeficientes preliminares sean numéricamente cercanos a cero. El exponente $\gamma$ está fijado en 1; no existe argumento `gamma`.

---

## 2. Parámetros principales

Todos los argumentos posteriores a `y_target` son exclusivamente nominales (*keyword-only*).

| Parámetro | Valor por defecto | Función |
|---|---|---|
| `X_panel` | *(obligatorio)* | DataFrame o ndarray $(T, P)$ de predictores fechados en $t$ |
| `y_target` | *(obligatorio)* | Serie o ndarray $(T,)$ de la variable objetivo |
| `horizon` | `1` | Horizonte directo $h$: se regresa $y_{t+h}$ sobre $X_t$ |
| `alpha` | `0.5` | Mezcla de penalización en $[0, 1]$ ($1.0 = \text{Lasso}$, $0.5 = \text{Elastic Net}$, $0.0 = \text{Ridge}$) |
| `adaptive` | `True` | Si es `True`, aplica los pesos adaptativos de Zou (2006) |
| `n_lambdas` | `40` | Número de puntos de la rejilla de $\lambda$ (escala logarítmica) |
| `lambda_min_ratio` | `1e-3` | Cociente $\lambda_{\min} / \lambda_{\max}$ |

Las tolerancias del descenso por coordenadas (`max_iter=1000`, `tol=1e-6`) son internas y no se exponen.

---

## 3. Selección de $\lambda$: BIC, sin validación cruzada

$\lambda$ se elige minimizando el criterio de información bayesiano sobre la muestra de estimación:

$$\text{BIC}(\lambda) = T_{\text{eff}} \log \text{MSE}(\lambda) + \text{df}(\lambda) \log T_{\text{eff}}$$

sobre `n_lambdas` puntos espaciados geométricamente entre $\lambda_{\max}$ y $\lambda_{\max} \cdot$ `lambda_min_ratio`. Para `alpha > 0`, $\lambda_{\max}$ es la forma cerrada habitual — la penalización más pequeña que anula todos los coeficientes, $\max_j |x_j'(y - \bar y)| / (T \alpha w_j)$ — y $\text{df}$ cuenta los coeficientes distintos de cero más uno.

**Ridge (`alpha=0`) sigue una trayectoria distinta.** Ningún $\lambda$ finito anula un coeficiente de Ridge, de modo que la rejilla se ancla en el espectro del diseño ponderado: con $e_{\max}$ el mayor autovalor de $X'X/T$ (para $X W^{-1/2}$), $\lambda_{\max} = 10\, e_{\max}$ — donde toda dirección de $X$ se contrae al menos por un factor 11 — y el `lambda_min_ratio` por defecto sitúa $\lambda_{\min}$ en $0{,}01\, e_{\max}$, donde la dirección principal se contrae un 1 %. La trayectoria se resuelve en forma cerrada a partir de una SVD y los grados de libertad del BIC son la traza de la matriz sombrero, $\sum_j d_j^2 / (d_j^2 + T\lambda)$ más uno. En el panel del ejemplo, `alpha=0.0, adaptive=False` alcanza un $R^2$ de 0,967 frente a 0,968 de MCO (en el extremo inferior de la rejilla, pues con $T = 159 \gg P = 30$ el BIC no tiene motivo para contraer) y `alpha=0.0, adaptive=True` encuentra un óptimo interior en $\lambda^* = 2{,}03$ con $R^2 = 0{,}964$. Ridge nunca selecciona: `selected_features` incluye los $P$ nombres.

**No hay validación cruzada en esta función ni en ningún lugar de `puremacro.forecast`**: ni *k-fold*, ni origen rodante, ni bloques temporales. $\lambda$ se elige para ajustar bien la muestra de estimación, no para pronosticar bien. Dos consecuencias:

- `optimal_lambda` no es comparable entre configuraciones: la rejilla depende de $\alpha$ y de los pesos adaptativos ($\lambda_{\max}$ es 1,57 con `alpha=1.0, adaptive=False` y 73,55 con `adaptive=True` en el ejemplo).
- Compruebe que el $\lambda$ elegido es interior. Con `alpha=0.02, adaptive=True` la rejilla por defecto se queda en $\lambda_{\min} = 3{,}68$ (11 predictores, $R^2 = 0{,}956$); `lambda_min_ratio=1e-5` desplaza el óptimo al interior en $\lambda^* = 2{,}29$ (13 predictores, $R^2 = 0{,}961$) y `n_lambdas=60` lo refina a $\lambda^* = 1{,}01$ (16 predictores, $R^2 = 0{,}966$).

```python
edges = (res.bic_path.index[0], res.bic_path.index[-1])
assert res.optimal_lambda not in edges, "amplíe lambda_min_ratio"
```

---

## 4. Pronóstico directo, nunca iterado

Con `horizon=h` se estima $y_{t+h} = \beta_0 + x_t'\beta$ sobre las filas $t = 0, \dots, T-h-1$ y se evalúa en la última fila de `X_panel`. No existe la ruta iterada: un pronóstico a horizonte 4 es un modelo *distinto*, no cuatro aplicaciones del modelo a horizonte 1, y el conjunto de predictores seleccionado cambia con $h$. En el ejemplo, el Lasso adaptativo selecciona 4 predictores en `h=1` ($R^2 = 0{,}963$) y otros 3 en `h=4` ($R^2 = 0{,}096$): la señal de un paso se ha disipado a cuatro pasos porque los predictores son AR(1) con $\rho$ entre 0,3 y 0,8.

`horizon=0` devuelve el valor ajustado *dentro de muestra* en la última observación, no un pronóstico; `horizon` no se valida y un valor negativo toma la misma rama.

---

## 5. Estandarización y lectura de los coeficientes

Los predictores se estandarizan con la media y la desviación típica poblacional (`ddof=0`) de las **filas de entrenamiento**; una columna constante recibe coeficiente exactamente `0.0`. La variable objetivo no se estandariza. Los coeficientes devueltos están **des-estandarizados**: `res.coefficients["x"]` es el efecto de una unidad de `x` en sus propias unidades, directamente utilizable en `res.intercept + X_row @ res.coefficients`. Para ordenar por importancia multiplique por la desviación típica de cada predictor.

---

## 6. Resultado

`PenalizedForecastResult` es un `dataclass` inmutable con `forecast`, `selected_features`, `coefficients` (Serie de longitud $P$, ceros incluidos), `intercept`, `optimal_lambda`, `in_sample_r2` (recortado a $[0, 1]$), `bic_path` (Serie indexada por $\lambda$ descendente) y `horizon`. Métodos de presentación: `summary()`, `to_frame()` (cada candidato con su coeficiente y un indicador `selected`), `to_markdown()` / `to_latex()` / `to_typst()` y `plot()` (barras horizontales de los coeficientes seleccionados, o la trayectoria del BIC si no sobrevivió ninguno; devuelve la figura).

No hay errores estándar ni intervalos: la inferencia tras la selección penalizada no es válida sin un paso de corrección de sesgo que este módulo no implementa.

---

## 7. Cinco advertencias

- **Los NaN se propagan en silencio y el $R^2$ miente.** Un NaN en cualquier lugar de `X_panel` devuelve `forecast=nan`, `selected_features=[]` e `in_sample_r2=1.0`. Elimine o interpole antes de llamar. Un NaN confinado a la **última** fila deja intactos coeficientes y $R^2$ y solo anula `forecast`.
- **La alineación es posicional; el índice se ignora.** Desplazar el índice de `y_target` no cambia nada. Longitudes distintas lanzan `ValueError: y_target has 155 rows but X_panel has 160; alignment is positional`.
- **`T - horizon` debe ser al menos 10**; por debajo lanza `ValueError`.
- **`in_sample_r2` es un diagnóstico del ajuste, no del pronóstico.** Con $P > T$ llega a 1,0 por interpolación.
- **`horizon` no se valida**; `alpha`, `n_lambdas` y `lambda_min_ratio` sí: `alpha=2.0`, que antes se ejecutaba, ahora lanza `ValueError`.

---

## 8. De un número a un historial

Evaluar el pronóstico exige reestimar en cada origen (seleccionar una vez sobre toda la muestra y «retro-probar» es sesgo de anticipación). Un ajuste cuesta unos 45 ms con $T = 200$, $P = 40$:

```python
import numpy as np
from puremacro.forecast import forecast_penalized, diebold_mariano

X, y = X_panel, y_target
h, start = 1, 120
preds, actuals = [], []
for t in range(start, len(y) - h):
    r = forecast_penalized(X.iloc[: t + 1], y.iloc[: t + 1], horizon=h, alpha=1.0)
    preds.append(r.forecast)
    actuals.append(y.iloc[t + h])

e_pen = np.array(preds) - np.array(actuals)
e_bench = np.array([y.iloc[: t + 1].mean() for t in range(start, len(y) - h)]) - np.array(actuals)
print(diebold_mariano(e_pen, e_bench, h=h))
```

`puremacro.forecast` aporta el resto del instrumental de evaluación: `diebold_mariano`, `giacomini_white`, `model_confidence_set` (Hansen-Lunde-Nason con *bootstrap* estacionario) con `losses_from_forecasts`, y las funciones de densidad `crps_gaussian`, `crps_ensemble`, `pit`, `pit_uniformity_test`, `berkowitz_test`, `klic_amisano_giacomini` y `combine_forecasts`.
