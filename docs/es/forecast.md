> 🇬🇧 [English](../forecast.md) · 🇪🇸 Español

# Pronóstico Macroeconómico Penalizado

`puremacro.forecast.forecast_penalized` responde a una pregunta concreta: a partir de un panel amplio de indicadores macroeconómicos candidatos, ¿cuáles aportan información predictiva genuina a un horizonte de $h$ pasos y cuál es el pronóstico puntual para el período $T+h$?

Es la herramienta adecuada para situaciones en las que el estimador de MCO deja de ser viable — por ejemplo, 200 indicadores mensuales frente a 120 trimestres históricos ($P > T$) — o cuando MCO está matemáticamente definido pero ajusta ruido muestral (*sobreajuste*). Devuelve un vector disperso (*sparse*) de coeficientes y una predicción puntual insesgada.

```python
from puremacro.forecast import forecast_penalized

res = forecast_penalized(X_panel, y_target, horizon=1, alpha=1.0, adaptive=True)
print("Pronóstico puntual:", res.forecast)
print("Predictores seleccionados:", res.selected_features)
print(res.summary())
```

---

## 1. Familia de estimadores y función objetivo

El algoritmo implementa descenso por coordenadas (*coordinate descent*) con umbralización suave (*soft-thresholding*) en NumPy puro, con la opción de ponderaciones adaptativas:

| `alpha` | `adaptive` | Modelo estimado |
|---|---|---|
| `1.0` | `False` | Lasso estándar |
| `1.0` | `True` | **Lasso Adaptativo** (Zou 2006) |
| `0 < \alpha < 1` | `False` | Elastic Net (Zou y Hastie 2005) |
| `0 < \alpha < 1` | `True` | Elastic Net Adaptativo |

La función objetivo sobre variables estandarizadas es:

$$\min_{\beta_0, \beta} \frac{1}{2T} \sum_{t=1}^{T-h} \left( y_{t+h} - \beta_0 - x_t'\beta \right)^2 + \lambda \sum_{j=1}^P w_j \left[ \alpha |\beta_j| + \frac{1}{2} (1 - \alpha) \beta_j^2 \right]$$

Cuando `adaptive=True`, los pesos $w_j$ se obtienen a partir de una estimación preliminar regularizada de Ridge:
$$w_j = \frac{1}{|\hat{\beta}_{\text{ridge}, j}| + 10^{-3}}$$
re-escalados para que la mediana de los pesos sea igual a 1. El suelo de $10^{-3}$ acota los pesos máximos en 1000, evitando la exclusión forzosa prematura de variables cuyos coeficientes preliminares sean numéricamente cercanos a cero.

---

## 2. Parámetros principales

| Parámetro | Valor por defecto | Función |
|---|---|---|
| `X_panel` | *(obligatorio)* | DataFrame o ndarray $(T, P)$ de predictores fechados en $t$ |
| `y_target` | *(obligatorio)* | Serie o ndarray $(T,)$ de la variable objetivo |
| `horizon` | `1` | Horizonte de pronóstico $h \ge 1$ |
| `alpha` | `1.0` | Mezcla de penalización ($1.0 = \text{Lasso}$, $0.5 = \text{Elastic Net}$) |
| `adaptive` | `True` | Si es `True`, aplica los pesos adaptativos de Zou (2006) |
| `n_lambdas` | `50` | Número de valores en la trayectoria de regularización $\lambda$ |
| `cv_folds` | `5` | Número de bloques para validación cruzada temporal |
