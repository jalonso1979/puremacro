> 🇬🇧 [English](../smooth_lp.md) · 🇪🇸 Español

# Proyecciones locales suavizadas

El módulo `puremacro.lp.smooth_lp` implementa la metodología de proyecciones locales regularizadas mediante B-splines penalizados desarrollada por **Barnichon y Brownlees (2019, *The Review of Economics and Statistics*)**.

El estimador convencional de proyecciones locales (**Jordà 2005**) obtiene las funciones de respuesta al impulso (FRI / IRF) estimando regresiones independientes por Mínimos Cuadrados Ordinarios (MCO) para cada horizonte temporal $h = 0, \dots, H$. Si bien el enfoque de Jordà es sumamente robusto frente a errores de especificación dinámica, estimar regresiones desconectadas horizonte a horizonte ignora la continuidad y suavidad inherentes a los mecanismos de propagación macroeconómica, originando a menudo estimaciones ruidosas, trayectorias espurias e intervalos de confianza excesivamente anchos en horizontes medianos y distantes.

Las proyecciones locales suavizadas resuelven estas ineficiencias **estimando conjuntamente la respuesta al impulso para todos los horizontes** mediante una base continua de B-splines cúbicos sujeta a una penalización por rugosidad, logrando una drástica reducción de la varianza muestral manteniendo la insesgadez asintótica.

---

## 1. Fundamentos econométricos

### 1.1 Formulación del modelo

Sea $y_t$ la variable de respuesta de interés, $x_t$ la perturbación estructural o choque de política, y $w_t$ un vector de variables de control contemporáneas y retardadas. En el esquema tradicional de proyecciones locales:

$$y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h' w_t + \varepsilon_{t+h}, \quad h = 0, \dots, H$$

Aplicando el teorema de Frisch-Waugh-Lovell (FWL), se proyectan los controles $w_t$ fuera de las series para obtener los residuos ortogonales de respuesta $\tilde{y}$ y del choque $\tilde{x}$.

Barnichon y Brownlees aproximan la función continua de respuesta al impulso $\beta(h)$ como una combinación lineal de $K$ funciones de base B-spline evaluadas sobre la malla discreta de horizontes $h \in \{0, 1, \dots, H\}$:

$$\beta(h) = \sum_{k=1}^K B_k(h) \theta_k = B_h \theta$$

donde $B$ representa la matriz de dimensión $(H+1) \times K$ de B-splines cúbicos anclados en los extremos y $\theta \in \mathbb{R}^K$ es el vector de coeficientes de la base spline.

El sistema apilado para todos los horizontes se estima mediante **Mínimos Cuadrados Penalizados (PLS)** o **Mínimos Cuadrados Generalizados Penalizados (PGLS)**:

$$\min_\theta \; (\tilde{Y} - X \theta)' \Omega^{-1} (\tilde{Y} - X \theta) + \lambda \theta' P \theta$$

donde:
- $X = B \otimes \tilde{x}$ es la matriz de diseño obtenida por producto de Kronecker.
- $\Omega$ es la matriz de covarianzas del error entre horizontes (la matriz identidad para PLS; la covarianza estimada inter-horizontes para PGLS).
- $P = D_d' D_d$ es la matriz de penalización por rugosidad generada por el operador de diferencias de orden $d$ (típicamente segundas diferencias $d=2$).
- $\lambda \ge 0$ es el hiperparámetro de regularización que calibra el dilema sesgo-varianza:
  - Cuando $\lambda \to 0$, las estimaciones convergen al estimador no penalizado de proyecciones locales de Jordà.
  - Cuando $\lambda \to \infty$, la trayectoria de la respuesta al impulso se constriñe hacia un polinomio suave de bajo grado.

### 1.2 Selección óptima de $\lambda$ guiada por datos

`puremacro.lp.smooth_lp` automatiza la calibración óptima del parámetro de penalización $\lambda$ según criterios formales:

1. **Criterio de Información de Akaike (`selection='aic'`)**: Minimiza el error de predicción dentro de la muestra penalizado por los grados de libertad efectivos $\text{df}(\lambda) = \text{tr}((X' \Omega^{-1} X + \lambda P)^{-1} X' \Omega^{-1} X)$.
2. **Criterio de Información Bayesiano (`selection='bic'`)**: Aplica una penalización más estricta ponderada por el logaritmo del tamaño muestral $\log(N) \cdot \text{df}(\lambda)$.
3. **Validación Cruzada Generalizada (`selection='gcv'`)**: Aproximación invariante a rotaciones del error cuadrático medio fuera de la muestra.
4. **Validación Cruzada en Bloques (`selection='cv'`)**: Validación temporal segmentada por particiones temporales consecutivas.

### 1.3 Inferencia y bandas de confianza

Se proporcionan dos esquemas robustos de inferencia:

1. **Errores estándar analíticos tipo sándwich HAC (`ci_type='analytic'`)**:  
   Calcula la matriz de varianzas robusta de Newey-West con núcleo de Bartlett para los parámetros $\theta$, proyectándola sobre la base de horizontes:
   $$\widehat{\text{Var}}(\hat{\beta}) = B \left( X' X + \lambda P \right)^{-1} X' \hat{\Sigma}_{HAC} X \left( X' X + \lambda P \right)^{-1} B'$$
2. **Remuestreo por bloques temporales (*Moving Block Bootstrap*, `ci_type='bootstrap'`)**:  
   Genera pseudomuestras mediante bloques continuos solapados para capturar no paramétricamente la autocorrelación residual y la heterocedasticidad condicional.

---

## 2. Reducción empírica de la varianza

En calibraciones dinámicas estándar (como modelos VAR monetarios o procesos AR(2) de inversión):
- Las proyecciones locales no penalizadas adolecen de una marcada varianza muestral que crece rápidamente conforme se incrementa el horizonte de proyección $h$.
- Las proyecciones locales suavizadas logran **reducir el error cuadrático medio (RMSE) entre un 30% y un 50%** en horizontes medios y largos ($h \ge 4$) preservando la cobertura nominal de los intervalos de confianza.

---

## 3. Ejemplo de uso y código ejecutable

### Inicio rápido

```python
import numpy as np
import pandas as pd
from puremacro.lp import smooth_lp

# 1. Generación de series temporales macroeconómicas sintéticas
rng = np.random.default_rng(42)
T = 200
choque = rng.standard_normal(T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7 * y[t-1] + 0.5 * choque[t] + 0.2 * rng.standard_normal()

df = pd.DataFrame({"y": y, "choque": choque})

# 2. Estimación de la proyección local suavizada con selección automática por AIC
res_smooth = smooth_lp(
    df=df,
    y="y",
    x="choque",
    horizons=20,
    n_lags=4,
    lam="auto",
    selection="aic",
    degree=3,
    penalty_order=2,
    ci_type="analytic",
    alpha=0.05,
)

# 3. Presentación de resultados y diagnósticos
print(res_smooth.summary())
print(f"Lambda óptimo: {res_smooth.attrs.get('lambda_optimal', 'N/A')}")

# 4. Gráfico de la respuesta al impulso suavizada con bandas de confianza
fig = res_smooth.plot()

# 5. Acceso a las series numéricas
estimaciones_puntuales = res_smooth.point   # ndarray (H+1,) con las respuestas
errores_estandar = res_smooth.se            # ndarray (H+1,) con errores HAC
banda_inferior = res_smooth.ci_lower        # Límites inferiores al 95%
banda_superior = res_smooth.ci_upper        # Límites superiores al 95%
```

### Comparación entre proyecciones convencionales y suavizadas

```python
# Proyecciones locales de Jordà aproximadas mediante penalización casi nula
res_ols = smooth_lp(df=df, y="y", x="choque", horizons=20, lam=1e-4)

# Proyección local suavizada guiada por BIC
res_opt = smooth_lp(df=df, y="y", x="choque", horizons=20, lam="auto", selection="bic")

# La regularización óptima produce bandas significativamente más estrechas y estables
print("Error estándar medio (MCO):", np.mean(res_ols.se))
print("Error estándar medio (Suave):", np.mean(res_opt.se))
```

---

## 4. Especificación completa de la API

### `smooth_lp`

```python
smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> LPResult
```

#### Parámetros:
- `df`: `DataFrame` de pandas o `ndarray` que contiene la variable objetivo, el choque y los controles.
- `y`: Nombre de columna o vector de la variable dependiente de respuesta.
- `x`: Nombre de columna o vector de la perturbación o política de interés.
- `horizons`: Horizonte máximo entero $H$ o secuencia iterable de horizontes.
- `n_lags`: Número de retardos autorregresivos incorporados para la variable objetivo, choque y controles.
- `controls`: Lista opcional de variables exógenas de control.
- `n_knots`: Cantidad de nudos interiores para los splines (seleccionado adaptativamente si es `None`).
- `degree`: Grado polinómico de los B-splines (por defecto `3` para splines cúbicos).
- `penalty_order`: Orden del operador de diferencias en la matriz de penalización $P = D_d' D_d$ (por defecto `2`).
- `lam`: Coeficiente de regularización $\lambda$. Ajustar a `'auto'` para optimización guiada por datos o ingresar un valor flotante positivo.
- `selection`: Criterio de optimización para $\lambda$: `'aic'`, `'bic'`, `'gcv'` o `'cv'`.
- `alpha`: Nivel de significancia para los intervalos de confianza (por defecto `0.05` para 95% de confianza).
- `ci_type`: Método inferencial: `'analytic'` (sándwich HAC) o `'bootstrap'` (bloques temporales).
- `n_boot`: Replicaciones bootstrap cuando `ci_type='bootstrap'`.
- `seed`: Semilla para asegurar reproducibilidad en el muestreo bootstrap.
- `gls`: Si es `True`, aplica ponderación por Mínimos Cuadrados Generalizados factibles entre horizontes.

---

## 5. Interfaz de resultados

`smooth_lp` devuelve un objeto especializado `LPResult` (subclase de `pandas.DataFrame`):

- **Columnas de datos**:
  - `point`: Estimaciones puntuales $\hat{\beta}(h)$.
  - `se`: Errores estándar.
  - `ci_lower`, `ci_upper`: Bandas de confianza puntuales.
  - `t_stat`: Estadísticos $t$ para el contraste de nulidad $H_0: \beta(h) = 0$.
  - `p_value`: Niveles de significación asintóticos ($p$-valores).
- **Métodos disponibles**:
  - `.plot()`: Gráfico en Matplotlib con la curva de respuesta al impulso y las bandas de confianza sombreadas.
  - `.summary()`: Resumen técnico con los diagnósticos de la regresión y el valor óptimo de $\lambda$.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas estructuradas listas para su incorporación directa en artículos académicos.
