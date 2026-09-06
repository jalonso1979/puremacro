> 🇬🇧 [English](../smooth_lp.md) · 🇪🇸 Español

# Proyecciones locales suavizadas

El módulo `puremacro.lp.smooth_lp` implementa la metodología de proyecciones locales regularizadas mediante B-splines penalizados desarrollada por **Barnichon y Brownlees (2019, *The Review of Economics and Statistics*)**.

El estimador convencional de proyecciones locales (**Jordà 2005**) obtiene las funciones de respuesta al impulso (FRI / IRF) estimando regresiones independientes por Mínimos Cuadrados Ordinarios (MCO) para cada horizonte temporal $h = 0, \dots, H$. Si bien el enfoque de Jordà es sumamente robusto frente a errores de especificación dinámica, estimar regresiones desconectadas horizonte a horizonte ignora la continuidad y suavidad inherentes a los mecanismos de propagación macroeconómica, originando a menudo estimaciones ruidosas, trayectorias espurias e intervalos de confianza excesivamente anchos en horizontes medianos y distantes.

Las proyecciones locales suavizadas resuelven estas ineficiencias **estimando conjuntamente la respuesta al impulso para todos los horizontes** mediante una base continua de B-splines cúbicos sujeta a una penalización por rugosidad, logrando una reducción sustancial de la varianza muestral manteniendo la insesgadez asintótica.

---

## 1. Fundamentos econométricos

### 1.1 Formulación del modelo

Sea $y_t$ la variable de respuesta de interés, $x_t$ la perturbación estructural o choque de política, y $z_t$ un vector de controles (constante, `n_lags` retardos de $y_t$, $x_t$ y de los controles, y los controles contemporáneos). En el esquema tradicional de proyecciones locales:

$$y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h' z_t + \varepsilon_{t+h}, \quad h = 0, \dots, H$$

Cada horizonte $h$ se estima sobre su propia muestra $S_h = \{t : t + h \le T\}$ (en el horizonte $h$ se pierden las últimas $h$ observaciones), exactamente igual que `lp_hac`. Aplicando el teorema de Frisch-Waugh-Lovell (FWL), los controles $z_t$ se proyectan fuera horizonte a horizonte para obtener el adelanto residualizado $\tilde{y}_{h,t}$ y el choque residualizado $\tilde{x}_{h,t}$.

Barnichon y Brownlees aproximan la función de respuesta al impulso $\beta(h)$ como una combinación lineal de $K$ funciones de base B-spline evaluadas sobre la malla de horizontes $h \in \{0, 1, \dots, H\}$:

$$\beta(h) = \sum_{k=1}^K B_k(h) \theta_k = B_h \theta$$

donde $B$ es la matriz $(H+1) \times K$ de B-splines cúbicos anclados en los extremos y $\theta \in \mathbb{R}^K$ es el vector de coeficientes de la base.

El sistema apilado para todos los horizontes se estima mediante **Mínimos Cuadrados Penalizados (PLS)**:

$$\min_\theta \; \sum_{h=0}^{H} \sum_{t \in S_h} \big( \tilde{y}_{h,t} - \tilde{x}_{h,t} B_h \theta \big)^2 + \lambda \, \theta' P \theta
\;=\; \min_\theta \; \| \tilde{Y} - X \theta \|^2 + \lambda \, \theta' P \theta$$

donde:
- $X = B \otimes \tilde{x}$ es la matriz de diseño apilada (producto de Kronecker) sobre las muestras específicas de cada horizonte.
- $P = D_d' D_d$ es la matriz de penalización por rugosidad generada por el operador de diferencias de orden $d$ (por defecto segundas diferencias $d=2$).
- $\lambda \ge 0$ es el hiperparámetro de regularización que calibra el dilema sesgo-varianza. **`lam` y `res.optimal_lambda` son el $\lambda$ de exactamente esta función objetivo**, es decir, la penalización es $\lambda\,\theta'P\theta$ frente a la suma de cuadrados apilada (que es $O(T)$, por lo que $\lambda$ no es adimensional: la malla automática descrita más abajo se adapta a la muestra).
  - Cuando $\lambda \to 0$, las estimaciones convergen a la proyección no penalizada de las proyecciones locales MCO horizonte a horizonte sobre la base spline; con una base saturada (`n_knots = H - degree`) coinciden *exactamente* con `lp_hac` en la muestra de cada horizonte.
  - Cuando $\lambda \to \infty$, la respuesta al impulso se constriñe a un polinomio de grado $d-1$ (una recta para $d=2$).

Con `gls=True` la suma de cuadrados se sustituye por la forma cuadrática de **Mínimos Cuadrados Generalizados Penalizados (PGLS)** $(\tilde{Y} - X\theta)'\,\Omega^{-1}\,(\tilde{Y} - X\theta)$, donde $\Omega$ es la covarianza $(H+1)\times(H+1)$ entre horizontes de los residuos MCO horizonte a horizonte. Estimar $\Omega$ requiere un panel de residuos balanceado, de modo que con `gls=True` todos los horizontes usan la muestra común $t + H \le T$ (`res.sample == "balanced"`); con el valor por defecto `gls=False` cada horizonte usa su propia muestra (`res.sample == "per-horizon"`, `res.n_obs` contiene $|S_h|$).

**Base.** `n_knots` es el número de nudos *interiores*, equiespaciados entre el primer y el último horizonte, de modo que la base tiene $K = $ `n_knots + degree + 1` funciones. Por defecto se usa aproximadamente un nudo por cada tres horizontes. $K$ se limita al número de horizontes (para que $\lambda \to 0$ recupere la proyección local no penalizada); un `n_knots` explícito por encima del límite se reduce emitiendo un `UserWarning`, y el valor efectivo se guarda en `res.n_knots`.

### 1.2 Selección óptima de $\lambda$ guiada por datos

Con `lam="auto"` (valor por defecto; no distingue mayúsculas, `None` es equivalente) `smooth_lp` explora la malla `res.lambda_grid = logspace(-5, 5, 50) * mean_h(x̃_h' x̃_h)`, que abarca desde respuestas prácticamente sin penalizar hasta respuestas prácticamente polinómicas para cualquier tamaño muestral, y minimiza:

1. **Criterio de Información de Akaike (`selection='aic'`)**: $\log(\text{RSS}/N) + 2\,\text{df}(\lambda)/N$ con grados de libertad efectivos $\text{df}(\lambda) = \text{tr}\big((X' \Omega^{-1} X + \lambda P)^{-1} X' \Omega^{-1} X\big)$ y $N = \sum_h |S_h|$.
2. **Criterio de Información Bayesiano (`selection='bic'`)**: $\log(\text{RSS}/N) + \log(N)\,\text{df}(\lambda)/N$.
3. **Validación Cruzada Generalizada (`selection='gcv'`)**: $(\text{RSS}/N) / (1 - \text{df}(\lambda)/N)^2$.
4. **Validación Cruzada en Bloques (`selection='cv'`)**: error cuadrático fuera de muestra sobre $K = \min(5, T/10)$ bloques temporales consecutivos; el estimador de cada bloque es el mismo estimador (G)LS penalizado que el ajuste final.

El valor seleccionado se reporta en `res.optimal_lambda`, `res.df_lambda`, en la columna `lambda` y en `res.summary()`. Un número suministrado por el usuario fija $\lambda$ (`res.selection_criterion == "fixed"`); `selection` se valida en ambos casos.

### 1.3 Inferencia y bandas de confianza

Se proporcionan dos esquemas robustos de inferencia:

1. **Errores estándar analíticos tipo sándwich HAC (`ci_type='analytic'`)**:  
   Errores estándar de Newey-West con núcleo de Bartlett para los coeficientes $\theta$ (ancho de banda `hac_lags`, por defecto $H$), proyectados sobre la base de horizontes:
   $$\widehat{\text{Var}}(\hat{\beta}) = B \left( X' X + \lambda P \right)^{-1} X' \hat{\Sigma}_{HAC} X \left( X' X + \lambda P \right)^{-1} B'$$
2. **Remuestreo por bloques temporales (*Moving Block Bootstrap*, `ci_type='bootstrap'`)**:  
   Genera pseudomuestras mediante bloques solapados de longitud $\lceil T^{1/3} \rceil$ para capturar no paramétricamente la autocorrelación y la heterocedasticidad entre horizontes (`n_boot` replicaciones, reproducible con `seed`).

---

## 2. Reducción empírica de la varianza

En calibraciones dinámicas estándar (como modelos VAR monetarios o series AR(2)):
- Las proyecciones locales no penalizadas adolecen de una marcada varianza muestral que crece rápidamente con el horizonte $h$.
- Las proyecciones locales suavizadas logran una varianza Monte Carlo y un error cuadrático medio claramente menores en horizontes medios y largos ($h \ge 4$); Barnichon y Brownlees documentan mejoras de RMSE del orden del 30-50% en sus simulaciones, y la batería de pruebas del paquete comprueba una reducción de varianza de al menos el 20% en un DGP AR(2). La ganancia depende del DGP.

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

# 3. Presentación de resultados y diagnósticos (el resumen reporta el lambda seleccionado)
print(res_smooth.summary())
print(f"Lambda óptimo: {res_smooth.optimal_lambda:.4g}  (grados de libertad efectivos = {res_smooth.df_lambda:.2f})")

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
# lambda = 0 con una base saturada (H - degree nudos interiores) es exactamente
# la proyección local MCO horizonte a horizonte (mismas estimaciones que lp_hac)
res_ols = smooth_lp(df=df, y="y", x="choque", horizons=20, n_knots=17, lam=0.0)

# Proyección local suavizada guiada por BIC
res_opt = smooth_lp(df=df, y="y", x="choque", horizons=20, lam="auto", selection="bic")

# La regularización óptima produce bandas significativamente más estrechas y estables
print("Error estándar medio (MCO):", np.mean(res_ols.se))
print("Error estándar medio (Suave):", np.mean(res_opt.se))
print("Lambda por BIC             :", res_opt.optimal_lambda, "en la malla", res_opt.lambda_grid[[0, -1]])
```

### Convenciones de llamada: vectores, entradas mixtas y controles como arrays

`smooth_lp` acepta las mismas dos convenciones que `lp_hac`:

```python
y_arr, x_arr = df["y"].to_numpy(), df["choque"].to_numpy()
controles = rng.standard_normal((T, 2))

# Vectores: primero la respuesta, después el choque (igual que lp_hac(y, choque, ...))
res_arr = smooth_lp(y_arr, x_arr, horizons=12, n_lags=2, controls=controles)

# DataFrame con nombres de columna; y/x también pueden ser vectores de longitud len(df),
# y controls puede ser una lista de nombres, un array (T,) / (T, k) o un DataFrame
res_mix = smooth_lp(df, "y", x_arr, horizons=12, n_lags=2, controls=controles)

assert np.allclose(res_arr.point, res_mix.point)
print(res_arr.y_name, res_arr.x_name)        # 'y' 'x'  (se usan los nombres de las Series si existen)
print(res_arr.n_obs)                          # tamaños muestrales por horizonte T - n_lags - h
```

---

## 4. Especificación completa de la API

### `smooth_lp`

```text
smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | pd.DataFrame | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    lambda_: float | None = None,     # alias de lam
    lags: int | None = None,          # alias de n_lags
    horizon: int | None = None,       # alias de horizons (horizonte máximo)
    ci: float | None = None,          # nivel de confianza, fija alpha = 1 - ci
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> SmoothLPResult
```

`lp_smooth` es un alias de `smooth_lp`.

#### Parámetros:
- `df`: `DataFrame` de pandas que contiene la variable objetivo, el choque y los controles; o, con entrada vectorial, la serie de respuesta 1-D.
- `y`: Nombre de columna (o vector de longitud `len(df)`) de la variable de respuesta; con entrada vectorial, la serie 1-D del choque (`smooth_lp(y_arr, x_arr)`).
- `x`: Nombre de columna (o vector de longitud `len(df)`) del choque; con entrada vectorial puede pasarse como `x=` en lugar del segundo argumento posicional (pasar ambos produce un error).
- `horizons`: Horizonte máximo entero $H$ (al menos 1) o iterable de al menos dos horizontes.
- `n_lags`: Número de retardos autorregresivos incorporados para la variable objetivo, el choque y los controles.
- `controls`: Controles exógenos opcionales: nombres de columna, un array `(T,)` / `(T, k)` o un DataFrame.
- `n_knots`: Número de nudos *interiores* del spline (tamaño de la base `n_knots + degree + 1`; valor adaptativo si es `None`; se limita al número de horizontes con un aviso).
- `degree`: Grado polinómico de los B-splines (por defecto `3` para splines cúbicos).
- `penalty_order`: Orden del operador de diferencias en la matriz de penalización $P = D_d' D_d$ (por defecto `2`).
- `lam`: Coeficiente de regularización $\lambda$ de la función objetivo apilada. `'auto'` (sin distinguir mayúsculas) o `None` para la selección guiada por datos, o un número no negativo; valores negativos, NaN u otras cadenas producen `ValueError`.
- `selection`: Criterio de optimización para $\lambda$: `'aic'`, `'bic'`, `'gcv'` o `'cv'` (se valida incluso cuando `lam` es fijo).
- `alpha`: Nivel de significación para los intervalos de confianza (por defecto `0.05` para 95% de confianza).
- `ci_type`: Método inferencial: `'analytic'` (sándwich HAC) o `'bootstrap'` (bloques temporales); cualquier otro valor produce `ValueError`.
- `n_boot`: Replicaciones bootstrap cuando `ci_type='bootstrap'`.
- `seed`: Semilla para asegurar la reproducibilidad del muestreo bootstrap.
- `hac_lags`: Ancho de banda de Bartlett del estimador HAC (por defecto: el horizonte máximo).
- `gls`: Si es `True`, aplica ponderación por Mínimos Cuadrados Generalizados factibles entre horizontes sobre la muestra común (balanceada).

---

## 5. Interfaz de resultados

`smooth_lp` devuelve un `SmoothLPResult`, subclase de `LPResult` (a su vez un `pandas.DataFrame`) con la misma estructura que el resto de estimadores de `puremacro.lp`:

- **Columnas de datos** (indexadas por `h`): `h`, `beta` (estimaciones puntuales $\hat\beta(h)$), `se`, `lo`, `hi` (bandas puntuales al nivel $1-\alpha$), `lambda` (el $\lambda$ utilizado, repetido en cada fila), `t` (estadístico $t$ para $H_0: \beta(h)=0$).
- **Propiedades vectoriales**: `.point`, `.se`, `.ci_lower`, `.ci_upper`, `.t_stat`, `.horizons` (alias de las columnas anteriores). No existe una columna `p_value`; si se necesita, use `2 * (1 - scipy.stats.norm.cdf(abs(res.t_stat)))`.
- **Atributos de estimación** (sobreviven a `.copy()`, a la selección de filas y de columnas): `optimal_lambda`, `df_lambda`, `lambda_grid`, `selection_criterion` (`'aic'`, `'bic'`, `'gcv'`, `'cv'` o `'fixed'`), `ci_type`, `theta`, `vcov` (de $\hat\beta$), `vcov_theta`, `B`, `P`, `n_knots` (nudos interiores efectivos), `n_basis`, `degree`, `penalty_order`, `gls`, `n_obs` (tamaños muestrales por horizonte), `sample` (`'per-horizon'` o `'balanced'`), `y_name`, `x_name`, `method == 'LP-smooth'`, y `.metadata` (un diccionario con lo anterior).
- **Métodos disponibles**:
  - `.plot()`: Figura de Matplotlib con la curva de respuesta al impulso y las bandas de confianza sombreadas.
  - `.summary()`: la tabla de la proyección local seguida de los diagnósticos de suavizado ($\lambda$ seleccionado y cómo se eligió, grados de libertad efectivos, base, método de inferencia, muestras).
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas estructuradas listas para su incorporación directa en artículos académicos.
