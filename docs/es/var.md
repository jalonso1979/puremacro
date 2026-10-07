> 🇬🇧 [English](../var.md) · 🇪🇸 Español

# Modelos VAR y FAVAR

`puremacro.var` ajusta un modelo de vectores autorregresivos (VAR) en forma reducida y aborda la cuestión fundamental que la forma reducida no puede dilucidar por sí misma: *cuál* de las innovaciones correlacionadas corresponde a cada perturbación estructural.

Todos los esquemas de identificación en `puremacro.var.identify` parten de la misma forma reducida y se diferencian exclusivamente en las restricciones económicas supuestas — un ordenamiento recursivo, neutralidad de largo plazo, patrones de signo, un instrumento externo o cambios en la varianza residual. Esta intercambiabilidad de la API permite que la elección metodológica obedezca a fundamentos teóricos y no a dificultades de implementación.

Todo el código de esta página se ejecuta de forma estrictamente local y sin conexión a internet.

```python
from puremacro.var import lag_select
from puremacro.var.identify import cholesky

Y = ...                                   # ndarray (T, n), filas ordenadas temporalmente
p = lag_select(Y, maxlags=8, ic="bic")
res = cholesky(Y, p=p, horizon=20, n_boot=500, ci=0.9, seed=0)
print(res.summary())
res.irf_point[4, 1, 0]                    # respuesta de la variable 1 en h=4 al choque 0
```

## 1. La forma reducida

`fit_var(Y, p)` estima el modelo por MCO con constante. Devuelve una dataclass congelada `VarEstimateResult`:

| Atributo | Dimensión | Descripción |
|---|---|---|
| `A_list` | lista de `p` matrices `(n, n)` | Matrices de coeficientes $A_1 \dots A_p$ |
| `c` | `(n,)` | Vector de constantes |
| `Sigma` | `(n, n)` | Matriz de covarianzas residual con corrección por grados de libertad $T - p - 1 - np$ |
| `resid` | `(T - p, n)` | Residuos estimados de forma reducida |
| `X` | `(T - p, 1 + np)` | Matriz de diseño con la constante en la columna 0 |

`lag_select(Y, maxlags=8, ic="bic")` selecciona el número de retardos minimizando el criterio informativo seleccionado (`"aic"`, `"bic"` o `"hq"`). `companion(A_list)` calcula la matriz compañera e `is_stable(A_list)` comprueba la condición de estabilidad $\max |\lambda_i| < 1$.

---

## 2. Esquemas de identificación estructural (`var.identify.*`)

### Cholesky (`cholesky` / `cholesky_svar`)
Identificación recursiva mediante factorización de Cholesky de la matriz de covarianzas: $\Sigma = B_0 B_0'$. Requiere justificar económicamente el ordenamiento causal contemporáneo de las variables:

```python
from puremacro.var.identify import cholesky_svar

res_chol = cholesky_svar(Y, p=2, horizon=20, n_boot=500, ci=0.90)
res_chol.plot(target_idx=0, shock_idx=0)
```

### Blanchard–Quah (`bq_svar`)
Identificación por restricciones de largo plazo (Blanchard y Quah 1989), imponiendo que perturbaciones transitorias (ej. choques de demanda) no tengan impacto acumulado sobre variables no estacionarias en el largo plazo (ej. PIB):
$$C(1) = (I - A_1 - \dots - A_p)^{-1} B_0$$
donde $C(1)$ se restringe a ser triangular inferior una vez que la variable `permanent_var_idx` se coloca primero.

**Qué respuestas se devuelven acumuladas depende de cómo entró cada columna, y lo indica `cumulate=`.** El VAR debe ser estacionario: una variable con raíz unitaria entra en primeras diferencias y una estacionaria puede entrar en niveles. Acumular a lo largo del horizonte la respuesta de una variable diferenciada da la respuesta de su nivel. Acumular una variable que ya está en niveles da la suma corrida de su respuesta, que converge a $((I - \sum A_i)^{-1} B_0)_{i\cdot}$ en lugar de volver a cero: un efecto permanente que la identificación descarta.

| especificación | columnas de `Y` | `cumulate` | `irf_point[h, i, j]` |
|---|---|---|---|
| Blanchard–Quah (1989) | `[100·Δlog PNB, tasa de desempleo]` | `[0]` | fila 0: % de desviación del nivel del PNB; fila 1: desviación de la tasa de desempleo, en puntos |
| Galí (1999), horas en diferencias | `[100·Δlog productividad, 100·Δlog horas]` | `True` (por defecto) | ambas filas: % de desviación del nivel |
| horas en niveles (la crítica CEV) | `[100·Δlog productividad, log horas]` | `[0]` | fila 1: desviación del log de las horas, sin acumular |
| respuestas brutas de las columnas tal como entraron | cualquiera | `False` | ninguna fila acumulada |

Blanchard y Quah definen $X = (\Delta Y, U)'$ con $Y$ el logaritmo del PNB y $U$ el *nivel* de la tasa de desempleo, y obtienen el efecto sobre el nivel de $Y$ tras $k$ periodos como la suma parcial de las respuestas de $\Delta Y$ (NBER WP 2737, 1988, p. 3); sus gráficas muestran el log del producto y la propia tasa de desempleo. Para su sistema:

```python
import numpy as np
from puremacro.var.identify import bq_svar

# pnb, desempleo: sus series trimestrales de PNB real y tasa de desempleo (%)
Y = np.column_stack([100 * np.diff(np.log(pnb)), desempleo[1:]])   # (Δ log PNB, U)
res = bq_svar(Y, p=4, horizon=40, permanent_var_idx=0, cumulate=[0])
res.irf_point[:, 0, 1]   # log PNB (%) tras el choque de demanda: vuelve a 0
res.irf_point[:, 1, 1]   # tasa de desempleo (puntos) tras el choque de demanda: vuelve a 0
```

`cumulate=True` es el valor por defecto y el comportamiento de todas las versiones anteriores (todas las filas acumuladas), correcto cuando todas las columnas están diferenciadas; es incorrecto para el propio sistema $(\Delta y, u)$ de BQ. Una lista de índices, o una máscara booleana de longitud `n`, acumula solo esas filas. La misma transformación se aplica a cada réplica bootstrap antes de tomar los percentiles, de modo que `irf_lower` e `irf_upper` son bandas del objeto pedido. No aplique `cumsum` de nuevo a una fila acumulada, ni `np.diff` a bandas acumuladas para recuperar una variable en niveles (la diferencia de dos cuantiles no es un cuantil): use `cumulate=`.

### Restricciones de signo (`sign_restrictions`)
Implementa el algoritmo de rotación ortogonal QR de Rubio-Ramírez, Waggoner y Zha (2010):

```python
from puremacro.var.identify import sign_restrictions

# Restricciones de signo por choque: {índice del choque: [signo por variable]},
# con +1 (positivo), -1 (negativo) y 0 (sin restricción) en el impacto
restricciones = {
    0: [+1, -1],   # choque de oferta: PIB sube, inflación baja
    1: [-1, -1],   # choque de política monetaria: PIB baja, inflación baja
}

res_signs = sign_restrictions(Y, restrictions=restricciones, p=2, horizon=20, n_draws=5000)
```

Para restricciones de signo combinadas con restricciones contemporáneas de cero exacto, utilice `sign_zero_restrictions` (Arias, Rubio-Ramírez y Waggoner 2018). Para inferencia robusta al conjunto identificado, utilice las bandas de Giacomini y Kitagawa (2021).

### SVAR con Proxy / Instrumentos externos (`proxy_svar`)
Identifica el choque estructural mediante una variable instrumental externa $z_t$ correlacionada con el choque de interés y ortogonal a las demás perturbaciones estructurales (Mertens y Ravn 2013, Stock y Watson 2018):

```python
from puremacro.var.identify import proxy_svar

res_proxy = proxy_svar(Y, p=2, instrument_series=z, horizon=20)
print("Estadístico F de primera etapa:", res_proxy.first_stage_F)
```

### Máxima participación espectral / News (`max_share_svar`)
Identifica choques que maximizan la contribución a la varianza del error de pronóstico de una variable objetivo en horizontes específicos (Barsky y Sims 2011, Francis et al. 2014).

### BVAR con a priori de Minnesota (`puremacro.var.bvar`)
Hay dos posteriores, y difieren en lo que puede hacer el factor de contracción cruzada λ₂. `minnesota_posterior` resuelve una regresión mixta de Theil–Goldberger por ecuación, de modo que respeta λ₂ (0,5 por defecto); devuelve solo la media posterior. `minnesota_gibbs`, `minnesota_optimal_lambda` y la verosimilitud marginal que usa este último emplean la a priori conjugada Normal–Wishart inversa, construida con las observaciones ficticias de Bańbura, Giannone y Reichlin (2010, ec. 5): primeros rezagos propios centrados en 1, desviación típica a priori $\lambda_1 \sigma_i / (k^{\lambda_3} \sigma_j)$. Su covarianza $\Psi \otimes \Omega_0$ da a todas las ecuaciones la misma forma a priori, así que solo admite λ₂ = 1: el artículo impone la a priori "bajo la condición de que ϑ = 1" (ECB WP 966, p. 11). Estas funciones usan λ₂ = 1 por defecto y, si se pide otro valor, emiten un aviso y usan 1; con λ₂ = 1 la media posterior exacta de Gibbs (`A_mean`) coincide con la de `minnesota_posterior`. En puremacro 4.3.0 y anteriores su bloque de observaciones ficticias centraba el primer rezago propio de todas las variables salvo la última en λ₂ (0,5 por defecto) en lugar de 1, de modo que la posterior dependía del orden de las columnas.

---

## 3. Inferencia y bandas de confianza bootstrap

`puremacro.var` proporciona cuatro algoritmos de remuestreo para construir bandas de confianza consistentes:

1. **Bootstrap residual estándar**: Remuestreo aleatorio con reemplazo de los residuos centrados.
2. **Wild Bootstrap**: Multiplica los residuos por perturbaciones Rademacher ($\pm 1$) o normales, preservando la heterocedasticidad condicional.
3. **Block Bootstrap**: Remuestreo de bloques temporales contiguos para preservar dependencia serial residual.
4. **Moving Block Bootstrap**: Remuestreo por bloques superpuestos.

---

## 4. VAR aumentado con factores (FAVAR)

`puremacro.var.favar` aplica el marco de Bernanke, Boivin y Eliasz (2005) para incorporar información macroeconómica de cientos de series temporales en un sistema VAR compacto:

```python
from puremacro.var import favar

res_favar = favar(
    panel_macro,          # DataFrame (T, N) de series informativas
    tasa_politica,        # Serie (T,) de la variable de política
    n_factors=3,
    p=2,
    horizon=20,
    ci=0.90,
)
print(res_favar.summary())
res_favar.plot()
```
