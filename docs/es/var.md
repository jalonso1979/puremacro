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
donde $C(1)$ se restringe a ser triangular inferior.

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
