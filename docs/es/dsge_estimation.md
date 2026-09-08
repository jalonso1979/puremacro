> 🇬🇧 [English](../dsge_estimation.md) · 🇪🇸 Español

# Estimar un archivo `.mod` de Dynare

Hasta la versión 2.6.0 puremacro podía *resolver* un archivo `.mod` pero no *estimarlo*. Todas las piezas estaban presentes —distribuciones a priori que leen las convenciones del bloque `estimated_params` de Dynare, un filtro y un suavizador de Kalman con inicialización difusa exacta, un muestreador de Metropolis auditado— y faltaba un único conector: nada convertía una declaración `varobs` en una ecuación de medida. En la práctica eso obligaba a escribir setenta líneas de aritmética de índices por modelo, algo que nadie hace para su propio modelo.

Ese conector ya existe. Un archivo `.mod` que declare `varobs` y `estimated_params` llega directamente a una distribución a posteriori.

```python
import numpy as np
import pandas as pd

from puremacro.dsge import load_mod

MOD = """
var y a;
varexo eps;
parameters rho;
rho = 0.7;
model;
  y = a;
  a = rho * a(-1) + eps;
end;
initval; a = 0; y = 0; end;
shocks; var eps; stderr 0.5; end;
varobs y;
estimated_params;
  rho, beta_pdf, 0.5, 0.2;
  stderr eps, inv_gamma_pdf, 0.5, 2;
end;
"""

model = load_mod(MOD)

# Datos simulados con rho = 0.7, sigma = 0.5, partiendo de la distribución
# estacionaria y no de cero (una serie que arranca lejos de su propia
# distribución estacionaria sesga la estimación a la baja, y eso es una
# propiedad de la muestra, no del estimador).
rng = np.random.default_rng(1)
burn, T = 200, 400
eps = rng.standard_normal(burn + T) * 0.5
a = np.zeros(burn + T)
for t in range(1, burn + T):
    a[t] = 0.7 * a[t - 1] + eps[t]
data = pd.DataFrame({"y": a[burn:]})

res = model.estimate(data, n_draws=400, n_chains=1, burn_in=500, seed=0)
print(res.summary().round(3))
```

`priors` y `varobs` toman por defecto los bloques del propio archivo, de modo que nada se declara dos veces. Pásalos explícitamente para sobrescribirlos.

## Qué debe declarar el archivo

`varobs` nombra las series observadas, en cualquier orden; el `DataFrame` aporta una columna por nombre. `estimated_params` indica qué se estima y cómo.

```text
estimated_params;
  // NOMBRE, VALOR_INICIAL, LB, UB, FORMA_PRIOR, P1, P2, P3, P4, JSCALE;
  alpha, 0.35, beta_pdf, 0.35, 0.02;
  rho, , 0, 1, beta_pdf, 0.5, 0.2;
  stderr eps_a, inv_gamma_pdf, 0.1, 2;
  corr eps_a, eps_b, normal_pdf, 0, 0.2;
end;
```

La gramática es posicionalmente ambigua —`NOMBRE, 0.35, beta_pdf, …` y `NOMBRE, beta_pdf, …` difieren solo en un campo inicial— y `corr a, b` lleva una coma dentro de su objetivo. Ambas cosas se resuelven consumiendo primero la palabra clave `stderr` o `corr` y localizando después el primer campo `*_pdf`: lo que lo precede se lee por longitud y lo que le sigue por posición. Las formas se reconocen sin distinguir mayúsculas, porque los archivos reales escriben `INV_GAMMA_PDF` mientras el manual escribe `inv_gamma_pdf`.

| Forma de Dynare | puremacro | notas |
|---|---|---|
| `beta_pdf` | `BetaPrior` | `P3`/`P4` dan la beta generalizada en `[P3, P4]` |
| `gamma_pdf` | `GammaPrior` | `P3` es la cota inferior |
| `normal_pdf` | `NormalPrior` | `P3`/`P4` truncan |
| `inv_gamma_pdf`, `inv_gamma1_pdf` | `InvGammaPrior` | a priori sobre una desviación típica |
| `inv_gamma2_pdf` | `InvGammaPrior(kind="type2")` | a priori sobre una varianza |
| `uniform_pdf` | `UniformPrior` | `(P1, P2)` como media/desviación, o `(P3, P4)` como cotas |
| `weibull_pdf` | `WeibullPrior` | |

En todas las familias `P1` y `P2` son la media y la desviación típica de la variable **efectiva**, desplazamiento y escala incluidos, siguiendo la convención de `beta_specification` y `gamma_specification` de Dynare. Invertirla reespecifica en silencio un bloque entero.

Los objetos estimados se nombran como los muestra Dynare: `SE_<perturbación>`, `CORR_<s1>_<s2>`, `ME_<observable>`.

```python
ep = model._estimated_params
print([s.name for s in ep.specs])
print({s.name: s.kind for s in ep.specs})
```

Ese `kind` no es decorativo. Solo `"param"` entra en las ecuaciones del modelo, de modo que solo `"param"` obliga a resolverlo de nuevo; una desviación típica de perturbación, una correlación entre perturbaciones y una desviación típica de error de medida se escriben directamente en `Q` y en `H`.

## La ecuación de medida

Un modelo resuelto entrega `v_t = ys + C x_t + D u_t`, de manera que una variable observable carga la innovación **contemporánea**, mientras que un espacio de estados estándar supone ruido de medida independiente de la perturbación del estado. No hay forma exacta de escribir `D u_t` como ruido de medida, así que la innovación se lleva dentro del estado:

$$\alpha_t = \begin{bmatrix} x_t \\ u_t \end{bmatrix},\quad
T = \begin{bmatrix} A & B \\ 0 & 0\end{bmatrix},\quad
R = \begin{bmatrix} 0 \\ I\end{bmatrix},\quad
Z = \begin{bmatrix} C_O & D_O\end{bmatrix},\quad d = ys_O$$

De esa elección se siguen dos cosas. Las observables en tasas de crecimiento no requieren tratamiento especial: `dy = y - y(-1) + ctrend` es una variable declarada, luego es una fila de `(C, D)` como cualquier otra. Y las perturbaciones estructurales suavizadas son literalmente las últimas filas del estado suavizado, de modo que no interviene ningún suavizador de perturbaciones.

```python
from puremacro.dsge import make_state_space_from_varobs

ssm = make_state_space_from_varobs(model, ["y"])
print(ssm.T.shape, ssm.R.shape, ssm.Z.shape, ssm.d)
```

El error de medida es **cero** por defecto, como en Dynare. Pasa `measurement_error={"y": 0.01}` para una desviación típica, o `ridge=` para puro acondicionamiento numérico. Si las observables no pueden generarse con las perturbaciones que el modelo tiene, eso se comunica como el error de especificación que es, antes de muestrear, y no como un mal punto de partida.

## Suavizado y predicción con parámetros calibrados

`smoother` es el `calib_smoother` de Dynare.

```python
sm = model.smoother(data)
print(sm.shocks.head(3).round(4))
print(f"log-verosimilitud {sm.loglik:.3f}")
print(sm.summary())
```

`sm.states`, `sm.shocks`, `sm.smoothed_obs` y `sm.filtered_states` son `DataFrame` indexados como la entrada. `sm.shock_decomposition()` devuelve la descomposición histórica para el mismo modelo y los mismos datos.

```python
fc = model.forecast(horizon=12, data=data, ci=0.90)
print(fc.mean.head(3).round(4))
```

La banda recoge únicamente la incertidumbre de las perturbaciones: los parámetros quedan fijos en los valores con los que se resolvió el modelo.

Una advertencia sobre el suavizador: **el primer periodo suavizado no es reproducible con precisión de máquina entre plataformas**. Llevar la innovación dentro del estado hace que la covarianza predicha sea estructuralmente deficiente de rango (el estado ampliado tiene `n_states + n_shocks` dimensiones impulsadas por `n_shocks` innovaciones), y la ganancia RTS se construye con `numpy.linalg.pinv` de esa matriz. En un RBC pequeño su número de condición ronda 1e15 y su menor valor singular queda justo en el umbral por defecto de `pinv`, de modo que la decisión de rango difiere entre compilaciones de LAPACK. El efecto se limita a `t = 0`: desde `t = 1` el ajuste reproduce los datos con precisión de máquina en todas partes, mientras que el primer periodo puede diferir en ~3e-06 sobre una observable de magnitud 2. Lee `shocks.iloc[0]` teniéndolo en cuenta.

## Comprobar la moda

`estimate_dsge` hacía una única ejecución acotada de L-BFGS-B, y una auditoría en la 2.5.0 la sorprendió devolviendo los valores iniciales mientras informaba convergencia. `mode_compute` selecciona ahora entre `lbfgs` (por defecto), `simplex`, `csminwel`, `cmaes` y `none`.

La forma más barata de ver que una moda declarada no lo es consiste en mirarla parámetro a parámetro:

```python
from puremacro.dsge import mode_check
from puremacro.dsge.priors import log_prior

names = tuple(res.param_names)
priors = model._estimated_params.priors()

def neg_log_post(v):
    """La posteriori, construida con piezas públicas: resolver, filtrar, sumar la priori."""
    theta = dict(zip(names, map(float, v)))
    m = load_mod(MOD, params={"rho": theta["rho"]})
    ll = m.smoother(data, shock_cov=np.array([[theta["SE_eps"] ** 2]])).loglik
    return -(ll + log_prior(theta, priors))

mc = mode_check(
    neg_log_post,
    np.array([res.mode[n] for n in names]), names,
    n_points=11, cov=res.mode_hessian_inv,
)
print(mc.summary())
```

Un corte que mejora alejándose de la moda declarada significa que ese punto no es una moda en esa dirección. `mc.summary()` nombra a los infractores; superar la prueba es necesario pero no suficiente, ya que son cortes de un solo parámetro y nada dicen de las direcciones intermedias.

## Verosimilitud marginal y comparación de modelos

```python
print(f"Laplace  log p(y) = {res.log_mdd('laplace'):.3f}")
```

Dos estimadores, y no son equivalentes. **Laplace** es exacto cuando la posteriori es gaussiana y, fuera de ese caso, vale exactamente lo que valga la moda: compruébala antes. La **media armónica modificada de Geweke** repondera las extracciones con una normal truncada y se evalúa en todos los niveles de truncamiento, devolviendo la dispersión entre niveles en lugar de ocultarla: el estimador debería ser invariante al truncamiento, de modo que una oscilación superior a un punto logarítmico indica que no ha convergido, y así lo declara.

`model_comparison` puntúa todos los modelos con el *mismo* estimador y se niega cuando alguno no puede aportarlo. Un valor de Laplace y uno de media armónica no están en pie de igualdad, y mezclarlos en silencio es la vía por la que un factor de Bayes se convierte en ficción.

## Diagnóstico del modelo y análisis de residuos

Antes de estimar un modelo o confiar en simulaciones de política, puremacro proporciona tres puntos de entrada de diagnóstico que igualan y amplían las herramientas de Dynare: `check()`, `resid()` y `model_diagnostics()`.

### Determinación de Blanchard-Kahn y valores propios (`check()`)

`model.check()` evalúa el espectro de valores propios generalizados del haz de matrices acompañante, clasifica las raíces en categorías estables, con raíz unitaria y explosivas, y verifica las condiciones de orden y rango de Blanchard-Kahn:

```python
table = model.check()
print(table.summary())
```

Cuando la determinación falla (indeterminación o explosividad), `check()` va más allá de un simple recuento escalar: calcula los vectores propios generalizados asociados a las raíces problemáticas y aísla las principales cargas por variable, señalando con precisión qué variables económicas originan el fallo:

```text
EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS (qz_criterium=1.000001)
========================================================================
Status       : 1 explosive roots for 2 forward-looking variables; missing root at |lambda| = 0.9421 loads chiefly on pi, y (indeterminacy).
Determinacy  : DETERMINACY FAILED
Forward vars : 2
Explosive    : 1
Stable       : 3
Unit roots   : 0
```

Visualización del espectro frente al círculo unitario complejo:

```python
fig = table.plot()  # grafica raíces estables, unitarias y explosivas con el círculo unitario
```

### Residuos de estado estacionario (`resid()`)

`model.resid()` evalúa las ecuaciones dinámicas de equilibrio en el estado estacionario, $f(ss, ss, ss, 0)$, y devuelve una `pd.Series` de residuos ordenados de forma descendente por su magnitud absoluta:

```python
resids = model.resid()
print(resids.head(5).round(6))
```

Las etiquetas y etiquetas identificadoras del archivo `.mod` se preservan, lo que permite aislar de inmediato las ecuaciones con residuos no nulos (por ejemplo, al diagnosticar discrepancias de calibración).

### Diagnósticos estructurales del modelo (`model_diagnostics()`)

`model.model_diagnostics()` ejecuta comprobaciones estructurales estáticas y dinámicas exhaustivas:
- **Análisis de rango del jacobiano estático**: evalúa $\text{rango}(J_{\text{estático}})$ frente al número de variables.
- **Ecuaciones y variables colineales**: emplea la Descomposición en Valores Singulares (SVD) para identificar subconjuntos exactos de ecuaciones colineales y variables no restringidas.
- **Matriz de incidencia numérica**: evalúa la incidencia de unión a lo largo de perturbaciones de vecindad para detectar variables no utilizadas y ecuaciones redundantes.
- **Regularidad del haz dinámico de matrices**: verifica que $\det(B - \lambda A) \not\equiv 0$.
- **Singularidad estocástica**: señala si el número de variables observadas declaradas en `varobs` supera la suma de choques y errores de medida.

```python
diag = model.model_diagnostics()
print(diag.summary())

# Graficar la matriz de incidencia booleana
fig = diag.plot()
```

## Resolución robusta del estado estacionario por bloques triangulares

El cómputo de estados estacionarios deterministas en modelos DSGE de mediana y gran escala falla con frecuencia al utilizar solucionadores de Newton multidimensionales ingenuos. Puremacro incorpora un motor de descomposición en bloques triangulares (`puremacro.dsge.steady`):

1. **Emparejamiento bipartito de Hopcroft-Karp**: halla un emparejamiento de máxima cardinalidad entre ecuaciones y variables en tiempo $O(|E|\sqrt{|V|})$.
2. **Descomposición de singularidad de Dulmage-Mendelsohn**: cuando no existe un emparejamiento completo, descompone el grafo bipartito de incidencia en subconjuntos sobredeterminados, subdeterminados y bien determinados, generando un error informativo `StructuralSingularityError` que nombra explícitamente los subconjuntos problemáticos:
   ```python
   from puremacro.dsge.steady import StructuralSingularityError

   try:
       ss, info = steady(equations, variables, guess, params)
   except StructuralSingularityError as err:
       print("Ecuaciones sobredeterminadas :", err.overdetermined_equations)
       print("Variables subdeterminadas    :", err.underdetermined_variables)
   ```
3. **Componentes fuertemente conexas de Tarjan (SCC)**: descompone el grafo de dependencias inducido por el emparejamiento en sub-bloques topológicos resueltos en secuencia. Las ecuaciones escalares aisladas se resuelven con métodos unidimensionales (Brent / secante), mientras que los bloques acoplados emplean solucionadores vectoriales.

### Menú de algoritmos y continuación por homotopía

El parámetro `solve_algo` permite seleccionar entre distintos algoritmos numéricos:
- `"block"` (por defecto): solucionador recursivo en bloques triangulares.
- `"hybr"`: método híbrido de Powell (MINPACK).
- `"lm"`: mínimos cuadrados no lineales de Levenberg-Marquardt.
- `"df-sane"`: método espectral libre de derivadas para sistemas de gran escala.

Cuando los métodos no lineales fallan debido a un punto inicial alejado, la **continuación por homotopía adaptativa** recorre una trayectoria paramétrica con bisección automática de pasos:

```python
from puremacro.dsge.steady import steady

# Continuación paramétrica desde alpha=0.20 hasta alpha=0.36
ss, info = steady(
    equations,
    variables,
    guess,
    params={"alpha": 0.36, "beta": 0.99},
    solve_algo="block",
    homotopy={"alpha": (0.20, 0.36)},
    homotopy_steps=10,
)
```

## Análisis de identificación de parámetros

Siguiendo a Iskrev (2010) y Ratto (2011), `model.identification()` evalúa si los parámetros estructurales pueden recuperarse unívocamente a partir de las variables observadas declaradas en `varobs`:

```python
ident = model.identification(varobs=["y", "pi", "i"], lags=2)
print(ident.summary())
```

El procedimiento evalúa dos jacobianos fundamentales:
1. **$J_1$ (Jacobiano de la solución en espacio de estados)**: $\frac{\partial \text{vec}(T, R, Q, Z, H)}{\partial \theta}$.
2. **$J_2$ (Jacobiano de momentos teóricos)**: $\frac{\partial m(\theta)}{\partial \theta}$, donde $m(\theta)$ apila las autocovarianzas del modelo hasta el orden `lags`.

El objeto `IdentificationResult` resultante reporta:
- **Deficiencia de rango**: determina si $J_1$ y $J_2$ tienen rango completo y la dimensión de sus espacios nulos.
- **Combinaciones de parámetros en el espacio nulo**: combinaciones lineales exactas que generan el espacio nulo (por ejemplo, `0.7071 * theta1 - 0.7071 * theta2 = 0`), revelando parámetros redundantes o proporcionales.
- **Colinealidad multivariada ($R^2$)**: $R^2$ resultante de la regresión de cada columna del jacobiano sobre las demás; valores cercanos a $1.0$ identifican parámetros colineales.
- **Fuerza de identificación**: métricas de sensibilidad y fuerza normalizada de Ratto.

```python
# Graficar colinealidad R^2 frente a fuerza de identificación
fig = ident.plot()
```

### Comprobación previa de identificación en la estimación

Para evitar el lanzamiento de cadenas MCMC computacionalmente costosas en modelos no identificados, pase `check_identification=True` a `model.estimate()`:

```python
# Emite una advertencia informativa si existen parámetros deficientes de rango
res = model.estimate(data, check_identification=True)

# O use check_identification="raise" para abortar inmediatamente ante deficiencias de rango
# res = model.estimate(data, check_identification="raise")
```

## Regímenes de política óptima y reglas simples óptimas

Puremacro ofrece soporte integral para el diseño de política monetaria y macroprudencial en tres regímenes estándar:

### Reglas simples óptimas (`osr()`)

`model.osr()` optimiza los coeficientes de respuesta en reglas de política simples (como reglas de Taylor) para minimizar una pérdida cuadrática sobre las varianzas teóricas:
$$L(\gamma) = \sum_i w_i \text{Var}(y_i; \gamma)$$
sujeta a la condición de determinación de Blanchard-Kahn. Una superficie de penalización numérica continua garantiza la evaluación suave del gradiente cuando los parámetros candidatos ingresan en regiones de indeterminación:

```python
osr_res = model.osr(
    rule_params=["phi_pi", "phi_y"],
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.5},
)
print(osr_res.summary())

# Gráfico de barras agrupadas comparando varianzas entre la regla base y la óptima
fig = osr_res.plot()
```

### Política discrecional (`discretionary_policy()`)

`discretionary_policy()` calcula el equilibrio discrecional markoviano perfecto y temporalmente consistente (Dennis 2007) mediante iteración de funciones de política sobre las matrices de respuesta del sector privado y del banco central:

```python
from puremacro.dsge import discretionary_policy

disc_res = discretionary_policy(
    model,
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.25},
    instruments=["i"],
    beta=0.99,
)
print(disc_res.summary())
```

### Compromiso lineal-cuadrático (`lq_commitment()`)

`lq_commitment()` resuelve la política óptima bajo compromiso desde la perspectiva intemporal ($\lambda_{-1} = 0$). Plantea el lagrangiano del planificador sobre las condiciones de equilibrio con expectativas racionales, ampliando el vector de estado con multiplicadores de Lagrange hacia adelante:

```python
from puremacro.dsge import lq_commitment

commit_res = lq_commitment(
    model,
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.25},
    instruments=["i"],
    beta=0.99,
)
print(commit_res.summary())

# Acceder a los multiplicadores de política y graficar funciones de respuesta a impulsos
augmented_model = commit_res.linear_model
print("Multiplicadores:", commit_res.multipliers)
fig = commit_res.plot(periods=16)
```

## Contrato unificado de presentación de resultados

Todos los objetos de resultados DSGE (`EigenvalueTable`, `ModelDiagnosticsResult`, `IdentificationResult`, `OSRResult`, `PolicyResult`) cumplen el contrato uniforme de presentación de puremacro:

| Método | Tipo devuelto | Descripción |
|---|---|---|
| `.to_frame()` | `pandas.DataFrame` | Representación tabular canónica de los resultados |
| `.summary()` | `str` | Salida limpia y legible en terminal |
| `.plot()` | `matplotlib` Figure/Axes | Diagnóstico visual con estilo listo para publicación |
| `.to_markdown()` | `str` | Tabla en formato GitHub-flavored Markdown |
| `.to_latex()` | `str` | Tabla LaTeX `tabular` lista para publicación con caracteres especiales escapados |
| `.to_typst()` | `str` | Tabla `#table(...)` para composición científica moderna en Typst |

## Lo que deliberadamente no hace

- **No hay procesador de macros.** `@#define`, `@#for`, `@#if`, `@#include` y `@{...}` lanzan `DynareFeatureError`. Antes se ignoraban, con lo que el archivo se cargaba limpio y se resolvía un *modelo distinto*. Expándelos con `dynare model.mod savemacro` y pasa el archivo expandido. Previsto para la 2.7.0.
- **No hay analizador de expresiones**, de modo que `STEADY_STATE()`, `EXPECTATION()`, `normcdf` y una variable local `#` definida sobre una variable endógena fallan. También para la 2.7.0.
- **Solo primer orden.** Una solución de segundo orden se rechaza en lugar de linealizarse en silencio; el filtro de partículas correspondiente queda para una versión posterior.
- **El muestreador adapta un escalar, no una matriz de covarianzas**, y su adaptación solo actúa cada 100 iteraciones, así que un `burn_in` inferior a 100 no adapta nunca y puede dejar la cadena atascada con una tasa de aceptación del 0 %. Dale al menos unos cientos.
- **`mode_compute` vale `"lbfgs"` por defecto**, y no el mejor `csminwel`, porque cambiarlo cambia todas las posterioris obtenidas hasta ahora.
- **observation_trends se aplica quitando la tendencia a los datos**, porque el espacio de estados es invariante en el tiempo por construcción.
- **Las verosimilitudes marginales anteriores a la 2.5.0 no son comparables** ni con estas ni entre sí: la recursión de Kalman partía entonces de un `P0` difuso, lo que en Smets–Wouters vale unos 114 puntos logarítmicos.

## Cuadernos de demostración

Para tutoriales interactivos completos con visualización y diagnósticos:

- **`notebooks/42_dsge_bayesian_estimation_and_diagnostics.py`** (y edición en español `42_dsge_bayesian_estimation_and_diagnostics_es.py`): Cuaderno insignia de estimación bayesiana que reproduce el flujo de trabajo completo de Smets-Wouters (2007). Demuestra la búsqueda de la moda con múltiples algoritmos comparando `"lbfgs"`, `"csminwel"` y `"cmaes"`; diagnósticos visuales de la moda mediante cortes de curvatura `mode_check`; muestreo MCMC de Metropolis-Hastings con métricas de convergencia de Gelman-Rubin; extracción de estados y choques con el suavizador de Kalman y descomposición histórica de choques estructurales; pronósticos fuera de muestra y condicionales con gráficos de abanico (fan charts); y cálculo de la densidad marginal de los datos (aproximación de Laplace frente a la media armónica modificada de Geweke) con comparación formal de modelos bayesianos.
- **`notebooks/41_dynare_frontier_showcase.py`** (y edición en español `41_dynare_frontier_showcase_es.py`): Demuestra capacidades de frontera de Dynare usando las funciones nativas 2.6.0 `.smoother()` y `.estimate()` sobre `sw07_pfeifer.mod` con datos trimestrales de EE.UU. empaquetados (`_sw07_data.csv`), restricciones ocasionalmente activas ZLB con OccBin y transiciones de Ramsey con previsión perfecta.

## Suite de replicación: familia `dsge_estimation`

El módulo `puremacro.replication` proporciona verificación automatizada de los principales resultados empíricos publicados en la literatura académica. La familia `dsge_estimation` verifica:

- **`dsge_estimation.sw07_log_posterior_at_mode`**: Evalúa el log-posteriori exacto de Kalman en la moda posterior sobre el conjunto de datos de EE.UU. 1966–2004 (`_sw07_data.csv`) bajo la inicialización de covarianza estacionaria de Lyapunov (`_stationary_init`). Objetivo: `-1673.72` (`Tol.TIGHT`).
- **`dsge_estimation.sw07_laplace_marginal_data_density`**: Evalúa la aproximación de Laplace a la densidad marginal de los datos a partir del hessiano inverso en la moda. Objetivo: `-1686.09` (`Tol.TIGHT`).
- **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`**: Evalúa la media armónica modificada de Geweke (1999) a través de los parámetros de truncamiento `[0.1, 0.3, 0.5, 0.7, 0.9]` y comprueba la consistencia entre niveles de truncamiento (dispersión $< 2.5$ puntos logarítmicos).
- **`dsge_estimation.sw07_structural_parameters_mode`**: Verifica los parámetros estructurales clave de la moda en la Tabla 1 de Smets y Wouters (2007) (`csadjcost`, `csigma`, `chabb`, `csigl`, `cprobp`, `cfc`, `crr`, `crdy`, `ctrend`).
