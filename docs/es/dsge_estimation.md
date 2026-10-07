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

Llevar la innovación dentro del estado hace que la covarianza predicha sea estructuralmente deficiente de rango (el estado ampliado tiene `n_states + n_shocks` dimensiones impulsadas por `n_shocks` innovaciones), por lo que `kalman_smoother` usa la recursión hacia atrás de Durbin–Koopman, que solo invierte la covarianza de las innovaciones `F_t` y nunca la covarianza predicha del estado. Hasta la 4.6.0 usaba la ganancia de Rauch–Tung–Striebel construida con `numpy.linalg.pinv` de esa matriz; en un RBC pequeño su menor valor singular quedaba justo en el umbral por defecto de `pinv`, de modo que el primer periodo suavizado difería hasta en ~3e-06 entre compilaciones de LAPACK. Sin error de medida, las observables ajustadas reproducen ahora los datos en todos los periodos, incluido el primero.

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
2. **Descomposición de singularidad de Dulmage-Mendelsohn**: descompone el grafo bipartito de incidencia en subconjuntos sobredeterminados, subdeterminados y bien determinados. Cuando no existe un emparejamiento completo en el punto inicial, `steady()` ejecuta una resolución `hybr` del sistema completo y vuelve a comprobar el emparejamiento en el punto que alcanza. Si ahí el emparejamiento es completo, la carencia era local al punto inicial y se devuelve la raíz (`info["fallback_reason"] == "incomplete_matching_at_guess"`). En caso contrario, el modelo es estructuralmente singular y su estado estacionario no es localmente único:

   - Si toda ecuación sobredeterminada se reduce a `0 = 0` (la forma estática de una ley de movimiento con raíz unitaria, como `z = z(-1) + e`), se devuelve el punto con un `StructuralSingularityWarning` que nombra las variables libres, e `info["structurally_singular"] = True`, como hace Dynare con los modelos con raíz unitaria.
   - Cualquier otra estructura (ecuaciones estructuralmente duplicadas, es decir, un subconjunto de ecuaciones en el que intervienen menos variables que ecuaciones, o una variable que no entra en ninguna ecuación) lanza `StructuralSingularityError`. El error nombra los subconjuntos sobredeterminado y subdeterminado y lleva la raíz comprobada en `hybr_point`. Para aceptar el punto, pasa `allow_singular=True` a `steady()`, `build()`, `build_dynare()` o `load_mod()`, o envuelve la llamada en `with allow_structural_singularity():` tras `from puremacro.dsge.steady import allow_structural_singularity`. El ajuste del gestor de contexto solo afecta al hilo actual (o a la tarea asyncio actual).
   - Con `allow_singular=False`, también las raíces unitarias lanzan el error.
   - Un sistema inconsistente lanza el error en todos los modos.

   La comprobación es estructural, como `dmperm` en Dynare: no detecta ecuaciones numéricamente colineales con incidencia completa (por ejemplo, `x + y = 2` y `2x + 2y = 4`). Cuando la pasada por bloques deja un residuo mayor que `tol`, se ejecuta un pulido `hybr` del sistema completo, que queda registrado en `info["full_system_polish"]` y en un mensaje de log de nivel INFO. Un modelo creado con `build()`, `build_dynare()` o `load_mod()` conserva ese diccionario `info` en `model.steady_state_info` (`None` cuando el estado estacionario se proporcionó en lugar de calcularse).

   ```python
   # requires: standalone snippet
   from puremacro.dsge.steady import StructuralSingularityError, steady

   try:
       ss, info = steady(equations, variables, guess, params)
   except StructuralSingularityError as err:
       print("Ecuaciones sobredeterminadas :", err.overdetermined_equations)
       print("Variables subdeterminadas    :", err.underdetermined_variables)
       print("Raíz comprobada              :", err.hybr_point)
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
# requires: standalone snippet
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
# requires: standalone snippet
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
# requires: standalone snippet
# Graficar colinealidad R^2 frente a fuerza de identificación
fig = ident.plot()
```

### Dónde se evalúan los jacobianos

La identificación local es una propiedad del jacobiano en un único vector de parámetros $\theta_0$: la condición de rango es $\operatorname{rank} J(\theta_0) = n_\theta$. Todas las columnas de $J_1$, $J_2$, $J_H$ y $J_S$ se calculan en el mismo $\theta_0$. El modelo se resuelve una sola vez en $\theta_0$, y la columna $j$ perturba solo $\theta_j$, con los demás parámetros fijos en su valor de $\theta_0$. $\theta_0$ es la calibración del modelo y la covarianza de perturbaciones declarada, sustituidas primero por `fixed_params` y después por el valor de cada parámetro analizado:

| `params` | Valor de cada parámetro analizado |
|---|---|
| un diccionario `{nombre: valor}` (o `p_dict=`) | el valor dado |
| una lista de nombres | su valor actual: la calibración para un parámetro estructural, $\sqrt{\Sigma_{u,ii}}$ para `SE_<perturbación>` (1.0 si no se declara covarianza), la correlación declarada para `CORR_<s1>_<s2>` (0.0 si no la hay), la entrada de `measurement_error` para `ME_<observable>` (0.0 si no la hay) |
| `EstimatedParams` / `EstimatedParamSpec` | el `start` de la especificación: su `INITVAL` si se declara; si no, su media a priori |
| omitido, en un modelo `.mod` con `estimated_params` | cada parámetro declarado en su `start`, como en la fila anterior |
| omitido, en cualquier otro modelo | cada parámetro calibrado en su calibración |

Una lista de nombres que declara el bloque `estimated_params` también toma el `start` de cada especificación.

`fixed_params` (valores estructurales fijados en $\theta_0$ pero no analizados) y `measurement_error` (desviaciones típicas fijas del error de medida, por observable) son argumentos tanto del método `model.identification()` como de la función `puremacro.dsge.identification(model, ...)`.

**En qué difiere de Dynare.** El comando `identification` de Dynare usa por defecto `parameter_set = prior_mean`. El valor por defecto de puremacro coincide con él solo cuando el bloque no declara ningún `INITVAL`. Para elegir el punto explícitamente, pásalo:

```python
# requires: standalone snippet
ep = model._estimated_params
at_prior_mean = model.identification(params={s.name: s.prior.mean for s in ep.specs})
at_calibration = model.identification(params={n: model._params[n] for n in ["kappa", "rho_u"]})
```

Los valores `SE_` son desviaciones típicas de las innovaciones ($Q_{ii} = \sigma_i^2$). Los valores `CORR_` son correlaciones ($Q_{ij} = \rho_{ij}\sigma_i\sigma_j$, con las desviaciones típicas de $\theta_0$). Los valores `ME_` son desviaciones típicas del error de medida ($H_{ii} = \sigma^2$). Es la misma correspondencia que `estimate()` aplica a cada extracción. Una covarianza declarada fuera de la diagonal que ningún parámetro `CORR_` nombra permanece fija *como covarianza* cuando se mueve un `SE_`. Dynare, en cambio, mantiene fija una correlación declarada como correlación.

Las columnas estructurales son diferencias centradas con paso $h = \max(10^{-5}, 10^{-4}|\theta_j|)$. La excepción se da cuando $\theta_j \pm h$ cruza una cota declarada o una resolución falla. La columna es entonces una diferencia unilateral de segundo orden, tomada del lado cuyos puntos quedan dentro de las cotas, lejos de la cota activa. Un paso nunca cruza una cota declarada. Las columnas de perturbaciones y de error de medida son analíticas.

**Conjunto de estados de un modelo `.mod` / `build_dynare`.** `build_dynare` detecta numéricamente las variables predeterminadas: una variable es un estado cuando su retardo tiene un coeficiente no nulo en la calibración. Un retardo cuyo coeficiente está calibrado en 0, como `crhoms`, `crhopinf`, `crhow`, `cmap` y `cmaw` en SW07, no es por tanto un estado del modelo calibrado. Todas las resoluciones de un mismo análisis de identificación usan el mismo conjunto de estados: los estados del modelo, más los retardos activos en $\theta_0$, más los retardos que activa cualquier parámetro estructural analizado al moverse. Cuando ese conjunto difiere del propio del modelo, el modelo se vuelve a resolver con él, incluso en la calibración. Los estados adicionales no cambian los momentos ni los espectros de las observables; sus derivadas, sí. Los modelos creados con `build()` conservan los estados que declaraste.

`prior_mc=N` repite el análisis de rango en `N` extracciones de $\theta_0$ a partir de las distribuciones a priori, truncadas a las cotas declaradas. Cada extracción se analiza en su propio punto completo. Las extracciones en las que el modelo no puede resolverse se omiten y se cuentan en `prior_mc_results["n_failed"]`; `n_draws` cuenta las extracciones analizadas. Si no puede resolverse ninguna, las tasas valen NaN y se emite un `RuntimeWarning`.

### Comprobación previa de identificación en la estimación

Para evitar el lanzamiento de cadenas MCMC computacionalmente costosas en modelos no identificados, pase `check_identification=True` a `model.estimate()`. La comprobación se ejecuta en el punto del que parte la estimación: cada parámetro estimado en su `start` (su `INITVAL`; si no, su media a priori), `fixed_params` y `measurement_error` tal como se pasan, y la calibración para todo lo demás. Un nombre de `fixed_params` que no sea un parámetro del modelo hace que `estimate()` lance `ValueError`, con la comprobación o sin ella.

```python
# requires: standalone snippet
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
sujeta a la condición de determinación de Blanchard-Kahn. En una regla candidata indeterminada, o en la que el modelo no puede resolverse, la pérdida se sustituye por una penalización continua, $10^8 + 10^4\lVert\gamma - \gamma_0\rVert^2$, de modo que los optimizadores sin gradiente (Nelder–Mead, Powell) se contraen alejándose de esa región:

```python
# requires: standalone snippet
osr_res = model.osr(
    rule_params=["phi_pi", "phi_y"],
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.5},
)
print(osr_res.summary())

# Gráfico de barras agrupadas comparando varianzas entre la regla base y la óptima
fig = osr_res.plot()
```

**Tolerancias y precisión.** `model.osr()` y la función `puremacro.dsge.osr(model, ...)` aceptan `xatol` (por defecto `1e-8`, una tolerancia absoluta sobre los coeficientes, en sus propias unidades), `fatol` (por defecto $10^{-12}$ veces la pérdida inicial, como mínimo $10^{-12}$) y `options` (que se pasa en último lugar a `scipy.optimize.minimize`, por ejemplo `{"xtol": ..., "ftol": ...}` para Powell). Una búsqueda que compara valores de la pérdida localiza los coeficientes, y la asignación que implican, solo hasta un error relativo de unos $\sqrt{2\varepsilon/c}$, donde $\varepsilon$ es la precisión relativa de la pérdida y $c$ su curvatura normalizada en el óptimo. Eso son unos $10^{-8}$ en un problema bien escalado, y un error mayor con una pérdida plana o mal escalada. Como `xatol` es absoluto, escálalo con los coeficientes. `xatol=1e-4, fatol=1e-4` reproduce los valores por defecto del propio Nelder–Mead de SciPy, que `osr` usó hasta la 4.3.0 inclusive. Hay ejemplos medidos en [Frontera DSGE, §1.6](dsge_phase_c.md).

`loss_opt` se recalcula resolviendo de nuevo el modelo en los coeficientes devueltos. Si esa resolución falla, o la regla es indeterminada en ese punto, `loss_opt` vale NaN con un `RuntimeWarning` y `optimal_model` es `None`; `loss_initial` vale NaN, también con una advertencia, cuando no pueden evaluarse los momentos de partida. Ninguno de los dos es nunca un valor provisional ni el valor de penalización del optimizador.

### Política discrecional (`discretionary_policy()`)

`discretionary_policy()` calcula el equilibrio discrecional markoviano perfecto y temporalmente consistente (Dennis 2007) mediante iteración de funciones de política sobre las matrices de respuesta del sector privado y del banco central:

```python
# requires: standalone snippet
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

`lq_commitment()` resuelve la política óptima bajo compromiso. Plantea el lagrangiano sobre las condiciones de equilibrio con expectativas racionales y amplía el vector de estado con los multiplicadores de Lagrange. La ley de movimiento que devuelve es la misma para el plan de Ramsey elegido en $t_0$ y para la regla de perspectiva atemporal; ambos difieren solo en el multiplicador inicial. El plan de Ramsey fija $\lambda_{-1} = 0$ sea cual sea la historia, mientras que la perspectiva atemporal usa el multiplicador implícito en la política pasada. Las respuestas al impulso y `conditional_loss` parten del estado estacionario, con $\lambda_{-1} = 0$, donde ambos coinciden. `loss` promedia sobre la distribución estacionaria, es decir, evalúa la regla atemporal en promedio, y vale NaN con un `RuntimeWarning` cuando no existe distribución estacionaria. El argumento `timeless` nunca cambió el resultado y está obsoleto. Véase [Frontera DSGE, §1.3](dsge_phase_c.md).

```python
# requires: standalone snippet
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

- **Las directivas de macro se expanden, nunca se ignoran.** Desde la 2.7.0, el preprocesador propio de puremacro expande `@#define`, `@#for`, `@#if`, `@#include` y `@{...}` antes de analizar el archivo ([Cuaderno de bocetos DSGE, §4b](dsge_build.md)). En la 2.6.0 lanzaban `DynareFeatureError`; antes se ignoraban, con lo que el archivo se cargaba limpio y se resolvía un *modelo distinto*.
- **Los parámetros se vuelven a leer solo donde Dynare los vuelve a leer.** El lector de `.mod` analiza `STEADY_STATE()`, `normcdf` y las variables locales `#` del modelo, definidas sobre parámetros o sobre variables endógenas. Las locales se sustituyen simbólicamente y nunca se congelan en la calibración, de modo que un parámetro estimado que solo entra en el modelo a través de ellas (el `constebeta` de SW07, vía `cbeta`, `cbetabar`, `cr`, `conster`, ...) mueve la verosimilitud en cada extracción; hasta la 4.3.0 inclusive esas locales se reducían a números al cargar el archivo y la verosimilitud de `constebeta` era exactamente plana. Las asignaciones de primer nivel como `cbeta = 1/(1+constebeta/100);`, fuera del bloque del modelo, se evalúan una sola vez al leer el archivo y **no** siguen a las extracciones —igual que en Dynare—, así que escribe una cantidad derivada de parámetros como una local `#`. `EXPECTATION(k)(...)` se analiza sintácticamente, pero su residuo compilado lanza `NameError`.
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

El módulo `puremacro.replication` contrasta resultados principales publicados con los cálculos del propio puremacro. La familia `dsge_estimation` ejecuta el modelo de Smets y Wouters (2007) programado a mano (`puremacro.dsge.smets_wouters`) sobre los datos trimestrales de EE.UU. 1966T1–2004T4 incluidos en el paquete (`_sw07_data.csv`, reconstruidos desde FRED con las definiciones del apéndice de datos de SW07). Su fixture, `puremacro/replication/data/sw07_parity_seed0_200draws.npz`, se distribuye como datos del paquete y contiene una moda posterior optimizada (L-BFGS-B desde dos puntos de partida, refinada con pasos de Newton), la inversa del hessiano en ella y 200 extracciones submuestreadas de Metropolis de paseo aleatorio. Tiene cuatro casos, y solo el primero se compara con una cifra publicada:

- **`dsge_estimation.sw07_structural_parameters_mode`** (objetivo publicado): la moda optimizada frente a la columna **Mode** (moda posterior) de las Tablas 1a y 1b de SW07 (ECB WP 722, pp. 35–36 del PDF), no frente a la columna Mean (media). Comprueba 13 parámetros: `csadjcost`, `csigma`, `chabb`, `csigl`, `cprobp`, `cfc`, `crr`, `crdy`, `ctrend` y los parámetros de los procesos de margen `crhopinf`, `cmap`, `crhow`, `cmaw`. Los 13 quedan a menos del 6.2 % de las modas publicadas (por ejemplo, $\varphi$ 5.694 frente a 5.48, $\sigma_c$ 1.405 frente a 1.39, $\mu_p$ 0.703 frente a 0.74); el caso usa `Tol.COARSE` (25 %) porque los datos son la versión actual de FRED, no los ficheros de SW de 2006.
- **`dsge_estimation.sw07_log_posterior_at_mode`**, **`dsge_estimation.sw07_laplace_marginal_data_density`**, **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`** (valores de regresión de puremacro, **no son cifras publicadas**): el log-posteriori en la moda, `-822.04`, recalculado en vivo; la log densidad marginal de Laplace, `-902.83`, a partir de ese valor y de la inversa del hessiano almacenada; y la media armónica modificada de Geweke (1999) sobre las 200 extracciones almacenadas, `-908.03`, con una dispersión de `2.66` puntos logarítmicos entre los niveles de truncamiento 0.1–0.9. Con solo 200 extracciones esa dispersión supera el umbral de convergencia de un punto logarítmico de `harmonic_mean_mdd`, así que este caso fija la salida del estimador, no una estimación convergida. Los tres casos usan `Tol.TIGHT` (2 % relativo, unos ±16 a ±18 puntos logarítmicos con estos valores), de modo que detectan cambios grandes en el modelo, los datos, las distribuciones a priori o los estimadores; la suite de tests fija el log-posteriori de la moda almacenada con una tolerancia de $10^{-6}$. SW07 no publican el log-posteriori en la moda, y su verosimilitud marginal (Tabla 2: −905.8) se calcula sobre 1966–2004 con 1956:1–1965:4 como muestra de entrenamiento. Desde 4.6.0 `sw07_laplace_mdd` reproduce ese cálculo sobre los datos de los autores (`presample=40`, `lik_init="diffuse"`): −932.3, y −921.4 con las opciones del `.mod` de replicación de los autores; en la moda de los propios autores puremacro coincide con Dynare 8 hasta 0.65 puntos logarítmicos (−840.81 frente a −841.46), y el propio Dynare 8 da −923.1 y no −905.8 sobre los archivos públicos. La cifra publicada no es, pues, reproducible a partir de esos archivos y no es un objetivo comparable; véase la [Galería de replicación](replication.md).

Hasta la 4.3.0 inclusive, esta página daba `-1673.72` y `-1686.09` como objetivos del log-posteriori y de Laplace, y describía el caso de la moda como una comprobación de la Tabla 1 en la moda posterior. Eran salidas de puremacro, no resultados de SW07, y el caso de la moda comparaba la mejor de 200 extracciones MCMC con valores de la columna Mean. La tabla de comparación completa está en la [Galería de Replicación](replication.md).
