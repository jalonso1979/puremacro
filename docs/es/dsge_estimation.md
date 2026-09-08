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

Una advertencia sobre el suavizador que conviene conocer: **el primer periodo suavizado es el menos fiable**. `t = 0` queda determinado por la covarianza del estado inicial, mientras que todos los periodos posteriores los determinan los datos; así, sin error de medida el ajuste interior reproduce las observaciones con precisión de máquina, pero el primer periodo hereda la exactitud de la solución de Lyapunov que hay detrás: perturbar `P0` en 1e-8 relativo lo mueve unos 2e-07 y deja `t >= 1` en 2e-16. Pasa `a0`/`P0` explícitamente cuando conozcas de verdad la condición inicial.

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

## Lo que deliberadamente no hace

- **No hay procesador de macros.** `@#define`, `@#for`, `@#if`, `@#include` y `@{...}` lanzan `DynareFeatureError`. Antes se ignoraban, con lo que el archivo se cargaba limpio y se resolvía un *modelo distinto*. Expándelos con `dynare model.mod savemacro` y pasa el archivo expandido. Previsto para la 2.7.0.
- **No hay analizador de expresiones**, de modo que `STEADY_STATE()`, `EXPECTATION()`, `normcdf` y una variable local `#` definida sobre una variable endógena fallan. También para la 2.7.0.
- **Solo primer orden.** Una solución de segundo orden se rechaza en lugar de linealizarse en silencio; el filtro de partículas correspondiente queda para una versión posterior.
- **El muestreador adapta un escalar, no una matriz de covarianzas**, y su adaptación solo actúa cada 100 iteraciones, así que un `burn_in` inferior a 100 no adapta nunca y puede dejar la cadena atascada con una tasa de aceptación del 0 %. Dale al menos unos cientos.
- **`mode_compute` vale `"lbfgs"` por defecto**, y no el mejor `csminwel`, porque cambiarlo cambia todas las posterioris obtenidas hasta ahora.
- **`observation_trends` se aplica quitando la tendencia a los datos**, porque el espacio de estados es invariante en el tiempo por construcción.
- **Las verosimilitudes marginales anteriores a la 2.5.0 no son comparables** ni con estas ni entre sí: la recursión de Kalman partía entonces de un `P0` difuso, lo que en Smets–Wouters vale unos 114 puntos logarítmicos.
