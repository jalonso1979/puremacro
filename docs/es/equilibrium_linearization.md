> 🇬🇧 [English](../equilibrium_linearization.md) · 🇪🇸 Español

# Derivadas comunes del equilibrio

`puremacro.core.EquilibriumSystem` describe ecuaciones cuadradas de equilibrio
\(F(x,\theta)=0\). `linearize_equilibrium` conserva sus jacobianos locales
y resuelve \(F_x\,\partial x/\partial\theta=-F_\theta\).
Esta primera implementación usa matrices densas y un límite configurable del
número de estados. La [continuación de ramas](equilibrium_continuation.md)
reutiliza estas ecuaciones. Los conjuntos de confianza estadísticos siguen
siendo una extensión separada, todavía no implementada en esta API.

## Ecuaciones explícitas

```python
import numpy as np
from puremacro.core import EquilibriumSystem, linearize_equilibrium

system = EquilibriumSystem(
    residual=lambda x, p: np.array([2*x[0] + x[1] - p[0], x[1] - p[0]]),
    state_names=("capital", "price"),
    parameter_names=("technology",),
)
lin = linearize_equilibrium(system, [0., 1.], [1.], branch_id="benchmark")
result = lin.sensitivity()
assert result.valid
print(result.to_frame())  # derivadas: capital=0, price=1
```

Las funciones opcionales `state_jacobian(x, theta)` y
`parameter_jacobian(x, theta)` se contrastan con diferencias centrales de los
residuos. Sin estas funciones, las diferencias centrales generan ambos
jacobianos: son **derivadas numéricas**, aunque la aplicación del teorema
de la función implícita resuelva un sistema lineal.

`lin.jvp(v)` y `lin.vjp(w)` aplican \(F_x\) y \(F_x^T\).
`lin.solve(rhs)` y `lin.solve(rhs, transpose=True)` reutilizan una
factorización LU. Estas operaciones forman matrices densas. Las ecuaciones
se conservan en `lin.system.evaluate(x, theta)`.
Las funciones genéricas deben ser puras y siguen bajo control del usuario;
los adaptadores de modelos conservan copias de sus entradas numéricas.

## Proyección del modelo de crecimiento

```python
from puremacro.vfi import CollocationProblem, linearize_projection

kss = (.36 * .96)**(1 / (1 - .36))
problem = CollocationProblem(
    domain=(.5*kss, 1.5*kss), orders=6, method="euler", beta=.96,
    params={"alpha": .36, "delta": 1., "sigma": 1.},
    options={"tol": 1e-12, "max_iter": 500},
)
solution = problem.solve()
lin = linearize_projection(solution, problem, parameters=("alpha", "beta"))
result = lin.sensitivity()
assert result.valid, result.reasons
policy_derivative = solution.basis.evaluate(kss) @ result.jacobian
```

Los estados son coeficientes de Chebyshev. La derivada de la política mantiene
fijos el estado de evaluación, la base y el dominio. Se admiten las ecuaciones
Euler incorporadas de una dimensión, `beta` y los parámetros declarados
`alpha`, `delta`, `sigma`/`gamma`, `z`/`A`. Se rechazan alias anulados por
su parámetro canónico. Bellman, funciones personalizadas, FEM y splines
requieren otro adaptador o ecuaciones explícitas. Se rechazan recortes físicos
de la política y continuación fuera del dominio en el punto inicial.

## Aiyagari continuo

```python
from puremacro.vfi import solve_aiyagari_continuous, linearize_aiyagari

equilibrium = solve_aiyagari_continuous(
    beta=.95, n_a=30, N_k=60, n_z=3,
    xtol=1e-11, egm_tol=1e-11, tol_ge=1e-6,
)
lin = linearize_aiyagari(equilibrium, parameters=("alpha", "beta", "gamma", "delta"))
result = lin.sensitivity()
assert result.valid, result.reasons
print(result.to_frame().loc[["interest_rate"]])
```

Las incógnitas son consumo en los nodos EGM, probabilidades del histograma y
tasa de interés de equilibrio, en ese orden. Las ecuaciones combinan el punto
fijo de Coleman, transporte estacionario con una ecuación sustituida por
normalización y equilibrio del mercado de capital. Al reconstruir matrices,
el orden es por filas: activos primero, ingreso después.

Se conserva la cadena de ingreso efectiva, incluidas cadenas personalizadas.
Se admiten `alpha`, `beta`, `gamma` y `delta`, con mallas y procesos de ingreso
fijos. La restricción de endeudamiento usa una extensión local del conjunto
activo: el ahorro es cero en nodos restringidos y su ecuación de consumo
impone el presupuesto. Los pasos de diferenciación comprueban el conjunto
activo, las celdas de interpolación y las celdas de la lotería del histograma.
Si el cambio persiste, no se entrega una derivada ordinaria. No se admiten
resultados genéricos de `continuous_stationary_equilibrium`.

Hay `(n_a + N_k) * n_z + 1` estados. El ejemplo tiene 271; las mallas por
defecto del solucionador exceden `max_states=1500`, el límite inicial del
linealizador. Aumentarlo exige considerar almacenamiento denso y costo cúbico
de factorización. Las mallas pequeñas verifican la API, no precisión económica
de la discretización ni escalabilidad a HANK.

## Diagnósticos y alcance

`state_scale`, `parameter_scale` y `residual_scale` son unidades positivas de
coordenadas y ecuaciones. El paso es `step * coordinate_scale`, con
`step=1e-5`, y puede reducirse ocho veces. Los valores singulares se calculan
sobre `diag(1/residual_scale) @ F_x @ diag(state_scale)`.
Al cambiar unidades, hay que ajustar las escalas; los diagnósticos de
condicionamiento dependen de ellas.

Las tolerancias iniciales son `residual_tol=1e-7`, `derivative_tol=1e-3`,
`singular_tol=1e-10` y `condition_limit=1e10`. Los motivos de rechazo incluyen
`not_equilibrium`, `regime_change`, `inconsistent_stencil`,
`derivative_mismatch`, `singular`, `ill_conditioned` y `linear_solve_failed`.
Las sensibilidades rechazadas contienen NaN y no se permite resolver el
sistema lineal rechazado. No hay regularización ni corrección implícita del
equilibrio. Entradas inválidas y excepciones del modelo se propagan; las
diferencias necesitan un entorno bilateral dentro del dominio declarado.

Los arreglos son copias de solo lectura. `snapshot_id` identifica por hash
datos numéricos, matrices y escalas, no código fuente. `branch_id` es una
etiqueta del usuario, no una certificación de ramas. Los pasos finitos y
residuos pequeños no prueban suavidad, unicidad, precisión de malla,
identificación estadística ni inferencia válida. Las
[API IFT especializadas](../vfi_analytic_gradients.md) conservan su comportamiento.

Los resultados ofrecen `summary`, `to_frame`/`to_dataframe`, `to_markdown`,
`to_latex`, `to_typst` y `plot(states=...)`. Los gráficos no llaman a `show` y
admiten como máximo 30 filas de estados.

Ejecute `python -m puremacro.examples.equilibrium_linearization --output /tmp/equilibrium-derivatives`
para generar CSV/JSON con sensibilidades de política, capital agregado y tasa
de interés, además de un caso singular. Las pruebas comparan soluciones
independientes en dos calibraciones de crecimiento y dos procesos de ingreso,
incluyendo toda la distribución Aiyagari y su derivada de masa total cero.
