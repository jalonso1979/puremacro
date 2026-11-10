> Español · [English](../hank_second_order.md)

# HANK experimental de segundo orden con un activo

`solve_hank_second_order` añade términos cuadráticos del estado y correcciones
por la varianza anticipada a una política HANK pequeña con un activo. Conserva
todos los controles marginales de los hogares y la base completa de distribución
sujeta a las restricciones de masa, ingreso y mercado de bonos. Las bases
reducidas, dos activos, capital productivo, precisión global y bienestar quedan
fuera de esta implementación.

```python
import numpy as np
from puremacro.models import (
    calibrate_one_asset_hank, solve_hank_reduced_policy,
    solve_hank_second_order,
)

ss = calibrate_one_asset_hank(n_a=20, n_e=2, a_max=120.)
linear = solve_hank_reduced_policy(
    ss, distribution_rank=None, marginal_rank=None, shock_std=0.0005,
)
policy = solve_hank_second_order(linear)
x = policy.state(monetary_shock=0.0005)
audit = policy.audit(x, tolerance=1e-5, quadrature_order=3)
assert audit.accepted
assert audit.diagnostics["mass_error"] < 1e-10
assert audit.diagnostics["budget_error"] < 1e-8
assert audit.diagnostics["equation_error"] < 4e-7
assert audit.diagnostics["euler_kkt_is_grid_diagnostic"]
```

## Cronología y aproximación

El estado contiene las coordenadas de distribución, la tasa nominal heredada
y el choque monetario actual, observado después de la innovación corriente.
Para el estado `x`, la política condicional es

\[
y = Gx + \tfrac12 G_{xx}[x,x] + \tfrac12 G_{\sigma\sigma}\sigma^2,
\qquad E[x'\mid x] = Hx + \tfrac12 H_{xx}[x,x]
                       + \tfrac12 H_{\sigma\sigma}\sigma^2.
\]

La siguiente innovación afecta solamente la coordenada del choque monetario.
Los valores marginales esperados se promedian **antes** del paso retrospectivo
del hogar; la continuación no lineal de Phillips se promedia por separado.
Las ecuaciones de curvatura del estado y de varianza se resuelven por separado,
incluyendo la curvatura de la política futura en la segunda. Los tensores usan
la convención de Taylor con factor un medio y todavía no incluyen la varianza
elegida para la innovación.

Las derivadas centradas con `difference_step` y la mitad del paso deben concordar
antes de la extrapolación de Richardson. Se rechazan primeras derivadas
incompatibles, curvatura no resuelta, ecuaciones de varianza singulares y
dimensiones que superan los límites. No se suavizan quiebres de política ni se
regulariza el sistema. La concordancia de dos pasos respalda el cálculo local;
no demuestra suavidad global al cruzar restricciones o nodos de interpolación.

La expansión de estado y varianza sigue a
[Schmitt-Grohé y Uribe](https://www.columbia.edu/~mu2166/2nd_order/2nd_order.pdf).
La simulación separa los componentes de primer y segundo orden, siguiendo el
tratamiento coherente por órdenes discutido por
[Kim, Kim, Schaumburg y Sims](https://www.federalreserve.gov/pubs/feds/2003/200361/200361pap.pdf).
Los términos cuadráticos usan únicamente estados de primer orden; realimentar
correcciones cuadráticas introduciría términos de orden superior no deseados.

## Riesgo anticipado y dispersión de estados

```python
mean = policy.mean_decomposition()
np.testing.assert_allclose(
    mean["state_mean"], mean["state_risk"] + mean["state_dispersion"],
)
np.testing.assert_allclose(
    mean["control_mean"], mean["control_risk"] + mean["control_dispersion"],
)
zero = policy.state()
np.testing.assert_allclose(
    policy.controls(zero, shock_std=0.0005),
    4 * policy.controls(zero, shock_std=0.00025),
)
path = policy.simulate(np.array([0.0002, -0.0001, 0., 0.]))
assert path.converged
assert path.diagnostics["pruned"]
```

`state_risk` es el límite con riesgo anticipado e innovaciones realizadas nulas.
`state_dispersion` recoge la curvatura de la regla evaluada en la covarianza
estacionaria de primer orden. Su suma es la media de la aproximación podada.
Las medias de controles son coordenadas: las últimas tres corresponden a
desviaciones logarítmicas de inflación, salario y producto. No son medias en
niveles transformados ni bienestar del hogar. Aplicar una función de utilidad
no lineal a estos objetos no produce una medida auditada de bienestar.

## Alcance de las comprobaciones

`controls` y `next_state` evalúan polinomios; `audit` evalúa las ecuaciones no
lineales **discretas** de hogares, distribución y mercados con cuadratura
gaussiana. `simulate` audita cada paso podado. Los fallos de ecuaciones,
probabilidad o dominio positivo siguen siendo fallos. Nunca se recorta ni
renormaliza una distribución para rescatar una trayectoria. La poda no garantiza
la factibilidad de la distribución.

Los residuos Euler/KKT se reportan aparte de la aceptación de ecuaciones en
malla finita. En la malla demostrada de 20 activos rondan 0.004, aunque los
errores de las ecuaciones discretas son mucho menores. No se certifica la
convergencia de la malla del hogar.

Los casos de ambos signos con 12 activos y los choques positivos con 20 activos
muestran una caída cúbica del error por encima de su piso numérico. Con 20
activos y escalas positivas de choque e
innovación de 0.001, 0.0005 y 0.00025, los errores máximos escalados son
aproximadamente 2.34e-6, 2.91e-7 y 3.62e-8. Casos polinómicos exactos y una
solución densa independiente verifican el álgebra. Equilibrios no lineales
finitos resueltos por separado aportan otra comparación, con su propia frontera
terminal de primer orden. La sensibilidad al horizonte estocástico sigue siendo
material; concordar con un árbol no certifica precisión global o infinita.

Los choques negativos de -0.001 y -0.0005 en la malla de 20 activos fallan
la tolerancia original de auditoría, 1e-5: los errores de la ecuación de
distribución rondan 2.55e-4 y 3.69e-5. Los casos menores, -0.00025 y -0.000125,
sí pasan. En total pasan 22 de los 24 casos de amplitud; los dos rechazos se
conservan. La precisión local debe comprobarse para cada dirección y tamaño
del choque, incluso cuando pasan las ecuaciones de coeficientes y las pruebas
de refinamiento de derivadas.

La comprobación más estricta de referencia, 1e-9, falla con la calibración de
12 activos por su piso de equilibrio de aproximadamente 1.47e-9. Esos fallos se
conservan; las referencias de 20 activos cumplen la misma tolerancia, sin
relajarla.

Los registros de amplitud, cuadratura, medias y referencias no lineales se
reproducen con `tools/reference_validation/benchmark_hank_second_order.py`.
Revise los fallos y las comparaciones de horizonte antes de ampliar el método.
