# Conjuntos de confianza robustos a identificación débil

El módulo estructural incorpora pruebas de hipótesis puntuales sobre momentos
y conjuntos de confianza continuos para una clase escalar explícita. Estos
métodos no invierten un jacobiano del modelo ni presuponen identificación
fuerte. Complementan la [inferencia estructural local](structural_bridge.md)
y el [SMM con incertidumbre de simulación](simulation_aware_smm.md).

## La prueba de hipótesis puntual

Para momentos empíricos \(\hat m\), momentos del modelo \(m(\theta)\) y la
covarianza completa \(V(\theta)\) de su diferencia, el estadístico es

\[
S(\theta)=[\hat m-m(\theta)]'V(\theta)^{-1}[\hat m-m(\theta)].
\]

Se compara con un valor crítico chi-cuadrado con **q grados de libertad**,
donde q es el número de momentos. No se resta el objetivo minimizado ni se
utiliza q-p. La covarianza ya está en la escala del estimador; no se aplica un
multiplicador de tamaño muestral. Es una prueba S de hipótesis puntual sobre
el vector completo, siguiendo el principio del
[Teorema 2 de Stock y Wright (2000)](https://users.ssc.wisc.edu/~behansen/718/StockWright2000.pdf).

```python
import numpy as np
from puremacro.structural import (
    MomentTargets, QuadraticMomentCurve, moment_restriction_test,
    invert_quadratic_moments, evaluate_moment_grid,
    SimulationPlan, simulate_moments, simulated_moment_test,
)

targets = MomentTargets([1.0], [[0.01]], ("second",))
point = moment_restriction_test(targets, [1.0], calibration="gaussian_known")
assert point.valid and point.accepted and point.df == 1
print(point.to_frame())
```

`calibration="gaussian_known"` declara que el error de momentos bajo la nula
es exactamente gaussiano, con covarianza conocida y definida positiva. Es un
supuesto del usuario que no se verifica automáticamente. Por defecto,
`calibration="asymptotic"` requiere un TCL válido bajo la nula y estimación
consistente de la covarianza; la validez uniforme exige los correspondientes
supuestos uniformes. Ninguna opción comprueba la especificación correcta.

Use `covariance=...` únicamente para la **covarianza total del error de momentos
en el candidato**, incluida cualquier incertidumbre del modelo o simulación.
Un peso arbitrario del ajuste o una covarianza evaluada en una estimación
potencialmente inconsistente no la sustituyen. Esta primera interfaz no
concentra parámetros de nuisance: todos los parámetros de una predicción
bajo la hipótesis puntual deben estar especificados.

`valid` indica evaluación numérica válida. Momentos del modelo no finitos o
covarianzas singulares, incluso numéricamente, producen resultados no resueltos
con `accepted=None`. No se introduce una pseudoinversa ni regularización para
cambiar silenciosamente la prueba. Etiquetas, dimensiones o matrices no
semidefinidas positivas incorrectas generan errores. Las exportaciones
conservan el estado.

## Inversión continua para una curva escalar declarada

`QuadraticMomentCurve` admite

\[
m(\theta)=a+b(\ell z+c z^2),\qquad z=(\theta-\text{center})/\text{scale}.
\]

a y b son vectores de q componentes; el índice común tiene coeficientes
escalares `linear` y `quadratic`. Es una clase limitada, no un vector arbitrario
de funciones cuadráticas. La covarianza debe ser constante respecto al
parámetro candidato. La interfaz obtiene el intervalo aceptado del índice y
su preimagen bajo la curva, conservando todos los componentes resueltos
numéricamente.

```python
curve = QuadraticMomentCurve(
    [0.0], [1.0], targets.labels, "gaussian-theta-squared-v1",
    linear=0.0, quadratic=1.0,
)
confidence = invert_quadratic_moments(curve, targets, calibration="gaussian_known")
assert confidence.resolved and len(confidence.intervals) == 2
assert confidence.contains(-1.0) and confidence.contains(1.0)
assert not confidence.contains(0.0)
print(confidence.summary())
print(confidence.to_frame())

clipped = invert_quadratic_moments(
    curve, targets, domain=(0.0, 1.0), calibration="gaussian_known",
)
assert clipped.truncated and clipped.boundary_contact == (False, True)
```

El ejemplo tiene dos componentes separados cerca de -1 y +1. No deben
reemplazarse por el intervalo entre los extremos exteriores. También se
conservan tangencias de un solo punto, conjuntos vacíos y toda la recta real.
Una carga exactamente nula no implica un intervalo estrecho: la prueba de
momentos puede aceptar todos los parámetros o rechazarlos todos. Las cargas
pequeñas no nulas nunca se convierten silenciosamente en cero mediante un
umbral de identificación.

`global_intervals` conserva la preimagen analítica; `intervals` la intersecta
con `domain`, una ventana de búsqueda/presentación. `global_geometry` distingue
conjuntos acotados, no acotados/toda la recta, vacíos y no resueltos.
`truncated` indica que parte del conjunto global queda fuera de la ventana,
incluso cuando el conjunto mostrado está vacío. `boundary_contact` describe
los extremos finitos de la ventana aceptados. La falta de acotación se deduce
de la curva declarada, no de alcanzar una frontera de malla. La cobertura
dentro de una ventana requiere que el parámetro verdadero pertenezca a esa
ventana preespecificada.

`resolved` se refiere a la geometría en punto flotante. Las tangencias casi
degeneradas ambiguas, raíces no representables y fallos en los residuos de los
extremos permanecen sin resolver; `contains()` genera entonces un error.
La incertidumbre numérica no se convierte en un conjunto vacío. `summary()`
presenta el estado separado de las filas de componentes. Las exportaciones
Markdown, LaTeX y Typst incluyen el estado aun para conjuntos vacíos/no resueltos.

## Modelos generales: las hipótesis muestreadas siguen siendo puntos

`evaluate_moment_grid` recibe un callback que devuelve `MomentTestResult`.
Puede utilizar una covarianza distinta en cada candidato y conservar
evaluaciones inválidas. No afirma haber completado la inversión continua.

```python
test_at = lambda theta: moment_restriction_test(
    targets, curve(theta), calibration="gaussian_known",
)
sampled = evaluate_moment_grid(test_at, [-1.5, 0.0, 1.5])
assert not sampled.accepted.any()
assert not sampled.continuous_complete
assert len(sampled.unresolved_cells()) == 2
assert len(confidence.intervals) == 2
```

Los tres puntos se rechazan, aunque el conjunto continuo contiene dos islas
aceptadas. Por ello, `accepted_runs()` devuelve **tramos de nodos muestreados**,
no extremos de intervalos de confianza. Cada celda entre nodos sigue sin
resolver, incluidas aquellas cuyos dos extremos se rechazan. Los nodos
inválidos tienen `accepted=None` en la tabla y separan los grupos aceptados.
Los errores inesperados del callback se propagan en vez de transformarse en
penalizaciones de rechazo. Todos los nodos deben compartir alpha, etiquetas,
q y calibración.

## Incertidumbre de simulación en el parámetro hipotetizado

El adaptador usa el lote conjunto realmente simulado en el candidato, con sus
conteos/estados y la covarianza completa de la media simulada. Declare
independencia respecto a los datos o proporcione covarianza cruzada, como en SMM.

```python
empirical = MomentTargets([0.3], [[0.02]], ("mean",))
plan = SimulationPlan(seed=618, replications=64, periods=20)

def simulator(theta, rng, budget):
    return [theta[0] + (1 + theta[0]**2) * rng.normal(size=budget.periods).mean()]

def simulated_test_at(theta):
    batch = simulate_moments(
        simulator, [theta], plan, labels=empirical.labels,
        definition_id="heteroskedastic-normal-mean-v1",
    )
    return simulated_moment_test(empirical, batch, assume_independent=True)

simulation_grid = evaluate_moment_grid(simulated_test_at, np.linspace(-0.5, 1.0, 31))
assert simulation_grid.valid.all()
assert simulation_grid.tests[0].calibration == "asymptotic"
print(simulation_grid.accepted_runs())
```

La covarianza total se recalcula en cada hipótesis; el ejemplo cambia
deliberadamente la varianza simulada con el parámetro. Nunca se reutiliza un
peso piloto fijo de SMM como covarianza de la prueba. `simulated_moment_test`
solo ofrece calibración asintótica: una covarianza estimada con un número
finito de réplicas no produce un pivote chi-cuadrado exacto. Los lotes
inválidos permanecen adjuntos a resultados no resueltos. Los sesgos de burn-in,
estadísticos de horizonte finito y discretización necesitan control separado.

## Interpretación y límites de validación

La nula mantiene tanto el parámetro como las restricciones de momentos. Un
conjunto vacío puede reflejar momentos incompatibles; un conjunto pequeño,
por sí solo, no establece identificación precisa. Esta primera versión no
incluye diagnóstico automático de identificación débil, concentración
arbitraria de parámetros de nuisance, QLR condicional, inferencia robusta a
singularidad ni inversión continua de callbacks arbitrarios. Las API
anteriores de bandas envolventes para IV débil no cambian.

Ejecute `python -m puremacro.examples.identification_robust_sets --output /tmp/weak-id`
para obtener tablas de componentes, una demostración de islas no detectadas,
una malla simulada, exportaciones, una figura y procedencia JSON. Las pruebas
independientes comprueban formas cuadráticas gaussianas, extremos analíticos,
cargas pequeñas no nulas, tangencias, truncamiento y evaluaciones inválidas.
El estudio local de aceptación utiliza cinco diseños gaussianos con covarianza
conocida y 2.000 ensayos por diseño, incluyendo identificación fuerte, débil,
ausente y en puntos de giro. Estas referencias no certifican modelos económicos
arbitrarios ni cobertura finita con covarianzas estimadas.
