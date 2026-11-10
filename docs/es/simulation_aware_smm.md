# SMM con incertidumbre de simulación

`puremacro.structural.fit_simulated_moments` ajusta momentos simulados teniendo
en cuenta tanto el error muestral empírico como el error de Monte Carlo. Esta
primera interfaz admite momentos localmente suaves con un presupuesto fijo de
simulación. Extiende el [puente empírico-estructural](structural_bridge.md) y
conserva los estimadores existentes de VFI, DMP y deuda soberana.

## Un ajuste reproducible

La covarianza empírica es **Cov(el estimador empírico)**, ya en su escala
muestral. El ejemplo utiliza una media observada sintética con varianza conocida
`1/1000`. Cada réplica simulada es una media de diez observaciones normales
independientes; 32 de esas medias forman el estimador simulado.

```python
from dataclasses import replace
import numpy as np
from puremacro.structural import (
    MomentReplication, MomentTargets, SimulationPlan,
    fit_simulated_moments, simulate_moments, moment_error_covariance,
)

targets = MomentTargets([0.4], [[1/1000]], ("mean",), units="level")
plan = SimulationPlan(seed=20261108, replications=32, periods=10)

def simulator(theta, rng, budget):
    values = theta[0] + rng.normal(size=budget.periods)
    return MomentReplication([values.mean()], counts=[budget.periods])

fit = fit_simulated_moments(
    simulator, targets, [0.0], plan=plan,
    definition_id="iid-normal-mean-v1", parameter_names=("mu",),
    jac=lambda theta: [[1.0]],
    assume_independent=True, assume_smooth=True,
    validation_plan=replace(plan, role="validation", replications=128),
)
assert fit.success and fit.inference_valid, fit.inference_reasons
print(fit.summary())
print(fit.moment_fit())
```

`success` describe al optimizador. `inference_valid` requiere además superar
las comprobaciones numéricas y de derivadas locales, sujeto a los supuestos
estadísticos siguientes. `summary()`, `to_frame()`, `to_dataframe()`,
`to_markdown()`, `to_latex()` y `to_typst()` presentan las mismas estimaciones
e intervalos. `moment_fit(validation=True)` ofrece una comparación descriptiva
separada.

## Dos convenciones distintas de covarianza

Para vectores conjuntos independientes \(s_1,\ldots,s_R\), la media simulada
es \(\bar s=R^{-1}\sum_r s_r\), con covarianza estimada

\[
V_s=\frac{1}{R(R-1)}\sum_r(s_r-\bar s)(s_r-\bar s)' .
\]

**No se divide otra vez por el número de períodos**: la variabilidad del
estadístico de cada trayectoria ya está incluida en su vector de réplica.
La dependencia temporal pertenece a cada trayectoria; las réplicas externas
deben ser independientes. En cambio, `MomentTargets.from_draws` estima la
covarianza de un estimador empírico con réplicas bootstrap/muestrales y **no**
divide por el número de réplicas.

```python
batch = simulate_moments(
    simulator, fit.theta, plan, labels=targets.labels, units=targets.units,
    definition_id="iid-normal-mean-v1",
)
np.testing.assert_array_equal(batch.replicates, fit.simulation.replicates)
np.testing.assert_allclose(batch.covariance,
                           batch.replicate_covariance / plan.replications)
total = moment_error_covariance(targets, batch, assume_independent=True)
np.testing.assert_allclose(total, targets.covariance + batch.covariance)
```

Si los datos y la simulación comparten aleatoriedad, proporcione
`cross_covariance=C`, donde \(C=\operatorname{Cov}(\hat m,\bar s)\), en vez
de afirmar independencia. La covarianza del error de momentos es
\(V_e+V_s-C-C'\). La covarianza conjunta completa debe ser semidefinida
positiva; etiquetas y unidades deben coincidir exactamente. Usar semillas
distintas no establece independencia si el callback reutiliza datos observados.
La matriz constante `C` debe corresponder al parámetro relevante y al número
de réplicas del ajuste.

## Pesos fijos y contribuciones separadas a la varianza

El objetivo es \((\bar s(\theta)-\hat m)'W(\bar s(\theta)-\hat m)\).
Por defecto se fija \(W=V_e^{-1}\). Se admite una matriz `weight` definida
positiva explícita. Alternativamente, `pilot_plan` estima los pesos en `theta0`
usando la covarianza de las réplicas piloto dividida por el número de réplicas
del **ajuste**. Los períodos y el burn-in del piloto y del ajuste deben
coincidir. Los pesos y el esfuerzo de simulación permanecen fijos durante
la optimización. No se aplica regularización ni truncamiento de autovalores.

```python
pilot_fit = fit_simulated_moments(
    simulator, targets, [0.0], plan=plan,
    pilot_plan=replace(plan, role="pilot", replications=64),
    definition_id="iid-normal-mean-v1", assume_independent=True,
    assume_smooth=True, jac=lambda theta: [[1.0]],
)
assert pilot_fit.inference_valid
np.testing.assert_allclose(
    pilot_fit.covariance,
    pilot_fit.sampling_covariance + pilot_fit.simulation_covariance
    + pilot_fit.cross_covariance_adjustment,
)
```

Con \(G=\partial\bar s/\partial\theta'\) y
\(A=(G'WG)^{-1}G'W\), la covarianza local es
\(A(V_e+V_s-C-C')A'\). El resultado conserva por separado las contribuciones
muestral, de simulación y cruzada. `moment_covariance` usa unidades de momentos;
las tres contribuciones y `covariance` usan unidades de parámetros. La
covarianza final de simulación se evalúa en la estimación, incluso cuando los
pesos proceden de un piloto.

## Aleatoriedad y contrato del callback

- `simulator(theta_copy, rng, plan)` devuelve un vector conjunto o un
  `MomentReplication`. Utilice el generador suministrado para **todas** las
  decisiones aleatorias. Implemente explícitamente `periods` observaciones
  retenidas y el `burn_in` declarado.
- Cada réplica recibe un generador PCG64 nuevo. Su clave SeedSequence es
  `[replication, stream, role_code, seed]`, con códigos 0/1/2 para
  fit/pilot/validation. Un mismo plan reproduce las secuencias independientemente
  del orden de evaluación. Aumentar el número de réplicas conserva las secuencias
  existentes. Los roles separan secuencias, no definiciones de momentos: use
  el mismo estadístico en piloto, ajuste y validación.
- `plan_id` resume mediante hash la configuración, el esquema de secuencias,
  el generador y la versión de NumPy. No incluye el código del modelo, no
  garantiza identidad binaria entre entornos ni establece convenciones
  económicas de medición. Conserve también el código y el entorno.
- `definition_id` identifica la convención de medición del usuario. Alinee
  condicionamiento, rezagos, transformaciones, ventanas muestrales, unidades
  y selección de ramas con los objetivos empíricos. Los identificadores y
  etiquetas no demuestran esa correspondencia.
- La ausencia de observaciones condicionales debe producir conteos nulos o
  momentos no finitos, nunca ceros fabricados. Valores no finitos, conteos
  nulos o un estado distinto de `ok` invalidan todo el lote conjunto. Todas
  las filas se conservan en `replicates`, `counts` y `statuses`; los conteos
  ausentes significan desconocidos. No se filtran las réplicas supervivientes.
  Los errores de programación del callback se propagan sin modificación.

`SimulationMomentError` conserva `.batch` y, cuando está disponible, una
`.reference` para fallos de reproducción. El ajuste comprueba reproducción
exacta al inicio y en la estimación; también reproduce piloto y validación.
Una caché acotada puede reutilizar evaluaciones idénticas. `n_evals` cuenta
lotes realmente simulados, incluidas comprobaciones, piloto y validación;
`n_replications_evaluated` suma sus presupuestos de réplicas.

## Alcance inferencial y validación

Los intervalos normales requieren `assume_smooth=True` explícito. Se omiten
ante fallo o falta de estacionariedad del optimizador, estimaciones en la
frontera, rango insuficiente, condicionamiento excesivo, covarianza singular
del error de momentos, covarianza no finita o comprobaciones de derivadas
inconsistentes. Un `jac` suministrado debe diferenciar la **media simulada del
plan fijo**. Dos radios de diferencias finitas y pendientes por izquierda y
derecha comprueban consistencia con números aleatorios comunes.
`derivative_step` controla el radio inicial escalado por parámetro;
`derivative_tol`, la tolerancia de discrepancia normalizada.

Superar estas comprobaciones no demuestra suavidad, identificación ni cobertura
estadística. La inferencia también supone especificación correcta,
identificación interior regular, estimaciones consistentes de covarianzas y
jacobiano, y sesgos despreciables de inicialización, estadístico de horizonte
finito y discretización. Un error de Monte Carlo grande puede deteriorar la
aproximación local. Políticas discretas, cambios de rama, identificación débil
y mala especificación requieren otros métodos. No se calcula automáticamente
un valor p de J ni un conjunto de confianza robusto a identificación débil.

Las secuencias nuevas de validación son descriptivas y nunca se agregan al
objetivo ajustado. Como la estimación depende de las secuencias del ajuste,
su diferencia frente a validación no constituye automáticamente una prueba
estándar de dos muestras independientes. Cambiar el presupuesto requiere un
plan y ajuste explícitos nuevos. Esta interfaz genérica no audita la medición
temporal ni los momentos condicionales del estimador soberano existente.

Ejecute `python -m puremacro.examples.simulation_aware_smm --output /tmp/smm`
para obtener CSV, Markdown, LaTeX, Typst y JSON. El ejemplo verifica momentos
brutos y varianza finita de la media de un AR(1) estacionario, además de momentos
de una cadena de Markov de dos estados con medición temporal coincidente.
Las pruebas comprueban independientemente la covarianza conjunta completa
del AR(1) mediante identidades gaussianas y el álgebra de GLS correlacionado.

Referencias: [Duffie y Singleton (1993)](https://web.stanford.edu/~duffie/jstorlinks/ds1.pdf)
sobre momentos simulados; [documentación de secuencias aleatorias de NumPy](https://numpy.org/doc/stable/reference/random/parallel.html)
para identificadores deterministas. Estas referencias no certifican callbacks
arbitrarios ni la inferencia en muestras finitas.
