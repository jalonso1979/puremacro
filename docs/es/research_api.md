# Composición de las API de investigación

[English](../research_api.md)

Los módulos de investigación mantienen contratos numéricos y económicos
separados. La tabla presenta las entradas públicas de las nueve primeras
implementaciones. Las importaciones desde cada paquete y su módulo remiten a
los mismos objetos. Las interfaces específicas de cada modelo siguen disponibles.

| Propósito | Módulo público y entrada | Significado de la aceptación |
| --- | --- | --- |
| Derivadas locales del equilibrio | `core.linearize_equilibrium`; `vfi.linearize_projection`, `vfi.linearize_aiyagari` | `sensitivity().valid`: regularidad de las ecuaciones retenidas y diferencias probadas |
| Continuación de ramas | `core.continue_equilibrium` | Revisar `status`, los `attempts` rechazados y el estado de cada pliegue |
| Estimación con incertidumbre de simulación | `structural.fit_simulated_moments` | `success` describe la optimización; `inference_valid` añade las condiciones de inferencia local |
| Inferencia robusta a identificación débil | `structural.simulated_moment_test`, `invert_quadratic_moments`, `evaluate_moment_grid` | `accepted=None` indica una prueba no resuelta; una malla no certifica sus interiores |
| Escalabilidad HANK | `models.solve_hank_reduced_policy` | Solución reducida de primer orden; la precisión en la malla completa y no lineal requiere otras auditorías |
| Comercio integrado | `trade.solve_reallocation_transition` | `converged` describe ecuaciones finitas; `run_reallocation_horizon_ladder` verifica horizontes ensayados |
| Información heterogénea | `expectations.build_gaussian_information`; `models.solve_information_nk` | Pronósticos gaussianos condicionales y ecuaciones NK finitas con las señales declaradas |
| Banca y riesgo soberano | `sovereign.solve_sovereign_banking` | Equilibrios puros auditados de tres fechas, con `price_selection="highest"` o `"lowest"` explícito |
| Adaptación y riesgo soberano | `sovereign.solve_sovereign_adaptation` | Ecuaciones de Bellman, precios y decisiones discretas auditadas en horizonte finito |

Consultar las guías de [derivadas](equilibrium_linearization.md),
[continuación](equilibrium_continuation.md), [SMM](simulation_aware_smm.md),
[inferencia robusta](identification_robust_inference.md), [HANK](hank_reduction.md),
[comercio](trade_reallocation.md), [información](heterogeneous_expectations.md),
[banca](sovereign_banking.md) y [adaptación](sovereign_adaptation.md).

La [referencia HANK de segundo orden](hank_second_order.md), desarrollada por
separado, añade `models.solve_hank_second_order` para modelos pequeños con un
activo y bases completas. Tiene comprobaciones propias de derivadas, ecuaciones
y factibilidad; no cambia el orden ni los criterios de las API HANK existentes.

## Composición explícita

Un residuo de equilibrio puede aportar sensibilidades y una rama local.
Después, el usuario define los momentos medidos para SMM y utiliza las
réplicas retenidas en una prueba de hipótesis puntual. Debe declarar la
medición, las unidades y los supuestos estadísticos. Este ejemplo tiene el
equilibrio `capital = 2 * technology` y error de simulación aditivo:

```python
import numpy as np
from puremacro.core import EquilibriumSystem, linearize_equilibrium, continue_equilibrium
from puremacro.structural import (
    MomentTargets, SimulationPlan, fit_simulated_moments, simulated_moment_test,
)

system = EquilibriumSystem(lambda x, theta: x - 2*theta, ("capital",), ("technology",))
linear = linearize_equilibrium(system, [2.], [1.])
assert linear.sensitivity().valid
branch = continue_equilibrium(system, [2.], [1.], max_steps=3)
np.testing.assert_allclose(branch.states, 2*branch.parameters, atol=1e-8)

targets = MomentTargets([2.], [[.04]], ("capital",), units="level")
plan = SimulationPlan(seed=12, replications=16, periods=20)

def simulator(theta, rng, budget):
    return 2*theta + rng.normal(size=budget.periods).mean()

def derivative(theta):
    return linearize_equilibrium(system, 2*theta, theta).sensitivity().jacobian

fit = fit_simulated_moments(simulator, targets, [.8], plan=plan,
    definition_id="capital-plus-sample-mean-v1", jac=derivative,
    parameter_names=("technology",), assume_independent=True, assume_smooth=True)
assert fit.success and fit.inference_valid
point_test = simulated_moment_test(targets, fit.simulation, assume_independent=True)
assert point_test.valid and point_test.accepted
```

La prueba en el valor estimado es aquí una comprobación descriptiva de
compatibilidad. No es una prueba de sobreidentificación ni una afirmación de
cobertura posterior a la selección. Para contrastar un parámetro hipotético,
simular en ese parámetro y utilizar su covarianza. Los planes conservan todas
las réplicas, incluidas las fallidas. Las excepciones de funciones suministradas
se propagan; los momentos numéricos inválidos quedan sin resolver y no se
convierten en rechazos o penalizaciones de optimización.

No existe una composición automática entre las economías de comercio, banca,
adaptación e información. Difieren en hogares, financiamiento, cronología y
condiciones terminales. Conectarlas exige un nuevo cierre económico explícito.
HANK de orden superior y las demás extensiones aplazadas quedan fuera de estas
primeras implementaciones.

## Propiedad de datos, controles y resultados guardados

Las instantáneas de resultados de equilibrio, SMM/inferencia robusta, comercio,
Gaussian/NK y riesgo soberano tienen almacenamiento numérico independiente e
inmutable. Usar `.copy()` para obtener un arreglo de trabajo modificable. Las
clases existentes de políticas HANK conservan su contrato de resultados.
Los mapas y secuencias anidados de diagnóstico también están congelados, y
las tablas no comparten datos numéricos retenidos. Los metadatos heredados
pueden contener objetos Python arbitrarios, cuya mutabilidad interna queda
fuera de este contrato. Las tolerancias de equilibrio y los parámetros escalares
de comercio, NK y riesgo soberano aceptan números reales finitos, incluidos
arreglos NumPy numéricos de dimensión cero. Se rechazan cadenas, booleanos,
complejos y arreglos con ejes. La opción `transpose` de equilibrio exige un
booleano; `direction` de continuación exige el entero -1 o 1.

Los diagnósticos pueden exportarse mediante una conversión JSON explícita:

```python
from collections.abc import Mapping
import json

def json_value(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

encoded = json.dumps(dict(fit.diagnostics), default=json_value)
assert json.loads(encoded)["assume_independent"] is True
```

Las clases de resultados permiten reconstruir arreglos guardados. Su
construcción no ejecuta una auditoría económica. Después de restaurar un
resultado, ejecutar su función `audit_*` y revisar `accepted`. Las estructuras
retenidas mal formadas de comercio/información devuelven rechazo y motivo;
un argumento principal incorrecto o una tolerancia inválida es un error de
entrada. Cada auditoría devuelve un informe nuevo, cuya edición no cambia
la solución.

Los valores `fingerprint` identifican especificaciones, no resultados ni código
fuente. `snapshot_id` identifica el punto, las matrices y las escalas retenidas
del equilibrio. Registrar por separado las huellas del código y los archivos,
la versión del paquete y el entorno. Los ejemplos CSV/NPZ/JSON ilustran la
persistencia; no son un formato universal de checkpoint versionado. Una
auditoría válida no demuestra convergencia de malla, calibración empírica,
precisión de horizonte infinito ni preparación para publicar una versión.
