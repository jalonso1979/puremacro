# Continuación estocástica no lineal de HANK

La [política recursiva local de HANK](hank_recursive.md) proporciona una frontera
terminal estocástica resuelta. La corrección no lineal resuelve conjuntamente
las políticas de los hogares, el transporte de la distribución y los mercados
en un árbol futuro condicional, partiendo de la **distribución observada y los
precios heredados reales**. Repetir este cálculo en cada observación produce una
política no lineal no anticipativa.

Es una aproximación condicional finita con una política terminal local. No es
una solución estocástica global ni de segundo orden para horizonte infinito.
Resolver las ecuaciones de mercado no certifica las mallas de hogares, la
compresión de distribuciones ni la incertidumbre futura omitida.

## Uso

```python
import numpy as np
from puremacro.models import calibrate_one_asset_hank, solve_hank_stochastic_policy
from puremacro.models.hank_nonlinear_continuation import (
    correct_hank_stochastic_state, simulate_hank_nonlinear,
    solve_hank_nonlinear_continuation,
)

ss = calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.)
local = solve_hank_stochastic_policy(ss, distribution_rank=12, shock_std=1e-4)

# Un equilibrio condicional: su estado certifica sólo las ecuaciones finitas.
finite = solve_hank_nonlinear_continuation(
    local, distribution=ss.distribution,
    aggregate_state=np.array([0., -1e-4]), lag_q=1.,
    horizon=8, branch_depth=2,
)

# Exigir refinamientos separados del horizonte y de la ramificación.
corrected = correct_hank_stochastic_state(
    local, ss.distribution, np.array([0., -1e-4]), lag_q=1.,
)

# Antes de cada corrección se revela únicamente la innovación actual.
history = simulate_hank_nonlinear(local, np.array([-1e-4, 4e-5, 1e-5]))
assert history.converged
```

Los estados agregados de los modelos de un activo y de consol son la
**desviación** de la tasa nominal heredada y el choque monetario actual. Los
modelos con capital productivo añaden el logaritmo del capital heredado. Los
métodos `aggregate_state_names`, `prices_from_state` y `next_aggregate_state`
determinan el orden y la transición. Se aceptan políticas locales completas y
reducidas; las pasadas de hogares siempre usan la malla completa.

En `simulate_hank_nonlinear`, `initial_aggregate_state` es el estado condicional
*antes* de la primera innovación proporcionada. La tasa nominal siguiente, el
capital físico, la distribución completa transportada y el precio cotizado
real `q` proceden de la solución no lineal anterior. En particular, `q` nunca se
infiere de nuevo de una media de riqueza ilíquida fuera del equilibrio.
`initial_q` y el `lag_q` de cada corrección son niveles de precio, con valor
predeterminado uno.

## Qué se resuelve

La fecha actual es la raíz. `horizon` cuenta las fechas incluyendo la raíz. Las
innovaciones de Gauss-Hermite se integran en cada una de las siguientes
`branch_depth` fechas. Las innovaciones posteriores del árbol finito toman su
media condicional; la frontera terminal vuelve a usar la esperanza estocástica
de la política local resuelta. En los nodos no terminales, el valor marginal
esperado es la media ponderada por probabilidades de las **políticas no
lineales optimizadas de los hijos**, en lugar de la regla lineal fija original.
Los precios, las decisiones de los hogares y las distribuciones de todos los
nodos responden conjuntamente a las ramas de choques condicionales.

Las incógnitas son los controles agregados logarítmicos en cada nodo y los
coeficientes de la distribución posterior al nodo terminal. Las pasadas hacia
atrás de hogares y hacia adelante de distribuciones generan las ecuaciones de
mercado y consistencia terminal. Los productos de Newton–Krylov amortiguado
diferencian todo este residuo no lineal. Se reutilizan los precondicionadores
de precios de un activo y de consol. Otros modelos usan el operador sin
precondicionador, salvo que se amplíen por separado. `method="hybr"` permite una
comparación densa por diferencias finitas, limitada por memoria.

No se estima ningún coeficiente usando la historia realizada suministrada. Las
innovaciones realizadas posteriores no entran en un árbol condicional anterior.
El mismo estado actual y las mismas opciones producen la misma acción actual
aunque cambien las realizaciones posteriores.

## Precisión y fallos

Los límites adaptativos predeterminados son `horizons=(4, 8, 16, 32, 64)` y
`branch_depths=(1, 2, 3, 4)`. Un árbol más profundo usa sólo horizontes mayores
que su profundidad de ramificación. Cada candidato se compara con un horizonte
más corto a profundidad fija y con un árbol menos ramificado al mismo horizonte.

Cada comparación informa tres cambios: máximo cambio de controles agregados
logarítmicos de la raíz, máximo cambio relativo del valor marginal esperado en
la malla de hogares y cambio L1 de la siguiente distribución completa. Su
máximo debe satisfacer la tolerancia correspondiente. Los valores
predeterminados son `1e-6` para cada criterio de refinamiento y `1e-8` para las
ecuaciones finitas. **Son criterios empíricos de sensibilidad, no cotas
rigurosas de error.** El orden de Gauss-Hermite se mantiene fijo durante la
adaptación; `quadrature_order_converged=None` explicita esta dimensión sin
certificar. Compare órdenes por separado con el solucionador de árbol finito.

| Dimensión | Evidencia reportada | Interpretación |
|---|---|---|
| Ecuaciones agregadas | `market_residual_max_abs`, presupuesto, masa y salidas de malla | Ecuaciones no lineales del árbol elegido |
| Malla de hogares | `euler_kkt_max`, `euler_kkt_mean_max`, `upper_grid_mass` | Diagnósticos de interpolación y malla; sin certificado de precisión |
| Compresión de distribuciones | L1 de proyección inicial/terminal y masa terminal reconstruida mínima | Error de representación terminal; se conserva el transporte real completo |
| Horizonte terminal finito | `horizon_difference`, `horizon_converged` | Sensibilidad de la raíz a profundidad fija |
| Ramas de choques futuros | `branching_difference`, `branching_converged` | Sensibilidad de la raíz a horizonte y orden de cuadratura fijos |
| Orden de cuadratura | Comparaciones de orden para el mismo árbol | No lo certifica aceptar la profundidad de ramificación |

Las distribuciones reales deben ser no negativas y conservar masa unitaria.
Una distribución terminal proyectada puede tener componentes negativos: se
reportan por separado y nunca sustituyen a la distribución real ni se recortan
silenciosamente. Los valores marginales esperados terminales deben ser positivos
y finitos. Los errores de Euler en mallas gruesas pueden seguir cerca de `1e-3`
aunque los residuos agregados sean inferiores a `1e-8`; `converged=True` no
elimina esa discrepancia.

Al agotar un límite de refinamiento, se lanza por defecto
`HANKContinuationAccuracyError`; su atributo `.result` conserva la corrección
fallida. Con `raise_on_failure=False`, la corrección devuelve
`converged=False` y los estados individuales. La simulación se detiene **antes
de transportar la acción fallida**, conservando el período y sus diagnósticos.
Los estados de ecuaciones, horizonte y ramificación permanecen separados,
incluso cuando el horizonte pasa y sólo falla la ramificación. `max_nodes`,
`max_workspace_bytes` y los límites de iteraciones acotan el trabajo; aumentar
la profundidad o el orden de cuadratura hace crecer rápidamente los árboles.

## Inspección independiente

`HANKNonlinearCorrection.tree` contiene probabilidades de nodos, trayectorias,
distribuciones completas de entrada y políticas optimizadas de hogares.
`expected_marginals`, `marginal_values`, `post_distributions`, `aggregate_states`,
`aggregate_coordinates` y `prices` permiten comprobar esperanzas y presupuestos
independientemente. Los precios y arreglos de hogares son **niveles**;
`tree.paths` contiene desviaciones en niveles y `aggregate_coordinates`
desviaciones logarítmicas.

La simulación conserva políticas de hogares de la raíz, esperanzas, precios
reales y todas las distribuciones transportadas. `keep_trees=True` conserva
además cada resultado condicional, con un posible coste elevado de memoria.

La evidencia congelada y los refinamientos más cortos fallidos están en
`tests/fixtures/hank_nonlinear_continuation/`. Reproduzca en un directorio
**nuevo**:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/puremacro-nonlinear-mpl PYTHONPATH=. \
python tools/reference_validation/validate_hank_nonlinear_continuation.py \
  --output /tmp/puremacro-nonlinear-new
```

El programa registra hashes de fuentes y artefactos, versiones de bibliotecas
numéricas y el estado verificado de un solo hilo de Accelerate cuando su API
está disponible. Los tiempos describen esa ejecución y no son garantías
portables de velocidad. La semilla, la historia reservada y el umbral original
`1e-5` de la auditoría de política fija no cambian; el refinamiento numérico
añade horizonte o ramas en lugar de relajar los umbrales de aceptación.
