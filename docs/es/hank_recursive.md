> 🇬🇧 [English](../hank_recursive.md) · 🇪🇸 Español

# Políticas recursivas HANK y equilibrio con dos activos

`solve_hank_stochastic_policy` resuelve una **política local de expectativas
racionales, de primer orden y horizonte infinito**, alrededor de un estado
estacionario con hogares heterogéneos. El estado contiene coeficientes de la
distribución, la tasa nominal heredada y una perturbación monetaria persistente.
Los valores marginales de los hogares y los precios agregados responden
conjuntamente; las innovaciones agregadas futuras continúan indefinidamente.

La distinción entre proyección y perturbación sigue a
[Reiter (2009)](https://doi.org/10.1016/j.jedc.2008.08.010): resolver primero la
heterogeneidad no lineal y después linealizar la dinámica agregada. El
[documento del autor](https://econ-papers.upf.edu/papers/972.pdf) explica por qué
unos pocos momentos pueden resultar insuficientes. PureMacro combina una base
restringida para la distribución y una descomposición de Schur generalizada
ordenada. El alcance es local: no calcula desplazamientos precautorios de
segundo orden, primas de riesgo ni bienestar estocástico global.

## Resolver, comprimir y auditar

```python
import numpy as np
from puremacro.models import (
    calibrate_one_asset_hank, solve_hank_stochastic_policy,
    solve_hank_stochastic_tree, monetary_scenario_tree,
)

# Malla ilustrativa; la precisión requiere refinamiento independiente.
ss = calibrate_one_asset_hank(n_a=80, n_e=3, a_max=150.)
full = solve_hank_stochastic_policy(ss, distribution_rank=None)
training = full.simulate(
    np.random.default_rng(1201).normal(0, 1e-7, 160)
).distributions
policy = solve_hank_stochastic_policy(
    ss, distribution_rank=16, snapshots=training,
    rho=.7, shock_std=1e-4,
)
held_out = np.random.default_rng(9917).normal(0, 1e-4, 48)
audit = policy.audit(held_out, tolerance=1e-5)
print(audit.converged, audit.diagnostics)

tree = monetary_scenario_tree(
    horizon=8, persistence=.7,
    innovations=(-.001, .001), probabilities=(.3, .7),
)
equilibrium = solve_hank_stochastic_tree(policy, tree)
print(equilibrium.converged, equilibrium.diagnostics)
```

Sin `snapshots`, la base procede de respuestas de los hogares a precios y
transporte repetido de la distribución. Con observaciones, usa una SVD de
desviaciones respecto al estado estacionario. Ambos métodos conservan primero
las direcciones necesarias para los observables de mercado contemporáneos;
un rango inferior a ese mínimo se rechaza. Los coeficientes de la política de
equilibrio resuelven las ecuaciones de hogares, distribución, mercados y
precios; no se ajustan mediante una regresión. `distribution_rank=None`
conserva toda la distribución restringida y sirve como referencia.

La base de un activo conserva cada marginal de ingresos y los activos
heredados agregados. La de dos activos conserva los marginales de ingresos y
los activos líquidos heredados; la riqueza ilíquida varía con el precio del
bono perpetuo. La proyección conserva además los agregados de las políticas
estacionarias, la condición laboral y la riqueza ilíquida heredada. Las
columnas son ortonormales y representan desviaciones con signo.
`basis.decode()` comprueba positividad por defecto. No recorta ni renormaliza.
Las simulaciones lineales informan explícitamente de masa reconstruida
negativa; un buen ajuste de momentos no prueba precisión distributiva.

`policy.simulate(innovations)` utiliza la recursión lineal resuelta. Sus
trayectorias agregadas son **desviaciones logarítmicas**, incluida
`log(1+inflación)`. Las trayectorias del solver no lineal son desviaciones en
niveles. `policy.covariance()` resuelve la covarianza estacionaria de primer
orden. `shock_std` modifica esa covarianza, pero no los coeficientes de la
política ni las medias estacionarias: es equivalencia cierta de primer orden.

`policy.audit()` transporta la distribución completa y resuelve los hogares
en historias externas al ajuste. Hace consistentes la distribución siguiente
y los valores de continuación, manteniendo fija la política agregada. Informa
errores de mercados y bienes, presupuestos, masa, salidas de malla, Euler/KKT
y pérdida de proyección. Incluye la distribución transportada final: su
arreglo distributivo tiene una fila más que las trayectorias. La auditoría
puede fallar aunque las ecuaciones proyectadas de Schur converjan. Los errores
Euler/KKT incluyen interpolación de hogares y se presentan separadamente del
umbral de aceptación agregado.

## Frontera terminal estocástica resuelta

`solve_hank_stochastic_tree` optimiza hogares y vacía mercados en cada nodo
finito. En cada hoja añade coeficientes distributivos desconocidos, que deben
igualar la proyección de la distribución realmente transportada más allá de
la hoja. La política terminal proporciona valores marginales, inflación y
producto sucesores, condicionados por esa distribución, la tasa nominal
elegida y la persistencia de la perturbación.

Las expectativas terminales de Phillips usan cuadratura gaussiana de dos
puntos. La regla lineal de valores marginales se integra exactamente. El
tramo finito es no lineal, pero la regla terminal sigue siendo de primer
orden: la aritmética no lineal de la frontera no constituye una solución
estocástica de segundo orden. Deben compararse horizontes, amplitudes y rangos.

La corrección utiliza Newton–GMRES amortiguado, productos centrados del
residuo no lineal completo y precondicionamiento disperso. El precondicionador
aproxima respuestas locales y el arrastre de activos; las expectativas y la
composición distributiva permanecen en el residuo externo.
`preconditioner="none"` permite contrastarlo. `method="hybr"` proporciona
una referencia densa por diferencias finitas. El
[solver analítico de árboles finitos](hank_optimized.md) permanece intacto.

`converged` exige residuos finitos de mercados y bienes, consistencia terminal,
presupuestos, masa conservada y no negativa, y salidas de malla despreciables.
La pérdida de proyección terminal, la masa terminal reconstruida mínima,
Euler/KKT y la masa superior de malla se informan aparte. Residuos finitos
pequeños no certifican la cola infinita proyectada. Los límites de iteraciones
y fallos de productos lineales o búsqueda de paso conservan el último iterado
factible y su historia. Un dominio inicial inválido genera una excepción; la
referencia MINPACK también puede hacerlo ante un ensayo fuera del dominio.

## Bono líquido y bono perpetuo real ilíquido

```python
from puremacro.models import calibrate_two_asset_equilibrium

two_ss = calibrate_two_asset_equilibrium(n_a=15, n_b=15, a_max=80.)
two_policy = solve_hank_stochastic_policy(two_ss, distribution_rank=16)
two_result = solve_hank_stochastic_tree(two_policy, tree)
print(two_result.paths[["liquid_mkt", "illiquid_mkt", "goods_mkt"]])
```

El cierre tiene bonos nominales líquidos del gobierno, con cantidad real
constante \(B\), y un bono perpetuo real ilíquido con oferta fija de unidades
\(S\), precio \(q_t\) y cupón real constante \(d\). Sus rendimientos son

\[
1+r^b_t=\frac{1+i_{t-1}}{1+\pi_t},\qquad
1+r^a_t=\frac{q_t+d}{q_{t-1}}.
\]

El hogar EGM de dos activos existente elige ahorro líquido e ilíquido y paga
el costo convexo de ajuste de cartera \(\chi\):

\[
c+a'+b'+\chi=(1+r^a_t)a+(1+r^b_t)b+z_t e.
\]

La productividad mantiene el proceso existente de dos estados. Salarios,
beneficios e impuestos se distribuyen proporcionalmente a productividad,
con media uno. La producción usa solamente trabajo, con horas sindicales
comunes \(N_t=Y_t\), y oferta flexible
\(w_t\int e c^{-\gamma}\,dD_t=\varphi Y_t^{1/\nu}\).
Los precios Rotemberg y la regla de Taylor completan el bloque agregado.

Los impuestos son \(T_t=r^b_tB+dS\). Las ganancias de capital del bono
perpetuo entran en rendimientos de hogares, no en gastos del gobierno. El
ingreso no financiero total es
\(z_t=Y_t-\text{costo de ajuste de precios}_t-T_t\).
Se vacían **ambos** mercados de activos:
\(\int b'\,dD_t=B\) y \(\int a'\,dD_t=q_tS\), junto con trabajo y Phillips.
El residuo independiente de bienes comprueba
\(Y_t=C_t+\int\chi\,dD_t+\text{costo de ajuste de precios}_t\).

La calibración recibe preferencias, costos de ajuste, rendimientos
estacionarios e ingreso no financiero. Las demandas determinan \(B\) y
\(S\); \(q=1\), el cupón iguala el rendimiento ilíquido estacionario y la
desutilidad laboral ajusta las horas al producto estacionario. Es una economía
de bonos perpetuos, distinta de un cierre con capital productivo o una réplica
publicada de ese modelo. `income` difiere del salario agregado `w`.

## Límites numéricos y evidencia

Las derivadas de política de un activo son analíticas. Las de dos activos
usan diferencias centradas de hogares almacenadas y transporte bilineal
diferenciado; las pruebas contrastan las ecuaciones conjuntas con evaluaciones
no lineales independientes. Ambos modelos comprueban determinación de Schur,
estabilidad, condicionamiento y residuos. No sustituyen un fallo de
determinación con pseudoinversas o políticas arbitrarias.

Se reduce la distribución, pero **se conservan todos los controles de valores
marginales de hogares**. QZ es denso. `max_system_size`, por defecto 1.800,
limita la dimensión antes de construir la base; no limita RSS.
`max_workspace_bytes` estima los arreglos de trabajo del árbol, excluyendo
la política ya preparada, factores dispersos y gastos de Python/BLAS. Las
mallas grandes necesitarán reducir también valores o resolver iterativamente
el subespacio estable.
En particular, la malla predeterminada de 500 × 7 de la calibración de un
activo excede este límite de Schur; los ejemplos usan mallas menores
deliberadamente. Elevar el límite permite una asignación densa mayor sin
cambiar su complejidad computacional.

`tools/reference_validation/validate_hank_recursive.py` registra políticas
completas y comprimidas, auditorías externas, horizontes, comparación densa
con Krylov y sensibilidad a mallas de dos activos. Los artefactos congelados
en `tests/fixtures/hank_recursive/` incluyen hashes de fuentes y archivos,
matrices de transición y política, semillas y trayectorias CSV. Conservan las
auditorías fallidas. La sensibilidad mantiene preferencias y precios de
hogares, recalibrando las ofertas de activos; no certifica convergencia de
malla para una economía con ofertas fijas.

La corrida conservada usa 80 puntos de activos y 3 estados de ingreso para
un activo, y 12 × 12 puntos de activos y 2 estados de ingreso para dos activos.
Las bases de rango 16 se entrenaron con 160 innovaciones pequeñas y se
evaluaron con 48 innovaciones independientes de desviación estándar `1e-4`.

| Comprobación | Un activo | Dos activos |
|---|---:|---:|
| Dimensión de distribución completa → comprimida | 236 → 16 | 285 → 16 |
| Dimensión de Schur completa → comprimida | 481 → 261 | 867 → 598 |
| Mayor diferencia agregada lineal externa | `4.44e-8` | `2.90e-10` |
| Residuo de auditoría no lineal, política completa | `1.65e-5` (falla) | `8.52e-6` (pasa) |
| Residuo de auditoría no lineal, política comprimida | `2.74e-5` (falla) | `9.49e-6` (pasa) |
| Auditoría comprimida con un cuarto de amplitud | `1.03e-6` (pasa) | `5.33e-7` (pasa) |
| Mayor residuo de árbol no lineal, horizontes 4/8/16/32 | `3.72e-10` | `5.54e-11` |
| Diferencia de trayectorias entre solución densa y Krylov | `2.07e-8` | `8.31e-10` |

El umbral es `1e-5` en todas las filas de auditoría. La falla de un activo
también aparece con la distribución restringida completa: una diferencia
lineal pequeña no elimina el error de aproximación no lineal. El resultado
de dos activos está cerca del límite de aceptación y no garantiza precisión
para otras historias de choques. Las auditorías usan el precio **cotizado**
anterior del consol al calcular rendimientos realizados, incluso cuando la
política aproximada deja un residuo en el mercado de activos. Los errores
máximos de Euler/KKT de hogares en estas mallas ilustrativas siguen cerca de
`1e-3` y `3e-3`, respectivamente. La convergencia del árbol no certifica esos
errores de malla.

`prior_variance_only/` conserva el primer experimento de compresión, sus
fuentes exactas y las fallas de determinación. Preservar observables de
mercado antes de truncar resuelve esas fallas en los casos reportados de dos
activos con rangos 8 y 16; ahora se rechaza explícitamente el rango 4 porque
omite direcciones necesarias. Los tiempos del JSON son observaciones
descriptivas, no un benchmark repetido de rendimiento.
