# Equilibrio HANK acoplado y optimizado

`HANKEquilibriumWorkspace` resuelve conjuntamente inflación, salarios,
producción, decisiones de los hogares y distribuciones en cada nodo de un
árbol finito de historias agregadas. Newton–GMRES usa productos analíticos
Jacobiano-vector a través del hogar y del transporte de la distribución,
sin construir un Jacobiano agregado denso. El precondicionador disperso y
los puntos iniciales explícitos permiten repetir experimentos.

Conserva la economía del [modelo en árbol finito](hank_scenario_tree.md): un
activo, deuda real fija, valoración empresarial neutral al riesgo en unidades
nominales, riesgo idiosincrático independiente y valor marginal terminal del
estado estacionario. **No es todavía un equilibrio estocástico estacionario**,
una política sobre el espacio de distribuciones, un modelo de dos activos ni
un modelo con límite inferior efectivo de la tasa de interés.

## Uso

```python
from puremacro.models import (
    HANKEquilibriumWorkspace,
    calibrate_one_asset_hank,
    monetary_scenario_tree,
)

# Malla ilustrativa; la precisión de investigación requiere refinamientos.
ss = calibrate_one_asset_hank(n_a=240, n_e=7, a_max=150.)
tree = monetary_scenario_tree(
    horizon=48, innovations=(-.005, .005), probabilities=(.3, .7),
)
workspace = HANKEquilibriumWorkspace(ss, tree)
base = workspace.solve()
cercano = workspace.solve(
    monetary_shocks=1.02 * tree.monetary_shock,
    initial=base,
)
print(cercano.converged, cercano.diagnostics["market_residual_max_abs"])
```

El espacio de trabajo copia los datos. Cambiar el objeto original no cambia
el problema preparado. Nuevos choques deben conservar el orden de nodos; una
Series debe tener exactamente el mismo índice. Cambios en parámetros, mallas,
probabilidades condicionales o datos terminales requieren otro espacio de
trabajo. La huella de esos datos comprueba la compatibilidad del resultado usado
como punto inicial. No se reutiliza implícitamente la solución anterior.

`solve_hank_equilibrium(ss, tree)` permite una solución individual. Ambos
interfaces devuelven `HANKScenarioTreeResult`, con políticas, distribuciones
condicionales, trayectorias y expectativas no condicionales por fecha.

## Algoritmo

La separación de bloques sigue la motivación del enfoque de espacio de
secuencias de [Auclert, Bardóczy, Rognlie y Straub
(2021)](https://doi.org/10.3982/ECTA17434). Este backend diferencia el punto
no lineal actual sobre el árbol:

1. EGM resuelve hacia atrás utilizando valores marginales esperados de los
   sucesores. Se guardan intervalos, pendientes y restricciones activas.
2. Las tangentes diferencian la malla endógena y las decisiones. En hogares
   restringidos se diferencian el presupuesto y la condición intratemporal.
3. El transporte hacia delante diferencia tanto la masa como los pesos de
   las loterías de activos; los cambios de composición afectan la agregación.
4. [GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
   obtiene la corrección de precios mediante estos productos. Una búsqueda
   amortiguada evalúa de nuevo todas las ecuaciones no lineales.

El precondicionador aproxima respuestas contemporáneas del hogar y la
persistencia de activos agregados, junto con las ecuaciones de Phillips y de
política monetaria. Reutiliza una factorización LU dispersa. Las expectativas
y la composición de la distribución se aproximan solamente dentro del
precondicionador: sus efectos completos están presentes en los productos
Jacobiano-vector y en los residuos finales. `preconditioner="none"` permite
comparar. En puntos angulosos se selecciona la derivada de la rama actual;
las diferencias finitas que cruzan ramas pueden discrepar. Los datos de
derivación se renuevan después de cada paso aceptado.

## Diagnósticos, memoria y evidencia

Las coordenadas concatenan `log(1+inflación)`, `log(salario/salario_ss)` y
`log(producción)` por nodo. `workspace.residual(x)` devuelve residuos de
Phillips, activos y trabajo efectivo. `jacobian_vector_product(x, direction)`
expone el producto analítico para comprobaciones independientes.

`converged` exige también el equilibrio de bienes, conservación de masa,
distribuciones no negativas salvo redondeo, presupuestos y masa despreciable
fuera de la malla. Los errores Euler/KKT y la masa en el límite superior se
informan por separado. Convergencia numérica no equivale a precisión de malla,
de horizonte o de bienestar estocástico estacionario.

Los datos inválidos y un dominio inicial inviable producen una excepción.
Agotar iteraciones, fallar la solución lineal o la búsqueda de paso conserva
el último punto viable con `converged=False`. `optimizer_message` e `history`
registran la causa, el residuo lineal recalculado y el paso aceptado. Un
precondicionador singular indica que puede intentarse `preconditioner="none"`.

El backend nuevo elimina el límite de 256 nodos del solver denso de referencia,
pero la memoria sigue creciendo con historias por celdas del hogar.
`max_workspace_bytes` limita una estimación de matrices de trabajo (un GiB
por defecto), no la memoria RSS; excluye objetos Python, BLAS y factores
dispersos. Reinicios Krylov mayores aumentan esa estimación.

Las pruebas contrastan derivadas con diferencias finitas, conservación de
masa y primer momento de activos; comparan con el solver denso y la transición
determinista publicada; resuelven 279 nodos y comprueban reutilización,
aislamiento de datos y fallos. El programa reproducible
`tools/reference_validation/benchmark_hank_equilibrium.py` registra fuentes,
entorno, calibración separada, tiempos repetidos con preparación incluida,
inicios cercanos, diferencias de políticas y trayectorias CSV. La equivalencia
usa las mismas mallas; los tiempos no son un criterio de CI.

El registro `tests/fixtures/hank_optimized/record.json` conserva estas medianas
de tres repeticiones en arm64, Python 3.11.17, NumPy 2.4.6 y SciPy
1.17.1. Ambas bibliotecas usan Apple Accelerate. El programa configura y
verifica su modo de un hilo mediante
[`BLASSetThreading` / `BLASGetThreading`](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))
en el hilo que ejecuta las llamadas, junto con límites de entorno para
OMP/OpenBLAS/MKL/vecLib. Todas las soluciones convergieron con tolerancia de
mercados `1e-8`; la mayor diferencia de trayectorias frente a la referencia
densa fue `9.07e-09`.

| Activos × ingresos | Periodos / nodos | Denso (s) | Optimizado inicial (s) | Razón | Inicio cercano (s) |
|---|---|---:|---:|---:|---:|
| 240 × 7 | 48 / 95 | 19.38 | 1.18 | 16.4× | 1.03 |
| 480 × 31 | 48 / 95 | 58.78 | 6.01 | 9.8× | 5.14 |
| 240 × 7 | 96 / 191 | 76.71 | 3.61 | 21.3× | 3.10 |

La calibración no está incluida en estos tiempos. Son observaciones de esta
máquina y estos casos, sin certificar precisión de malla. Los CSV permiten
recalcular diferencias agregadas; los grandes arreglos de hogares no se
archivan y sus diferencias requieren repetir el programa. El directorio
`prior_uncontrolled/` conserva las observaciones iniciales y el programa
exacto, cuyos límites de entorno no fijaban explícitamente Accelerate;
esta ejecución con modo de un hilo verificado sustituye aquellos tiempos.

## Hacia HANK estocástico estacionario

El siguiente paso es representar la distribución sin expandir un árbol
exponencial: una representación comprimida y una política terminal
estocástica resuelta, auditadas en historias agregadas externas al ajuste.
Deben comprobarse la distribución completa, Euler/KKT y los mercados;
aproximar unos momentos no basta. Después pueden incorporarse activos líquidos
e ilíquidos, restricciones de cartera y reglas fiscales más amplias, con
referencias independientes para las nuevas ecuaciones. Los kernels compilados
o acelerados requieren mediciones de los bloques costosos y conservar este
backend NumPy/SciPy como comparación portátil.
