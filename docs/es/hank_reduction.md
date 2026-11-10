# Reducir los controles de los hogares antes de resolver el equilibrio HANK

`solve_hank_reduced_policy` comprime las desviaciones de los valores
marginales junto con los estados de la distribución. Aplica las derivadas de
los hogares a las direcciones retenidas y construye las matrices de equilibrio
en coordenadas reducidas. La ruta de dos activos nunca construye la caché
anterior de derivadas completas de los hogares.

La solución estacionaria, los pasos hacia atrás con grilla endógena, las
restricciones activas sobre activos y el transporte de probabilidades
conservan la grilla completa. La aproximación local de primer orden incluye
una continuación estocástica infinita. No calcula un cambio precautorio del
estado estacionario, una política global ni bienestar de segundo orden.

```python
import numpy as np
from puremacro.models import calibrate_one_asset_hank
from puremacro.models.hank_reduction import solve_hank_reduced_policy

steady = calibrate_one_asset_hank()  # 500 puntos de activos, 7 estados de ingreso
policy = solve_hank_reduced_policy(
    steady, distribution_rank=96, marginal_rank=128, krylov_steps=96,
    rho=0.7, shock_std=0.0001,
)
linear = policy.simulate(np.r_[-1e-5, np.zeros(39)])
audit = policy.audit(np.r_[-1e-5, np.zeros(9)], tolerance=1e-5)
print(policy.diagnostics)
print(audit.converged, audit.diagnostics)
```

La base de la distribución preserva la masa, las marginales estacionarias del
ingreso y las direcciones observables de mercado del solucionador recursivo.
La base marginal contiene las respuestas independientes a los precios
agregados actuales. Las demás direcciones se obtienen mediante aplicaciones
repetidas de la respuesta marginal hacia atrás y una SVD. Esto incluye
respuestas a precios anticipados sin ajustar los coeficientes de política de
equilibrio. Como alternativa, `marginal_snapshots` admite valores marginales
positivos de la grilla completa con forma `(observaciones,
numero_de_marginales)`; las respuestas a precios actuales siguen siendo
obligatorias.

El rango marginal solicitado se limita al espacio independiente de
entrenamiento y se registra el rango efectivo. `krylov_steps` controla el
horizonte de entrenamiento, que también debe comprobarse al refinar el rango.
`marginal_rank=None` conserva todos los controles para comparaciones pequeñas.
El límite de dimensión se comprueba antes de construir una matriz identidad
marginal completa o una matriz de Schur.

## Formas de Schur real y compleja

Ambas funciones de solución reducida usan `schur_output="real"` por defecto.
La forma de Schur generalizada real conserva los modos conjugados complejos
en bloques de 2×2; el solucionador utiliza los bloques estables completos
para recuperar la transición y la respuesta de los controles. Los valores
propios complejos no implican una política económica compleja. Véase la
[documentación de Schur generalizada de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ordqz.html).

Use `schur_output="complex"` para comparar con la descomposición anterior.
La elección se registra en `policy.diagnostics["schur_output"]`. Ambas rutas
usan las mismas bases y ecuaciones proyectadas, número de raíces estables,
margen respecto al círculo unitario, límite de condición de la proyección de
estados y tolerancias de ecuaciones. No hay un reintento automático con otra
descomposición. Un fallo se conserva para la configuración solicitada.

La forma real reduce la aritmética del paso de Schur denso. La mejora total
depende del modelo: las respuestas de hogares y el entrenamiento de bases
pueden dominar una solución de dos activos. El algoritmo sigue usando un
sistema reducido denso; no proporciona un método iterativo de subespacios
estables. Para medir ambas rutas en el equipo actual, con auditorías de grilla
completa y rangos que históricamente fallaron, ejecute:

```sh
PYTHONPATH=. python tools/reference_validation/benchmark_hank_schur.py \
    --output reviews/hank-schur-benchmark --repeats 3
```

Use un directorio nuevo. Cada solución usa un proceso nuevo y secuencial;
el registro separa el tiempo total, Schur, construcción de bases y memoria
residente máxima. La descomposición está incluida en la fase de política de
Schur, por lo que esos dos tiempos no deben sumarse. El tiempo no es una
prueba de exactitud; una descomposición más rápida no certifica la precisión
no lineal.

## Ecuaciones y aceptación

Sean `B` las direcciones retenidas de la distribución y `V` las de los
valores marginales. La política usa `D = D_ss + B z` y `m = m_ss + V v`. Cada
coordenada reducida se expande antes de evaluar la derivada direccional de los
hogares en la grilla completa; la ecuación marginal se proyecta con `V.T`.
QZ ordenado resuelve este sistema proyectado. Un número incorrecto de raíces
estables, una raíz unitaria, una proyección singular del estado o errores en
las ecuaciones proyectadas producen una excepción. No se elimina ninguna
raíz para fabricar determinación.

`policy.converged` y `simulate(...).converged` certifican **solamente las
ecuaciones proyectadas**. Deben leerse junto con los siguientes diagnósticos:

| Diagnóstico | Significado |
| --- | --- |
| `projected_equilibrium_residual_max_abs` | Residuo de las matrices reducidas resueltas |
| `full_marginal_relative_residual_max_abs` | Máximo coeficiente de una ecuación omitida, dividido por `1 + abs(m_ss)`, ante una dirección unitaria del estado |
| `full_marginal_stationary_relative_rms_max` | Mayor RMS marginal normalizado bajo la covarianza estacionaria de los choques |
| `directional_projection_error` | Diferencia entre una evaluación direccional nueva y la solución proyectada ensamblada |
| `reduced_jacobian_bytes`, `basis_bytes` | Tamaños de los arreglos almacenados; no son la memoria máxima del proceso |

Los errores por coordenada unitaria dependen de la normalización del estado.
El RMS estacionario y los errores en trayectorias de evaluación consideran la
escala de los choques. Un error de entrenamiento o un residuo proyectado
pequeños no certifican las ecuaciones omitidas.

`audit` transporta la distribución completa, real y no negativa. Resuelve la
consistencia entre continuación y transporte manteniendo fijos los
coeficientes de política agregada y comprueba **todas** las ecuaciones
marginales. La aceptación exige errores marginales normalizados y errores de
mercados/bienes inferiores a `tolerance`, error presupuestario inferior a
`1e-8`, error de masa inferior a `1e-10`, ausencia de masa negativa
material y de salidas de la grilla, y soluciones de continuación satisfactorias.
Los errores de interpolación Euler/KKT se reportan por separado. Nunca se
recortan silenciosamente las distribuciones lineales con signo ni los
valores marginales no positivos.

## Refinamiento y límites

Compare rangos de distribución y marginales, horizontes de entrenamiento y
tamaños de choques. Use el solucionador estocástico denso existente como
comparación independiente en grillas pequeñas y transiciones largas en
espacio de secuencias cuando estén disponibles. El ejemplo de 500 por 7
requiere más direcciones de distribución que los valores predeterminados
para grillas pequeñas: las pruebas preliminares con rangos de 4 a 32 fallaron
la condición de determinación; con 48 o más, el sistema se resolvió. Esto
justifica revisar los rangos, no relajar la comprobación de raíces.

En una grilla de dos activos de 40 por 40, el refinamiento marginal puede
lograr trayectorias lineales muy similares mientras una auditoría no lineal
con choques mayores aún falla el criterio marginal completo. Un residuo de
mercado pequeño no basta. Los registros finales deben reportar estos fallos
junto con los casos satisfactorios de menor amplitud.

El enfoque sigue la separación entre funciones estacionarias de los hogares
y perturbaciones agregadas reducidas descrita por [Bayer y Lütticke (2020)](https://www.ralphluetticke.com/publication/bl_qe/).
Esta implementación utiliza una base Krylov/SVD de respuestas de los hogares;
no pretende reproducir su implementación de cosenos dispersos ni su
calibración.

## Otros cierres agregados

`solve_hank_reduced_kernel` acepta el protocolo compartido de pasos hacia
atrás, transporte y mercados. Los métodos opcionales `aggregate_state_names`,
`prices_from_state(controls, aggregate_state, distribution, *, lag_q=None)` y
`next_aggregate_state(prices, aggregate_state, rho)` permiten agregar estados
predeterminados. La innovación monetaria se ubica por nombre, no por la última
posición del arreglo. El método de precios debe respetar un precio rezagado
observado cuando se suministra durante auditorías fuera del equilibrio. Los
kernels deben proporcionar derivadas direccionales; el kernel anterior de
dos activos con caché completa se rechaza antes de iniciar la construcción.

`feedback` se expande a todos los controles marginales seguidos por los
agregados; `reduced_feedback` y las matrices proyectadas usan los coeficientes
de la base marginal. `_forecast(next_state, current_nominal)` conserva la
semántica existente de continuación estocástica con valores marginales
completos.

## Evidencia numérica conservada

El registro histórico con Schur complejo de `tests/fixtures/hank_scaling/`
vincula fuentes, matrices y trayectorias por SHA-256. Su manifiesto apunta a
una copia exacta archivada de `hank_reduction.py`; se conservan los hashes
originales y los artefactos numéricos. Usa procesos nuevos y verifica un solo
hilo de BLAS.
La grilla de un activo de 500×7, con rangos 96 y 128, tiene 229 ecuaciones;
la de dos activos de 40×40×2, con rangos 24 y 48, tiene 78. Ambas usan 96
pasos de entrenamiento de Krylov. El rango marginal 128 también funcionó en
la segunda grilla, pero el rango 80 con ese horizonte falló la comprobación
de una política real en ese registro de Schur complejo. Es evidencia
histórica, no una afirmación para todas las descomposiciones. Los rangos no
forman una regla monótona de aceptación.

En grillas pequeñas, reducir los controles cambia las trayectorias agregadas
retenidas como máximo en `5.8e-9` para un activo y `5.1e-11` para dos. Para
500×7, las primeras 30 respuestas difieren del solucionador independiente en
el espacio de secuencias, con horizonte 200, en menos de `7.33e-8`.

Las auditorías no lineales de 48 fechas, semilla 9917 y desviación estándar
`1e-4`, fallan con estas políticas fijas de primer orden. Con una cuarta parte
de la amplitud, 40×40 pasa; 500×7 aún presenta error de mercado `1.36e-5`,
superior al umbral sin cambios de `1e-5`. Se conservan por separado los errores
de ecuaciones marginales, de Euler/grilla y de mercados. La
[continuación no lineal](hank_nonlinear_continuation.md) resuelve el equilibrio
condicional finito y mantiene sus propias comprobaciones de horizonte y ramas.

La memoria residente incluye importaciones y calibración; los bytes de las
matrices se registran aparte. Los tiempos se midieron mientras otras pruebas
compartían la máquina y no garantizan una aceleración en un equipo exclusivo.
