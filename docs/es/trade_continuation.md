> 🇬🇧 [English](../trade_continuation.md) · 🇪🇸 Español

# Continuación de parámetros y arranques múltiples para equilibrios de política

`puremacro.trade.continuation` añade dos ayudas pequeñas sobre
[`solve_policy_equilibrium`](trade_policy.md): una trayectoria auditada en un
parámetro del modelo con una política arancelaria completa y fija, y un
arranque múltiple en un objetivo exacto. Ambas aceptan una etapa solo después
de reauditar el estado devuelto con el evaluador de contabilidad consistente, y
ambas adjuntan la misma salvedad: un equilibrio alcanzado a lo largo de una
trayectoria o desde un arranque alternativo no queda establecido como
perteneciente a ninguna otra rama, y una búsqueda sin resolver no demuestra ni
un pliegue ni la inexistencia.

```python
import numpy as np
from puremacro.trade import solve_policy_equilibrium
from puremacro.trade.continuation import sigma_path, try_starts
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
from puremacro.trade._oecd_icio import condense_final_demand
from dataclasses import replace

# Three countries, three sectors, OECD final-use layout condensed to C/I/Cx.
raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=3, seed=2)
raw = replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C*raw.K_F))
calibration = package_mrio_to_calibration_result(condense_final_demand(raw))

# A 10% import tariff by the first country on every foreign origin and use.
nc, ns, nfd = calibration.nc, calibration.ns, calibration.n_final_demand
tau, tau_fd = np.ones((ns*nc, ns, nc)), np.ones((ns*nc, nfd, nc))
tau[ns:, :, 0], tau_fd[ns:, :, 0] = 1.1, 1.1

# Follow the intermediate sourcing elasticity from Leontief (0) to 2 at that policy.
path = sigma_path(calibration, tau, tau_fd, 2., tol=1e-9)
print(path.summary())
print(path.to_dataframe()[["fraction", "sigma", "method", "iterations", "audited_max_residual"]])

# The endpoint is an ordinary consistent-accounting equilibrium.
direct = solve_policy_equilibrium(calibration, tau, tau_fd, sigma=2., tol=1e-9)
print(np.max(np.abs(path.final.x_sol - direct.x_sol)))  # 3.1e-11 on 2026-09-22

# Several warm starts at one exact target; the first audited root wins.
starts = [{"name": "calibrated", "x0": None},
          {"name": "from_path", "x0": path.final, "provenance": {"path": "sigma 0 -> 2"}}]
multi = try_starts(calibration, dict(tau=tau, tau_fd=tau_fd, sigma=2., tol=1e-9), starts)
print(multi.to_dataframe())
```

Se usa la semilla 2 porque la semilla sintética por defecto produce una
participación de gasto negativa tras la condensación, que la contabilidad
consistente rechaza. El ejemplo condensa los seis usos finales sintéticos con
`condense_final_demand` del módulo interno `puremacro.trade._oecd_icio` (no
forma parte de la API pública, así que puede cambiar sin aviso de
obsolescencia) y pasa impuestos nulos sobre el uso final porque esa ayuda no
acepta una tabla sin ellos.

## Qué calculan las ayudas

`continue_parameter(solve, path, start, ...)` es el motor genérico. `path(f)`
devuelve los argumentos con nombre del solucionador en la fracción `f` de la
trayectoria, con `f` en `[0, 1]`, por ejemplo `{"tau": tau, "tau_fd": tau_fd,
"sigma": 2*f, "tol": 1e-9}`. `solve` es o bien un `TradeCalibrationResult`, en
cuyo caso cada prueba llama a `solve_policy_equilibrium` con esos argumentos,
`method="newton"` y el estado aceptado anterior como arranque (véase la
sección siguiente), o bien un invocable `solve(kwargs, x0)` que devuelve un
`TradeEquilibriumResult` de contabilidad consistente (entonces `calib=` es
obligatorio para la auditoría). `start` debe resolver ya las ecuaciones en
`path(start_fraction)`: se audita, nunca se vuelve a resolver, de modo que una
etapa guardada se reanuda en su fracción real.

`sigma_path(calib, tau, tau_fd, sigma_target, sigma_start=0., ...)` es el
envoltorio de conveniencia para la elasticidad de sustitución de insumos
intermedios. La trayectoria es

```
sigma(f) = sigma_start + f (sigma_target - sigma_start),   f in [0, 1]
```

con los extremos exactos en `f = 0` y `f = 1` (`f in [0, 1]` significa `f` en
el intervalo `[0, 1]`). `f` nunca significa una fracción arancelaria: los
calendarios arancelarios intermedios y finales son completos y fijos a lo
largo de la trayectoria. Un `start` suministrado se audita en
`sigma(start_fraction)` sin volver a resolverlo (y entonces se rechaza `x0`).
Si se omite `start`, se resuelve en `sigma(start_fraction)` a partir de `x0`
con la escalera auditada completa de `solve_policy_equilibrium`
(`method="auto"` salvo que se indique un método): el arranque se audita como
uno suministrado, no se sigue, de modo que la regla de solo Newton descrita
más abajo se aplica únicamente a las pruebas. La primera etapa registra el
método, la semilla y el uso de la escalera de esa resolución, y
`metadata["start_source"]` indica `"solved"` o `"supplied"`. Si la resolución
del arranque falla, se lanza `PolicyEquilibriumError` indicando la fracción,
antes de que exista ninguna etapa. Cuando `sigma_start == sigma_target` la
trayectoria solo audita el arranque.

`try_starts(solve, target, starts, max_iter=35)` prueba arranques con nombre
en los argumentos del objetivo, que no cambian. Cada arranque es un mapeo con
un `name`, un `x0` (un vector de equilibrio, un `TradeEquilibriumResult`, o
`None` para el estado calibrado) y una `provenance` opcional, que se guarda
literalmente. Un arranque cuyo vector no es finito o no tiene la longitud de
estado de la calibración se registra como no resuelto sin resolver nada. La
búsqueda se detiene en la primera raíz auditada; un arranque rechazado nunca
sustituye a una raíz, y una búsqueda sin resolver devuelve `final=None`.

## Arranques en caliente y `allow_fallback`

Como en las ayudas de IO, cada prueba es por defecto una resolución de Newton
desde el estado aceptado anterior (en `try_starts`, desde el arranque con
nombre), y cualquier otro resultado es una prueba rechazada que reduce el
paso a la mitad. Con el valor por defecto `allow_fallback=False`:

- el adaptador de la calibración llama a `solve_policy_equilibrium` con
  `method="newton"` y rechaza `method="auto"`, `"hybr"` y `"keller_pac"` con
  un `ValueError` antes de resolver nada, porque `"auto"` recorre la escalera
  de respaldo, `"hybr"` parte del estado calibrado y `"keller_pac"` parte del
  calendario sin aranceles, de modo que ninguno sigue la trayectoria;
- un estado devuelto cuyo registro del solucionador muestra la escalera de
  respaldo (`metadata["policy_solver_fallback_used"]`), o cuyo último intento
  registrado partió de algo distinto del arranque suministrado, es una prueba
  rechazada. Esto también se aplica a los estados que devuelve un invocable
  `solve` propio.

`allow_fallback=True` activa la escalera: el adaptador usa entonces por
defecto `method="auto"`, y una etapa aceptada puede ser una raíz auditada
alcanzada por `hybr` desde el estado calibrado o por la continuación
arancelaria de Keller desde el calendario sin aranceles. Esa etapa resuelve
las ecuaciones en los parámetros solicitados, pero no se alcanzó siguiendo la
trayectoria. Cada etapa y cada intento de `try_starts` registra `seed`
(`"warm"` o `"calibrated"`) y `fallback_used`, de modo que el registro deja
ver una raíz atribuida a un arranque con nombre cuyo vector nunca se usó.

## Ley de pasos

El avance sigue la ayuda de IO `continue_origin`. Con fracción actual `f` y
paso `h` (prueba, fallo y éxito en el bloque siguiente):

```
trial     = min(1, f + h)
failure   -> h <- h / 2; if h < min_step: stop, keep the last audited state
success   -> f <- trial; if iterations <= fast_iterations: h <- min(max_step, growth h)
```

Es decir: tras un fallo el paso se reduce a la mitad y, si queda por debajo de
`min_step`, la trayectoria se detiene conservando el último estado auditado;
tras un éxito la fracción avanza y, si la prueba necesitó como mucho
`fast_iterations` iteraciones, el paso crece por `growth` hasta `max_step`.

Valores por defecto: `initial_step = 0.25`, `min_step = 1/1024`,
`max_step = 0.5`, `growth = 1.5`, `fast_iterations = 6`, los de
`continue_origin`; la ayuda de IO `continue_ces` usaba un tope de 0.4, cinco
iteraciones rápidas y `min_step = 1/128`, que pueden pasarse. `iterations` es
el recuento propio del solucionador: iteraciones de Newton, evaluaciones de
función de `hybr` de SciPy, o pasos de continuación de Keller. Hay dos
diferencias deliberadas con el bucle de IO. Una prueba a menos de `1e-12` de
la fracción uno se toma como uno, lo que evita una etapa final duplicada por
acumulación de coma flotante. Las ayudas de IO también rechazaban toda prueba
que necesitara más de 25 iteraciones de Newton (`continue_ces`: 12); aquí el
tope por prueba es el `max_iter` del solucionador (100 en
`solve_policy_equilibrium`), así que pase `max_iter=25` para reproducir esa
regla. Una parada lanza `ParameterContinuationFailure`, cuyo `state` es el
último equilibrio auditado (nunca la prueba rechazada) y cuyo `result` es el
`ParameterContinuationResult` parcial con `status="unresolved"`, todas las
etapas aceptadas y todas las pruebas rechazadas.

## Aceptación e invariantes

Una prueba se acepta solo cuando el estado devuelto

1. declara convergencia,
2. lleva contabilidad consistente, transferencias de suma fija y exactamente
   el `sigma` solicitado y los calendarios arancelarios intermedios y finales
   solicitados (por igualdad de arrays),
3. supera la reevaluación de contabilidad consistente de todas las ecuaciones
   de equilibrio y contables, incluidos el mercado de bienes omitido y los
   saldos exteriores, en su tolerancia registrada, con todos los campos
   reportados (precios, producciones, salarios, rentas del capital,
   transferencias, índices de precios, demanda final, recaudación arancelaria
   y PIB) iguales al equilibrio recalculado con `rtol = atol = 1e-8`,
4. tiene un residuo absoluto máximo auditado no mayor que la `tol` solicitada
   (por defecto `1e-8`, la de `solve_policy_equilibrium`; absoluta en las
   unidades de la calibración), y
5. salvo con `allow_fallback=True`, no se alcanzó mediante la escalera de
   respaldo del solucionador ni, según su último intento registrado, partió de
   una semilla distinta del arranque en caliente.

La auditoría es el propio evaluador del modelo, el mismo que usa
`solve_policy_equilibrium`; no es un oráculo derivado por separado. Un estado
que incumple el criterio 2 es un contrato roto y lanza `ValueError` de
inmediato; un estado que incumple cualquier otro criterio es una prueba
rechazada. Un `ValueError` o `TypeError` lanzado por el solucionador significa
entradas inválidas y se propaga; `RuntimeError` (por ejemplo
`PolicyEquilibriumError`), `ArithmeticError` y `LinAlgError` son fallos
numéricos que reducen el paso a la mitad. Los equilibrios heredan el
numerario y el cierre del saldo exterior de `solve_policy_equilibrium` (véase
la [contabilidad consistente](trade_accounting.md)). En la tabla sintética
(observado el 2026-09-23, no fijado por una prueba), reescalar todos los
flujos por 1000 con `tol` escalada igual, o permutar países y sectores
manteniendo en su lugar el primer país y el primer sector, reprodujo la
trayectoria con una precisión de alrededor de `1e-14`; poner otro país en
primer lugar, y con ello cambiar el numerario, cambió las producciones en
torno a un 1% y los salarios en 0.15.

Por defecto, todo argumento con nombre que `path` deja sin cambiar entre las
fracciones 0 y 1 se resume con una huella (calendarios arancelarios por sus
bytes, escalares por valor, estados de equilibrio por su resumen) y debe ser
idéntico en cada prueba; `continued=` nombra explícitamente los argumentos que
pueden cambiar e `invariant=` sustituye la huella. Una violación se lanza antes
de resolver la prueba. El resumen de la huella se guarda como `invariant_id`.
Una trayectoria identidad (nada continuado) audita igualmente el estado
suministrado y reporta la fracción uno.

## Resultados

`ParameterContinuationResult` contiene `fractions`, `parameter_values`, las
`stages` y `failures` serializables en JSON (sin vectores de estado), `final`,
`status`, `start_fraction`, `last_fraction`, `invariant_id`, `qualification` y
`metadata`. Cada etapa registra el parámetro, el método, la semilla, si se usó
la escalera de respaldo, las iteraciones, el residuo del solucionador, el
residuo auditado, la tolerancia solicitada, los segundos, el paso usado, y el
mínimo de precio, de razón de producción, de salario y de gasto final. Cada
prueba rechazada registra el error y, cuando el solucionador lanzó
`PolicyEquilibriumError`, `solver_attempts`: el método, la semilla, el
indicador de convergencia, los residuos y el error de cada intento; el mensaje
de `ParameterContinuationFailure` termina con el último de ellos.
`keep_states=True` conserva cada `TradeEquilibriumResult` aceptado en
`states`; `stage_callback(state, stage)` recibe cada estado aceptado con una
copia independiente de su registro para que quien llama pueda persistirlo;
`progress(event)` recibe los eventos de prueba, aceptación y rechazo.
`metadata=` se fusiona con los metadatos del resultado, y las claves que el
resultado registra por sí mismo (`method`, `solver`, `audit`,
`sigma_target`, ...) se rechazan con un `ValueError`. `to_dataframe()`,
`to_markdown()`, `to_latex()`, `to_typst()` y `summary()` siguen las
convenciones del paquete, y `plot(values, ax=...)` dibuja un escalar por etapa
frente al parámetro: el residuo auditado por defecto, una clave de etapa, una
secuencia, o un invocable aplicado a los estados conservados (por ejemplo la
VE hicksiana de un país con `compute_hicksian_welfare`).

`DirectTargetResult` contiene `attempts` (nombre, procedencia, estado, error o
diagnósticos del solucionador, incluidos `seed` y `fallback_used`),
`accepted_start`, `final`, `status`, el `target` con huellas, `qualification`
y `metadata`, con los mismos exportadores de tablas.

`final` es un `TradeEquilibriumResult` ordinario de contabilidad consistente:
[`compute_hicksian_welfare`](trade_welfare.md) y las funciones de
posprocesamiento lo aceptan sin cambios.

## Qué está validado y qué no

Las pruebas de `tests/test_trade_continuation.py` comprueban, en la tabla
sintética 3 x 3 anterior (`tol=1e-9`): el extremo de la trayectoria en sigma
frente a un `solve_policy_equilibrium(sigma=2)` directo (diferencia máxima en
el vector de estado `3.1e-11`, tolerancia `1e-9`; fracciones aceptadas 0,
0.25, 0.625 y 1, cada prueba una resolución de Newton en caliente); la
reanudación desde una etapa guardada (`1.2e-11`); la conservación del
arranque tras un fallo, con los intentos del solucionador registrados; las
violaciones de invariantes y del contrato de parámetros lanzadas antes de
resolver o sin resolver; la trayectoria identidad; la validación de ajustes,
métodos y metadatos; el contrato de `allow_fallback` (estados de la escalera
de respaldo o de semilla calibrada rechazados por defecto y aceptados y
etiquetados con la opción); `try_starts` devolviendo solo raíces auditadas;
registros JSON, serialización con pickle, tablas y gráficos; y la ley de pasos
bajo resultados guionizados del solucionador. Una prueba de paridad ejecuta
`continue_origin` y `try_direct_target` de IO en un subproceso (omitida cuando
el volumen de investigación no está montado) con los mismos resultados
guionizados, incluida una etapa con exactamente `fast_iterations` iteraciones
y un paso limitado a 0.5, y exige fracciones aceptadas, pruebas rechazadas,
estados y últimas fracciones idénticos.

En el fixture OECD incluido de 3 regiones x 3 sectores en millones de USD,
`sigma_path(calib, tau, tau_fd, 0.5, tol=1e-5)` con un arancel uniforme del
10% a las importaciones de EE. UU. coincide con
`solve_policy_equilibrium(calib, tau, tau_fd, sigma=0.5, tol=1e-5)` con una
precisión relativa de `6.6e-12` en precios y `6.7e-12` en PIB (2026-09-23,
tras las correcciones del solucionador en modo consistente; la prueba exige
`1e-6`). Todas las etapas, incluido el arranque en sigma 0, son resoluciones
de Newton (el arranque requiere 6 iteraciones hasta `4.8e-8`). Antes de esas
correcciones, Newton se detenía en sigma 0 con `converged=False` en un residuo
de `8.3e-6` porque su regla de parada no comprobaba las ecuaciones físicas
auditadas, y el arranque procedía del respaldo de Keller. La trayectoria de 0
a 2 queda **sin resolver** en este fixture: se detiene en la fracción 0.450
(sigma 0.900), y la trayectoria de 2 a 0 se detiene en la fracción 0.479
(sigma 1.041). Las llamadas directas a `solve_policy_equilibrium` en sigma
0.91, 0.95, 1.0, 1.01 y 1.04 fallan con Newton, híbrido y continuación de
Keller; en sigma 1, 1.01 y 1.04 el intento de Keller alcanza residuos de
`4.9e-8`, `9.9e-6` y `1.0e-7`, pero en una raíz con gasto final negativo, que
la auditoría rechaza por demanda no factible. Sigma 0.9 se resuelve con Newton
y 1.05 mediante el respaldo de Keller. La herramienta reporta esa región como
etapas auditadas más un `ParameterContinuationFailure`; no la resuelve.
Es una observación sobre la pila de solucionadores existente cerca de la
elasticidad unitaria en esta tabla y tolerancia, no una propiedad del
algoritmo de trayectoria; una prueba marcada como lenta fija el contrato
(etapas auditadas, estado conservado auditado) y no el fallo.

La tabla incluida de 77 países x 11 sectores no puede usarse con estas ayudas:
su calibración tiene tres celdas de inversión negativas, que la contabilidad
consistente rechaza antes de resolver nada (una prueba fija el `ValueError`).
La [documentación contable](trade_accounting.md) explica por qué un agregado
de existencias con signo negativo requiere agregación o un modelo explícito
de inventarios.

Las ayudas de IO se escribieron para recuperar equilibrios GTAP de baja
elasticidad y no lo lograron: los once registros `recovery.json` bajo
`retaliation_2026-09-22/results_latest/cases/*/runs/gtap/low/` están
`unresolved`, con la homotopía en el origen deteniéndose entre las fracciones
0.92 y 0.99 y todos los arranques directos rechazados. La salvedad de IO dice
lo siguiente (original en inglés); `result.qualification` adjunta una versión
adaptada a trayectorias de parámetros:

> An admissible exact-target equilibrium found from an alternative start or
> production-elasticity path is not established as belonging to the forward
> low-elasticity tariff branch. Local stability is reported separately;
> neither convergence nor unsuccessful searches prove global uniqueness or
> nonexistence.

y, sobre el rechazo temprano de pruebas lentas, "This can reject a slowly
converging trial; it does not diagnose a fold or prove nonexistence".
puremacro no implementa esa guardia de rechazo temprano: una prueba se rechaza
solo cuando el solucionador lanza una excepción, el registro del solucionador
muestra una semilla distinta del arranque en caliente, o la auditoría falla.
Este módulo no evalúa la estabilidad local de los estados aceptados.

## Documentación relacionada

- [Política arancelaria hicksiana y recuperación auditada del equilibrio](trade_policy.md):
  el solucionador auditado que llama cada etapa (Newton desde el arranque en
  caliente por defecto; la escalera completa Newton, híbrida y de Keller con
  `allow_fallback=True`).
- [Contabilidad de comercio consistente](trade_accounting.md): las ecuaciones
  que reevalúa la auditoría de cada etapa, y el cierre del numerario y del
  saldo exterior.
- [Bienestar hicksiano de consumo](trade_welfare.md): bienestar a lo largo de
  una trayectoria a partir de los estados conservados.
- [Estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
