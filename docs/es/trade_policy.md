> 🇬🇧 [English](../trade_policy.md) · 🇪🇸 Español

# Política arancelaria hicksiana y recuperación auditada del equilibrio

Seleccione `metric="hicksian_ev"` explícitamente en la búsqueda del arancel
unilateral, en la búsqueda de Nash multilateral o en la matriz de pagos de
bienestar. Esto usa [contabilidad consistente](trade_accounting.md), devoluciones
fiscales de suma fija y la [función de gasto de consumo](trade_welfare.md). Los
valores por defecto históricos y los alias `ev`, `equivalent_variation` y
`consumption` conservan sus objetivos heredados. No seleccionan el nuevo cálculo.

```python
from puremacro.trade import (
    solve_policy_equilibrium, compute_unilateral_optimal_tariff,
    solve_multilateral_nash_tariffs, compute_welfare_payoff_matrix,
)

# Choose an absolute tolerance appropriate for the calibration units.
ge_tol = 1e-5  # used for the bundled 77x11 table (a regression fixture)
base = solve_policy_equilibrium(calibration, sigma=2., tol=ge_tol)
options = dict(metric="hicksian_ev", base_equilibrium=base, sigma=2.,
               consumption_categories=(0,), ge_tol=ge_tol)
opt = compute_unilateral_optimal_tariff(
    calibration, country_idx="USA", tariff_max=.30, num_grid=21, **options,
)
game = solve_multilateral_nash_tariffs(
    calibration, player_countries=["USA", "CHN"], tariff_max=.30,
    best_response_grid_size=13, tol=1e-4, regret_tol=1e-6, **options,
)
matrix = compute_welfare_payoff_matrix(
    calibration, player_a="USA", player_b="CHN",
    optimal_a=.10, optimal_b=.20, **options,
)
```

Los países nombrados deben existir en la calibración. `sigma` es la sustitución
entre orígenes de los insumos intermedios; las cestas de uso final siguen siendo
Leontief. Las categorías de consumo son `(0,)` por defecto. Seleccione
categorías adicionales distintas de la inversión solo cuando pertenezcan a la
función de utilidad declarada. Es bienestar de consumo condicional, que incluye
el consumo público cuando los datos de entrada lo agregan.

## Una sola base de comparación

Cada búsqueda mantiene fijos el mismo equilibrio sin aranceles, las preferencias,
el numerario, los saldos externos y las categorías de consumo para cada
candidato y cada desviación. Si se omite `base_equilibrium`, la búsqueda lo
resuelve una vez. Una referencia suministrada debe superar el `ge_tol`
solicitado, usar el `sigma` solicitado y tener aranceles intermedios y finales
nulos. Los cambios en los impuestos domésticos quedan fuera de este experimento
de política.

El cierre del ahorro externo forma parte de esa referencia. Las búsquedas
unilateral, de Nash y de la matriz de pagos aceptan `foreign_saving_units`
(`"numeraire"` por defecto, o `"world_income"`) y lo transmiten a cada
equilibrio que resuelven: la referencia, cada candidato y cada desviación. Un
`base_equilibrium` suministrado debe haberse resuelto con el mismo cierre; en
caso contrario, la búsqueda lanza `ValueError`. El cierre se registra en
`metadata["foreign_saving_units"]`. Con saldos externos de referencia distintos
de cero, el cierre en numerario hace que los pagos dependan de qué país se
enumera primero, y `"world_income"` no (véase
[contabilidad consistente](trade_accounting.md)):

```python
base = solve_policy_equilibrium(calibration, sigma=2., tol=ge_tol,
                                foreign_saving_units="world_income")
options.update(base_equilibrium=base, foreign_saving_units="world_income")
```

En la tabla de dos países del cuaderno 63 (`sigma=2`, categorías de consumo
`(0, 2)`, `ge_tol=1e-9`), la VE del país A por su propio arancel del 10% es el
2.540428% del consumo de referencia con `"world_income"`, sea cual sea el país
que se enumere primero. Con `"numeraire"` es el 2.630850% si A va primero y el
2.495185% si va B. En 4.3.0 las búsquedas lanzaban `TypeError` con esta opción.

Los campos y curvas de bienestar en niveles contienen la variación equivalente
(VE) en unidades de valor de la calibración; la VE de referencia es cero. Las
ganancias porcentuales usan el **gasto de consumo seleccionado de referencia**,
nunca la VE de referencia. Los pagos de un bloque suman la VE monetaria de sus
miembros; los bloques estratégicos no deben solaparse. El bienestar mundial
incluye todos los países calibrados, también los que no son jugadores
estratégicos. Los diagnósticos de términos de intercambio son índices a nivel de
país; no están disponibles (`NaN`) para un bloque multipaís porque no hay
definido un índice de comercio exterior del bloque.

Los metadatos de Nash distinguen `player_regrets`, en unidades de valor, de
`relative_max_regret`, una **fracción** del gasto de consumo de referencia fijo
de cada jugador. Así, `regret_tol=1e-6` significa una millonésima de ese gasto,
no una millonésima de la VE ni un punto porcentual. Estas convenciones se
registran junto con la referencia y las ejecuciones del solver en los metadatos
del resultado.

## Qué establece la convergencia

El `method="grid"` unilateral evalúa la malla declarada. `method="bounded"`
refina además cada intervalo con un máximo local muestreado, incluidos los
intervalos extremos. El `method="best_response"` de Nash usa esta búsqueda de
malla/refinamiento para cada jugador y después evalúa desviaciones unilaterales
simultáneas en el perfil final. `converged=True` exige que tanto la brecha
arancelaria de la mejor respuesta sin amortiguar como el arrepentimiento
normalizado por el consumo cumplan sus tolerancias. Las actualizaciones
amortiguadas pequeñas por sí solas no establecen la convergencia. Alcanzar
`max_iter` puede devolver un candidato de equilibrio general auditado con
`converged=False` y diagnósticos finitos de la desviación final.

Lea `best_response_boundaries`, el techo arancelario y la resolución de la
malla. Una solución en la frontera está restringida por el intervalo impuesto.
El refinamiento local puede pasar por alto picos estrechos no muestreados; estas
comprobaciones no demuestran optimalidad global ni existencia o unicidad de un
equilibrio de Nash del juego continuo.

Las etiquetas de frontera (`metadata["boundary"]` en la búsqueda unilateral,
`metadata["best_response_boundaries"]` en Nash) valen `"lower"` o `"upper"`
cuando una tasa está a menos de `tol` de 0 o de `tariff_max`, e `"interior"` en
otro caso. Cuando las dos bandas de tolerancia se solapan (`tariff_max <= 2 *
tol`, registrado como `metadata["boundary_bands_overlap"]`), gana la cota más
cercana y el punto medio exacto es `"interior"`. En 4.3.0 se comprobaba primero
la cota inferior, de modo que con `tariff_max <= tol` una respuesta en el techo
se etiquetaba como `"lower"`. Los techos superiores a `2 * tol`, incluidos
todos los valores por defecto, no cambian. La ruta de Nash heredada (no
hicksiana) aplica la misma regla a sus `best_response_boundaries`.

La matriz de pagos usa las mismas dos acciones, cero y un arancel positivo fijo,
para cada jugador en las cuatro celdas. Si se omite, la acción positiva es el
óptimo unilateral del jugador frente a aranceles extranjeros nulos. Los `nash_a`
o `nash_b` suministrados se aceptan solo como acciones fijas, explícitamente no
verificadas como tasas de Nash. Tasas `optimal_*` y `nash_*` en conflicto lanzan
un error. Los aranceles positivos mutuos no son automáticamente un equilibrio de
Nash del juego continuo ni un dilema del prisionero.

## Recuperación y desviaciones fallidas

`solve_policy_equilibrium(method="auto")` prueba Newton, el híbrido de SciPy y
después la continuación por pseudo-longitud de arco de Keller. El primer intento
puede usar un arranque en caliente aceptado previamente; los intentos
posteriores reinician desde el estado calibrado. La continuación parte de
aranceles nulos y usa los calendarios objetivo completos de insumos intermedios
y de uso final. Todos los intentos conservan `sigma`, el cierre fiscal y la
tolerancia solicitada. La recuperación no relaja el umbral de aceptación.

Antes de cualquier intento, un calendario Leontief (`sigma=0`) se examina con
cotas de Collatz-Wielandt del radio espectral de la matriz de costes aumentada
con aranceles `B_tau = a * multiplier / (1 - t)`, con la cuña exacta del impuesto
a la producción. Una cota inferior de al menos uno certifica que no existe un
sistema de precios positivo: entonces se lanza `PolicyEquilibriumError` con una
lista `attempts` vacía y las cotas en su atributo `viability`, sin ejecutar
ningún solver. Cualquier otro resultado deja correr los intentos: `viable`,
`near_critical` (cotas en `[1 - 1e-6, 1)`; existe un sistema de precios, con
precios muy grandes), `unresolved` (las cotas abarcan el umbral) y
`not_applicable` para insumos intermedios CES (`sigma > 0`), donde la cota
Leontief no certifica nada. El examen se registra en
`equilibrium.metadata["policy_viability"]`. `foreign_saving_units`
(`"numeraire"` por defecto, o `"world_income"`) se transmite a cada intento;
véase [contabilidad consistente](trade_accounting.md).

Un resultado aceptado debe declarar convergencia, conservar los calendarios y la
tecnología solicitados y superar una reevaluación independiente de los campos
reportados y de todos los residuos de equilibrio y contables, incluidos el
mercado de bienes omitido y los saldos externos. Los métodos de los intentos, las
semillas, los avisos, los errores y los residuos aceptados se registran en
`equilibrium.metadata["policy_solver_attempts"]`.

Si todos los intentos fallan, `PolicyEquilibriumError` expone `attempts`; las
búsquedas de política adjuntan además el `profile` que falló. La búsqueda lanza
la excepción de inmediato. No produce ningún pago ni certificado de mejor
respuesta para esa desviación, no almacena en caché ningún estado fallido y
nunca lo usa como siguiente arranque en caliente. Un fallo del refinamiento
escalar también deja la optimización sin resolver. Use `ge_method`,
`ge_max_iter`, `ge_max_steps` y `ge_tol` para controlar las soluciones del
equilibrio de política con independencia de los límites de iteración del juego
exterior. Las opciones desconocidas se rechazan.

`ge_tol` acota los residuos absolutos en las unidades de la calibración. Debe
ser apropiado para su escala numérica: la tabla 77x11 incluida, nominalmente en
millones de USD, usa un `ge_tol=1e-5` solicitado explícitamente. Esa tabla es un
fixture de regresión de software, no una fuente de magnitudes de la OCDE (véase
[el aviso](ADVISORY.md), 2026-09-22). Una petición por debajo
de la resolución de coma flotante puede fallar incluso en la referencia; la
recuperación deliberadamente no relaja esa petición en silencio. Cambiar las
unidades monetarias puede exigir cambiar la tolerancia absoluta del equilibrio
general, mientras que el bienestar porcentual y el arrepentimiento normalizado
son invariantes a las unidades.

## Evidencia y alcance

Una ecuación de precios relativos CES de dos países derivada por separado,
cuentas fiscales algebraicas y una minimización primal del gasto proporcionan la
referencia. Los 1.681 perfiles de una malla arancelaria de 41 por 41 se evalúan
de forma independiente. Las elecciones unilaterales, el candidato de Nash
restringido, las desviaciones y los pagos con acciones fijas coinciden con esa
referencia. Las pruebas ejercitan además un fallo real de Newton con
recuperación híbrida, continuación forzada, recuperación agotada, falsa
convergencia, cobertura de desviaciones fallidas, escalado de moneda y
relajación diminuta.

Véanse [los resultados registrados](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-hicksian-policy/REPORT.md) y
[el cuaderno 63](https://github.com/jalonso1979/puremacro/blob/v4.3.0/notebooks/63_trade_wars_and_nash_tariffs_es.ipynb). El benchmark
alcanza su techo del 40%; no identifica un arancel óptimo interior ni valida una
predicción empírica de guerra comercial. Los instrumentos soportados son
`universal`, `final_only` e `intermediate_only`. Los juegos con gradientes, otros
cierres fiscales, los modelos flexibles/GPU y el envoltorio histórico
`benchmark_real_world_tariffs` no están disponibles para esta métrica. Evalúe
calendarios explícitos con el solver público y el evaluador de bienestar en
lugar de tratar las etiquetas históricas de escenarios como datos actuales de
política real.
