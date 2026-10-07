> 🇬🇧 [English](../trade_distributional.md) · 🇪🇸 Español

# Incidencia distributiva en los hogares

`puremacro.trade.distributional` combina canastas observadas de gasto por grupo
con precios de comprador, cambios en ingresos nominales y transferencias
fiscales proporcionados por el usuario. Calcula variación equivalente (EV) y
variación compensatoria (CV) estáticas exactas mediante el
[motor de preferencias de los hogares](trade_household.md).

Es **incidencia de equilibrio parcial condicionada a los precios e ingresos
proporcionados**. Los precios pueden proceder de una solución de equilibrio
general auditada o de un escenario supuesto explícitamente. La demanda de los
hogares no retroalimenta producción, comercio ni precios de factores. El
resultado no es un nuevo equilibrio general ni una estimación causal del
efecto de un arancel. Una canasta observada no identifica transmisión a precios,
respuestas del ingreso ni elasticidades de preferencias.

## Medias de grupo y factores de expansión

`prepare_household_groups` recibe:

| Entrada | Significado requerido |
|---|---|
| `expenditure` | DataFrame con grupos en filas y sectores etiquetados en columnas; **gasto monetario medio por unidad representada**, no totales del grupo |
| `population_weights` | Conteos de expansión positivos por grupo, en la misma unidad representada |
| `baseline_budget` | Series opcional que debe coincidir con la suma del gasto de cada fila |
| `income_exposure` | DataFrame opcional de grupo por fuente de ingreso, con ingresos nominales iniciales por unidad representada |
| `monetary_unit`, `period`, `population_unit` | Escala monetaria, periodo de gasto y unidad representada explícitos |
| `provenance` | Diccionario de procedencia; `source` es obligatorio y se conservan las marcas de datos sintéticos o fixtures de regresión |

Las etiquetas deben ser cadenas únicas y no vacías. Las entradas se alinean
por etiquetas; las etiquetas faltantes, adicionales o duplicadas producen un
error. Los precios deben ser positivos y finitos. Gasto y exposición de ingreso
deben ser finitos y no negativos; cada grupo necesita gasto total positivo.
No se eliminan grupos ni se imputan observaciones silenciosamente.

Con microdatos y factores de expansión `w_i`, calcule el gasto medio sectorial
como `sum(w_i * gasto_i_sector) / sum(w_i)` y el conteo del grupo como
`sum(w_i)`. Use la misma muestra completa y el mismo denominador para **todos
los sectores**, incluidos los hogares que no compraron en un sector. Un
promedio publicado entre compradores positivos no es el promedio incondicional
del grupo. No sume repetidamente los factores de expansión en cada registro
sectorial. La unidad de gasto debe concordar con ponderaciones de hogares o
personas.

Las medias definen grupos representativos. El bienestar no lineal calculado
con esas medias no equivale al bienestar individual promedio ponderado de los
hogares. Cuando esa diferencia importe y los datos lo permitan, use una fila
por hogar.

El ingreso expuesto puede superar el gasto observado: ahorro, recursos no
monetarios y otros ingresos pueden diferir. El escenario supone que **todo el
ingreso nominal incremental expuesto se gasta**, manteniendo fijo el resto del
gasto inicial. Es un supuesto de comportamiento explícito, no una propensión
a consumir estimada. Las fuentes pueden distinguir factor y sector, por
ejemplo `manufacturing_labor` o `agriculture_capital`.

## Un escenario sintético pequeño

Estas cifras son inventadas para mostrar la interfaz. No constituyen una
economía calibrada, una estimación de encuesta ni un pronóstico arancelario.

```python
import pandas as pd
from puremacro.trade.distributional import (
    prepare_household_groups, compute_distributional_welfare,
)

spending = pd.DataFrame(
    [[60.0, 40.0], [40.0, 160.0]],
    index=["lower_income", "higher_income"], columns=["food", "other"],
)
groups = prepare_household_groups(
    spending,
    pd.Series([300.0, 100.0], index=spending.index),
    income_exposure=pd.DataFrame(
        {"labor": [80.0, 100.0], "capital": [20.0, 100.0]}, index=spending.index,
    ),
    monetary_unit="illustrative currency units", period="quarter",
    population_unit="households",
    provenance={"source": "two-group synthetic illustration", "is_synthetic": True},
)
result = compute_distributional_welfare(
    groups,
    base_prices=pd.Series({"food": 1.0, "other": 1.0}),
    prices=pd.Series({"food": 1.20, "other": 0.90}),
    factor_income_changes=pd.Series({"labor": 0.10, "capital": -0.05}),
    transfer_total=1400.0,
    transfer_weights=pd.Series({"lower_income": 2.0, "higher_income": 1.0}),
    rule="cobb_douglas",
)
print(result.groups[["baseline_budget", "income_change", "transfer", "ev", "ev_pct"]])
print(result.aggregate[["total_transfer", "total_ev", "ev_pct", "mean_ev_pct"]])
print(result.summary())
result.plot()
```

`factor_income_changes` usa **fracciones**: `0.10` significa +10% y `-1` la
pérdida completa de esa fuente. Sus etiquetas deben coincidir exactamente con
las columnas de exposición. Omitir cambios significa mantener el ingreso.

La razón `prices / base_prices` normaliza los precios sectoriales respecto al
gasto inicial suministrado. Los dos estados y los cambios nominales de ingreso
deben usar un numerario común. Escalar uniformemente precios y todos los
recursos nominales deja intacto el bienestar real.

## Transferencias y agregación

`transfer_total` es el **cambio agregado** en la recaudación asignada a los
hogares representados, en la moneda y periodo del gasto. Un valor negativo es
un retiro agregado. Se proporciona explícitamente: el puente no lo infiere del
ingreso del gobierno del modelo GE ni del ingreso factorial del hogar.

`transfer_weights` contiene puntajes no negativos de asignación **por unidad
representada**, `a_g`:

```text
transferencia_por_unidad[g] = transfer_total * a_g / sum_h(population_weight[h] * a_h)
sum_g(population_weight[g] * transferencia_por_unidad[g]) = transfer_total
```

Por defecto cada unidad recibe la misma transferencia; los totales por grupo
pueden diferir. En el ejemplo, cada hogar de menor ingreso recibe cuatro
unidades monetarias y cada hogar de mayor ingreso dos: `300*4 + 100*2 = 1400`.
Evite contar esas transferencias de nuevo dentro de los cambios de ingreso.
Los pesos normalizados representan una población de tamaño uno; el monto
agregado debe usar esa misma escala. Convierta unidades explícitamente si una
tabla GE usa millones y la encuesta usa moneda por hogar.

`result.groups` informa montos por unidad representada y porcentajes respecto
al gasto inicial de cada grupo. `result.aggregate.total_ev` y `total_cv` suman
esos montos con los factores de expansión. `ev_pct` agregado divide EV total
entre gasto inicial total; `mean_ev_pct` promedia los porcentajes de los grupos
usando pesos de población. Responden preguntas distintas. Ninguna agregación
es una función de bienestar social ni una comparación interpersonal de utilidad.

## Preferencias y atribución exacta

Las reglas son `"fixed_baskets"`, `"cobb_douglas"`, `"ces"` y `"stone_geary"`.
CES acepta `ces_elasticity`. LES acepta `expenditure_elasticities` como
DataFrame de grupo por sector y `supernumerary_share` como escalar o Series
por grupo. Omitir las elasticidades LES supone valores unitarios; omitir su
fracción supernumeraria usa el supuesto explícito 0.5 del motor existente.
La auditoría, la reparación de suma ponderada, la marca de fracción asumida
y los ajustes de elasticidades quedan en `metadata["preference_calibration"]`.
El gasto observado por sí solo no estima esos parámetros.

El motor define `EV = e(p0,u1) - e(p0,u0)` y
`CV = e(p1,u1) - e(p1,u0)`. Ambas son positivas ante ganancias; CV es el monto
que podría retirarse a los precios nuevos. Los porcentajes usan el gasto
inicial observado como denominador.

`price_ev`, `income_ev` y `transfer_ev` suman exactamente `ev`; las columnas
análogas `_cv` suman `cv`. Son diferencias de bienestar ancladas a la **misma
base original**, siguiendo **precios → ingreso factorial → transferencias**.
Esta atribución descriptiva depende del orden; no es causal ni un promedio
Shapley. Se verifica el dominio de preferencias en cada etapa. Una etapa de
precios LES que agote los recursos de subsistencia produce un error aunque
una transferencia posterior pudiera rescatar el estado final.

## Puente desde un modelo de comercio resuelto

`distributional_welfare_from_results(groups, calib, baseline, counterfactual,
country=..., factor_income_changes=..., **scenario_options)` obtiene los
precios sectoriales de un país mediante `household_prices_from_result` y
reutiliza su auditoría del equilibrio. Ambos estados deben haber convergido,
usar el mismo modo contable y conservar exactamente las etiquetas ordenadas
de países y sectores de la calibración. Los sectores de los hogares deben
coincidir con los de la calibración.

El mapeo explícito del ingreso es una Series desde las fuentes de ingreso del
hogar hacia cambios nominales fraccionales. Asociar salarios nacionales con
ingresos laborales de encuesta sigue siendo una decisión del investigador.
El presupuesto agregado de consumo GE **no** reemplaza los presupuestos de
los grupos. `baseline_tau_fd` y `tau_fd` suministran aranceles finales del modo
heredado; el modo consistente utiliza los esquemas registrados y reauditados.
Las canastas y restricciones sobre modelos flexibles son las del
[puente de precios de los hogares](trade_household.md).

Dentro de cada sector, todos los grupos heredan la canasta GE de orígenes
del país seleccionado. Las diferencias de gasto sectorial no identifican
cuotas domésticas/importadas particulares de cada grupo. Si un sector tiene
peso nulo en esa categoría GE pero gasto positivo en la encuesta, el puente
produce un error: el precio unitario de relleno del motor no puede valorar
ese gasto. Para emplear un precio sectorial estimado por separado o un mapeo
de orígenes explícito diferente, construya esos precios de comprador y llame
directamente a `compute_distributional_welfare`. El supuesto de orígenes
heredados y los sectores con ceros estructurales no utilizados quedan
registrados en los metadatos.

El resultado conserva procedencia de los hogares, calibración y metadatos de
ambos equilibrios. Se propaga la marca superior `is_regression_fixture` cuando
alguna entrada la contiene. Los resultados basados en un fixture de regresión
siguen siendo ejercicios con ese fixture; añadir gasto de encuesta no convierte
el escenario de precios GE en evidencia empírica.

Los resultados ofrecen `summary()`, `to_frame()` / `to_dataframe()`,
`to_markdown()`, `to_latex()`, `to_typst()` y `plot()`.
