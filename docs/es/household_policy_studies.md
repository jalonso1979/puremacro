> 🇬🇧 [English](../household_policy_studies.md) · 🇪🇸 Español

# Estudios de política con hogares observados

Los ejemplos de desarrollo 6.1 conectan gasto observado, compensación con
financiamiento explícito y comparaciones robustas. Los precios contrafactuales
y las elasticidades son supuestos; no son estimaciones causales de un arancel.

```sh
python -m puremacro.examples.household_trade_policy --output estudio-enigh
python -m puremacro.examples.trade_adjustment_households --output estudio-ajuste
```

Los directorios deben ser nuevos. Se conservan tablas CSV, manifiestos JSON,
supuestos, reglas fiscales y evidencia numérica. El primer ejemplo también
copia los insumos autenticados al directorio de resultados.

## Fuente y población analizada

Fuente: [INEGI, ENIGH 2022 nueva serie](https://www.inegi.org.mx/programas/enigh/nc/2022/),
tabla `concentradohogar`. Se conservan las 90 102 filas originales, el factor
de expansión, el estrato y la UPM. Las 69 filas con gasto monetario cero,
que representan 21 945 hogares, están fuera del dominio de preferencias con
presupuesto positivo. Permanecen en el archivo original y en el registro de
exclusiones. El análisis utiliza 90 033 observaciones que representan
37 538 178 hogares.

Los importes son pesos trimestrales a precios de agosto de 2022. `alimentos`
incluye alimentos, bebidas y tabaco; el resto es `gasto_mon - alimentos`.
Los quintiles ordenan el ingreso corriente por integrante, ponderado por el
factor de expansión del hogar. Los empates permanecen en el quintil inferior.
No son quintiles ponderados por personas. Se conservan las huellas SHA-256 del
ZIP, CSV y archivo distribuido. El ingreso corriente medio de todos los hogares
es 63 695,46 pesos, consistente con la cifra publicada redondeada.

Se aplican los [términos de libre uso del INEGI](https://www.inegi.org.mx/inegi/terminos.html).
Debe conservarse la atribución y los metadatos. Las transformaciones y el
análisis son de PureMacro, sin aval del INEGI.

## Políticas, bienestar y financiamiento

El estudio estático combina aumentos del precio de alimentos del 5%, 10% y
20% con elasticidades CES de 0, 1 y 2. Cada hogar conserva su propia canasta.
El bienestar no lineal se calcula antes de agregar con factores de expansión.

Los programas destinan el mismo presupuesto bruto, 1% del gasto monetario
agregado inicial: beneficios iguales por hogar, dirigidos a los dos quintiles
inferiores o proporcionales al gasto alimentario inicial. Impuestos de suma
fija proporcionales al gasto monetario inicial financian los beneficios.
El caso sin compensación es un control con presupuesto cero. No se modelan
respuestas de trabajo, ahorro, base fiscal o elegibilidad; las transferencias
netas se gastan íntegramente.

`balanced_compensation` devuelve beneficios, impuestos, transferencias netas
y el balance agregado. `compute_distributional_welfare` acepta estas
transferencias mediante `household_transfers`; su suma ponderada debe coincidir
con `transfer_total`, cero en un programa equilibrado. Una EV positiva indica
una ganancia a precios iniciales.

Se comparan por separado la EV media de toda la población analizada y la de
los dos quintiles inferiores. Son objetivos distintos y pueden ordenar las
políticas de forma diferente. No se reportan intervalos de confianza muestrales:
los estados de incertidumbre representan supuestos económicos. La identificación
causal y la inferencia con el diseño muestral requieren trabajo adicional.

## Resultados conservados

La ejecución desde el wheel instalado se conserva en
`tests/fixtures/household_trade_policy/`. Con aumento del primer precio del 10%
y elasticidad CES uno, la EV media en pesos trimestrales iniciales es:

| Política | Todos los hogares | Dos quintiles inferiores |
|---|---:|---:|
| Sin compensación | -1406,05 | -1140,51 |
| Beneficios iguales | -1408,01 | -1018,04 |
| Beneficios focalizados | -1409,81 | -444,96 |
| Beneficios por gasto alimentario | -1408,62 | -1090,04 |

La focalización reduce la pérdida del grupo de menores ingresos en todos los
estados estáticos enumerados; no la elimina en este ejemplo. Redistribuir
recursos nominales entre hogares con índices de precios distintos también
cambia la EV monetaria media de la población. Por ello cada objetivo tiene
sus propias tablas de resultados y arrepentimiento. El ordenamiento depende
del modelo y la muestra; no establece una política óptima empírica.

La fórmula CES independiente de dos bienes coincide con la EV por hogar con
error inferior a `1.1e-15` del presupuesto inicial. Desde el repositorio se
regeneran ambos estudios, huellas de código, tablas y verificaciones mediante:

```sh
PYTHONPATH=. python tools/reference_validation/check_household_trade.py --output evidencia
```

## Comparaciones robustas

`puremacro.study.compare_policy_payoffs` recibe una matriz política por estado.
Reporta dominancia, resultado más desfavorable y arrepentimiento máximo,
definido como la distancia al mejor resultado disponible en cada estado.
Población, horizonte, numerario y definición de bienestar deben ser comunes y
explícitos. Una celda ausente, no finita o fallida impide ordenar toda la matriz.
Solo se calculan resultados esperados con probabilidades proporcionadas por el
usuario que sumen uno. La densidad de una cuadrícula no implica probabilidades.

## Ajuste comercial e incidencia intertemporal

El segundo ejemplo usa [abastecimiento dinámico](trade_adjustment.md): un arancel
del 10% o 25% encarece el insumo importado entre los periodos 2 y 5. Las
participaciones iniciales 0,7/0,3, elasticidad 3 y costos de ajuste 0 o 2 son
supuestos. Los costos de compra y ajuste se trasladan íntegramente al precio
del primer rubro de gasto. La recaudación arancelaria no se distribuye ni
financia la compensación; no es un cierre de cuentas nacionales.

`household_incidence_path` calcula el cambio constante de consumo que produce
la misma utilidad CES-CRRA durante un horizonte finito común. El descuento
trimestral es 0,99 y la curvatura 1 o 2; no hay ahorro. Los primeros doce
periodos reciben el mismo presupuesto bruto de compensación, con sus impuestos,
incluso antes y después del arancel. Los primeros doce
periodos del costo de abastecimiento se comparan con una solución de 24,
con tolerancia `1e-4`. Esto no establece el bienestar posterior al periodo doce.

Las pruebas incluyen fórmulas analíticas CES, conservación del financiamiento,
una solución independiente por programación dinámica discreta y la equivalencia
entre EV estática y bienestar intertemporal cuando la trayectoria es constante.
