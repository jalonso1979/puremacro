> 🇬🇧 [English](../trade_accounting.md) · 🇪🇸 Español

# Contabilidad a precios de productor y de comprador en el modelo de comercio

Use el modo consistente para los contrafactuales nuevos. Este ejemplo se ejecuta
tal cual sobre la tabla de dos países equilibrada a mano que usa la comprobación
de referencia descrita más abajo:

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium

# Two countries (A, B), one good, final uses C and I, values in one currency unit.
Z = np.array([[10., 12.], [8., 14.]])                      # intermediate sales
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])  # final sales to A-C, A-I, B-C, B-I
production_tax = np.array([4., 6.])
final_tax = np.array([2., 1., 3., 2.])
value_added = Z.sum(1) + F.sum(1) - Z.sum(0) - production_tax
data = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                  np.r_[2*value_added/3, np.zeros(4)], np.r_[value_added/3, np.zeros(4)]])
calibration = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"],
                                    sector_codes=["GOOD"])

# Tariff MULTIPLIERS (1 + rate), indexed (origin node, use, destination country):
# A charges 20% on imports from B, B charges 10% on imports from A.
intermediate_tariff_multipliers = np.ones((2, 1, 2))
intermediate_tariff_multipliers[1, 0, 0] = 1.2
intermediate_tariff_multipliers[0, 0, 1] = 1.1
final_tariff_multipliers = np.ones((2, 2, 2))
final_tariff_multipliers[1, :, 0] = 1.2
final_tariff_multipliers[0, :, 1] = 1.1

result = solve_trade_equilibrium(
    calibration,
    tau=intermediate_tariff_multipliers,
    tau_fd=final_tariff_multipliers,
    accounting="consistent",
    tol=1e-9,  # absolute, in the table's value units
)
assert result.converged
print(result.metadata["account_residuals"])
```

`tol` acota residuos **absolutos** en las unidades de valor de la calibración. El
piso alcanzable es aproximadamente `1e-16` veces el mayor valor de producción:
`tol=1e-9` es adecuado para la tabla anterior (valores en decenas), mientras que
el fixture OECD congelado de 3 regiones x 3 sectores en millones de USD
(producción de hasta `6.4e7`) se estanca cerca de `5e-8`; use allí de `1e-5` a
`1e-7` y reescale la tolerancia al cambiar la unidad monetaria. La contabilidad
consistente requiere una calibración con coeficientes de uso final y
participaciones de gasto no negativos: la tabla heredada incluida de 77 países x
11 sectores (`load_icio_data(source="legacy")`) se rechaza con un `ValueError` (tres celdas de
inversión negativas y una participación de inversión negativa); use la ingesta
nativa con `package_mrio_to_calibration_result`, el fixture OECD congelado o su
propia tabla equilibrada.

`accounting="legacy"` sigue siendo el valor por defecto por compatibilidad.
Reproduce las convenciones históricas de MATLAB, incluidas sus decisiones de
precedencia de precios y de valoración de la demanda final.
`tariff_revenue_mode="schedule"` por sí solo **no** selecciona la nueva
contabilidad. El modo consistente selecciona el calendario arancelario completo,
costes de factores homogéneos y el cierre explícito descrito más abajo. Anula
los interruptores heredados de precedencia y de tasas nacionales, y registra las
convenciones efectivas en los metadatos. Los envoltorios históricos de juegos
arancelarios y de política conservan sus convenciones salvo que pasen
explícitamente la opción consistente; este cambio no certifica sus medidas de
bienestar.

## Precios, cantidades e ingresos fiscales

Sea i el país/sector de origen, j el nodo productor de destino, k una categoría
de uso final y c un país de destino. Los precios son relativos a los precios de
productor de referencia, que valen uno. En este modelo no hay márgenes iceberg
ni de distribución; requerirían ecuaciones separadas de recursos y de valoración.

| Concepto | Definición |
|---|---|
| Precio de productor/en frontera | p_i |
| Precio de comprador intermedio | p_i τ^Z_ij |
| Coste del compuesto final antes del impuesto local | Q_kc = Σ_i a^F_ikc p_i τ^F_ikc |
| Precio de comprador del compuesto final | P_kc = Q_kc / (1 − δ_kc) |
| Valor bilateral a precios de productor | p_i × cantidad entregada |
| Arancel a la importación | (τ − 1) × p_i × cantidad entregada |

Los aranceles usan el precio del bien exportado tanto para las entregas
intermedias **como** para las finales. Los paga una vez el comprador y se
devuelven una vez al país importador. El impuesto final local se aplica al gasto
con aranceles incluidos. Las entradas domésticas del calendario de aranceles
deben valer uno. Se admiten importaciones subvencionadas mediante multiplicadores
positivos menores que uno.

Para un gasto de uso final E_kc, las entregas son

```
C_kc = E_kc / P_kc
F_ikc = a^F_ikc C_kc
local final tax = δ_kc E_kc
```

Así, valor a precios de productor + aranceles + impuesto final local es igual al
gasto de comprador en cada categoría/país. La calibración de δ es el impuesto
final de la fuente dividido por las compras finales efectivas de la fuente
**incluido ese impuesto**. El ahorro financiero se excluye de ese denominador.
El `tax_fd` heredado almacenado tiene otro denominador; el modo consistente
recupera la participación apropiada de la tabla de calibración (o la
reconstruye a partir de los coeficientes presupuestarios de referencia cuando la
tabla no está disponible).

La producción combina requerimientos de insumos calibrados (aprovisionamiento
Leontief o CES) con valor añadido Cobb–Douglas. El coste unitario del valor
añadido es

```
v_j = (r_c / α_j)^α_j (w_c / (1 − α_j))^(1 − α_j) / β_j
(1 − t_j) p_j = v_j + intermediate purchaser cost per unit
production-tax receipt = t_j p_j Y_j
L_j = (1 − α_j) v_j Y_j / w_c
K_j = α_j v_j Y_j / r_c
```

Estas ecuaciones son homogéneas en los precios nominales. Una carga calibrada
como participación del **ingreso** por ventas debe recaudar t_j p_j Y_j, no
t_j Y_j. La fila TLS bruta original de la OCDE agrega impuestos y subvenciones a
los productos; modelar su importe en la columna de producción como esta carga
sobre el ingreso por ventas es un supuesto explícito de calibración. La fuente
no identifica un calendario bilateral completo de impuestos a los productos o de
IVA. Este modo ofrece un modelo reducido coherente, no la afirmación de que esas
instituciones fiscales se hayan identificado de forma independiente.

Estas distinciones siguen los principios estándar de contabilidad de precios y
fiscal:
[valoración a precios de comprador frente a precios básicos en las cuentas nacionales de la ONU](https://digitallibrary.un.org/record/525790/files/National_accounts_introduction.pdf)
e [ingresos por producción/aranceles y un numerario en el ejemplo CGE estándar de GAMS](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html).
Esas fuentes motivan las convenciones; la referencia cuantitativa aquí es la
derivación separada de dos países descrita más abajo, no una replicación de ese
modelo de GAMS.

## Presupuestos, ahorro externo y numerario

Todos los impuestos domésticos y aranceles entran en los ingresos públicos G_c.
En el cierre de suma fija soportado, la transferencia resuelta T_c es igual a
G_c. La renta de los hogares es I_c = w_c L_c + r_c K_c + T_c. Las
participaciones de gasto calibradas θ la reparten, deduciendo el ahorro externo
B_c de la inversión:

```
E_kc = θ_kc I_c − 1[k = investment] B_c
Σ_k E_kc + B_c = I_c
B_c = baseline foreign saving, fixed in numeraire units
Σ_c B_c = 0
```

La inversión es la categoría de índice 1, o de índice 0 en un modelo con una
sola categoría. La ingesta nativa de la OCDE asigna primero los seis usos finales
a C/I/Cx. El gasto final debe ser no negativo en una solución aceptada. El modo
consistente rechaza cualquier celda de uso final negativa. En la tabla nativa de
la OCDE de 2019, I = GFCF + INVNT tiene 90 celdas negativas aunque el total de I
de cada país es positivo; el recuento por categoría se registra en
`calib.metadata["negative_final_demand_cells"]`. Una tabla así necesita antes
agregación o un modelo explícito de existencias. Los cargadores de FIGARO,
EXIOBASE, WIOD y Eora condensan sus usos finales en C (consumo final), I
(formación bruta de capital fijo) y Cx (variación de existencias y objetos
valiosos, y las exportaciones de EXIOBASE); a diferencia del mapeo de la OCDE,
su Cx no son compras de los residentes en el exterior. El mapeo está en
`metadata["final_use_mapping"]`.

Fijar el ahorro externo es una decisión sustantiva de cierre. Permitir que todos
los saldos externos sean endógenos mientras se imponen los presupuestos
nacionales deja sin especificar decisiones independientes de transferencias
entre países. El modo coherente sustituye esa redundancia por los saldos externos
de referencia. El precio de productor del primer país/sector se fija en uno. Su
ecuación redundante de vaciado de bienes se omite del sistema numérico cuadrado
y después **se comprueba de forma independiente antes de que pueda declararse la
convergencia**. Las exportaciones menos importaciones realizadas también deben
recuperar B_c en cada país, incluido el último.

Como B_c se fija por defecto en unidades del numerario
(`foreign_saving_units="numeraire"`), el equilibrio no es homogéneo de grado cero
en precios nominales cuando el ahorro externo es distinto de cero: introducir la
misma tabla con otro país en primer lugar cambia la asignación real y la VE/VC
hicksiana (en la tabla de referencia de bienestar de dos países, la VE del país A
es -27.55% del consumo de referencia con A primero y -45.99% con B primero). Pase
`foreign_saving_units="world_income"` para fijar el ahorro de referencia de cada
país como proporción del ingreso factorial mundial,
`B_c = (B_c^0 / Σ_d (L_d + K_d)) Σ_d (w_d L_d + r_d K_d)`; el sistema es entonces
homogéneo de grado cero y los resultados no dependen del orden de los países (el
mismo ejemplo da -36.95% con cualquier orden). El valor por defecto no cambia.

El modo heredado (`accounting="legacy"`, la réplica de MATLAB) mantiene endógenos
todos los saldos externos y no elimina esa redundancia. Para cada país, la suma de
sus residuos de mercado de bienes ponderados por precios, sus residuos de
beneficio nulo ponderados por producción, sus residuos de mercado de factores
ponderados por precios de factores y su ecuación fiscal, menos su ecuación de
saldo externo, es idénticamente igual a cuatro cuñas contables del código MATLAB:
la precedencia `w/(1-α)^(1-α)`, el impuesto a la producción cobrado sobre la
cantidad y no sobre el valor, los aranceles no acreditados y el comercio de
demanda final valorado al precio compuesto del importador. El nivel de precios y
el saldo externo de cada país quedan por tanto determinados solo por la respuesta
de esas cuñas, y el jacobiano equilibrado tiene una dirección débil por país.
Donde las respuestas casi se cancelan, choques pequeños no tienen equilibrio
cercano. En la tabla limpia OCDE 2020 (`load_icio_data(source="oecd2020")`) la
dirección de Costa Rica pierde su anclaje con un arancel uniforme de EE. UU. de
alrededor del 0.89%, y la senda del arancel tiene ahí un pliegue, de modo que los
escenarios arancelarios del modelo heredado no tienen solución en esa tabla. Con
`replicate_matlab_precedence=False` sí se resuelven. El análisis completo está en
`reviews/2026-10-04-clean-table-tariff-scenarios/REPORT.md`. Para tablas de
tamaño completo use `solve_trade_equilibrium(method="equilibrated_newton")`; si
falla, `metadata["near_singular_country"]` indica el país cuya dirección se ha
vuelto singular.

El PIB a coste de factores es igual a la renta de los factores. El PIB a precios
de mercado añade impuestos domésticos y aranceles; debe ser igual al gasto final
de comprador más las exportaciones netas valoradas a precios de productor/en
frontera. Los presupuestos de hogares y gobierno y todas las ecuaciones físicas
de equilibrio quedan expuestos en `metadata["account_residuals"]`. El éxito del
solver por sí solo no puede certificar la convergencia cuando esas ecuaciones
físicas fallan o la demanda es inviable.

## Reporte y alcance del solver

`p_sol` es el precio de productor efectivamente resuelto. `p_fd` es Q;
`Pfd_final` es P. `c_fd` es la cantidad compuesta efectiva tras asignar el
ahorro externo. El comercio bilateral, las importaciones y las exportaciones
usan valores a precios de productor. Los metadatos exponen matrices separadas de
valor a precios de productor y de comprador, ingresos por impuestos domésticos y
aranceles.

`data_model_vf` contiene valores de transacción a precios de productor, ingresos
por impuestos locales y pagos nominales a los factores. `data_tariff_vf` incluye
además los aranceles: las columnas de producción suman el ingreso por ventas a
precios de productor y las columnas de uso final suman el gasto de comprador.
Estas tablas contienen valores nominales en todos sus elementos.

El `cpi` reportado es un índice de Laspeyres de precios de comprador con cesta
fija dividido por el salario relativo a la referencia. `metadata["nominal_cpi"]`
conserva el índice sin deflactar. Una referencia suministrada también debe usar
contabilidad consistente. Es un índice, no bienestar hicksiano ni una derivación
de la función de gasto.

El modo soporta NumPy, aprovisionamiento intermedio Leontief/CES y devoluciones
de suma fija. Newton, LU disperso, Krylov, Broyden, híbrido/LM de SciPy y la
continuación completa de Keller resuelven las mismas ecuaciones. Los solvers
Newton, LU disperso, Krylov, Broyden y Keller se detienen solo cuando los residuos
resueltos y las ecuaciones físicas comprobadas de forma independiente (la ecuación
de bienes omitida y los saldos externos realizados) cumplen `tol`, de modo que un
estado aceptado por el solver no se rechaza después con la misma tolerancia; sus
jacobianos por diferencias finitas escalan el paso de las incógnitas de
transferencias y saldo externo, expresadas en unidades de valor, con su magnitud,
por lo que la resolución no depende de la unidad monetaria. Híbrido/LM de SciPy
usan su propio criterio de parada y se auditan después. Una petición de
`method="condensed"` usa Newton completo y registra `effective_method="newton"`;
la reducción de Schur heredada encarna ecuaciones distintas. El modo
cuasi-condensado, otros esquemas de reciclaje fiscal, la ejecución en GPU y las
penalizaciones por capacidad lanzan explícitamente `NotImplementedError` en este
modo a la espera de sus propias derivaciones económicas. El modo heredado
conserva esas interfaces existentes sin adquirir una nueva afirmación de
validación.

## Comprobación independiente con dos países

[El script de referencia](https://github.com/jalonso1979/puremacro/blob/v4.3.0/tools/reference_validation/validate_trade_accounting.py)
usa una tabla de transacciones equilibrada a mano con dos países, un sector
productor, dos usos finales, ahorro externo distinto de cero, impuestos locales y
aranceles heterogéneos. Deriva la referencia directamente de la tabla, sin llamar
dentro del oráculo a las rutinas de calibración, residuos o equilibrio general de
puremacro.

Con un sector por país, el vaciado de factores fija las producciones. Para cada
precio de productor relativo candidato, las cantidades intermedias quedan fijadas
por los insumos Leontief. Sea h_kc la participación del gasto final de comprador
que retorna como impuesto local más arancel. El oráculo resuelve la renta
algebraicamente:

```
I_c = [p_c Y_c − Σ_i p_i Z_ic − h_investment,c B_c]
      / [1 − Σ_k h_kc θ_kc]
```

Después vacía un mercado de bienes con una raíz escalar de Brent. El otro mercado
de bienes se comprueba por separado. Esto proporciona precios, pagos a factores,
entregas e ingresos fiscales independientes en la referencia y bajo un choque
arancelario no uniforme. Las pruebas también comprueban la homogeneidad nominal,
el rango completo del jacobiano tras el cierre, todos los libros presupuestarios,
el escalado de moneda y la agregación conservativa de la OCDE.

[Los resultados y los comandos de reproducción](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-trade-accounting/REPORT.md)
registran los errores alcanzados y las limitaciones pendientes. La posterior
[interfaz de bienestar hicksiano de consumo](trade_welfare.md) proporciona VE/VC
derivadas y una atribución por extremos. La descomposición histórica de términos
de intercambio/eficiencia y la certificación de teoremas siguen no disponibles.
