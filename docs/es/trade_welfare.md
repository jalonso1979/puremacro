> 🇬🇧 [English](../trade_welfare.md) · 🇪🇸 Español

# Bienestar hicksiano de consumo

Use equilibrios explícitos de referencia y contrafactual obtenidos con
contabilidad consistente. Este ejemplo se ejecuta tal cual sobre una tabla de dos
países equilibrada a mano:

```python
import numpy as np
from puremacro.trade import (calibrate_trade_model, compute_hicksian_welfare,
                             solve_trade_equilibrium)

# Two countries (A, B), one good, final uses C and I, values in one currency unit.
Z = np.array([[10., 12.], [8., 14.]])
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])  # to A-C, A-I, B-C, B-I
production_tax = np.array([4., 6.])
final_tax = np.array([2., 1., 3., 2.])
value_added = Z.sum(1) + F.sum(1) - Z.sum(0) - production_tax
data = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                  np.r_[2*value_added/3, np.zeros(4)], np.r_[value_added/3, np.zeros(4)]])
calibration = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"],
                                    sector_codes=["GOOD"])

import_rates = np.array([0.20, 0.10])  # A and B tariff RATES on all imports
tol = 1e-10  # absolute, in the table's value units; see "Tolerances" below
base = solve_trade_equilibrium(calibration, accounting="consistent", tol=tol)
counter = solve_trade_equilibrium(
    calibration, tau=import_rates, tau_fd=import_rates,
    accounting="consistent", tol=tol,
)
welfare = compute_hicksian_welfare(
    calibration, counter, base_result=base, target_country="A",
    consumption_categories=(0,),
)
print(welfare.summary())
print(welfare.to_dataframe())
```

Ambos estados deben converger y usar la misma calibración, y deben resolverse
con el mismo `sigma`, el mismo cierre fiscal y el mismo `foreign_saving_units`
(una discrepancia genera `ValueError`).

**Tolerancias.** Pase explícitamente el `tol` del solver. Acota residuos
absolutos en las unidades de valor de la calibración, y las identidades de
bienestar heredan ese residuo: con el valor por defecto del solver (`2.5e-3`),
una tabla medida en unidades monetarias simples deja errores de cuentas fiscales
cercanos a `1e-4`, que no superan la identidad fiscal con el `tol=1e-8` por
defecto del bienestar. Use `1e-10` para tablas de escala unitaria como la
anterior y aproximadamente de `1e-5` a `1e-7` para el fixture OECD congelado en
millones de USD (véase [contabilidad consistente](trade_accounting.md)). La
función de bienestar reevalúa las ecuaciones de equilibrio, los precios, las cantidades y
las cuentas fiscales usando los calendarios arancelarios registrados. Los estados
heredados, los calendarios ausentes, las soluciones fallidas, los campos
obsoletos y los cierres fiscales no soportados lanzan errores. Los resultados
creados antes de que se registraran los calendarios arancelarios deben resolverse
de nuevo.

## Qué entra en la utilidad

El valor por defecto es la categoría de uso final 0, convencionalmente C. Dentro
de cada categoría de uso final el modelo tiene coeficientes fijos de
aprovisionamiento por origen: una cesta Leontief. Para varias categorías
seleccionadas, las preferencias son Cobb–Douglas con pesos
`omega_k = theta_k / sum_selected(theta)`. Estos pesos y coeficientes de
aprovisionamiento permanecen fijos a lo largo de la comparación. Las categorías
con peso cero no contribuyen.

Es bienestar de **consumo** condicional. La inversión (índice 1 en un modelo con
varios usos finales) queda excluida. Su gasto resta el ahorro externo y no
obedece las participaciones de gasto de consumo. Un modelo con una sola categoría
se soporta cuando el ahorro externo es cero. Un modelo de utilidad a lo largo de
la vida, de ahorro o de bienes públicos necesitaría sus propias preferencias y su
propio cierre intertemporal.

La categoría C nativa de la OCDE es HFCE + NPISH + GGFC. Así, su resultado valora
la cesta de consumo agregada del modelo, incluido el consumo final del gobierno;
no es una estimación de bienestar solo de los hogares. Ese supuesto de agregación
es explícito. El usuario puede seleccionar categorías adicionales distintas de la
inversión con `consumption_categories=(0, 2)`, declarando que también pertenecen
a la utilidad.

## Derivación y signo

Sea c_k la cantidad compuesta y P_k su precio de **comprador**, incluidos
aranceles e impuestos locales sobre el uso final. Para pesos positivos
seleccionados omega que suman uno, las funciones de utilidad y de gasto son

```
U(c) = product(c_k ** omega_k)
e(P, U) = U * product((P_k / omega_k) ** omega_k)
h_k(P, U) = omega_k * e(P, U) / P_k
```

Minimizar el gasto de comprador sujeto a U(c) >= U da la h_k anterior. Como el
presupuesto de consumo seleccionado es m = kappa * (renta de factores +
transferencias), con kappa la suma de sus participaciones theta, las demandas
observadas satisfacen `P_k c_k = omega_k m`. La implementación comprueba esta
dualidad en ambos estados.

Con 0 denotando la referencia y 1 el contrafactual, la variación equivalente
(VE, campo `ev`) y la variación compensatoria (VC, campo `cv`) son

```
EV = e(P0, U1) - e(P0, U0)
CV = e(P1, U1) - e(P1, U0)
```

Ambas son **positivas para las ganancias de bienestar**. La VE es la
transferencia equivalente al **presupuesto de consumo** seleccionado a precios de
referencia; una VC positiva es el importe que puede retirarse de ese presupuesto
a precios contrafactuales conservando la utilidad de referencia. Si una
transferencia de renta debe seguir la participación fija de consumo kappa del
modelo, la transferencia de renta total requerida es EV/kappa (o CV/kappa). Los
campos reportados usan unidades del presupuesto de consumo. Esta es la
convención de la función de gasto de las
[notas de teoría del consumidor del MIT](https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/1b0f90b4fdada65148edd1e4b16aab18_MIT14_121F15_3S.pdf).

El tiempo de ejecución evalúa la utilidad a partir de cantidades y el gasto a
partir de precios de comprador. Usa cocientes logarítmicos y `expm1` para
cambios pequeños. El `cpi` de Laspeyres deflactado por el salario, el PIB nominal
y los ingresos arancelarios por sí solos no determinan la VE. No se usa ningún
componente residual para construir el total de bienestar.

`ev` y `cv` usan las unidades de valor de la calibración. `ev_pct_consumption`
divide la VE por el gasto de consumo seleccionado de referencia; `ev_pct_gdp`
usa el PIB de referencia. Ningún porcentaje se sustituye silenciosamente por la
atribución en niveles. El resultado no supone que toda calibración esté en
millones de USD.

## Atribución exacta por extremos

Defina A = product((P0_k/P1_k) ** omega_k), la renta de factores F y la
transferencia total de suma fija resuelta T. La VE de consumo también es igual a
`kappa * [A*(F1+T1) - (F0+T0)]` cuando se cumplen las identidades de demanda
comprobadas. Asignar simétricamente cada interacción precio/renta da

```
price_effect           = kappa * (A-1) * (F0+F1+T0+T1) / 2
factor_income_effect   = kappa * (1+A) * (F1-F0) / 2
fiscal_transfer_effect = kappa * (1+A) * (T1-T0) / 2
```

Estas expresiones son la asignación de Shapley entre los tres bloques de
extremos: se promedia la contribución marginal de cada bloque sobre los seis
órdenes posibles. Su suma se contrasta con la **VE de la función de gasto
calculada por separado**. El término fiscal se divide en efectos de devolución
de aranceles y de impuestos domésticos con el mismo peso. Son subcomponentes, no
ganancias adicionales. Los aranceles se incluyen una vez en los precios de
comprador y una vez en la cuenta de transferencias, tal como especifica el
modelo de equilibrio. Un ingreso fiscal no es en sí mismo una ganancia de
bienestar social.

Esta atribución está condicionada al numerario fijo del modelo. Con ahorro
externo distinto de cero, el cierre por defecto (`foreign_saving_units="numeraire"`)
hace que la propia VE/VC dependa de qué país/sector aparece primero; resuelva
ambos estados con `foreign_saving_units="world_income"` para obtener resultados
que no dependan del orden de los países (véase
[contabilidad consistente](trade_accounting.md)). Los bloques de extremos mixtos no tienen por qué ser equilibrios, así que es una atribución
contable, no un experimento causal de equilibrio general ni un teorema de
términos de intercambio/eficiencia asignativa. Cambiar las unidades nominales de
forma consistente escala todos los efectos monetarios; la VE porcentual no
cambia.

## Evidencia y límites

La referencia independiente resuelve una economía de dos países con un sector
productor, dos cestas de consumo distintas y una categoría separada de
inversión. Después minimiza numéricamente las compras **a nivel de origen**
sujetas a restricciones Leontief y de utilidad en cada par precio/utilidad. Esta
optimización primal no llama a la fórmula de gasto del tiempo de ejecución. Los
seis órdenes de Shapley se evalúan por separado. Véanse
[el código de referencia](https://github.com/jalonso1979/puremacro/blob/v4.3.0/tools/reference_validation/validate_trade_welfare.py)
y [la validación registrada](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-hicksian-welfare/REPORT.md).

Comprobaciones adicionales cubren la identidad, comparaciones invertidas,
escalado de moneda, subvenciones, choques pequeños, estados obsoletos/fallidos y
la agregación congelada de la OCDE. La nueva interfaz soporta el
aprovisionamiento Leontief/CES del [modelo consistente](trade_accounting.md) con
devoluciones de suma fija. Los optimizadores arancelarios, las búsquedas de Nash
y las matrices de pagos seleccionan ahora este cálculo con
`metric="hicksian_ev"`; véase
[integración con la política y recuperación del solver](trade_policy.md). Los
alias históricos de objetivos y los proxies de bienestar del modelo flexible no
cambian.

`decompose_hicksian_ev_3way` sigue no disponible: sus campos `TOT`, `Alloc` y
`TariffRec` describen una descomposición distinta y no soportada. Use
`compute_hicksian_welfare` para la VE/VC derivada y la atribución anterior. La
certificación de teoremas también sigue no disponible. Los contenedores
heredados existentes y las pruebas históricas en cuarentena se conservan por
compatibilidad, no se cuentan como validación.
