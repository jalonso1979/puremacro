> 🇬🇧 [English](../trade_condensed.md) · 🇪🇸 Español

# Modelo arancelario Leontief condensado de un factor

`puremacro.trade.condensed` resuelve un modelo arancelario estático de varios
países y sectores con coeficientes fijos, un factor compuesto por país y
aranceles a la importación devueltos como transferencia de suma fija; elimina
precios y producciones de forma exacta, certifica cada equilibrio reportado a
partir de los flujos brutos de la tabla y reporta únicamente medidas libres de
numerario. Es una adaptación del motor de investigación usado en la
reconstrucción IO de 2026
(`headlinePaper/rebuild/vendor/puremacro/trade/corrected`) a las convenciones
de puremacro. Es un modelo distinto de `solve_trade_equilibrium` y de su modo
`accounting="consistent"`; la tabla comparativa de abajo indica en qué
difieren.

## Ejemplo ejecutable con la tabla 77x11 incluida

```python
from puremacro.trade import calibrate_trade_model, load_icio_data
from puremacro.trade.condensed import (
    BalancedIOTable, build_tariff_wedges, calibrate_condensed, compute_measures, solve_condensed,
)

legacy = calibrate_trade_model(load_icio_data(source="legacy"))
table = BalancedIOTable.from_trade_calibration(legacy, negative_investment="to_inventory")
print(table.report.negative_investment_cells)   # ('LTU_MINQ->LTU', 'UKR_MINQ->UKR', 'VNM_MANU->VNM')
calib = calibrate_condensed(table, allow_empty_purchases_abroad=True)
wedges = build_tariff_wedges(calib, 0.10, importer="USA")   # 10% on all merchandise imports
result = solve_condensed(calib, wedges)                      # gate, Newton, certificate
print(result.summary())
measures = compute_measures(calib, wedges, result.state, country="USA", partners=("CHN", "MEX", "CAN"))
print(measures.to_markdown())
```

Con la tabla incluida esto imprime un residuo escalado máximo de `2.7e-13`, un
residuo de Walras de `7.9e-14`, un bloque máximo del certificado de `2.7e-13`,
cuatro iteraciones de Newton en alrededor de un segundo y, para Estados Unidos,
precios al consumo relativos a los salarios de `+0.7234%`, una variación
equivalente de `-0.0291%` del PIB base, términos de intercambio Fisher de
`-0.1055%`, ingresos arancelarios del `0.800%` del PIB y una tasa efectiva
sobre las importaciones gravables de exactamente `10%`. La tabla incluida funde
las existencias con la inversión y tiene tres celdas negativas, que el puente
traslada a la categoría exógena de existencias y registra. Hay ejemplos con
tres regiones y sintéticos en `tests/test_trade_condensed.py`.

La tabla incluida procede de una exportación dañada de la OCDE (véase el
[aviso de procedencia](ADVISORY.md)), así que las cifras anteriores son valores
de regresión del software, no estimaciones de la OCDE. Con la edición limpia
OCDE 2019 el mismo experimento reproduce las cifras publicadas con la
reconstrucción IO:

```python
# requires: the OECD ICIO 2023-edition 2019_SML.csv (not bundled)
from puremacro.trade import (
    BalancedIOTable, aggregate_mrio, build_tariff_wedges, calibrate_condensed,
    compute_measures, read_oecd_native, regularize_table, solve_condensed,
)

table, report = regularize_table(read_oecd_native("2019_SML.csv", 2019))
for layout in (table, aggregate_mrio(table, "agregar")):
    calib = calibrate_condensed(BalancedIOTable.from_io_table(layout))
    wedges = build_tariff_wedges(calib, 0.10, importer="USA")
    result = solve_condensed(calib, wedges)
    print(compute_measures(calib, wedges, result.state, country="USA").to_markdown())
```

Con 77 x 45 esto da precios al consumo relativos a los salarios de
`+0.722154%` y una variación equivalente de `-0.005924%` del PIB base (ocho
iteraciones de Newton, certificado `6.9e-13`, unos cinco segundos). La
agregación Agregar de 11 sectores da `+0.705964%` y `-0.009092%`: la tabla
gruesa exagera la pérdida de bienestar en torno a la mitad y la de términos de
intercambio en torno al triple (`-0.0393%` frente a `-0.0128%`).

## Qué calcula

Índices: países `k`, celdas `i = country * S + sector` (M = N S), categorías
finales `f` en {C, G, X} con participaciones Cobb-Douglas y existencias
exógenas V. En el punto de referencia todos los precios, salarios y
multiplicadores valen uno.

Calibración a partir de una tabla balanceada (`table_from_arrays`, luego
`calibrate_condensed`):

```
a_ij = Z_ij / y0_j     b_j = VA_j / y0_j     t_j = TLS_j / y0_j
omega^f_ik = F^f_ik / B^f_k        A0^f_k = B^f_k + TFD^f_k       t^f_k = TFD^f_k / A0^f_k
theta^f_k = A0^f_k / sum_f A0^f_k  qV_ik = F^V_ik                 TV_k = TFD^V_k
Fbar_k = sum_{j in k} VA_j         Y0_k = Fbar_k + sum_{j in k} TLS_j + sum_f TFD^f_k + TV_k
XN0_k = fob exports - imports      s_k = XN0_k / sum_l Fbar_l
```

Equilibrio en niveles (2M + 3N incógnitas, una ecuación redundante por la ley
de Walras):

```
(E1) (1 - t_j) p_j = b_j w_c(j) + sum_i a_ij tau_ij p_i                       zero profit
(E2) y_i = sum_j a_ij y_j + sum_k (sum_f omega^f_ik q^f_k + qV_ik)             goods clearing
(E3) sum_{j in k} b_j y_j = Fbar_k                                              factor market
(E4) Y_k = w_k Fbar_k + PT_k + TR_k                                             income
(E5) XN_k = s_k sum_l w_l Fbar_l                                                closure
(E6) sum_l w_l Fbar_l = sum_l Fbar_l                                            numeraire
P^f_k = sum_i omega^f_ik tau^f_ik p_i     E^f_k = theta^f_k (Y_k - XN_k - E^V_k)
q^f_k = (1 - t^f_k) E^f_k / P^f_k         E^V_k = sum_i tau^V_ik p_i qV_ik + TV_k P^G_k
```

`PT_k` recoge los impuestos sobre los productos en la producción y en las
compras finales; `TR_k` recoge todos los aranceles (`(tau - 1) x valor a
precios de productor`) sobre las compras intermedias, finales y de existencias,
y se devuelve mediante (E4).

Condensación: dado `z = (log w, Y / Y0)`, (E1) se resuelve de forma exacta con
una LU de `I - B_tau^T`, con `B_tau,ij = a_ij tau_ij / (1 - t_j)`, y (E2) con
una LU de `I - a`. Los residuos son (E3) y (E4) escalados por cantidades base,
con la ecuación de ingreso del país de referencia sustituida por el numerario;
esa ecuación omitida es la comprobación de Walras que acompaña a cada estado.
El jacobiano usa diferencias hacia adelante (`h = 1e-7`) y reutiliza los
precios en las columnas de ingreso.

Solucionadores (`solve_condensed`): Newton amortiguado con retroceso de Armijo,
un tope de `0.3` en los desplazamientos de los log-salarios y la regla de
parada `max |r| <= 1e-12`, `|Walras| <= 1e-10`; continuación de parámetro
natural en la escala arancelaria (el paso se divide por dos al fallar, hasta
`1/256`, con predictor secante); un respaldo anidado que resuelve el sistema
afín de ingresos dados los salarios; continuación opcional por pseudolongitud
de arco con monitores de pliegue (`dlambda/ds`, `sign det J`, `sigma_min /
sigma_max`) y una comprobación de aterrizaje a `1e-10`; multiarranque cercano
opcional que lanza `MultipleEquilibria` cuando un arranque convergido aterriza
a más de `1e-9`. Una puerta de existencia de Collatz-Wielandt `rho(B_tau) < 1 -
1e-3` (`puremacro.trade.regularize.compute_spectral_radius`) se ejecuta antes
de cada resolución y falla de forma cerrada: unas cotas que certifican `rho <
1` pero no el margen lanzan `ProductivityUncertified` ("margin not met", con
las cotas adjuntas), unas cotas que no certifican nada la lanzan como
"inconclusive", y ninguna de las dos afirma la inexistencia; las cuñas
construidas para otra tabla se rechazan con `CalibrationError`. Cuando la
continuación se estanca, la etiqueta de diagnóstico del respaldo anidado y su
registro de etapa (`continuation_lam_reached`) nombran el estancamiento.

Certificado (`certify_raw_flows`): diez bloques recalculados a partir de `Z, F,
VA, TLS, TFD` y las cuñas, nunca a partir de `a, b, omega, theta` ni de las
factorizaciones: beneficio nulo en las celdas activas, equilibrio de bienes,
mercados de factores, presupuesto de los hogares, ingreso nacional, identidad
de ingresos arancelarios, cierre, identidad de cuenta corriente (`NX` a partir
de valores bilaterales a precios de productor), suma mundial y numerario. Cada
bloque debe ser como máximo `1e-10`; en caso contrario `solve_condensed` lanza
`CertificationFailure` y nunca devuelve `converged=True` tras un fallo. El
certificado usa los precios y cantidades de las cestas del estado, así que es
independiente de los coeficientes calibrados pero no del sistema de demanda que
los produjo. Los dos bloques por celda (beneficio nulo, equilibrio de bienes)
se escalan por `max(p_j y_j, cell_floor)` y `max(y_i, cell_floor)` con
`cell_floor = 1` en las unidades de la tabla (la convención de millones de USD
del motor IO), así que las celdas por debajo de una unidad se juzgan a `1e-10`
absoluto; pase un `cell_floor` menor a `certify_raw_flows` para tablas en
unidades grandes. Las celdas fantasma (celdas vacías que la construcción
rellena con flujos de `1e-6`) quedan fuera solo del bloque de beneficio nulo;
todos los demás bloques usan su cociente de producción verdadero `y_j / y0_j`,
porque el modelo deja que la producción fantasma se mueva y el valor añadido
fantasma forma parte de `Fbar`. `solve_condensed(..., certify=False)` omite el
certificado solo con fines de diagnóstico: ese resultado es `converged` pero ni
`passed` ni `certified`.

Medidas (`compute_measures`, todas libres de numerario, `P^U_k = prod_f (P^f_k
/ P0^f_k)^theta^f_k`):

| Medida | Definición |
|---|---|
| precios al consumo relativos a los salarios (M1) | `P^C_k / (P0^C_k w_k) - 1`, el inverso del salario real de consumo (no es un IPC) |
| variación equivalente (M2) | `Atil^1_k / P^U_k - Atil^0_k`, porcentaje del PIB base, con la descomposición exacta en ingreso real de los factores, ingresos netos reales (aranceles y otros impuestos) y la transferencia por déficit y existencias |
| ingreso nacional real (M3) | `(Y_k / P^U_k) / Y0_k - 1` |
| términos de intercambio Fisher (M4) | fob, sin aranceles, `sqrt(L^X P^X / (L^M P^M)) - 1` a cantidades base |
| ingresos arancelarios (M5) | `TR_k / Y_k` y la tasa efectiva sobre las importaciones gravables (las existencias solo cuentan como gravables cuando el estado se resolvió con `inventories="tariffed"`) |
| PIB real a precios constantes (M6) | `sum (VA_j + TLS_j) yhat_j + sum_f TFD^f qhat^f + TFD^V`, solo un efecto composición |
| cota PE de coste unitario (M8) | `p^PE = (I - B_tau^T)^{-1} b / (1 - t)`, `ln(1 + Pi^C) = ln(1 + Pi^{C,PE}) + Delta^rel`, y la descomposición de primer orden en efecto directo y cascada |
| brechas de agregación (M9), incidencia sectorial (M10) | `aggregation_gaps(fine, coarse)` devuelve `fine`, `coarse`, `absolute_gap`, `relative_gap` para cada clave común; `sector_incidence(..., groups=...)` con dispersión intragrupo ponderada por la producción |

`separate_baseline_tariffs(calib, wedges)` reexpresa los aranceles observados
en el punto de referencia como cuñas brutas explícitas con impuestos residuales
sobre los productos, de modo que los incrementos contrafactuales se apilan
sobre aranceles iniciales no nulos sin contar dos veces la recaudación aduanera
de referencia (puertas a `1e-11`). Es una separación contable, no evidencia de
que un arancel no observado sea cero.

## Contrato de datos

`table_from_arrays` (o `BalancedIOTable.from_arrays`) acepta la disposición
compartida de la ola 1: `Z (M, M)`, `F (M, N, K)`, `VA (M,)`, `TLS (M,)`, `TFD
(N, K)`, `country_codes`, `sector_codes` y `fd_codes` con los seis códigos de
la OCDE, `("C", "G", "X", "V")`, `("C", "G", "X", "V", "VAL")` (VAL se funde en
V y se registra), `("C", "G", "X")`, el heredado `("C", "I", "Cx")`, o los
vocabularios de demanda final de los cargadores de `puremacro.trade.data`:
FIGARO (`P3_S14`, `P3_S15`, `P3_S13` sumados en C, `P51G` en G, `P5M` en V),
EXIOBASE (`HFCE`, `NPISH`, `GGFC`, `GFCF`, `INVNT`, `VALUABLES` fundido en V, y
una columna `EXPORT` que debe ser idénticamente cero en una tabla cerrada),
WIOD (`CONS_h`, `CONS_np`, `CONS_g`, `GFCF`, `INVT`) y Eora (`HFCE`, `NPISH`,
`GGFC`, `GFCF`, `INVNT`, `ACQ_VAL` fundido en V). Los códigos que caen en una
misma categoría se suman; C y G son obligatorios, X y V valen cero cuando
faltan. Las disposiciones sin columna de compras en el extranjero (todas menos
la de la OCDE) dan una cesta X vacía, así que `calibrate_condensed` necesita
`allow_empty_purchases_abroad=True` con ellas (X copia la cesta C como precio
auxiliar; su gasto es idénticamente cero). Las disposiciones de tres categorías
y cualquier tabla con celdas G negativas necesitan
`negative_investment="raise"` (por defecto) o `"to_inventory"`.
`BalancedIOTable.from_io_table` toma cualquier objeto con esos atributos
(incluida una `Z` dispersa), `from_raw` toma un
`puremacro.trade.data.RawIOData` de cualquiera de los cinco cargadores, y
`from_trade_calibration` hace de puente desde un `TradeCalibrationResult`
(trabajo y capital se suman en el factor único; G contiene las existencias; Cx
pasa a X solo para la disposición incluida y el mapeo `DPABR` de la OCDE,
mientras que el Cx de los cargadores FIGARO, WIOD y Eora pasa a la V exógena,
de modo que esas tablas se calibran con `allow_empty_purchases_abroad=True`, y
el Cx de EXIOBASE requiere un `cx_category` explícito). La
construcción añade fantasmas de `1e-6` a las celdas vacías, eleva el valor
añadido no positivo a `1e-3` de la producción, recalcula TLS como residuo de
columna y vigila el cambio (`1e-4` del |TLS| mundial, `1e-2` de la producción
de una celda) y asigna el redondeo mundial de las balanzas comerciales al país
de referencia (el último código por defecto). La máscara de mercancías se
infiere para los registros de 45 industrias de la OCDE, el de 11 sectores
incluido, Agregar, las secciones CIIU y el fixture; otros registros (entre
ellos los códigos sectoriales de FIGARO, EXIOBASE, WIOD y Eora) pasan
`merchandise_mask`, por ejemplo desde `puremacro.trade.mrio.goods_mask`.

Los aranceles son tasas ad valorem `r >= 0`; las transacciones domésticas nunca
se gravan y las compras de residentes en el extranjero (X) están exentas.
`TariffWedges.from_arrays` rechaza los multiplicadores menores que uno
(subvenciones a la importación), lo que mantiene la matriz de costes no
decreciente en la escala arancelaria, de modo que un solo certificado de
existencia con el arancel completo cubre todas las etapas de la continuación.
`build_tariff_wedges` toma un escalar, una tabla `(N, S)` origen por sector
para un importador, un arreglo `(N, N, S)`, un mapeo `{importer: {origin | "*":
rate}}` (represalias; un origen con nombre siempre prevalece sobre `"*"`, sea
cual sea el orden de las claves) o un invocable como `TariffScenario.get_rate`.
`wedges.to_dataframe()` lista las tasas no nulas por origen, importador y
sector. La cobertura es de mercancías por defecto (`coverage="all"` incluye
servicios); `fd_tariffed` elige cuáles de C, G, V soportan el arancel. Con
`coverage="goods"`, una especificación explícita por sector (un arreglo con eje
sectorial, un arreglo `(S,)` dentro de un mapeo o un invocable) que ponga
aranceles en sectores de servicios se recorta a mercancías con un
`RuntimeWarning`, y los sectores recortados se registran en
`wedges.metadata["dropped_service_rates"]`; las tasas escalares significan
"todos los sectores cubiertos" y no avisan. Las reglas gruesas de 45 a 11
sectores y las concordancias pertenecen a `puremacro.trade.mrio` (sus tablas
agregadas entran por `from_io_table`).

## Tres modelos, no uno

| | `solve_trade_equilibrium` (heredado) | `accounting="consistent"` | `puremacro.trade.condensed` |
|---|---|---|---|
| Factores | trabajo y capital, valor añadido Cobb-Douglas | trabajo y capital, valor añadido Cobb-Douglas | un factor compuesto por país |
| Demanda final | participaciones C, I, Cx del ingreso; ahorro exterior en I | participaciones C, I, Cx del ingreso; ahorro exterior fijo deducido de I | C, G, X Cobb-Douglas sobre la absorción neta de existencias exógenas V |
| Balanza comercial | fija en unidades del numerario | ahorro exterior de referencia fijo | participaciones del ingreso mundial de los factores (también PIB mundial o PIB propio) |
| Numerario | primer precio de productor = 1 | primer precio de productor = 1 | ingreso mundial de los factores (o el salario de cualquier país) |
| Incógnitas | sistema completo o condensación de Schur heredada 4N-1 | sistema completo (Newton, LU dispersa, Krylov, Keller PAC) | condensación exacta 2N `(log w, Y / Y0)` |
| Tolerancia y aceptación | tolerancia del solucionador (`2.5e-3` por defecto en la condensada) | residuo y libro de cuentas a la tolerancia del solucionador | residuo `1e-12`, Walras `1e-10`, diez bloques de flujos brutos a `1e-10` |
| Bienestar | basado en índices | EV/CV hicksianas con atribución de Shapley entre extremos | EV con descomposición aditiva exacta, ingreso nacional real, ToT Fisher |
| Aranceles iniciales | los flujos observados son flujos a precios básicos | igual | `separate_baseline_tariffs` |

Los resultados no son intercambiables entre las tres columnas. Use el modelo
condensado cuando quiera una referencia Leontief de un factor certificada con
medidas libres de numerario; use la contabilidad consistente cuando necesite la
estructura de dos factores o las herramientas de política hicksianas.

## Qué está validado y qué no

Verificado en este repositorio (`tests/test_trade_condensed.py`): reproducción
del punto de referencia a `1e-14` para todos los cierres y numerarios;
certificados iguales o inferiores a `1e-10` en una tabla sintética 3x4, el
fixture congelado OCDE 3x3 y la tabla 77x11 incluida (observados `1.1e-13`,
`9.8e-13`, `2.7e-13`); el certificado reacciona a perturbaciones de `1e-5` en
`w, p, y, Y, XN, TR` en el bloque nombrado y a una perturbación salarial de
`1e-9`; certificados de tablas en unidades naturales con entre una y seis
celdas vacías (fantasma); aditividad de la descomposición de la EV a `1e-12`;
invariancia al numerario de dieciocho medidas y componentes de la
descomposición a `1e-9` (observado `1.8e-11`); un arancel sobre cantidades
fijas de existencias es neutral bajo el cierre de factores y no bajo el cierre
de PIB mundial; homogeneidad al escalar la tabla por 1000 (`1.4e-11`); una tasa
efectiva exacta del `10%`, también con existencias no gravadas; los jacobianos
hacia adelante y centrales coinciden a `1e-6` relativo (observado `5e-8`); la
cota PE queda por debajo del precio al consumo de equilibrio general; etapas de
continuación, detección de estancamiento, el respaldo anidado, monitores y
aterrizaje de la longitud de arco, multiarranque cercano y la recuperación
desde arranques lejanos en la tabla 77x11; una frontera de existencia que falla
de forma cerrada en la economía sintética (un arancel del 500% de EE. UU. sobre
mercancías pasa la puerta de existencia, pero el equilibrio con salario
estadounidense positivo deja de existir cerca de un arancel del 259%, `lambda*
= 0.51773` del 500%: a lo largo de la rama `lambda` crece de forma monótona
mientras el salario de EE. UU. tiende a cero; la continuación de parámetro
natural se estanca en `lambda = 0.515625`, la pseudolongitud de arco lanza
`FoldDetected` en `lambda = 0.5177`, donde el jacobiano es numéricamente
singular, `sigma_min / sigma_max = 3e-11`, y `solve_condensed` termina en
`EquilibriumNotFound` nombrando el estancamiento); las cinco disposiciones de
los cargadores de `puremacro.trade.data` en tablas sintéticas 3x3; y paridad
con el motor IO copiado en el repositorio de investigación (`vendor`),
ejecutado en un subproceso, en los tres fixtures: salarios, precios,
producciones y medidas idénticos bit a bit en la tabla sintética (también con
aranceles de referencia del 5% separados, el cierre de PIB mundial con el
salario de ROW como numerario, inversión real fija y existencias no gravadas),
y dentro de la tolerancia de prueba `1e-12` en el fixture OCDE 3x3 y el puente
77x11 incluido (observado: salarios y precios a `6.4e-14`, medidas principales
a `4.3e-13` puntos porcentuales).

No validado aquí, citando la documentación IO: "No global uniqueness, universal
fold threshold, or global Newton guarantee is asserted."; "Numerical residual
checks do not establish global equilibrium existence, uniqueness, or
optimality."; "These are floating-point verification criteria, not formal
interval root enclosures."; el diagnóstico de Gale-Nikaido es "OA evidence
only; never called a proof" y no se ha adaptado. La longitud de arco y el
multiarranque son evidencia, no pruebas. No se ha exhibido ningún punto de
retorno en un ejemplo económico: el monitor de pliegue se ejercita con un
cambio de signo sintético del determinante y con la frontera de existencia de
arriba, así que `FoldDetected` significa que la trayectoria no puede
continuarse de forma regular (un punto de retorno o una frontera de
existencia), y esa frontera no es un umbral de pliegue universal. La EV
estática incluye la inversión en el compuesto de absorción. Desviaciones
documentadas respecto al motor IO: el último predictor de longitud de arco
apunta a `lambda = 1` en vez de sobrepasarlo; `certify_raw_flows` recibe el
tratamiento de existencias; el certificado y el PIB real usan el cociente de
producción verdadero de las celdas fantasma (el motor IO lo fijaba en uno, de
modo que su bloque de mercados de factores rechazaba equilibrios válidos de
tablas en unidades pequeñas con celdas vacías); la tasa efectiva sigue el
tratamiento de existencias de la resolución y se reporta para el país
solicitado (el motor IO siempre contaba como gravables las existencias con
multiplicador arancelario y siempre reportaba Estados Unidos); se rechazan los
multiplicadores menores que uno y un origen con nombre prevalece sobre `"*"`
sea cual sea el orden de las claves (en el motor IO ganaba la última clave); el
margen de existencia vale `1e-3` por defecto, el margen de diseño de la
especificación del motor IO (no distribuida), en lugar del `1e-12` del código
IO; el corrector de longitud de arco comprueba la admisibilidad;
`multistart_near` usa ocho arranques hasta once sectores (el motor IO usaba
ocho solo con exactamente once); y `aggregation_gaps` usa nombres genéricos
(`fine`, `coarse`, `absolute_gap` en lugar de los `measures_45`, `measures_11`,
`val_45`, `val_11`, `absolute_gap_pp` del IO). No se han adaptado el puente
heredado `fd_rule="invest"`, `inactive="mask"`, el lector CSV de la OCDE y su
lista blanca md5, el registro de escenarios del artículo y sus cargadores JSON
de política, ni las reglas gruesas de aranceles de 45 a 11 sectores. Las
ejecuciones completas con 45 sectores usan factorizaciones LU densas de
matrices `3465 x 3465` (alrededor de 0.7 GB de pico en el motor IO) y no son un
objetivo para el navegador. La tabla 77x11 incluida es la descrita en la
[documentación MRIO](trade_mrio.md); sus límites de procedencia se aplican a
cualquier número calculado con ella.

Relacionado: [contabilidad comercial consistente](trade_accounting.md),
[bienestar de consumo hicksiano](trade_welfare.md), [política arancelaria
hicksiana](trade_policy.md), [estado de la validación
estructural](STRUCTURAL_VALIDATION_STATUS.md).
