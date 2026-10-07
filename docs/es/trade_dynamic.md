> 🇬🇧 [English](../trade_dynamic.md) · 🇪🇸 Español

# MRIO dinámico con capital específico por sector

`puremacro.trade.dynamic` es una extensión dinámica, determinista y con
previsión perfecta del modelo insumo-producto multipaís de coeficientes fijos,
en la que cada celda país-industria de origen conserva su propio stock de
capital instalado con costes de ajuste a la Tobin-q. Es una portabilidad fiel
del motor de investigación `dynamic_model/native_*` (septiembre de 2026) a las
convenciones de puremacro: objetos de resultado congelados con
`to_dataframe()`, `to_markdown()`, `to_latex()` y `to_typst()`, constructores
que validan, errores estructurados y ningún solucionador que devuelva
`converged=True` sin superar un certificado contable independiente en cada
fecha.

**No** es el CGE estático de `solve_trade_equilibrium`: no hay sustitución
Armington, el trabajo es inelástico y móvil dentro de cada país, los países
están en autarquía financiera y el consumo público forma parte de la cesta del
hogar. Su medida de bienestar es un equivalente de consumo CRRA, no la EV
hicksiana de [`compute_hicksian_welfare`](trade_welfare.md). Lea las
limitaciones al final antes de interpretar cualquier cifra.

## Inicio rápido con la tabla 77x11 incluida

```python
import numpy as np
from puremacro.trade import load_icio_data
from puremacro.trade.dynamic import (
    DynamicAccounts, DynamicTariff, DynamicEconomy, calibrate_dynamic,
    solve_dynamic_steady_state, solve_dynamic_transition, tariff_path,
    stability_report,
)

icio = load_icio_data(source="legacy", return_structured=True)
# Three cells of the bundled investment column are negative: choose a policy.
accounts = DynamicAccounts.from_icio(icio, negative_investment="to_inventory")
calibration = calibrate_dynamic(
    accounts, beta=.96, delta=.08,
    factor_policy="reclassify_losses", investment_policy="reallocate",
    accounting_policy="reconcile_rounding",
)
print(calibration.summary())          # 847 active cells, 21 countries rebasketed
print(calibration.to_dataframe().head())

economy = DynamicEconomy(calibration, adjustment_cost=2., risk_aversion=2.)
baseline = solve_dynamic_steady_state(economy)            # the benchmark, 0 iterations
tariff = DynamicTariff.uniform(calibration, "USA", .01, sectors="merchandise")
terminal = solve_dynamic_steady_state(economy, tariff)     # exact 2C condensation
print(terminal.to_dataframe().loc[["USA", "MEX", "CHN"]])

report = stability_report(economy, baseline)               # Blanchard-Kahn count
print(report.summary())   # 846 stable roots for 847 predetermined stocks: NOT determinate
```

La tabla incluida procede de una exportación dañada de la OCDE (véase el
[aviso de procedencia](ADVISORY.md)); aquí sus cifras son valores de regresión
del software, no estimaciones de la OCDE. La última línea es el hallazgo comentado en *Limitaciones*: con este cierre la
tabla incluida es localmente indeterminada, de modo que
`solve_dynamic_transition` se niega a ejecutarse por defecto. Con
`require_determinacy=False` se resuelve igualmente la pila de horizonte finito
(a 40 fechas convergen aranceles de EE. UU. sobre mercancías del 1% al 4%; los
del 5% y 10% chocan con la frontera de inversión interior en Irlanda y lanzan
`DynamicSolveError`):

```python
policies = tariff_path(economy.zero_policy(), tariff, horizon=40)
run = solve_dynamic_transition(economy, policies, terminal=terminal,
                               require_determinacy=False)
print(run.summary())
print(run.welfare.to_dataframe().loc[["USA", "MEX"]])
```

## Un ejemplo pequeño reproducido por la suite de pruebas

Las cuentas analíticas de dos países y dos sectores son el fixture que usan
todas las pruebas de este repositorio y las del propio motor IO. Son un
fixture de software, nunca evidencia empírica.

```python
import numpy as np
from puremacro.trade.dynamic import (
    analytic_two_country_accounts, calibrate_dynamic, DynamicEconomy,
    DynamicTariff, tariff_path, solve_dynamic_steady_state,
    solve_dynamic_transition, compare_horizons, run_horizon_ladder,
)

accounts = analytic_two_country_accounts()
calibration = calibrate_dynamic(accounts)
economy = DynamicEconomy(calibration)
zero = economy.zero_policy()
rates = np.zeros((calibration.n_cells, calibration.n_countries))
rates[calibration.country == 0, 1] = .06     # BBB taxes AAA's goods at 6%
rates[calibration.country == 1, 0] = .08     # AAA taxes BBB's goods at 8%
shock = DynamicTariff.build(rates, consumption_rates=rates * .7,
                            investment_rates=rates * 1.3, label="analytic tariff")
terminal = solve_dynamic_steady_state(economy, shock, tol=1e-11)

# Announced at date 0, implemented at date 2, permanent.
ladder = run_horizon_ladder(
    economy, lambda T: tariff_path(zero, shock, horizon=T, announcement=2),
    horizons=(80, 160, 320), terminal=terminal, tol=1e-10,
)
print(ladder.to_dataframe())
accepted = ladder.accepted            # the 320-date solution (status "verified")
print(accepted.welfare.summary())      # AAA -0.24%, BBB +0.11% of benchmark consumption
```

En este fixture la comparación 80 frente a 160 coincide en las primeras 20
fechas (3.6e-7 en logaritmos) pero el bienestar todavía se mueve 0.0055 puntos
porcentuales, así que se rechaza; 160 frente a 320 supera la ventana
(2.7e-12), el bienestar (4.3e-5 pp) y la brecha del estado terminal (4.3e-5).
La raíz estable más lenta es 0.979, y por eso hacen falta cientos de fechas.

## El modelo

Celda `j` en el país `c(j)`, fecha `t`, precios de referencia iguales a uno.
El valor añadido es Cobb-Douglas normalizado en la referencia sobre capital
específico del sector y trabajo nacional; los insumos intermedios y las cestas
finales son de coeficientes fijos:

```
v[j,t] = w[c(j),t]^(1-alpha[j]) (R[j,t]/R0[j])^alpha[j]
(1-tax[j]) p[j,t] = sum_i A[i,j] (1+tau[i,c(j),t]) p[i,t] + b[j] v[j,t]
L[j,t] = (1-alpha[j]) b[j] v[j,t] y[j,t] / w[c(j),t]
K[j,t] = alpha[j] b[j] v[j,t] y[j,t] / R[j,t]
y = A y + omegaC C + omegaI I_national
PC[c] = sum_i (1+tauC[i,c]) omegaCtax[i,c] p[i] / (1-tC[c])   (exempt X part added)
PI[c] = sum_i (1+tauI[i,c]) omegaI[i,c] p[i] / (1-tI[c])
```

Capital, instalación y ecuación de Euler:

```
K[j,t+1] = (1-delta) K[j,t] + K[j,t] Phi(x[j,t]),   x = I/K
Phi(x) = x - phi/2 (x-delta)^2
q[j,t] = PI[c(j),t] / Phi'(x[j,t])
q[j,t] = m[c,t+1] (R[j,t+1] + q[j,t+1] G[j,t+1]),  G = 1-delta+Phi(x)-x Phi'(x)
m[c,t+1] = beta (C[c,t+1]/C[c,t])^(-sigma) PC[c,t]/PC[c,t+1]
```

El presupuesto del hogar paga exactamente el consumo y la inversión que
entrega el equilibrio de bienes:

```
PC C + PI sum_{j in c} I[j] + XN0[c] w_last
    = w L0[c] + sum_{j in c} R K + product and production tax receipts + tariff receipts
```

No hay bono negociado. Las exportaciones netas de referencia `XN0` son
transferencias mundiales equilibradas, fijas en unidades del salario del
último país (el numerario). Un presupuesto redundante (el del país con mayor
escala presupuestaria) se sustituye por la ecuación del numerario y se
comprueba después junto con todas las demás cuentas. Como el último país del
registro es a la vez el numerario salarial y el ancla de las transferencias,
el orden del registro de países tiene contenido económico: reordenar los
países cambia el cierre y, por tanto, las asignaciones reales siempre que se
muevan los salarios relativos (aproximadamente el cambio del salario relativo
del ancla por `XN0`), exactamente como en el motor IO. Reordenar los sectores
es un mero reetiquetado.

**Eliminación.** Las incógnitas retenidas por fecha son
`z_t = [log w (C), log C/C0 (C), log K_{t+1}/K0 (N)]`, es decir `N + 2C` por
fecha. La instalación se invierte en su rama creciente,
`x = delta + 2g/(1 + sqrt(1 - 2 phi g))` con `g = K'/K - 1`; el equilibrio de
bienes usa las respuestas de Leontief cacheadas `YC = (I-A)^-1 omegaC`,
`YI = (I-A)^-1 omegaI`; los precios de los factores se obtienen de las
condiciones Cobb-Douglas; los precios del productor resuelven un sistema
lineal ajustado por aranceles por cada política distinta (LU disperso hasta
600 celdas; por encima de 600 celdas, LU denso cuando la matriz de
coeficientes tiene densidad superior a .25 y, en otro caso, BiCGSTAB
precondicionado con respaldo GMRES). El
residuo retenido son `C` ecuaciones de equilibrio del trabajo, `C`
presupuestos con un numerario y `N` ecuaciones de Euler.
`DynamicEconomy.state_jvp`/`states_jvp` dan productos jacobiano-vector
analíticos exactos, por lotes por política.

**Certificado.** Cada fecha aceptada supera 17 comprobaciones reconstruidas
de forma independiente (bienes, beneficio nulo, trabajo, acumulación de
capital, pagos a factores, demanda de capital y de trabajo, tecnología,
condición de primer orden de la inversión, Euler, recaudación arancelaria,
presupuesto público, todos los presupuestos incluido el omitido, cuenta
corriente, equilibrio mundial de transferencias, valor de los bienes de
inversión) a `max(1e-8, tol)`.

**Estado estacionario.** `solve_dynamic_steady_state(method="condensed")`
resuelve una condensación exacta de `2C` incógnitas en salarios nacionales y
precios de inversión con jacobiano analítico denso (por defecto por encima de
240 incógnitas); `"full"` resuelve las `N + 2C` ecuaciones estacionarias con
Newton sin matriz. Ambos reevalúan el residuo completo y el certificado.
Ambos son métodos de Newton locales sin continuación de respaldo, y
`method="auto"` no reintenta con el otro método si falla: un arancel grande
puede necesitar `start` desde el estado estacionario de un arancel menor, y
un arancel prohibitivo puede no tener ningún estado estacionario interior (en
el fixture analítico un arancel uniforme del 100% lanza `DynamicSolveError`,
mientras que uno del 1000% converge a un estado de esquina con
`C_BBB/C0 = 0.056` que el diagnóstico de determinación clasifica como
indeterminado, 3 raíces estables para 4 stocks). Un `start` inviable o una
política cuyo sistema de precios ajustado por aranceles no tiene solución
positiva lanza `EconomicDomainError` (un `ValueError`) antes de cualquier paso
de Newton.

**Transición.** `solve_dynamic_transition` apila `T (N + 2C)` incógnitas con
capital inicial fijo y un estado estacionario terminal resuelto por separado
bajo la última política (mantenida más allá del horizonte), y ejecuta un
Newton inexacto sin matriz: `lgmres` (con respaldo `gmres`) precondicionado
por una tridiagonal aproximada capital-tiempo más bloques nacionales
(`preconditioner="capital"`), término de forzamiento
`max(1e-5, min(.1, sqrt(||F||)))`, tope de `.75` en los movimientos en
logaritmos y bisección de Armijo. Un estancamiento de la búsqueda lineal lanza
`DynamicSolveError` nombrando la ecuación, la fecha y la celda, la menor tasa
de instalación del último iterado aceptado (incluida la fecha frontera
terminal) y lo que el paso de Newton completo haría con ella. La conjetura
inicial por defecto es la mezcla exponencial del motor IO entre el estado
inicial y el terminal; cuando un choque grande haría desinvertir esa
conjetura (tasa de instalación `x <= 0`), el solucionador usa en su lugar una
mezcla lineal, y cuando incluso la mezcla lineal es infactible ninguna senda
de ese horizonte puede mantener la inversión interior, lo que se informa como
un `DynamicSolveError` que nombra la celda (`metadata["start"]` registra la
conjetura usada). Los fallos de la búsqueda lineal del estado estacionario
nombran en `location` la ecuación con el mayor residuo.

## Libro de ajustes de la calibración estacionaria

Un año IO no es un estado estacionario. `calibrate_dynamic` deriva
`R0 = PI0 (1/beta - 1 + delta)`, `K0 = alpha VA0 / R0`, la inversión de
reposición `delta K0`, reasigna el flujo estacionario de inversión dentro de
la absorción observada (recomponiendo la cesta solo donde la composición
observada no puede financiar la reposición, con
`investment_policy="reallocate"`), agrupa variaciones de existencias, objetos
valiosos y compras exentas en el extranjero en la cesta de consumo, y
reconcilia los impuestos sobre usos finales. `factor_policy="reclassify_losses"`
convierte componentes factoriales negativos en subvenciones a la producción y
recorta las participaciones a `capital_share_bounds`. Todo queda registrado en
`calibration.report` y en
`calibration.to_dataframe("countries" | "factors" | "investment")`.

Puentes: `DynamicAccounts.from_icio` acepta `ICIOData` (formato C/I/Cx
incluido: I contiene existencias y tres celdas son negativas, por lo que
`negative_investment` debe ser `"raise"` o `"to_inventory"`; las filas de
trabajo y capital son el reparto mecánico 2/3-1/3, así que las participaciones
del capital son uniformes 1/3), `RawIOData` (las categorías de los cargadores
como HFCE/NPISH/GGFC/GFCF/INVNT/DPABR se proyectan sobre el contrato
`("C","G","X","V","VAL")`; sin detalle factorial se asume y registra el reparto
2/3-1/3) y cualquier objeto de tabla de la sección 2. La columna `EXPORT` del
cargador de EXIOBASE (exportaciones a regiones fuera de la tabla) no tiene
mapeo por defecto porque no son compras de residentes en el extranjero; pase
`fd_map={"EXPORT": "X"}` u otro destino de forma explícita. `from_arrays`
acepta `F` como `(M, N, K)` o `(M, N*K)` y `TFD` como `(N, K)` o plano
`(N*K,)`; una tabla de impuestos transpuesta `(K, N)` se rechaza cuando
`N != K` (con tantos países como códigos de uso final la orientación no puede
detectarse, así que pase los destinos en filas y las categorías en columnas).
El valor por defecto `merchandise_sectors="isic_a_c"` marca como bienes las
etiquetas incluidas AGRI/MINQ/MANU, los 22 códigos de bienes de los 45
sectores de la OCDE y los códigos que son la letra A, B o C sola o que
empiezan por una de ellas seguida de un dígito, salvo C33 de FIGARO
(reparación e instalación); otros registros, como los códigos `i01...` de
EXIOBASE, necesitan una lista explícita de sectores. La regla y la lista de
bienes resultante se registran en `metadata["merchandise_sectors"]`.
`from_trade_calibration` lee `calib.data_calibra`; su columna Cx pasa a X
(compras de los residentes en el exterior) solo para la disposición incluida y
el mapeo `DPABR` de la OCDE, el Cx de los cargadores FIGARO, WIOD y Eora
(existencias con signo y objetos valiosos) pasa a V, y el Cx de EXIOBASE, que
incluye su columna de exportaciones, requiere un `cx_category` explícito.

## Políticas

`DynamicTariff` guarda tres tablas `N x C` de tipos ad valorem (entregas
intermedias, de consumo y de inversión) por celda de origen y país de destino;
las entregas nacionales y las compras de residentes en el extranjero nunca
pagan. Los aranceles específicos por sector de destino no son representables.
Los tipos deben superar `-1`: cero es libre comercio y los tipos negativos son
subvenciones a la importación, como en el motor IO.
`DynamicTariff.uniform(cal, "USA", .1, sectors="merchandise", exporters=("CHN",),
uses=("intermediate",))`, `from_rates` y `build` validan las tablas. La huella
(sha256 de las tres tablas) se deriva siempre de las tablas guardadas y nunca
se pasa como argumento, de modo que `dataclasses.replace(policy, rates=...)`
produce una huella nueva; las políticas idénticas comparten una factorización
de la red de precios y se rechaza una política cuyas tablas ya no coinciden
con su huella. Una selección sin ninguna celda de origen extranjera (por
ejemplo `exporters=("USA",)` para el importador `"USA"`) lanza `ValueError` en
lugar de devolver una política nula con etiqueta de arancel.
`tariff_path(baseline, shock, horizon=T, announcement=2,
duration=5)` construye calendarios anticipados, permanentes o temporales.

## Aceptación por horizonte

Una pila finita es una aproximación a una senda de horizonte infinito.
`compare_horizons(short, long)` exige: (1) que los primeros `periods=20`
estados retenidos coincidan dentro de `1e-5` en logaritmos; (2) que el
bienestar equivalente en consumo coincida dentro de `1e-4` puntos
porcentuales; (3) una comprobación terminal sobre la solución larga. El valor
por defecto `terminal_check="state_gap"` exige `max |z_T - z*| <= 1e-4` y da
el estado `verified`. El modo opcional `"discounted_wealth"` exige que el
valor descontado del capital final

```
D[c] = beta^(T-1) (C[c,T-1]/C0[c])^(-sigma) / PC[c,T-1] * sum_{j in c} q[j,T-1] K[j,T] / C0[c]
```

sea como mucho `1e-6` y da el estado explícitamente más débil
`verified_window_and_welfare`. En palabras de la documentación IO: "it does
not certify full terminal-state convergence or a rigorous utility-tail error
bound" y "These tolerances are not error bounds against the unknown
infinite-horizon solution." Ambas soluciones deben proceder de la misma
economía (el resultado de la transición registra una huella de las matrices
de calibración, el coste de ajuste, la aversión al riesgo y el numerario salarial en
`metadata["economy_fingerprint"]`), partir del mismo estado inicial, terminar
en el mismo estado terminal y seguir la misma política anunciada.
`run_horizon_ladder` resuelve horizontes crecientes, arrancando cada uno de la
senda anterior, y se detiene en la primera comparación aceptada.

## Bienestar

`consumption_equivalent_welfare` devuelve, por país, el cambio proporcional
permanente del consumo de referencia `e` con
`sum_t beta^t u(e) = sum_t beta^t u(C_t/C0)` para `u` CRRA (el mismo
`risk_aversion` que la ecuación de Euler; logarítmica en uno), la cola
estacionaria ponderada por `beta^T/(1-beta)`, el equivalente anual de gasto a
precios de referencia y su porcentaje del PIB de referencia. No hay agregación
entre países ni descomposición en canales.

## Diagnóstico de determinación

`stability_report(economy, steady)` linealiza las ecuaciones de cada fecha con
los propios JVP del modelo, forma el haz
`[[A_plus,0],[0,I]] s_{t+1} = [[-A_0,-A_minus_K],[E_K,0]] s_t` con
`s_t = (dz_t, dK_t)` y cuenta los autovalores generalizados: la determinación
exige exactamente `N` raíces estables para los `N` stocks predeterminados. El
resultado informa de la raíz estable más lenta, la raíz inestable más pequeña
y las cuotas de carga por país de su autovector. El punto debe ser
estacionario bajo la política pasada (o la que lleva un resultado de estado
estacionario): el informe rechaza con un `ValueError` un estado cuyo residuo
`F(z, z, z)` supere `stationarity_tol` (por defecto `1e-8`). El problema denso tiene orden
`n_vars + N` (1848 en la tabla incluida, decenas de segundos: de 24 a 36 s con
2 hilos BLAS en la máquina de referencia). `solve_dynamic_transition` guarda
el informe en la economía por huella de la política terminal, de modo que una
escalera de horizontes lo paga una sola vez.

## Qué está validado aquí y qué no

Verificado por pruebas de este repositorio (`tests/test_trade_dynamic.py`):

* Estacionariedad de la referencia, identidades de factores e impuestos,
  cuentas escalares independientes en estados estacionarios y a lo largo de
  transiciones, JVP analíticos frente a diferencias centrales (2e-6),
  condensación exacta frente al problema completo, jacobiano condensado frente
  a diferencias finitas, evaluación por lotes, sincronización de la
  anticipación, políticas temporales, salvaguardas de la comparación de
  horizontes, el diagnóstico descontado del punto final y todas las
  identidades de bienestar de las pruebas IO.
* Paridad con el motor IO en el fixture analítico: matrices de calibración,
  estados estacionarios, una senda de 24 fechas y una de 60 fechas resuelta
  con `lgmres` y el precondicionador de capital coinciden a 0.0 (idénticos bit
  a bit), y con `puremacro.dsge.solve_perfect_foresight` sobre las mismas
  ecuaciones a 40 fechas hasta 3.4e-13. Una prueba lenta y opcional repite la
  comparación IO en la tabla incluida (calibración permisiva, ecuaciones con
  LU densa, estado estacionario y una senda de 40 fechas al 1%). Los backends
  iterativos de precios y bienes coinciden con la LU dispersa hasta 1e-10 en
  una tabla dispersa con semilla fija.
* Salvaguardas: una política reconstruida con `dataclasses.replace` se
  resuelve como la política nueva, las comparaciones de horizonte rechazan
  soluciones de economías distintas y `stability_report` rechaza puntos no
  estacionarios.
* Tabla 77x11 incluida: libro de calibración (847 celdas activas, 0 ajustes
  factoriales, 21 países recompuestos, L1 de cestas 1.0% de la inversión
  mundial), estados estacionarios base y al 10%, una transición de 40 fechas
  al 1%, el contrato de fallo al 5% (ecuación de Euler de IRL:GOV en la fecha
  39, tasa de instalación 1e-10 en la fecha frontera terminal) y (lento,
  opcional) el recuento 846/847 con raíz inestable 1.00185 cargando el 90% en
  Irlanda y la escalera 160/320/640 (ventana 320 a 640 de 1e-14, bienestar
  5e-6 pp, brecha de estado 7e-2 falla, riqueza descontada 3e-11 pasa).

No validado aquí, citando la documentación IO:

* Parámetros: "A single IO table does not identify historical capital
  stocks, depreciation, adjustment costs, or intertemporal preferences. The
  demonstrations use beta=.96, delta=.08, adjustment cost 2, and CRRA risk
  aversion 2." La aceptación numérica "does not validate the economic
  parameter choices."
* Inversión: "This is an interior investment model; it does not claim to
  solve an investment-irreversibility complementarity problem." En la tabla
  incluida el choque admisible depende del horizonte: aranceles de EE. UU.
  sobre mercancías del 1% al 4% dan sendas interiores certificadas a 40 fechas
  y el 3% también a 160 fechas, mientras que la búsqueda lineal se estanca
  con el 4% a 160 fechas y con el 5% y el 10% a 40 fechas, cuando la
  instalación en IRL:GOV se acerca a cero en la fecha frontera terminal. Como
  el cierre es indeterminado en esta tabla, el choque admisible se reduce al
  crecer el horizonte. Un fallo de la búsqueda lineal es un indicio, no una
  prueba, de que no existe senda interior, y la frontera no se exploró de
  forma exhaustiva.
* Alcance: "fixed sourcing, Cobb-Douglas value added, fixed national labor,
  and financial autarky with balanced transfers. It does not expose inactive
  Armington or labor-supply elasticities." Las variaciones de existencias y
  los objetos valiosos registrados "do not identify inventory stocks." El
  consumo "includes government and nonprofit final demand in the
  representative household basket."
* Horizontes: los horizontes largos son intrínsecos (raíces estables más
  lentas de aproximadamente 0.98 en el fixture y en la tabla incluida). La
  ejecución IO de EXIOBASE 49x163 se aceptó solo como
  `verified_window_and_welfare` (brecha logarítmica terminal 0.00989 por
  encima de 1e-4) y "No GTAP horizon is accepted yet." Ninguna de las dos se
  reproduce aquí.
* La indeterminación 77x11 es una propiedad local del cierre linealizado en
  esta tabla (846 raíces estables para 847 stocks), no un defecto del
  solucionador; ninguna prueba de este repositorio la atribuye a un rasgo
  concreto de los datos, y otras tablas pueden ser también indeterminadas,
  así que ejecute `stability_report` antes de confiar en
  `require_determinacy=False`. El criterio `state_gap` rechaza correctamente
  la escalera al 1% mientras que `discounted_wealth` la aceptaría. Trate los
  resultados `discounted_wealth` solo como afirmaciones sobre la ventana
  inicial.

## Relacionado

* [Contabilidad de comercio consistente](trade_accounting.md), [bienestar
  hicksiano de consumo](trade_welfare.md) y [política arancelaria](trade_policy.md)
  para el modelo estático que esta extensión no sustituye.
* `puremacro.dsge.solve_perfect_foresight` como Newton apilado genérico usado
  de oráculo independiente.
* [Estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
