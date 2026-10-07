> 🇬🇧 [English](../trade_household.md) · 🇪🇸 Español

# Preferencias del hogar con bienestar exacto por función de gasto

`puremacro.trade.household` es una capa de demanda del hogar independiente del
solucionador: cuatro reglas de preferencias sobre compuestos sectoriales, una
función de gasto normalizada al punto de referencia, demanda marshalliana y
hicksiana, matrices de Slutsky analíticas y EV/CV monetarias exactas. Se portó
del módulo de investigación IO `headlinePaper/preferences_2026-09-22/household.py`
(2026-09-22) y se conecta con las calibraciones y equilibrios de comercio de
puremacro mediante dos puentes.

```python
import numpy as np
from puremacro.trade.household import calibrate_household, fit_supernumerary_share

x0 = np.array([[25., 10.], [45., 60.], [30., 30.], [0., 0.]])   # benchmark spending, sectors x regions
eta = np.array([[.6, .5], [.9, 1.1], [1.3, 1.2], [np.nan, np.nan]])  # Engel targets (NaN on absent sectors)

prefs = calibrate_household(x0, "stone_geary", expenditure_elasticities=eta,
                            supernumerary_share=[.4, .5],
                            sector_codes=["AGR", "MAN", "SRV", "NONE"], country_codes=["USA", "ROW"])
print(prefs.summary())
print(prefs.calibration.to_dataframe())      # adding-up repair, admissible bounds, subsistence shares

prices = np.array([[1.3, .8], [.9, 1.25], [1.1, 1.05], [2., 3.]])
budget = np.array([105., 110.])
state = prefs.evaluate(prices, budget, validate=True)
welfare = prefs.welfare(prices, budget, validate=True)     # EV/CV against the calibration benchmark
print(welfare.to_dataframe())
S = prefs.slutsky(prices, budget)                           # (4, 4, 2) substitution matrices
```

## Qué calcula

Todos los arreglos tienen forma `(sector, región)`, salvo los presupuestos, la
utilidad y el bienestar, que tienen forma `(región,)`. Una unidad de compuesto
sectorial es la cantidad que cuesta una unidad monetaria a su precio comprador
de referencia, de modo que la matriz de gasto de referencia `x0` es también la
matriz de cantidades de referencia y los precios y la utilidad de referencia
valen uno. Los presupuestos regionales son `m0 = x0.sum(0)` y las
participaciones `w = x0 / m0`. La sustitución entre orígenes, los impuestos y el
presupuesto de equilibrio general son externos.

| regla | demanda | función de gasto |
|---|---|---|
| `fixed_baskets` | `q = x0 m / (m0 sum_i w_i p_i)` | `e(p, u) = m0 u sum_i w_i p_i` |
| `cobb_douglas` | `q_i = w_i m / p_i` | `e(p, u) = m0 u prod_i p_i^w_i` |
| `stone_geary` | `q_i = gamma_i + beta_i (m - sum_j p_j gamma_j) / p_i` | `e(p, u) = sum_i p_i gamma_i + S0 P(p) u`, `P(p) = prod_i p_i^beta_i`, `S0 = lambda m0` |
| `ces` | `q_i = w_i m p_i^-sigma P^(sigma-1)` | `e(p, u) = m0 P(p) u`, `P(p) = (sum_i w_i p_i^(1-sigma))^(1/(1-sigma))` |

Para Stone-Geary el excedente es `S = m - sum_j p_j gamma_j` y la utilidad
normalizada al punto de referencia es `u = S / (S0 P(p))`. El índice completo de
costo de vida a la utilidad de referencia es `e(p, 1) / m0`; `P` por sí solo
valora el gasto discrecional y no es ese índice. `ces` con `sigma = 0`
reproduce las canastas fijas y con `sigma = 1` Cobb-Douglas (una forma
`expm1`/`log1p` mantiene el índice estable cerca de uno). La demanda hicksiana
es `h(p, u) = grad_p e(p, u)`; las matrices de sustitución analíticas son

```
LES / Cobb-Douglas:  S_ij = S * (beta_i beta_j / (p_i p_j) - delta_ij beta_i / p_i^2)
CES:                 S_ij = sigma * (h_i h_j / m - delta_ij h_i / p_i)
fixed baskets:       S = 0
```

Son simétricas, semidefinidas negativas y satisfacen `S p = 0`.

### Bienestar

Con 0 el estado base y 1 el estado de comparación,

```
EV = e(p0, u1) - e(p0, u0)
CV = e(p1, u1) - e(p1, u0)
```

Ambas son positivas cuando hay ganancia de bienestar; `cv` es la cantidad que
puede retirarse a los nuevos precios, no la compensación requerida. Frente al
punto de referencia de la calibración esto se reduce a `EV = S0 (u1 - 1)` para
toda regla (incluidas las canastas fijas) y `CV = m1 - e(p1, 1)`; para LES
`CV = P(p1) EV`. `HouseholdWelfareResult` informa `ev`, `cv`,
`ev_pct_consumption`, `cv_pct_consumption` (ambas divididas por el presupuesto
base del hogar `e(p0, u0)`, de modo que los sistemas de preferencias comparten
un denominador), `utility`, `base_utility`, `cost_index` y `price_index` por
región. `compute_household_welfare(prefs, p1, m1, base_prices=p0, base_budget=m0)`
acepta un estado base arbitrario; `prefs.welfare(p, m)` usa el punto de referencia.

### Calibración a partir de objetivos de Engel

En el punto de referencia la elasticidad gasto LES es `eta_i = beta_i / w_i`,
así que `beta = w eta`. La adición de Engel `sum_i w_i eta_i = 1` se impone
dividiendo los objetivos de cada región por su media ponderada por gasto; las
medias originales, el ajuste máximo y los objetivos reparados quedan registrados
en `prefs.calibration` (un `HouseholdCalibrationResult`; `repair_adding_up=False`
rechaza objetivos inconsistentes). La subsistencia es `gamma = x0 (1 - lambda eta)`
con `lambda = S0 / m0` la participación supernumeraria, que debe cumplir
`0 < lambda <= 1 / max_i eta_i` en cada región para que `gamma >= 0`. `lambda = 1`
da Cobb-Douglas. Los sectores no observados permanecen como ceros estructurales
exactos; los objetivos omitidos significan explícitamente elasticidades unitarias.

`lambda` no está identificada por los datos IO. `supernumerary_share=None` (el
valor por defecto) usa el valor supuesto 0.5 y registra
`metadata["supernumerary_share_default_is_assumption"] = True` (una
participación suministrada registra `False`); la auditoría de calibración lo
dice literalmente: "Stone-Geary surplus shares are supplied assumptions or external
estimates, not identified by IO accounts and Engel targets".
`fit_supernumerary_share` proyecta objetivos externos de elasticidad precio
compensada propia sobre la identidad LES de referencia
`xi_ii = -lambda eta_i (1 - beta_i)` por mínimos cuadrados ponderados regionales,
recortada al intervalo admisible, y devuelve un `SupernumeraryFitResult` con el
ajuste sin restricción, las cotas, los indicadores de recorte, los residuos por
sector y la RMSE ponderada. Las categorías con peso cero no influyen en el
ajuste (sus valores objetivo pueden ser marcadores). Una región sin respuesta
de precios ponderada lanza un error.

```python
own = -prefs.supernumerary_share * prefs.eta * (1 - prefs.beta)   # implied targets
fit = fit_supernumerary_share(x0, eta, own)
print(fit.summary(), fit.to_dataframe())
```

### Dominio y paso complejo

Los núcleos de evaluación conservan entradas complejas, de modo que
`prefs.evaluate(p + 1e-25j * dp, m)` da derivadas direccionales exactas, y
nunca recortan. Pase `validate=True` (o llame a `prefs.validate_domain(p, m)`)
en los estados reales aceptados: precios no positivos, entradas no finitas y
excedente agotado lanzan `HouseholdDomainError`, una subclase de `ValueError`.
Se requiere un excedente estrictamente positivo para una derivada de demanda
interior y una utilidad normalizada positiva.

## Relación con el módulo flexible

`puremacro.trade.flexible` también incluye un sistema Stone-Geary
(`FlexiblePreferenceConfig.mu_s`). Con objetivos de Engel unitarios y
`lambda = 1 - mu_s` ambos dan los mismos niveles de demanda al ingreso de
referencia (la batería de pruebas compara `compute_stone_geary_final_demand`
con `prefs.demand` a 1e-12 en la calibración sintética de 2 países y 2
sectores). Un `mu_s = 1 - lambda*eta` por sector y región (con la `eta`
reparada) hace lo mismo con objetivos no unitarios y una `lambda` regional
(también comprobado a 1e-12); la reparación de adición y la comprobación de
admisibilidad `lambda <= 1/max(eta)` deben hacerse aquí primero. El
solucionador de equilibrio general flexible usa un LES con cesta supernumeraria
de proporciones fijas que anida la cesta heredada (allí un `mu_s` uniforme es
neutral); la paridad de niveles de demanda solo vale para
`compute_stone_geary_final_demand`.

Solo coinciden los niveles de demanda. El módulo flexible escala la
subsistencia con el ingreso (suavizado `tanh`), por lo que su demanda no tiene
función de gasto: en la calibración sintética su matriz de Slutsky por
diferencias finitas tiene una asimetría de 0.06 incluso al ingreso de
referencia, de 0.15 a la razón de ingreso 0.8 y de 0.02 a la razón 1.2, frente
a unos 2e-8 para la LES exacta de aquí (la prueba exige una asimetría flexible
mayor que 1e-3 y una exacta menor que 1e-7 en las tres razones), y sus niveles
de demanda se apartan de la LES exacta lejos del ingreso de referencia.
`FlexibleTradeEquilibriumResult.welfare_decomposition` sigue siendo, por tanto,
una aproximación, como afirma su docstring; este módulo ofrece la alternativa
exacta por función de gasto.

## Puentes con calibraciones y equilibrios de comercio

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade.household import (calibrate_household, household_expenditure_from_calibration,
                                       household_prices_from_result, compute_household_welfare_from_results)

# Analytic 2-country x 2-sector table with three final uses (C, I, X); rows: sellers, taxes, labour, capital.
Z = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
y = np.array([100., 150., 120., 180.])
va = y - Z.sum(0) - .05 * y
data = np.zeros((7, 10))
data[:4, :4] = Z
data[4, :4], data[5, :4], data[6, :4] = .05 * y, 2 / 3 * va, va / 3
fd = y - Z.sum(1)
data[:2, 4:] = fd[:2, None] * np.array([.50, .25, .05, .10, .08, .02])
data[2:4, 4:] = fd[2:, None] * np.array([.10, .08, .02, .50, .25, .05])
data[4, 4:] = .02 * data[:4, 4:].sum(0)
calib = calibrate_trade_model(data, ns=2, nc=2, nfd=3, country_codes=["HOME", "ROW"], sector_codes=["GOODS", "SERV"])

x0 = household_expenditure_from_calibration(calib, category=0)       # (2, 2) benchmark purchaser spending
rates = np.array([.15, 0.])                                           # HOME taxes all its imports at 15 percent
base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
cf = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-10)
prices, budget = household_prices_from_result(cf, calib)              # (2, 2) composite prices, (2,) budgets
for rule, kw in (("fixed_baskets", {}), ("cobb_douglas", {}), ("stone_geary", {"supernumerary_share": .6}),
                 ("ces", {"ces_elasticity": .5})):
    prefs = calibrate_household(x0, rule, sector_codes=calib.sector_codes, country_codes=calib.country_codes, **kw)
    welfare = compute_household_welfare_from_results(prefs, calib, cf, base_result=base)
    print(rule, np.round(welfare.ev_pct_consumption, 4))
hicks = compute_hicksian_welfare(calib, cf, base_result=base, target_country="HOME")
print("Leontief basket", round(hicks.ev_pct_consumption, 4))          # equals the fixed_baskets household
```

`household_expenditure_from_calibration(calib, category)` devuelve
`x0[s, c] = A[s, c] theta[k, c] (l + k + T)[c]`, donde `A` es la participación
del sector `s` en la canasta Leontief de orígenes de la categoría
(`sum_i afd[i, s, k, c]`). El gasto a precios comprador incluye el impuesto al
uso final, una proporción uniforme de la categoría en ambos modos contables,
de modo que las participaciones sectoriales a precios básicos y comprador
coinciden. La categoría 1 (inversión) se rechaza porque su gasto resta el
ahorro externo; las celdas negativas lanzan un error que las nombra. En la
tabla OCDE 77x11 incluida la categoría 0 está limpia (las 847 participaciones
sectoriales son positivas) mientras que la categoría 1 tiene celdas negativas.

`household_prices_from_result(result, calib, category=0, tau_fd=None)` devuelve
los precios compuestos Leontief con aranceles relativos al punto de referencia,
`P[s, c] = sum_i afd[i, s, k, c] p_i tf[i, s, k, c] / sum_i afd[i, s, k, c]`, y el
presupuesto comprador de la categoría `theta[k] (w l + r k + T)`.

* Los resultados con `accounting="consistent"` se reauditan exactamente como lo
  hace `compute_hicksian_welfare` (esquemas arancelarios registrados, residuos,
  campos) y se valoran con `_accounting.evaluate`: `tf` es el
  `final_tariff_multipliers` registrado y el presupuesto es el `expenditure` del
  bloque. Aquí `tau_fd` es opcional; si se da (las tasas nacionales de la
  solución o un arreglo completo de multiplicadores) se resuelve con las
  convenciones del solucionador y se contrasta con el esquema registrado, y una
  discrepancia lanza un error.
* Los resultados legacy no registran esquemas arancelarios, por lo que sus
  residuos no pueden reevaluarse: solo se aceptan si los residuos registrados
  cumplen la tolerancia propia de la solución (`metadata["tol"]`, 1e-8 si
  falta). Se valoran a los precios de estado de `x_sol`, a partir de los cuales
  el evaluador legacy construyó `p_fd` y los flujos de demanda final; el
  `p_sol` registrado es el precio de beneficio cero y difiere de ellos hasta el
  residuo de la solución (1.8e-7 en la calibración sintética con la
  tolerancia por defecto 2.5e-3). Pase el `tau_fd` usado en la solución
  (`None` significa libre comercio); se audita contra el `p_fd` almacenado
  (`ppfd_k = sum_i afd_ik p_i tf_ik`) y una discrepancia lanza un error. Ese
  agregado identifica un multiplicador extranjero por destino, de modo que las
  tasas nacionales quedan verificadas. Un esquema que varía entre orígenes
  extranjeros o sectores no puede verificarse con `p_fd`: en la calibración
  sintética un esquema distinto con los mismos agregados desplaza la EV del
  importador un 12 por ciento con Cobb-Douglas y un 22 por ciento con CES de
  elasticidad 3 (las canastas fijas no se ven afectadas). Por eso un esquema
  así lanza un error salvo con `trust_tau_fd_detail=True`, que acepta el
  detalle verificando solo sus agregados; es preferible
  `accounting="consistent"`, que registra el esquema. Los presupuestos legacy
  solo están disponibles para cierres de suma fija.
* Los resultados del modelo flexible no están soportados por el puente: calcule
  los precios compuestos Armington con `puremacro.trade.flexible` y llame
  directamente a `compute_household_welfare`.

`tol` (1e-8 por defecto) compara los campos registrados con sus valores
recalculados (las comprobaciones de campos de la auditoría consistente; en
legacy `p_fd`, `c_fd`, `w_sol`, `r_sol` y `T_sol`), un `tau_fd` suministrado con
un esquema consistente registrado y la dispersión de un esquema legacy entre
orígenes extranjeros. Los residuos de equilibrio se comprueban en cambio con la
tolerancia registrada por el solucionador, y los multiplicadores de demanda
final domésticos deben valer uno a 1e-14. Los errores nombran la función
llamada y, en `compute_household_welfare_from_results`, el estado que falla
(`eq_result` o `base_result`); la clave de metadatos del bienestar
`tariff_schedule_verification` registra cómo se verificó el esquema de cada
estado. Las etiquetas de las preferencias deben coincidir con `country_codes` y
`sector_codes` de la calibración salvo que sean las generadas por defecto
(`C00`, `S00`, ...), y el resultado de bienestar lleva las etiquetas de país de
la calibración.

Con `rule="fixed_baskets"` y una sola categoría de consumo el puente reproduce
`compute_hicksian_welfare` (canasta Leontief) de forma exacta; la batería de
pruebas verifica EV, CV y porcentajes a 1e-10 relativo para cada país en la
calibración sintética con un arancel del 15 por ciento (categorías 0 y 2) y en
una tabla no cuadrada de 3 países y 2 sectores, y la vía legacy contra la
identidad Leontief cerrada `EV = m1 ppfd0 / ppfd1 - m0`. Los hogares no
homotéticos y CES dan cifras distintas por construcción. La EV/CV de los
puentes hereda las convenciones de numerario y de ahorro externo del
equilibrio: con ahorro externo no nulo la convención del modelo consistente
depende del orden de los países, y también estas cifras, como en
`compute_hicksian_welfare` (en la tabla de 3 países, invertir el orden mueve la
EV Stone-Geary del primer país de 1.26 a 1.19 por ciento de su presupuesto).

Con la tabla OCDE incluida, la categoría de uso final 0 es HFCE + NPISH + GGFC,
de modo que un "hogar" calibrado a partir de ella valora el consumo agregado,
incluido el consumo final del gobierno; los metadatos del resultado lo
registran. Una separación solo de hogares requiere una fuente nativa con usos
finales separados (véase [bienestar hicksiano del consumo](trade_welfare.md)).

## Qué está validado y qué no

Comprobado en `tests/test_trade_household.py`:

* las doce pruebas de comportamiento y dualidad del módulo IO sobre su
  fixture analítico 4x2 (reproducción del punto de referencia para cada regla y
  elasticidad CES en {0, 0.4, 1, 2}, adición presupuestaria, homogeneidad,
  derivadas de Engel a 2e-10, `lambda = 1` igual a Cobb-Douglas, dualidad del
  gasto, lema de Shephard, simetría de Slutsky y semidefinición negativa por
  diferencias centradas a 7e-8, derivadas por paso complejo a 2e-9, errores de
  dominio, frontera de admisibilidad, límites CES, recuperación exacta de una
  `lambda` generadora a 2e-16, cotas activas e influencia nula de pesos cero);
* matrices de Slutsky analíticas contra diferencias centradas a 5e-8 en cada
  región, simetría exacta, `S p = 0` y semidefinición negativa;
* un oráculo primal SLSQP de minimización del gasto para Stone-Geary,
  Cobb-Douglas y CES (0.7, 1.8): gasto minimizado dentro de 1e-6 relativo de
  `e(p, u)` y minimizador dentro de 2e-4 de la demanda hicksiana;
* bienestar con base general: la identidad da exactamente cero, las
  comparaciones invertidas intercambian EV y CV, la base de referencia reproduce
  las formas cerradas; el cambio de moneda escala los niveles y conserva
  porcentajes, utilidad e índices;
* el puente de calibración sobre las tablas sintética y 77x11 incluida, la
  paridad con el módulo flexible al ingreso de referencia y su divergencia fuera
  de él, y el contraste de integrabilidad a las razones de ingreso 0.8, 1 y 1.2;
* los puentes de resultados consistentes y legacy sobre la tabla sintética
  cuadrada y una tabla no cuadrada de 3 países y 2 sectores: EV/CV de canastas
  fijas contra `compute_hicksian_welfare` para cada país (categorías 0 y 2), un
  certificado independiente (el gasto comprador por sector leído de los flujos
  de demanda final del modelo es igual a `P_s q_s` a 1e-12), la identidad
  Leontief legacy a 1e-11, soluciones legacy con la tolerancia por defecto del
  solucionador, la barrera de residuos, el rechazo de campos manipulados y de
  esquemas erróneos, la confianza explícita exigida para esquemas legacy por
  sector, las comprobaciones de etiquetas y los mensajes de error;
* paridad en proceso con la implementación IO sobre entradas aleatorias
  (arreglos de calibración, evaluación, demanda hicksiana, gasto, bienestar,
  auditorías de calibración y ajuste): la diferencia máxima observada es 0.0
  (idéntica bit a bit), omitida cuando el volumen de investigación no está montado.

No validado aquí, citando la documentación IO: la calibración "is a disclosed
transfer and reconciliation, not native microdata estimation"; "The inherited
targets generally cannot be matched exactly by LES, and a calibrated parameter
is not thereby empirically identified for 2019"; las restricciones de
sustitución LES "are therefore substantive: matching income elasticities and
projecting compensated own-price targets does not reproduce a general CDE
substitution matrix"; "These remain static policy comparisons, not lifetime
welfare". No se incluyen objetivos CDE de GTAP (los datos tienen licencia), y
la inserción en equilibrio general del IO (`PreferenceModel`,
`PreferenceInverse`, continuación arancelaria, estabilidad local) no forma parte
de este port: use los puentes con los solucionadores propios de puremacro.

Documentación relacionada: [contabilidad de comercio consistente](trade_accounting.md),
[bienestar hicksiano del consumo](trade_welfare.md), [política arancelaria hicksiana](trade_policy.md),
[economía espacial cuantitativa y equilibrio general de comercio](spatial_and_trade_ge.md).
