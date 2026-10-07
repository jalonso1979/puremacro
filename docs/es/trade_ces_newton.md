> 🇬🇧 [English](../trade_ces_newton.md) · 🇪🇸 Español

# Newton exacto por bloques con CES anidada sobre la contabilidad consistente

`puremacro.trade.ces_newton` resuelve el equilibrio comercial con contabilidad
consistente de [contabilidad comercial consistente](trade_accounting.md) con
una tecnología de producción CES anidada de tres niveles, usando el jacobiano
exacto en lugar de diferencias finitas. Cada paso de Newton cuesta dos
factorizaciones LU densas `M x M` más un pequeño complemento de Schur
macroeconómico. En una tabla sintética de 77 países por 11 sectores (`M = 847`,
2001 incógnitas), un arancel del 10% con elasticidades moderadas (0.1, 0.1,
0.5) se resuelve en menos de un segundo en un portátil (0.7 a 0.9 s medidos, 4
iteraciones de Newton), y un caso de elasticidad alta (elasticidad de origen 4)
con una tabla arancelaria aleatoria del 0 al 30% tarda unos segundos (2.6 a
3.7 s, 10 iteraciones). Todo estado devuelto ha superado un certificado de
flujos independiente; no se devuelve nada más.

El módulo procede del espacio de investigación IO (bajo
`headlinePaper/rebuild/`: `ces_newton.py`, `ces_continuation.py` y el
`vendor/puremacro/trade/corrected/ces.py` incluido, con la demostración
`empirical_2019/CES_NEWTON_PROOF.md`). Las citas de este documento proceden del
`README.md` y del `CHANGES.md` de ese espacio, y la auditoría de elasticidades
de su `results/elasticity_2019.json`. El motor IO resuelve un modelo distinto
(un factor compuesto, cierres DEK); lo que se traslada es la eliminación, las
pruebas de oráculo y la regla de homotopía.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model
from puremacro.trade.ces_newton import (NestedCESTechnology, solve_ces_block_newton,
                                        continue_tariff_homotopy, certify_ces_equilibrium)

# Two countries, one sector, two final uses (C, I): a hand-balanced table in the
# calibrate_trade_model layout (rows: deliveries, net taxes, labour, capital).
Z = np.array([[10., 12.], [8., 14.]])
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
production_tax = np.array([4., 6.]); final_tax = np.array([2., 1., 3., 2.])
output = Z.sum(1) + F.sum(1)
va = output - Z.sum(0) - production_tax
table = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                   np.r_[2 * va / 3, np.zeros(4)], np.r_[va / 3, np.zeros(4)]])
calib = calibrate_trade_model(table, ns=1, nc=2, nfd=2, country_codes=["A", "B"])

tau = np.ones((2, 1, 2)); tau[1, 0, 0] = 1.2        # A taxes intermediate imports from B at 20%
tau_fd = np.ones((2, 2, 2)); tau_fd[1, :, 0] = 1.2  # and final imports from B at 20%
technology = NestedCESTechnology(sigma_va_materials=0.3, sigma_sectors=0.5,
                                 sigma_origins=1.5, rho_va=0.8)
result = solve_ces_block_newton(calib, tau, tau_fd, technology=technology)
print(result.summary())
print(result.equilibrium.w_sol.ravel(), result.certificate["current_account"])

path = continue_tariff_homotopy(calib, tau, tau_fd, technology=technology)
print(path.stages_frame())
audit = certify_ces_equilibrium(calib, path.equilibrium.x_sol, tau=tau, tau_fd=tau_fd,
                                technology=technology)
print(max(audit.values()))
```

`result.equilibrium` es un `TradeEquilibriumResult` ordinario con
`metadata["accounting"] == "consistent"` y
`metadata["effective_method"] == "ces_block_newton"`. Dentro del repositorio,
la tabla congelada de la OCDE 2019 con tres regiones que usan las pruebas se
carga con `tools.reference_validation.validate_oecd.load_fixture()` seguido de
la función auxiliar privada `puremacro.trade._oecd_icio.condense_final_demand`
y de `puremacro.trade.data.package_mrio_to_calibration_result`. La tabla
incluida de 77 por 11 es rechazada por la contabilidad consistente (tres celdas
de inversión fusionada son negativas), así que las tablas nativas deben
condensarse primero en agregados C/I/X no negativos.

## Qué calcula

Las ecuaciones son exactamente las de `accounting="consistent"`: valoración a
precios de productor y de comprador, costes de factores homogéneos, un impuesto
a la producción sobre el ingreso por ventas, participaciones de gasto final con
impuestos incluidos, devoluciones de suma fija, ahorro exterior fijo en unidades
del numerario y primer precio de productor igual a uno. Solo cambia la
tecnología intermedia. La celda `j` (país `c`, sector `s`) compra a la celda de
origen `i` a `r_ij = p_i tau_ij` y combina los insumos mediante

```
h_sj = CES_so({r_ij}_{i in s}; u0_ij)            origins within a material sector
m_j  = CES_ss({h_sj}_s; eta0_sj)                 material sectors
v_j  = CES_rho({w_c, r_c}; (1 - alpha_j, alpha_j))   labour and capital
c_j  = CES_sv({v_j, m_j}; (v0_j, A0_j) / (1 - t_j))  value added and materials
```

con `CES_sigma(x; s) = (sum_i s_i x_i^(1 - sigma))^(1 / (1 - sigma))`, la media
geométrica en `sigma = 1` y el valor uno en un nido vacío. Las participaciones
de referencia son `u0_ij = a0_ij / sum_{k in s} a0_kj`,
`eta0_sj = sum_{i in s} a0_ij / A0_j`, `A0_j = sum_i a0_ij` y
`v0_j = (l_j + k_j) / y_j`; la identidad `A0_j + v0_j = 1 - t_j` se cumple celda
a celda en toda tabla de `calibrate_trade_model`. El lema de Shephard da los
requerimientos de insumos por unidad de producto

```
a_ij  = a0_ij (c_j / m_j)^sv (m_j / h_sj)^ss (h_sj / r_ij)^so
b_Lj  = b0_Lj (c_j / v_j)^sv (v_j / w_c)^rho,   b0_Lj = (1 - alpha_j) v0_j
b_Kj  = b0_Kj (c_j / v_j)^sv (v_j / r_c)^rho,   b0_Kj = alpha_j v0_j
(1 - t_j) c_j = sum_i a_ij r_ij + b_Lj w_c + b_Kj r_c
```

y la fila de beneficio nulo es `c_j - p_j`. Todo lo demás (renta de los hogares
`w L + r K + T`, gasto `theta` por renta menos el ahorro exterior en inversión,
cestas de origen Leontief `afd`, aranceles sobre valores de productor, el
presupuesto público, el cierre de ahorro fijo y la fila del numerario que
sustituye a la primera ecuación de bienes) no cambia, así que con **tecnología
plana** (`sigma_va_materials = 0`, `sigma_sectors = sigma_origins`,
`rho_va = 1`) el residuo iguala a `_accounting.evaluate(sigma=sigma_origins)`
fila por fila. La CES plana de `_accounting` usa pesos `a_ij / A_j` sobre los
`M` orígenes; una CES anidada con elasticidades iguales en los niveles de sector
y origen colapsa exactamente en ese índice. `NestedCESTechnology.is_flat`
informa de este caso y `NestedCESTechnology.from_flexible` traduce una
`FlexibleTechnologyConfig` (`sigma_y`, `sigma_inter`, `rho_va`). Las
elasticidades son insumos estilizados: el registro IO afirma que "all CES
elasticities are explicitly stylized" (`CHANGES.md`).

## La acción exacta del jacobiano

Las incógnitas son `x = [log p (M); log y (M); log r (N); log w (N); T (N); XN (N - 1)]`
y las filas son bienes (la fila 0 es el numerario `p_0 - 1`), precios, trabajo,
capital, balanzas `XN - B0` y presupuestos, en el orden y la escala en niveles
de `_accounting.evaluate`. Con participaciones de coste corrientes
`alpha_ij = a_ij r_ij / ((1 - t_j) c_j)`, `beta_Lj = b_Lj w_c / ((1 - t_j) c_j)`
y `beta_Kj = b_Kj r_c / ((1 - t_j) c_j)` (Euler: suman uno), participaciones de
nido `u_ij`, `eta_sj`, `theta_Lj` construidas a partir de ellas, y direcciones
logarítmicas `l` (precios), `lw`, `lr`:

```
dlog c_j = sum_i alpha_ij l_i + beta_Lj lw_c + beta_Kj lr_c
dlog h_sj = sum_{i in s} u_ij l_i,   dlog m_j = sum_s eta_sj dlog h_sj,   dlog v_j = theta_Lj lw_c + theta_Kj lr_c
da_ij = a_ij [sv dlog c_j + (ss - sv) dlog m_j + (so - ss) dlog h_{s(i)j} - so l_i]
db_Lj = b_Lj [sv dlog c_j + (rho - sv) dlog v_j - rho lw_c]      (db_Kj likewise with lr_c)
d[y_j sum_i (tau_ij - 1) p_i a_ij] = duty_j dlog y_j + y_j sum_i (tau_ij - 1) p_i (da_ij + a_ij l_i)
```

La eliminación sigue la demostración IO (`CES_NEWTON_PROOF.md`) con dos cambios
impuestos por el cierre de puremacro: hay dos factores, y el numerario sustituye
a la primera ecuación de bienes mientras todos los presupuestos nacionales
permanecen en el sistema. `CESBlockJacobian.solve(r)` calcula `J^{-1} r`:

1. leyendo directamente el bloque de balanzas (`dXN = r_bal`, un bloque identidad);
2. resolviendo el bloque de precios `(diag(p) - diag(c) alpha^T) l = diag(c)(beta_L C lw + beta_K C lr) - r_p`
   una vez para el término forzante y una vez por cada dirección unitaria de precio de factor (`L_r`, `L_w`);
3. resolviendo el bloque de bienes `(I - a) diag(y) dlog y = (da) y + afd . dq + rhs`, donde la
   ecuación de bienes omitida lleva un valor desconocido `mu` en `rhs[0]`;
4. ensamblando el complemento de Schur `(3N + 1) x (3N + 1)` de las filas de trabajo,
   capital, presupuesto y numerario en `(log r, log w, T, mu)` empujando direcciones
   unitarias a través de `response` en bloques de `block_size` columnas, factorizándolo
   por LU y sustituyendo hacia atrás.

`apply(v)` evalúa `J v` con los mismos diferenciales; `as_preconditioner()` y
`as_operator()` envuelven ambos como `scipy.sparse.linalg.LinearOperator`.
`macro_condition` es el número de condición del bloque de Schur escalado por
filas y columnas (`3N + 1 = 232` filas con `N = 77` en lugar de
`4N - 1 = 307`: las `N - 1` filas de balanzas forman un bloque identidad y se
eliminan primero). `CESBlockJacobian.n` es el número de celdas `M`; `dim`
(igual a `shape[0]`) es el número de incógnitas `2M + 4N - 1`.

La invertibilidad es una condición, no un teorema. El bloque de precios
`diag(p) - diag(c) alpha^T` es una Z-matriz cuya fila `j` es dominante cuando
`p_j > c_j (1 - beta_Lj - beta_Kj)`, lo que se cumple cerca de cualquier raíz
(donde `p = c`) siempre que cada celda pague alguna renta de factores; allí es
una M-matriz y su inversa es la serie de Neumann convergente. `I - a` es
invertible cuando la matriz de insumos corriente tiene radio espectral menor que
uno (Hawkins-Simon). El bloque de Schur se factoriza con pivoteo parcial y se
registra su número de condición. El README de IO lo expresa así: "The proof is a
conditional local identity, not a convergence or global-equilibrium theorem."

## Aceptación y contrato de fallo

`solve_ces_block_newton` es un Newton amortiguado: dirección `-J^{-1} R(x)`,
primer paso `min(1, 1 / max|dx_log|)`, descenso de Armijo del residuo escalado al
cuadrado con factor `1 - 1e-4 step`, como máximo `max_backtracks` bisecciones,
y un filtro de admisibilidad (precios, producciones, precios de factores y
rentas finitos y positivos; cantidades finales positivas en las categorías
activas y exactamente cero en las estructuralmente ausentes).

El bucle tiene una sola prueba de aceptación, la misma que aplican después el
posprocesado y `compute_hicksian_welfare`. Se detiene solo cuando (i) cada fila
impuesta del residuo, dividida por su escala de referencia (producción,
dotación o renta), es como máximo `tol` (por defecto `2e-11`), y (ii) cada fila
en niveles es como máximo `tol * max(scale)`, incluidas las filas que ninguna
ecuación de Newton impone y que solo se cumplen por la ley de Walras: la
ecuación de bienes omitida y las balanzas exteriores realizadas. Además exige
que ningún gasto final sea negativo. `equilibrium.metadata["tol"]` registra
esa cota absoluta. Cuando (i) se cumple pero (ii) no, el bucle da pasos de
Newton de pulido adicionales; normalmente basta uno. Cuando los flujos son tan
grandes respecto de las escalas de referencia que el redondeo por sí solo
supera la cota, el bucle lanza un error. Por ejemplo, un multiplicador de
importación de 1e8 con cestas de origen finales fijas hace que los aranceles
devueltos alcancen unas 3e7 veces la renta de referencia; el error indica
entonces la auditoría en niveles y la cota, y el remedio es un `tol` más laxo.

Un estado se devuelve solo si, además, `certify_ces_equilibrium` es como máximo
`certificate_tol` (por defecto `1e-8`). El certificado reconstruye los flujos
intermedios y de factores a partir de la tecnología en `(p, r, w)` con bucles
explícitos por país. Las cantidades finales, el gasto final y los ingresos
arancelarios registrados proceden del ensamblaje de la demanda del residuo, y
el certificado los comprueba mediante las claves de valores de bienes, ecuación
omitida, presupuestos, cuenta corriente, aranceles e identidad del PIB. Informa
de beneficio nulo, agregación de costes, valores de bienes, la ecuación de
bienes omitida, la cuenta corriente, aranceles reconstruidos bilateralmente,
vaciado de factores, presupuestos de hogares y gobierno, la identidad del PIB y
el residuo raíz escalado. Cada clave se prueba frente a la violación que
nombra: perturbaciones de 1e-5 en cada incógnita, fallos inyectados en el
ensamblaje y una tecnología que rompe la identidad de Euler en 1e-6.

Los límites de iteraciones, los fallos de búsqueda lineal, los iterados
inadmisibles, los bloques macro singulares y los certificados fallidos lanzan
`CESNewtonError` con el `history` de iteraciones y el último `macro_condition`.
Un fallo de búsqueda lineal indica qué prueba rechazó los pasos de prueba. Si
los rechazó la prueba de Armijo, el mensaje informa del número de condición,
porque un jacobiano casi singular es la causa habitual. Si todos los intentos
salieron del dominio admisible, el mensaje nombra la cantidad, la categoría y
el país que restringen en el paso de prueba más pequeño. Cada registro del
historial cuenta ambos tipos de rechazo (`rejected_armijo`,
`rejected_inadmissible`) y guarda la auditoría en niveles
(`level_audit_before`). Nunca se devuelve un estado no convergido con
`converged=True`.

`continue_tariff_homotopy` mueve linealmente las tablas arancelarias
intermedias y finales desde una tabla inicial (libre comercio por defecto)
hasta el objetivo con una fracción adaptativa: primer paso `0.25`, crecimiento
`1.5` con tope `0.4` tras una etapa que necesite como máximo `fast_iterations`
pasos de Newton, bisección tras un fallo, abandono por debajo de `min_step`.
Cada etapa aceptada supera las mismas pruebas de residuo, auditoría en niveles
y certificado que una resolución directa, y se empaqueta al aceptarse (una
etapa cuyo empaquetado falla cuenta como intento fallido). Los intentos
fallidos se registran en `failed_stages`. Todo abandono lanza `CESNewtonError`
con `stages`, `failed_stages` y `last_result` (la última etapa aceptada, o
`None` cuando la propia tabla inicial queda sin resolver). `progress` recibe
`{"fraction", "step"}` antes de cada etapa de prueba y los registros del
historial de Newton de cada etapa.

En una tabla aleatoria de 20 por 11 con elasticidad de origen 4, un Newton
directo desde la referencia falla en la búsqueda lineal: la prueba de Armijo
rechaza todos los intentos y el número de condición macro llega a 8e10. La
homotopía alcanza el objetivo en tres etapas; ese es el propósito de la
herramienta. No todo fallo es una singularidad. Con otras tablas arancelarias
aleatorias sobre la misma tabla, el Newton directo falla porque el filtro de
admisibilidad restringe: el iterado queda contra la inversión nula de un país,
y el error indica que todos los pasos de prueba salieron del dominio admisible.
En el caso examinado (semilla 0, elasticidad plana 2), relajar el filtro lleva
a una raíz cuyo gasto de inversión en un país es -865, porque el ahorro
exterior nominal fijo (7268) supera la participación de la inversión en la
renta (6402). Esa raíz es inadmisible bajo este cierre, y una homotopía
arancelaria no puede cambiarlo.

Los puntos singulares son una propiedad del modelo, no de la portación. En la
tabla congelada de la OCDE con tres regiones, el determinante del jacobiano de
referencia cambia de signo dos veces entre elasticidades de origen 0.5 y 1. El
valor singular mínimo del jacobiano escalado es 4.8e-6 en 0.99, frente a 5e-3
en 0 o 2. Las pruebas lo comprueban con diferencias centrales de
`_accounting.evaluate`, con independencia de este módulo. Con un arancel
estadounidense del 10% y elasticidad de origen unitaria falla el Newton
directo, y también fallaron el Newton por diferencias finitas de puremacro y
`hybr` cuando se comprobó. La homotopía certifica el punto final de (1,1,1,1),
pero la trayectoria de (0,1,1,1) se estanca junto a la referencia. Ambos fallos
se lanzan con el crecimiento del número de condición en `history`. La
homotopía no es una garantía de recuperación: en
la auditoría IO de elasticidades, 12 de 21 perfiles nativos fueron aceptados y
9 quedaron sin resolver, con sus primeros intentos de homotopía estancados en
residuos entre 1.7e-4 y 1.4e-3 (`results/elasticity_2019.json`; todos los
perfiles `unit_origin` y `middle`, dos `cobb_douglas` y un `top`). El README de
IO afirma: "The discrete checks do not certify a continuous branch between
stages. The sign is not robust across the accepted elasticity profiles; failed
procedures are not nonexistence certificates."

## Bienestar y resultados

`CESBlockNewtonResult` envuelve el `TradeEquilibriumResult` (`equilibrium`), la
tecnología, `iterations`, `calls`, `seconds`, el `residual_max` escalado, el
`certificate`, `macro_condition`, el `history` de Newton y las `stages` y
`failed_stages` de la homotopía; `to_dataframe()` muestra el historial y
`stages_frame()` y `certificate_frame()` los otros dos, con `to_markdown()`,
`to_latex()`, `to_typst()` y `summary()` como en el resto de puremacro.

Con tecnología plana el resultado es un equilibrio de contabilidad consistente
en todos los sentidos y [`compute_hicksian_welfare`](trade_welfare.md) lo acepta
sin cambios (`metadata["hicksian_welfare_supported"]` es verdadero y
`metadata["sigma"]` es `sigma_origins`). Con tecnología anidada
`metadata["hicksian_welfare_supported"]` es falso. La bandera es informativa:
`compute_hicksian_welfare` no la lee. En su lugar, su auditoría reevalúa el
estado con la CES plana de `_accounting.evaluate`, que rechaza (con un
`ValueError`) todo estado anidado cuya raíz difiera de la del modelo plano. Un
estado anidado que también resuelve el modelo plano se acepta. Un ejemplo es la
tecnología (0, .5, .5, .7) sobre la tabla de la OCDE con tres regiones, donde
las participaciones de capital uniformes hacen inerte el nido capital/trabajo
(`w = r`). La EV que devuelve sigue siendo correcta, porque la tecnología no
cambia las preferencias de consumo. Por la misma razón, el bienestar hicksiano
de consumo para resultados anidados requiere las funciones de gasto de los
hogares evaluadas a los precios de comprador del resultado (`Pfd_final`) en
vez de una nueva resolución; véase [preferencias del hogar](trade_household.md).

## Guarda de memoria

Medido con `tracemalloc` en una tabla sintética de 77 por 11 (`M = 847`; una
matriz `M x M` de dobles ocupa 5.7 MB). Los datos del problema retienen unas
3.3 matrices de ese tamaño, los flujos del estado actual 5.9 y el propio
operador 6.2 (dos factores LU, participaciones de coste y de nido, flujos
arancelarios). En total son unas 15 matrices (88 MB). Una resolución de Newton
completa alcanza un pico de unas 22 matrices (127 MB), porque la búsqueda lineal
evalúa un estado de prueba mientras el actual sigue vivo. El pico crece con
`M^2`: unos 2.1 GB con `M = 3465` (77 países por 45 sectores) y 2.8 GB con el
valor por defecto `max_cells = 4000`. `max_cells` rechaza tablas mayores antes
de reservar memoria, y el mensaje de error indica ambas estimaciones. Nada se
reserva al importar, así que el módulo es importable en el playground del
navegador. Allí el límite de memoria de WebAssembly (unos pocos GB como
máximo) hace del valor por defecto una cota superior más que un tamaño de
trabajo.

## Qué está validado y qué no

`tests/test_trade_ces_newton.py` comprueba, sobre una tabla no negativa de 3 por
2 construida a mano con una categoría de compras en el extranjero ausente y
contraaranceles bilaterales, su variante con X activa, la tabla congelada de la
OCDE 2019 con tres regiones y una tabla sintética de 20 por 11 con semilla (77
por 11 marcada `slow`):

- el oráculo de inversa exacta en un punto aleatorio que no es equilibrio: el
  jacobiano por paso complejo del residuo del módulo satisface `J solve(r) = r`
  y `apply(solve(r)) = r` hasta `3e-10` en unidades escaladas para las
  tecnologías (0,0,0,1), (.1,.3,1.5,1), (1,1,1,1), (.5,1,4,.7), (.1,.1,.5,.7) y
  tres planas (observado `6e-14` en las tablas pequeñas, `2e-12` en 20 por 11);
- las diferencias centrales de `_accounting.evaluate` con tecnología plana
  invierten el operador hasta `1e-6` y coinciden con el jacobiano por paso
  complejo;
- paridad fila a fila del residuo con `_accounting.evaluate` para `sigma` en
  {0, .5, 1, 2} hasta `1e-12` relativo, el límite Leontief frente a
  `solve_trade_equilibrium(accounting="consistent")` hasta `1e-10`, equilibrios
  CES planos y EV hicksiana frente a los solucionadores de diferencias finitas,
  y los flujos posprocesados frente a `_accounting.postprocess`;
- reproducción de la referencia (`a = a0`, `b = b0`, residuo por debajo de
  `1e-12`) para toda tecnología, la identidad de Euler, derivadas de Shephard
  por diferencias centrales, participaciones que suman uno;
- certificado como máximo `1e-8` y libro contable como máximo `1e-10` veces la
  mayor escala de referencia (producción, dotación o renta; en torno a `1e-11`
  relativo observado en la tabla de la OCDE) en toda solución aceptada, toda
  fila en niveles dentro de `metadata["tol"]`, homogeneidad nominal,
  invariancia de precios y porcentajes de bienestar ante la unidad monetaria;
- una sola prueba de aceptación: un barrido de 66 resoluciones sobre las tablas
  construidas a mano, 4 de las cuales necesitan un paso de pulido, devuelve solo
  estados que superan la auditoría del posprocesado, incluido un caso de
  regresión que un bucle anterior detenía en un residuo escalado de `1.45e-11`
  con una brecha de exportaciones netas de `4.8e-9` frente a la cota `3.9e-9`;
- sensibilidad del certificado: cada clave salta (por encima de `1e-7`) ante la
  violación que nombra, y se rechaza una tecnología que viola la identidad de
  Euler en `1e-6`;
- mensajes de fallo que nombran su causa (dirección limitada por la
  admisibilidad, fallo de Armijo, suelo de redondeo con multiplicadores
  prohibitivos), abandonos de la homotopía que conservan su registro, y la
  cuasi singularidad de la OCDE mediante diferencias centrales de
  `_accounting.evaluate`;
- el contrato de fallo, la certificación de etapas de la homotopía, el registro
  de fallos forzados, la reanudación desde una tabla parcial, la independencia
  del tamaño de bloque hasta `1e-12`, una raíz densa `hybr` independiente del
  mismo residuo, un golden autogenerado de la OCDE, y el núcleo CES compartido
  frente al `ces_index` de IO en un subproceso cuando el volumen de
  investigación está montado.

No validado: paridad con los equilibrios del motor IO (un factor compuesto,
cierres DEK, inventarios exógenos: un modelo distinto), convergencia para
cualquier perfil de elasticidades más allá de las tablas anteriores, bienestar
con tecnología anidada, ejecución en GPU, preferencias LES, márgenes,
penalizaciones de capacidad y cierres fiscales distintos de la suma fija (el
módulo flexible conserva todo eso). Los umbrales de residuo y certificado son
comprobaciones en coma flotante, no demostraciones por intervalos.

Relacionado: [contabilidad comercial consistente](trade_accounting.md),
[bienestar hicksiano de consumo](trade_welfare.md),
[política arancelaria hicksiana y recuperación del solucionador](trade_policy.md),
[estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md).
