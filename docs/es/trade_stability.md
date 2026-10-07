> 🇬🇧 [English](../trade_stability.md) · 🇪🇸 Español

# Estabilidad local reducida de un equilibrio comercial (experimental)

`reduced_stability` plantea una única pregunta acotada sobre un equilibrio comercial convergido: si los precios relativos de los factores respondieran a su propio exceso de demanda mientras los precios al productor, las producciones brutas y los presupuestos fiscales nacionales se vaciaran al instante, ¿un pequeño desplazamiento se extinguiría o crecería? La respuesta es una clasificación (`stable`, `unstable`, `nonhyperbolic` o `degenerate`) de los autovalores de un jacobiano reducido, junto con las comprobaciones que la justifican. Es evidencia sobre ese único proceso de ajuste. Nunca es un resultado de unicidad, y cada `StabilityResult` lleva el texto de calificación del proyecto IO en `qualification`.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.stability import reduced_stability

# Two countries, one sector, no intermediates, no taxes, balanced trade.
# Rows: goods A, goods B, taxes, labour, capital; columns: 2 intermediate, 2 final uses per country.
y0, alpha = np.array([100., 150.]), np.array([.3, .4])
F = np.array([[60., 40.], [40., 110.]])          # origin x destination final deliveries
data = np.zeros((5, 6))
data[3, :2], data[4, :2] = (1-alpha)*y0, alpha*y0
for c in range(2):
    data[:2, 2+2*c:4+2*c] = F[:, c][:, None]*np.array([.7, .3])[None, :]
calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-12)

report = reduced_stability(calib, base)
print(report.summary())
print(report.to_dataframe())
print(report.maximum_real_eigenvalue)         # 0.2222... = 2/9, the closed-form unstable root
print(report.qualification)
```

El jacobiano reducido de este ejemplo tiene forma cerrada (el archivo de pruebas la deriva) y su raíz inestable es exactamente `2/9`. La inestabilidad es económica, no numérica: con cestas físicas fijas la demanda del bien de un país es inelástica al precio, de modo que subir sus precios de factores eleva su ingreso y su exceso de demanda de factores.

## Qué se calcula

Escriba los residuos físicos del modelo como `r(x)` en la disposición de incógnitas de puremacro `x = [log p; log y; log r; log w; T; XN]` (`n = ns*nc` celdas, `nc` países). Las incógnitas se parten en un bloque lento `z = (log r, log w)` y un bloque rápido `u = (log p, log y, T)`; los saldos exteriores `XN` se mantienen fijos. Las filas se parten en las filas de factores `f` (brechas de trabajo y capital) y las filas rápidas `g` (vaciado de bienes, beneficio nulo, presupuestos fiscales). Las filas de cuenta corriente no se imponen al bloque rápido: certifican el estado convergido y nada más.

En cada vector de precios de factores se vacía el bloque rápido, `g(u(z), z) = 0`, y los cocientes de exceso de demanda de factores son `e(z) = -f(u(z), z) / endowment` (dotación; positivo significa demanda por encima de la dotación). Su jacobiano es el complemento de Schur

```
J_F = f_z - f_u g_u^{-1} g_z,        A = -J_F / endowment
```

con filas ordenadas (capital, trabajo) para coincidir con las columnas (log r, log w); `factor_labels` da el orden `r[c0], ..., w[c0], ...`. `A` se construye a partir de un jacobiano por diferencias finitas centradas de `puremacro.trade._accounting.evaluate` (resultados consistentes) o de `puremacro.trade.equilibrium.evaluate_equilibrium_residuals` (resultados legacy). El paso complejo no está disponible porque el desempaquetador del estado convierte a float. Los pasos de las incógnitas en niveles (`T`, y `XN` en la partición ingenua descrita abajo) se escalan por la renta nacional, porque esas incógnitas entran linealmente en unidades de calibración y un paso sin escalar ahogaría su efecto en el error de redondeo.

Los saldos exteriores se redenominan para que la aplicación sea homogénea de grado cero en todos los precios de factores, `A 1 = 0`. El valor por defecto `balance_units="world_factor_income"` los fija en unidades de `sum_k (w_k L_k + r_k K_k)`, que es el numerario del código de estabilidad reducida del proyecto IO. Bajo la contabilidad consistente la identidad

```
p.goods + (1 - tax) y.(pp - p) + w.labour_gap + r.capital_gap + 1.budget = 0
```

se cumple en todo `x`, así que sobre la variedad de vaciado rápido el exceso de demanda de factores ponderado por valor `W.e` se anula (`W = (r K, w L)`), y al derivarlo se obtiene `W A + W*e = 0`, que en un equilibrio es `(r K, w L) A = 0`. El código IO restaura una fila presupuestaria eliminada a partir de esa misma identidad; puremacro evalúa todas las filas presupuestarias directamente y reporta la identidad como comprobación (`metadata['world_identity_error']`).

La ley relativa `d log(f_i / f_ref) / dt = e_i - e_ref` para cada precio de factor distinto de la referencia (por defecto: el salario del país 0) da la matriz de dimensión `(2 nc - 1)` `R = A[keep, keep] - A[ref, keep]`. Como `A 1 = 0`, toda referencia produce una `R` semejante, así que el espectro no depende de la referencia. Las velocidades opcionales `speeds` por país y factor sustituyen `A` por `diag(speeds) A` en la ley, de modo que el espectro de `R` es entonces el de `diag(speeds) A` sin su cero.

La comprobación densa opcional (`dense=True`, por defecto) recalcula `A` de forma independiente: resuelve de nuevo el bloque rápido en precios de factores perturbados mediante iteración de cuerda (como máximo 30 iteraciones, deteniéndose cuando las filas de bienes quedan por debajo de `1e-12 max(1, max y)`, las de beneficio nulo por debajo de `1e-12 max(1, max p)` y las presupuestarias por debajo de `1e-12 max(1, max renta)`) y diferencia los cocientes de exceso de demanda resultantes. Esto difiere del código IO, que elimina las incógnitas rápidas de un jacobiano por paso complejo del sistema completo: la nueva resolución comprueba la propia aplicación no lineal de vaciado rápido en lugar de repetir la misma eliminación lineal.

Clasificación, con banda de signo `tol` y umbral de rango `rank_tol` (ambos 1e-7 por defecto):

| resultado | condición |
|---|---|
| `degenerate` | `R` es numéricamente deficiente de rango: valores singulares iguales o inferiores a `rank_tol * max(1, mayor)` (un espacio nulo estructural o un pliegue) |
| `unstable` | `max Re eig(R) > tol` |
| `stable` | `max Re eig(R) < -tol` |
| `nonhyperbolic` | en otro caso |

La partición ingenua que mantiene las filas de cuenta corriente en el bloque rápido junto con `XN` (`current_account="fast"`) es estructuralmente degenerada: su matriz reducida tiene rango `nc`, sean cuales sean las unidades de la tabla. El diagnóstico lo reporta en lugar de un signo.

## Cierres y campos del resultado

| argumento | significado |
|---|---|
| `balance_units` | `"world_factor_income"` (por defecto, numerario IO, espectro invariante a la referencia), `"reference_factor_price"` (la convención del prototipo de evaluación; homogénea, pero el espectro se mueve con la referencia siempre que los saldos no sean nulos, en 3.5e-2 en los fixtures sintéticos), `"numeraire"` (el cierre propio del modelo, saldos fijos en unidades del numerario; no homogéneo, se reporta con un aviso). No se aplica con `current_account="fast"` |
| `reference`, `reference_factor` | país y factor (`"w"` o `"r"`) del precio de referencia |
| `current_account` | `"certificate"` (por defecto) o `"fast"` (ingenua, degenerada) |
| `tau`, `tau_fd`, `tauf`, `tauf_fd` | esquemas arancelarios; los resultados consistentes llevan los suyos en `metadata` y los valores explícitos deben coincidir, los resultados legacy no llevan ninguno y los necesitan |
| `speeds` | velocidades de ajuste positivas opcionales, longitud `2 nc` |
| `step`, `fd_order` | paso de diferencias finitas para incógnitas logarítmicas, como máximo 1e-2 (los niveles `T` y `XN` se escalan por la renta; entran linealmente), y orden del esquema 2 o 4 |
| `dense`, `dense_step`, `dense_tol` | comprobación independiente resolviendo de nuevo el bloque rápido en precios de factores perturbados (`dense_step` como máximo 1e-2); lanza `StabilityError` si discrepa |
| `tol`, `rank_tol` | banda de signo de la clasificación y umbral relativo de valores singulares para el rango, fijados por separado para que una banda de signo amplia no convierta equilibrios ordinarios en degenerados |
| `homogeneity_tol`, `equilibrium_tol` | umbral para las identidades de homogeneidad y de Walras, y para aceptar el estado como equilibrio de sus propias ecuaciones |

Cuando una identidad es exacta para la aplicación evaluada, su incumplimiento es un error, no una salvedad. Bajo la contabilidad consistente la homogeneidad es exacta (con `balance_units` distinto de `"numeraire"`, o en la partición ingenua) y también lo son las identidades de Walras y mundial; una violación por encima de `homogeneity_tol` (identidad mundial: 1e-10) solo puede deberse a error de diferencias finitas, por ejemplo un paso demasiado pequeño o demasiado grande, y lanza `StabilityError`.

`StabilityResult` expone `classification`, `maximum_real_eigenvalue`, `eigenvalues` (ordenados por parte real), `reduced_jacobian` (`A`), `relative_matrix` (`R`), `rank`, `homogeneity_error` (`max|A 1| / max|A|`), `walras_error` (`max|W A + W*e| / max|W_i A_ij|`, relativo), `fast_block_condition` (tras equilibrar filas y columnas, porque el bloque bruto mezcla incógnitas logarítmicas con niveles de transferencias en unidades de calibración), `equilibrium_residual` y `dense_check_error` (ambos absolutos), los textos `adjustment`, `closure` y `qualification`, `reference`, `factor_labels` y `metadata` (autovectores, valores singulares, espectro absoluto de `A`, la comprobación de la identidad mundial, la partición y todas las tolerancias; sus arrays son de solo lectura). `to_dataframe()` lista los autovalores; `mode_loadings(k)` da el autovector por la derecha del modo `k` sobre las etiquetas de precios de factores (cero en la referencia); `to_markdown()`, `to_latex()` y `to_typst()` renderizan la tabla.

## Resultados legacy

El diagnóstico funciona con resultados legacy, pero la contabilidad legacy no es homogénea de grado uno: los impuestos a la producción gravan cantidades y la fórmula de coste por defecto `replicate_matlab_precedence=True` es homogénea de grado `1 + alpha`. La aplicación reducida no es entonces homogénea de grado cero, la reducción relativa queda condicionada al precio de referencia, y `reduced_stability` emite un `RuntimeWarning`, fija `metadata['homogeneous'] = False` y añade la salvedad a `closure`, de modo que `summary()` la muestra junto al veredicto; `metadata['absolute_eigenvalues']` guarda el espectro de la aplicación completa. Con impuestos a la producción nulos, impuestos a usos finales nulos y `replicate_matlab_precedence=False`, los resultados legacy y consistentes dan el mismo jacobiano reducido hasta 1.3e-10.

## Qué establecen las pruebas y qué no

Verificado en `tests/test_trade_stability.py` (tolerancias indicadas allí; valores observados entre paréntesis):

- jacobiano reducido en forma cerrada del ejemplo de dos países para las tres denominaciones de saldos y ambas referencias, atol 1e-8 (2.3e-10 con el paso de segundo orden por defecto 1e-6; 1.8e-13 con `fd_order=4`, `step=1e-3`);
- complemento de Schur frente a una resolución independiente del bloque rápido con `fsolve` en el fixture de dos países, 1e-8;
- la comprobación de cuerda incorporada en todos los fixtures, 1e-8 (1.1e-9 a 2.8e-9);
- homogeneidad e identidad de Walras ponderada por valor, 1e-8 (por debajo de 2e-9), también en un punto fuera del equilibrio de la variedad de vaciado rápido donde `W A` por sí solo no es cero;
- invarianza del espectro a la referencia bajo unidades de renta mundial de factores (1.8e-10) y su dependencia documentada bajo unidades del precio de factor de referencia;
- velocidades de ajuste: 8 sorteos aleatorios en [1/4, 4] en el fixture de tres países, donde el espectro relativo coincide con el de `diag(speeds) A` sin su cero (1e-8; observado como máximo 2.2e-10) y la clasificación sigue siendo `unstable`;
- las cargas de los modos son autovectores por la derecha de la ley módulo el vector constante (1e-8; observado 5e-16);
- partición ingenua degenerada (rango `nc`) en ambos fixtures sintéticos, en la tabla de tres países con flujos multiplicados por 1e6 y en el fixture OCDE 3x3 en millones de USD; ese reescalado no cambia la propia `A` (1e-8);
- la banda de signo y el umbral de rango actúan por separado;
- continuidad de los autovalores ante cambios arancelarios de 1e-3 y 1e-2;
- tanteo de Euler sobre la aplicación no lineal de vaciado rápido: la norma del exceso de demanda cae a lo largo del autovector estable y crece a lo largo del inestable;
- acuerdo legacy frente a consistente cuando las contabilidades coinciden (1.3e-10) y la discrepancia por precedencia MATLAB (0.19);
- el fixture OCDE 3x3, el modelo CES hicksiano de dos países y una ejecución legacy 77x11 marcada como lenta (unos 50 segundos para el jacobiano por diferencias finitas).

Observaciones registradas por esas pruebas, todas dependientes del cierre: los dos fixtures sintéticos de puremacro se clasifican `unstable` con coeficientes de abastecimiento fijos (max Re +0.245 y +0.239); el mayor autovalor real cae monótonamente con la elasticidad de sustitución de intermedios y se vuelve negativo en `sigma = 1` (2c2s: +0.245, +0.114, -0.016, -0.278 en sigma 0, 0.5, 1, 2); el fixture OCDE 3x3 hace lo mismo bajo la contabilidad consistente (+0.145, +0.050, -0.022 en sigma 0, 0.5, 1); es `stable` bajo la contabilidad legacy con un arancel del 10 por ciento de EE. UU. (-0.115, error de homogeneidad 0.6), así que la misma tabla produce veredictos distintos bajo contabilidades distintas.

No establecido, citando la documentación IO: "A local stability result is conditional on the specified relative-wage adjustment process, not a uniqueness theorem" (README) y "Numerical verification does not establish empirical identification, global uniqueness, or stability under every possible adjustment process" (VALIDATION.md). La clasificación nunca es un certificado de unicidad, no dice nada sobre cuencas de atracción y no estima cómo se ajustan realmente los precios de los factores. Solo se admite el cierre fiscal de suma fija sin márgenes de capacidad ni reciclaje de ingresos. El jacobiano por diferencias finitas cuesta `2 (2 n + 3 nc)` evaluaciones del residuo, lo que es asumible al tamaño 77x11 incluido e impracticable a tamaños nativos de 45 sectores sin un jacobiano analítico.

## Relacionado

- [Contabilidad comercial consistente](trade_accounting.md) para la disposición de residuos que el diagnóstico diferencia.
- [Bienestar hicksiano del consumo](trade_welfare.md) y [Política arancelaria hicksiana](trade_policy.md) para los equilibrios a los que se aplica este diagnóstico.
- [Estado de validación estructural](STRUCTURAL_VALIDATION_STATUS.md) para la fila de validación.
