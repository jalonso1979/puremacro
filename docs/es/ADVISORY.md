> 🇬🇧 [English](../ADVISORY.md) · 🇪🇸 Español

# Avisos de corrección

Se emite un aviso de corrección cuando una versión publicada de
puremacro devolvió un **número equivocado** — no un fallo, no una función
ausente, sino una respuesta bien formada y falsa. La distinción importa
porque un fallo se denuncia solo y un número equivocado no: acaba en un
cuadro, en una figura, en un dictamen.

Cada aviso nombra las versiones afectadas, la condición exacta bajo la
cual el error **se anula** (para que pueda descartar su propia estimación
sin volver a correrla) y qué hacer.

---

## 2026-09-03 — doce más, versiones hasta 1.9.0 inclusive

**Corregido en 1.10.0.** Una auditoría posterior
preguntó, de cada fixture de estimador del paquete, *¿qué condición elimina este
fixture?* — la pregunta sobre la que giraban los siete defectos de arriba.
Encontró doce más, cada uno reproducido contra una verdad conocida antes de
tocar nada.

| Estimador | Qué estaba mal | No afectado cuando | Dirección del error |
|---|---|---|---|
| `cointegration_modern.dols` | Omitía el término contemporáneo `dX_t`, que es el que elimina el sesgo. El estimador era MCO con regresores extra. | `dX_t` no está correlacionado con `u_t` | El sesgo seguía al de MCO en todos los tamaños: **+0.036 frente a +0.030** en T=100, donde el DOLS correcto da −0.0005 |
| `cointegration_modern.dols` | Conservaba una fila cuyo rezago de `dX` faltante se **rellenaba con ceros**, fabricando un valor de regresor | `lags = 0` | Una observación de más, construida con un valor nunca medido |
| `cointegration_modern.fm_ols` | Construía la corrección de Phillips–Hansen sólo con `Lambda`, omitiendo el bloque `Sigma` de rezago 0 | `Omega` es proporcional a `Sigma` | Sobre-corregía. **ECM peor que MCO** en T=200, 800 y 3200 |
| `var.irf.gfevd` → `connectedness.spillover_index` | Dividía por un piso absoluto de `1e-12`, rompiendo una invarianza de escala que el estimando tiene por construcción | Ninguna varianza residual cae cerca del piso | La conectividad total pasó de **13.43 a 39.48** en un mismo VAR al reescalar una variable |
| `dsge.gensys` | `Impact` omitía el término `Pi N`: resolvía como si los errores de expectativas no respondieran a los choques | `Pi N = 0` | En `y_t = a E_t y_{t+1} + eps_t` (verdad `y_t = eps_t`) devolvía **`[0, -2]`**: la variable no respondía a su propio choque |
| `dsge.gensys` | Lanzaba excepción con cualquier modelo **sin raíces inestables**; la prueba de unicidad era vacua | El modelo tiene al menos una raíz inestable | `x_t = rho x_{t-1} + eps` no podía resolverse |
| `dsge.fertility_adj_costs` `irf`/`fevd` | Avanzaba el estado antes de aplicar `F`, pero los controles leen el estado *rezagado* | Sólo el horizonte 0 | Toda IRF de control **un periodo adelantada**: el `y(h)` reportado era el `y(h+1)` correcto |
| `did.sun_abraham` | `lo`/`hi` promediaban los extremos por cohorte en vez de usar el `se` agregado | Una sola cohorte por tiempo de evento | Bandas **`sqrt(K)` veces más anchas** — 1.73 con K=3, 1.37 con K=2 — e inconsistentes con el `se` de su propia fila |
| `inference.weak_iv.kleibergen_paap_f` | El sándwich HC0 dividía entre `n` dos veces mientras el pan invertía el producto cruzado crudo `Z'Z`, que ya lleva ambos factores | Nunca — toda llamada estaba afectada | Devolvía **exactamente `n^2` veces** el estadístico correcto: 235,470 frente a una verdad de 5.89 con n=200. Los umbrales de Stock-Yogo rondan 10, así que reportaba "instrumentos fortísimos" para todo conjunto de datos y **jamás podía diagnosticar un instrumento débil** |
| `inference.weak_iv.kleibergen_paap_f` | La función de influencia usaba `kron(Z_t, V_t)` donde el `vec` por columnas y el pan exigen `kron(V_t, Z_t)` | `k = 1` (un regresor endógeno) — ahí coinciden | Varianza robusta incorrecta con dos o más regresores endógenos |
| `inference.weak_iv.anderson_rubin_band` | Comparaba un estadístico F contra `chi2.ppf(ci, df)`, pero es `df*F` lo que es `chi2(df)` | `df = 1` — el valor por defecto y el único ejercitado | Corte `df` veces mayor. La banda nominal del 90% tenía **100.0% de cobertura** con df=2 y df=4 |
| `lp.panel_lp_dk` (vía `_focal_dk_se`) | El ancho de banda de Bartlett salía del número de filas `N*T`, pero el núcleo corre sobre la suma transversal de longitud `T` | `N = 1` | Ancho inflado por `N^(2/9)`: **7 -> 10 -> 13** con T=100 fijo al pasar N de 10 a 50 a 200. Agregar países cambiaba la autocorrelación supuesta |
| `lp.mean_group_panel_lp` | No ordenaba su entrada, mientras `lp_hac` construye rezagos con `.shift()` posicional | Quien llama ya pasaba un marco ordenado | En un panel cuya respuesta verdadera en h=1 es 0.8: **0.765 ordenado, 0.055 con las mismas filas barajadas**. Silencioso, y atenuado hacia cero |
| 25 sitios de valor-p en 16 módulos | `1 - cdf` en vez de la función de supervivencia | El valor-p supera ~1e-16 | Devolvían **exactamente 0.0** en la cola. Normal bilateral en \|z\| = 9 es 2.3e-19; chi-cuadrada 200 con 5 gl es 2.8e-41 |

### Qué hay que volver a correr

- **Cualquier estimación de `dols` o `fm_ols`.** Ambas estaban sesgadas
  sistemáticamente sobre los datos para los que existen. `fm_ols` quedaba más
  lejos de la verdad que MCO.
- **Cualquier índice de conectividad de Diebold–Yilmaz** calculado con el
  `identification="gfevd"` por defecto sobre datos cuyas varianzas residuales
  sean pequeñas en las unidades empleadas.
- **Cualquier IRF de `gensys`.** La matriz de impacto era incorrecta para todo
  modelo con comportamiento prospectivo, que son todos los que gensys atiende.
- **Cualquier IRF o FEVD del DSGE de fecundidad**, incluidas las figuras de
  `docs/research/fertility_bk_diagnosis/`. Desplace las sendas de control un
  periodo hacia atrás, o vuelva a correrlas.
- **Cualquier banda de estudio de eventos de Sun–Abraham** en un tiempo de
  evento con más de una cohorte. Las estimaciones puntuales y el `se` se
  sostienen; los intervalos eran demasiado anchos.
- **Cualquier valor-p reportado como exactamente 0.0.** Nunca fue cero.

### Una nota sobre las pruebas

Cuatro artefactos distintos afirmaban el defecto de `gensys` en vez de
detectarlo: tres pruebas en `tests/test_dsge_gensys_coverage.py` y el caso de
validación `dsge.gensys_shock_impact_identity`, cuya propia cita enunciaba el
modelo **sin** su término `Pi eta_t` — álgebra correcta a partir de una premisa
truncada. La banda de `sun_abraham` contradecía el error estándar impreso a su
lado en la misma fila, así que ésa no requería referencia externa alguna. Y el
docstring de `FertilitySolution` describía `F` actuando sobre el estado actual
mientras el solver lo construye contra el rezagado; el bucle de IRF se escribió
siguiendo el docstring.

---

## 2026-09-02 — siete estimadores, versiones 0.92.0 a 1.8.0

**Corregido en 1.9.0.** Siete estimadores públicos devolvieron números
equivocados en todas las versiones desde 0.92.0 hasta 1.8.0 inclusive.
Los siete fallos comparten una forma: la respuesta equivocada era
*internamente consistente*, de modo que ninguna invariante que el paquete
verificaba podía detectarla, y en todos los casos el fixture de prueba
cumplía exactamente la condición bajo la cual el error desaparece.

### ¿Le afecta?

```python
import puremacro
puremacro.__version__          # < "1.9.0" → lea el cuadro
```

Si **publicó** un número de alguno de los siete estimadores de abajo con
una versión anterior a 1.9.0, vuelva a correrlo. Si sólo lo corrió sobre
datos que cumplen la columna "no afectado cuando", no hay nada que hacer.

| Estimador | Qué estaba mal | No afectado cuando | Dirección del error |
|---|---|---|---|
| `var.identify.proxy_svar` | El vector de impacto se devolvía en la métrica de `Sigma` y no en la de `Sigma^-1` — proporcional a `Sigma b_1`, no a `b_1`. El choque identificado era una mezcla de todos los choques estructurales. | `Sigma` es proporcional a la identidad (residuos i.i.d.) — ahí el error es exactamente cero | Crece con la estructura fuera de la diagonal de `Sigma`. 31% en un elemento de un DGP de 3 variables, con el patrón de signos relativos equivocado |
| `var.panel` | Importa el mismo `_proxy_impact_factory` | ídem | ídem |
| `inference.swamy_test` | La forma cuadrática se centraba en la media aritmética y no en `beta_bar_W`, la ponderada por precisión | Todas las unidades se estiman con la **misma** precisión | Sobre-rechaza la homogeneidad de pendientes, nunca sub-rechaza. Tamaño al 5% nominal: 0.050 con precisión igual, 0.078 con dispersión 2×, **0.975** con 0.1 frente a 3.0 |
| `garch.dcc_fit` | Los rendimientos crudos se estandarizaban con una volatilidad ajustada sobre los centrados, de modo que `Qbar` estimaba `mu_i · mu_j` | `mean="zero"` — **el valor por defecto**, e idéntico bit a bit al anterior sobre datos ya centrados. Sólo `mean="constant"` está afectado | Correlaciones atraídas hacia `m²/(m²+1)`, con `m = mu/sd`. Un valor verdadero de 0 se reportaba como **+0.94** con media 5 y desviación 1 |
| `state_space.simulation_smoother` | Los interceptos del modelo quedaban en el segundo paso de Durbin–Koopman, sumando `b` una segunda vez | `c` y `d` son ambos cero — todos los fixtures de la suite, e idéntico bit a bit ahí | Cada draw desplazado por el intercepto completo. Con `d = 5` los draws quedaban exactamente −5.0 de la media posterior, contra un error Monte Carlo de 0.012 |
| `var.wild_bootstrap_var` | Los draws fallidos se escribían en la pila de percentiles como la estimación puntual, sin contador ni advertencia | Ningún draw falló | Bandas demasiado **estrechas**, monótonamente en la fracción de fallos `f`; ancho cero una vez que `f ≥ 1−2a`. Peor precisamente cuando el `impact_fn` de `proxy_svar` falla por instrumento débil — la banda se estrechaba cuando debía ensancharse |
| `var.identify.rigobon_svar` | El bootstrap emparejaba bloques de residuos remuestreados con las etiquetas de régimen en orden de calendario, destruyendo la identificación en cada draw | Sólo la estimación puntual; lo que estaba mal es la **banda** | Bandas unas **8 veces más anchas** en un DGP con razón de varianzas verdadera 3.0 (los draws promediaban 1.14 y nunca superaron 1.50 en 500) |
| `var.estimate_var` | Aceptaba datos no finitos y devolvía coeficientes todo-NaN sin lanzar excepción | La entrada es finita | No es un número equivocado sino uno silencioso: `Sigma`, los residuos y toda IRF, FEVD, descomposición histórica y banda construidas sobre el ajuste salían todo-NaN y perfectamente bien formadas. Ahora es un `LinAlgError` con nombre |

Las derivaciones completas, las magnitudes medidas y por qué cada fixture
no podía alcanzar su propio error están en
[`CHANGELOG.md`](https://github.com/jalonso1979/puremacro/blob/main/CHANGELOG.md),
bajo 1.9.0, "Fixed — affects results published in every release from
0.92.0 to 1.8.0".

### Qué hay que volver a correr

- **Cualquier respuesta a impulso proxy-SVAR publicada** de `proxy_svar`
  o `var.panel`. Cambian la estimación puntual y la banda.
- **Cualquier banda de Rigobon.** La estimación puntual se sostiene; la
  banda no.
- **Cualquier rechazo de `swamy_test` en un panel con precisión desigual
  entre unidades** — muestras cortas mezcladas con largas, países
  pequeños con grandes. Ése es el caso normal, no el exótico.
- **Cualquier correlación de `dcc_fit(mean="constant")`.** La ruta por
  defecto `mean="zero"` no requiere nada.
- **Cualquier draw de `simulation_smoother` de un modelo con deriva de
  estado o intercepto de medición distintos de cero.**
- **Cualquier banda de `wild_bootstrap_var` cuya corrida haya reportado
  fallos de bootstrap** — cosa que, antes de 1.9.0, no reportaba. Si el
  estimador era `proxy_svar` con instrumento débil, suponga que la banda
  quedó demasiado estrecha.

### Hasta dónde llegó la corrección en 1.9.0

Se dice aquí en vez de dejar que un usuario lo descubra:

- **`matlab/` — en parte.** La caja de herramientas de MATLAB es una
  implementación aparte, así que una corrección en Python no la alcanza.
  Dos se portaron a mano: `+puremacro/+var/proxy.m` tenía el mismo error
  de métrica en el proxy-SVAR y está corregido, y
  `+puremacro/+var/estimate.m` ahora lanza excepción ante entradas no
  finitas. **Toda respuesta a impulso proxy-SVAR que esa caja produjo
  antes del 2026-09-02 es incorrecta y debe volver a correrse.** Los
  otros cinco estimadores del cuadro **no** han sido auditados ahí;
  donde la caja los implemente, suponga los mismos defectos hasta
  comprobarlo. Vea
  [`matlab/README.md`](https://github.com/jalonso1979/puremacro/blob/main/matlab/README.md).
- **Los cuadernos de este repositorio ya se volvieron a ejecutar.**
  `notebooks/14_tax_multiplier_three_ways`,
  `notebooks/17_identification_spec_curve`, sus gemelos `_es`,
  `notebooks/course/06_lp_narrativa_es` y la compilación de
  `playground/` muestran números posteriores a la corrección desde
  1.9.0. `notebooks/08_garch_volatility` no requirió cambio: llama a
  `dcc_fit(panel)`, y la ruta por defecto `mean="zero"` es idéntica bit
  a bit.
- **Los suyos no.** Un `.ipynb` versionado guarda las salidas de la
  corrida que lo produjo. Toda celda suya que muestre un resultado de un
  estimador de arriba, ejecutada antes de 1.9.0, sigue mostrando el
  número previo a la corrección hasta que la vuelva a ejecutar.

---

## Cómo se decide un aviso

Se emite un aviso cuando se cumplen **todas** estas condiciones:

1. Una versión publicada devolvió un resultado numéricamente equivocado
   de un estimador público, o un resultado sin la cobertura que declaraba.
2. El fallo fue silencioso — sin excepción, sin advertencia, sin una
   salida visiblemente mal formada.
3. Un usuario pudo plausiblemente haber publicado ese número.

Un error que lanza excepción, uno en una ruta no publicada y uno en un
auxiliar privado sin consecuencia pública son entradas del CHANGELOG, no
avisos.

La regla que se sigue es la de
[`CONTRIBUTING.md`](https://github.com/jalonso1979/puremacro/blob/main/CONTRIBUTING.md):
el paquete no sustituye un valor faltante por uno plausible, y no se
queda callado sobre un número que calculó mal.
