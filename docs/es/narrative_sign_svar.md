> 🇬🇧 [English](../narrative_sign_svar.md) · 🇪🇸 Español

# Restricciones narrativas de signo en modelos SVAR

El módulo `puremacro.var.identify.narrative_sign` implementa la metodología de identificación estructural para vectores autorregresivos (SVAR) desarrollada por **Antolín-Díaz y Rubio-Ramírez (2018, *American Economic Review*)**, junto con las extensiones de cotas de magnitud de **Ludvigson, Ma y Ng (2021, *Journal of Political Economy*)**.

En los modelos SVAR identificados mediante restricciones de signo convencionales (Faust 1998; Uhlig 2005; Rubio-Ramírez, Waggoner y Zha 2010), las regiones identificadas por conjuntos (*set identification*) suelen ser excesivamente amplias, arrojando intervalos de credibilidad que abarcan respuestas cualitativamente contradictorias. Las restricciones narrativas resuelven esta limitación al condicionar las matrices de rotación estructural sobre evidencia histórica concreta —tales como anuncios documentados de la banca central, perturbaciones geopolíticas en el mercado petrolero o episodios específicos de crisis financiera—, descartando aquellas extracciones ortogonales que contradicen el registro histórico establecido.

---

## 1. Fundamentos metodológicos

### 1.1 La forma reducida del VAR y las rotaciones estructurales

Considérese un modelo $\text{VAR}(p)$ en forma reducida con $n$ variables endógenas:

$$y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma)$$

donde $\Sigma$ es la matriz simétrica y definida positiva de covarianzas residuales. Sea $P = \text{chol}(\Sigma)$ el factor triangular inferior de Cholesky tal que $P P' = \Sigma$.

Las innovaciones de forma reducida $u_t$ se vinculan con las perturbaciones estructurales ortonormales $\varepsilon_t \sim \mathcal{N}(0, I_n)$ mediante:

$$u_t = B \varepsilon_t = P Q \varepsilon_t$$

donde $Q \in \mathcal{O}(n)$ es una matriz ortogonal de rotación que satisface $Q Q' = Q' Q = I_n$. Las respuestas al impulso estructurales en el horizonte $h$ se obtienen como:

$$\Psi_h = \Phi_h B = \Phi_h P Q$$

siendo $\Phi_h$ la matriz de coeficientes de media móvil de la forma reducida en el horizonte $h$ ($\Phi_0 = I_n$).

### 1.2 Tipología de restricciones narrativas

Antolín-Díaz y Rubio-Ramírez (2018) formalizan las restricciones narrativas sobre las realizaciones de las perturbaciones estructurales $\varepsilon_t = B^{-1} u_t = Q' P^{-1} u_t$ y su descomposición histórica en cuatro categorías fundamentales:

1. **Tipo I: Restricción sobre el signo de la perturbación (`'shock_sign'`)**  
   Fija el signo de la perturbación estructural $j$ en una fecha histórica precisa $t^*$:
   $$\text{sign}(\varepsilon_{j, t^*}) = s, \quad s \in \{-1, +1\}$$
   *Ejemplo*: «La perturbación de política monetaria en el cuarto trimestre de 1979 (endurecimiento de Paul Volcker) fue estrictamente positiva (contractiva)». El signo es un campo obligatorio de una restricción de Tipo I (las tuplas `(fecha, choque, signo)` son la forma abreviada).

2. **Tipo II: Dominancia en la descomposición histórica (`'hd_dominance'`, `dominance='most'`)**  
   Exige que la perturbación estructural $j$ sea la principal contribuyente individual al cambio inesperado acumulado en la variable $i$ a lo largo de una ventana de $L$ períodos finalizada en $t_1^*$:
   $$|H_{i, j}(t_1^*, L)| \ge \max_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   donde la contribución histórica de la perturbación $k$ se define como:
   $$H_{i, k}(t_1^*, L) = \sum_{l=0}^L (\Phi_l B)_{i, k} \, \varepsilon_{k, t_1^* - l}$$

3. **Tipo III: Dominancia abrumadora en la descomposición histórica (`'hd_dominance'`, `dominance='overwhelming'`)**  
   Exige que la contribución absoluta de la perturbación $j$ supere la suma de las contribuciones de todas las demás perturbaciones estructurales combinadas:
   $$|H_{i, j}(t_1^*, L)| \ge \sum_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   *Ejemplo*: «La perturbación monetaria fue la causa abrumadora del incremento imprevisto de la tasa de fondos federales en 1979Q4».

4. **Tipo IV: Cotas de magnitud de la perturbación (`'shock_bound'`)**  
   Impone desigualdades sobre la magnitud absoluta de la perturbación estructural conforme a Ludvigson, Ma y Ng (2021):
   $$\underline{m} \le |\varepsilon_{j, t^*}| \le \bar{m}$$
   *Ejemplo*: «La perturbación monetaria durante el cambio de régimen de Volcker tuvo una magnitud de al menos 2 desviaciones estándar ($|\varepsilon_{j, t^*}| \ge 2.0$)». La cota es **sin signo por defecto** (`sign=None`); indique `sign=+1` o `sign=-1` para restringir además el signo de la perturbación.

### 1.3 Fechas de las restricciones

`date` puede ser un **entero** o una **fecha de calendario**:

- Un entero es siempre el **índice de fila (base 0) en `Y`** (debe ser $\ge p$ para que exista un residuo), se pase o no `dates`.
- Un `pd.Timestamp`, una cadena ISO (`"1979-10-06"`), un `datetime.date` o un `np.datetime64` se localiza en `dates`. Un `DataFrame` indexado por un `DatetimeIndex` o `PeriodIndex` proporciona `dates` automáticamente (un `dates=` explícito tiene prioridad). Si hay una coincidencia exacta con una marca temporal, se usa; en caso contrario la fecha se asigna a la observación cuyo **período la contiene**, con la longitud del período inferida del espaciado del índice: mismo trimestre en un índice trimestral (de modo que el anuncio del 1979-10-06 cae en 1979Q4 tanto si el índice está marcado en 1979-10-01 como en 1979-12-31), mismo mes en un índice mensual, coincidencia exacta en índices más finos. Una fecha cuyo período no figura en el índice produce un `ValueError` en lugar de reasignarse a un período vecino.

### 1.4 Muestreo y ponderación por importancia (Algoritmo 1 de AD-RR)

1. **Generación de rotaciones de Haar**: Se muestrean matrices aleatorias gaussianas $Z \sim \mathcal{N}(0, I_n)$ y se calcula $Q$ mediante descomposición QR con normalización de signos positivos en la diagonal ($R_{ii} > 0$). Con `bayes_draws=True` se extrae un nuevo $(A, c, \Sigma)$ de la distribución a posteriori conjugada Normal-Inversa-Wishart para cada rotación (se vuelve a muestrear hasta obtener un VAR estable, con un máximo de 50 intentos; una extracción sin candidato estable se omite y se contabiliza en `n_unstable_draws`).
2. **Filtrado por restricciones de signo tradicionales**: Se verifica si las respuestas al impulso $\Psi_h = \Phi_h P Q$ satisfacen las restricciones de signo especificadas en `sign_matrix`.
3. **Validación narrativa**: Se calculan las perturbaciones estructurales históricas $\varepsilon = u (B^{-1})'$ y se comprueba el cumplimiento simultáneo de todas las restricciones narrativas en sus fechas correspondientes.
4. **Ponderación por importancia**: A fin de corregir el sesgo que favorecería a parámetros bajo los cuales los eventos narrativos fuesen fortuitamente probables, cada extracción superviviente se pondera por:
   $$w = \frac{1}{\omega}$$
   donde $\omega$ representa la probabilidad de que las restricciones narrativas se verifiquen cuando las perturbaciones en las fechas restringidas se extraen de forma independiente e idénticamente distribuida $\mathcal{N}(0, I_n)$:
   - Para restricciones puras de Tipo I (`'shock_sign'`) sobre $m$ pares distintos (fecha, perturbación), $\omega = 0.5^m$ se conoce de forma analítica cerrada ($w = 2^m$).
   - Para restricciones de Tipo II, III o IV, $\omega$ se estima mediante simulación de Monte Carlo con $S = \text{n\_weight\_sims}$ extracciones. Una estimación $\hat\omega = 0$ se acota inferiormente en $1/S$ (el ponderador queda limitado a $S$); el número de extracciones en las que se activa esa cota se reporta como `n_weight_floor`.
5. **Inferencia ponderada**: Las bandas de credibilidad y la mediana se obtienen mediante cuantiles ponderados puntuales sobre las extracciones aceptadas. La matriz de impacto de cada extracción superviviente (y, en modo bayesiano, sus coeficientes autorregresivos) se conserva en el resultado, de modo que `.irf(h)` y `.fevd(h)` para horizontes más allá del $H$ estimado son medianas ponderadas de las extracciones *extendidas*, nunca una única extracción representativa.

---

## 2. Diagnósticos de aceptación y métricas

El objeto `NarrativeSignResult` generado proporciona métricas diagnósticas exhaustivas:

| Atributo | Tipo | Significado analítico |
|---|---|---|
| `n_draws` | `int` | Total de rotaciones ortogonales de Haar evaluadas. |
| `n_traditional_accepted` | `int` | Número de extracciones que satisfacen las restricciones de signo tradicionales. |
| `n_narrative_accepted` | `int` | Extracciones que satisfacen tanto las restricciones tradicionales como las narrativas. |
| `acceptance_rate` | `float` | Tasa de aceptación global (`n_narrative_accepted / n_draws`). |
| `traditional_acceptance_rate`| `float` | Fracción de extracciones válidas tradicionales (`n_traditional_accepted / n_draws`). |
| `narrative_acceptance_rate`  | `float` | Tasa condicional de aceptación narrativa (`n_narrative_accepted / n_traditional_accepted`). |
| `weights` | `np.ndarray` | Ponderadores de importancia en bruto $1/\hat{\omega}$ para las extracciones supervivientes. |
| `ess` / `effective_draws` | `float` | Tamaño muestral efectivo de Kish: $(\sum w_i)^2 / \sum w_i^2$. |
| `n_weight_floor` | `int` | Extracciones supervivientes cuya $\hat\omega$ fue 0 y se acotó en $1/\text{n\_weight\_sims}$ (sus ponderadores están limitados, por lo que `ess` sobreestima la eficiencia). |
| `n_unstable_draws` | `int` | Solo en modo bayesiano: extracciones a posteriori omitidas por no hallarse un VAR estable en 50 intentos. |
| `restriction_labels` | `tuple[str]` | Etiquetas legibles que identifican cada restricción impuesta. |
| `restriction_fail_counts` | `tuple[int]` | Diagnóstico de rigidez: extracciones tradicionales rechazadas por cada restricción individual. |
| `accepted_B`, `accepted_A` | `np.ndarray` | Matrices de impacto $(m, n, n)$ de las $m$ extracciones supervivientes y, en modo bayesiano, sus matrices autorregresivas $(m, p, n, n)$ (`None` en modo MCO). |
| `A_list`, `intercept`, `residuals`, `Sigma`, `B` | — | Objetos de forma reducida y matriz de impacto de la extracción representativa (la más cercana a la mediana); `B B' = Sigma` en ambos modos. |
| `init_y` | `np.ndarray` | Las primeras $p$ observaciones, condición inicial por defecto de la descomposición histórica. |

### Avisos

`identify_narrative_sign` emite un `RuntimeWarning` (nunca degrada en silencio) cuando:

- **sobreviven demasiado pocas extracciones** para resolver las bandas solicitadas — menos de $\max(10, \lceil 2/(1-\text{ci}) \rceil)$ extracciones aceptadas, en cuyo caso las bandas puntuales pueden colapsar sobre la mediana;
- **los ponderadores de importancia están concentrados** — `ess` de Kish por debajo del 10 % de `n_narrative_accepted`, de modo que unas pocas extracciones dominan las bandas ponderadas;
- **se activa la cota inferior de $\omega$** en al menos una extracción superviviente (`n_weight_floor > 0`; aumente `n_weight_sims`);
- **se omitieron extracciones a posteriori inestables** en modo bayesiano (`n_unstable_draws > 0`). Si *todas* las extracciones a posteriori son inestables, se lanza un `RuntimeError`.

`summary()` muestra el modo de forma reducida, el número de extracciones inestables y el número de cotas activadas junto a los recuentos de aceptación.

---

## 3. Aplicación: Choque de política monetaria de Volcker (1979Q4)

En octubre de 1979, la Reserva Federal presidida por Paul Volcker adoptó un nuevo esquema operativo centrado en objetivos cuantitativos de reservas no prestadas, provocando un aumento histórico sin precedentes en la tasa de fondos federales con el propósito de erradicar la persistente inflación estadounidense.

Antolín-Díaz y Rubio-Ramírez (2018) retoman el SVAR monetario identificado por signos de Uhlig (2005) —un VAR **mensual** de seis variables para EE. UU. sobre los datos de Uhlig— y lo afinan con restricciones narrativas sobre el episodio de Volcker, principalmente:

- **Restricción narrativa 1 (Tipo I)**: la perturbación de política monetaria de octubre de 1979 es positiva (contractiva).
- **Restricción narrativa 2 (Tipo III)**: la perturbación monetaria es la contribuyente abrumadora al movimiento inesperado de la tasa de fondos federales en octubre de 1979.

Su resultado principal es que estas restricciones estrechan sustancialmente el conjunto identificado por signos y eliminan las respuestas contraintuitivas que las restricciones de signo tradicionales admiten por sí solas.

El ejemplo ejecutable siguiente aplica **las mismas dos restricciones de Volcker** a un pequeño VAR trimestral **sintético** de 3 variables (TFF, inflación, crecimiento del producto) con un choque de +3.5 desviaciones estándar inyectado en 1979Q4. Ilustra el mecanismo y la API; **no** reproduce las cifras del artículo (los datos de Uhlig no se descargan sin conexión y AD-RR presentan sus FIR gráficamente). `puremacro.examples.narrative_sign_adrr.run_demo()` es una versión más completa del mismo ejercicio sintético, y `tests/test_replication_adrr2018.py` fija una instantánea de regresión de *la propia salida de esa demostración*: es una prueba de regresión, no una comprobación frente a valores publicados.

### Ejemplo ejecutable

```python
import numpy as np
import pandas as pd
from puremacro.var.identify import (
    NarrativeRestriction,
    identify_narrative_sign,
)

# 1. Panel trimestral sintético (TFF, Inflación, Crecimiento del PIB), 1965Q1-2007Q4
rng = np.random.default_rng(42)
T = 172
fechas = pd.date_range("1965-01-01", periods=T, freq="QS")  # marcas de inicio de trimestre
volcker_idx = fechas.get_loc("1979-10-01")                   # fila de 1979Q4

# Simulación de un VAR estacionario calibrado
Y = np.zeros((T, 3))
for t in range(1, T):
    Y[t] = 0.5 * Y[t-1] + rng.standard_normal(3)
# Inyección del episodio de ajuste de Volcker en 1979Q4
Y[volcker_idx, 0] += 3.5

# 2. Restricciones de signo tradicionales en h = 0
# Choque 0 = Política monetaria contractiva (+TFF, -Inflación, -PIB)
sign_matrix = {0: np.array([+1, -1, -1])}

# 3. Restricciones narrativas. Las fechas de calendario se resuelven mediante
#    `dates` (el anuncio del FOMC del 1979-10-06 cae en la observación de
#    1979Q4); un entero como `volcker_idx` se leería como índice de fila.
restricciones = [
    # Tipo I: perturbación monetaria positiva en 1979Q4
    ("1979-10-06", 0, +1),
    # Tipo III: choque monetario como causa abrumadora del repunte de la TFF
    NarrativeRestriction(
        kind="hd_dominance",
        date="1979-10-06",
        shock=0,
        variable=0,  # TFF
        window=0,
        dominance="overwhelming",
    ),
]

# 4. Estimación del SVAR con restricciones narrativas
res_narr = identify_narrative_sign(
    Y,
    restricciones,
    p=2,
    horizon=16,
    sign_matrix=sign_matrix,
    dates=fechas,
    n_draws=3000,
    n_weight_sims=300,
    ci=0.68,
    seed=123,
)

# 5. Diagnósticos e informe resumen
print(res_narr.summary())
print(f"Aceptación tradicional: {res_narr.traditional_acceptance_rate:.2%}")
print(f"Aceptación narrativa  : {res_narr.narrative_acceptance_rate:.2%}")
print(f"Tamaño efectivo (ESS) : {res_narr.effective_draws:.1f}")

# 6. Figuras, tablas y objetos derivados
fig = res_narr.plot(shock_idx=0, target_idx=None)   # un panel por variable
tabla_md = res_narr.to_markdown(target_idx=0, shock_idx=0)
tabla_tex = res_narr.to_latex(target_idx=0, shock_idx=0)
fevd_20 = res_narr.fevd(horizon=20)                 # mediana ponderada más allá de H=16
dh_tff = res_narr.historical_decomposition(variable=0)
```

---

## 4. Especificación completa de la API

### `identify_narrative_sign` (alias `narrative_sign_svar`)

```text
identify_narrative_sign(
    Y: np.ndarray | pd.DataFrame | VarEstimateResult,
    restrictions: list | None = None,
    *,
    p: int | None = None,
    lags: int | None = None,          # alias de p
    horizon: int | None = None,       # por defecto 20
    horizons: int | None = None,      # alias de horizon
    sign_matrix: dict | np.ndarray | None = None,
    dates: Sequence[datetime-like] | None = None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int | None = 0,
) -> NarrativeSignResult
```

#### Parámetros:
- `Y`: Matriz temporal $(T, n)$, `DataFrame` o resultado previo `VarEstimateResult`. Un `DataFrame` con `DatetimeIndex`/`PeriodIndex` proporciona `dates` automáticamente y sus nombres de columna pasan a `result.names`.
- `restrictions`: Lista de restricciones narrativas. Admite instancias de `NarrativeRestriction`, tuplas breves `(fecha, indice_choque, signo)` (Tipo I) o eventos `puremacro.narrative.NarrativeEvent` (asignados automáticamente a Tipo I sobre el choque 0, usando la fecha de anuncio y el signo).
- `p` / `lags`: Orden autorregresivo del VAR (inferido automáticamente si `Y` es un `VarEstimateResult`; obligatorio en otro caso). Pasar ambos con valores distintos produce `ValueError`.
- `horizon` / `horizons`: Horizonte de proyección $H$ para las respuestas al impulso (por defecto `20`). Pasar ambos con valores distintos produce `ValueError`.
- `sign_matrix`: Diccionario `{h: S}` con matrices de signo tradicionales $S \in \{-1, 0, 1\}$ de forma $(n, n)$, o $(n,)$ aplicado a la columna del choque 0; un array suelto equivale a `{0: S}`. `None` (por defecto) no impone restricciones de signo tradicionales.
- `dates`: Índices o etiquetas temporales de longitud $T$ para resolver fechas de calendario (véase §1.3). Tiene prioridad sobre el índice del `DataFrame`.
- `bayes_draws`: Si es `True`, muestrea coeficientes $(A, \Sigma)$ de la distribución a posteriori Normal-Inversa-Wishart para cada extracción (véase §1.4).
- `n_draws`: Cantidad total de rotaciones de Haar candidatas (por defecto `2000`, debe ser $\ge 1$).
- `n_weight_sims`: Muestras de Monte Carlo empleadas en el cómputo de $\omega$ para restricciones de Tipo II, III o IV (por defecto `500`, debe ser $\ge 1$).
- `ci`: Nivel de cobertura para las bandas de credibilidad (por defecto `0.90`); debe estar estrictamente entre 0 y 1.
- `seed`: Semilla del generador de rotaciones y del simulador de ponderadores (por defecto `0`; `None` usa entropía nueva).

Los argumentos con nombre desconocidos producen `TypeError`: una errata como `n_draw=` nunca se ejecuta en silencio con el valor por defecto.

---

## 5. Interfaz de resultados y capacidades analíticas

La clase `NarrativeSignResult` implementa las herramientas del protocolo de presentación de `puremacro`:

- `.plot(*, target_idx=0, shock_idx=0, title="", ylabel="Response", scale=1.0, ax=None)`: con índices enteros, un único panel (mediana de la respuesta y banda de credibilidad sombreada) para ese par (variable, choque), opcionalmente dibujado sobre `ax`. `target_idx=None` dibuja un panel por variable de respuesta para el choque dado; `shock_idx=None` un panel por choque; ambos `None` la rejilla completa $n \times n$. Devuelve la `Figure` de matplotlib.
- `.summary()`: Informe en texto plano con el modo de forma reducida, las tasas de aceptación, el tamaño muestral efectivo, los recuentos de cotas/extracciones inestables y las frecuencias de rechazo de cada restricción.
- `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tablas ordenadas de la mediana de respuesta y sus bandas (`target_idx`/`shock_idx` seleccionan el par).
- `.irf(horizon=None)`: Respuestas al impulso de mediana ponderada con forma $(h+1, n, n)$. Para $h \le H$, un corte de `irf_median`; para $h > H$, la mediana ponderada de la FIR de cada extracción aceptada extendida hasta $h$ (las primeras $H+1$ filas coinciden con `irf_median`).
- `.fevd(horizon=None)`: Descomposición de la varianza del error de pronóstico de mediana ponderada, con filas renormalizadas para sumar 1; extendida más allá de $H$ del mismo modo.
- `.historical_decomposition(variable=None, shock=None, init_y=None)`: Descomposición histórica que usa la $B$ de la extracción representativa junto con los objetos de forma reducida de esa misma extracción (MCO en modo MCO; los propios $(A, c)$ y residuos de la extracción a posteriori en modo bayesiano). `init_y` toma por defecto las primeras $p$ observaciones almacenadas, de modo que `deterministic + shocks.sum(axis=2)` reproduce $y_t$ exactamente para $t \ge p$. Devuelve un diccionario (`'shocks'` de forma $(T-p, n, n)$, `'deterministic'` de forma $(T-p, n)$) o un `DataFrame` ordenado cuando se indican `variable` y/o `shock`.
